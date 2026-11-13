"""render: turn an edit plan (v2) into a finished video, effects included.

Per shot (rendered in parallel, cached by its settings):
  speed       constant, or a ramp (`ramp_to`): source time follows a quadratic, so the speed changes
              smoothly. Slow motion uses the clip's real frames when it was shot at 50/60 fps.
  framing     centre crop to the output shape, then a zoom path: a slow push-in/pull-out (`zoom`)
              plus instant "punch-ins" that ease back (`punches`, seconds into the shot).
  look        one grade for the whole edit (`look` preset), with a per-shot exposure and
              saturation correction (`grade`) so neighbouring shots match.
  flashes     white flashes that decay quickly (`flash`); fades from/to black on the first/last shot.
  handles     extra frames before/after when a transition overlaps the cut, so a transition is
              centred on the cut and the edit keeps its exact length.

Then shots joined by transitions are mixed with `xfade` in small chunks (dissolve, dip to
black/white, zoom, directional slide/whip), the chunks are joined without re-encoding, and the
soundtrack is muxed (music from its chosen start with fades and loudness normalisation, or a
reference's own sound untouched).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import autocut as ac
from autocut import AutocutError, log

LOOKS = {
    "natural": {"contrast": 1.04, "saturation": 1.08, "extra": ""},
    "cinematic": {"contrast": 1.10, "saturation": 1.02, "gamma": 0.97,
                  "extra": "colorbalance=rs=-0.05:gs=-0.01:bs=0.06:rh=0.06:gh=0.01:bh=-0.05,vignette=angle=0.32"},
    "vivid": {"contrast": 1.08, "saturation": 1.28, "extra": "vibrance=intensity=0.15"},
    "warm": {"contrast": 1.06, "saturation": 1.12, "extra": "colortemperature=temperature=5000:mix=0.6,vignette=angle=0.3"},
    "mono": {"contrast": 1.18, "saturation": 0.0, "extra": "vignette=angle=0.35"},
}
XFADE = {
    "dissolve": "fade",
    "fadeblack": "fadeblack",
    "fadewhite": "fadewhite",
    "zoomin": "zoomin",
    "whipleft": "smoothleft",
    "whipright": "smoothright",
    "whipup": "smoothup",
    "whipdown": "smoothdown",
}
SHOT_BITRATE = "24M"  # intermediate quality (1080p); chunks with transitions are encoded once more


def _f(x: float) -> str:
    return f"{x:.6f}".rstrip("0").rstrip(".") if abs(x) > 1e-9 else "0"


def source_time(shot: dict, t: float) -> float:
    """Source seconds from the shot's in-point at output time t (seconds from the cut)."""
    v0 = float(shot.get("speed", 1.0))
    D = shot["frames"] / shot["_fps"]
    a = (float(shot.get("ramp_to", v0)) - v0) / D if D > 0 else 0.0
    return v0 * t + 0.5 * a * t * t


def pts_expr(shot: dict, pre: float) -> str:
    """setpts expression mapping input time (from the seek point) to output time (from the
    first rendered frame, i.e. pre seconds before the cut)."""
    v0 = float(shot.get("speed", 1.0))
    D = shot["frames"] / shot["_fps"]
    a = (float(shot.get("ramp_to", v0)) - v0) / D if D > 0 else 0.0
    if abs(a) < 1e-6:
        return f"(PTS-STARTPTS)/{_f(v0)}"
    t0 = -pre
    s0 = v0 * t0 + 0.5 * a * t0 * t0
    # t = (-v0 + sqrt(v0^2 + 2a(u + s0))) / a ; out = t - t0
    u = "((PTS-STARTPTS)*TB)"
    return f"((-{_f(v0)}+sqrt(max(0,{_f(v0 * v0)}+{_f(2 * a)}*({u}+{_f(s0)}))))/{_f(a)}-({_f(t0)}))/TB"


def zoom_expr(shot: dict, pre_frames: int, fps: float) -> str:
    t = f"(on/{_f(fps)}-{_f(pre_frames / fps)})"
    D = max(shot["frames"] / fps, 1e-3)
    z = shot.get("zoom") or {}
    z0, z1 = float(z.get("from", 1.0)), float(z.get("to", 1.0))
    p = f"clip({t}/{_f(D)},0,1)"
    ease = f"({p}*{p}*(3-2*{p}))"
    expr = f"({_f(z0)}+{_f(z1 - z0)}*{ease})"
    for tp in shot.get("punches", []):
        expr += f"+0.14*exp(-max(0,{t}-{_f(tp)})/0.22)*gte({t},{_f(tp)})"
    return f"max(1,{expr})"


def grade_filters(plan: dict, shot: dict) -> str:
    look = LOOKS.get(plan.get("look", "natural"), LOOKS["natural"])
    if plan.get("look") == "match" and plan.get("match"):
        m = plan["match"]
        look = {"contrast": m.get("contrast", 1.04), "saturation": m.get("saturation", 1.05), "extra": m.get("extra", "")}
    g = shot.get("grade") or {}
    eq = f"eq=contrast={_f(look['contrast'])}:saturation={_f(look['saturation'] * float(g.get('sat', 1.0)))}:brightness={_f(float(g.get('exposure', 0.0)))}"
    if look.get("gamma"):
        eq += f":gamma={_f(look['gamma'])}"
    out = [eq]
    if look.get("extra"):
        out.append(look["extra"])
    return ",".join(out)


def track_expr(keys: list, axis: int, pre: float) -> str:
    """Piecewise-linear crop position over output time (keys: [t, fx, fy], t in seconds from the cut)."""
    pts = sorted((float(k[0]), float(k[axis])) for k in keys)
    if not pts:
        return "0.5"
    T = f"(t-{_f(pre)})"
    expr = _f(pts[-1][1])
    for (t0, v0), (t1, v1) in reversed(list(zip(pts, pts[1:]))):
        seg = _f(v0) if t1 - t0 < 1e-6 else f"({_f(v0)}+{_f(v1 - v0)}*({T}-{_f(t0)})/{_f(t1 - t0)})"
        expr = f"if(lt({T},{_f(t1)}),{seg},{expr})"
    return f"if(lt({T},{_f(pts[0][0])}),{_f(pts[0][1])},{expr})"


def frame_filters(shot: dict, W: int, H: int, zw: int, zh: int, pre: float) -> list[str]:
    """Source picture -> zw x zh canvas: the active picture (black bars removed), then a tracked
    crop that fills the shape ("smart"/"center"), or the whole picture fitted over a blurred copy
    ("blur") or between black bars ("bars")."""
    out = []
    act = shot.get("active")
    if act and len(act) == 4:
        x0, y0, x1, y1 = (float(v) for v in act)
        out.append(f"crop=w='trunc(iw*{_f(x1 - x0)}/2)*2':h='trunc(ih*{_f(y1 - y0)}/2)*2':x='iw*{_f(x0)}':y='ih*{_f(y0)}'")
    fit = shot.get("fit")
    if fit == "bars":
        out.append(f"scale={zw}:{zh}:force_original_aspect_ratio=decrease:flags=bicubic,pad={zw}:{zh}:(ow-iw)/2:(oh-ih)/2:black,setsar=1")
        return out
    if fit == "blur":
        bw, bh = max(2, W // 8 // 2 * 2), max(2, H // 8 // 2 * 2)
        out.append(
            f"split[bg][fg];[bg]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},boxblur=6:2,"
            f"scale={zw}:{zh},eq=brightness=-0.12:saturation=1.15[bb];"
            f"[fg]scale={zw}:{zh}:force_original_aspect_ratio=decrease:flags=bicubic[ff];"
            f"[bb][ff]overlay=x='(W-w)/2':y='(H-h)/2',setsar=1"
        )
        return out
    track = shot.get("track")
    if track:
        fx, fy = track_expr(track, 1, pre), track_expr(track, 2, pre)
    else:
        f = shot.get("focus") or [0.5, 0.5]
        fx, fy = _f(float(f[0])), _f(float(f[1]))
    out.append(f"crop=w='min(iw,ih*{W}/{H})':h='min(ih,iw*{H}/{W})':x='(iw-ow)*({fx})':y='(ih-oh)*({fy})'")
    out.append(f"scale={zw}:{zh}:flags=bicubic,setsar=1")
    return out


def shot_filter(plan: dict, shot: dict, pre: int, post: int) -> str:
    W, H, fps = int(plan["width"]), int(plan["height"]), float(plan["fps"])
    total = shot["frames"] + pre + post
    zoomed = bool(shot.get("punches")) or any(abs(float((shot.get("zoom") or {}).get(k, 1.0)) - 1.0) > 1e-6 for k in ("from", "to"))
    # Work at 1.5x the output size when zooming: zoom paths stay smooth (sub-pixel) without 4K-sized frames.
    zw, zh = (int(W * 1.5) // 2 * 2, int(H * 1.5) // 2 * 2) if zoomed else (W, H)
    chain = [f"setpts='{pts_expr(shot, pre / fps)}'", f"fps={_f(fps)}", *frame_filters(shot, W, H, zw, zh, pre / fps)]
    if zoomed:
        chain.append(f"zoompan=z='{zoom_expr(shot, pre, fps)}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps={_f(fps)}")
    chain.append(grade_filters(plan, shot))
    t = f"(t-{_f(pre / fps)})"
    for tf in shot.get("flash", []):
        chain.append(f"eq=brightness='0.55*exp(-max(0,{t}-{_f(tf)})*9)*gte({t},{_f(tf)})':eval=frame")
    if shot.get("fade_in"):
        chain.append(f"fade=t=in:st={_f(pre / fps)}:d={_f(shot['fade_in'] / fps)}")
    if shot.get("fade_out"):
        st = (pre + shot["frames"] - shot["fade_out"]) / fps
        chain.append(f"fade=t=out:st={_f(st)}:d={_f(shot['fade_out'] / fps)}")
    chain.append("format=yuv420p")
    _ = total
    return ",".join(chain)


def handles(shots: list[dict]) -> list[tuple[int, int]]:
    """Frames each shot needs before/after its cut for centred transitions."""
    out = [[0, 0] for _ in shots]
    for i in range(1, len(shots)):
        tr = shots[i].get("transition_in") or {}
        d = int(tr.get("frames", 0)) if tr.get("type", "cut") != "cut" else 0
        if d > 0:
            out[i - 1][1] += (d + 1) // 2
            out[i][0] += d // 2
    return [tuple(x) for x in out]


def render_shot(plan: dict, shot: dict, pre: int, post: int, dst: Path, codec: str = "h264") -> None:
    fps = float(plan["fps"])
    shot = {**shot, "_fps": fps}
    t_start = -pre / fps
    t_end = (shot["frames"] + post) / fps
    s_start = float(shot["in"]) + source_time(shot, t_start)
    s_len = source_time(shot, t_end) - source_time(shot, t_start)
    seek = max(0.0, s_start)
    vf = shot_filter(plan, shot, pre, post)
    total = shot["frames"] + pre + post
    vf += f",tpad=stop_mode=clone:stop_duration={_f(total / fps + 1)}"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-hwaccel", "videotoolbox",
           "-ss", _f(seek), "-t", _f(s_len + 0.5), "-i", shot["clip"], "-an", "-vf", vf,
           "-frames:v", str(total), "-fps_mode", "cfr", "-r", ac.rate_arg(fps)]
    cmd += ["-c:v", "h264_videotoolbox", "-b:v", SHOT_BITRATE] if codec == "h264" else ac.video_encoder_args(codec, SHOT_BITRATE)
    tmp = dst.with_suffix(".part.mp4")
    ac.run(cmd + [str(tmp)])
    tmp.replace(dst)


def shot_key(plan: dict, shot: dict, pre: int, post: int) -> str:
    st = os.stat(shot["clip"])
    keys = {k: v for k, v in shot.items() if k not in ("score", "why", "tags", "energy")}
    raw = json.dumps([keys, st.st_size, int(st.st_mtime), pre, post, plan["width"], plan["height"], plan["fps"], plan.get("look"), plan.get("match"), "r3"], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def assemble_chunk(plan: dict, files: list[Path], shots: list[dict], hand: list[tuple[int, int]], dst: Path) -> None:
    """Join shots whose cuts carry transitions with xfade (one encode)."""
    fps = float(plan["fps"])
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for f in files:
        cmd += ["-i", str(f)]
    parts = [f"[{i}:v]settb=AVTB,fps={_f(fps)},format=yuv420p[s{i}]" for i in range(len(files))]
    cur, cur_len = "s0", shots[0]["frames"] + hand[0][0] + hand[0][1]
    for i in range(1, len(files)):
        tr = shots[i]["transition_in"]
        d = int(tr["frames"])
        kind = XFADE.get(tr["type"], "fade")
        off = (cur_len - d) / fps
        out = f"x{i}"
        parts.append(f"[{cur}][s{i}]xfade=transition={kind}:duration={_f(d / fps)}:offset={_f(off)}[{out}]")
        cur = out
        cur_len = cur_len + shots[i]["frames"] + hand[i][0] + hand[i][1] - d
    total = sum(s["frames"] for s in shots) + hand[0][0] + hand[-1][1]
    cmd += ["-filter_complex", ";".join(parts), "-map", f"[{cur}]", "-frames:v", str(total),
            "-fps_mode", "cfr", "-r", ac.rate_arg(fps), "-c:v", "h264_videotoolbox", "-b:v", "20M", str(dst)]
    ac.run(cmd)


def render(plan_path: Path, out: str | None = None, workers: int = 2) -> Path:
    ac.require_tools()
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text())
    shots = [s for s in plan.get("shots", []) if int(s.get("frames", 0)) > 0]
    if not shots:
        raise AutocutError("the plan has no shots")
    for s in shots:
        if not os.path.isfile(s["clip"]):
            raise AutocutError(f"clip missing: {s['clip']}")
    shots[0] = {**shots[0], "transition_in": {"type": "cut"}}
    fps = float(plan["fps"])
    total_frames = sum(int(s["frames"]) for s in shots)
    total = total_frames / fps
    stem = plan_path.name.replace(".montage.json", "").replace(".json", "")
    out_path = Path(out).resolve() if out else plan_path.with_name(stem + ".mp4")
    audio = plan.get("audio") or {}
    if any(out_path == Path(p).resolve() for p in [s["clip"] for s in shots] + ([audio["path"]] if audio.get("path") else [])):
        raise AutocutError("refusing to overwrite an original file")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = plan_path.with_name(stem + ".work")
    work.mkdir(exist_ok=True)
    hand = handles(shots)
    files = [work / f"shot_{shot_key(plan, s, *h)}.mp4" for s, h in zip(shots, hand)]
    todo = [(s, h, f) for s, h, f in zip(shots, hand, files) if not f.is_file()]
    log(f"rendering {len(todo)} of {len(shots)} shots ({len(shots) - len(todo)} reused), {workers} workers")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(render_shot, plan, s, h[0], h[1], f) for s, h, f in todo]
        for i, fut in enumerate(futs, 1):
            fut.result()
            log(f"progress {i}/{len(todo)} shots")
    log(f"shots done in {time.time() - t0:.1f}s")
    # chunks: runs of shots linked by transitions
    chunks: list[list[int]] = []
    for i, s in enumerate(shots):
        if i > 0 and (s.get("transition_in") or {}).get("type", "cut") != "cut":
            chunks[-1].append(i)
        else:
            chunks.append([i])
    keep = set(files)
    with tempfile.TemporaryDirectory(prefix="ei-render-") as tmp:
        pieces = []
        for k, ch in enumerate(chunks):
            if len(ch) == 1:
                pieces.append(files[ch[0]])
                continue
            dst = Path(tmp) / f"chunk{k:03d}.mp4"
            assemble_chunk(plan, [files[i] for i in ch], [shots[i] for i in ch], [hand[i] for i in ch], dst)
            pieces.append(dst)
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{str(p).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for p in pieces))
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst)]
        if audio.get("path"):
            if not ac.probe(audio["path"]).has_audio:
                raise AutocutError(f"no sound in {audio['path']}")
            cmd += ["-ss", _f(float(audio.get("start", 0))), "-t", _f(total + 0.5), "-i", audio["path"]]
            chain = [f"aresample=48000,apad,atrim=0:{_f(total)},asetpts=PTS-STARTPTS"]
            if float(audio.get("fade_in", 0)) > 0:
                chain.append(f"afade=t=in:st=0:d={_f(float(audio['fade_in']))}")
            if float(audio.get("fade_out", 0)) > 0:
                fo = min(float(audio["fade_out"]), total / 2)
                chain.append(f"afade=t=out:st={_f(max(0.0, total - fo))}:d={_f(fo)}")
            if audio.get("normalize"):
                chain.append("loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000")
            cmd += ["-filter_complex", f"[1:a]{','.join(chain)}[a]", "-map", "0:v", "-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-map", "0:v"]
        cmd += ["-c:v", "copy", "-frames:v", str(total_frames), "-movflags", "+faststart", str(out_path)]
        ac.run(cmd)
    for stale in work.glob("shot_*.mp4"):
        if stale not in keep:
            stale.unlink()
    log(f"rendered {total:.1f}s ({len(shots)} shots) in {time.time() - t0:.1f}s -> {out_path}")
    return out_path


def quality_check(path: str, expect_frames: int, fps: float) -> list[str]:
    """Problems a viewer would notice: wrong length, black stretches, frozen picture."""
    import re
    import subprocess

    issues = []
    proc = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        n = int(proc.stdout.strip())
        if n != expect_frames:
            issues.append(f"length: {n} frames, expected {expect_frames}")
    except ValueError:
        issues.append("could not count frames")
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-i", path, "-an", "-vf", "blackdetect=d=0.4:pix_th=0.08,freezedetect=n=0.002:d=1.2", "-f", "null", "-"], capture_output=True, text=True)
    for m in re.finditer(r"black_start:([\d.]+) black_end:([\d.]+)", proc.stderr):
        a, b = float(m.group(1)), float(m.group(2))
        if a > 0.6 and b < expect_frames / fps - 1.0:  # fades at the ends are intended
            issues.append(f"black {a:.1f}–{b:.1f}s")
    for m in re.finditer(r"freeze_start: ([\d.]+)", proc.stderr):
        issues.append(f"frozen picture from {float(m.group(1)):.1f}s")
    return issues
