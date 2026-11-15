"""Your own camera as the presenter of a generated video, step 2: the takes over the video.

Only the picture of each take is used. For every piece of the voice the owner recorded themself saying,
- the moment they start and stop speaking is found in the take's own sound;
- the take's sound is lined up with how the generated voice says the same words (the two say the same
  thing at different speeds), which gives, for every moment of the voice, the moment of the take to show;
- the take is cut around the face, so the face stays in one place from take to take.
Where nothing is said for a while, the idle look is shown. The result is a plan
(`<name>.presenter/plan.json`, editable), and a render of it: the takes in a circle over the video, up to
any time that is recorded so far. The video's own sound is kept as it is.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import presenter as pr
from autocut import AutocutError, log

SR = 16000  # the sound of takes and voice is compared at this rate
HOP = 0.01  # seconds between the frames of that comparison
MAX_SPEED = 2.5  # a take is never shown faster or slower than this
LEAN = 0.3  # a take may show this much of itself before and after its words
IDLE_GAP = 0.8  # a pause longer than this (after the takes' own lead-in and out) shows the idle look
BLEND = 3  # frames over which one take fades into the next
SHAPES = ("circle", "rounded", "square")
POSITIONS = ("bottom-right", "bottom-left", "bottom-center", "top-right", "top-left", "top-center")
LOOK = {"shape": "circle", "size": 0.4, "position": "bottom-right", "margin": 40, "border": True}


def _ffmpeg(args: list[str], what: str) -> bytes:
    r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", *args], capture_output=True)
    if r.returncode != 0:
        raise AutocutError(f"{what}: {r.stderr.decode(errors='replace').strip()[-300:]}")
    return r.stdout


def sound_of(path: Path, start: float | None = None, seconds: float | None = None) -> np.ndarray:
    """The sound of a file as mono samples at SR (a take's sound starts at the take's time 0)."""
    cut = ([] if start is None else ["-ss", f"{max(0.0, start):.3f}"]) + ([] if seconds is None else ["-t", f"{max(0.01, seconds):.3f}"])
    raw = _ffmpeg([*cut, "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-af", "aresample=async=1:first_pts=0", "-f", "f32le", "-"], f"the sound of {path.name} could not be read")
    return np.frombuffer(raw, np.float32).astype(np.float64)


def levels(x: np.ndarray) -> np.ndarray:
    """Loudness in dB, one value per HOP."""
    hop = int(SR * HOP)
    n = len(x) // hop
    if n == 0:
        return np.zeros(0)
    return 10 * np.log10(np.maximum(1e-10, (x[: n * hop].reshape(n, hop) ** 2).mean(axis=1)))


def speech_span(x: np.ndarray) -> tuple[float, float]:
    """From when to when someone speaks in a recording, in seconds (the whole of it when that is unclear)."""
    db = levels(x)
    whole = (0.0, len(x) / SR)
    if len(db) < 20:
        return whole
    floor, peak = float(np.percentile(db, 10)), float(np.percentile(db, 98))
    if peak - floor < 12:
        return whole
    loud = db > max(floor + 10, peak - 30)
    run = np.convolve(loud.astype(int), np.ones(5, dtype=int), mode="same") >= 4  # (50 ms of it, not a click)
    idx = np.flatnonzero(run)
    if len(idx) == 0:
        return whole
    return max(0.0, idx[0] * HOP - 0.04), min(whole[1], (idx[-1] + 1) * HOP + 0.04)


def features(x: np.ndarray) -> np.ndarray:
    """What the sound is like every HOP (26 bands of loudness over pitch, evened out per recording), so
    that the same words by two voices look alike."""
    win, hop, nfft, bands = int(SR * 0.025), int(SR * HOP), 512, 26
    n = (len(x) - win) // hop + 1
    if n < 2:
        return np.zeros((0, bands))
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(win), nfft)) ** 2
    mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    edges = 700 * (10 ** (np.linspace(mel(100), mel(7000), bands + 2) / 2595) - 1)
    bins = np.fft.rfftfreq(nfft, 1 / SR)
    bank = np.maximum(0, np.minimum((bins[None, :] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None]),
                                    (edges[2:, None] - bins[None, :]) / (edges[2:, None] - edges[1:-1, None])))
    f = np.log(spec @ bank.T + 1e-8)
    return (f - f.mean(axis=0)) / (f.std(axis=0) + 1e-6)


def follow(take: np.ndarray, voice: np.ndarray) -> np.ndarray:
    """For every frame of the voice, the frame of the take that says the same thing (both start and end
    together; never backwards; neither may run more than MAX_SPEED times faster than the other)."""
    n, m = len(voice), len(take)
    if n < 2 or m < 2:
        return np.linspace(0, max(0, m - 1), max(1, n))
    line = np.linspace(0, m - 1, n)
    if not (1 / MAX_SPEED < m / n < MAX_SPEED):
        return line
    cost = np.sqrt(((voice[:, None, :] - take[None, :, :]) ** 2).mean(axis=2))
    band = max(8.0, 0.25 * max(n, m))  # (and never far from an even stretch)
    cost[np.abs(np.arange(m)[None, :] - line[:, None]) > band] = np.inf
    acc = np.full((n, m), np.inf)
    acc[0, 0] = cost[0, 0]
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0 or not np.isfinite(cost[i, j]):
                continue
            best = min(acc[i - 1, j - 1] if i and j else np.inf, (acc[i - 1, j] if i else np.inf) + 0.15, (acc[i, j - 1] if j else np.inf) + 0.15)
            acc[i, j] = cost[i, j] + best
    if not np.isfinite(acc[-1, -1]):
        return line
    i, j, sums, counts = n - 1, m - 1, np.zeros(n), np.zeros(n)
    while True:
        sums[i] += j
        counts[i] += 1
        if i == 0 and j == 0:
            break
        steps = [(acc[i - 1, j - 1] if i and j else np.inf, i - 1, j - 1), (acc[i - 1, j] if i else np.inf, i - 1, j), (acc[i, j - 1] if j else np.inf, i, j - 1)]
        _, i, j = min(steps, key=lambda s: s[0])
    path = sums / np.maximum(1, counts)
    k = min(len(path) // 2 * 2 - 1, 15)  # smooth over 150 ms: lips do not jump
    if k >= 3:
        padded = np.concatenate([np.full(k // 2, path[0]), path, np.full(k // 2, path[-1])])
        path = np.convolve(padded, np.ones(k) / k, mode="valid")
    path[0], path[-1] = 0, m - 1
    return np.maximum.accumulate(np.clip(path, 0, m - 1))


def anchors_of(take_sound: np.ndarray, voice_sound: np.ndarray) -> tuple[list[list[float]], dict]:
    """[[seconds into the voice piece, seconds into the take], ...] and how well the two fit."""
    ts, te = speech_span(take_sound)
    vs, ve = speech_span(voice_sound)
    if te - ts < 0.2 or ve - vs < 0.2:
        ts, te, vs, ve = 0.0, len(take_sound) / SR, 0.0, len(voice_sound) / SR
    ft, fv = features(take_sound[int(ts * SR) : int(te * SR)]), features(voice_sound[int(vs * SR) : int(ve * SR)])
    path = follow(ft, fv)
    step = max(1, int(0.1 / HOP))
    pick = sorted(set(list(range(0, len(path), step)) + [len(path) - 1]))
    anchors = [[round(vs + i * HOP, 3), round(ts + float(path[i]) * HOP, 3)] for i in pick]
    if len(anchors) < 2:
        anchors = [[round(vs, 3), round(ts, 3)], [round(ve, 3), round(te, 3)]]
    anchors[-1] = [round(ve, 3), round(te, 3)]
    # how alike the loudness of the two is once lined up, and with an even stretch instead (-1..1)
    lt, lv = levels(take_sound), levels(voice_sound)
    def alike(at):  # noqa: E306
        u = np.arange(int(vs / HOP), min(len(lv), int(ve / HOP)))
        i = np.clip(np.round(at(u * HOP) / HOP).astype(int), 0, max(0, len(lt) - 1))
        return float(np.corrcoef(lv[u], lt[i])[0, 1]) if len(u) > 5 and len(lt) and lv[u].std() > 0 and lt[i].std() > 0 else 0.0
    a = np.array(anchors)
    fit = alike(lambda u: np.interp(u, a[:, 0], a[:, 1]))
    even = alike(lambda u: ts + (u - vs) * (te - ts) / max(1e-6, ve - vs))
    if fit < even:  # the even stretch is the better one
        anchors, fit = [[round(vs, 3), round(ts, 3)], [round(ve, 3), round(te, 3)]], even
    return anchors, {"speech": [round(ts, 3), round(te, 3)], "speed": round((te - ts) / max(1e-6, ve - vs), 3), "fit": round(fit, 3), "even": round(even, 3)}


def face_of(path: Path) -> list[float] | None:
    """Where the face is in a take: [centre x, centre y, height] in fractions of the picture."""
    with tempfile.TemporaryDirectory(prefix="ei-face-") as tmp:
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(path), "-vf", "fps=1,scale=640:-2", "-frames:v", "5", str(Path(tmp) / "f%d.jpg")], capture_output=True)
        boxes = [b for b in pr.face_boxes(sorted(Path(tmp).glob("f*.jpg"))) if b]
    if not boxes:
        return None
    b = np.median(np.array(boxes), axis=0)
    return [round(float((b[0] + b[2]) / 2), 4), round(float((b[1] + b[3]) / 2), 4), round(float(b[3] - b[1]), 4)]


def crop_of(face: list[float] | None, face_h: float, w: int, h: int) -> tuple[int, int, int]:
    """(x, y, side) of the square shown from a take: the face a little above the middle, head and
    shoulders in it, the same size in every take."""
    side = int(min(w, h, max(0.3 * h, 2.6 * face_h * h))) // 2 * 2
    cx, cy = (face[0] * w, (face[1] + 0.12 * face[2]) * h) if face else (w / 2, h / 2)
    return int(min(max(0, cx - side / 2), w - side)), int(min(max(0, cy - side / 2), h - side)), side


def size_of(path: Path) -> tuple[int, int, float, float]:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
        s = d["streams"][0]
        num, den = s["r_frame_rate"].split("/")
        fps = float(num) / float(den) if float(den) else 30.0
        return int(s["width"]), int(s["height"]), fps if 1 <= fps <= 120 else 30.0, float(d.get("format", {}).get("duration") or 0)
    except (ValueError, KeyError, IndexError, ZeroDivisionError) as e:
        raise AutocutError(f"{path.name} could not be read as a video") from e


def make_plan(session: pr.Session, look: dict | None = None, *, fresh: bool = False) -> dict:
    """The plan of the presenter picture: per recorded piece, which take, where the face is and how the
    take follows the voice. What an earlier plan says about a take is kept (it may have been edited)."""
    path = session.dir / "plan.json"
    old = {}
    if path.is_file() and not fresh:
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
    kept = {p.get("take"): p for p in old.get("pieces") or [] if p.get("anchors")}
    pieces = []
    for ln in session.lines:
        take = session.takes.get(str(ln["index"]))
        if not take:
            continue
        p = kept.get(take["file"])
        if not p or p.get("text") != ln["text"]:
            tp = session.dir / "takes" / take["file"]
            anchors, how = anchors_of(sound_of(tp), sound_of(session.voice, ln["src"], ln["end"] - ln["start"]))
            p = {"take": take["file"], "text": ln["text"], "anchors": anchors, **how, "face": face_of(tp)}
            log(f"presenter: piece {ln['index'] + 1} follows the voice (fit {how['fit']:.2f}, speed {how['speed']:.2f}×)")
        pieces.append({**p, "index": ln["index"], "start": ln["start"], "end": ln["end"]})
    idle = session.takes.get(pr.IDLE)
    idle_plan = None
    if idle:
        was = old.get("idle") or {}
        idle_plan = was if was.get("take") == idle["file"] else {"take": idle["file"], "face": face_of(session.dir / "takes" / idle["file"])}
    plan = {"version": 1, "video": str(session.video), "look": {**LOOK, **(old.get("look") or {}), **{k: v for k, v in (look or {}).items() if v is not None}},
            "idle": idle_plan, "pieces": pieces}
    session.dir.mkdir(parents=True, exist_ok=True)
    pr._atomic(path, json.dumps(plan, indent=1, ensure_ascii=False))
    return plan


def schedule(pieces: list[dict], frames: int, fps: float, has_idle: bool) -> list[tuple[int, float] | None]:
    """For every frame of the video: (which piece, the moment of its take to show), (-1, seconds of idle
    look so far) or None (nothing recorded for it and no idle look: the take before it stands still)."""
    out: list[tuple[int, float] | None] = [None] * frames
    spans = []
    for k, p in enumerate(pieces):
        a = np.array(p["anchors"], dtype=float)
        lo = pieces[k - 1]["end"] if k else 0.0
        hi = pieces[k + 1]["start"] if k + 1 < len(pieces) else frames / fps
        first, last = p["start"] + a[0, 0], p["start"] + a[-1, 0]  # when the words start and stop, in the video
        gap_before, gap_after = max(0.0, first - lo), max(0.0, hi - last)
        # a short pause is shared by the two takes around it; in a long one each leans in, the idle look between
        begin = first - (gap_before / 2 if k and gap_before <= IDLE_GAP + 2 * LEAN or not has_idle and k else min(gap_before, LEAN) if has_idle else gap_before)
        end = last + (gap_after / 2 if k + 1 < len(pieces) and gap_after <= IDLE_GAP + 2 * LEAN else min(gap_after, LEAN) if has_idle else gap_after / 2 if k + 1 < len(pieces) else gap_after)
        spans.append((begin, end, a))
    idle_t = 0.0
    k = 0
    for f in range(frames):
        t = f / fps
        while k + 1 < len(spans) and t >= spans[k][1] and t >= spans[k + 1][0]:
            k += 1
        if spans and spans[k][0] <= t < spans[k][1]:
            begin, end, a = spans[k]
            u = t - pieces[k]["start"]
            if u < a[0, 0]:
                out[f] = (k, a[0, 1] - (a[0, 0] - u))
            elif u > a[-1, 0]:
                out[f] = (k, a[-1, 1] + (u - a[-1, 0]))
            else:
                out[f] = (k, float(np.interp(u, a[:, 0], a[:, 1])))
        elif has_idle:
            out[f] = (-1, idle_t)
            idle_t += 1 / fps
    return out


def _take_frames(path: Path, crop: tuple[int, int, int], d: int, fps: float) -> np.ndarray:
    x, y, side = crop
    raw = _ffmpeg(["-i", str(path), "-an", "-vf", f"fps={fps:.6f}:start_time=0,crop={side}:{side}:{x}:{y},scale={d}:{d}:flags=bicubic", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                  f"the take {path.name} could not be read")
    n = len(raw) // (d * d * 3)
    if n == 0:
        raise AutocutError(f"the take {path.name} has no picture")
    return np.frombuffer(raw[: n * d * d * 3], np.uint8).reshape(n, d, d, 3)


def _shape_files(tmp: Path, d: int, look: dict) -> tuple[Path, Path, Path, int]:
    """The cut-out (mask), the shadow under it and the border over it; and the room around for the shadow."""
    from PIL import Image, ImageDraw, ImageFilter

    pad, ss = max(8, d // 14), 4
    c = d + 2 * pad
    radius = {"circle": d / 2, "rounded": d * 0.16, "square": 0}.get(look.get("shape"), d / 2)
    def box(img, xy, **kw):  # noqa: E306
        ImageDraw.Draw(img).rounded_rectangle(xy, radius=int(radius * ss), **kw)
    mask = Image.new("L", (d * ss, d * ss), 0)
    box(mask, (0, 0, d * ss - 1, d * ss - 1), fill=255)
    mask.resize((d, d), Image.LANCZOS).save(tmp / "mask.png")
    under = Image.new("RGBA", (c * ss, c * ss), (0, 0, 0, 0))
    box(under, (pad * ss, (pad + pad // 3) * ss, (pad + d) * ss, (pad + pad // 3 + d) * ss), fill=(0, 0, 0, 150))
    under.filter(ImageFilter.GaussianBlur(pad * ss * 0.45)).resize((c, c), Image.LANCZOS).save(tmp / "under.png")
    ring = Image.new("RGBA", (c * ss, c * ss), (0, 0, 0, 0))
    if look.get("border", True):
        box(ring, (pad * ss, pad * ss, (pad + d) * ss - 1, (pad + d) * ss - 1), outline=(255, 255, 255, 255), width=max(3, d // 60) * ss)
    ring.resize((c, c), Image.LANCZOS).save(tmp / "ring.png")
    return tmp / "mask.png", tmp / "under.png", tmp / "ring.png", pad


def render(video: str, *, until: float | None = None, out: str | None = None, look: dict | None = None, fresh: bool = False) -> dict:
    """The video with the presenter picture over it, up to `until` seconds (default: as far as is recorded)."""
    session = pr.Session(video)
    if not any(k != pr.IDLE for k in session.takes):
        raise AutocutError("nothing is recorded for this video yet: record some lines first (presenter record)")
    plan = make_plan(session, look, fresh=fresh)
    lk = plan["look"]
    if lk.get("shape") not in SHAPES or lk.get("position") not in POSITIONS:
        raise AutocutError(f"the shape is one of {', '.join(SHAPES)} and the position one of {', '.join(POSITIONS)}")
    W, H, fps, length = size_of(session.video)
    recorded = session.recorded_until()
    whole = len(plan["pieces"]) == len(session.lines)
    end = min(length, float(until)) if until and until > 0 else length if whole else min(length, recorded if recorded > 0 else plan["pieces"][-1]["end"])
    frames = max(1, int(round(end * fps)))
    d = int(min(W, H) * min(0.9, max(0.1, float(lk.get("size", 0.4))))) // 2 * 2
    margin = max(0, int(lk.get("margin", 40)))
    dst = Path(out).expanduser() if out else session.video.with_name(f"{session.video.stem} - presenter.mp4") if whole and end >= length - 0.05 else session.dir / "preview.mp4"
    if dst.resolve() == session.video:
        raise AutocutError("the result cannot replace the video itself")
    faces = [p["face"][2] for p in plan["pieces"] if p.get("face")]
    face_h = float(np.median(faces)) if faces else 0.3
    idle = plan.get("idle")
    plan_frames = schedule(plan["pieces"], frames, fps, bool(idle))
    takes_dir = session.dir / "takes"
    tw, th = 0, 0
    loaded: dict[int, np.ndarray] = {}

    def frames_for(k: int) -> np.ndarray:
        nonlocal tw, th
        if k not in loaded:
            for other in [o for o in loaded if o != -1]:  # (one take in memory at a time, and the idle look)
                del loaded[other]
            p = idle if k == -1 else plan["pieces"][k]
            path = takes_dir / p["take"]
            tw, th, _, _ = size_of(path)
            loaded[k] = _take_frames(path, crop_of(p.get("face"), face_h, tw, th), d, fps)
        return loaded[k]

    with tempfile.TemporaryDirectory(prefix="ei-presenter-") as tmp:
        mask, under, ring, pad = _shape_files(Path(tmp), d, lk)
        c = d + 2 * pad
        side = lk["position"].split("-")[1]
        x = {"left": margin - pad, "right": W - c - margin + pad, "center": (W - c) // 2}[side]
        y = margin - pad if lk["position"].startswith("top") else H - c - margin + pad
        graph = (f"[1:v]format=rgba[f];[2:v]format=gray[m];[f][m]alphamerge[fc];[0:v][3:v]overlay={x}:{y}[a];"
                 f"[a][fc]overlay={x + pad}:{y + pad}:eof_action=pass[b];[b][4:v]overlay={x}:{y}[v]")
        part = dst.with_name(dst.stem + ".part.mp4")
        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(session.video),
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{d}x{d}", "-r", f"{fps:.6f}", "-i", "pipe:0",
               "-loop", "1", "-i", str(mask), "-loop", "1", "-i", str(under), "-loop", "1", "-i", str(ring),
               "-filter_complex", graph, "-map", "[v]", "-map", "0:a?", "-c:a", "copy", "-c:v", "h264_videotoolbox", "-b:v", "12M",
               "-frames:v", str(frames), "-t", f"{frames / fps:.3f}", "-movflags", "+faststart", str(part)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        last, last_k, fade = np.zeros((d, d, 3), np.uint8), None, 0
        try:
            for f, what in enumerate(plan_frames):
                if what is None:
                    frame = last
                else:
                    k, t = what
                    src = frames_for(k)
                    if k == -1:  # the idle look goes forwards, then backwards, and so on: no jump where it repeats
                        i = int(round(t * fps)) % max(1, 2 * (len(src) - 1))
                        i = i if i < len(src) else 2 * (len(src) - 1) - i
                    else:
                        i = min(len(src) - 1, max(0, int(round(t * fps))))
                    frame = src[i]
                    if k != last_k and last_k is not None:
                        fade, held = BLEND, last
                    if fade:
                        w = fade / (BLEND + 1)
                        frame = (held.astype(np.float32) * w + frame.astype(np.float32) * (1 - w)).astype(np.uint8)
                        fade -= 1
                    last_k = k
                last = frame
                proc.stdin.write(frame.tobytes())
                if f % 150 == 0:
                    log(f"presenter: {f / fps:.0f}s of {frames / fps:.0f}s")
            proc.stdin.close()
            err = proc.stderr.read().decode(errors="replace")
            if proc.wait() != 0:
                raise AutocutError(f"the video could not be made: {err.strip()[-300:]}")
        except BrokenPipeError as e:
            err = proc.stderr.read().decode(errors="replace")
            proc.wait()
            raise AutocutError(f"the video could not be made: {err.strip()[-300:]}") from e
        finally:
            if proc.poll() is None:
                proc.kill()
            if part.is_file() and proc.returncode != 0:
                part.unlink()
        part.replace(dst)
    log(f"presenter: {dst.name}, {frames / fps:.1f}s of {length:.1f}s, {len(plan['pieces'])} of {len(session.lines)} pieces recorded")
    return {"outputs": [str(dst)], "plan": str(session.dir / "plan.json"), "seconds": round(frames / fps, 2), "whole": bool(whole and end >= length - 0.05),
            "pieces": len(plan["pieces"]), "of": len(session.lines)}
