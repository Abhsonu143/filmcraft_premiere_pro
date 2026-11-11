#!/usr/bin/env python3
"""voiceover_cut — narration-driven editor (travel videos: silent footage + a recorded voiceover).

Pipeline:  voiceover -> transcript (script as a hint) -> beats -> shots (~3 s, exact frames)
           clips -> CLIP index (cached) -> match each beat's words to footage -> plan.json -> render.

    python voiceover_cut.py plan   vo.wav --clips footage/ --script script.txt --music bed.mp3
    python voiceover_cut.py render vo.vo.plan.json --workers 2
    python voiceover_cut.py auto   vo.wav --clips a.mp4 b.mp4 --captions
    python voiceover_cut.py export vo.vo.plan.json          # -> .otio (+ .srt) for FilmCraft

Every shot in the plan (`clip`, `in`) can be changed by hand; render re-encodes only shots that changed.
Without open_clip installed (or with --no-clip) shots rotate through the clips instead of matching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import autocut as ac
from autocut import AutocutError, log

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mts", ".m2ts", ".webm", ".mxf"}
CLIP_MODEL = ("ViT-B-32-quickgelu", "openai")  # ~600 MB on the GPU; MIT-licensed weights
SAMPLE_EVERY = 2.0  # seconds between indexed frames
EMBED_DIM = 512


# ---------------------------------------------------------------------------------------------
# Beats and shots (pure logic, unit-tested)
# ---------------------------------------------------------------------------------------------


def split_beats(words: list[dict], duration: float, *, max_beat: float = 9.0, pause: float = 0.5) -> list[dict]:
    """Group narration words into beats (sentences or pause-separated phrases). Beats tile the whole
    voiceover: each runs from its first word to the next beat's first word, so nothing is lost."""
    groups: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        text = w["word"].strip()
        if not text:
            continue
        if cur and (w["start"] - cur[-1]["end"] > pause or w["end"] - cur[0]["start"] > max_beat):
            groups.append(cur)
            cur = []
        cur.append(w)
        if text[-1] in ".?!;":
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    if not groups:
        return [{"start": 0.0, "end": duration, "text": ""}] if duration > 0 else []
    beats = []
    for i, g in enumerate(groups):
        start = 0.0 if i == 0 else g[0]["start"]
        end = groups[i + 1][0]["start"] if i + 1 < len(groups) else duration
        beats.append({"start": start, "end": max(end, start), "text": " ".join(w["word"].strip() for w in g)})
    return [b for b in beats if b["end"] > b["start"]]


def fixed_beats(duration: float, length: float) -> list[dict]:
    """Beats without a transcript: equal chunks."""
    n = max(1, round(duration / max(length, 0.5)))
    return [{"start": duration * i / n, "end": duration * (i + 1) / n, "text": ""} for i in range(n)]


def make_shots(beats: list[dict], fps: float, shot_len: float = 3.0) -> list[dict]:
    """Split beats into shots of about `shot_len`. Boundaries are whole frames from one grid, so the
    shots add up exactly to the voiceover length (no drift)."""
    shots = []
    for bi, b in enumerate(beats):
        dur = b["end"] - b["start"]
        n = max(1, round(dur / shot_len))
        for k in range(n):
            f0 = round((b["start"] + dur * k / n) * fps)
            f1 = round((b["start"] + dur * (k + 1) / n) * fps)
            if f1 > f0:
                shots.append({"beat": bi, "start_frame": f0, "frames": f1 - f0})
    # Close any 1-frame gaps/overlaps left by rounding between beats.
    for a, b in zip(shots, shots[1:]):
        a["frames"] = b["start_frame"] - a["start_frame"]
    return [s for s in shots if s["frames"] > 0]


def assign_shots(
    shots: list[dict],
    samples: list[tuple[int, float]],
    sims: list[list[float]],
    clip_durations: list[float],
    fps: float,
    *,
    reuse_penalty: float = 0.08,
    same_clip_penalty: float = 0.03,
    short_penalty: float = 0.05,
) -> list[dict]:
    """Pick footage for each shot. `samples[j]` = (clip index, time) of an indexed frame and
    `sims[beat][j]` its similarity to the beat's text. Already-used moments and the previous shot's
    clip are penalised so the edit doesn't repeat itself. With all-zero sims this rotates through
    the clips and unused moments. Returns {clip, in, score} per shot."""
    if not samples:
        raise AutocutError("no usable footage")
    uses = [0] * len(samples)
    prev_clip = -1
    out = []
    for shot in shots:
        dur = shot["frames"] / fps
        row = sims[shot["beat"]] if shot["beat"] < len(sims) else [0.0] * len(samples)
        best, best_score = 0, -math.inf
        for j, (ci, _t) in enumerate(samples):
            score = row[j] - reuse_penalty * uses[j]
            if ci == prev_clip:
                score -= same_clip_penalty
            if clip_durations[ci] < dur:
                score -= short_penalty
            if score > best_score + 1e-9:
                best, best_score = j, score
        ci, t = samples[best]
        start = min(max(0.0, t - dur / 2), max(0.0, clip_durations[ci] - dur))
        start = math.floor(start * fps) / fps
        for j, (cj, tj) in enumerate(samples):
            if cj == ci and start <= tj < start + dur:
                uses[j] += 1
        prev_clip = ci
        out.append({"clip": ci, "in": round(start, 6), "score": round(row[best], 4)})
    return out


# ---------------------------------------------------------------------------------------------
# CLIP index (child process; cached per clip in ~/.cache/autocut/clip)
# ---------------------------------------------------------------------------------------------


def _frames_rgb(path: str, every: float, size: int = 224):
    import numpy as np

    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-hwaccel", "videotoolbox", "-i", path, "-an",
         "-vf", f"fps=1/{every:g},scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        return np.zeros((0, size, size, 3), dtype=np.uint8)
    n = len(proc.stdout) // (size * size * 3)
    return np.frombuffer(proc.stdout[: n * size * size * 3], dtype=np.uint8).reshape(n, size, size, 3)


def _clip_worker(to_index: list[str], texts: list[str], every: float) -> dict:
    """Runs in its own process: load CLIP once, embed frames of new clips and the beat texts."""
    import numpy as np
    import open_clip
    import torch
    from PIL import Image

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL[0], pretrained=CLIP_MODEL[1], device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL[0])
    images: dict[str, list] = {}
    with torch.no_grad():
        for path in to_index:
            frames = _frames_rgb(path, every)
            embs = []
            for i in range(0, len(frames), 16):
                batch = torch.stack([preprocess(Image.fromarray(f)) for f in frames[i : i + 16]]).to(device)
                e = model.encode_image(batch)
                embs.append((e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy())
            images[path] = (np.concatenate(embs) if embs else np.zeros((0, EMBED_DIM), np.float32)).tolist()
        text_embs = []
        if texts:
            for i in range(0, len(texts), 64):
                tok = tokenizer(texts[i : i + 64]).to(device)
                e = model.encode_text(tok)
                text_embs.append((e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy())
    return {"images": images, "texts": np.concatenate(text_embs).tolist() if text_embs else []}


def clip_similarities(clips: list[str], texts: list[str]) -> tuple[list[tuple[int, float]], list[list[float]]]:
    """(samples, sims[beat][sample]) using cached frame embeddings where possible."""
    import numpy as np

    cache_dir = ac.CACHE_DIR / "clip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{CLIP_MODEL[0]}-{CLIP_MODEL[1]}-{SAMPLE_EVERY}"
    keys = {c: cache_dir / f"{ac.file_key(c, tag)}.npy" for c in clips}
    missing = [c for c in clips if not keys[c].is_file()]
    log(f"CLIP index: {len(clips) - len(missing)} cached, {len(missing)} to index")
    t0 = time.time()
    res = ac.run_isolated(_clip_worker, missing, [t or " " for t in texts], SAMPLE_EVERY)
    for c, emb in res["images"].items():
        np.save(keys[c], np.asarray(emb, dtype=np.float32).reshape(-1, EMBED_DIM))
    log(f"CLIP done in {time.time() - t0:.1f}s")
    samples: list[tuple[int, float]] = []
    mats = []
    for ci, c in enumerate(clips):
        m = np.load(keys[c])
        mats.append(m)
        samples += [(ci, k * SAMPLE_EVERY) for k in range(len(m))]
    if not samples:
        return [], []
    img = np.concatenate(mats)
    txt = np.asarray(res["texts"], dtype=np.float32).reshape(-1, EMBED_DIM)
    sims = txt @ img.T
    # Empty beats (no words) carry no meaning: zero their row so only the penalties decide.
    for i, t in enumerate(texts):
        if not t.strip():
            sims[i] = 0
    return samples, sims.tolist()


# ---------------------------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------------------------


def collect_clips(items: list[str]) -> list[str]:
    out: list[str] = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            out += sorted(str(f.resolve()) for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT and not f.name.startswith("."))
        elif p.is_file():
            out.append(str(p.resolve()))
        else:
            raise AutocutError(f"clip not found: {it}")
    seen = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def make_plan(args: argparse.Namespace) -> dict:
    ac.require_tools()
    voice = str(Path(args.voice).resolve())
    vinfo = ac.probe(voice)
    if not vinfo.has_audio:
        raise AutocutError("the voiceover file has no audio")
    preset = dict(ac.PRESETS[args.preset])
    fps = float(args.fps or preset["fps"])
    clips, durations, infos = [], [], []
    for c in collect_clips(args.clips):
        info = ac.probe(c)
        if not info.has_video or info.duration <= 0:
            log(f"skipping (no video): {c}")
            continue
        clips.append(c)
        durations.append(info.duration)
        infos.append(info)
    if not clips:
        raise AutocutError("no video clips found")
    script = Path(args.script).read_text(encoding="utf-8") if args.script else None
    words: list[dict] = []
    if not args.no_transcribe:
        # Whisper's prompt window is ~224 tokens: the start of the script is the useful part.
        tr = ac.transcribe(voice, model=args.whisper_model, prompt=script[:800] if script else None, language=args.language)
        words = tr["words"]
    beats = split_beats(words, vinfo.duration) if words else fixed_beats(vinfo.duration, args.shot_len)
    shots = make_shots(beats, fps, args.shot_len)
    texts = [b["text"] for b in beats]
    if args.no_clip:
        samples, sims = [(ci, k * SAMPLE_EVERY) for ci, d in enumerate(durations) for k in range(max(1, int(d // SAMPLE_EVERY)))], []
    else:
        try:
            samples, sims = clip_similarities(clips, texts)
        except ImportError:
            log("open_clip not installed: rotating through clips instead of matching")
            samples, sims = [(ci, k * SAMPLE_EVERY) for ci, d in enumerate(durations) for k in range(max(1, int(d // SAMPLE_EVERY)))], []
    if not sims:
        sims = [[0.0] * len(samples) for _ in beats]
    picks = assign_shots(shots, samples, sims, durations, fps)
    plan_shots = []
    for s, p in zip(shots, picks):
        b = beats[s["beat"]]
        plan_shots.append({"clip": clips[p["clip"]], "in": p["in"], "frames": s["frames"], "beat": s["beat"], "text": b["text"], "score": p["score"]})
    total = sum(s["frames"] for s in shots)
    log(f"{len(beats)} beats -> {len(plan_shots)} shots, {total / fps:.1f}s (voice {vinfo.duration:.1f}s)")
    return {
        "version": ac.PLAN_VERSION,
        "mode": "voiceover",
        "voice": voice,
        "preset": args.preset,
        "width": preset["width"],
        "height": preset["height"],
        "fps": fps,
        "video_bitrate": preset["video_bitrate"],
        "clips": [{"path": c, "duration": d, "vfr": i.vfr, "rotation": i.rotation} for c, d, i in zip(clips, durations, infos)],
        "beats": beats,
        "shots": plan_shots,
        "music": {"path": str(Path(args.music).resolve()), "db": args.music_db} if args.music else None,
        "captions": {"style": preset["caption_style"], "words": words} if args.captions and words else None,
    }


# ---------------------------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------------------------


def shot_key(plan: dict, shot: dict, codec: str) -> str:
    st = os.stat(shot["clip"])
    raw = json.dumps([shot["clip"], st.st_size, int(st.st_mtime), shot["in"], shot["frames"], plan["width"], plan["height"], plan["fps"], plan.get("video_bitrate"), codec])
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def render_shot(plan: dict, shot: dict, dst: Path, codec: str) -> None:
    fps = float(plan["fps"])
    dur = shot["frames"] / fps
    vf = ac.cover_filter(plan["width"], plan["height"], fps) + f",tpad=stop_mode=clone:stop_duration={dur:.3f}"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-hwaccel", "videotoolbox",
           "-ss", f"{float(shot['in']):.6f}", "-i", shot["clip"], "-an", "-vf", vf,
           "-frames:v", str(int(shot["frames"])), "-fps_mode", "cfr", "-r", ac.rate_arg(fps)]
    cmd += ac.video_encoder_args(codec, plan.get("video_bitrate", "10M"))
    tmp = dst.with_suffix(".part.mp4")
    run_cmd = cmd + [str(tmp)]
    ac.run(run_cmd)
    tmp.replace(dst)


def render_plan(plan_path: Path, *, out: str | None, codec: str, workers: int) -> Path:
    ac.require_tools()
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text())
    shots = [s for s in plan.get("shots", []) if int(s.get("frames", 0)) > 0]
    if not shots:
        raise AutocutError("the plan has no shots")
    for s in shots:
        if not os.path.isfile(s["clip"]):
            raise AutocutError(f"clip missing: {s['clip']}")
    voice = plan["voice"]
    fps = float(plan["fps"])
    total = sum(int(s["frames"]) for s in shots) / fps
    stem = plan_path.name.replace(".vo.plan.json", "").replace(".json", "")
    out_path = Path(out) if out else plan_path.with_name(stem + ".vo.mp4")
    if any(out_path.resolve() == Path(p).resolve() for p in [voice] + [s["clip"] for s in shots]):
        raise AutocutError("refusing to overwrite an original file")
    work = plan_path.with_name(stem + ".work")
    work.mkdir(exist_ok=True)
    # Shots are encoded with the hardware encoder; h264 everywhere so `-c copy` concat is valid.
    shot_codec = "h264" if codec in ("h264", "hevc") else codec
    files = [work / f"shot_{shot_key(plan, s, shot_codec)}.mp4" for s in shots]
    todo = [(s, f) for s, f in zip(shots, files) if not f.is_file()]
    log(f"rendering {len(todo)} of {len(shots)} shots with {workers} workers ({len(shots) - len(todo)} reused)")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for fut in [ex.submit(render_shot, plan, s, f, shot_codec) for s, f in todo]:
            fut.result()
    log(f"shots done in {time.time() - t0:.1f}s")
    for stale in work.glob("shot_*.mp4"):
        if stale not in files:
            stale.unlink()
    music = (plan.get("music") or {}).get("path")
    if music and not os.path.isfile(music):
        log(f"music missing, rendering without it: {music}")
        music = None
    with tempfile.TemporaryDirectory(prefix="vocut-") as tmp:
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{str(f).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for f in files))
        joined = Path(tmp) / "joined.mp4"
        ac.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined)])
        srt = Path(tmp) / "captions.srt"
        cap = plan.get("captions")
        has_caps = False
        if cap and cap.get("words"):
            big = cap.get("style") == "big"
            cues = ac.group_words(cap["words"], max_words=3 if big else 7, max_chars=18 if big else 38)
            srt.write_text(ac.to_srt(cues), encoding="utf-8")
            has_caps = bool(cues)
            if has_caps:
                shutil.copy(srt, out_path.with_suffix(".srt"))
        graph = [f"[1:a]aresample=48000,apad,atrim=0:{total:.6f},asetpts=PTS-STARTPTS[voice]"]
        if music:
            graph.append(ac.music_mix_filter("voice", "2:a", "aout", total, float(plan["music"].get("db", -16))))
        else:
            graph.append("[voice]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]")
        if has_caps:
            graph.insert(0, "[0:v]" + ac.subtitles_filter(str(srt), cap.get("style", "bottom")) + "[vout]")
        script = Path(tmp) / "graph.txt"
        script.write_text(";\n".join(graph))
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(joined), "-i", voice]
        if music:
            cmd += ["-stream_loop", "-1", "-i", music]
        cmd += ac.filter_script_args(str(script))
        if has_caps:
            cmd += ["-map", "[vout]"] + ac.video_encoder_args(codec, plan.get("video_bitrate", "10M"))
        else:
            cmd += ["-map", "0:v", "-c:v", "copy"]
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-t", f"{total:.6f}", "-movflags", "+faststart", str(out_path)]
        t1 = time.time()
        ac.run(cmd)
        log(f"final mux {time.time() - t1:.1f}s -> {out_path}")
    return out_path


def add_plan_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("voice", help="recorded narration (wav/m4a/mp3/...)")
    p.add_argument("--clips", nargs="+", required=True, help="video files and/or folders of footage")
    p.add_argument("--script", help="the script text (helps Whisper spell names right)")
    p.add_argument("--preset", choices=sorted(ac.PRESETS), default="travel")
    p.add_argument("--fps", type=float)
    p.add_argument("--shot-len", type=float, default=3.0, help="target seconds per shot")
    p.add_argument("--music")
    p.add_argument("--music-db", type=float, default=-16.0)
    p.add_argument("--captions", action="store_true")
    p.add_argument("--whisper-model", default=ac.DEFAULT_WHISPER)
    p.add_argument("--language")
    p.add_argument("--no-transcribe", action="store_true", help="skip Whisper; equal-length shots")
    p.add_argument("--no-clip", action="store_true", help="skip CLIP matching; rotate through clips")
    p.add_argument("-o", "--plan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_plan_args(sub.add_parser("plan", help="analyse and write an editable plan"))
    pr = sub.add_parser("render", help="render a plan")
    pr.add_argument("plan")
    for p in (pr, pa := sub.add_parser("auto", help="plan + render")):
        p.add_argument("--out")
        p.add_argument("--codec", choices=["hevc", "h264", "x264"], default="h264")
        p.add_argument("--workers", type=int, default=2, help="parallel shot encodes (2 suits 8 GB)")
    add_plan_args(pa)
    pe = sub.add_parser("export", help="write an OpenTimelineIO timeline for FilmCraft / Resolve")
    pe.add_argument("plan")
    pe.add_argument("-o", "--out")
    args = ap.parse_args(argv)
    try:
        if args.cmd in ("plan", "auto"):
            plan = make_plan(args)
            path = Path(args.plan) if args.plan else ac.default_plan_path(args.voice, ".vo.plan.json")
            ac.save_plan(plan, path)
            if args.cmd == "auto":
                render_plan(path, out=args.out, codec=args.codec, workers=args.workers)
        elif args.cmd == "render":
            render_plan(Path(args.plan), out=args.out, codec=args.codec, workers=args.workers)
        else:
            import otio_export

            otio_export.export_plan_file(Path(args.plan), Path(args.out) if args.out else None)
    except AutocutError as e:
        log(f"error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
