#!/usr/bin/env python3
"""montage: auto-edit many short clips to music, or to the rhythm of a reference video.

    # Clips + music -> a 30 s vertical edit cut on the beat
    python montage.py music --clips clips/ --music song.mp3 --length 30 --aspect vertical

    # Same soundtrack and cut rhythm as a reference video, with your own clips
    python montage.py reference ref.mp4 --clips clips/ [--aspect same]

    python montage.py render plan.montage.json      # re-render after editing the plan
    python montage.py export plan.montage.json      # .otio for EI / FilmCraft / Resolve

How it edits
  music mode: beats are found in the music (onset strength + tempo + dynamic-programming beat
  tracking). Shots change every 1/2/4/8 beats so each lasts about the --pace length. With
  --music-start auto the loudest stretch of the song is used (snapped to a bar), with a short fade in
  and out. Every clip is scanned for motion and brightness. Busy shots go on energetic beats and calm
  shots on calm ones; dark or frozen moments are avoided, and no moment is used twice.
  reference mode: the reference's cuts are detected (scene changes), and its soundtrack is used
  unchanged. Each reference shot gets a clip moment that looks like it (CLIP image similarity
  when available) and moves like it (motion level).

Everything is a plan first (`<name>.montage.json`): every shot's clip and in-point can be edited.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import autocut as ac
import voiceover_cut as vc
from autocut import AutocutError, log

ASPECTS = {"horizontal": (1920, 1080), "vertical": (1080, 1920), "square": (1080, 1080)}
PACE = {"fast": 1.2, "medium": 2.0, "slow": 3.2}
ANALYSIS_FPS = 5.0
SR = 22050
HOP = 512


# ---------------------------------------------------------------------------------------------
# Audio: beats (pure numpy; unit-tested on a synthetic click track)
# ---------------------------------------------------------------------------------------------


def decode_mono(path: str, sr: int = SR, start: float = 0.0, duration: float | None = None):
    import numpy as np

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise AutocutError(f"could not read audio from {path}: {proc.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def onset_envelope(y, sr: int = SR, hop: int = HOP):
    """Spectral flux: how much new energy appears in each ~23 ms frame."""
    import numpy as np

    n_fft = 2048
    if len(y) < n_fft:
        return np.zeros(0, dtype=np.float32)
    frames = 1 + (len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    win = np.hanning(n_fft).astype(np.float32)
    env = np.empty(frames, dtype=np.float32)
    prev = None
    for i in range(0, frames, 2048):  # chunked: bounded memory for long songs
        mag = np.log1p(10 * np.abs(np.fft.rfft(y[idx[i : i + 2048]] * win, axis=1)))
        d = np.diff(mag, axis=0, prepend=mag[:1] if prev is None else prev[None, :])
        env[i : i + len(mag)] = np.maximum(d, 0).sum(axis=1)
        prev = mag[-1]
    env -= env.mean()
    sd = env.std()
    return env / sd if sd > 0 else env


def estimate_period(env, sr: int = SR, hop: int = HOP, lo_bpm: float = 70, hi_bpm: float = 180) -> float:
    """Beat period in frames from the onset autocorrelation, with a soft preference near 120 BPM."""
    import numpy as np

    fps = sr / hop
    n = len(env)
    if n < 8:
        return fps / 2
    ac_ = np.correlate(env, env, mode="full")[n - 1 :]
    lags = np.arange(len(ac_))
    lo, hi = int(fps * 60 / hi_bpm), int(fps * 60 / lo_bpm)
    hi = min(hi, len(ac_) - 1)
    if hi <= lo:
        return fps / 2
    bpm = 60 * fps / np.maximum(lags[lo : hi + 1], 1)
    weight = np.exp(-0.5 * (np.log2(bpm / 120.0) / 0.9) ** 2)
    best = lo + int(np.argmax(ac_[lo : hi + 1] * weight))
    # Parabolic interpolation between lags: whole-frame lags alone quantise 120 BPM to ~117.
    if 0 < best < len(ac_) - 1:
        y0, y1, y2 = ac_[best - 1], ac_[best], ac_[best + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            return float(best + 0.5 * (y0 - y2) / den)
    return float(best)


def track_beats(env, period: float, tightness: float = 100.0) -> list[int]:
    """Dynamic-programming beat tracker (Ellis 2007): beats sit on strong onsets, about one
    period apart."""
    import numpy as np

    n = len(env)
    if n == 0 or period <= 1:
        return []
    score = env.astype(np.float64).copy()
    back = np.full(n, -1, dtype=np.int64)
    lo, hi = int(round(period / 2)), int(round(period * 2))
    for t in range(lo, n):
        start = max(0, t - hi)
        stop = t - lo
        if stop <= start:
            continue
        prev = np.arange(start, stop)
        cand = score[prev] - tightness * np.log((t - prev) / period) ** 2
        k = int(np.argmax(cand))
        if cand[k] > 0:
            score[t] += cand[k]
            back[t] = prev[k]
    # Start from the best-scoring frame in the last period and walk back.
    tail = max(0, n - int(period))
    t = tail + int(np.argmax(score[tail:]))
    beats = []
    while t >= 0:
        beats.append(t)
        t = int(back[t])
    return beats[::-1]


def beat_times(path: str, start: float = 0.0, duration: float | None = None) -> tuple[list[float], float, list[float]]:
    """(beat times in seconds from `start`, tempo in BPM, onset strength per beat)."""
    y = decode_mono(path, SR, start, duration)
    env = onset_envelope(y)
    if len(env) < 16 or float(abs(env).max(initial=0)) < 1e-6:
        return [], 0.0, []
    period = estimate_period(env)
    frames = track_beats(env, period)
    fps = SR / HOP
    strength = [float(env[max(0, f - 2) : f + 3].max()) for f in frames]
    # A frame's spectrum spans 2048 samples from its start, so an onset registers ~1.5 hops
    # before the sound: shift by window/2 + hop (measured: -66 ms -> about 0 on a kick track).
    lag = (1024 + HOP) / SR
    return [f / fps + lag for f in frames], 60 * fps / period, strength


def loudest_window(path: str, length: float, total: float) -> float:
    """Start time of the loudest `length`-second stretch (RMS over 0.5 s blocks)."""
    import numpy as np

    if total <= length + 0.5:
        return 0.0
    y = decode_mono(path, 8000)
    block = 4000
    nb = len(y) // block
    if nb == 0:
        return 0.0
    rms = np.sqrt((y[: nb * block].reshape(nb, block) ** 2).mean(axis=1))
    w = max(1, int(length / 0.5))
    if nb <= w:
        return 0.0
    c = np.concatenate([[0], np.cumsum(rms)])
    sums = c[w:] - c[:-w]
    # Prefer not to start in the first 3% (intros); ties go to the earlier start.
    return float(int(np.argmax(sums)) * 0.5)


def cut_frames_from_beats(beats: list[float], strength: list[float], length: float, fps: float, target: float) -> tuple[list[int], list[float]]:
    """Shot boundaries in frames (0 … total) on beats, about `target` seconds apart, and an
    energy value per shot (0–1 rank of beat strength)."""
    total = round(length * fps)
    if total <= 0:
        return [0], []
    usable = [(b, s) for b, s in zip(beats, strength) if 0 < b < length]
    if len(usable) >= 2:
        period = (usable[-1][0] - usable[0][0]) / max(1, len(usable) - 1)
        per = max(1, round(target / period)) if period > 0 else 1
        per = min([1, 2, 4, 8], key=lambda p: abs(p - per))
        cuts = [round(b * fps) for b, _ in usable[::per]]
    else:  # no beat found: an even grid
        n = max(1, round(length / target))
        cuts = [round(total * i / n) for i in range(1, n)]
    bounds = sorted({0, total, *[c for c in cuts if 0 < c < total]})
    # Drop slivers shorter than a third of the target (merge into the previous shot).
    min_len = max(1, round(target * fps / 3))
    clean = [bounds[0]]
    for b in bounds[1:]:
        if b - clean[-1] < min_len and b != total:
            continue
        if b == total and b - clean[-1] < min_len and len(clean) > 1:
            clean[-1] = total
            continue
        clean.append(b)
    energy = []
    for a, b in zip(clean, clean[1:]):
        vals = [s for t, s in zip(beats, strength) if a <= t * fps < b]
        energy.append(max(vals) if vals else 0.0)
    return clean, rank01(energy)


def rank01(values: list[float]) -> list[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    for r, i in enumerate(order):
        out[i] = r / max(1, len(values) - 1)
    return out


# ---------------------------------------------------------------------------------------------
# Video analysis: motion + brightness per clip (cached)
# ---------------------------------------------------------------------------------------------


def analyse_clip(path: str) -> dict:
    """Per-frame (at 5 fps) motion (mean abs difference) and brightness of a tiny grey copy."""
    import numpy as np

    key = ac.file_key(path, "motion-v1", str(ANALYSIS_FPS))
    cache = ac.CACHE_DIR / "motion" / f"{key}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    w, h = 64, 36
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-hwaccel", "videotoolbox", "-i", path, "-an",
         "-vf", f"fps={ANALYSIS_FPS:g},scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
        capture_output=True,
    )
    n = len(proc.stdout) // (w * h)
    if proc.returncode != 0 or n == 0:
        res = {"motion": [], "brightness": []}
    else:
        f = np.frombuffer(proc.stdout[: n * w * h], dtype=np.uint8).reshape(n, h, w).astype(np.float32)
        motion = np.concatenate([[0.0], np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))])
        if n > 1:
            motion[0] = motion[1]
        res = {"motion": [round(float(x), 3) for x in motion], "brightness": [round(float(x), 2) for x in f.mean(axis=(1, 2))]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(res))
    return res


def window_stats(series: list[float], start: float, dur: float) -> float:
    a = int(start * ANALYSIS_FPS)
    b = max(a + 1, int((start + dur) * ANALYSIS_FPS))
    vals = series[a:b]
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------------------------
# Choosing footage (pure logic, unit-tested)
# ---------------------------------------------------------------------------------------------


def candidate_windows(clips: list[dict], step: float = 0.5) -> list[dict]:
    """Candidate in-points every `step` s of every clip, with motion/brightness at that point."""
    out = []
    for ci, c in enumerate(clips):
        dur = float(c["duration"])
        t = 0.0
        while t < max(dur, 0.01):
            out.append({"clip": ci, "t": round(t, 3)})
            t += step
    return out


def choose_footage(
    shots: list[dict],
    clips: list[dict],
    fps: float,
    *,
    sims: list[list[float]] | None = None,
    samples: list[tuple[int, float]] | None = None,
    step: float = 0.5,
) -> list[dict]:
    """Pick (clip, in) for each shot. A shot has `frames` and `energy` (0–1, how busy the moment
    should look). Each clip has `duration`, `motion`, `brightness` (5 fps series). Optional
    `sims[shot][sample]` (CLIP similarity to the shot's look) with `samples[j] = (clip, time)`.

    Scoring: match motion rank to energy, avoid dark/frozen moments, never reuse a moment, spread
    use across clips, avoid the same clip twice in a row."""
    if not clips:
        raise AutocutError("no usable clips")
    cands = candidate_windows(clips, step)
    # Motion of every candidate for each distinct shot length is costly; use a 2 s window as the
    # descriptor (the look of the moment), ranked across all candidates.
    desc = [window_stats(clips[c["clip"]].get("motion", []), c["t"], 2.0) for c in cands]
    bright = [window_stats(clips[c["clip"]].get("brightness", []), c["t"], 2.0) for c in cands]
    mrank = rank01(desc)
    by_clip: dict[int, list[tuple[float, int]]] = {}
    for j, (c, t) in enumerate(samples or []):
        by_clip.setdefault(c, []).append((t, j))
    used: list[list[tuple[float, float]]] = [[] for _ in clips]
    uses = [0] * len(clips)
    prev = -1
    picks = []
    for si, shot in enumerate(shots):
        dur = shot["frames"] / fps
        energy = float(shot.get("energy", 0.5))
        best, best_score = None, -math.inf
        sim_row = sims[si] if sims and si < len(sims) else None
        for k, c in enumerate(cands):
            ci, t = c["clip"], c["t"]
            cdur = float(clips[ci]["duration"])
            start = min(t, max(0.0, cdur - dur))
            score = 1.0 - abs(mrank[k] - energy)
            if bright[k] < 18:  # nearly black
                score -= 1.0
            if desc[k] < 0.05 and len(clips[ci].get("motion", [])) > 2:  # frozen frame
                score -= 0.3
            if any(not (start + dur <= a or start >= b) for a, b in used[ci]):
                score -= 2.0
            score -= 0.15 * uses[ci]
            if ci == prev:
                score -= 0.4
            if cdur < dur:
                score -= 0.5
            if sim_row is not None and samples:
                score += 4.0 * best_sim(sim_row, by_clip.get(ci, []), start, dur)
            if score > best_score + 1e-9:
                best, best_score = (ci, start), score
        if best is None:
            raise AutocutError("could not choose footage")
        ci, start = best
        start = math.floor(start * fps) / fps
        used[ci].append((start, start + dur))
        uses[ci] += 1
        prev = ci
        picks.append({"clip": ci, "in": round(start, 6), "score": round(best_score, 3)})
    return picks


def best_sim(row: list[float], clip_samples: list[tuple[float, int]], start: float, dur: float) -> float:
    """Best similarity among one clip's indexed frames that fall in (or next to) the window."""
    vals = [row[j] for t, j in clip_samples if start - 1.0 <= t <= start + dur + 1.0 and j < len(row)]
    return max(vals) if vals else (min(row) if row else 0.0)


# ---------------------------------------------------------------------------------------------
# Reference analysis
# ---------------------------------------------------------------------------------------------


def detect_cuts(path: str, threshold: float = 0.3, min_gap: float = 0.25) -> list[float]:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-hwaccel", "videotoolbox", "-i", path, "-an",
         "-vf", f"scale=320:-2,select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    times = [float(x) for x in re.findall(r"pts_time:([\d.]+)", proc.stderr)]
    out: list[float] = []
    for t in sorted(times):
        if t > min_gap and (not out or t - out[-1] >= min_gap):
            out.append(t)
    return out


def _embed_worker(frame_requests: list[tuple[str, float]], to_index: list[str], every: float) -> dict:
    """CLIP embeddings of single frames (path, time) and of whole clips (every `every` s)."""
    import numpy as np
    import open_clip
    import torch
    from PIL import Image

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(vc.CLIP_MODEL[0], pretrained=vc.CLIP_MODEL[1], device=device)
    model.eval()

    def embed(frames) -> list:
        out = []
        with torch.no_grad():
            for i in range(0, len(frames), 16):
                batch = torch.stack([preprocess(Image.fromarray(f)) for f in frames[i : i + 16]]).to(device)
                e = model.encode_image(batch)
                out.append((e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy())
        return np.concatenate(out).tolist() if out else []

    singles = []
    for path, t in frame_requests:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1", "-an",
             "-vf", "scale=224:224:force_original_aspect_ratio=increase,crop=224:224", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True,
        )
        if len(proc.stdout) >= 224 * 224 * 3:
            singles.append(np.frombuffer(proc.stdout[: 224 * 224 * 3], dtype=np.uint8).reshape(224, 224, 3))
        else:
            singles.append(np.zeros((224, 224, 3), dtype=np.uint8))
    clips = {p: embed(vc._frames_rgb(p, every)) for p in to_index}
    return {"frames": embed(singles), "clips": clips}


def clip_image_sims(clip_paths: list[str], frame_requests: list[tuple[str, float]]):
    """(samples, sims[request][sample]) by CLIP image-to-image similarity, sharing the
    voiceover mode's per-clip cache."""
    import numpy as np

    cache_dir = ac.CACHE_DIR / "clip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{vc.CLIP_MODEL[0]}-{vc.CLIP_MODEL[1]}-{vc.SAMPLE_EVERY}"
    keys = {c: cache_dir / f"{ac.file_key(c, tag)}.npy" for c in clip_paths}
    missing = [c for c in clip_paths if not keys[c].is_file()]
    log(f"CLIP: {len(frame_requests)} reference frames, {len(missing)} clips to index ({len(clip_paths) - len(missing)} cached)")
    res = ac.run_isolated(_embed_worker, frame_requests, missing, vc.SAMPLE_EVERY)
    for c, emb in res["clips"].items():
        np.save(keys[c], np.asarray(emb, dtype=np.float32).reshape(-1, vc.EMBED_DIM))
    samples, mats = [], []
    for ci, c in enumerate(clip_paths):
        m = np.load(keys[c])
        mats.append(m)
        samples += [(ci, k * vc.SAMPLE_EVERY) for k in range(len(m))]
    if not samples:
        return [], []
    img = np.concatenate(mats)
    ref = np.asarray(res["frames"], dtype=np.float32).reshape(-1, vc.EMBED_DIM)
    return samples, (ref @ img.T).tolist()


# ---------------------------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------------------------


def load_clips(items: list[str]) -> list[dict]:
    clips = []
    paths = vc.collect_clips(items)
    log(f"analysing {len(paths)} clips (motion, brightness)")
    t0 = time.time()
    for p in paths:
        info = ac.probe(p)
        if not info.has_video or info.duration <= 0.2:
            log(f"skipping (no usable video): {p}")
            continue
        a = analyse_clip(p)
        clips.append({"path": p, "duration": info.duration, "motion": a["motion"], "brightness": a["brightness"], "rotation": info.rotation})
    log(f"clip analysis {time.time() - t0:.1f}s")
    if not clips:
        raise AutocutError("no usable video clips")
    return clips


def output_size(aspect: str, ref: ac.MediaInfo | None = None) -> tuple[int, int]:
    if aspect == "same" and ref is not None:
        w, h = ref.display_size
        if w <= 0 or h <= 0:
            return ASPECTS["horizontal"]
        scale = min(1.0, 1920 / max(w, h))  # keep at most 1080p-class
        return max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)
    if aspect not in ASPECTS:
        raise AutocutError(f"unknown aspect {aspect!r} (horizontal, vertical, square, same)")
    return ASPECTS[aspect]


def plan_shots(clips: list[dict], bounds: list[int], energy: list[float], fps: float, sims=None, samples=None) -> list[dict]:
    shots = [{"frames": b - a, "energy": e} for a, b, e in zip(bounds, bounds[1:], energy or [0.5] * (len(bounds) - 1))]
    picks = choose_footage(shots, clips, fps, sims=sims, samples=samples)
    return [{"clip": clips[p["clip"]]["path"], "in": p["in"], "frames": s["frames"], "energy": round(s["energy"], 3), "score": p["score"]} for s, p in zip(shots, picks)]


def make_music_plan(args) -> dict:
    ac.require_tools()
    music = str(Path(args.music).resolve())
    minfo = ac.probe(music)
    if not minfo.has_audio:
        raise AutocutError("the music file has no audio")
    length = float(args.length)
    if length <= 0:
        raise AutocutError("--length must be positive")
    if minfo.duration < length:
        log(f"music is only {minfo.duration:.1f}s: the edit will be {minfo.duration:.1f}s")
        length = minfo.duration
    fps = float(args.fps)
    width, height = output_size(args.aspect)
    clips = load_clips(args.clips)
    if args.music_start == "auto":
        rough = loudest_window(music, length, minfo.duration)
    else:
        rough = max(0.0, min(float(args.music_start), max(0.0, minfo.duration - length)))
    # Beats over a slightly larger window, then snap the start to the nearest bar-ish beat.
    pre = min(rough, 4.0)
    beats, tempo, strength = beat_times(music, rough - pre, length + pre + 1.0)
    start = rough
    if beats:
        near = min(beats, key=lambda b: abs(b - pre))
        start = max(0.0, rough - pre + near)
        beats = [b - near for b in beats]
        keep = [(b, s) for b, s in zip(beats, strength) if b >= 0]
        beats, strength = [b for b, _ in keep], [s for _, s in keep]
    start = max(0.0, min(start, max(0.0, minfo.duration - length)))
    log(f"music: tempo ~{tempo:.0f} BPM, using {start:.1f}s–{start + length:.1f}s, {len(beats)} beats")
    bounds, energy = cut_frames_from_beats(beats, strength, length, fps, PACE.get(args.pace, 2.0))
    shots = plan_shots(clips, bounds, energy, fps)
    log(f"{len(shots)} shots from {len(clips)} clips")
    return {
        "version": ac.PLAN_VERSION,
        "mode": "montage",
        "kind": "music",
        "width": width, "height": height, "fps": fps, "video_bitrate": "12M",
        "audio": {"path": music, "start": round(start, 3), "fade_in": 0.3 if start > 0.5 else 0.0, "fade_out": 1.5, "normalize": True},
        "tempo": round(tempo, 1),
        "clips": [{"path": c["path"], "duration": c["duration"]} for c in clips],
        "shots": shots,
    }


def make_reference_plan(args) -> dict:
    ac.require_tools()
    ref = str(Path(args.reference).resolve())
    rinfo = ac.probe(ref)
    if not rinfo.has_video:
        raise AutocutError("the reference has no video")
    fps = float(args.fps) if args.fps else (rinfo.fps if rinfo.fps and round(rinfo.fps) in (24, 25, 30, 50, 60) else 30.0)
    fps = round(fps, 3)
    width, height = output_size(args.aspect, rinfo)
    clips = load_clips(args.clips)
    length = rinfo.duration
    cuts = detect_cuts(ref, args.threshold)
    if not cuts and rinfo.has_audio:
        log("no scene cuts found in the reference: cutting on its beats instead")
        beats, _, strength = beat_times(ref)
        bounds, _ = cut_frames_from_beats(beats, strength, length, fps, 2.0)
    else:
        total = round(length * fps)
        bounds = sorted({0, total, *[round(t * fps) for t in cuts if 0 < round(t * fps) < total]})
    rmotion = analyse_clip(ref)["motion"]
    energy = rank01([window_stats(rmotion, a / fps, (b - a) / fps) for a, b in zip(bounds, bounds[1:])])
    sims = samples = None
    if not args.no_clip:
        try:
            requests = [(ref, (a + b) / 2 / fps) for a, b in zip(bounds, bounds[1:])]
            samples, sims = clip_image_sims([c["path"] for c in clips], requests)
        except ImportError:
            log("open_clip not installed: matching on motion only")
    log(f"reference: {len(bounds) - 1} shots over {length:.1f}s, output {width}x{height} @ {fps:g}")
    shots = plan_shots(clips, bounds, energy, fps, sims, samples)
    return {
        "version": ac.PLAN_VERSION,
        "mode": "montage",
        "kind": "reference",
        "reference": ref,
        "width": width, "height": height, "fps": fps, "video_bitrate": "12M",
        "audio": {"path": ref, "start": 0.0, "fade_in": 0.0, "fade_out": 0.0, "normalize": False} if rinfo.has_audio else None,
        "clips": [{"path": c["path"], "duration": c["duration"]} for c in clips],
        "shots": shots,
    }


# ---------------------------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------------------------


def render_plan(plan_path: Path, *, out: str | None, workers: int = 2, codec: str = "h264") -> Path:
    ac.require_tools()
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text())
    shots = [s for s in plan.get("shots", []) if int(s.get("frames", 0)) > 0]
    if not shots:
        raise AutocutError("the plan has no shots")
    for s in shots:
        if not os.path.isfile(s["clip"]):
            raise AutocutError(f"clip missing: {s['clip']}")
    fps = float(plan["fps"])
    total = sum(int(s["frames"]) for s in shots) / fps
    stem = plan_path.name.replace(".montage.json", "").replace(".json", "")
    out_path = Path(out).resolve() if out else plan_path.with_name(stem + ".mp4")
    audio = plan.get("audio") or {}
    originals = [s["clip"] for s in shots] + ([audio["path"]] if audio.get("path") else [])
    if any(out_path == Path(p).resolve() for p in originals):
        raise AutocutError("refusing to overwrite an original file")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = plan_path.with_name(stem + ".work")
    work.mkdir(exist_ok=True)
    files = [work / f"shot_{vc.shot_key(plan, s, codec)}.mp4" for s in shots]
    todo = [(s, f) for s, f in zip(shots, files) if not f.is_file()]
    log(f"rendering {len(todo)} of {len(shots)} shots ({len(shots) - len(todo)} reused), {workers} workers")
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, fut in enumerate([ex.submit(vc.render_shot, plan, s, f, codec) for s, f in todo], 1):
            fut.result()
            log(f"progress {i}/{len(todo)} shots")
    log(f"shots done in {time.time() - t0:.1f}s")
    for stale in work.glob("shot_*.mp4"):
        if stale not in files:
            stale.unlink()
    with tempfile.TemporaryDirectory(prefix="montage-") as tmp:
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{str(f).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for f in files))
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst)]
        if audio.get("path") and os.path.isfile(audio["path"]):
            cmd += ["-ss", f"{float(audio.get('start', 0)):.3f}", "-t", f"{total + 0.5:.3f}", "-i", audio["path"]]
            chain = [f"aresample=48000,apad,atrim=0:{total:.6f},asetpts=PTS-STARTPTS"]
            if float(audio.get("fade_in", 0)) > 0:
                chain.append(f"afade=t=in:st=0:d={float(audio['fade_in']):.3f}")
            if float(audio.get("fade_out", 0)) > 0:
                fo = min(float(audio["fade_out"]), total / 2)
                chain.append(f"afade=t=out:st={max(0.0, total - fo):.3f}:d={fo:.3f}")
            if audio.get("normalize"):
                chain.append("loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000")
            cmd += ["-filter_complex", f"[1:a]{','.join(chain)}[a]", "-map", "0:v", "-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-map", "0:v"]
        cmd += ["-c:v", "copy", "-t", f"{total:.6f}", "-movflags", "+faststart", str(out_path)]
        ac.run(cmd)
    log(f"done in {time.time() - t0:.1f}s -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def default_plan(out: str | None, fallback_stem: str) -> Path:
    if out:
        o = Path(out).resolve()
        return o.with_name(o.stem + ".montage.json")
    return Path.cwd() / f"{fallback_stem}.montage.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pm = sub.add_parser("music", help="clips + music -> beat-cut edit")
    pm.add_argument("--clips", nargs="+", required=True, help="video files and/or folders")
    pm.add_argument("--music", required=True)
    pm.add_argument("--length", type=float, default=30.0, help="seconds (default 30)")
    pm.add_argument("--aspect", choices=["horizontal", "vertical", "square"], default="vertical")
    pm.add_argument("--pace", choices=sorted(PACE), default="medium", help="shot length: fast ~1.2 s, medium ~2 s, slow ~3.2 s")
    pm.add_argument("--music-start", default="auto", help="'auto' (loudest part) or seconds into the song")
    pm.add_argument("--fps", type=float, default=30.0)
    pr = sub.add_parser("reference", help="same soundtrack and cut rhythm as a reference video")
    pr.add_argument("reference")
    pr.add_argument("--clips", nargs="+", required=True)
    pr.add_argument("--aspect", choices=["same", "horizontal", "vertical", "square"], default="same")
    pr.add_argument("--threshold", type=float, default=0.3, help="scene-cut sensitivity (lower finds more cuts)")
    pr.add_argument("--fps", type=float)
    pr.add_argument("--no-clip", action="store_true", help="match on motion only (no CLIP model)")
    for p in (pm, pr):
        p.add_argument("--out", help="output video (default: <name>.mp4 in the current folder)")
        p.add_argument("--plan-only", action="store_true", help="write the plan, don't render")
        p.add_argument("--workers", type=int, default=2)
    pre = sub.add_parser("render")
    pre.add_argument("plan")
    pre.add_argument("--out")
    pre.add_argument("--workers", type=int, default=2)
    pe = sub.add_parser("export")
    pe.add_argument("plan")
    pe.add_argument("-o", "--out")
    args = ap.parse_args(argv)
    try:
        result: dict = {}
        if args.cmd in ("music", "reference"):
            if args.cmd == "music":
                plan = make_music_plan(args)
                plan_path = default_plan(args.out, f"montage_{Path(args.music).stem}")
            else:
                plan = make_reference_plan(args)
                plan_path = default_plan(args.out, f"like_{Path(args.reference).stem}")
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            ac.save_plan(plan, plan_path)
            result["plan"] = str(plan_path)
            if not args.plan_only:
                video = render_plan(plan_path, out=args.out, workers=args.workers)
                result["outputs"] = [str(video)]
            import otio_export

            result["otio"] = str(otio_export.export_plan_file(plan_path))
        elif args.cmd == "render":
            result["outputs"] = [str(render_plan(Path(args.plan), out=args.out, workers=args.workers))]
        else:
            import otio_export

            result["otio"] = str(otio_export.export_plan_file(Path(args.plan), Path(args.out) if args.out else None))
        print(json.dumps(result))
    except AutocutError as e:
        log(f"error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
