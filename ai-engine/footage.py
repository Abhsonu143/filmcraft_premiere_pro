"""footage: what is in each clip and which moments of it are worth using (cached per clip).

For every clip, one decode pass at 5 fps on a small copy measures, per sample:
  motion     how much the picture changes (mean absolute difference)
  dx, dy     global camera movement (phase correlation), pixels of a 160-px-wide frame per sample
  sharp      sharpness (variance of the Laplacian); low = blurry or out of focus
  bright     mean luma (0–255); sat = mean saturation (0–1); warm = red minus blue balance
  shake      jitter of the camera path (high-frequency part of dx/dy)

Optional CLIP embeddings (one frame per second) give what the shot shows (wide aerial,
people, temple, sunset …), a look score ("cinematic, beautiful" vs "dull, blurry"), and
near-duplicate detection between shots.

`usable(clip)` then scores every 0.2 s moment, so the director can pick the best window of any
length inside a clip (never the shaky start of a drone take-off or a blurry frame).
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import autocut as ac
from autocut import log

FPS = 5.0
CLIP_EVERY = 1.0
VERSION = "footage-v2"

# Zero-shot vocabulary (CLIP text prompts). Groups are mutually exclusive scores per frame.
SHOT_SIZE = {
    "aerial": "an aerial drone shot of a landscape from high above",
    "wide": "a wide landscape shot",
    "medium": "a medium shot of people",
    "close": "a close-up shot",
}
SUBJECTS = {
    "people": "people",
    "face": "a smiling face of a person",
    "temple": "a temple",
    "village": "a village with houses",
    "fields": "green rice fields",
    "water": "a river or lake",
    "waterfall": "a waterfall",
    "trees": "trees and forest",
    "animals": "animals in a field",
    "sunset": "a golden sunset sky",
    "night": "a town at night with lights",
    "clouds": "dramatic clouds in the sky",
    "road": "a road",
    "rocks": "rocks and cliffs",
}
LOOK_GOOD = ["a beautiful cinematic photograph", "a stunning professional travel photo", "an epic landscape photo"]
LOOK_BAD = ["a blurry photo", "a boring dull photo", "a badly exposed dark photo", "a photo of the ground"]


# ---------------------------------------------------------------------------------------------
# Signal analysis (numpy)
# ---------------------------------------------------------------------------------------------


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Translation (dx, dy) that maps frame a onto b, and the correlation peak (confidence)."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1])).astype(np.float32)
    fa = np.fft.rfft2((a - a.mean()) * win)
    fb = np.fft.rfft2((b - b.mean()) * win)
    r = fb * np.conj(fa)
    r /= np.abs(r) + 1e-6
    c = np.fft.irfft2(r, s=a.shape)
    k = int(np.argmax(c))
    py, px = divmod(k, a.shape[1])
    peak = float(c.flat[k])
    # Parabolic sub-pixel refinement along each axis.
    def sub(m1: float, m0: float, p1: float) -> float:
        d = m1 - 2 * m0 + p1
        return 0.5 * (m1 - p1) / d if abs(d) > 1e-9 else 0.0
    H, W = a.shape
    fy = sub(c[(py - 1) % H, px], c[py, px], c[(py + 1) % H, px])
    fx = sub(c[py, (px - 1) % W], c[py, px], c[py, (px + 1) % W])
    py, px = py + fy, px + fx
    if py > a.shape[0] // 2:
        py -= a.shape[0]
    if px > a.shape[1] // 2:
        px -= a.shape[1]
    return float(px), float(py), peak


def laplacian_var(g: np.ndarray) -> float:
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def frame_stats(rgb: np.ndarray) -> dict:
    f = rgb.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    mx, mn = f.max(axis=2), f.min(axis=2)
    sat = np.where(mx > 1, (mx - mn) / np.maximum(mx, 1), 0)
    return {"bright": float(y.mean()), "sat": float(sat.mean()), "warm": float((r - b).mean() / 255), "sharp": laplacian_var(y), "gray": y}


def creation_time(path: str) -> float:
    """Capture time (seconds since epoch) from the container, else the file's modified time."""
    proc = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time", "-of", "csv=p=0", path], capture_output=True, text=True)
    s = proc.stdout.strip()
    if s:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return Path(path).stat().st_mtime


def analyse(path: str) -> dict:
    """Signals for one clip (cached by file identity)."""
    key = ac.file_key(path, VERSION)
    cache = ac.CACHE_DIR / "footage" / f"{key}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    info = ac.probe(path)
    dw, dh = info.display_size
    if dw <= 0 or dh <= 0:
        dw, dh = 16, 9
    w = 160 if dw >= dh else 90
    h = max(2, int(round(w * dh / dw / 2)) * 2)
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-hwaccel", "videotoolbox", "-i", path, "-an",
         "-vf", f"fps={FPS:g},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
    )
    n = len(proc.stdout) // (w * h * 3)
    frames = np.frombuffer(proc.stdout[: n * w * h * 3], dtype=np.uint8).reshape(n, h, w, 3) if n else np.zeros((0, h, w, 3), np.uint8)
    stats = [frame_stats(f) for f in frames]
    motion, dx, dy, conf = [0.0], [0.0], [0.0], [1.0]
    for a, b in zip(stats, stats[1:]):
        motion.append(float(np.abs(b["gray"] - a["gray"]).mean()))
        x, y, p = phase_shift(a["gray"], b["gray"])
        dx.append(x)
        dy.append(y)
        conf.append(p)
    if n > 1:
        motion[0], dx[0], dy[0] = motion[1], dx[1], dy[1]
    motion, dx, dy = motion[:n], dx[:n], dy[:n]
    # Shake: what is left of the camera path after smoothing.
    def hf(v):
        v = np.asarray(v, dtype=np.float32)
        if len(v) < 5:
            return np.zeros_like(v)
        k = np.ones(5) / 5
        return np.abs(v - np.convolve(v, k, mode="same"))
    shake = (hf(dx) + hf(dy)).tolist()
    res = {
        "path": path,
        "duration": info.duration,
        "fps": info.fps,
        "width": dw,
        "height": dh,
        "created": creation_time(path),
        "motion": [round(v, 3) for v in motion],
        "dx": [round(v, 2) for v in dx],
        "dy": [round(v, 2) for v in dy],
        "shake": [round(float(v), 3) for v in shake],
        "sharp": [round(s["sharp"], 1) for s in stats],
        "bright": [round(s["bright"], 1) for s in stats],
        "sat": [round(s["sat"], 3) for s in stats],
        "warm": [round(s["warm"], 3) for s in stats],
        "scale": w / max(1, dw),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(res))
    return res


# ---------------------------------------------------------------------------------------------
# CLIP semantics (one model load for all clips; per-clip cache)
# ---------------------------------------------------------------------------------------------


def keyframes_rgb(path: str, every: float, size: int = 224) -> np.ndarray:
    """One frame per `every` seconds, taken from keyframes only (no full 4K decode): each
    sample is the nearest keyframe. Falls back to a full decode when the file has none."""
    pr = subprocess.run(["ffprobe", "-v", "error", "-skip_frame", "nokey", "-select_streams", "v:0", "-show_entries", "frame=pts_time", "-of", "csv=p=0", path], capture_output=True, text=True)
    times = []
    for line in pr.stdout.split():
        try:
            times.append(float(line.strip(",")))
        except ValueError:
            pass
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-skip_frame", "nokey", "-i", path, "-an",
                           "-vf", f"scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}",
                           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    n = len(proc.stdout) // (size * size * 3)
    if n == 0 or not times:
        import voiceover_cut as vc

        return vc._frames_rgb(path, every)
    frames = np.frombuffer(proc.stdout[: n * size * size * 3], dtype=np.uint8).reshape(n, size, size, 3)
    times = np.asarray(times[:n]) - (times[0] if times else 0.0)
    dur = float(times[-1]) + every
    want = np.arange(0, max(dur, every), every)
    idx = [int(np.argmin(np.abs(times - t))) for t in want]
    return frames[idx]


def _semantic_worker(paths: list[str], every: float, prompts: list[str]) -> dict:
    import open_clip
    import torch
    from PIL import Image

    import voiceover_cut as vc

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(vc.CLIP_MODEL[0], pretrained=vc.CLIP_MODEL[1], device=device)
    model.eval()
    tok = open_clip.get_tokenizer(vc.CLIP_MODEL[0])
    out = {}
    with torch.no_grad():
        for p in paths:
            frames = keyframes_rgb(p, every)
            embs = []
            for i in range(0, len(frames), 16):
                batch = torch.stack([preprocess(Image.fromarray(f)) for f in frames[i : i + 16]]).to(device)
                e = model.encode_image(batch)
                embs.append((e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy())
            out[p] = np.concatenate(embs).tolist() if embs else []
        t = model.encode_text(tok(prompts).to(device))
        text = (t / t.norm(dim=-1, keepdim=True)).float().cpu().numpy().tolist()
    return {"images": out, "text": text}


def semantics(paths: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """(per-clip image embeddings [n x 512] at 1 per second, prompt embeddings by name)."""
    cache_dir = ac.CACHE_DIR / "footage-clip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    keys = {p: cache_dir / f"{ac.file_key(p, VERSION, str(CLIP_EVERY))}.npy" for p in paths}
    missing = [p for p in paths if not keys[p].is_file()]
    names = list(SHOT_SIZE) + list(SUBJECTS) + [f"good{i}" for i in range(len(LOOK_GOOD))] + [f"bad{i}" for i in range(len(LOOK_BAD))]
    prompts = list(SHOT_SIZE.values()) + list(SUBJECTS.values()) + LOOK_GOOD + LOOK_BAD
    log(f"understanding shots with CLIP: {len(missing)} clips to index, {len(paths) - len(missing)} cached")
    t0 = time.time()
    res = ac.run_isolated(_semantic_worker, missing, CLIP_EVERY, prompts)
    for p, e in res["images"].items():
        np.save(keys[p], np.asarray(e, dtype=np.float32).reshape(-1, 512))
    log(f"CLIP {time.time() - t0:.1f}s")
    imgs = {p: np.load(keys[p]) for p in paths}
    text = {n: np.asarray(v, dtype=np.float32) for n, v in zip(names, res["text"])}
    return imgs, text


def describe(emb: np.ndarray, text: dict[str, np.ndarray]) -> dict:
    """Per-frame descriptors from CLIP embeddings: shot size, top subjects, look score."""
    if len(emb) == 0:
        return {"size": [], "tags": [], "look": []}

    def softmax(names: list[str]) -> np.ndarray:
        m = np.stack([text[n] for n in names]) @ emb.T * 100.0
        m -= m.max(axis=0, keepdims=True)
        e = np.exp(m)
        return e / e.sum(axis=0, keepdims=True)

    size = softmax(list(SHOT_SIZE))
    subj = softmax(list(SUBJECTS))
    good = np.stack([text[f"good{i}"] for i in range(len(LOOK_GOOD))]) @ emb.T
    bad = np.stack([text[f"bad{i}"] for i in range(len(LOOK_BAD))]) @ emb.T
    look = (good.max(axis=0) - bad.max(axis=0)) * 10.0  # roughly -1 … +1
    names = list(SUBJECTS)
    return {
        "size": [list(SHOT_SIZE)[int(i)] for i in size.argmax(axis=0)],
        "size_p": size.T.tolist(),
        "tags": [[names[j] for j in np.argsort(-subj[:, i])[:2]] for i in range(subj.shape[1])],
        "look": look.tolist(),
    }


# ---------------------------------------------------------------------------------------------
# Usability of moments
# ---------------------------------------------------------------------------------------------


def moment_quality(a: dict) -> np.ndarray:
    """0–1 per 5-fps sample: sharp, well exposed, steady, and not a frozen frame."""
    sharp = np.asarray(a["sharp"], dtype=np.float32)
    if len(sharp) == 0:
        return np.zeros(0, np.float32)
    bright = np.asarray(a["bright"], dtype=np.float32)
    shake = np.asarray(a["shake"], dtype=np.float32)
    # sharpness relative to the clip's own best (scenes differ in texture)
    s = np.clip(sharp / max(1e-3, float(np.percentile(sharp, 90))), 0, 1)
    expo = np.clip(1 - np.abs(bright - 120) / 120, 0, 1)
    expo = np.where(bright < 25, expo * 0.3, expo)  # night shots stay possible but rank low
    steady = np.clip(1 - shake / 3.0, 0, 1)
    return 0.45 * s + 0.3 * expo + 0.25 * steady


def window_value(series, start: float, dur: float, fps: float = FPS) -> float:
    a = max(0, int(start * fps))
    b = max(a + 1, int(math.ceil((start + dur) * fps)))
    v = series[a:b]
    return float(np.mean(v)) if len(v) else 0.0


def camera_motion(a: dict, start: float, dur: float) -> tuple[float, float, float]:
    """(speed in frame-widths per second, direction x, direction y) over a window."""
    w = 1.0 / max(a.get("scale", 0.1), 1e-3)  # analysis px -> source px
    dx = window_value(a["dx"], start, dur) * FPS
    dy = window_value(a["dy"], start, dur) * FPS
    width = a.get("width", 1920) * a.get("scale", 0.1)
    speed = math.hypot(dx, dy) / max(width, 1)
    _ = w
    return speed, dx, dy
