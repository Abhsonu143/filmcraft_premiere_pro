"""reframe: true vertical / square / horizontal from any source, by following what matters.

For every piece of a clip a few frames per second are looked at with Apple Vision (on the
Neural Engine, no download): faces, people, the "objects" that stand out, and where the eye goes
(attention saliency). The crop window is put where it covers the most of that, with a pull
towards the centre and towards where it already is (no nervous jumps). The path is then
smoothed like a camera operator would: it holds still when the subject barely moves, pans
smoothly (speed-limited) when it does, and jumps only where the picture itself cuts.

Cinematic black bars (letterboxing) are found and cropped away first, so the shape is filled
with picture, not with black.

Without Vision (not macOS, or pyobjc missing) the crop stays centred.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from autocut import log

SAMPLE_FPS = 4.0
SAMPLE_W = 512
BINS = 100


def crop_fractions(src_w: int, src_h: int, W: int, H: int) -> tuple[float, float]:
    """Width and height of the crop window as fractions of the (active) source picture."""
    if src_w <= 0 or src_h <= 0:
        return 1.0, 1.0
    a_src, a_out = src_w / src_h, W / H
    if a_out < a_src:
        return a_out / a_src, 1.0
    return 1.0, a_src / a_out


def best_centre(profile: np.ndarray, width: float, prev: float | None, stick: float = 0.35) -> float:
    """Centre (0–1) of the window of `width` covering the most weight, preferring to stay put."""
    if width >= 0.999 or profile.sum() <= 0:
        return 0.5 if prev is None else prev
    n = len(profile)
    k = max(1, int(round(width * n)))
    cs = np.concatenate([[0.0], np.cumsum(profile)])
    sums = cs[k:] - cs[:-k]  # window starting at bin i
    centres = (np.arange(len(sums)) + k / 2) / n
    score = sums / max(1e-9, profile.sum())
    if prev is not None:
        score = score - stick * np.abs(centres - prev)
    i = int(np.argmax(score))
    # centre on what is inside the chosen window (not just "contain" it)
    seg = profile[i : i + k]
    if seg.sum() <= 0:
        return float(centres[i])
    c = float(((np.arange(i, i + k) + 0.5) / n * seg).sum() / seg.sum())
    if prev is not None and abs(c - prev) < 0.03:
        c = prev
    return float(np.clip(c, width / 2, 1 - width / 2))


def smooth_path(times: list[float], centres: list[float], breaks: set[int], width: float,
                hold: float = 0.06, max_speed: float = 0.35) -> list[tuple[float, float]]:
    """Camera-operator smoothing: hold when the subject barely moves, else a smooth speed-limited pan.
    `breaks`: sample indices where the picture cuts (the path may jump there)."""
    if not centres:
        return []
    lo, hi = width / 2, 1 - width / 2
    out: list[tuple[float, float]] = []
    segs, start = [], 0
    for i in range(1, len(centres) + 1):
        if i == len(centres) or i in breaks:
            segs.append((start, i))
            start = i
    for a, b in segs:
        c = np.asarray(centres[a:b], dtype=np.float64)
        t = np.asarray(times[a:b], dtype=np.float64)
        if len(c) >= 3:  # median of 3 removes single-frame detector glitches
            c = np.asarray([np.median(c[max(0, i - 1) : i + 2]) for i in range(len(c))])
        if np.ptp(c) < hold or len(c) == 1:
            v = float(np.clip(np.median(c), lo, hi))
            out += [(float(t[0]), v), (float(t[-1]), v)]
            continue
        g = np.exp(-0.5 * (np.arange(-4, 5) / 1.6) ** 2)
        pad = np.pad(c, 4, mode="edge")
        s = np.convolve(pad, g / g.sum(), mode="same")[4:-4]
        for i in range(1, len(s)):  # speed limit (fraction of the frame per second)
            dt = max(1e-3, t[i] - t[i - 1])
            s[i] = s[i - 1] + float(np.clip(s[i] - s[i - 1], -max_speed * dt, max_speed * dt))
        out += [(float(ti), float(np.clip(si, lo, hi))) for ti, si in zip(t, s)]
    return out


def to_fraction(centre: float, width: float) -> float:
    """Crop centre -> the renderer's position (0 = left/top edge, 1 = right/bottom edge)."""
    if width >= 0.999:
        return 0.5
    return float(np.clip((centre - width / 2) / (1 - width), 0, 1))


def letterbox(grays: list[np.ndarray]) -> tuple[float, float, float, float] | None:
    """Active picture (x0, y0, x1, y1 as fractions) when every sample has the same black bars."""
    if not grays:
        return None
    stack = np.stack(grays).astype(np.float32)
    # per row: the median across the width (subtitle text inside a bar doesn't count), then the
    # brightest of all samples (a bar is black in every frame)
    rows = np.median(stack, axis=2).max(axis=0)
    cols = np.median(stack, axis=1).max(axis=0)
    h, w = rows.shape[0], cols.shape[0]

    def bar(v: np.ndarray) -> int:
        n = 0
        while n < len(v) and v[n] < 20:
            n += 1
        return n

    top, bottom = bar(rows), bar(rows[::-1])
    left, right = bar(cols), bar(cols[::-1])
    box = [0.0, 0.0, 1.0, 1.0]
    found = False
    if top / h >= 0.05 and bottom / h >= 0.05 and abs(top - bottom) / h < 0.04 and (top + bottom) / h < 0.5:
        box[1], box[3] = (top + 1) / h, (h - bottom - 1) / h
        found = True
    if left / w >= 0.05 and right / w >= 0.05 and abs(left - right) / w < 0.04 and (left + right) / w < 0.5:
        box[0], box[2] = (left + 1) / w, (w - right - 1) / w
        found = True
    return (box[0], box[1], box[2], box[3]) if found else None


def _vision_boxes(paths: list[str]) -> list[dict]:
    """Per image: faces, humans, salient objects and attention boxes (x0, y0, x1, y1, weight), top-left origin."""
    import objc
    import Vision
    from Foundation import NSURL

    out = []
    for p in paths:
        with objc.autorelease_pool():
            h = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(p), None)
            face = Vision.VNDetectFaceRectanglesRequest.alloc().init()
            hum = Vision.VNDetectHumanRectanglesRequest.alloc().init()
            try:
                hum.setUpperBodyOnly_(False)
            except AttributeError:
                pass
            obj = Vision.VNGenerateObjectnessBasedSaliencyImageRequest.alloc().init()
            att = Vision.VNGenerateAttentionBasedSaliencyImageRequest.alloc().init()
            ok = h.performRequests_error_([face, hum, obj, att], None)
            res: dict = {"faces": [], "humans": [], "objects": [], "attention": []}
            if ok[0] if isinstance(ok, tuple) else ok:
                def box(r, wgt):
                    b = r.boundingBox()
                    return (b.origin.x, 1 - b.origin.y - b.size.height, b.origin.x + b.size.width, 1 - b.origin.y, wgt)

                res["faces"] = [box(r, float(r.confidence())) for r in (face.results() or [])]
                res["humans"] = [box(r, float(r.confidence())) for r in (hum.results() or [])]
                for key, rq in (("objects", obj), ("attention", att)):
                    for r in rq.results() or []:
                        res[key] += [box(o, float(o.confidence())) for o in (r.salientObjects() or [])]
            out.append(res)
    return out


def profile_of(det: dict, axis: int, talk: bool, active: tuple[float, float, float, float] | None) -> np.ndarray:
    """Importance along one axis (0 = x, 1 = y) in BINS bins, in active-picture coordinates."""
    prof = np.zeros(BINS)
    a0, a1 = (active[axis], active[axis + 2]) if active else (0.0, 1.0)
    span = max(1e-3, a1 - a0)
    weights = {"faces": 3.0 if talk else 1.5, "humans": 1.5, "objects": 1.0, "attention": 0.35}
    for key, wgt in weights.items():
        for b in det.get(key, []):
            lo = (b[axis] - a0) / span
            hi = (b[axis + 2] - a0) / span
            lo, hi = max(0.0, lo), min(1.0, hi)
            if hi <= lo:
                continue
            i0, i1 = int(lo * BINS), max(int(lo * BINS) + 1, int(np.ceil(hi * BINS)))
            area = (b[2] - b[0]) * (b[3] - b[1])
            mass = wgt * max(0.2, float(b[4])) * (1.0 if key in ("faces", "attention") else min(1.0, 0.3 + 4 * area))
            prof[i0:i1] += mass / (i1 - i0)
    x = (np.arange(BINS) + 0.5) / BINS
    prof += 0.25 * np.exp(-0.5 * ((x - 0.5) / 0.22) ** 2) / BINS * 10  # the game camera / cameraman centres things
    return prof


def sample_frames(src: str, start: float, dur: float, folder: Path, tag: str) -> list[tuple[float, Path]]:
    pattern = folder / f"{tag}_%04d.jpg"
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{max(0.0, start):.3f}", "-t", f"{max(0.3, dur):.3f}",
                    "-i", src, "-an", "-vf", f"fps={SAMPLE_FPS:g},scale={SAMPLE_W}:-2", "-q:v", "5", str(pattern)], capture_output=True)
    files = sorted(folder.glob(f"{tag}_*.jpg"))
    return [((i + 0.5) / SAMPLE_FPS, f) for i, f in enumerate(files)]


def reframe_pieces(src: str, pieces: list[tuple[float, float]], src_w: int, src_h: int, W: int, H: int, talk: bool) -> list[dict]:
    """Per source piece: {"active": box|None, "path": [(source_time, cx, cy)], "w": crop_w, "h": crop_h}
    with cx, cy the crop centre in active-picture fractions."""
    try:
        import Vision  # noqa: F401
        have_vision = True
    except ImportError:
        have_vision = False
        log("Apple Vision unavailable: centred crop")
    from PIL import Image

    out = []
    with tempfile.TemporaryDirectory(prefix="ei-reframe-") as tmp:
        for k, (a, b) in enumerate(pieces):
            samples = sample_frames(src, a, b - a, Path(tmp), f"p{k}")
            grays = []
            for _, f in samples:
                try:
                    grays.append(np.asarray(Image.open(f).convert("L"), dtype=np.uint8))
                except OSError:
                    pass
            active = letterbox(grays) if len(grays) >= 2 else None
            aw = src_w * ((active[2] - active[0]) if active else 1.0)
            ah = src_h * ((active[3] - active[1]) if active else 1.0)
            cw, ch = crop_fractions(int(aw), int(ah), W, H)
            res = {"active": active, "w": cw, "h": ch, "path": [(a, 0.5, 0.5), (b, 0.5, 0.5)]}
            if have_vision and samples and (cw < 0.999 or ch < 0.999):
                dets = _vision_boxes([str(f) for _, f in samples])
                # where the picture itself cuts, the crop may jump
                breaks = set()
                for i in range(1, len(grays)):
                    if grays[i].shape == grays[i - 1].shape:
                        g0 = grays[i - 1][::8, ::8].astype(np.float32)
                        g1 = grays[i][::8, ::8].astype(np.float32)
                        if np.abs(g1 - g0).mean() > 38:
                            breaks.add(i)
                times = [a + t for t, _ in samples][: len(dets)]
                paths = []
                for axis, width in ((0, cw), (1, ch)):
                    if width >= 0.999:
                        paths.append([(t, 0.5) for t in times])
                        continue
                    cs, prev = [], None
                    for i, d in enumerate(dets):
                        c = best_centre(profile_of(d, axis, talk, active), width, None if i in breaks else prev)
                        cs.append(c)
                        prev = c
                    paths.append(smooth_path(times, cs, breaks, width))
                # merge both axes onto the x path's times
                px, py = paths
                ty = [t for t, _ in py]
                vy = [v for _, v in py]
                res["path"] = [(t, x, float(np.interp(t, ty, vy))) for t, x in px] if px else res["path"]
            out.append(res)
    return out


def track_for_shot(frame: dict, shot_in: float, speed: float, out_len: float) -> tuple[list[list[float]], list[float] | None]:
    """The renderer's `track` keys ([t_out, fx, fy]) for a shot that plays source shot_in.. at `speed`."""
    keys = []
    src_end = shot_in + out_len * speed
    pts = frame["path"]
    ts = [p[0] for p in pts]
    for t, cx, cy in pts:
        if shot_in - 0.3 <= t <= src_end + 0.3:
            keys.append([round((t - shot_in) / speed, 3), round(to_fraction(cx, frame["w"]), 4), round(to_fraction(cy, frame["h"]), 4)])
    if not keys and pts:  # a short shot between two samples
        mid = shot_in + out_len * speed / 2
        cx = float(np.interp(mid, ts, [p[1] for p in pts]))
        cy = float(np.interp(mid, ts, [p[2] for p in pts]))
        keys = [[0.0, round(to_fraction(cx, frame["w"]), 4), round(to_fraction(cy, frame["h"]), 4)]]
    # thin out keys that add nothing (holds)
    thin = [k for i, k in enumerate(keys) if i in (0, len(keys) - 1) or abs(k[1] - keys[i - 1][1]) > 1e-3 or abs(k[1] - keys[i + 1][1]) > 1e-3
            or abs(k[2] - keys[i - 1][2]) > 1e-3 or abs(k[2] - keys[i + 1][2]) > 1e-3]
    active = frame.get("active")
    return thin, (list(active) if active else None)
