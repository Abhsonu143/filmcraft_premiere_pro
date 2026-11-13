#!/usr/bin/env python3
"""tv_light: play your clip on the big TV in a recording of your room, with the TV backlight
(Apex / Ambilight style) matching your clip's colours: done in software, no HDMI sync box.

Recordings: a folder of videos of the room (horizontal and/or vertical). Each recording is
analysed once and cached (~/.cache/autocut/tvlight):
  * orientation (rotation metadata), duration, frame rate;
  * the TV screen corners: from a one-time calibration (four corners on one frame; EI suggests
    them) then tracked through the whole recording as the camera moves (SIFT on the room, never
    on the screens, whose pictures change), 10 times a second;
  * the backlight glow around the TV (left, top, right, below) and the colours at the edges of
    what the TV showed, every 0.1 s, from which EI learns how this room answers the screen:
    "this colour at this edge -> this glow on this wall" (the recordings were made with real
    HDMI sync, so they teach it);
  * the other screens (monitor, laptop) and lamps: bright in every frame -> never recoloured.

Making a video for a clip:
  1. the glow your clip would cause is predicted for every 0.1 s (from its edge colours, through
     the learned room response);
  2. the recording moments whose real glow is closest are chosen as one smooth path (mostly
     continuous playback; jumps only where the camera view nearly matches, preferably on a cut
     in your clip; short stretches may repeat);
  3. what is left of the difference is closed by relighting: the glow near each TV edge is
     recoloured towards the predicted colour (weaker with distance; screens and lamps untouched);
  4. your clip is placed on the TV with the tracked perspective and matched to how the camera
     sees that screen (level, contrast, a touch of softness and grain).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

import autocut as ac
from autocut import AutocutError, log

HZ = 10.0
AW = 960  # analysis width (long side) for tracking and colours
VERSION = "tvlight-v1"
ZONES = ("left", "top", "right", "bottom")


def cache_dir() -> Path:
    d = ac.CACHE_DIR / "tvlight"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------
# Recordings
# ---------------------------------------------------------------------------------------------


def probe_rec(path: str) -> dict:
    """Orientation (after rotation), size, duration, fps."""
    info = ac.probe(path)
    dw, dh = info.display_size
    return {"path": path, "duration": info.duration, "fps": info.fps or 30.0, "width": dw, "height": dh,
            "orientation": "vertical" if dh > dw else "horizontal"}


def scan(folder: str) -> list[dict]:
    import voiceover_cut as vc

    recs = []
    for p in vc.collect_clips([folder]):
        try:
            recs.append(probe_rec(p))
        except AutocutError as e:
            log(f"skipping {Path(p).name}: {e}")
    return recs


def decode(path: str, start: float, dur: float | None, hz: float, w_long: int = AW):
    """Frames (BGR uint8) at `hz` from `start`, scaled so the long side is w_long; rotation applied."""
    info = probe_rec(path)
    if info["width"] >= info["height"]:
        W, H = w_long, int(round(w_long * info["height"] / info["width"] / 2)) * 2
    else:
        H, W = w_long, int(round(w_long * info["width"] / info["height"] / 2)) * 2
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-i", path, "-an", "-vf", f"fps={hz:g},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = W * H * 3
    try:
        while True:
            buf = proc.stdout.read(n) if proc.stdout else b""
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    finally:
        proc.kill()
        proc.wait()


def frame_at(path: str, t: float, w_long: int = AW) -> np.ndarray:
    for f in decode(path, t, 0.5, 2.0, w_long):
        return f.copy()
    raise AutocutError(f"no frame at {t:.1f}s in {Path(path).name}")


# ---------------------------------------------------------------------------------------------
# Tracking the TV
# ---------------------------------------------------------------------------------------------


def _quad_mask(shape, quad_px: np.ndarray, grow: float = 0.06) -> np.ndarray:
    import cv2

    h, w = shape[:2]
    c = quad_px.mean(axis=0)
    q = c + (quad_px - c) * (1 + grow)
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, q.astype(np.int32), 255)
    return m


def track(path: str, ref_t: float, quad_norm: list[list[float]], hz: float = HZ, others: list[list[list[float]]] | None = None) -> dict:
    """TV corners (normalised) every 1/hz s through the whole recording, following the room."""
    import cv2

    ref = frame_at(path, ref_t)
    h, w = ref.shape[:2]
    scale = np.array([w, h], np.float32)
    q_ref = np.float32(quad_norm) * scale
    sift = cv2.SIFT_create(1500)
    clahe = cv2.createCLAHE(3.0, (8, 8))

    def feats(img: np.ndarray, quads_px: list[np.ndarray]):
        g = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        m = np.full(g.shape, 255, np.uint8)
        for q in quads_px:  # no features on screens: their pictures change
            m[_quad_mask(g.shape, q) > 0] = 0
        m[g > 245] = 0
        return sift.detectAndCompute(g, m)

    def homography(k1, d1, k0, d0):
        if d1 is None or d0 is None or len(k1) < 12 or len(k0) < 12:
            return None, 0
        m = [a for a, b in (x for x in bf.knnMatch(d1, d0, k=2) if len(x) == 2) if a.distance < 0.75 * b.distance]
        if len(m) < 12:
            return None, 0
        H, inl = cv2.findHomography(np.float32([k1[x.queryIdx].pt for x in m]), np.float32([k0[x.trainIdx].pt for x in m]), cv2.RANSAC, 3.0)
        return H, int(inl.sum()) if inl is not None else 0

    bf = cv2.BFMatcher()
    others_px = [np.float32(o) * scale for o in (others or [])]
    k_ref, d_ref = feats(ref, [q_ref, *others_px])
    times, quads, quality = [], [], []
    H_cur = None  # frame -> ref
    prev = None
    t0 = time.time()
    for i, f in enumerate(decode(path, 0.0, None, hz)):
        q_prev = (cv2.perspectiveTransform(q_ref[None], np.linalg.inv(H_cur))[0] if H_cur is not None else q_ref)
        k, d = feats(f, [q_prev] + [cv2.perspectiveTransform(o[None], np.linalg.inv(H_cur))[0] if H_cur is not None else o for o in others_px])
        H_direct, n_direct = homography(k, d, k_ref, d_ref) if (i % 5 == 0 or H_cur is None) else (None, 0)
        if H_direct is not None and n_direct >= 60:
            H_cur, q = H_direct, min(1.0, n_direct / 200)
        elif prev is not None:
            H_step, n_step = homography(k, d, prev[0], prev[1])  # frame -> previous frame
            if H_step is not None and n_step >= 25 and H_cur is not None:
                H_cur, q = H_cur @ H_step, min(1.0, n_step / 200) * 0.8
            else:
                q = 0.0  # lost: keep the last position
        else:
            q = 0.0
        if H_cur is None:
            H_cur = np.eye(3)
        quad = cv2.perspectiveTransform(q_ref[None], np.linalg.inv(H_cur))[0] / scale
        times.append(i / hz)
        quads.append(quad.tolist())
        quality.append(round(q, 3))
        prev = (k, d)
        if i % 200 == 0 and i:
            log(f"tracking {Path(path).name}: {i / hz:.0f}s")
    Q = smooth_quads(np.float32(quads), np.float32(quality))
    log(f"tracked the TV through {len(times) / hz:.0f}s of {Path(path).name} in {time.time() - t0:.0f}s "
        f"({100 * np.mean(np.array(quality) > 0):.0f}% of frames locked)")
    return {"times": times, "quads": Q.tolist(), "quality": quality}


def smooth_quads(Q: np.ndarray, quality: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Remove tracking jitter (a small Gaussian over time; lost frames follow their neighbours)."""
    if len(Q) < 3:
        return Q
    k = np.exp(-0.5 * (np.arange(-3, 4) / sigma) ** 2)
    out = np.empty_like(Q)
    wts = np.clip(quality, 0.05, 1.0)
    for i in range(len(Q)):
        a, b = max(0, i - 3), min(len(Q), i + 4)
        kw = k[a - i + 3 : b - i + 3] * wts[a:b]
        out[i] = (Q[a:b] * kw[:, None, None]).sum(0) / kw.sum()
    return out


# ---------------------------------------------------------------------------------------------
# Light: what the room does with each screen colour
# ---------------------------------------------------------------------------------------------

# TV space: u = 0..1 left -> right edge of the screen, v = 0..1 top -> bottom; the glow lives outside.
TS = 160  # TV-space grid (pixels per screen width/height)
U0, U1, V0, V1 = -0.45, 1.45, -0.55, 1.35  # extent of the TV-space grid around the screen
GW, GH = int(TS * (U1 - U0)), int(TS * (V1 - V0))


_LIN = ((np.arange(256, dtype=np.float32) / 255.0) ** 2.2).astype(np.float32)
_SRGB = (np.clip(np.arange(4097, dtype=np.float32) / 4096.0, 0, 1) ** (1 / 2.2) * 255.0 + 0.5).astype(np.uint8)


def lin(x: np.ndarray) -> np.ndarray:
    """sRGB-ish 8-bit -> linear light (table lookup)."""
    x = np.asarray(x)
    return _LIN[x] if x.dtype == np.uint8 else (np.asarray(x, np.float32) / 255.0) ** 2.2


def srgb(x: np.ndarray) -> np.ndarray:
    """Linear light -> 8-bit (table lookup)."""
    return _SRGB[(np.clip(x, 0, 1) * 4096).astype(np.int32)]


def unit_to_quad(quad_px: np.ndarray):
    """Homography: TV space (u, v) in grid pixels -> image pixels."""
    import cv2

    src = np.float32([[(0 - U0) * TS, (0 - V0) * TS], [(1 - U0) * TS, (0 - V0) * TS], [(1 - U0) * TS, (1 - V0) * TS], [(0 - U0) * TS, (1 - V0) * TS]])
    return cv2.getPerspectiveTransform(src, np.float32(quad_px))


def to_tv_space(img: np.ndarray, quad_px: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.warpPerspective(img, unit_to_quad(quad_px), (GW, GH), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)


def zone_slices() -> dict[str, tuple[slice, slice]]:
    """Glow zones outside the screen, in TV-space grid pixels (rows, cols)."""
    def col(u): return int(round((u - U0) * TS))
    def row(v): return int(round((v - V0) * TS))
    return {
        "left": (slice(row(0.05), row(0.95)), slice(col(-0.30), col(-0.04))),
        "top": (slice(row(-0.45), row(-0.06)), slice(col(0.05), col(0.95))),
        "right": (slice(row(0.05), row(0.95)), slice(col(1.04), col(1.30))),
        "bottom": (slice(row(1.04), row(1.25)), slice(col(0.05), col(0.95))),
    }


def edge_slices() -> dict[str, tuple[slice, slice]]:
    """The screen's own edge bands (what drives each part of the backlight)."""
    def col(u): return int(round((u - U0) * TS))
    def row(v): return int(round((v - V0) * TS))
    return {
        "left": (slice(row(0.0), row(1.0)), slice(col(0.0), col(0.12))),
        "top": (slice(row(0.0), row(0.15)), slice(col(0.0), col(1.0))),
        "right": (slice(row(0.0), row(1.0)), slice(col(0.88), col(1.0))),
        "bottom": (slice(row(0.85), row(1.0)), slice(col(0.0), col(1.0))),
    }


def analyse(path: str, ref_t: float, quad_norm: list[list[float]], others: list | None = None) -> dict:
    """Track the TV and measure glow + screen edges every 0.1 s (cached per recording + calibration)."""
    key = ac.file_key(path, VERSION, json.dumps([ref_t, quad_norm, others or []]))
    cache = cache_dir() / f"{key}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    meta = probe_rec(path)
    tr = track(path, ref_t, quad_norm, others=others)
    Q = np.float32(tr["quads"])
    zs, es = zone_slices(), edge_slices()
    tv_imgs = []
    t0 = time.time()
    for i, f in enumerate(decode(path, 0.0, None, HZ)):
        if i >= len(Q):
            break
        h, w = f.shape[:2]
        tv_imgs.append(to_tv_space(f, Q[i] * [w, h]))
    stack = np.stack(tv_imgs)  # n, GH, GW, 3 (uint8)
    v = stack.max(axis=3).astype(np.float32)
    # self-lit: bright in (almost) every frame whatever the backlight does -> other screens, lamps
    selflit = (np.percentile(v, 15, axis=0) > 150) | (stack.sum(axis=3) == 0).all(axis=0)
    glow = {z: [] for z in ZONES}
    edge = {z: [] for z in ZONES}
    L = lin(stack)
    for z in ZONES:
        r, c = zs[z]
        m = ~selflit[r, c]
        m &= stack[:, r, c].sum(axis=3).min(axis=0) > 0  # inside the picture in every frame
        if m.sum() < 20:
            m = np.ones_like(m)
        glow[z] = L[:, r, c][:, m].mean(axis=1).tolist()
        r, c = es[z]
        edge[z] = L[:, r, c].reshape(len(L), -1, 3).mean(axis=1).tolist()
    scr = es["left"][0], slice(es["left"][1].start, es["right"][1].stop)
    lum = (L[:, scr[0], scr[1]] @ np.float32([0.2126, 0.7152, 0.0722]))
    screen_stats = {"p05": float(np.percentile(lum, 5)), "p50": float(np.percentile(lum, 50)), "p95": float(np.percentile(lum, 95))}
    out = {"version": VERSION, **meta, "ref_t": ref_t, "quad_ref": quad_norm, "times": tr["times"][: len(L)], "quads": Q[: len(L)].tolist(),
           "quality": tr["quality"][: len(L)], "glow": glow, "edge": edge, "selflit": np.packbits(selflit).tolist(), "selflit_shape": list(selflit.shape),
           "screen": screen_stats}
    out["model"] = fit_light(out)
    cache.write_text(json.dumps(out))
    log(f"light measured in {time.time() - t0:.0f}s; room response fit: " + ", ".join(f"{z} R²={out['model'][z]['r2']:.2f}" for z in ZONES)
        + f", lag {out['model']['lag'] / HZ:.1f}s")
    return out


def fit_light(a: dict) -> dict:
    """Per zone: glow ≈ A·edge(t - lag) + b (linear light), lag shared, chosen by best fit."""
    best = None
    for lag in range(0, 5):
        res, tot = {}, 0.0
        for z in ZONES:
            E = np.float32(a["edge"][z])
            G = np.float32(a["glow"][z])
            if lag:
                E, G = E[:-lag], G[lag:]
            X = np.hstack([E, np.ones((len(E), 1), np.float32)])
            W, *_ = np.linalg.lstsq(X, G, rcond=None)
            pred = X @ W
            ss = float(((G - pred) ** 2).sum())
            r2 = 1 - ss / max(1e-9, float(((G - G.mean(0)) ** 2).sum()))
            res[z] = {"W": W.tolist(), "r2": r2}
            tot += r2
        if best is None or tot > best[0]:
            best = (tot, lag, res)
    return {**best[2], "lag": best[1]}


# ---------------------------------------------------------------------------------------------
# Your clip -> the glow it should cause
# ---------------------------------------------------------------------------------------------


def clip_edges(clip: str, hz: float = HZ) -> dict:
    """Edge colours (linear) of the clip every 1/hz s, plus scene cuts and overall luminance."""
    info = ac.probe(clip)
    W, H = 160, 90
    proc = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", clip, "-an", "-vf",
                           f"fps={hz:g},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
    n = len(proc.stdout) // (W * H * 3)
    if n == 0:
        raise AutocutError("could not read the clip")
    F = lin(np.frombuffer(proc.stdout[: n * W * H * 3], np.uint8).reshape(n, H, W, 3))
    bands = {"left": (slice(0, H), slice(0, int(W * 0.12))), "top": (slice(0, int(H * 0.15)), slice(0, W)),
             "right": (slice(0, H), slice(W - int(W * 0.12), W)), "bottom": (slice(H - int(H * 0.15), H), slice(0, W))}
    edges = {z: F[:, r, c].reshape(n, -1, 3).mean(axis=1) for z, (r, c) in bands.items()}
    lum = F @ np.float32([0.2126, 0.7152, 0.0722])
    diff = np.abs(np.diff(F.mean(axis=(1, 2)), axis=0)).sum(axis=1)
    cuts = [0.0] + [float(x) for x in diff]
    return {"edges": edges, "lum_p50": float(np.median(lum)), "cuts": cuts, "n": n, "duration": info.duration, "fps": info.fps or 30.0}


def predict_glow(ce: dict, a: dict) -> dict[str, np.ndarray]:
    """Glow per zone (linear RGB) the clip would cause in this room, via the learned response."""
    k = a["screen"]["p50"] / max(1e-4, ce["lum_p50"])  # the clip as bright as the camera saw the TV
    k = float(np.clip(k, 0.3, 3.0))
    out = {}
    lag = int(a["model"]["lag"])
    for z in ZONES:
        E = np.clip(ce["edges"][z] * k, 0, 1)
        if lag:
            E = np.vstack([np.repeat(E[:1], lag, axis=0), E[:-lag]])
        W = np.float32(a["model"][z]["W"])
        out[z] = np.clip(np.hstack([E, np.ones((len(E), 1), np.float32)]) @ W, 1e-4, 1.0)
    return out


def glow_distance(target: dict[str, np.ndarray], a: dict) -> np.ndarray:
    """Cost [clip step, recording frame]: log-colour difference per zone, zones weighted by how well
    the room model explains them."""
    n_rec = len(a["glow"]["left"])
    T = len(target["left"])
    D = np.zeros((T, n_rec), np.float32)
    for z in ZONES:
        w = max(0.1, a["model"][z]["r2"])
        G = np.log(np.float32(a["glow"][z]) + 0.01)  # n_rec x 3
        P = np.log(target[z] + 0.01)  # T x 3
        D += w * np.abs(P[:, None, :] - G[None, :, :]).sum(axis=2)
    return D / sum(max(0.1, a["model"][z]["r2"]) for z in ZONES)


def min_runs(path: np.ndarray, n_rec: int, min_len: int = 10) -> np.ndarray:
    """No stretch shorter than min_len steps (1 s): a short stretch is replaced by playing on from
    the stretch before it (a jump every few frames reads as a glitch, not as light)."""
    p = path.copy()
    t = 1
    while t < len(p):
        if p[t] != p[t - 1] + 1:  # a jump starts a stretch at t
            end = t + 1
            while end < len(p) and p[end] == p[end - 1] + 1:
                end += 1
            if end - t < min_len:
                for k in range(t, end):
                    p[k] = min(n_rec - 1, p[k - 1] + 1)
            t = end
        else:
            t += 1
    return p


def choose_path(D: np.ndarray, quads: np.ndarray, cuts: np.ndarray, jump: float = 3.0, pose_w: float = 40.0, hold: float = 0.15) -> tuple[np.ndarray, float]:
    """Viterbi over recording frames: play on (free), hold a frame (small cost), or jump to another
    moment (costly, more so the more the camera view differs; cheaper on a cut in the clip)."""
    T, n = D.shape
    Qf = quads.reshape(n, -1)
    pose = np.sqrt(((Qf[:, None, :] - Qf[None, :, :]) ** 2).sum(axis=2) / 4.0).astype(np.float32)  # mean corner distance
    jump_cost = jump + pose_w * pose  # [from, to]
    V = D[0].copy()
    back = np.zeros((T, n), np.int32)
    cut_thr = np.percentile(cuts, 97) if len(cuts) > 5 else np.inf
    for t in range(1, T):
        cont = np.full(n, np.inf, np.float32)
        cont[1:] = V[:-1]  # j-1 -> j
        stay = V + hold
        jc = jump_cost * (0.4 if cuts[t] >= cut_thr else 1.0)
        jump_from = np.argmin(V[:, None] + jc, axis=0)
        jump_val = V[jump_from] + jc[jump_from, np.arange(n)]
        choices = np.stack([cont, stay, jump_val])
        which = np.argmin(choices, axis=0)
        V = choices[which, np.arange(n)] + D[t]
        back[t] = np.where(which == 0, np.arange(n) - 1, np.where(which == 1, np.arange(n), jump_from))
    j = int(np.argmin(V))
    total = float(V[j])
    path = np.zeros(T, np.int32)
    for t in range(T - 1, -1, -1):
        path[t] = j
        j = back[t, j]
    path = min_runs(path, n)
    return path, float(D[np.arange(T), path].mean())


# ---------------------------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------------------------


def relight_weights(a: dict, falloff: float = 0.45) -> np.ndarray:
    """How strongly each zone's backlight reaches every point around the TV (TV space, 4 channels):
    strongest at the screen's edge, fading with distance; zero on the screen, other screens, lamps."""
    import cv2

    u = (np.arange(GW) + 0.5) / TS + U0
    v = (np.arange(GH) + 0.5) / TS + V0
    U, Vv = np.meshgrid(u, v)
    inside_v = np.clip(1 - np.maximum(0, np.maximum(-Vv, Vv - 1)) / 0.35, 0, 1)
    inside_u = np.clip(1 - np.maximum(0, np.maximum(-U, U - 1)) / 0.35, 0, 1)
    w = np.zeros((GH, GW, 4), np.float32)
    w[..., 0] = np.exp(-np.maximum(0, -U) / falloff) * (U < 0) * inside_v
    w[..., 1] = np.exp(-np.maximum(0, -Vv) / falloff) * (Vv < 0) * inside_u
    w[..., 2] = np.exp(-np.maximum(0, U - 1) / falloff) * (U > 1) * inside_v
    w[..., 3] = np.exp(-np.maximum(0, Vv - 1) / (falloff * 0.7)) * (Vv > 1) * inside_u
    s = w.sum(axis=2, keepdims=True)
    w = np.where(s > 1, w / np.maximum(s, 1e-6), w)
    shape = tuple(a["selflit_shape"])
    selflit = np.unpackbits(np.uint8(a["selflit"]))[: shape[0] * shape[1]].reshape(shape).astype(np.float32)
    selflit = cv2.dilate(selflit, np.ones((5, 5), np.uint8))
    keep = cv2.GaussianBlur(1 - selflit, (0, 0), 2)
    return w * keep[..., None]


def screen_match(clip_lin: np.ndarray, a: dict) -> np.ndarray:
    """The clip as the camera would see the TV showing it: luminance levels of the real screen."""
    lum = clip_lin @ np.float32([0.2126, 0.7152, 0.0722])
    c05, c50, c95 = np.percentile(lum, [5, 50, 95])
    s = a["screen"]
    # piecewise-linear luminance map clip -> camera's screen, applied as a ratio (keeps colour)
    xs, ys = [0.0, c05, c50, c95, 1.0], [0.0, s["p05"], s["p50"], s["p95"], min(1.0, s["p95"] * 1.15)]
    xs = np.maximum.accumulate(np.float32(xs) + np.arange(5) * 1e-4)
    target = np.interp(lum, xs, ys)
    return np.clip(clip_lin * (target / np.maximum(lum, 1e-4))[..., None], 0, 1)


def add_bezel(img: np.ndarray, side: float, bottom_factor: float = 2.0) -> np.ndarray:
    """The picture inside a thin dark TV bezel: `side` = bezel width as a fraction of the picture
    width (left, top, right), the bottom bezel `bottom_factor` times wider, like a real TV. A faint
    lighter line on the bezel's inner edge catches the light as a real bezel does."""
    import cv2

    h, w = img.shape[:2]
    b = max(1, int(round(side * w)))
    bb = max(b, int(round(side * bottom_factor * w)))
    out = np.empty_like(img)
    out[:] = (14, 13, 12)  # near-black, slightly warm (BGR)
    # a soft vertical gradient: real bezels are a touch lighter at the top edge
    grad = np.linspace(1.25, 0.85, h, dtype=np.float32)[:, None, None]
    out = np.clip(out.astype(np.float32) * grad, 0, 255).astype(np.uint8)
    inner = cv2.resize(img, (w - 2 * b, h - b - bb), interpolation=cv2.INTER_AREA)
    out[b : h - bb, b : w - b] = inner
    cv2.rectangle(out, (b - 1, b - 1), (w - b, h - bb), (38, 38, 40), 1)  # the inner edge highlight
    return out


def render(clip: str, a: dict, path: np.ndarray, target: dict[str, np.ndarray], out: str, W: int, H: int, fps: float = 30.0,
           relight: float = 1.0, bezel: float = 0.0) -> None:
    """Background = the chosen recording moments; glow relit to the clip; the clip on the TV."""
    import cv2

    rec = a["path"]
    T = len(path)
    dur = min(T / HZ, ac.probe(clip).duration)
    n_out = int(dur * fps)
    Qrec = np.float32(a["quads"])
    weights = relight_weights(a)
    # glow gains per step (smoothed so relighting never flickers)
    G = {z: np.float32(a["glow"][z]) for z in ZONES}
    gains = np.stack([np.clip((target[z] + 0.01) / (G[z][path] + 0.01), 0.35, 2.8) for z in ZONES], axis=1)  # T, 4, 3
    k = np.exp(-0.5 * (np.arange(-3, 4) / 1.5) ** 2)
    k /= k.sum()
    gains = np.stack([np.stack([np.convolve(np.pad(gains[:, z, ch], 3, mode="edge"), k, mode="valid") for ch in range(3)], axis=1) for z in range(4)], axis=1)
    gains = 1 + (gains - 1) * relight
    # background: decode each continuous stretch of the path once, in order
    segs: list[tuple[int, int]] = []  # (start step, end step) where path increases by 1
    s0 = 0
    for t in range(1, T + 1):
        if t == T or path[t] != path[t - 1] + 1:
            segs.append((s0, t))
            s0 = t

    def background():
        for a0, b0 in segs:
            t_rec = path[a0] / HZ
            need = int(round((b0 - a0) / HZ * fps))
            got = 0
            if b0 - a0 == 1 and a0 + 1 < T and path[a0 + 1] == path[a0]:
                need = int(round(1 / HZ * fps))  # a held frame
            for f in decode(rec, t_rec, (b0 - a0) / HZ + 0.2, fps, max(W, H)):
                if got >= need:
                    break
                yield f, t_rec + got / fps, a0 + got / fps * HZ
                got += 1
            while got < need:  # end of the recording: hold the last frame
                yield f, t_rec + got / fps, a0 + got / fps * HZ
                got += 1

    clip_proc = subprocess.Popen(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", clip, "-an", "-vf",
                                  f"fps={fps:g},scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
                                  "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{W}x{H}", "-r", f"{fps:g}", "-i", "-", "-i", clip, "-map", "0:v", "-map", "1:a?", "-shortest",
                            "-c:v", "h264_videotoolbox", "-b:v", "16M", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out],
                           stdin=subprocess.PIPE)
    rng = np.random.default_rng(1)
    grain = [rng.normal(0, 1.6, (H, W, 3)).astype(np.int16) for _ in range(6)]  # camera grain, reused
    cw, ch = 1280, 720
    src_quad = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    t0 = time.time()
    try:
        for i, (bg, t_rec, step) in enumerate(background()):
            if i >= n_out:
                break
            buf = clip_proc.stdout.read(cw * ch * 3) if clip_proc.stdout else b""
            if len(buf) < cw * ch * 3:
                break
            frame = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA) if bg.shape[:2] != (H, W) else bg
            qi = min(len(Qrec) - 1, t_rec * HZ)
            lo = int(qi)
            hi = min(len(Qrec) - 1, lo + 1)
            quad = (Qrec[lo] * (1 - (qi - lo)) + Qrec[hi] * (qi - lo)) * [W, H]
            ts = min(T - 1, int(step))
            # relight the room: gain = 1 + Σ zone weight × (zone gain − 1), in linear light
            Hm = unit_to_quad(quad)
            wmap = cv2.warpPerspective(weights, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            gain = wmap.reshape(-1, 4) @ (gains[ts] - 1).astype(np.float32)
            img = lin(frame) * (1 + gain.reshape(H, W, 3))
            # the clip on the TV
            c = np.frombuffer(buf, np.uint8).reshape(ch, cw, 3)
            c = srgb(screen_match(lin(c), a))
            if bezel > 0:  # the corners sit on the TV's outer edge: draw the bezel inside them
                c = add_bezel(c, bezel)
            c = cv2.GaussianBlur(c, (0, 0), 0.6)
            M = cv2.getPerspectiveTransform(src_quad, np.float32(quad))
            warped = cv2.warpPerspective(c, M, (W, H), flags=cv2.INTER_LINEAR)
            mask = cv2.warpPerspective(np.full((ch, cw), 255, np.uint8), M, (W, H))
            mask = cv2.GaussianBlur(mask, (0, 0), 0.8).astype(np.float32)[..., None] / 255.0
            img = img * (1 - mask) + lin(warped) * mask
            outf = srgb(img)
            outf = np.clip(outf.astype(np.int16) + grain[i % len(grain)], 0, 255).astype(np.uint8)  # camera grain
            enc.stdin.write(outf.tobytes())
            if i % (int(fps) * 5) == 0 and i:
                log(f"progress {i}/{n_out} frames ({i / max(1e-3, time.time() - t0):.1f} fps)")
    finally:
        if enc.stdin:
            enc.stdin.close()
        enc.wait()
        clip_proc.kill()
        clip_proc.wait()
    log(f"rendered {out} in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------------------------
# Calibration (once per recording) and suggestions
# ---------------------------------------------------------------------------------------------


def _vision_rects(img: np.ndarray) -> list[np.ndarray]:
    """Rectangles (4 corners, normalised, top-left origin) found by Apple Vision in one frame."""
    import tempfile

    import cv2

    try:
        import objc
        import Vision
        from Foundation import NSURL
    except ImportError:
        return []
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tf:
        cv2.imwrite(tf.name, img)
        with objc.autorelease_pool():
            h = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(tf.name), None)
            r = Vision.VNDetectRectanglesRequest.alloc().init()
            r.setMaximumObservations_(12)
            r.setMinimumSize_(0.08)
            r.setMinimumAspectRatio_(0.25)
            r.setMaximumAspectRatio_(1.0)
            r.setQuadratureTolerance_(35.0)
            r.setMinimumConfidence_(0.3)
            h.performRequests_error_([r], None)
            out = []
            for o in r.results() or []:
                pts = [o.topLeft(), o.topRight(), o.bottomRight(), o.bottomLeft()]
                out.append(np.float32([[p.x, 1 - p.y] for p in pts]))
            return out


def order_quad(p: np.ndarray) -> np.ndarray:
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    p = np.float32(p)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.float32([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]])


def good_quad(q: np.ndarray, min_side: float = 0.05, min_area: float = 0.01) -> bool:
    """A usable TV outline: four distinct corners, convex, not tiny."""
    import cv2

    q = np.float32(q)
    sides = [float(np.linalg.norm(q[i] - q[(i + 1) % 4])) for i in range(4)]
    return min(sides) >= min_side and cv2.contourArea(q) >= min_area and cv2.isContourConvex(q.reshape(-1, 1, 2))


def suggest(path: str, n: int = 16) -> dict:
    """A first guess of the TV corners: the biggest cluster of rectangles Vision finds over many frames
    (the TV's picture panels and the screen itself sit inside it; the monitor is a separate cluster)."""
    import cv2

    meta = probe_rec(path)
    times = np.linspace(0.08, 0.92, n) * meta["duration"]
    rects: list[tuple[float, np.ndarray]] = []
    frames = {}
    for t in times:
        try:
            f = frame_at(path, float(t))
        except AutocutError:
            continue
        frames[float(t)] = f
        rects += [(float(t), r) for r in _vision_rects(f)]
    if not rects:
        return {"ref_t": float(times[len(times) // 2]), "quad": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.75], [0.2, 0.75]], "others": [], "confidence": 0.0}
    # group by overlap of centres (camera moves a little; good enough to separate TV from monitor)
    cents = np.float32([r.mean(0) for _, r in rects])
    areas = np.float32([cv2.contourArea(r) for _, r in rects])
    groups: list[list[int]] = []
    for i in np.argsort(-areas):
        for g in groups:
            ref = rects[g[0]][1]
            if cv2.pointPolygonTest(ref.reshape(-1, 1, 2), (float(cents[i][0]), float(cents[i][1])), False) >= 0:
                g.append(int(i))
                break
        else:
            groups.append([int(i)])
    score = [sum(areas[g]) for g in groups]
    tv = groups[int(np.argmax(score))]
    # the frame where the biggest rectangle of the TV group was seen: its hull is the suggestion
    best_t = rects[tv[0]][0]
    same_frame = [rects[i][1] for i in tv if rects[i][0] == best_t]
    hull = cv2.convexHull(np.vstack(same_frame).reshape(-1, 1, 2))
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            quad = order_quad(ap.reshape(4, 2))
            break
    if quad is not None and not good_quad(quad):
        quad = None  # e.g. two corners on top of each other (a triangle): use the TV area's box instead
    if quad is None:  # the largest single rectangle Vision saw on the TV, else the area's box
        biggest = order_quad(max(same_frame, key=lambda r: cv2.contourArea(np.float32(r))))
        if good_quad(biggest):
            quad = biggest
        else:
            x, y, w, h = cv2.boundingRect(hull)
            quad = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    others = [order_quad(rects[g[0]][1]).tolist() for k, g in enumerate(groups) if g is not tv and rects[g[0]][0] == best_t]
    conf = float(len(tv) / max(1, len(frames)))
    return {"ref_t": best_t, "quad": quad.tolist(), "others": others, "confidence": round(min(1.0, conf), 2)}


def calib_path() -> Path:
    return cache_dir() / "calibration.json"


def load_calibrations() -> dict:
    try:
        return json.loads(calib_path().read_text())
    except (OSError, ValueError):
        return {}


def calibration(path: str) -> dict | None:
    return load_calibrations().get(ac.file_key(path, "tv-calib"))


def save_calibration(path: str, ref_t: float, quad: list, others: list | None = None, confirmed: bool = True) -> dict:
    cal = load_calibrations()
    entry = {"path": path, "ref_t": float(ref_t), "quad": [[float(x), float(y)] for x, y in quad], "others": others or [], "confirmed": confirmed,
             "saved": time.strftime("%Y-%m-%d %H:%M")}
    cal[ac.file_key(path, "tv-calib")] = entry
    calib_path().write_text(json.dumps(cal, indent=1))
    return entry


# ---------------------------------------------------------------------------------------------
# Making a video
# ---------------------------------------------------------------------------------------------

SIZES = {"horizontal": (1920, 1080), "vertical": (1080, 1920)}


def scan_status(folder: str) -> list[dict]:
    """Recordings with their orientation and calibration (cached: the probe is cheap)."""
    out = []
    for r in scan(folder):
        cal = calibration(r["path"])
        out.append({**r, "calibrated": bool(cal and cal.get("confirmed")), "calibration": cal})
    return out


def cached_analysis(path: str) -> dict | None:
    """The analysis of a recording for its current calibration, if it was already made (never computes)."""
    cal = calibration(path)
    if not cal:
        return None
    key = ac.file_key(path, VERSION, json.dumps([cal["ref_t"], cal["quad"], cal.get("others") or []]))
    f = cache_dir() / f"{key}.json"
    try:
        return json.loads(f.read_text()) if f.is_file() else None
    except ValueError:
        return None


def frame_for_calibration(path: str, out_png: str, t: float | None = None) -> dict:
    """A frame to place the corners on, and the corners to start from: at time `t` the TV where
    the tracking found it (when the recording was analysed), else the saved or suggested corners."""
    import cv2

    meta = probe_rec(path)
    cal = calibration(path)
    sug = cal if cal else suggest(path)
    if not cal:
        save_calibration(path, sug["ref_t"], sug["quad"], sug.get("others"), confirmed=False)
    at = float(sug["ref_t"] if t is None else np.clip(t, 0.0, max(0.0, meta["duration"] - 0.2)))
    quad, source = sug["quad"], "saved" if (cal and cal.get("confirmed")) else "suggested"
    if t is not None and abs(at - float(sug["ref_t"])) > 0.05:
        a = cached_analysis(path)
        if a and a.get("quads"):
            i = int(np.clip(round(at * HZ), 0, len(a["quads"]) - 1))
            quad, source = a["quads"][i], "tracked"
    f = frame_at(path, at, 1280)
    cv2.imwrite(out_png, f)
    return {"image": out_png, "width": f.shape[1], "height": f.shape[0], "ref_t": at, "quad": quad, "duration": meta["duration"],
            "confirmed": bool(cal and cal.get("confirmed")), "source": source, "others": sug.get("others", [])}


def preview(path: str, out_png: str, n: int = 6) -> dict:
    """Moments across the recording with the tracked TV outline drawn: shows drift at a glance
    (analyses the recording first if needed)."""
    import cv2

    cal = calibration(path)
    if not cal:
        s = suggest(path)
        cal = save_calibration(path, s["ref_t"], s["quad"], s.get("others"), confirmed=False)
    a = analyse(path, cal["ref_t"], cal["quad"], cal.get("others"))
    Q = np.float32(a["quads"])
    tiles = []
    times = []
    for k in range(n):
        t = float(a["duration"]) * (0.05 + 0.9 * k / max(1, n - 1))
        i = int(np.clip(round(t * HZ), 0, len(Q) - 1))
        f = frame_at(path, t, 640)
        h, w = f.shape[:2]
        cv2.polylines(f, [(Q[i] * [w, h]).astype(np.int32)], True, (60, 230, 90), 2, cv2.LINE_AA)
        cv2.putText(f, f"{t:.0f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(f)
        times.append(round(t, 1))
    cols = 3
    rows = [np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range((len(tiles) + cols - 1) // cols) if len(tiles[r * cols : (r + 1) * cols]) == cols]
    sheet = np.vstack(rows) if rows else tiles[0]
    cv2.imwrite(out_png, sheet)
    locked = float(np.mean(np.float32(a["quality"]) > 0))
    return {"image": out_png, "times": times, "locked": round(locked, 3)}


def make_batch(clips: list[str], folder: str, out_dir: str, orientation: str = "horizontal", relight: float = 1.0,
               recording: str | None = None, bezel: float = 0.0) -> dict:
    """Several clips one after another (the room analysis is shared, so later clips go faster).
    A clip that fails is reported and the rest carry on."""
    outd = Path(out_dir).expanduser().resolve()
    outd.mkdir(parents=True, exist_ok=True)
    done, failed = [], []
    for k, c in enumerate(clips, 1):
        out = outd / f"{Path(c).stem} - on TV {orientation}.mp4"
        n = 2
        while out.exists():
            out = outd / f"{Path(c).stem} - on TV {orientation} {n}.mp4"
            n += 1
        log(f"clip {k}/{len(clips)}: {Path(c).name}")
        try:
            make(c, folder, str(out), orientation, relight, recording=recording, bezel=bezel)
            done.append(str(out))
        except AutocutError as e:
            log(f"{Path(c).name} failed: {e}")
            failed.append({"clip": c, "error": str(e)})
        log(f"progress {k}/{len(clips)} clips")
    if not done:
        raise AutocutError("no clip could be made" + (f": {failed[0]['error']}" if failed else ""))
    return {"outputs": done, "out_dir": str(outd), "failed": failed}


def make(clip: str, folder: str, out: str, orientation: str = "horizontal", relight: float = 1.0, fps: float = 30.0,
         recording: str | None = None, bezel: float = 0.0) -> dict:
    """Clip on the TV of the best-matching recording of that orientation, backlight matched."""
    ac.require_tools()
    if orientation not in SIZES:
        raise AutocutError("orientation must be horizontal or vertical")
    if not 0.0 <= bezel <= 0.05:
        raise AutocutError("bezel must be between 0 (none) and 0.05 (5% of the picture width)")
    if not Path(clip).is_file():
        raise AutocutError(f"clip not found: {clip}")
    recs = [r for r in scan(folder) if r["orientation"] == orientation and (not recording or Path(r["path"]).name == Path(recording).name)]
    if not recs:
        raise AutocutError(f"no {orientation} recordings in {folder}")
    t0 = time.time()
    ce = clip_edges(clip)
    best = None
    for r in recs:
        cal = calibration(r["path"])
        if not cal:
            s = suggest(r["path"])
            cal = save_calibration(r["path"], s["ref_t"], s["quad"], s.get("others"), confirmed=False)
            log(f"{Path(r['path']).name}: TV corners guessed automatically (check them once in the app for a perfect fit)")
        a = analyse(r["path"], cal["ref_t"], cal["quad"], cal.get("others"))
        tgt = predict_glow(ce, a)
        D = glow_distance(tgt, a)
        path, cost = choose_path(D, np.float32(a["quads"]), np.float32(ce["cuts"]))
        fit = float(np.mean([a["model"][z]["r2"] for z in ZONES]))
        log(f"{Path(r['path']).name}: light match {cost:.2f} (lower is better), room model fit {fit:.2f}, {int((np.diff(path) != 1).sum())} jumps")
        if best is None or cost < best[0]:
            best = (cost, r, a, path, tgt)
    cost, r, a, path, tgt = best
    W, H = SIZES[orientation]
    outp = Path(out).resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    part = outp.with_name(outp.stem + ".part.mp4")
    render(clip, a, path, tgt, str(part), W, H, fps, relight, bezel)
    part.replace(outp)
    plan = {"version": 1, "clip": str(Path(clip).resolve()), "recording": r["path"], "orientation": orientation, "match": round(cost, 3),
            "path": path.tolist(), "relight": relight}
    outp.with_suffix(".tvlight.json").write_text(json.dumps(plan))
    log(f"done in {time.time() - t0:.0f}s using {Path(r['path']).name}")
    return {"outputs": [str(outp)], "out_dir": str(outp.parent), "recording": r["path"], "match": round(cost, 3)}


def add_args(sub) -> None:
    import argparse

    p = sub.add_parser("tv-scan", help="recordings of the room: orientation and calibration")
    p.add_argument("folder")
    p = sub.add_parser("tv-frame", help="a frame and suggested TV corners for calibration")
    p.add_argument("recording")
    p.add_argument("--out", required=True, help="PNG to write")
    p.add_argument("--t", type=float, help="this moment of the recording (the tracked corners there)")
    p = sub.add_parser("tv-preview", help="moments across a recording with the tracked TV outline (checks alignment)")
    p.add_argument("recording")
    p.add_argument("--out", required=True, help="PNG to write")
    p = sub.add_parser("tv-calibrate", help="save the TV corners of a recording")
    p.add_argument("recording")
    p.add_argument("--quad", required=True, help="x1,y1,x2,y2,x3,y3,x4,y4 normalised (top-left, top-right, bottom-right, bottom-left)")
    p.add_argument("--ref-t", type=float, required=True)
    p = sub.add_parser("tv-make", help="your clip(s) on the TV, backlight matched in software")
    p.add_argument("clips", nargs="+")
    p.add_argument("--recordings", required=True)
    p.add_argument("--out", help="output file (one clip)")
    p.add_argument("--out-dir", help="output folder (several clips: '<clip> - on TV <shape>.mp4')")
    p.add_argument("--orientation", default="horizontal", choices=list(SIZES))
    p.add_argument("--relight", type=float, default=1.0, help="0 = only choose matching moments, 1 = also recolour the glow fully")
    p.add_argument("--recording", help="use this recording only")
    p.add_argument("--bezel", type=float, default=0.0,
                   help="draw a thin TV bezel around the clip (fraction of its width; 0.012 is thin; the bottom is twice as wide). "
                        "Use it when the corners are placed on the TV's outer edge")
    _ = argparse


def run_args(a) -> dict:
    if a.cmd == "tv-scan":
        recs = scan_status(a.folder)
        return {"recordings": [{k: r[k] for k in ("path", "orientation", "duration", "width", "height", "calibrated")} for r in recs]}
    if a.cmd == "tv-frame":
        return frame_for_calibration(a.recording, a.out, a.t)
    if a.cmd == "tv-preview":
        return preview(a.recording, a.out)
    if a.cmd == "tv-calibrate":
        v = [float(x) for x in a.quad.split(",")]
        if len(v) != 8 or not all(-0.5 <= x <= 1.5 for x in v):
            raise AutocutError("--quad needs 8 numbers between -0.5 and 1.5")
        e = save_calibration(a.recording, a.ref_t, [v[0:2], v[2:4], v[4:6], v[6:8]], confirmed=True)
        return {"calibration": e}
    if a.cmd == "tv-make":
        import voiceover_cut as vc

        clips, missing = [], []
        for item in a.clips:  # a missing clip is reported, the others still run
            try:
                clips += vc.collect_clips([item])
            except AutocutError:
                missing.append({"clip": item, "error": "not found"})
                log(f"skipping {item}: not found")
        if not clips:
            raise AutocutError("none of the clips were found")
        if len(clips) == 1 and a.out and not a.out_dir:
            return make(clips[0], a.recordings, a.out, a.orientation, a.relight, recording=a.recording, bezel=a.bezel)
        out_dir = a.out_dir or (str(Path(a.out).parent) if a.out else str(Path(clips[0]).parent / "EI on TV"))
        res = make_batch(clips, a.recordings, out_dir, a.orientation, a.relight, recording=a.recording, bezel=a.bezel)
        res["failed"] = missing + res.get("failed", [])
        return res
    raise AutocutError(f"unknown command {a.cmd}")
