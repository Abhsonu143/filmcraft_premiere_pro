"""sfx: cinematic sound effects synthesised in code (no downloads, no licences, the same every time).

    impact     a trailer hit: a low boom falling in pitch, a noise crack, a long tail
    riser      noise and a tone climbing for `seconds`, ending exactly on the moment it leads into
    whoosh     air passing by (swept, band-passed noise), panned left to right
    sub        a sub drop: a sine sliding from 70 to 28 Hz
    braam      a dark brassy low chord (detuned saws, slowly opening filter): awe, dread, mystery
    shimmer    high bell partials with a slow sparkle: wonder, love, hope
    chime      a short bright rising arpeggio: joy, delight
    heartbeat  two soft low thumps per beat: fear, dread, anxiety
    swell      a reverse-cymbal-like build of 1–2 s that blooms into the next line

All are 48 kHz stereo float32, peak about -1 dBFS before the mix sets their level.
"""

from __future__ import annotations

import numpy as np

SR = 48000
KINDS = ("impact", "riser", "whoosh", "sub", "braam", "shimmer", "chime", "heartbeat", "swell")


def _t(seconds: float) -> np.ndarray:
    return np.arange(int(max(0.01, seconds) * SR)) / SR


def _norm(x: np.ndarray, peak: float = 0.89) -> np.ndarray:
    m = float(np.abs(x).max(initial=0.0))
    return (x * (peak / m)).astype(np.float32) if m > 0 else x.astype(np.float32)


def _onepole(x: np.ndarray, cutoff: np.ndarray | float) -> np.ndarray:
    """A one-pole low-pass with a (possibly moving) cutoff in Hz."""
    c = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), x.shape)
    a = 1.0 - np.exp(-2 * np.pi * c / SR)
    y = np.empty_like(x, dtype=np.float64)
    acc = 0.0
    for i in range(len(x)):  # short effects: fine in Python, still well under a second
        acc += a[i] * (x[i] - acc)
        y[i] = acc
    return y


def _bandnoise(n: int, lo: np.ndarray | float, hi: np.ndarray | float, rng: np.random.Generator, steep: bool = False) -> np.ndarray:
    w = rng.standard_normal(n)
    b = _onepole(w, hi) - _onepole(w, lo)
    return _onepole(b, hi) - _onepole(_onepole(b, lo), lo) if steep else b


def _stereo(l: np.ndarray, r: np.ndarray | None = None) -> np.ndarray:
    return np.stack([l, l if r is None else r], axis=1).astype(np.float32)


def _reverb(x: np.ndarray, seconds: float = 1.6, mix: float = 0.35, seed: int = 7) -> np.ndarray:
    """A cheap stereo tail: convolution with decaying noise."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = x.copy()
    k = min(len(x), int(0.05 * SR))
    x[len(x) - k :] *= np.linspace(1, 0, k)[:, None]  # no click where the dry sound ends
    env = np.exp(-np.linspace(0, 7, n))
    out = []
    for ch in range(2):
        ir = rng.standard_normal(n) * env
        ir /= np.sqrt((ir ** 2).sum())
        size = len(x) + n
        f = np.fft.rfft(x[:, ch], size) * np.fft.rfft(ir, size)
        out.append(np.fft.irfft(f, size)[:size])
    wet = np.stack(out, axis=1)
    dry = np.pad(x, ((0, n), (0, 0)))
    return (dry * (1 - mix) + wet * mix).astype(np.float32)


def impact(size: float = 1.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = _t(2.5 * size + 0.5)
    f = 30 + 90 * np.exp(-t * 9)  # 120 Hz falling to 30 Hz
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * (2.2 / size))
    crack = _bandnoise(len(t), 300, 6000, rng) * np.exp(-t * 40)
    body = _bandnoise(len(t), 60, 900, rng) * np.exp(-t * 8) * 0.5
    x = 1.0 * boom + 0.6 * crack / (np.abs(crack).max() + 1e-9) + body / (np.abs(body).max() + 1e-9) * 0.4
    return _norm(_reverb(_stereo(x), 2.2 * size, 0.3, seed))


def riser(seconds: float = 3.0, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = _t(seconds)
    p = t / max(t[-1], 1e-6)
    hi = 800 + 9000 * p ** 2
    noise = _bandnoise(len(t), hi * 0.25, hi, rng)
    noise /= np.abs(noise).max() + 1e-9
    f = 180 * 2 ** (3 * p ** 1.5)  # three octaves up
    tone = sum(np.sin(2 * np.pi * np.cumsum(f * m) / SR) / m for m in (1, 2, 3))
    x = (0.7 * noise + 0.35 * tone) * p ** 2.2
    x[-int(0.004 * SR):] *= np.linspace(1, 0, int(0.004 * SR))  # stops dead on the moment
    pan = 0.5 + 0.3 * np.sin(2 * np.pi * 0.7 * t)
    return _norm(_stereo(x * (1 - pan) * 1.4, x * pan * 1.4))


def whoosh(seconds: float = 0.7, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = _t(seconds)
    p = t / max(t[-1], 1e-6)
    env = np.sin(np.pi * p ** 0.7) ** 2
    hi = 400 + 3600 * np.sin(np.pi * p) ** 2
    x = _bandnoise(len(t), hi * 0.45, hi, rng, steep=True) * env
    return _norm(_stereo(x * (1.2 - p), x * (0.2 + p)))


def sub(seconds: float = 1.6, seed: int = 4) -> np.ndarray:
    t = _t(seconds)
    f = 28 + 42 * np.exp(-t * 2.5)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, t / 0.02) * np.exp(-t * 1.6)
    return _norm(_stereo(np.tanh(1.6 * x)))


def braam(seconds: float = 3.0, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = _t(seconds)
    x = np.zeros(len(t))
    for f0 in (55.0, 82.4, 110.0):  # A1, E2, A2
        for det in (-0.6, 0.0, 0.7):
            ph = rng.uniform(0, 1)
            x += 2 * (((f0 + det) * t + ph) % 1.0) - 1  # saw
    cutoff = 200 + 1400 * np.minimum(1, t / (0.35 * seconds)) * np.exp(-t / seconds)
    x = _onepole(x, cutoff)
    env = np.minimum(1, t / 0.05) * np.exp(-t * (1.2 / seconds))
    return _norm(_reverb(_stereo(np.tanh(1.5 * x * env)), 2.5, 0.3, seed))


def shimmer(seconds: float = 2.5, seed: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = _t(seconds)
    x = np.zeros(len(t))
    for k, f in enumerate((1318.5, 1760.0, 2217.5, 2637.0, 3520.0)):  # E6 A6 C#7 E7 A7
        start = k * 0.09
        tt = np.clip(t - start, 0, None)
        x += np.sin(2 * np.pi * f * tt + rng.uniform(0, 6.28)) * (t >= start) * np.exp(-tt * 1.8) * (0.9 ** k)
    x *= 1 + 0.3 * np.sin(2 * np.pi * 6 * t)
    return _norm(_reverb(_stereo(x), 3.0, 0.5, seed), 0.7)


def chime(seed: int = 8) -> np.ndarray:
    t = _t(1.4)
    x = np.zeros(len(t))
    for k, f in enumerate((1046.5, 1318.5, 1568.0, 2093.0)):  # C6 E6 G6 C7
        tt = np.clip(t - k * 0.07, 0, None)
        x += (t >= k * 0.07) * np.exp(-tt * 4) * (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 2 * f * tt))
    return _norm(_reverb(_stereo(x), 1.2, 0.3, seed), 0.7)


def heartbeat(beats: int = 4, bpm: float = 62.0, seed: int = 9) -> np.ndarray:
    period = 60.0 / bpm
    t = _t(beats * period + 0.4)
    x = np.zeros(len(t))
    for b in range(beats):
        for off, amp in ((0.0, 1.0), (0.28, 0.7)):
            t0 = b * period + off
            tt = np.clip(t - t0, 0, None)
            x += (t >= t0) * amp * np.sin(2 * np.pi * (45 + 30 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 14)
    return _norm(_stereo(np.tanh(2 * x)))


def swell(seconds: float = 1.8, seed: int = 10) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = _t(seconds)
    p = t / max(t[-1], 1e-6)
    x = _bandnoise(len(t), 2000, 12000, rng) * p ** 3
    x[-int(0.01 * SR):] *= np.linspace(1, 0, int(0.01 * SR))
    return _norm(_stereo(x, np.roll(x, 37)), 0.8)


def make(kind: str, seconds: float | None = None, seed: int = 0) -> np.ndarray:
    """One effect; `seconds` sets the length of risers, swells, whooshes, braams and shimmers.
    Every effect but the risers and swells (which stop dead on their moment) fades out cleanly."""
    y = _make(kind, seconds, seed)
    if kind not in ("riser", "swell"):
        k = min(len(y), int(0.25 * SR))
        y = y.copy()
        y[len(y) - k :] *= (np.linspace(1, 0, k) ** 2)[:, None]
    return y


def _make(kind: str, seconds: float | None = None, seed: int = 0) -> np.ndarray:
    if kind == "impact":
        return impact(1.0 if seconds is None else max(0.5, seconds / 2.5), 1 + seed)
    if kind == "riser":
        return riser(seconds or 3.0, 2 + seed)
    if kind == "whoosh":
        return whoosh(seconds or 0.7, 3 + seed)
    if kind == "sub":
        return sub(seconds or 1.6, 4 + seed)
    if kind == "braam":
        return braam(seconds or 3.0, 5 + seed)
    if kind == "shimmer":
        return shimmer(seconds or 2.5, 6 + seed)
    if kind == "chime":
        return chime(8 + seed)
    if kind == "heartbeat":
        return heartbeat(max(2, int(round((seconds or 4.0) / (60 / 62.0)))), seed=9 + seed)
    if kind == "swell":
        return swell(seconds or 1.8, 10 + seed)
    raise ValueError(f"unknown effect {kind!r}")
