"""music: the structure of a song, as an editor hears it (cached per file).

  beats, tempo      where the pulse is (onset strength + dynamic-programming beat tracking)
  downbeats         the "1" of each bar (the beat phase whose low end hits hardest)
  energy per beat   loudness 0–1 across the song; onset per beat: how sharp the attack is
  sections          stretches between big changes in loudness or timbre, with an energy level
                    (calm / mid / high) and the biggest rise marked as the "drop"
  hits              the strongest accents (a cut or a punch-in belongs there)

Any file with a sound track works, including videos (their audio is read; no extraction step).
"""

from __future__ import annotations

import json
import math

import numpy as np

import autocut as ac
import montage as mt
from autocut import AutocutError, log

VERSION = "music-v2"


def _stft_mag(y: np.ndarray, n_fft: int = 2048, hop: int = mt.HOP):
    if len(y) < n_fft:
        return np.zeros((0, n_fft // 2 + 1), np.float32)
    frames = 1 + (len(y) - n_fft) // hop
    win = np.hanning(n_fft).astype(np.float32)
    out = np.empty((frames, n_fft // 2 + 1), np.float32)
    idx = np.arange(n_fft)
    for i in range(0, frames, 1024):
        j = np.arange(i, min(frames, i + 1024))
        out[j] = np.abs(np.fft.rfft(y[(j[:, None] * hop) + idx[None, :]] * win, axis=1))
    return out


def _bands(mag: np.ndarray, sr: int = mt.SR, n: int = 16) -> np.ndarray:
    """Log energy in n log-spaced bands (a crude mel spectrum) per frame."""
    freqs = np.linspace(0, sr / 2, mag.shape[1])
    edges = np.geomspace(40, sr / 2, n + 1)
    out = np.zeros((mag.shape[0], n), np.float32)
    for k in range(n):
        m = (freqs >= edges[k]) & (freqs < edges[k + 1])
        if m.any():
            out[:, k] = np.log1p(mag[:, m].mean(axis=1))
    return out


def analyse(path: str) -> dict:
    key = ac.file_key(path, VERSION)
    cache = ac.CACHE_DIR / "music" / f"{key}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    y = mt.decode_mono(path)
    if len(y) < mt.SR:
        raise AutocutError("the audio is shorter than one second")
    duration = len(y) / mt.SR
    env = mt.onset_envelope(y)
    fps = mt.SR / mt.HOP
    lag = (1024 + mt.HOP) / mt.SR
    if len(env) < 16 or float(np.abs(env).max(initial=0)) < 1e-6:
        beats_f: list[int] = []
        period = fps / 2
    else:
        period = mt.estimate_period(env)
        beats_f = mt.track_beats(env, period)
    beats = [f / fps + lag for f in beats_f]
    tempo = 60 * fps / period if period > 0 else 0.0
    if len(beats) < 4:  # no pulse (speech, ambience): a 120 BPM grid keeps the editor working
        tempo = 120.0
        beats = list(np.arange(0.0, duration, 0.5))
    mag = _stft_mag(y)
    bands = _bands(mag)
    bass = mag[:, : max(2, int(150 / (mt.SR / 2) * mag.shape[1]))].sum(axis=1) if len(mag) else np.zeros(0)
    frame_t = np.arange(len(mag)) / fps + 1024 / mt.SR
    edges = beats + [duration]
    rms, onset, bassh, feat = [], [], [], []
    for a, b in zip(edges, edges[1:]):
        s0, s1 = int(a * mt.SR), max(int(a * mt.SR) + 1, int(b * mt.SR))
        rms.append(float(np.sqrt(np.mean(y[s0:s1] ** 2))) if s1 <= len(y) else 0.0)
        fm = (frame_t >= a) & (frame_t < b)
        f0 = int(max(0, (a - lag) * fps))
        onset.append(float(env[f0 : f0 + 4].max()) if f0 < len(env) else 0.0)
        bassh.append(float(bass[fm].max()) if fm.any() else 0.0)
        feat.append(bands[fm].mean(axis=0) if fm.any() else np.zeros(bands.shape[1], np.float32))
    # loudness in dB, mapped to 0–1 between the song's quiet and loud ends
    db = 20 * np.log10(np.maximum(np.asarray(rms), 1e-5))
    lo, hi = np.percentile(db, 5), np.percentile(db, 95)
    energy = np.clip((db - lo) / max(hi - lo, 1e-3), 0, 1)
    # downbeat phase: the beat in each group of four with the strongest bass attack
    phase_score = [float(np.mean(bassh[k::4])) if bassh[k::4] else 0.0 for k in range(4)]
    phase = int(np.argmax(phase_score))
    downbeats = list(range(phase, len(beats), 4))
    sections = find_sections(np.asarray(feat), energy, downbeats, beats, duration)
    on = np.asarray(onset)
    thr = max(float(np.percentile(on, 90)), float(on.mean() + 1.2 * on.std())) if len(on) else 0.0
    hits = [i for i, v in enumerate(onset) if v >= thr and energy[i] > 0.4]
    res = {
        "path": path,
        "duration": duration,
        "tempo": round(tempo, 2),
        "beats": [round(b, 4) for b in beats],
        "downbeats": downbeats,
        "energy": [round(float(e), 3) for e in energy],
        "onset": [round(v, 3) for v in onset],
        "sections": sections,
        "hits": hits,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(res))
    log(f"music: {tempo:.0f} BPM, {len(beats)} beats, {len(sections)} sections, {len(hits)} hits")
    return res


def find_sections(feat: np.ndarray, energy: np.ndarray, downbeats: list[int], beats: list[float], duration: float) -> list[dict]:
    """Section boundaries on bar lines where timbre or loudness changes most (≥ 4 bars apart)."""
    bars = [d for d in downbeats if d < len(feat)]
    if len(bars) < 4:
        return [{"start": 0.0, "end": duration, "energy": float(energy.mean()) if len(energy) else 0.5, "level": "mid", "drop": False}]
    bar_feat = np.stack([feat[b : b + 4].mean(axis=0) for b in bars])
    bar_en = np.asarray([energy[b : b + 4].mean() for b in bars])
    # Short songs (a 15 s reel) have few bars: allow sections of 2 bars there.
    gap = 4 if len(bars) >= 16 else 2
    w = 2 if len(bars) >= 8 else 1
    nov = np.zeros(len(bars))
    rise = np.zeros(len(bars))
    for i in range(1, len(bars)):
        a = bar_feat[max(0, i - w) : i].mean(axis=0)
        b = bar_feat[i : i + w].mean(axis=0)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        ea, eb = bar_en[max(0, i - w) : i].mean(), bar_en[i : i + w].mean()
        rise[i] = eb - ea
        nov[i] = (1 - cos) * 10 + 1.5 * abs(eb - ea)
    cuts = [0]
    # A big loudness rise is always a section change (the drop), whatever the timbre does.
    big = int(np.argmax(rise))
    if rise[big] > 0.25 and len(bars) - big >= 2:
        cuts.append(big)
    order = np.argsort(-nov)
    thr = float(nov.mean() + 0.5 * nov.std())
    for i in order:
        if nov[i] < thr:
            break
        if all(abs(i - c) >= gap for c in cuts) and len(bars) - i >= gap // 2:
            cuts.append(int(i))
    cuts = sorted(cuts)
    secs = []
    for k, c in enumerate(cuts):
        s = beats[bars[c]] if c > 0 else 0.0
        e = beats[bars[cuts[k + 1]]] if k + 1 < len(cuts) else duration
        b0 = bars[c]
        b1 = bars[cuts[k + 1]] if k + 1 < len(cuts) else len(energy)
        secs.append({"start": round(s, 3), "end": round(e, 3), "energy": round(float(energy[b0:b1].mean()) if b1 > b0 else 0.0, 3)})
    levels = np.asarray([x["energy"] for x in secs])
    lo, hi = np.percentile(levels, 33), np.percentile(levels, 67)
    for x in secs:
        x["level"] = "high" if x["energy"] >= max(hi, 0.55) else ("calm" if x["energy"] <= min(lo, 0.45) else "mid")
    rises = [secs[i]["energy"] - secs[i - 1]["energy"] for i in range(1, len(secs))]
    for x in secs:
        x["drop"] = False
    if rises and max(rises) > 0.12:
        secs[1 + int(np.argmax(rises))]["drop"] = True
    return secs


def choose_window(m: dict, length: float, start: str | float = "auto") -> float:
    """Start time (on a bar line) of the best `length`-second stretch: high energy overall, contains
    the drop or the loudest section, starts at a section start, ends near a section end."""
    dur = m["duration"]
    if length >= dur - 0.05:
        return 0.0
    if start != "auto":
        return max(0.0, min(float(start), dur - length))
    beats, energy = m["beats"], np.asarray(m["energy"])
    cands = sorted({0.0, *[beats[d] for d in m["downbeats"] if beats[d] + length <= dur + 0.01]})
    sec_starts = [s["start"] for s in m["sections"]]
    sec_ends = [s["end"] for s in m["sections"]]
    drops = [s["start"] for s in m["sections"] if s.get("drop")]
    best, best_s = 0.0, -math.inf
    for c in cands:
        e = c + length
        idx = [i for i, b in enumerate(beats) if c <= b < e]
        if not idx:
            continue
        mean_e = float(energy[idx].mean())
        score = mean_e
        if drops and any(c + 1.5 < d < e - 2.0 for d in drops):
            score += 0.35  # the build and its drop are both in the edit
        if any(abs(c - s) < 0.3 for s in sec_starts):
            score += 0.15
        if any(abs(e - s) < 1.0 for s in sec_ends):
            score += 0.1
        if c < 0.05 * dur and mean_e < 0.4:
            score -= 0.1  # quiet intro
        if score > best_s:
            best, best_s = c, score
    return best


def window(m: dict, start: float, length: float) -> dict:
    """The song re-timed to the edit: beats, downbeats, hits and sections relative to `start`."""
    end = start + length
    beats = m["beats"]
    keep = [i for i, b in enumerate(beats) if start - 1e-6 <= b < end]
    remap = {i: k for k, i in enumerate(keep)}
    secs = []
    for s in m["sections"]:
        a, b = max(s["start"], start), min(s["end"], end)
        if b - a > 0.2:
            secs.append({**s, "start": round(a - start, 3), "end": round(b - start, 3)})
    return {
        "start": start,
        "length": length,
        "tempo": m["tempo"],
        "beats": [beats[i] - start for i in keep],
        "energy": [m["energy"][i] for i in keep],
        "onset": [m["onset"][i] for i in keep],
        "downbeats": [remap[i] for i in m["downbeats"] if i in remap],
        "hits": [remap[i] for i in m["hits"] if i in remap],
        "sections": secs or [{"start": 0.0, "end": length, "energy": 0.5, "level": "mid", "drop": False}],
    }
