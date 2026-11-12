#!/usr/bin/env python3
"""director: plan an edit the way a human editor would, then render it with effects.

    python director.py music     --clips CLIPS/ --music SONG_OR_VIDEO --length 30 --aspect vertical --out edit.mp4
    python director.py reference REF.mp4 --clips CLIPS/ --out like_ref.mp4 [--audio OTHER]
    python director.py render    edit.montage.json

What it decides, and why
  Cuts      dynamic programming over every beat: a cut wants a strong accent or a bar line, and
            shot length follows the music's energy (long shots in calm parts, quick cuts in
            loud ones; the bar before a drop accelerates). Section changes are always cuts.
  Shots     for every cut slot, each clip's best window at each speed is scored. The score covers
            sharp/steady/well-exposed footage, a CLIP "looks good" score, motion that matches the
            music's energy, the slot's role (opening wide shot, the drop's most striking shot,
            a calm wide or sunset to close), and the trip's chronological order.
  Sequence  a beam search over whole sequences: no clip twice, no near-identical shots back to back
            (a jump cut), camera moves that flow in the same direction, varied shot sizes, and no
            harsh brightness jumps.
  Effects   motivated, not random: real slow motion (60 fps sources) in calm parts, speed-ups and
            ramps into the drop, slow push-ins on static shots, punch-in zooms on hits, a white flash
            and punch on the drop, dissolves only at calm section changes, whip transitions only
            between shots panning the same way, one colour grade with per-shot exposure matching, and
            fades at the ends.
  Reference the reference's style, not its content: its cut points (same soundtrack), its
            transitions, flashes and zoom punches, its motion energy per shot (busy shot for busy
            shot), and optionally its colour look.

Everything lands in an editable plan (`<out>.montage.json`) and an `.otio` timeline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import autocut as ac
import footage as fo
import music as mu
import render as rd
import voiceover_cut as vc
from autocut import AutocutError, log

ASPECTS = {"horizontal": (1920, 1080), "vertical": (1080, 1920), "square": (1080, 1080)}
PACE = {"fast": 0.6, "medium": 1.0, "slow": 1.6}
SPEEDS = [0.5, 0.75, 1.0, 1.5, 2.0]


# ---------------------------------------------------------------------------------------------
# Footage
# ---------------------------------------------------------------------------------------------


def _prefix(v) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.asarray(v, dtype=np.float64))])


def _mean(pref: np.ndarray, a: float, b: float, rate: float) -> float:
    i = int(max(0, math.floor(a * rate)))
    j = int(min(len(pref) - 1, max(i + 1, math.ceil(b * rate))))
    if j <= i:
        return float(pref[-1] / max(1, len(pref) - 1))
    return float((pref[j] - pref[i]) / (j - i))


def load_footage(paths: list[str], use_clip: bool = True) -> list[dict]:
    t0 = time.time()
    log(f"analysing {len(paths)} clips (sharpness, exposure, camera motion)")
    clips = []
    for i, p in enumerate(paths, 1):
        a = fo.analyse(p)
        if a["duration"] <= 0.3 or not a["motion"]:
            continue
        clips.append({"path": p, "a": a})
        if i % 25 == 0:
            log(f"progress {i}/{len(paths)} clips analysed")
    if not clips:
        raise AutocutError("no usable video clips")
    log(f"footage signals {time.time() - t0:.1f}s")
    sem = None
    if use_clip:
        try:
            sem = fo.semantics([c["path"] for c in clips])
        except ImportError:
            log("open_clip not installed: no shot understanding (quality and motion only)")
    looks_all = []
    for c in clips:
        a = c["a"]
        c["dur"] = float(a["duration"])
        c["fps_src"] = float(a.get("fps") or 30)
        q = fo.moment_quality(a)
        c["q"] = _prefix(q)
        c["qmin"] = q
        mot = np.asarray(a["motion"], dtype=np.float32)
        cam = np.hypot(np.asarray(a["dx"]), np.asarray(a["dy"])) / max(1.0, a["width"] * a.get("scale", 0.1))
        c["mot"] = _prefix(mot)
        c["cam"] = _prefix(cam * fo.FPS)  # frame widths per second
        c["dx"] = _prefix(a["dx"])
        c["dy"] = _prefix(a["dy"])
        c["bright"] = _prefix(a["bright"])
        c["sat"] = _prefix(a["sat"])
        if sem is not None:
            emb = sem[0].get(c["path"], np.zeros((0, 512), np.float32))
            d = fo.describe(emb, sem[1])
            c["emb"] = emb
            c["size"] = d["size"]
            c["tags"] = d["tags"]
            c["look_raw"] = np.asarray(d["look"], dtype=np.float32)
            looks_all += d["look"]
        else:
            c["emb"] = np.zeros((0, 512), np.float32)
            c["size"], c["tags"], c["look_raw"] = [], [], np.zeros(0, np.float32)
    # look scores relative to this batch of footage (0–1)
    if looks_all:
        mu_, sd = float(np.mean(looks_all)), float(np.std(looks_all) + 1e-6)
        for c in clips:
            c["look"] = _prefix(1 / (1 + np.exp(-(c["look_raw"] - mu_) / sd)))
    else:
        for c in clips:
            c["look"] = _prefix([0.5])
    created = sorted(range(len(clips)), key=lambda i: (clips[i]["a"].get("created", 0), clips[i]["path"]))
    for r, i in enumerate(created):
        clips[i]["chrono"] = r / max(1, len(clips) - 1)
    # global distribution of motion (log) for energy matching
    lm = np.log1p(np.concatenate([np.asarray(c["a"]["motion"]) for c in clips]))
    for c in clips:
        c["_lm"] = (float(lm.mean()), float(lm.std() + 1e-6))
    return clips


def at_second(c: dict, key: str, t: float):
    arr = c.get(key)
    if arr is None or len(arr) == 0:
        return None
    return arr[min(len(arr) - 1, max(0, int(t / fo.CLIP_EVERY)))]


def emb_at(c: dict, t: float) -> np.ndarray | None:
    e = c.get("emb")
    if e is None or len(e) == 0:
        return None
    return e[min(len(e) - 1, max(0, int(t / fo.CLIP_EVERY)))]


# ---------------------------------------------------------------------------------------------
# Cuts
# ---------------------------------------------------------------------------------------------


def target_len(energy: float, pace: float) -> float:
    """Ideal shot length (s) for an energy level: ~3.6 s when calm down to ~0.8 s when loud."""
    return pace * 3.6 * (0.8 / 3.6) ** float(np.clip(energy, 0, 1))


def cut_points(w: dict, fps: float, pace: float, min_len: float = 0.45) -> list[dict]:
    """Choose cut beats by dynamic programming; returns slots with frame ranges and context."""
    L = w["length"]
    beats = [b for b in w["beats"] if 0.05 < b < L - min_len]
    energy = np.asarray(w["energy"] or [0.5])
    onset = np.asarray(w["onset"] or [0.0])
    on_n = (onset - onset.min()) / max(1e-6, float(np.ptp(onset))) if len(onset) else onset
    down = set(w["downbeats"])
    beat_idx = {round(b, 4): i for i, b in enumerate(w["beats"])}
    sec_starts = [s["start"] for s in w["sections"][1:]]
    drops = [s["start"] for s in w["sections"] if s.get("drop")]

    def e_at(t: float) -> float:
        idx = [i for i, b in enumerate(w["beats"]) if b <= t]
        i = idx[-1] if idx else 0
        lo, hi = max(0, i - 2), min(len(energy), i + 3)
        return float(energy[lo:hi].mean()) if hi > lo else 0.5

    pts = [0.0] + beats + [L]
    forced = {j for j, t in enumerate(pts) if any(abs(t - s) < 0.08 for s in sec_starts)}
    n = len(pts)
    best = [-math.inf] * n
    back = [-1] * n
    best[0] = 0.0
    for j in range(1, n):
        tj = pts[j]
        bi = beat_idx.get(round(tj, 4))
        reward = 0.0
        if bi is not None:
            reward = 0.9 * float(on_n[bi]) + (0.45 if bi in down else 0.0)
        for i in range(j - 1, -1, -1):
            d = tj - pts[i]
            if d < min_len and j != n - 1:
                continue
            if d > 7.0:
                break
            if any(i < k < j for k in forced):
                break  # never skip over a section change
            mid_e = e_at((tj + pts[i]) / 2)
            tl = target_len(mid_e, pace)
            # the bar before a drop accelerates
            if any(0 < dr - tj < 2.2 for dr in drops):
                tl *= 0.55
            if j == n - 1:
                tl = max(tl, 1.8)  # the closing shot holds
            dev = math.log(max(d, 1e-3) / tl)
            score = best[i] + reward - 2.2 * dev * dev
            if score > best[j]:
                best[j], back[j] = score, i
    path = []
    j = n - 1
    while j > 0 and back[j] >= 0:
        path.append(j)
        j = back[j]
    path.append(0)
    times = [pts[k] for k in reversed(path)]
    frames = sorted({round(t * fps) for t in times} | {0, round(L * fps)})
    frames = [f for f in frames if 0 <= f <= round(L * fps)]
    slots = []
    for k, (a, b) in enumerate(zip(frames, frames[1:])):
        if b - a < 2:
            continue
        ta, tb = a / fps, b / fps
        sec = next((s for s in w["sections"] if s["start"] - 0.05 <= ta < s["end"]), w["sections"][-1])
        ids = [i for i, bt in enumerate(w["beats"]) if ta <= bt < tb]
        hits = [w["beats"][h] - ta for h in w["hits"] if ta + 0.2 < w["beats"][h] < tb - 0.15]
        slots.append({
            "start_f": a, "frames": b - a,
            "energy": float(np.mean([energy[i] for i in ids])) if ids else e_at(ta),
            "level": sec.get("level", "mid"),
            "section": w["sections"].index(sec),
            "section_start": any(abs(ta - s) < 0.08 for s in sec_starts),
            "drop": any(abs(ta - d) < 0.08 for d in drops),
            "build": any(0 < d - ta < 2.2 for d in drops),
            "hits": hits,
        })
    if slots:
        slots[0]["first"] = True
        slots[-1]["last"] = True
    return slots


def split_long(slots: list[dict], w: dict, fps: float, max_len: float) -> list[dict]:
    """Slots longer than the footage can fill (even slowed down) are split on beats."""
    out = []
    for s in slots:
        d = s["frames"] / fps
        if d <= max_len:
            out.append(s)
            continue
        n = math.ceil(d / max_len)
        ta = s["start_f"] / fps
        cand = [b - ta for b in w["beats"] if ta < b < ta + d] if w.get("beats") else []
        cuts = []
        for k in range(1, n):
            ideal = d * k / n
            c = min(cand, key=lambda b: abs(b - ideal)) if cand else ideal
            cuts.append(c)
        edges = sorted({0, *[round((ta + c) * fps) - s["start_f"] for c in cuts], s["frames"]})
        for k, (a, b) in enumerate(zip(edges, edges[1:])):
            if b - a < 2:
                continue
            piece = {**s, "start_f": s["start_f"] + a, "frames": b - a, "hits": [h - a / fps for h in s["hits"] if a / fps < h < b / fps]}
            if k > 0:
                piece.update({"section_start": False, "drop": False, "first": False, "ref_transition": None, "ref_punches": [], "ref_flash": []})
            if k < len(edges) - 2:
                piece["last"] = False
            out.append(piece)
    return out


# ---------------------------------------------------------------------------------------------
# Choosing shots
# ---------------------------------------------------------------------------------------------


def window_features(c: dict, start: float, src_len: float, speed: float) -> dict:
    end = start + src_len
    q = _mean(c["q"], start, end, fo.FPS)
    qmin = float(c["qmin"][int(start * fo.FPS) : max(int(start * fo.FPS) + 1, int(end * fo.FPS))].min()) if len(c["qmin"]) else 0.5
    mot = _mean(c["mot"], start, end, fo.FPS)
    cam = _mean(c["cam"], start, end, fo.FPS)
    look = _mean(c["look"], start, end, 1.0 / fo.CLIP_EVERY) if len(c["look"]) > 1 else 0.5
    mu_, sd = c["_lm"]
    dyn = 1 / (1 + math.exp(-((math.log1p(mot * speed) - mu_) / sd)))
    return {"q": q, "qmin": qmin, "mot": mot, "cam": cam * speed, "look": look, "dyn": dyn,
            "dx": _mean(c["dx"], start, end, fo.FPS), "dy": _mean(c["dy"], start, end, fo.FPS),
            "bright": _mean(c["bright"], start, end, fo.FPS), "sat": _mean(c["sat"], start, end, fo.FPS)}


def unary(slot: dict, c: dict, f: dict, speed: float, progress: float, opts: dict, start: float) -> float:
    e = slot["energy"]
    s = 1.0 * f["q"] + 1.3 * f["look"] + 1.1 * (1 - abs(f["dyn"] - e))
    if f["qmin"] < 0.3:
        s -= 0.6
    if f["bright"] < 30 and not slot.get("last"):
        s -= 0.4
    # speed: slow motion is cinematic when calm, speed-ups belong to energetic parts
    if speed < 1:
        s += 0.25 if e < 0.45 else -0.25
    elif speed > 1:
        s += 0.15 if e > 0.6 or slot.get("build") else -0.35
    s -= 0.05 * abs(math.log2(speed))
    size = at_second(c, "size", start) or ""
    tags = at_second(c, "tags", start) or []
    if speed > 1 and any(t in ("people", "face", "animals") for t in tags):
        s -= 0.9  # sped-up people look comical
    if slot.get("first"):
        s += 0.35 if size in ("aerial", "wide") else 0.0
        s += 0.6 * f["look"]
    if slot.get("last"):
        s += 0.3 if any(t in ("sunset", "night", "clouds") for t in tags) else 0.0
        s += 0.2 if size in ("aerial", "wide") else 0.0
        s += 0.2 * (1 - f["dyn"])
        s -= 0.6 if speed > 1 else 0.0  # an ending breathes
    if slot.get("drop"):
        s += 0.6 * f["look"] + 0.4 * f["dyn"]
    if "ref_peak" in slot and slot["ref_peak"] is not None:
        s += 0.3 * float(slot.get("_peak_match", 0.0))
    if opts.get("order") == "time":
        s -= 0.9 * abs(c["chrono"] - progress)
    if slot.get("sims") is not None:
        s += 3.0 * float(slot["sims"])
    return s


def candidates(slot: dict, clips: list[dict], fps: float, progress: float, opts: dict, k: int = 28, ref_sims=None) -> list[dict]:
    d = slot["frames"] / fps
    out = []
    for ci, c in enumerate(clips):
        best = None
        for sp in SPEEDS:
            if sp < 1 and c["fps_src"] < fps / sp * 0.95:
                continue  # no real frames for that slow motion
            src = d * sp
            lo, hi = 0.2, c["dur"] - src - 0.1
            if hi < lo:
                continue
            starts = np.arange(lo, hi + 1e-6, max(0.3, (hi - lo) / 8)) if hi > lo else np.asarray([lo])
            for st in starts:
                f = window_features(c, float(st), src, sp)
                sl = slot
                if ref_sims is not None:
                    sl = {**slot, "sims": ref_sims(ci, float(st), src)}
                if slot.get("ref_peak") is not None:
                    pk = peak_offset(c, float(st), src) / sp
                    sl = {**sl, "_peak_match": max(0.0, 1 - abs(pk - slot["ref_peak"]) / 0.6)}
                sc = unary(sl, c, f, sp, progress, opts, float(st))
                if best is None or sc > best["score"]:
                    best = {"clip": ci, "in": float(st), "speed": sp, "src": src, "score": sc, "f": f}
        if best:
            out.append(best)
    out.sort(key=lambda x: -x["score"])
    return out[:k]


def peak_offset(c: dict, start: float, src: float) -> float:
    m = c["a"]["motion"]
    a, b = int(start * fo.FPS), max(int(start * fo.FPS) + 1, int((start + src) * fo.FPS))
    seg = m[a:b]
    return (int(np.argmax(seg)) / fo.FPS) if seg else 0.0


def pair_cost(prev: dict, cur: dict, clips: list[dict], slot: dict, used_embs: list) -> float:
    cp, cc = clips[prev["clip"]], clips[cur["clip"]]
    s = 0.0
    e1, e2 = emb_at(cp, prev["in"] + prev["src"] - 0.05), emb_at(cc, cur["in"])
    if e1 is not None and e2 is not None:
        sim = float(e1 @ e2)
        if sim > 0.93:
            s -= 1.2
        elif sim > 0.87:
            s -= 0.4
    em = emb_at(cc, cur["in"] + cur["src"] / 2)
    if em is not None and used_embs:
        if max(float(em @ u) for u in used_embs) > 0.95:
            s -= 0.6
    a, b = prev["f"], cur["f"]
    if a["cam"] > 0.05 and b["cam"] > 0.05:
        dot = a["dx"] * b["dx"] + a["dy"] * b["dy"]
        s += 0.15 if dot > 0 else -0.3
    if abs(a["bright"] - b["bright"]) > 60 and not slot.get("section_start"):
        s -= 0.25
    t1 = set(at_second(cp, "tags", prev["in"]) or [])
    t2 = set(at_second(cc, "tags", cur["in"]) or [])
    if t1 and t1 == t2:
        s -= 0.2
    return s


def recent_subject_penalty(seq: list[dict], cur: dict, clips: list[dict], window: int = 6) -> float:
    """The same main subject (waterfall, temple …) again within a few shots feels repetitive."""
    main = (at_second(clips[cur["clip"]], "tags", cur["in"]) or [None])[0]
    if not main:
        return 0.0
    recent = [(at_second(clips[x["clip"]], "tags", x["in"]) or [None])[0] for x in seq[-window:-1]]
    return -0.45 if main in recent else 0.0


def sequence(slots: list[dict], clips: list[dict], fps: float, opts: dict, beam: int = 14, ref_sims=None) -> list[dict]:
    n = len(slots)
    cands = []
    t0 = time.time()
    for i, sl in enumerate(slots):
        cs = candidates(sl, clips, fps, i / max(1, n - 1), opts, ref_sims=(lambda ci, st, src, i=i: ref_sims(i, ci, st, src)) if ref_sims else None)
        if not cs:
            raise AutocutError(f"no clip can fill shot {i + 1} ({sl['frames'] / fps:.1f}s)")
        cands.append(cs)
    log(f"scored footage for {n} shots in {time.time() - t0:.1f}s")
    states = [(0.0, [], frozenset(), [])]
    for i in range(n):
        nxt = []
        for score, seq, used, embs in states:
            for c in cands[i]:
                pen = 0.0
                if c["clip"] in used:
                    if len(used) < len(clips):
                        continue
                    pen -= 1.5
                if seq:
                    pen += pair_cost(seq[-1], c, clips, slots[i], embs)
                    pen += recent_subject_penalty(seq + [c], c, clips)
                    if len(seq) >= 2:
                        z = [at_second(clips[x["clip"]], "size", x["in"]) for x in (seq[-2], seq[-1])] + [at_second(clips[c["clip"]], "size", c["in"])]
                        if z[0] and z[0] == z[1] == z[2]:
                            pen -= 0.15
                e = emb_at(clips[c["clip"]], c["in"] + c["src"] / 2)
                nxt.append((score + c["score"] + pen, seq + [c], used | {c["clip"]}, embs + ([e] if e is not None else [])))
        if not nxt:  # every clip used: allow repeats
            for score, seq, used, embs in states:
                c = cands[i][0]
                nxt.append((score + c["score"] - 1.5, seq + [c], used, embs))
        nxt.sort(key=lambda x: -x[0])
        states = nxt[:beam]
    return states[0][1]


# ---------------------------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------------------------


def add_effects(shots: list[dict], slots: list[dict], clips: list[dict], fps: float, opts: dict) -> None:
    level = opts.get("effects", "full")
    vertical = opts.get("height", 1080) > opts.get("width", 1920)
    last_whip = -10
    for i, (sh, sl) in enumerate(zip(shots, slots)):
        d = sl["frames"] / fps
        f = sh.pop("f")
        sh["grade"] = {"exposure": round(float(np.clip((118 - f["bright"]) / 255 * 0.6, -0.08, 0.08)), 3),
                       "sat": 1.12 if f["sat"] < 0.18 else 1.0}
        if level == "none":
            sh["transition_in"] = {"type": "cut"}
            continue
        # framing: static shots get a slow push-in (pull-out to close the film)
        if f["cam"] < 0.03 and d >= 1.1:
            if sl.get("last"):
                sh["zoom"] = {"from": 1.1, "to": 1.0}
            else:
                sh["zoom"] = {"from": 1.0, "to": 1.06 if sl["energy"] > 0.6 else 1.09}
        if level == "full":
            sh["punches"] = [round(h, 3) for h in sl["hits"] if sl["energy"] > 0.55] [:2]
            if sl.get("ref_punches"):
                sh["punches"] = sl["ref_punches"]
            if sl.get("ref_flash"):
                sh["flash"] = sl["ref_flash"]
        # speed ramp into the drop: the last shot before it accelerates
        if level == "full" and sl.get("build") and i + 1 < len(slots) and slots[i + 1].get("drop") and sh["speed"] <= 1.0 and d >= 0.8:
            c = clips[sh["clip"]]
            ramp_to = 2.0
            need = d * (sh["speed"] + ramp_to) / 2
            if sh["in"] + need < c["dur"] - 0.05:
                sh["ramp_to"] = ramp_to
        # transition into this shot
        tr = {"type": "cut"}
        if i == 0:
            sh["fade_in"] = 4 if vertical else 12
        elif sl.get("ref_transition"):
            tr = dict(sl["ref_transition"])
        elif sl.get("drop"):
            tr = {"type": "fadewhite", "frames": 6}
            sh["punches"] = [0.0] + [p for p in sh.get("punches", []) if p > 0.3]
        elif sl.get("section_start") and (sl["level"] == "calm" or slots[i - 1]["level"] == "calm") and d > 1.0 and slots[i - 1]["frames"] / fps > 1.0:
            tr = {"type": "dissolve", "frames": int(round(min(0.5, d / 3) * fps))}
        elif level == "full" and sl["energy"] > 0.5 and i - last_whip >= 4 and d > 0.6:
            pf, cf = shots[i - 1].get("_f"), f
            if pf and pf["cam"] > 0.05 and cf["cam"] > 0.05 and (pf["dx"] * cf["dx"] + pf["dy"] * cf["dy"]) > 0:
                horiz = abs(pf["dx"]) >= abs(pf["dy"])
                kind = ("whipleft" if pf["dx"] < 0 else "whipright") if horiz else ("whipup" if pf["dy"] < 0 else "whipdown")
                tr = {"type": kind, "frames": 6}
                last_whip = i
        sh["transition_in"] = tr
        sh["_f"] = f
        if sl.get("last"):
            sh["fade_out"] = int(round(min(0.8, d / 2) * fps))
    for sh in shots:
        sh.pop("_f", None)


# ---------------------------------------------------------------------------------------------
# Reference style
# ---------------------------------------------------------------------------------------------


def reference_style(path: str, fps_out: float) -> dict:
    """Cuts, transitions (fades, flashes), zoom punches, per-shot motion and the colour look."""
    import subprocess

    info = ac.probe(path)
    dw, dh = info.display_size
    w = 90 if dh > dw else 160
    h = max(2, int(round(w * dh / max(dw, 1) / 2)) * 2)
    rate = 15.0
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-an", "-vf", f"fps={rate:g},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    n = len(proc.stdout) // (w * h * 3)
    if n < 3:
        raise AutocutError("could not read the reference video")
    fr = np.frombuffer(proc.stdout[: n * w * h * 3], dtype=np.uint8).reshape(n, h, w, 3).astype(np.float32)
    gray = fr @ np.asarray([0.299, 0.587, 0.114], np.float32)
    luma = gray.mean(axis=(1, 2))
    diff = np.concatenate([[0], np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))])
    flashes = []
    for t in range(2, n - 1):
        if luma[t] - min(luma[t - 2], luma[t - 1]) > 30 and luma[t] > 150:
            if not flashes or t / rate - flashes[-1] > 0.4:
                flashes.append(round(t / rate, 3))
    cuts = []
    for t in vc_cuts(path):
        if any(abs(t - f) < 0.12 for f in flashes):
            a, b = int((t - 0.2) * rate), int((t + 0.3) * rate)
            if 0 <= a and b < n and float(np.abs(gray[a] - gray[b]).mean()) < 40:
                continue  # a flash on the same scene
        # A flash or a zoom punch also changes every pixel, but the picture after it is the same
        # scene as before it: only a real change of picture is a cut.
        a, b = int((t - 0.2) * rate), int((t + 0.3) * rate)
        if 0 <= a and b < n and float(np.abs(gray[a] - gray[b]).mean()) < 14:
            continue
        cuts.append(round(t, 3))
    # zoom rate between frames: best scale of frame t that matches frame t+1 (centre crop)
    scales = np.linspace(0.94, 1.12, 19)
    zoom = np.ones(n)
    for t in range(n - 1):
        a, b = gray[t], gray[t + 1]
        best, bs = 1.0, math.inf
        for s in scales:
            ch, cw = int(h / s), int(w / s)
            if ch < 4 or cw < 4 or ch > h or cw > w:
                continue
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            crop = a[y0 : y0 + ch, x0 : x0 + cw]
            yi = (np.arange(h) * ch / h).astype(int)
            xi = (np.arange(w) * cw / w).astype(int)
            err = float(np.abs(crop[yi][:, xi] - b).mean())
            if err < bs:
                best, bs = s, err
        zoom[t + 1] = best
    near_cut = lambda t: any(abs(t - c) < 0.15 for c in cuts)  # noqa: E731
    punches = [round(t / rate, 3) for t in range(1, n) if zoom[t] >= 1.05 and zoom[t - 1] < 1.03 and not near_cut(t / rate)]
    edges = [0.0] + cuts + [info.duration]
    shots = []
    for k, (a, b) in enumerate(zip(edges, edges[1:])):
        ia, ib = int(a * rate), max(int(a * rate) + 1, int(b * rate))
        seg = diff[ia:ib]
        tr = "cut"
        if k > 0:
            lo = luma[max(0, ia - 4) : ia + 4]
            if len(lo) and lo.min() < 18:
                tr = "fadeblack"
            elif len(lo) and lo.max() > 230 and lo.max() - luma[max(0, ia - 6)] > 40:
                tr = "fadewhite"
        shots.append({
            "start": a, "end": b,
            "motion": float(seg[1:].mean()) if len(seg) > 1 else 0.0,
            "peak": float(np.argmax(seg[1:]) / rate) if len(seg) > 2 else None,
            "transition": tr,
            "punches": [round(p - a, 3) for p in punches if a + 0.1 < p < b - 0.1],
            "flash": [round(x - a, 3) for x in flashes if a + 0.1 < x < b - 0.1],
        })
    r, g, bch = fr[..., 0], fr[..., 1], fr[..., 2]
    mx, mn = fr.max(axis=3), fr.min(axis=3)
    sat = float(np.where(mx > 1, (mx - mn) / np.maximum(mx, 1), 0).mean())
    look = {"sat": sat, "contrast": float(gray.std(axis=(1, 2)).mean()), "warm": float((r - bch).mean() / 255), "bright": float(luma.mean())}
    log(f"reference: {len(shots)} shots, {len(punches)} zoom punches, {len(flashes)} flashes, transitions {[s['transition'] for s in shots[1:]]}")
    return {"duration": info.duration, "shots": shots, "look": look, "punches": punches, "flashes": flashes,
            "size": (dw, dh), "fps": info.fps}


def vc_cuts(path: str) -> list[float]:
    import montage as mt

    return mt.detect_cuts(path, 0.3)


def match_look(ref_look: dict, clips: list[dict]) -> dict:
    sats = [float(np.mean(c["a"]["sat"])) for c in clips if c["a"]["sat"]]
    ours = float(np.mean(sats)) if sats else 0.3
    warm = float(np.mean([np.mean(c["a"]["warm"]) for c in clips if c["a"]["warm"]] or [0.0]))
    s = float(np.clip(ref_look["sat"] / max(ours, 1e-3), 0.7, 1.45))
    dw = float(np.clip((ref_look["warm"] - warm) * 0.6, -0.08, 0.08))
    extra = f"colorbalance=rm={dw:.3f}:bm={-dw:.3f}" if abs(dw) > 0.01 else ""
    return {"contrast": 1.06, "saturation": round(s, 3), "extra": extra}


# ---------------------------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------------------------


def finish(shots: list[dict], slots: list[dict], clips: list[dict], fps: float) -> list[dict]:
    out = []
    for sh, sl in zip(shots, slots):
        c = clips[sh["clip"]]
        rec = {"clip": c["path"], "in": round(sh["in"], 4), "frames": sl["frames"], "speed": sh["speed"],
               "energy": round(sl["energy"], 3), "score": round(sh["score"], 3),
               "why": " ".join(x for x in [
                   (at_second(c, "size", sh["in"]) or ""), ",".join(at_second(c, "tags", sh["in"]) or []),
                   "slow-mo" if sh["speed"] < 1 else ("sped up" if sh["speed"] > 1 else ""),
                   "DROP" if sl.get("drop") else "", "opener" if sl.get("first") else "", "closer" if sl.get("last") else ""] if x)}
        for k in ("zoom", "punches", "flash", "transition_in", "fade_in", "fade_out", "grade", "ramp_to"):
            if sh.get(k):
                rec[k] = sh[k]
        out.append(rec)
    return out


def plan_music(args) -> dict:
    ac.require_tools()
    audio = str(Path(args.music).resolve())
    if not ac.probe(audio).has_audio:
        raise AutocutError(f"no sound in {audio}")
    m = mu.analyse(audio)
    length = min(float(args.length), m["duration"])
    if length < float(args.length) - 0.05:
        log(f"the audio is only {m['duration']:.1f}s: the edit will be {length:.1f}s")
    start = mu.choose_window(m, length, args.music_start)
    w = mu.window(m, start, length)
    fps = float(args.fps)
    width, height = ASPECTS[args.aspect]
    clips = load_footage(vc.collect_clips(args.clips), use_clip=not args.no_clip)
    slots = cut_points(w, fps, PACE.get(args.pace, 1.0))
    med = float(np.median([c["dur"] for c in clips]))
    slots = split_long(slots, w, fps, max(1.2, min(6.0, 1.8 * med)))
    log(f"music {m['tempo']:.0f} BPM, using {start:.1f}–{start + length:.1f}s; {len(slots)} shots, sections {[s['level'] + ('*' if s.get('drop') else '') for s in w['sections']]}")
    opts = {"order": args.order, "effects": args.effects, "width": width, "height": height}
    seq = sequence(slots, clips, fps, opts)
    add_effects(seq, slots, clips, fps, opts)
    shots = finish(seq, slots, clips, fps)
    return {
        "version": 2, "mode": "montage", "kind": "music", "engine": "director",
        "width": width, "height": height, "fps": fps, "look": args.look,
        "audio": {"path": audio, "start": round(start, 3), "fade_in": 0.05 if start < 0.1 else 0.25, "fade_out": 1.0, "normalize": True},
        "tempo": m["tempo"], "sections": w["sections"],
        "clips": [{"path": c["path"], "duration": c["dur"]} for c in clips],
        "shots": shots,
    }


def plan_reference(args) -> dict:
    ac.require_tools()
    ref = str(Path(args.reference).resolve())
    rinfo = ac.probe(ref)
    if not rinfo.has_video:
        raise AutocutError("the reference has no video")
    fps = float(args.fps) if args.fps else (30.0 if rinfo.fps > 31 or not rinfo.fps else round(rinfo.fps, 3))
    if args.aspect == "same":
        dw, dh = rinfo.display_size
        scale = min(1.0, 1920 / max(dw, dh, 1))
        width, height = max(2, int(dw * scale) // 2 * 2), max(2, int(dh * scale) // 2 * 2)
    else:
        width, height = ASPECTS[args.aspect]
    style = reference_style(ref, fps)
    audio = str(Path(args.audio).resolve()) if args.audio else ref
    clips = load_footage(vc.collect_clips(args.clips), use_clip=not args.no_clip)
    if args.audio:  # another soundtrack: the reference's pace and effects on the new music
        m = mu.analyse(audio)
        length = min(m["duration"], float(args.length or rinfo.duration))
        start = mu.choose_window(m, length, "auto")
        w = mu.window(m, start, length)
        lens = [s["end"] - s["start"] for s in style["shots"]]
        pace = float(np.clip(np.median(lens) / 2.0, 0.5, 2.5))
        slots = cut_points(w, fps, pace)
        for sl in slots:
            sl["ref_transition"] = None
        audio_spec = {"path": audio, "start": round(start, 3), "fade_in": 0.05, "fade_out": 1.0, "normalize": True}
    else:
        m = mu.analyse(ref) if rinfo.has_audio else None
        length = rinfo.duration
        w = mu.window(m, 0.0, length) if m else {"beats": [], "energy": [], "onset": [], "downbeats": [], "hits": [], "sections": [{"start": 0, "end": length, "level": "mid", "energy": 0.5, "drop": False}], "length": length}
        motions = [s["motion"] for s in style["shots"]]
        ranks = mt_rank(motions)
        slots = []
        total = round(length * fps)
        for k, (s, rk) in enumerate(zip(style["shots"], ranks)):
            a, b = round(s["start"] * fps), min(total, round(s["end"] * fps))
            if b - a < 2:
                continue
            tr = s["transition"]
            slots.append({
                "start_f": a, "frames": b - a, "energy": 0.5 * rk + 0.5 * _energy_between(w, s["start"], s["end"]),
                "level": "mid", "section": 0, "section_start": k > 0, "drop": False, "build": False,
                "hits": [], "ref_peak": s["peak"], "first": k == 0, "last": False,
                "ref_transition": {"type": tr, "frames": 8} if tr != "cut" else {"type": "cut"},
                "ref_punches": s["punches"], "ref_flash": s["flash"],
            })
        if slots:
            slots[-1]["last"] = True
        audio_spec = {"path": ref, "start": 0.0, "fade_in": 0.0, "fade_out": 0.0, "normalize": False} if rinfo.has_audio else None
        med = float(np.median([c["dur"] for c in clips]))
        slots = split_long(slots, w, fps, max(1.2, min(6.0, 1.8 * med)))
    opts = {"order": args.order, "effects": args.effects, "width": width, "height": height}
    seq = sequence(slots, clips, fps, opts)
    add_effects(seq, slots, clips, fps, opts)
    shots = finish(seq, slots, clips, fps)
    plan = {
        "version": 2, "mode": "montage", "kind": "reference", "engine": "director", "reference": ref,
        "width": width, "height": height, "fps": fps, "look": args.look,
        "audio": audio_spec,
        "clips": [{"path": c["path"], "duration": c["dur"]} for c in clips],
        "shots": shots,
    }
    if args.look == "match":
        plan["match"] = match_look(style["look"], clips)
    return plan


def mt_rank(v: list[float]) -> list[float]:
    import montage as mt

    return mt.rank01(v)


def _energy_between(w: dict, a: float, b: float) -> float:
    vals = [e for t, e in zip(w.get("beats", []), w.get("energy", [])) if a <= t < b]
    return float(np.mean(vals)) if vals else 0.5


def add_common(p: argparse.ArgumentParser, workers: bool = True) -> None:
    p.add_argument("--clips", nargs="+", required=True, help="video files and/or folders")
    p.add_argument("--out", required=True)
    p.add_argument("--look", default="cinematic", choices=sorted(rd.LOOKS) + ["match"])
    p.add_argument("--effects", default="full", choices=["full", "subtle", "none"])
    p.add_argument("--order", default="time", choices=["time", "free"], help="keep the trip's order (time) or best-first (free)")
    p.add_argument("--fps", type=float)
    p.add_argument("--no-clip", action="store_true", help="no CLIP shot understanding (faster, less smart)")
    if workers:
        p.add_argument("--workers", type=int, default=2)
    p.add_argument("--plan-only", action="store_true")


def add_music_args(pm: argparse.ArgumentParser, workers: bool = True) -> None:
    pm.add_argument("--music", required=True, help="a song, or any video: its sound is used")
    pm.add_argument("--length", type=float, default=30.0)
    pm.add_argument("--aspect", default="vertical", choices=sorted(ASPECTS))
    pm.add_argument("--pace", default="medium", choices=sorted(PACE))
    pm.add_argument("--music-start", default="auto")
    add_common(pm, workers)


def add_reference_args(pr: argparse.ArgumentParser, workers: bool = True) -> None:
    pr.add_argument("reference")
    pr.add_argument("--audio", help="use this soundtrack (audio or video file) instead of the reference's")
    pr.add_argument("--length", type=float)
    pr.add_argument("--aspect", default="same", choices=["same", *sorted(ASPECTS)])
    add_common(pr, workers)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_music_args(sub.add_parser("music"))
    add_reference_args(sub.add_parser("reference"))
    pre = sub.add_parser("render")
    pre.add_argument("plan")
    pre.add_argument("--out")
    pre.add_argument("--workers", type=int, default=2)
    return ap


def run(args) -> dict:
    if args.cmd == "render":
        p = Path(args.plan).resolve()
        v = rd.render(p, args.out, args.workers)
        return {"outputs": [str(v)], "plan": str(p)}
    if getattr(args, "fps", None) is None and args.cmd == "music":
        args.fps = 30.0
    plan = plan_music(args) if args.cmd == "music" else plan_reference(args)
    out = Path(args.out).resolve()
    plan_path = out.with_name(out.stem + ".montage.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    ac.save_plan(plan, plan_path)
    for i, s in enumerate(plan["shots"]):
        tr = (s.get("transition_in") or {}).get("type", "cut")
        log(f"  shot {i + 1:2d} {s['frames'] / plan['fps']:4.1f}s x{s['speed']:<4} {Path(s['clip']).stem:<16} {tr:<9} {s.get('why', '')}")
    res = {"plan": str(plan_path), "out_dir": str(out.parent)}
    if not args.plan_only:
        video = rd.render(plan_path, str(out), args.workers)
        issues = rd.quality_check(str(video), sum(s["frames"] for s in plan["shots"]), plan["fps"])
        for i in issues:
            log(f"quality check: {i}")
        res.update({"outputs": [str(video)], "issues": issues})
    import otio_export

    try:
        res["otio"] = str(otio_export.export_plan_file(plan_path))
    except AutocutError as e:
        log(f"OTIO export skipped: {e}")
    return res


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(json.dumps(run(args)))
    except AutocutError as e:
        log(f"error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
