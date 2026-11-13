"""clip_edit: build short clips from the action in a long video the way an editor would.

Why not just cut out the most intense minute: that plays as one continuous stretch with a few
jumps. An editor watches a whole fight or scene and keeps only its beats:

  1. Stretches: each clip draws from one scene (a fight, a chase), usually 2–4x longer than
     the clip, so most of it is thrown away. Scene edges are moved to quiet moments.
  2. Moments: inside the stretch, every hit/explosion/shout (sharp onsets in the sound, peaks in
     loudness and motion) is a candidate. Each is cut on action: in ~0.4–1.4 s before the impact
     (the wind-up shows) and out once the reaction has played, never inside a spoken word.
     Lines of dialogue are candidates too (whole sentences) for story context.
  3. Structure: a 2 s teaser of the best moment first (the hook), then the scene in story order
     with the strongest moments, pieces getting shorter as it builds, the climax (best moment)
     in slow motion when the source has the frames for it, and the payoff held longer at the end.
  4. Effects only where motivated: a flash cut out of the teaser, punch-in zooms on the biggest
     impacts, a slow push-in on longer calm pieces, a short fade at the very end.

With a music track, cut points move onto its beats.
"""

from __future__ import annotations

import json
import time

import numpy as np

import autocut as ac
from autocut import log

HZ = 10  # signal rate (10 values per second)
PACE = {"slow": 1.4, "medium": 1.0, "fast": 0.72}


# ---------------------------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------------------------


def _flux(y: np.ndarray, hop: int) -> np.ndarray:
    """Spectral flux per hop (50 ms at 8 kHz): how much new sound starts."""
    n = len(y) // hop
    if n < 2:
        return np.zeros(max(0, n), np.float32)
    win = np.hanning(hop).astype(np.float32)
    out = np.zeros(n, np.float32)
    prev = None
    for i in range(0, n, 4096):
        fr = y[i * hop : min(n, i + 4096) * hop].reshape(-1, hop) * win
        mag = np.log1p(10 * np.abs(np.fft.rfft(fr, axis=1)))
        d = np.diff(mag, axis=0, prepend=mag[:1] if prev is None else prev[None, :])
        out[i : i + len(mag)] = np.maximum(d, 0).sum(axis=1)
        prev = mag[-1]
    return out


def signals(src: str, duration: float) -> dict[str, np.ndarray]:
    """Loudness (dB), onsets, motion, brightness and picture cuts at 10 Hz (cached)."""
    import footage as fo
    import montage as mt

    key = ac.file_key(src, "action-v2")
    cache = ac.CACHE_DIR / "action" / f"{key}.json"
    if cache.is_file():
        d = json.loads(cache.read_text())
        return {k: np.asarray(v, dtype=np.float32) for k, v in d.items()}
    t0 = time.time()
    log("measuring the action (sound onsets, loudness, motion, scene changes); cached afterwards")
    a = fo.analyse(src)
    n = max(1, int(duration * HZ))
    mot5 = np.asarray(a["motion"], dtype=np.float32)
    br5 = np.asarray(a["bright"], dtype=np.float32)
    if len(mot5) == 0:
        mot5, br5 = np.zeros(1, np.float32), np.full(1, 100.0, np.float32)
    idx = np.minimum(len(mot5) - 1, (np.arange(n) / HZ * fo.FPS).astype(int))
    med = float(np.median(mot5))
    cut5 = mot5 > max(30.0, 5 * med)
    cut = np.zeros(n, np.float32)
    for j in np.nonzero(cut5)[0]:
        i = int(round(j / fo.FPS * HZ))
        if 0 <= i < n:
            cut[i] = 1.0
    y = mt.decode_mono(src, 8000)
    hop = 400  # 50 ms
    m = len(y) // hop
    rms = np.sqrt((y[: m * hop].reshape(m, hop) ** 2).mean(axis=1)) if m else np.zeros(0)
    db = 20 * np.log10(np.maximum(rms, 1e-5))
    fl = _flux(y, hop)

    def per10(v: np.ndarray, how) -> np.ndarray:
        out = np.full(n, float(v.min()) if len(v) else 0.0, np.float32)
        k = min(n, len(v) // 2)
        if k:
            out[:k] = how(v[: 2 * k].reshape(k, 2), axis=1)
        return out

    sig = {"loud": per10(db, np.mean), "flux": per10(fl, np.max), "motion": mot5[idx], "bright": br5[idx], "cut": cut}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({k: [round(float(x), 3) for x in v] for k, v in sig.items()}))
    log(f"action signals in {time.time() - t0:.0f}s")
    return sig


def zs(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return (v - np.median(v)) / (v.std() + 1e-6)


def smooth(v: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(v) == 0:
        return np.asarray(v, dtype=np.float64)
    pad = np.pad(np.asarray(v, dtype=np.float64), k // 2, mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")[: len(v)]


def speech_mask(words: list[dict], n: int) -> np.ndarray:
    m = np.zeros(n, np.float32)
    for w in words:
        a, b = int(w["start"] * HZ), int(np.ceil(w["end"] * HZ))
        m[max(0, a) : min(n, b)] = 1.0
    return m


def interest(sig: dict[str, np.ndarray], speech: np.ndarray) -> np.ndarray:
    """How much is happening, 10 Hz (z-score-ish; dead air, menus and black strongly negative)."""
    loud, mot = sig["loud"], sig["motion"]
    s = 0.35 * zs(smooth(loud, 5)) + 0.3 * zs(smooth(sig["flux"], 5)) + 0.25 * zs(np.log1p(mot)) + 0.12 * speech
    still = (smooth(mot, 20) < np.percentile(mot, 15)) & (smooth(loud, 20) < np.percentile(loud, 40))
    s[still] -= 1.0
    s[sig["bright"] < 14] -= 2.0
    return s


# ---------------------------------------------------------------------------------------------
# Stretches (one scene per clip)
# ---------------------------------------------------------------------------------------------


def stretches(score: np.ndarray, count: int, target: float, duration: float) -> list[tuple[float, float]]:
    """`count` non-overlapping source stretches (seconds), each about 2.5–3.5x the clip length,
    around the busiest scenes, with edges moved to the quietest nearby moment."""
    per_s = score[: len(score) // HZ * HZ].reshape(-1, HZ).mean(axis=1) if len(score) >= HZ else np.asarray([score.mean()])
    secs = len(per_s)
    R = int(np.clip(3.0 * target, 45, 240))
    R = int(min(R, max(20, duration / max(1, count))))
    act = smooth(per_s, max(3, R // 2))
    calm = smooth(per_s, 9)
    used = np.zeros(secs, bool)
    out = []
    for _ in range(count):
        best = None
        for c in np.argsort(-act):
            c = int(c)
            s0, s1 = max(0, c - R // 2), min(secs, c + R // 2)
            if s1 - s0 >= min(R, secs) * 0.6 and not used[s0:s1].any():
                best = (s0, s1)
                break
        if best is None:
            break
        s0, s1 = best
        # edges at the calmest point within ±15% of the stretch (between fights, not inside one)
        m = max(3, R // 7)

        def calmest(c0: int, lo_ok: int, hi_ok: int) -> int:
            lo, hi = max(lo_ok, c0 - m), min(hi_ok, c0 + m)
            if hi <= lo:
                return c0
            return lo + int(np.argmin(calm[lo:hi]))

        free_lo = s0
        while free_lo > 0 and not used[free_lo - 1] and s0 - free_lo < m:
            free_lo -= 1
        free_hi = s1
        while free_hi < secs and not used[free_hi] and free_hi - s1 < m:
            free_hi += 1
        a = calmest(s0, free_lo, s0 + m)
        b = calmest(s1, s1 - m, free_hi)
        if b - a < R * 0.5:
            a, b = s0, s1
        used[max(0, a - 2) : min(secs, b + 2)] = True
        out.append((float(a), float(b)))
    return sorted(out)


# ---------------------------------------------------------------------------------------------
# Moments (cut on action)
# ---------------------------------------------------------------------------------------------


def word_safe(t: float, words: list[dict], later: bool) -> float:
    """Move a cut point out of a spoken word."""
    for w in words:
        if w["start"] < t < w["end"]:
            return float(w["end"]) + 0.06 if later else float(w["start"]) - 0.06
    return t


def phrases(words: list[dict], max_len: float = 6.0, pause: float = 0.3) -> list[dict]:
    """Spoken phrases: split at sentence ends, commas, breaths (pauses) and ~6 s at most, so
    narration can be cut between phrases without cutting into one."""
    out: list[dict] = []
    cur: list[dict] = []

    def flush() -> None:
        if cur:
            out.append({"start": float(cur[0]["start"]), "end": float(cur[-1]["end"]), "words": list(cur),
                        "text": " ".join(w["word"].strip() for w in cur)})
            cur.clear()

    for w in words:
        t = w["word"].strip()
        if not t:
            continue
        if cur and (w["start"] - cur[-1]["end"] > pause or w["end"] - cur[0]["start"] > max_len):
            flush()
        cur.append(w)
        if t[-1] in ".?!,;:":
            flush()
    flush()
    return out


def sentence_safe(a: float, b: float, sents: list[dict], grow_in: float = 1.5, grow_out: float = 2.5) -> tuple[float, float]:
    """Never start or end a piece in the middle of a spoken sentence: grow to include the whole
    sentence when that is short, else leave the sentence out."""
    for s in sents:
        if s["start"] < a < s["end"]:
            a = s["start"] - 0.1 if a - s["start"] <= grow_in else s["end"] + 0.1
        if s["start"] < b < s["end"]:
            b = s["end"] + 0.2 if s["end"] - b <= grow_out else s["start"] - 0.1
    return a, b


def usable(a: float, b: float, sig: dict[str, np.ndarray]) -> tuple[float, float] | None:
    """Trim black (fades) off the ends of a piece; None when what is left is short, mostly dark
    or a frozen picture (pause screens, held frames)."""
    br, mot = sig["bright"], sig["motion"]
    n = len(br)
    i, j = max(0, int(a * HZ)), min(n, int(np.ceil(b * HZ)))
    while i < j and br[i] < 18:
        i += 1
    while j > i and br[j - 1] < 18:
        j -= 1
    if j - i < 10:
        return None
    if float((br[i:j] < 18).mean()) > 0.2 or float((mot[i:j] < 0.3).mean()) > 0.5:
        return None
    return (max(a, i / HZ), min(b, j / HZ))


def moments(score: np.ndarray, sig: dict[str, np.ndarray], words: list[dict], s0: float, s1: float, pace: float = 1.0) -> list[dict]:
    """Candidate pieces inside [s0, s1]: action peaks cut on action, plus whole lines of dialogue."""
    import highlights as hl

    n = len(score)
    i0, i1 = max(0, int(s0 * HZ)), min(n, int(s1 * HZ))
    if i1 - i0 < 20:
        return []
    f = smooth(score, 5)
    flux = sig["flux"]
    loud = sig["loud"]
    cut = sig["cut"]
    sents = phrases([w for w in words if s0 - 10 <= w["start"] < s1 + 10])
    seg = f[i0:i1]
    order = np.argsort(-seg)
    peaks: list[int] = []
    for j in order:
        p = i0 + int(j)
        if f[p] < np.percentile(seg, 35):
            break
        if all(abs(p - q) >= 20 for q in peaks):
            peaks.append(p)
    lo_s, hi_s = float(np.percentile(seg, 20)), float(np.percentile(seg, 98))
    out = []
    for p in peaks:
        a = max(i0, p - 15)
        ti = a + int(np.argmax(flux[a : min(i1, p + 4)]))  # the impact: the sharpest onset near the peak
        strength = float(np.clip((f[p] - lo_s) / max(1e-6, hi_s - lo_s), 0, 1))
        # in: a picture cut 0.4–1.4 s before the impact, else the quietest moment there
        lo, hi = max(i0, ti - 14), max(i0, ti - 4)
        cuts = [i for i in range(lo, hi) if cut[i] > 0]
        start = (max(cuts) if cuts else lo + int(np.argmin(loud[lo:hi]))) if hi > lo else ti
        # out: once the reaction has played (sound back down) and before the next big onset
        L = (1.6 + 3.2 * strength) * pace
        e_lo, e_hi = ti + 8, min(i1, ti + int(L * HZ) + 1)
        end = e_hi
        if e_hi > e_lo:
            base = float(loud[ti]) - 9
            thr = float(np.percentile(flux[i0:i1], 92))
            for i in range(e_lo, e_hi):
                if loud[i] < base or cut[i] > 0 or (flux[i] > thr and i > ti + 12):
                    end = i
                    break
        a_t, b_t = sentence_safe(start / HZ, end / HZ, sents)
        a_t = word_safe(a_t, words, later=False)
        b_t = word_safe(b_t, words, later=True)
        if b_t - a_t < 1.0:
            continue
        ok = usable(a_t, b_t, sig)
        if ok is None:
            continue  # fades to black, loading screens, held frames
        a_t, b_t = ok
        out.append({"in": round(a_t, 2), "out": round(b_t, 2), "impact": round(ti / HZ, 2), "score": round(strength, 3), "kind": "action"})
    # dialogue: whole lines give the story (who, why); scored lower than the best action
    for s in phrases([w for w in words if s0 <= w["start"] < s1]):
        nwords = len([w for w in s["words"] if hl.norm_word(w["word"]) not in hl.FILLERS])
        dur = s["end"] - s["start"]
        if nwords >= 3 and 1.0 <= dur <= 7.0 and usable(s["start"] - 0.12, s["end"] + 0.3, sig) == (s["start"] - 0.12, s["end"] + 0.3):
            mid = int((s["start"] + s["end"]) / 2 * HZ)
            val = float(np.clip((f[min(n - 1, mid)] - lo_s) / max(1e-6, hi_s - lo_s), 0, 1))
            out.append({"in": round(s["start"] - 0.12, 2), "out": round(s["end"] + 0.3, 2), "impact": None,
                        "score": round(0.25 + 0.5 * val + (0.15 if "?" in s["text"] or "!" in s["text"] else 0), 3), "kind": "line", "text": s["text"]})
    return out


# ---------------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------------


def overlaps(m: dict, chosen: list[dict], margin: float = 0.4) -> bool:
    return any(not (m["out"] + margin <= c["in"] or m["in"] >= c["out"] + margin) for c in chosen)


def piece_cap(x: float, pace: float) -> float:
    """Longest a piece may be at position x (0 = start, 1 = end): pieces get shorter as it builds."""
    return (6.0 - 2.8 * x) * pace


def normalise(chosen: list[dict], pace: float) -> list[dict]:
    """Story order; neighbours closer than 0.8 s are joined when the result stays short, else the
    weaker one goes (a 0.5 s jump would look like a mistake); long pieces capped by position."""
    out: list[dict] = []
    for m in sorted((dict(c) for c in chosen), key=lambda c: c["in"]):
        if out and m["in"] - out[-1]["out"] < 0.8:
            prev = out[-1]
            if max(prev["out"], m["out"]) - prev["in"] <= 6.5 * pace:
                prev["out"] = max(prev["out"], m["out"])
                if m["score"] > prev["score"]:
                    prev["score"], prev["impact"] = m["score"], m.get("impact") or prev.get("impact")
                prev["kind"] = "action" if "action" in (prev["kind"], m["kind"]) else "line"
                prev.setdefault("text", m.get("text"))
            elif m["score"] > prev["score"]:
                out[-1] = m
            continue
        out.append(m)
    total = sum(m["out"] - m["in"] for m in out) or 1.0
    acc = 0.0
    for k, m in enumerate(out):
        length = m["out"] - m["in"]
        if m["kind"] == "action" and 0 < k < len(out) - 1:
            cap = max(1.6, piece_cap(acc / total, pace))
            imp = m.get("impact")
            if length > cap:
                if imp is not None:  # keep the impact inside: trim both ends around it
                    a = max(m["in"], min(imp - 0.5, m["out"] - cap))
                    m["in"], m["out"] = round(a, 2), round(a + cap, 2)
                else:
                    m["out"] = round(m["in"] + cap, 2)
        acc += m["out"] - m["in"]
    return out


def assemble(cands: list[dict], budget: float, mn: float, mx: float, max_lines: int = 3, pace: float = 1.0) -> list[dict]:
    """Best moments up to `budget` seconds, in story (source) order."""
    chosen: list[dict] = []
    pieces: list[dict] = []
    lines = 0
    for m in sorted(cands, key=lambda c: -c["score"]):
        if overlaps(m, chosen, 0.2):
            continue
        if m["kind"] == "line":
            if lines >= max_lines:
                continue
            lines += 1
        chosen.append(m)
        pieces = normalise(chosen, pace)
        if sum(p["out"] - p["in"] for p in pieces) >= budget:
            break

    def total_len() -> float:
        return sum(m["out"] - m["in"] for m in pieces)

    def floor(m: dict) -> float:
        return max(m["in"] + 1.6, (m.get("impact") or m["in"]) + 0.9)

    for _ in range(400):  # trim to the budget: longest action tails first, then the weakest piece
        excess = total_len() - min(mx, budget + 1.0)
        if excess <= 0 or len(pieces) <= 1:
            break
        act = [m for m in pieces if m["kind"] == "action" and m["out"] - floor(m) > 0.25]
        if act:
            m = max(act, key=lambda c: c["out"] - floor(c))
            m["out"] = round(m["out"] - min(0.5, excess, m["out"] - floor(m)), 2)
        else:
            pieces.remove(min(pieces, key=lambda c: c["score"]))
    return pieces


def edit_clip(cands: list[dict], s0: float, s1: float, target: float, mn: float, mx: float, *, hook: bool,
              src_fps: float, words: list[dict], pace: float = 1.0) -> dict | None:
    """One clip: hook teaser + story-order pieces + slow-motion climax + held payoff.
    Returns {"pieces": [...], "shots": [...] (renderer v2, without framing), "climax": t} or None."""
    if not cands:
        return None
    teaser_len = 2.0 if hook else 0.0
    goal = float(np.clip(target, mn, mx))
    slow_extra = 1.0  # the slowed climax adds about this much
    # a stretch carried by speech (narration, a story scene) may use more whole lines
    span = max(1.0, s1 - s0)
    spoken = sum(max(0.0, min(s1, w["end"]) - max(s0, w["start"])) for w in words if w["end"] > s0 and w["start"] < s1) / span
    max_lines = 3 + int(round(12 * min(1.0, spoken)))
    pieces = assemble(cands, goal - teaser_len - slow_extra, mn, mx, max_lines=max_lines, pace=pace)
    if not pieces:
        return None
    acts = [p for p in pieces if p["kind"] == "action" and p.get("impact") is not None]
    late = [p for p in acts if pieces.index(p) >= len(pieces) // 3]
    climax = max(late or acts or pieces, key=lambda p: p["score"])  # the peak belongs in the second part
    best = max(acts or pieces, key=lambda p: p["score"])  # the teaser shows the very best moment
    # payoff: hold the last piece a little longer (the outcome, the breath after the fight)
    last = pieces[-1]
    if last["kind"] == "action":
        last["out"] = round(min(s1, last["out"] + 1.5 * pace), 2)
        last["out"] = word_safe(last["out"], words, later=True)
        _, last["out"] = sentence_safe(last["in"], last["out"], phrases([w for w in words if s0 <= w["start"] < s1 + 10]))
    # setup: the opening piece may start a bit earlier (establish where we are)
    first = pieces[0]
    if first["kind"] == "action" and first is not climax:
        first["in"] = round(max(s0, first["in"] - 0.8), 2)
        first["in"] = word_safe(first["in"], words, later=False)
    shots: list[dict] = []
    ti = climax.get("impact")
    tb = best.get("impact")
    show_teaser = hook and tb is not None and (sum(p["out"] - p["in"] for p in pieces[: pieces.index(best)]) > 5.0)
    if show_teaser and tb is not None:
        a = max(best["in"], tb - 0.7)
        shots.append({"in": a, "len": 1.9, "speed": 1.0, "role": "hook", "zoom": {"from": 1.04, "to": 1.12}, "punches": [round(tb - a, 2)]})
    words_at = lambda a, b: any(w["end"] > a and w["start"] < b for w in words)  # noqa: E731
    big = sorted((p["score"] for p in acts), reverse=True)
    punch_cut = big[min(len(big) - 1, max(0, len(big) // 3))] if big else 2.0
    for k, p in enumerate(pieces):
        length = p["out"] - p["in"]
        slow_ok = p is climax and ti is not None and src_fps >= 47 and not words_at(ti - 0.2, ti + 0.7) and length >= 1.8
        if slow_ok and ti is not None:
            s_a = max(p["in"] + 0.3, ti - 0.15)
            s_b = min(p["out"] - 0.2, ti + 0.55)
            if s_a - p["in"] >= 0.3:
                shots.append({"in": p["in"], "len": s_a - p["in"], "speed": 1.0, "role": "build"})
            slow = max(0.4, 30.0 / src_fps)  # one real source frame per output frame (60 fps -> 0.5x)
            shots.append({"in": s_a, "len": (s_b - s_a) / slow, "speed": slow, "role": "climax", "punches": [0.0], "flash": [0.0]})
            if p["out"] - s_b >= 0.3:
                shots.append({"in": s_b, "len": p["out"] - s_b, "speed": 1.0, "role": "climax-after"})
            continue
        sh: dict = {"in": p["in"], "len": length, "speed": 1.0, "role": "line" if p["kind"] == "line" else "action"}
        imp = p.get("impact")
        if imp is not None and p["score"] >= punch_cut and k % 2 == 0 and p is not climax:
            sh["punches"] = [round(max(0.0, imp - p["in"]), 2)]
        elif length >= 3.5:
            sh["zoom"] = {"from": 1.0, "to": 1.07} if k % 2 == 0 else {"from": 1.07, "to": 1.0}
        shots.append(sh)
    if show_teaser and len(shots) > 1:
        shots[1]["transition_in"] = {"type": "fadewhite", "frames": 6}
    if shots:
        shots[-1]["role"] = shots[-1].get("role", "") + " payoff"
        shots[-1]["fade_out_s"] = 0.35
    return {"pieces": [(p["in"], p["out"]) for p in pieces], "shots": shots, "climax": ti, "span": [s0, s1],
            "lines": [p.get("text") for p in pieces if p["kind"] == "line"]}


def snap_to_beats(shots: list[dict], beats: list[float], tol: float = 0.3, min_len: float = 0.8) -> None:
    """Move each cut onto the nearest music beat (by lengthening/shortening the shot before it)."""
    if not beats:
        return
    b = np.asarray(beats)
    t = 0.0
    for k, s in enumerate(shots[:-1]):
        end = t + s["len"]
        j = int(np.argmin(np.abs(b - end)))
        d = float(b[j] - end)
        if abs(d) <= tol and s["len"] + d >= min_len and s.get("speed", 1.0) == 1.0:
            s["len"] += d
        t += s["len"]


def describe(clip: dict) -> str:
    sh = clip["shots"]
    return f"{len(sh)} shots, {sum(s['len'] for s in sh):.0f}s from {clip['span'][1] - clip['span'][0]:.0f}s of source"
