"""score: emotional sound design under a voice (or under a clip's own sound).

Input: the lines of a narration with their emotion and intensity (from the script, chosen by
Claude, or set by you), the picture cuts, and where the music comes from (music_library). Output:
an editable sound plan (`.score.json`) and the mixed sound.

How the plan is made, like a trailer editor would:
  - Acts: consecutive lines of the same emotional family share one music cue; very short acts join
    a neighbour. Each act gets its own track (no track twice when the library allows).
  - Transitions: a line that lands a surprise, a triumph, rage or fear after a calmer act gets a
    hard transition: the music stops just before it (a beat of silence), an impact hits on the
    first word and the new cue starts on it. Other acts crossfade in the pause between lines.
  - Peaks (intense, high-energy lines): a riser leads into them and an impact lands on them.
  - The emotion's own colour: a shimmer for wonder, a chime for joy, a heartbeat under dread, a
    sub drop under grief or rage, a swell into love and hope, whooshes on fast cuts.
  - The hook: the first line opens on an impact (or a braam for mystery, dread and awe).
  - Mix: music levelled per track, ducked under the voice (less under game sound), swelling in the
    pauses, louder for intense acts; effects at fixed levels; loudness -14 LUFS at the end.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import autocut as ac
import emotions as em
import sfx as fx
from autocut import AutocutError, log

SR = fx.SR
SFX_DB = {"impact": -5.0, "riser": -11.0, "whoosh": -15.0, "sub": -9.0, "braam": -8.0, "shimmer": -15.0, "chime": -14.0, "heartbeat": -13.0,
          "swell": -14.0}
HARD_IN = {"Surprise", "Amazement", "Triumph", "Rage", "Anger", "Fear", "Euphoria", "Thrill"}
BRAAM_HOOK = {"mystery", "fear", "tension", "wonder"}


# ---------------------------------------------------------------------------------------------
# Lines -> acts and events (pure logic, unit-tested)
# ---------------------------------------------------------------------------------------------


def normalise_lines(lines: list[dict]) -> list[dict]:
    out = []
    for ln in lines:
        name = em.resolve(ln.get("emotion"))
        inten = ln.get("intensity", ln.get("energy"))
        try:
            inten = float(inten) if inten is not None else 0.4 + 0.5 * em.EMOTIONS[name]["arousal"]
        except (TypeError, ValueError):
            inten = 0.6
        out.append({**ln, "start": float(ln["start"]), "end": float(ln["end"]), "emotion": name, "intensity": float(np.clip(inten, 0, 1)),
                    "family": em.EMOTIONS[name]["family"], "arousal": em.EMOTIONS[name]["arousal"]})
    return sorted(out, key=lambda x: x["start"])


def make_acts(lines: list[dict], total: float, min_len: float = 6.0, change_cost: float = 3.0) -> list[dict]:
    """Split the lines into acts covering [0, total), one music cue each, following the emotional
    arc: the split that best fits every line's mood (each act takes the mood closest to its lines,
    weighted by how long and how strongly they are felt), with a cost per music change and no act
    shorter than `min_len` (the payoff at the end may be 6 s). Dynamic programming over the line
    boundaries."""
    n = len(lines)
    if n == 0:
        return []
    fams = list(em.FAMILIES)
    w = np.asarray([(l["end"] - l["start"]) * (0.4 + l["intensity"]) for l in lines])
    # another mood is always another kind of music (a base cost), more so the further apart they are
    dist = np.asarray([[0.0 if f == l["family"] else 0.6 + em.family_distance(f, l["family"]) for l in lines] for f in fams])

    def span(i: int, j: int) -> tuple[float, float]:
        return (0.0 if i == 0 else lines[i]["start"]), (lines[j]["start"] if j < n else total)

    best = [0.0] + [np.inf] * n
    back = [0] * (n + 1)
    fam_of: dict[tuple[int, int], int] = {}
    for j in range(1, n + 1):
        for i in range(j):
            s0, s1 = span(i, j)
            short = s1 - s0 < (min(min_len, 6.0) if j == n else min_len)
            if short and not (i == 0 and j == n):
                continue
            mis = dist[:, i:j] @ w[i:j]  # per family
            f = int(np.argmin(mis))
            c = best[i] + float(mis[f]) + change_cost
            if c < best[j]:
                best[j], back[j] = c, i
                fam_of[(i, j)] = f
    if not np.isfinite(best[n]):  # too short for any split: one act
        back[n] = 0
        fam_of[(0, n)] = int(np.argmin(dist @ w))
    bounds, j = [], n
    while j > 0:
        bounds.append((back[j], j))
        j = back[j]
    acts = []
    for i, j in reversed(bounds):
        fam = fams[fam_of[(i, j)]]
        s0, s1 = span(i, j)
        ls = list(range(i, j))
        # the act's emotion: its strongest line of that mood (or the nearest mood)
        top = max(ls, key=lambda k: w[k] * (1.0 / (0.3 + em.family_distance(fam, lines[k]["family"]))))
        acts.append({"family": fam, "lines": ls, "start": s0, "end": s1, "emotion": lines[top]["emotion"],
                     "intensity": round(float(max(lines[k]["intensity"] for k in ls)), 2)})
    return acts


def _clear(events: list[dict], t: float, kinds: tuple[str, ...], gap: float) -> bool:
    return all(abs(e["t"] - t) >= gap for e in events if e["kind"] in kinds)


def plan_events(lines: list[dict], acts: list[dict], cuts: list[float], total: float, *, music: bool = True) -> tuple[list[dict], list[dict]]:
    """(effects, transitions): effects [{kind, t, seconds?, why}], transitions per act boundary."""
    ev: list[dict] = []
    trans: list[dict] = []
    if not lines:
        return ev, trans
    first = lines[0]
    if first["arousal"] >= 0.55 or first["intensity"] >= 0.7 or first["family"] in BRAAM_HOOK:
        kind = "braam" if first["family"] in BRAAM_HOOK and first["family"] != "tension" or first["emotion"] in ("Awe", "Dread", "Mystery") else "impact"
        ev.append({"kind": kind, "t": max(0.0, first["start"] - 0.02), "why": f"hook ({first['emotion']})"})
    for k in range(1, len(acts)):
        a, prev = acts[k], acts[k - 1]
        ln = lines[a["lines"][0]]
        prev_end = lines[prev["lines"][-1]]["end"]
        jump = em.EMOTIONS[a["emotion"]]["arousal"] - em.EMOTIONS[prev["emotion"]]["arousal"]
        hard = (ln["emotion"] in HARD_IN and ln["intensity"] >= 0.55) or (jump >= 0.35 and ln["intensity"] >= 0.6)
        if hard:
            gap_start = max(prev_end, ln["start"] - 0.6)
            trans.append({"at": ln["start"], "type": "hit", "silence_from": round(gap_start, 3)})
            before = lines[a["lines"][0] - 1] if a["lines"][0] > 0 else None
            builds = prev["family"] in ("tension", "heroic", "euphoria", "fear", "mystery") or (before is not None and before["family"] in ("tension", "fear"))
            if (builds or ln["emotion"] in ("Triumph", "Euphoria", "Thrill")) and ln["start"] >= 2.5 and _clear(ev, ln["start"], ("riser",), 6):
                ev.append({"kind": "riser", "t": ln["start"], "seconds": float(min(3.0, ln["start"] - 0.5)), "why": f"into {ln['emotion']}"})
            ev.append({"kind": "impact" if ln["family"] != "wonder" or ln["emotion"] == "Surprise" else "braam", "t": ln["start"], "why": f"{ln['emotion']} lands"})
        else:
            mid = (prev_end + ln["start"]) / 2
            trans.append({"at": round(mid, 3), "type": "crossfade", "seconds": round(float(np.clip(ln["start"] - prev_end + 1.0, 1.0, 2.5)), 2)})
    for i, ln in enumerate(lines):
        e = em.EMOTIONS[ln["emotion"]]
        t = ln["start"]
        if i == 0:
            continue
        # peaks: a riser into the line and an impact on it
        if ln["intensity"] >= 0.85 and ln["arousal"] >= 0.7 and _clear(ev, t, ("impact", "braam"), 4.0):
            if t >= 2.5 and _clear(ev, t, ("riser",), 8.0):
                ev.append({"kind": "riser", "t": t, "seconds": float(min(2.5, t - 0.5)), "why": f"peak ({ln['emotion']})"})
            ev.append({"kind": "impact", "t": t, "why": f"peak ({ln['emotion']})"})
            continue
        # a reveal inside an act: the music drops out for a beat, then an impact lands the line
        if ln["emotion"] in HARD_IN and ln["intensity"] >= 0.75 and t not in {a["start"] for a in acts}:
            if _clear(ev, t, ("impact", "braam"), 4.0):
                ev.append({"kind": "dip", "t": t, "seconds": float(min(0.6, max(0.3, e["pause_before"]))), "why": f"beat before {ln['emotion']}"})
                ev.append({"kind": "impact", "t": t, "why": f"{ln['emotion']} lands"})
                continue
        if ln["intensity"] < 0.6:
            continue
        for kind in e["sfx"]:
            if kind in ("impact", "riser", "whoosh"):
                continue  # placed by the rules above and at cuts
            if kind == "heartbeat" and _clear(ev, t, ("heartbeat",), 12.0):
                ev.append({"kind": "heartbeat", "t": t, "seconds": float(min(6.0, ln["end"] - t + 1.5)), "why": ln["emotion"]})
            elif kind in ("shimmer", "chime", "sub", "braam") and _clear(ev, t, (kind,), 8.0) and _clear(ev, t, ("impact", "braam"), 1.5):
                ev.append({"kind": kind, "t": t, "why": ln["emotion"]})
            elif kind == "swell" and t >= 1.8 and _clear(ev, t, ("swell", "riser"), 8.0):
                ev.append({"kind": "swell", "t": t, "seconds": 1.6, "why": f"into {ln['emotion']}"})
            break  # one colour per line
    # whooshes on the cuts of energetic acts
    for c in cuts:
        # the line being heard at the cut (or the one just before it) sets the energy
        before = [ln for ln in lines if ln["start"] <= c]
        if not before or before[-1]["arousal"] <= 0.65:
            continue
        if _clear(ev, c, ("whoosh",), 2.5) and _clear(ev, c, ("impact", "braam", "riser"), 0.6):
            ev.append({"kind": "whoosh", "t": c, "why": "fast cut"})
    if not music:
        trans = []
    return sorted(ev, key=lambda e: e["t"]), trans


# ---------------------------------------------------------------------------------------------
# Choosing the music
# ---------------------------------------------------------------------------------------------


def choose_music(acts: list[dict], source: str, *, folder: str | None = None, licence: str = "monetized", lib=None) -> None:
    """Adds {track, offset, title, credit} to every act."""
    import music as mu
    import music_library as ml

    used: set[str] = set()
    for k, a in enumerate(acts):
        length = a["end"] - a["start"] + 3.0
        try:
            pool = ml.candidates(a["family"], source, folder=folder, licence=licence, length=length, seed=k, lib=lib)
        except AutocutError as e:
            log(f"no music for act {k + 1} ({e})")
            pool = []
        if not pool:
            continue
        fresh = [t for t in pool if t["file"] not in used] or pool
        best = fresh[0]
        used.add(best["file"])
        try:
            m = mu.analyse(best["file"])
            off = float(mu.choose_window(m, min(length, m["duration"] - 0.5))) if m["duration"] > length else 0.0
        except AutocutError:
            off = 0.0
        a.update({"track": best["file"], "offset": round(off, 3), "title": best.get("title", Path(best["file"]).stem),
                  "credit": {k2: best.get(k2, "") for k2 in ("title", "author", "license", "page", "source")}})


# ---------------------------------------------------------------------------------------------
# Mixing
# ---------------------------------------------------------------------------------------------


def decode(path: str, start: float = 0.0, seconds: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(0.0, start):.3f}"]
    if seconds is not None:
        cmd += ["-t", f"{seconds:.3f}"]
    cmd += ["-i", path, "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy() if len(raw) >= 8 else np.zeros((0, 2), np.float32)


def presence(y: np.ndarray, attack: float = 0.03, release: float = 0.3) -> np.ndarray:
    """0 (silence) … 1 (full voice) per sample, with a fast attack and a slow release."""
    hop = int(0.01 * SR)
    mono = np.abs(y).mean(axis=1) if y.ndim == 2 else np.abs(y)
    n = len(mono) // hop + 1
    rms = np.sqrt(np.add.reduceat(np.pad(mono, (0, n * hop - len(mono))) ** 2, np.arange(0, n * hop, hop)) / hop)
    db = 20 * np.log10(rms + 1e-7)
    p = np.clip((db + 45) / 20, 0, 1)
    out = np.empty_like(p)
    acc = 0.0
    ka, kr = 1 - np.exp(-0.01 / attack), 1 - np.exp(-0.01 / release)
    for i, v in enumerate(p):
        acc += (ka if v > acc else kr) * (v - acc)
        out[i] = acc
    return np.repeat(out, hop)[: len(mono)]


def loop_to(y: np.ndarray, n: int, xf: int = int(1.0 * SR)) -> np.ndarray:
    """A cue looped (with crossfades) to n samples."""
    if len(y) >= n or len(y) < 2 * xf:
        return np.pad(y, ((0, max(0, n - len(y))), (0, 0)))[:n]
    out = y.copy()
    ramp = np.linspace(0, 1, xf)[:, None]
    while len(out) < n:
        out = np.concatenate([out[:-xf], out[-xf:] * (1 - ramp) + y[:xf] * ramp, y[xf:]])
    return out[:n]


def level(y: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Every track to the same loudness (RMS of its louder half), so cues sit at one level."""
    if len(y) == 0:
        return y
    frames = np.sqrt((y[: len(y) // 4800 * 4800].reshape(-1, 4800, 2) ** 2).mean(axis=(1, 2)) + 1e-12)
    ref = float(np.sqrt(np.mean(np.sort(frames)[len(frames) // 2 :] ** 2))) if len(frames) else 1.0
    return y * (10 ** (target_db / 20) / max(ref, 1e-6))


def music_bed(acts: list[dict], trans: list[dict], total: float, music_db: float) -> np.ndarray:
    n = int(total * SR)
    bed = np.zeros((n, 2), np.float32)
    for k, a in enumerate(acts):
        if not a.get("track"):
            continue
        # transitions[k - 1] leads into act k (one per boundary)
        tin = trans[k - 1] if 0 < k <= len(trans) else None
        tout = trans[k] if k < len(trans) else None
        fade_in = 0.25 if k == 0 else (0.03 if tin and tin["type"] == "hit" else (tin or {}).get("seconds", 1.5))
        fade_out = 2.5 if k == len(acts) - 1 else (0.06 if tout and tout["type"] == "hit" else (tout or {}).get("seconds", 1.5))
        s = a["start"] - (0.0 if k == 0 or (tin and tin["type"] == "hit") else fade_in / 2)
        e = a["end"] + (0.0 if k == len(acts) - 1 or (tout and tout["type"] == "hit") else fade_out / 2)
        if tout and tout["type"] == "hit":
            e = min(e, tout["silence_from"])
        s, e = max(0.0, s), min(total, e)
        if e - s < 0.2:
            continue
        y = decode(a["track"], a.get("offset", 0.0), e - s + 0.1)
        y = level(loop_to(y, int((e - s) * SR)))
        m = len(y)
        env = np.ones(m)
        fi, fo = min(m, int(fade_in * SR)), min(m, int(fade_out * SR))
        if fi:
            env[:fi] = np.linspace(0, 1, fi) ** 1.5
        if fo:
            env[m - fo :] *= np.linspace(1, 0, fo) ** 1.5
        gain = 10 ** ((music_db + 4.0 * (a.get("intensity", 0.6) - 0.6)) / 20)
        i0 = int(s * SR)
        seg = (y * env[:, None] * gain)[: n - i0]
        bed[i0 : i0 + len(seg)] += seg.astype(np.float32)
    return bed


def dips(events: list[dict], n: int) -> np.ndarray:
    """Music gain for the beats of silence before reveals: down to -30 dB over the beat, back up
    over 0.4 s once the line lands."""
    g = np.ones(n)
    for e in events:
        if e["kind"] != "dip":
            continue
        t, d = float(e["t"]), float(e.get("seconds", 0.4))
        a, b, c = int((t - d) * SR), int(t * SR), int((t + 0.4) * SR)
        a, b, c = max(0, a), max(0, min(n, b)), max(0, min(n, c))
        floor = 10 ** (-30 / 20)
        if b > a:
            g[a:b] = np.minimum(g[a:b], np.linspace(1, floor, b - a) ** 1.5)
        if c > b:
            g[b:c] = np.minimum(g[b:c], np.linspace(floor, 1, c - b))
    return g


def effects_track(events: list[dict], total: float, effects_db: float = 0.0) -> np.ndarray:
    n = int(total * SR)
    out = np.zeros((n, 2), np.float32)
    for k, e in enumerate(events):
        if e["kind"] not in fx.KINDS:  # (dips shape the music, they make no sound)
            continue
        y = fx.make(e["kind"], e.get("seconds"), seed=k)
        gain = 10 ** ((SFX_DB.get(e["kind"], -12) + effects_db) / 20)
        # a riser or swell ends on its moment; everything else starts on it
        i0 = int(e["t"] * SR) - (len(y) if e["kind"] in ("riser", "swell") else 0)
        if i0 < 0:
            y, i0 = y[-i0:], 0
        seg = y[: max(0, n - i0)] * gain
        out[i0 : i0 + len(seg)] += seg
    return out


def mix(primary: str, plan: dict, dst: Path, *, duck_db: float = 10.0, primary_db: float = 0.0, loudness: float = -14.0,
        duck_source: str | None = None) -> None:
    """primary (voice, or a clip's own sound) + music bed (ducked) + effects -> dst (48 kHz wav).
    The music ducks under `duck_source` (the voice alone) when the primary also carries other sound."""
    total = float(plan["total"])
    n = int(total * SR)
    v = decode(primary)
    v = np.pad(v, ((0, max(0, n - len(v))), (0, 0)))[:n] * (10 ** (primary_db / 20))
    if duck_source:
        d = decode(duck_source)
        pres = presence(np.pad(d, ((0, max(0, n - len(d))), (0, 0)))[:n])
    else:
        pres = presence(v)
    bed = music_bed(plan["acts"], plan["transitions"], total, plan.get("music_db", -2.0)) if plan.get("music", True) else np.zeros((n, 2), np.float32)
    bed *= (10 ** (-duck_db * pres / 20))[:, None]
    bed *= dips(plan["events"], n)[:, None]
    fxt = effects_track(plan["events"], total, plan.get("effects_db", 0.0)) if plan.get("effects", True) else np.zeros((n, 2), np.float32)
    fxt *= (10 ** (-3.0 * pres / 20))[:, None]
    out = v + bed + fxt
    peak = float(np.abs(out).max(initial=0.0))
    if peak > 0.98:
        out *= 0.98 / peak
    with tempfile.TemporaryDirectory(prefix="ei-score-") as tmp:
        raw = Path(tmp) / "mix.f32"
        out.astype(np.float32).tofile(raw)
        ac.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", str(raw),
                "-af", f"loudnorm=I={loudness}:TP=-1.5:LRA=11,aresample={SR}", "-c:a", "pcm_s16le", str(dst)])


# ---------------------------------------------------------------------------------------------
# One call
# ---------------------------------------------------------------------------------------------


def plan(lines: list[dict], total: float, *, cuts: list[float] | None = None, source: str = "internet", folder: str | None = None,
         licence: str = "monetized", music: bool = True, effects: bool = True, music_db: float = -2.0, lib=None) -> dict:
    ls = normalise_lines(lines)
    # a short video holds 2–4 cues, a long one changes music at most every ~25 s
    acts = make_acts(ls, total, min_len=float(np.clip(total / 5, 6.0, 25.0)))
    events, trans = plan_events(ls, acts, sorted(cuts or []), total, music=music)
    if music:
        choose_music(acts, source, folder=folder, licence=licence, lib=lib)
    for a in acts:
        a["lines"] = [int(i) for i in a["lines"]]
    log("score: " + " | ".join(f"{a['start']:.0f}s {a['emotion']} ({a['family']}) {('“' + a['title'][:30] + '”') if a.get('title') else 'no music'}"
                               for a in acts) + f"; {len(events)} effects")
    return {"version": 1, "total": round(total, 3), "music": music, "effects": effects, "music_db": music_db, "effects_db": 0.0, "source": source,
            "lines": [{k: l[k] for k in ("start", "end", "text", "emotion", "intensity") if k in l} for l in ls],
            "acts": acts, "transitions": trans, "events": events}


def credits(plan_: dict) -> list[str]:
    out, seen = [], set()
    for a in plan_.get("acts", []):
        c = a.get("credit") or {}
        if not c or c.get("source") == "folder" or a.get("track") in seen:
            continue
        seen.add(a.get("track"))
        out.append(f"Music: “{c.get('title', '')}” by {c.get('author') or 'unknown'} ({c.get('license', '')}) {c.get('page', '')}".strip())
    return out


def render(primary: str, lines: list[dict], total: float, dst: str | Path, *, duck_db: float = 10.0, duck_source: str | None = None,
           plan_path: str | Path | None = None, **kw) -> dict:
    """Plan + mix; writes `dst` and the plan (`<dst>.score.json` unless `plan_path`); returns the plan."""
    dst = Path(dst)
    p = plan(lines, total, **kw)
    mix(primary, p, dst, duck_db=duck_db, duck_source=duck_source)
    if plan_path:
        Path(plan_path).write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
        return p
    dst.with_suffix(".score.json").write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def remix(primary: str, plan_path: str, dst: str, duck_db: float = 11.0) -> None:
    """Mix again from an edited `.score.json` (change emotions' tracks, levels, effects, times)."""
    mix(primary, json.loads(Path(plan_path).read_text(encoding="utf-8")), Path(dst), duck_db=duck_db)


# ---------------------------------------------------------------------------------------------
# Command-line options shared by every tool
# ---------------------------------------------------------------------------------------------

MUSIC_SOURCES = ("internet", "folder", "ai", "none")


def add_sound_args(p) -> None:
    g = p.add_argument_group("sound design (optional)")
    g.add_argument("--sound-design", action="store_true", help="score the emotions: mood-matched music per act, hits, risers, whooshes, pauses")
    g.add_argument("--music-source", default="internet", choices=list(MUSIC_SOURCES),
                   help="internet: free music by mood (credited); folder: your music; ai: composed on this Mac; none: effects only")
    g.add_argument("--music-folder", help="your music folder (with --music-source folder)")
    g.add_argument("--licence", default="monetized", choices=["monetized", "personal"],
                   help="internet music: monetized = CC0/BY/BY-SA only; personal = also non-commercial licences")
    g.add_argument("--no-effects", action="store_true", help="music only, no sound effects")
    g.add_argument("--music-level", type=float, default=-2.0, help="music level in dB (0 louder, -6 quieter)")


def options(a) -> dict | None:
    """Sound-design options from parsed arguments, or None when it is off."""
    if not getattr(a, "sound_design", False):
        return None
    src = a.music_source
    if src == "folder" and not a.music_folder:
        raise AutocutError("choose your music folder (--music-folder)")
    return {"source": src if src != "none" else "internet", "music": src != "none", "folder": a.music_folder, "licence": a.licence,
            "effects": not a.no_effects, "music_db": a.music_level}


def score_narration(voice_wav: str, narration: dict, dst: str | Path, opts: dict, tail: float = 2.5) -> dict:
    """A spoken narration (voice.speak output) with its score: writes dst + dst.score.json."""
    total = float(narration.get("duration") or 0.0) + tail
    p = render(voice_wav, narration["segments"], total, dst, **opts)
    return {"output": str(dst), "plan": str(Path(dst).with_suffix(".score.json")), "credits": credits(p)}


def add_args(p) -> None:
    p.add_argument("audio", help="the voice (or any main sound) to score")
    p.add_argument("--narration", required=True, help=".narration.json (lines with emotions), or a .score.json to mix again")
    p.add_argument("--out", required=True)
    add_sound_args(p)


def run_args(a) -> dict:
    a.sound_design = True
    data = json.loads(Path(a.narration).read_text(encoding="utf-8"))
    if "acts" in data:  # an edited plan: mix again exactly as written
        mix(a.audio, data, Path(a.out))
        return {"outputs": [a.out], "out_dir": str(Path(a.out).parent)}
    res = score_narration(a.audio, data, a.out, options(a) or {})
    return {"outputs": [res["output"]], "out_dir": str(Path(a.out).parent), "plan": res["plan"], "credits": res["credits"]}


# ---------------------------------------------------------------------------------------------
# Emotions for lines that have none (your own recorded voiceover, talk clips)
# ---------------------------------------------------------------------------------------------

_CUES = [
    (r"\?$", "Curiosity", 0.6), (r"!$", "Excitement", 0.75),
    (r"\b(never|secret|hidden|nobody|mystery|unknown|why)\b", "Mystery", 0.6),
    (r"\b(wow|incredible|unbelievable|insane|crazy|amazing)\b", "Amazement", 0.8),
    (r"\b(won|win|victory|finally|record|best|greatest|champion)\b", "Triumph", 0.75),
    (r"\b(lost|died|death|miss|gone|sad|cry|tears)\b", "Sorrow", 0.6),
    (r"\b(remember|used to|back then|years ago|childhood|old days)\b", "Nostalgia", 0.55),
    (r"\b(danger|scary|afraid|fear|dark|terrifying|horror)\b", "Fear", 0.65),
    (r"\b(love|heart|together|beautiful)\b", "Love", 0.55),
    (r"\b(calm|peace|quiet|slow|relax|breathe)\b", "Calmness", 0.4),
    (r"\b(hope|dream|future|believe|someday)\b", "Hope", 0.6),
    (r"\b(angry|hate|furious|enough)\b", "Anger", 0.7),
    (r"\b(wait|but then|suddenly|until)\b", "Suspense", 0.6),
]


def guess_emotions(lines: list[dict]) -> None:
    """Rule-of-thumb emotions from the words (when no language model is at hand)."""
    import re

    for i, ln in enumerate(lines):
        t = str(ln.get("text", "")).strip().lower()
        hit = next(((e, x) for pat, e, x in _CUES if re.search(pat, t)), None)
        if hit:
            ln["emotion"], ln["intensity"] = hit
        else:
            ln["emotion"], ln["intensity"] = ("Curiosity", 0.6) if i == 0 else ("Neutral", 0.5)


def tag_emotions(lines: list[dict], how: str = "claude", model: str | None = None) -> str:
    """Give lines without a real emotion one (Claude reads them all at once; rules otherwise).
    Returns who chose them."""
    if any(em.resolve(ln.get("emotion")) != "Neutral" for ln in lines):
        return "script"
    if how == "claude" and lines:
        try:
            import script_writer as sw

            numbered = "\n".join(f"{i + 1}. {ln['text']}" for i, ln in enumerate(lines))
            prompt = ("For each line of this video narration, choose the one emotion the listener should feel, from: "
                      f"{em.prompt_list()}. Give an intensity 0-1 (0.3 gentle, 0.6 clear, 0.9 peak). Think of the whole arc.\n\n{numbered}\n\n"
                      'Reply with only JSON: {"lines": [{"n": 1, "emotion": "...", "intensity": 0.6}, ...]}')
            data = sw.extract_json(sw.ask_claude(prompt, model, timeout=240))
            for r in data.get("lines", []):
                k = int(r.get("n", 0)) - 1
                if 0 <= k < len(lines):
                    lines[k]["emotion"] = em.resolve(r.get("emotion"))
                    lines[k]["intensity"] = float(np.clip(float(r.get("intensity", 0.6)), 0, 1))
            log("emotions chosen by Claude: " + ", ".join(sorted({ln["emotion"] for ln in lines})))
            return "claude"
        except Exception as e:  # no Claude, no network, odd reply: the rules still give an arc
            log(f"Claude could not tag the emotions ({type(e).__name__}); using word cues")
    guess_emotions(lines)
    return "rules"
