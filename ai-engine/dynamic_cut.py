"""dynamic_cut: "Dynamic" pace for Auto Edit Clips to Music, variety between runs, and feedback.

Dynamic pace: the cuts follow what the audio and the footage do, not one shot length.

  The audio:
    - how busy it is: dense onsets (drums, fast notes) shorten the shots, sparse ones lengthen them,
      along with the energy (as before) and the sections, drops and hits;
    - phrases: the pauses between sung or spoken phrases (sentences in a voice-over) become cut
      points, so a cut lands where a line ends, also off the beat.
  The footage (per clip, cached):
    - scene changes inside a clip (trailers and edited clips have their own cuts): a shot never
      runs across one;
    - dialogue: sentences found by Whisper (small, once per clip): a shot keeps a sentence whole or
      leaves it out, never cuts it mid-word, and dialogue is never sped up;
    - silent footage (a drone, a gimbal): long, smooth moves are preferred for long, calm shots.
  Then the length of each shot is chosen with the footage: neighbouring music slots in a calm part
  are joined into one longer shot when the best footage for it is a whole sentence or one long
  smooth move; the busy parts stay short.

Variety: every run uses a new seed (a little randomness in close calls) and avoids the windows the
previous version of the same edit used, so running it again gives a different edit, not the same one.

Feedback (Edit & Retry ▸ "What should be better?"): Claude turns what you say ("too many car shots,
more faces, slower at the start, less zooming") into changes: faster or slower, footage to show
more of or less of (found with CLIP across all the clips), fewer or more effects, more variety. The
feedback is kept with the edit (`<out>.feedback.json`) and every later retry follows all of it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

import autocut as ac
import footage as fo
from autocut import AutocutError, log

SPEECH_MODEL = ac.DEFAULT_WHISPER  # whisper-small: quick, good enough for "is someone talking, and when"
VERSION = "dyn1"


# ---------------------------------------------------------------------------------------------
# The audio: busy or sparse, and its phrases
# ---------------------------------------------------------------------------------------------


def phrase_gaps(audio: str, start: float, length: float) -> list[float]:
    """Pauses between phrases (sung or spoken) in the edit's stretch of audio, relative times."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", audio, "-vn",
           "-af", "highpass=f=180,lowpass=f=3800,silencedetect=noise=-34dB:d=0.22", "-f", "null", "-"]
    try:
        err = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stderr
    except (OSError, subprocess.TimeoutExpired):
        return []
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end: (-?[\d.]+)", err)]
    gaps = []
    for a, b in zip(starts, ends):
        if 0.3 < b < length - 0.3 and b - a < 2.5:
            gaps.append(round((a + b) / 2, 3))
    # a track with no phrases (all instruments) or one that is mostly silence says nothing here
    return gaps if 0 < len(gaps) <= length / 1.2 else []


def pace_profile(w: dict, base: float = 1.0):
    """A pace multiplier over time: busy music (dense onsets) gets shorter shots, sparse music longer."""
    beats = np.asarray(w.get("beats") or [], dtype=np.float64)
    onset = np.asarray(w.get("onset") or [], dtype=np.float64)
    if len(beats) < 4 or len(onset) != len(beats):
        return lambda t: base
    strong = onset > np.median(onset)
    dens = np.array([strong[max(0, i - 4):i + 5].mean() for i in range(len(beats))])
    lo, hi = float(dens.min()), float(dens.max())
    rank = (dens - lo) / max(1e-6, hi - lo)

    def at(t: float) -> float:
        i = int(np.clip(np.searchsorted(beats, t), 0, len(beats) - 1))
        return base * float(1.35 - 0.6 * rank[i])

    return at


# ---------------------------------------------------------------------------------------------
# The footage: scene changes, dialogue, silent footage
# ---------------------------------------------------------------------------------------------


def scene_cuts(a: dict) -> list[float]:
    """Hard cuts inside a clip, from the frame-difference series (seconds)."""
    m = np.asarray(a.get("motion") or [], dtype=np.float32)
    if len(m) < 5:
        return []
    out = []
    for i in range(1, len(m)):
        local = np.median(m[max(0, i - 6):i + 6])
        if m[i] > 16 and m[i] > 3.5 * max(local, 1.0):
            t = i / fo.FPS
            if not out or t - out[-1] > 0.4:
                out.append(round(t, 2))
    return out


def _speech_worker(items: list[tuple[str, str]], model: str) -> dict:
    """Runs in its own process: Whisper once, every clip's sentences."""
    import mlx_whisper

    out = {}
    for path, wav in items:
        try:
            res = mlx_whisper.transcribe(wav, path_or_hf_repo=model, word_timestamps=False, verbose=None, condition_on_previous_text=False)
        except Exception:  # (one odd clip never stops the edit)
            out[path] = []
            continue
        segs = []
        for s in res.get("segments", []):
            text = str(s.get("text", "")).strip()
            if float(s.get("no_speech_prob", 0)) > 0.55 or len(text.split()) < 2 or float(s.get("compression_ratio", 1)) > 2.6:
                continue  # (music and game sound make Whisper imagine words)
            segs.append([round(float(s["start"]), 2), round(float(s["end"]), 2)])
        out[path] = segs
    return out


def clip_signals(clips: list[dict], speech: bool = True) -> None:
    """Adds to each clip: has_audio, cuts (scene changes), speech ([[start, end], ...]); cached."""
    todo = []
    tmp = Path(tempfile.mkdtemp(prefix="ei-dyn-"))
    for c in clips:
        info = ac.probe(c["path"])
        c["has_audio"] = bool(info.has_audio)
        c["cuts"] = scene_cuts(c["a"])
        c["speech"] = []
        if not (speech and info.has_audio):
            continue
        key = ac.CACHE_DIR / "speech" / f"{ac.file_key(c['path'], SPEECH_MODEL, VERSION)}.json"
        if key.is_file():
            try:
                c["speech"] = json.loads(key.read_text())
                continue
            except ValueError:
                pass
        wav = tmp / f"{len(todo)}.wav"
        r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", c["path"], "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                           capture_output=True)
        if r.returncode == 0 and wav.is_file():
            todo.append((c, key, str(wav)))
    if todo:
        log(f"listening for dialogue in {len(todo)} clip(s) (Whisper small, once per clip)")
        t0 = time.time()
        try:
            res = ac.run_isolated(_speech_worker, [(c["path"], w) for c, _, w in todo], SPEECH_MODEL)
        except Exception as e:
            log(f"could not listen for dialogue ({e.__class__.__name__}); cutting by picture and music only")
            res = {}
        for c, key, _ in todo:
            segs = res.get(c["path"], [])
            c["speech"] = segs
            key.parent.mkdir(parents=True, exist_ok=True)
            key.write_text(json.dumps(segs))
        log(f"dialogue found in {sum(1 for c, _, _ in todo if c['speech'])} of {len(todo)} clip(s) in {time.time() - t0:.0f}s")
    for f in tmp.glob("*.wav"):
        f.unlink(missing_ok=True)
    try:
        tmp.rmdir()
    except OSError:
        pass
    n_cuts = sum(len(c["cuts"]) for c in clips)
    if n_cuts:
        log(f"{n_cuts} scene change(s) inside the clips: no shot runs across one")


# ---------------------------------------------------------------------------------------------
# Scoring a window (added to the director's own score)
# ---------------------------------------------------------------------------------------------


def window_extra(slot: dict, ci: int, c: dict, st: float, src: float, speed: float, opts: dict) -> float:
    """Extra score for using clip `c` from `st` for `src` seconds of source in this slot."""
    end = st + src
    s = 0.0
    for t in c.get("cuts") or []:
        if st + 0.08 < t < end - 0.08:
            return -9.0  # a scene change in the middle of a shot reads as a mistake
    sp = c.get("speech") or []
    if sp and opts.get("dynamic"):
        inside = [seg for seg in sp if seg[0] < end and seg[1] > st]
        if inside:
            if speed != 1.0:
                s -= 1.2  # sped-up or slowed dialogue sounds and looks wrong
            whole = [seg for seg in inside if seg[0] >= st - 0.05 and seg[1] <= end + 0.05]
            cut = [seg for seg in inside if not (seg[0] >= st - 0.05 and seg[1] <= end + 0.05)]
            s += 0.55 * min(2, len(whole)) - 0.45 * len(cut)
    if opts.get("dynamic") and not c.get("has_audio", True):
        # silent footage (drone, gimbal): a long smooth move carries a long calm shot
        d = slot["frames"] / max(1.0, opts.get("fps", 30.0))
        if d > 2.2 and slot.get("energy", 0.5) < 0.5:
            shake = c["a"].get("shake") or []
            a, b = int(st * fo.FPS), max(int(st * fo.FPS) + 1, int(end * fo.FPS))
            sh = float(np.mean(shake[a:b])) if shake[a:b] else 1.0
            s += 0.3 if sh < 0.6 else 0.0
    # variety between runs: a little randomness in close calls, and not the windows used last time
    seed = opts.get("seed")
    if seed is not None:
        h = hashlib.blake2b(f"{seed}:{ci}:{int(st * 2)}".encode(), digest_size=4).digest()
        s += (int.from_bytes(h, "little") / 2**32 - 0.5) * 0.5
    for (p, a, b) in opts.get("avoid", ()):
        if p == c["path"] and a < end and b > st:
            s -= 0.8
            break
    fb = opts.get("fb")
    if fb is not None:
        s += fb(c["path"], st, end)
    return s


# ---------------------------------------------------------------------------------------------
# Choosing shot lengths with the footage: joining calm slots into one longer shot
# ---------------------------------------------------------------------------------------------


def merged(slots: list[dict], fps: float = 30.0) -> dict:
    first, last = slots[0], slots[-1]
    t0 = first["start_f"]
    frames = sum(s["frames"] for s in slots)
    energy = sum(s["energy"] * s["frames"] for s in slots) / max(1, frames)
    hits = []
    for s in slots:
        off = (s["start_f"] - t0)
        hits += [h + off / fps for h in s.get("hits", [])]
    m = {**first, "frames": frames, "energy": energy, "hits": hits, "last": bool(last.get("last")), "build": bool(last.get("build")), "joined": len(slots)}
    return m


def can_join(slots: list[dict], fps: float) -> bool:
    if len(slots) < 2:
        return True
    if any(s.get("drop") or s.get("section_start") or s.get("first") for s in slots[1:]):
        return False
    if len({s.get("section") for s in slots}) > 1:
        return False
    d = sum(s["frames"] for s in slots) / fps
    e = sum(s["energy"] * s["frames"] for s in slots) / max(1, sum(s["frames"] for s in slots))
    return d <= 6.5 and e < 0.62


def sequence_dynamic(dr, slots: list[dict], clips: list[dict], fps: float, opts: dict, beam: int = 12, max_join: int = 3) -> tuple[list[dict], list[dict]]:
    """Like the director's sequence, but each shot may cover 1–3 consecutive music slots. Returns
    (shots, the slots they fill)."""
    n = len(slots)
    cache: dict[tuple[int, int], list[dict]] = {}

    def cands(i: int, k: int) -> list[dict]:
        if (i, k) not in cache:
            sl = slots[i] if k == 1 else merged(slots[i:i + k], fps)
            cache[(i, k)] = dr.candidates(sl, clips, fps, i / max(1, n - 1), opts, k=None)
        return cache[(i, k)]

    # states[i]: best partial edits that end at slot i: (score, seq, slots_used, used_clips, embs)
    states: dict[int, list] = {0: [(0.0, [], [], frozenset(), [])]}
    for i in range(n):
        here = sorted(states.pop(i, []), key=lambda x: -x[0])[:beam]
        for score, seq, used_slots, used, embs in here:
            for k in range(1, max_join + 1):
                if i + k > n or not can_join(slots[i:i + k], fps):
                    break
                sl = slots[i] if k == 1 else merged(slots[i:i + k], fps)
                tried = 0
                for c in cands(i, k):
                    if tried >= dr.CANDIDATES_PER_SHOT:
                        break
                    pen = 0.0
                    if c["clip"] in used:
                        if len(used) < len(clips):
                            continue
                        last = max((q for q, x in enumerate(seq) if x["clip"] == c["clip"]), default=-99)
                        if len(seq) - last <= min(8, len(clips) - 1):
                            continue
                        pen -= 1.5
                    tried += 1
                    if seq:
                        pen += dr.pair_cost(seq[-1], c, clips, sl, embs)
                        pen += dr.recent_subject_penalty(seq + [c], c, clips)
                    # a joined shot counts for the slots it fills, a little less: it must be clearly better
                    gain = c["score"] * k * (1.0 if k == 1 else 0.9) + pen
                    e = dr.emb_at(clips[c["clip"]], c["in"] + c["src"] / 2)
                    states.setdefault(i + k, []).append((score + gain, seq + [c], used_slots + [sl], used | {c["clip"]}, embs + ([e] if e is not None else [])))
        if i + 1 not in states and i + 1 <= n and here:
            # (only when nothing fits: the best candidate again)
            score, seq, used_slots, used, embs = here[0]
            c = cands(i, 1)[0]
            states[i + 1] = [(score + c["score"] - 1.5, seq + [c], used_slots + [slots[i]], used, embs)]
    final = sorted(states.get(n, []), key=lambda x: -x[0])
    if not final:
        raise AutocutError("could not fill the edit with these clips")
    best = final[0]
    joined = sum(1 for s in best[2] if s.get("joined", 1) > 1)
    if joined:
        log(f"dynamic: {joined} longer shot(s) where the footage carries them (a whole sentence, one long move)")
    return best[1], best[2]


# ---------------------------------------------------------------------------------------------
# Feedback: what did not come out right, turned into changes
# ---------------------------------------------------------------------------------------------


def feedback_path(out: str | Path) -> Path:
    o = Path(out)
    return o.with_name(o.stem + ".feedback.json")


def load_feedback(out: str | Path) -> list[dict]:
    try:
        v = json.loads(feedback_path(out).read_text(encoding="utf-8"))
        return v if isinstance(v, list) else []
    except (OSError, ValueError):
        return []


def add_feedback(out: str | Path, text: str) -> list[dict]:
    """Keeps the new feedback with the earlier ones (every retry follows all of it)."""
    items = load_feedback(out)
    t = re.sub(r"\s+", " ", text or "").strip()
    if t and (not items or items[-1].get("text") != t):
        items.append({"text": t, "time": time.strftime("%Y-%m-%d %H:%M")})
        feedback_path(out).write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    return items


def rules_feedback(text: str) -> dict:
    t = text.lower()
    out = {"pace": "same", "more": [], "less": [], "effects": "same", "variety": False}
    if re.search(r"\b(faster|quicker|more cuts|too slow|boring)\b", t):
        out["pace"] = "faster"
    elif re.search(r"\b(slower|calmer|too fast|too many cuts|hectic)\b", t):
        out["pace"] = "slower"
    if re.search(r"\b(less|fewer|no|too much)\b.*\b(effect|zoom|shake|flash)", t):
        out["effects"] = "less"
    if re.search(r"\b(same|repeat|repetitive|variety|different)\b", t):
        out["variety"] = True
    for m in re.finditer(r"more ([a-z ]{3,30}?)(?:[,.;]|$)", t):
        out["more"].append(m.group(1).strip())
    for m in re.finditer(r"(?:less|fewer|no|too many) ([a-z ]{3,30}?)(?:[,.;]|$)", t):
        w = m.group(1).strip()
        if not re.match(r"(effect|zoom|shake|flash|cuts)", w):
            out["less"].append(w)
    return out


def interpret(texts: list[str], writer: str = "claude") -> dict:
    """All the feedback so far, as changes: {pace, more, less, effects, variety}."""
    joined = " | ".join(texts)
    if not joined.strip():
        return {"pace": "same", "more": [], "less": [], "effects": "same", "variety": False}
    if writer == "claude":
        try:
            import script_writer as sw

            prompt = f"""A creator watched an automatic music edit of their clips and said what was wrong (oldest first, all still apply):
{joined}

Turn it into changes for the next edit. Footage descriptions must be short English phrases a vision model can match in a frame
("close-up of a face", "cars on a road", "aerial view of the sea", "text on screen"). Answer only JSON:
{{"pace": "faster|slower|same", "more": ["what to show more of"], "less": ["what to show less of"], "effects": "less|more|same", "variety": true}}"""
            got = sw.extract_json(sw.ask_claude(prompt, timeout=180))
            res = rules_feedback("")
            res.update({k: got[k] for k in res if k in got})
            res["more"] = [str(x)[:60] for x in res.get("more") or []][:4]
            res["less"] = [str(x)[:60] for x in res.get("less") or []][:4]
            if res["pace"] not in ("faster", "slower", "same"):
                res["pace"] = "same"
            if res["effects"] not in ("less", "more", "same"):
                res["effects"] = "same"
            return res
        except AutocutError as e:
            log(f"Claude could not read the feedback ({e}); using simple rules")
    return rules_feedback(joined)


def feedback_scorer(clips: list[dict], more: list[str], less: list[str]):
    """A score bonus for windows that show more of / less of what was asked (CLIP over every clip)."""
    if not more and not less:
        return None
    import voiceover_cut as vc

    paths = [c["path"] for c in clips]
    texts = list(more) + list(less)
    try:
        samples, sims = vc.clip_similarities(paths, texts)
    except Exception as e:
        log(f"could not match the feedback to the footage ({e.__class__.__name__})")
        return None
    if not samples:
        return None
    s = np.asarray(sims, dtype=np.float64)
    z = (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-6)
    val = z[: len(more)].sum(axis=0) - z[len(more):].sum(axis=0)
    per: dict[str, dict[int, float]] = {}
    for (ci, t), v in zip(samples, val):
        per.setdefault(paths[ci], {})[int(round(t / vc.SAMPLE_EVERY))] = float(v)

    def score(path: str, a: float, b: float) -> float:
        d = per.get(path)
        if not d:
            return 0.0
        ks = range(int(a / vc.SAMPLE_EVERY), int(b / vc.SAMPLE_EVERY) + 1)
        vals = [d[k] for k in ks if k in d]
        return 0.35 * float(np.mean(vals)) if vals else 0.0

    log("feedback: " + ", ".join([f"more {m}" for m in more] + [f"less {x}" for x in less]))
    return score


def previous_windows(plan_path: Path) -> list[tuple[str, float, float]]:
    """The windows the previous version of this edit used (avoided this time, for a different edit)."""
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    fps = float(plan.get("fps") or 30)
    out = []
    for sh in plan.get("shots", []):
        a = float(sh.get("in", 0))
        out.append((sh.get("clip", ""), a, a + sh.get("frames", 0) / fps * float(sh.get("speed", 1.0))))
    return out
