#!/usr/bin/env python3
"""narrate: the voiceover editor. Cuts your footage to a narration the way an editor would.

Input: a voiceover (any audio/video, your recording or a cloned-voice narration from voice.py),
optionally the script it came from, and footage: many short clips, a few long ones, or both.

What decides which picture goes where
  1. The narration is split into lines (phrases, at breaths and sentence ends). Each line gets
     a picture brief: the script's "visual" note when it came from a script, else a short
     English shot description written by the language model from the words (works for Hindi),
     else the English translation of the line.
  2. The footage is understood: long files are split at their own cuts into shots; every second
     is described with CLIP (what is in frame, aerial/wide/medium/close), and rated for
     sharpness, exposure, shake and camera motion (cached, so a second edit is quick).
  3. For every line, candidate windows of exactly the right length are scored: how well they
     show what is being said (CLIP text–image match, averaged over the window), how good they
     look, whether the camera energy fits the delivery (excited lines get motion, calm lines get
     stillness), and for the first line, an establishing wide/aerial shot.
  4. The whole edit is then chosen together (beam search), not line by line: no window is used
     twice, the same clip is not used back to back, shot sizes vary (wide -> medium -> close),
     and footage keeps its real-life order when the narration tells a story in order.
  5. Timing: cuts land on line boundaries, a few frames before the words (picture leads sound,
     as editors do); long lines get a second shot at a breath; pauses are covered, never black.
     New sections get a dissolve; the edit fades in and out.
  6. Sound: the voice, a little of each clip's natural sound under it (so it feels real, not a
     slideshow), optional music ducked under the voice, loudness for social platforms.
     Captions (optional) use the script's exact words, Devanagari for Hindi.

    python narrate.py voice.wav --clips footage/ --out edit.mp4
    python narrate.py narration.wav --narration narration.narration.json --clips a.mp4 b.mp4 --aspect vertical --captions
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import autocut as ac
from autocut import AutocutError, log

ASPECTS = {"horizontal": (1920, 1080), "vertical": (1080, 1920), "square": (1080, 1080)}
PACE_MAX = {"slow": 6.5, "medium": 4.8, "fast": 3.2}  # longest single shot (s)
EMOTION_MOTION = {"excited": 0.8, "hook": 0.7, "cta": 0.5, "story": 0.5, "curious": 0.45, "suspense": 0.35, "serious": 0.3, "neutral": 0.4, "warm": 0.3, "calm": 0.15}


# ---------------------------------------------------------------------------------------------
# Narration lines
# ---------------------------------------------------------------------------------------------


def lines_from_words(words: list[dict], duration: float, min_len: float = 1.2, max_len: float = 7.0) -> list[dict]:
    """Phrases from a transcript: split at sentence ends, commas and breaths; tiny bits merged."""
    import clip_edit as ce

    ph = ce.phrases(words, max_len=max_len, pause=0.35)
    out: list[dict] = []
    for p in ph:
        if out and (p["end"] - p["start"] < min_len or out[-1]["end"] - out[-1]["start"] < min_len) and p["end"] - out[-1]["start"] <= max_len:
            out[-1]["end"] = p["end"]
            out[-1]["text"] += " " + p["text"]
            out[-1]["words"] += p["words"]
        else:
            out.append({"start": p["start"], "end": p["end"], "text": p["text"], "words": list(p["words"])})
    for l in out:
        l.setdefault("emotion", "neutral")
        l.setdefault("visual", "")
        l.setdefault("section", None)
    return out


def lines_from_narration(nar: dict) -> list[dict]:
    return [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"], "emotion": s.get("emotion", "neutral"),
             "visual": s.get("visual", ""), "section": s.get("section"), "words": []} for s in nar.get("segments", [])]


def align_script(lines: list[dict], script: dict) -> None:
    """Borrow emotion/visual/section from a script for lines of a recorded read (text overlap)."""
    import voice as vo

    segs = script.get("segments", [])
    if not segs:
        return
    for l in lines:
        best, best_s = None, 0.0
        lt = set(vo.norm_text(w) for w in l["text"].split())
        for s in segs:
            st = set(vo.norm_text(w) for w in s["text"].split())
            j = len(lt & st) / max(1, len(lt | st))
            if j > best_s:
                best, best_s = s, j
        if best and best_s >= 0.25:
            l["emotion"] = best.get("emotion", l["emotion"])
            l["visual"] = l["visual"] or best.get("visual", "")
            l["section"] = best.get("section")


def describe_lines(lines: list[dict], how: str, model: str | None) -> None:
    """Picture briefs (English) for lines without one."""
    todo = [l for l in lines if not l.get("visual")]
    if not todo or how == "none":
        return
    numbered = "\n".join(f"{i + 1}. {l['text']}" for i, l in enumerate(todo))
    prompt = ("These are consecutive lines of a video voiceover (they may be Hindi, English or Hinglish). For each line, write the ideal "
              "B-roll shot to show while it is spoken, in English, 4 to 12 words, concrete and visual (what is in frame, and the shot size: "
              "aerial, wide, medium or close-up). Keep the story's place and subject consistent between lines.\n"
              "Reply with only a JSON list of strings, one per line, in order.\n\n" + numbered)
    try:
        if how == "claude":
            import script_writer as sw

            raw = sw.ask_claude(prompt, None, timeout=300)
        else:
            import llm

            outs = llm.ask([prompt], model, max_tokens=60 * len(todo) + 50)
            raw = outs[0] if outs else ""
        m = re.search(r"\[.*\]", raw, re.S)
        arr = json.loads(m.group(0)) if m else []
    except Exception as e:  # briefs are a help, not a requirement
        log(f"no shot descriptions ({type(e).__name__}: {str(e)[:100]}); matching on the words")
        arr = []
    for l, v in zip(todo, arr):
        if isinstance(v, str) and v.strip():
            l["visual"] = v.strip()


def _translate_worker(wav: str, model: str) -> list[dict]:
    import mlx_whisper

    r = mlx_whisper.transcribe(wav, path_or_hf_repo=model, task="translate", condition_on_previous_text=False)
    return [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in r.get("segments", [])]


def english_gloss(voice: str, lines: list[dict], model: str) -> None:
    """Lines still without a brief get the English translation of what is said at that time."""
    todo = [l for l in lines if not l.get("visual")]
    if not todo:
        return
    if all(not re.search(r"[ऀ-ॿ]", l["text"]) for l in todo):
        for l in todo:
            l["visual"] = l["text"]
        return
    with tempfile.TemporaryDirectory(prefix="ei-tr-") as tmp:
        wav = str(Path(tmp) / "v.wav")
        ac.extract_audio(voice, wav)
        segs = ac.run_isolated(_translate_worker, wav, model)
    for l in todo:
        txt = " ".join(s["text"] for s in segs if s["end"] > l["start"] and s["start"] < l["end"])
        l["visual"] = txt.strip() or l["text"]


CAMERA_WORDS = re.compile(r"\b(drone|aerial|shot|shots|wide|medium|close-?up|closeup|tracking|tracks?|orbit(ing)?|reveal(ing)?|slow(ly)?|low|high|"
                          r"angle|pan(ning)?|tilt(ing)?|push(ing)?|pull(ing)?|dolly|handheld|cinematic|dramatic|footage|b-roll|view|camera|"
                          r"top-?down|overhead|establishing|static|steady|follow(ing)?|frame|framed|fpv|pov|flying|flies|fly|over|above|"
                          r"from|of|the|a|an|in|with|and|at|into|toward|towards|up|down)\b", re.I)


def subject_only(brief: str) -> str:
    """'Low drone tracking shot following a winding river up a green valley' -> 'winding river green valley'
    (camera words match every clip of a drone-only library equally; the subject is what differs)."""
    t = re.sub(r"[,.;:!?()]", " ", brief)
    t = CAMERA_WORDS.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip() or brief


def queries(line: dict) -> list[str]:
    q = [line.get("visual") or line["text"], subject_only(line.get("visual") or line["text"])]
    if line["text"] and not re.search(r"[\u0900-\u097F]", line["text"]):
        q.append(line["text"])
    return [x[:300] for x in q]


def line_embeddings(lines: list[dict]) -> np.ndarray:
    """One query vector per line: the brief, its subject alone, and the words, averaged."""
    qs = [queries(l) for l in lines]
    flat = [x for q in qs for x in q]
    emb = np.asarray(ac.run_isolated(_text_worker, flat), dtype=np.float32)
    out, k = [], 0
    weights = [1.0, 1.3, 0.6]
    for q in qs:
        v = sum(emb[k + j] * weights[j] for j in range(len(q)))
        out.append(v / (np.linalg.norm(v) + 1e-9))
        k += len(q)
    return np.asarray(out)


# ---------------------------------------------------------------------------------------------
# Footage
# ---------------------------------------------------------------------------------------------


def _text_worker(texts: list[str]) -> list[list[float]]:
    import open_clip
    import torch

    import voiceover_cut as vc

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(vc.CLIP_MODEL[0], pretrained=vc.CLIP_MODEL[1], device=device)
    model.eval()
    tok = open_clip.get_tokenizer(vc.CLIP_MODEL[0])
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            e = model.encode_text(tok(texts[i : i + 64]).to(device))
            out += (e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy().tolist()
    return out


def index_footage(paths: list[str]) -> list[dict]:
    """Per clip: duration, fps, created time, 1-Hz CLIP embeddings, shot size per second,
    quality and motion per second, and its own internal cuts (so long files become shots)."""
    import footage as fo

    clips = []
    for p in paths:
        info = ac.probe(p)
        if not info.has_video or info.duration < 0.8:
            log(f"skipping (no usable video): {Path(p).name}")
            continue
        clips.append({"path": p, "duration": info.duration, "fps": info.fps or 30.0, "audio": info.has_audio})
    if not clips:
        raise AutocutError("no usable video clips")
    t0 = time.time()
    log(f"analysing {len(clips)} clip(s) (motion, sharpness, cuts; cached afterwards)")
    for k, c in enumerate(clips, 1):
        a = fo.analyse(c["path"])
        q = fo.moment_quality(a)
        mot = np.asarray(a["motion"], dtype=np.float32)
        n = max(1, int(np.ceil(c["duration"])))
        per = lambda v: np.asarray([float(np.mean(v[int(i * fo.FPS) : int((i + 1) * fo.FPS)])) if int(i * fo.FPS) < len(v) else (float(v[-1]) if len(v) else 0.0) for i in range(n)])  # noqa: E731
        c["quality"] = per(q) if len(q) else np.full(n, 0.6)
        c["motion"] = per(mot) if len(mot) else np.zeros(n)
        med = float(np.median(mot)) if len(mot) else 1.0
        cuts = [i / fo.FPS for i in np.nonzero(mot > max(30.0, 6 * med))[0]] if len(mot) else []
        c["cuts"] = [t for t in cuts if 0.5 < t < c["duration"] - 0.5]
        c["created"] = a.get("created") or 0.0
        if k % 10 == 0:
            log(f"progress {k}/{len(clips)} clips analysed")
    imgs, text = fo.semantics([c["path"] for c in clips])
    for c in clips:
        e = imgs.get(c["path"])
        c["emb"] = e if e is not None and len(e) else np.zeros((1, 512), np.float32)
        d = fo.describe(c["emb"], text)
        c["size"] = d["size"] or ["wide"]
        c["look"] = np.asarray(d["look"] or [0.0])
    log(f"footage understood in {time.time() - t0:.0f}s")
    return clips


# ---------------------------------------------------------------------------------------------
# Choosing pictures
# ---------------------------------------------------------------------------------------------


def shots_for_lines(lines: list[dict], total: float, pace: str, lead: float = 0.12) -> list[dict]:
    """Timeline slots: one per line (two or three when a line is long), covering 0..total
    without gaps. Cuts sit `lead` s before the line starts (picture leads sound)."""
    mx = PACE_MAX.get(pace, 4.8)
    starts = [0.0] + [max(0.0, l["start"] - lead) for l in lines[1:]]
    ends = starts[1:] + [total]
    slots = []
    for i, (a, b, l) in enumerate(zip(starts, ends, lines)):
        n = max(1, int(np.ceil((b - a) / mx)))
        cuts = [a + (b - a) * k / n for k in range(1, n)]
        if n > 1 and l.get("words"):  # move each even split onto the nearest breath (within 1 s)
            breaths = [((w1["end"] + w2["start"]) / 2 - lead, w2["start"] - w1["end"]) for w1, w2 in zip(l["words"], l["words"][1:])]
            snapped = []
            for c in cuts:
                near = [(abs(t - c) - 0.5 * g, t) for t, g in breaths if abs(t - c) <= 1.0 and g >= 0.12]
                snapped.append(min(near)[1] if near else c)
            cuts = sorted(c for c in snapped if a + 1.0 < c < b - 1.0)
        pts = [a, *cuts, b]
        for k, (x, y) in enumerate(zip(pts, pts[1:])):
            slots.append({"start": x, "end": y, "line": i, "part": k, "of": len(pts) - 1})
    return [s for s in slots if s["end"] - s["start"] > 0.2]


def candidates(slot: dict, line: dict, q_emb: np.ndarray, clips: list[dict], first: bool, k: int = 30) -> list[dict]:
    """Best windows (clip, in) of the slot's length, with their unary score."""
    dur = slot["end"] - slot["start"]
    want_motion = EMOTION_MOTION.get(line.get("emotion", "neutral"), 0.4)
    out = []
    for ci, c in enumerate(clips):
        if c["duration"] < dur + 0.1:
            continue
        sims = c["emb"] @ q_emb  # per second
        mot_all = c["motion"]
        mot_n = np.clip(mot_all / max(1e-3, float(np.percentile(mot_all, 90)) if len(mot_all) else 1.0), 0, 1.5)
        step = 0.5 if c["duration"] < 120 else 1.0
        for a in np.arange(0.0, c["duration"] - dur - 0.05, step):
            b = a + dur
            if any(a + 0.15 < t < b - 0.15 for t in c["cuts"]):
                continue  # never across the clip's own cut
            i0, i1 = int(a), max(int(a) + 1, int(np.ceil(b)))
            sem = float(np.mean(sims[i0:i1])) if i1 <= len(sims) else float(np.mean(sims[i0:])) if i0 < len(sims) else 0.0
            qual = float(np.mean(c["quality"][i0:i1])) if i0 < len(c["quality"]) else 0.5
            look = float(np.mean(c["look"][i0:i1])) if i0 < len(c["look"]) else 0.0
            mot = float(np.mean(mot_n[i0:i1])) if i0 < len(mot_n) else 0.3
            size = c["size"][min(i0, len(c["size"]) - 1)]
            out.append({"clip": ci, "in": round(float(a), 2), "sem": sem, "qual": qual, "look": look, "mot": mot, "size": size})
    if not out:
        return []
    # meaning first: how much better than the footage's typical match for this line
    sems = np.asarray([x["sem"] for x in out])
    mu, sd = float(np.median(sems)), float(sems.std() + 1e-6)
    for x in out:
        s = 1.0 * (x["sem"] - mu) / sd + 0.5 * x["qual"] + 0.2 * float(np.clip(x["look"], -1, 1)) - 0.3 * abs(min(1.0, x["mot"]) - want_motion)
        if first and x["size"] in ("aerial", "wide"):
            s += 0.2  # establish the place first
        x["score"] = float(s)
    out.sort(key=lambda x: -x["score"])
    # keep variety in the shortlist: at most 4 windows per clip
    per: dict[int, int] = {}
    short = []
    for x in out:
        if per.get(x["clip"], 0) >= 4:
            continue
        per[x["clip"]] = per.get(x["clip"], 0) + 1
        short.append(x)
        if len(short) >= k:
            break
    return short


def choose(slots: list[dict], cands: list[list[dict]], clips: list[dict], story_order: bool, beam: int = 12) -> list[dict]:
    """The whole edit at once (beam search over slots)."""
    State = tuple  # (score, picks list, used {clip: [(a,b)]})
    beams: list[tuple[float, list[dict], dict]] = [(0.0, [], {})]
    for si, (slot, cs) in enumerate(zip(slots, cands)):
        dur = slot["end"] - slot["start"]
        nxt: list[tuple[float, list[dict], dict]] = []
        for score, picks, used in beams:
            prev = picks[-1] if picks else None
            for c in cs:
                spans = used.get(c["clip"], [])
                if any(not (c["in"] + dur + 0.5 <= a or c["in"] >= b + 0.5) for a, b in spans):
                    continue  # never show the same moment twice
                s = c["score"]
                if prev is not None:
                    if prev["clip"] == c["clip"]:
                        s -= 0.6 if slot["part"] == 0 else 0.2  # same clip again (softer inside one long line)
                    if prev["size"] == c["size"] and len(picks) >= 2 and picks[-2]["size"] == c["size"]:
                        s -= 0.2  # three of the same shot size in a row
                    if story_order and clips[c["clip"]]["created"] and clips[prev["clip"]]["created"]:
                        if clips[c["clip"]]["created"] + 1 < clips[prev["clip"]]["created"]:
                            s -= 0.12  # going back in time
                s -= 0.1 * len(spans)  # spread across the footage
                nu = dict(used)
                nu[c["clip"]] = spans + [(c["in"], c["in"] + dur)]
                nxt.append((score + s, picks + [c], nu))
        if not nxt:  # ran out of fresh footage: allow reuse rather than fail
            nxt = [(score + cs[0]["score"] - 1.0, picks + [cs[0]], used) for score, picks, used in beams[:1] if cs]
            if not nxt:
                raise AutocutError("not enough footage for the narration")
        nxt.sort(key=lambda x: -x[0])
        beams = nxt[:beam]
    return beams[0][1]


# ---------------------------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------------------------


def build_shots(slots: list[dict], picks: list[dict], lines: list[dict], clips: list[dict], fps: float, W: int, H: int) -> list[dict]:
    shots = []
    push = True
    for i, (s, p) in enumerate(zip(slots, picks)):
        frames = max(1, int(round(s["end"] * fps)) - int(round(s["start"] * fps)))
        c = clips[p["clip"]]
        mot = float(np.mean(c["motion"][int(p["in"]) : int(p["in"] + frames / fps) + 1])) if len(c["motion"]) else 0.0
        sh = {"clip": c["path"], "in": p["in"], "frames": frames, "speed": 1.0, "line": s["line"], "text": lines[s["line"]]["text"],
              "visual": lines[s["line"]].get("visual", ""), "match": round(p["sem"], 3), "size": p["size"]}
        if mot < float(np.percentile(c["motion"], 40) if len(c["motion"]) else 1.0) and frames / fps >= 2.0:
            sh["zoom"] = {"from": 1.0, "to": 1.06} if push else {"from": 1.06, "to": 1.0}  # still shots breathe
            push = not push
        if i > 0 and s["part"] == 0:
            prev_sec = lines[slots[i - 1]["line"]].get("section")
            cur_sec = lines[s["line"]].get("section")
            gap = lines[s["line"]]["start"] - lines[slots[i - 1]["line"]]["end"]
            if (cur_sec and prev_sec and cur_sec != prev_sec) or gap > 0.9:
                sh["transition_in"] = {"type": "dissolve", "frames": int(round(0.4 * fps))}
        shots.append(sh)
    if shots:
        shots[0]["fade_in"] = int(round(0.4 * fps))
        shots[-1]["fade_out"] = int(round(0.8 * fps))
    return shots


def caption_words(lines: list[dict]) -> list[dict]:
    """Word timings from each line's exact text (spread over the line by length)."""
    out = []
    for l in lines:
        if l.get("words"):
            out += [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in l["words"]]
            continue
        ws = l["text"].split()
        if not ws:
            continue
        total = sum(len(w) + 1 for w in ws)
        t = l["start"]
        span = l["end"] - l["start"]
        for w in ws:
            d = span * (len(w) + 1) / total
            out.append({"word": " " + w, "start": t, "end": t + d * 0.92})
            t += d
    return out


def build_sound(voice: str, shots: list[dict], clips: list[dict], fps: float, total: float, dst: Path, music: str | None,
                ambience_db: float, music_db: float) -> None:
    """Voice + natural sound of the shown clips (low) + optional music, ducked under the voice."""
    import music as mu

    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", voice]
    parts = [f"[0:a]aresample=48000,apad,atrim=0:{total:.4f},asetpts=PTS-STARTPTS[voice]"]
    mix = ["[voice]"]
    n = 1
    with_audio = {c["path"] for c in clips if c["audio"]}
    if ambience_db > -60 and shots:
        labels = []
        for k, s in enumerate(shots):
            d = s["frames"] / fps
            if s["clip"] in with_audio:
                cmd += ["-ss", f"{s['in']:.3f}", "-t", f"{d + 0.2:.3f}", "-i", s["clip"]]
                parts.append(f"[{n}:a]aresample=48000,asetpts=PTS-STARTPTS,atrim=0:{d:.4f},apad,atrim=0:{d:.4f},afade=t=in:d=0.08,afade=t=out:st={max(0, d - 0.08):.4f}:d=0.08[n{k}]")
                n += 1
            else:
                parts.append(f"anullsrc=r=48000:cl=mono,atrim=0:{d:.4f}[n{k}]")
            labels.append(f"[n{k}]")
        parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1,aformat=channel_layouts=mono,volume={10 ** (ambience_db / 20):.4f}[amb]")
        mix.append("[amb]")
    if music:
        m = mu.analyse(music)
        start = mu.choose_window(m, total)
        cmd += ["-ss", f"{start:.3f}", "-t", f"{total + 0.5:.3f}", "-i", music]
        parts.append(f"[{n}:a]aresample=48000,apad,atrim=0:{total:.4f},afade=t=in:d=1.0,afade=t=out:st={max(0, total - 2.5):.4f}:d=2.5,volume={10 ** (music_db / 20):.4f}[mus]")
        n += 1
        mix.append("[mus]")
    if len(mix) > 1:
        parts.append("[voice]asplit=2[v1][v2]")
        bed = mix[1:]
        if len(bed) > 1:
            parts.append(f"{''.join(bed)}amix=inputs={len(bed)}:normalize=0[bed]")
        else:
            parts.append(f"{bed[0]}anull[bed]")
        parts.append("[bed][v2]sidechaincompress=threshold=0.03:ratio=6:attack=15:release=450[duck]")
        parts.append("[v1][duck]amix=inputs=2:normalize=0:duration=first[mixed]")
        out = "[mixed]"
    else:
        out = "[voice]"
    parts.append(f"{out}loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000,atrim=0:{total:.4f}[ao]")
    graph = ";\n".join(parts)
    with tempfile.TemporaryDirectory(prefix="ei-nsnd-") as tmp:
        g = Path(tmp) / "g.txt"
        g.write_text(graph)
        ac.run(cmd + ac.filter_script_args(str(g)) + ["-map", "[ao]", "-c:a", "pcm_s16le", str(dst)])


def make(voice: str, clips_in: list[str], out: str, *, narration: str | None = None, script: str | None = None, aspect: str = "horizontal",
         pace: str = "medium", captions: bool = False, caption_style: str = "clean", music: str | None = None, music_db: float = -20.0,
         ambience_db: float = -24.0, describe: str = "local", model: str | None = None, story_order: bool = True, look: str = "natural",
         whisper_model: str = ac.DEFAULT_WHISPER, workers: int = 2, plan_only: bool = False) -> dict:
    import render
    import voiceover_cut as vc

    ac.require_tools()
    voice = str(Path(voice).resolve())
    vinfo = ac.probe(voice)
    if not vinfo.has_audio:
        raise AutocutError("the voiceover has no sound")
    if aspect not in ASPECTS:
        raise AutocutError(f"aspect must be one of {', '.join(ASPECTS)}")
    W, H = ASPECTS[aspect]
    fps = 30.0
    total = vinfo.duration
    t0 = time.time()
    sibling = Path(voice).with_suffix(".narration.json")
    if not narration and not script and sibling.is_file():
        narration = str(sibling)  # a cloned-voice narration from voice.py: exact lines, emotions, shot ideas
        log(f"using {sibling.name}")
    if narration:
        lines = lines_from_narration(json.loads(Path(narration).read_text(encoding="utf-8")))
        lang = next((s.get("lang") for s in json.loads(Path(narration).read_text(encoding="utf-8")).get("segments", []) if s.get("lang")), None)
    else:
        sc = json.loads(Path(script).read_text(encoding="utf-8")) if script and script.endswith(".json") else None
        prompt = " ".join(s["text"] for s in sc["segments"])[:800] if sc else (Path(script).read_text(encoding="utf-8")[:800] if script else None)
        tr = ac.transcribe(voice, model=whisper_model, prompt=prompt)
        lang = tr.get("language")
        lines = lines_from_words(tr["words"], total)
        if sc:
            align_script(lines, sc)
    if not lines:
        raise AutocutError("no speech found in the voiceover")
    log(f"{len(lines)} narration lines")
    describe_lines(lines, describe, model)
    english_gloss(voice, lines, whisper_model)
    clips = index_footage(vc.collect_clips(clips_in))
    q = line_embeddings(lines)
    slots = shots_for_lines(lines, total, pace)
    cands = [candidates(s, lines[s["line"]], q[s["line"]], clips, first=(i == 0)) for i, s in enumerate(slots)]
    picks = choose(slots, cands, clips, story_order)
    shots = build_shots(slots, picks, lines, clips, fps, W, H)
    outp = Path(out).resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    plan = {"version": 2, "mode": "narrate", "voice": voice, "width": W, "height": H, "fps": fps, "look": look, "lines": lines, "shots": shots,
            "music": music, "language": lang}
    plan_path = outp.with_suffix(".narrate.json")
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    used = len({s["clip"] for s in shots})
    log(f"planned {len(shots)} shots from {used} clip(s) for {total:.0f}s of narration in {time.time() - t0:.0f}s; "
        f"mean picture match {np.mean([s['match'] for s in shots]):.2f}")
    if plan_only:
        return {"outputs": [], "plan": str(plan_path), "out_dir": str(outp.parent)}
    with tempfile.TemporaryDirectory(prefix="ei-narrate-") as tmp:
        rplan = Path(tmp) / "plan.json"
        rplan.write_text(json.dumps({**plan, "lines": []}, default=float))
        video = Path(tmp) / "video.mp4"
        render.render(rplan, str(video), workers=workers)
        sound = Path(tmp) / "sound.wav"
        build_sound(voice, shots, clips, fps, sum(s["frames"] for s in shots) / fps, sound, str(Path(music).resolve()) if music else None, ambience_db, music_db)
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-i", str(sound)]
        words = caption_words(lines)
        if captions and words:
            import captions as cap

            ass = cap.build_ass(words, None, W, H, total, caption_style, "off")
            if any(re.search(r"[ऀ-ॿ]", w["word"]) for w in words):
                ass = ass.replace("Avenir Next Heavy", "Kohinoor Devanagari").replace("Avenir Next Demi Bold", "Kohinoor Devanagari")
            af = Path(tmp) / "c.ass"
            af.write_text(ass, encoding="utf-8")
            e = str(af).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
            cmd += ["-vf", f"subtitles=filename='{e}'", "-c:v", "h264_videotoolbox", "-b:v", "14M"]
        else:
            cmd += ["-c:v", "copy"]
        partial = outp.with_name(outp.stem + ".part.mp4")
        cmd += ["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(partial)]
        ac.run(cmd)
        partial.replace(outp)
    if words:
        outp.with_suffix(".srt").write_text(ac.to_srt(ac.group_words(words, max_words=7, max_chars=42)), encoding="utf-8")
    log(f"done: {outp.name} in {time.time() - t0:.0f}s")
    return {"outputs": [str(outp)], "plan": str(plan_path), "out_dir": str(outp.parent)}


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("voice", help="the voiceover (audio or video)")
    p.add_argument("--clips", nargs="+", required=True, help="footage files and/or folders")
    p.add_argument("--out", required=True)
    p.add_argument("--narration", help=".narration.json from voice.py (exact lines, emotions, visuals)")
    p.add_argument("--script", help=".script.json or .txt the voiceover was read from")
    p.add_argument("--aspect", default="horizontal", choices=list(ASPECTS))
    p.add_argument("--pace", default="medium", choices=list(PACE_MAX))
    p.add_argument("--captions", action="store_true")
    p.add_argument("--caption-style", default="clean", choices=["pop", "clean", "boxed"])
    p.add_argument("--music")
    p.add_argument("--music-db", type=float, default=-20.0)
    p.add_argument("--ambience-db", type=float, default=-24.0, help="natural sound of the clips under the voice (-60 = off)")
    p.add_argument("--describe", default="local", choices=["claude", "local", "none"], help="who writes picture briefs for lines")
    p.add_argument("--llm", default="default")
    p.add_argument("--no-story-order", action="store_true")
    p.add_argument("--look", default="natural")
    p.add_argument("--plan-only", action="store_true")


def run_args(a) -> dict:
    return make(a.voice, a.clips, a.out, narration=a.narration, script=a.script, aspect=a.aspect, pace=a.pace, captions=a.captions,
                caption_style=a.caption_style, music=a.music, music_db=a.music_db, ambience_db=a.ambience_db, describe=a.describe,
                model=a.llm, story_order=not a.no_story_order, look=a.look, workers=getattr(a, "workers", 2), plan_only=a.plan_only)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    try:
        print(json.dumps(run_args(ap.parse_args(argv)), ensure_ascii=False))
    except AutocutError as e:
        log(f"error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
