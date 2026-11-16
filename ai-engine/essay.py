#!/usr/bin/env python3
"""essay: a narrated video essay made from your own analysed recordings.

    python ei_engine.py essay "Jin broke his code to save others. Is Atsu breaking hers only to avenge the past?" \
        --plans ".../Ghost of Yotei - plans/plans.json" --minutes 6 --language hi --voice nxstory-a1

You give an idea (a question, a comparison, an argument) and the plans file(s) of Plan Videos.
Nothing is analysed again: what was found in each recording is read from next to it.

1. Plan (essay.md to read, essay.json to change):
   - outline: Claude reads the idea, what is known about the game, and every recording in rough
     steps, and lays out the argument in chapters, naming the stretches of the recordings that
     could carry each one;
   - script: it then reads those stretches closely (every 5 seconds, every line said with its
     time) and writes the essay as beats. A beat is either your voice saying one line over
     picked shots ("say" + "show"), or a line of the game played with its own sound, whole
     ("play"). What the idea claims but the recordings do not show is listed as a doubt.
   You can stop here (`--to plan`), read it, and change it with `--notes`.
2. Voice: the lines are spoken once in your cloned voice (or a macOS voice).
3. Video: each line gets its shots for exactly as long as it is spoken, the game's own sound
   stays quietly under your voice, and the game's lines play in the clear between yours. Wide,
   or delivered tall with the picture dimmed and blurred behind.

Built from units other tools can use: recordings.Library (the analysed recordings), pieces.render
(pieces of recordings + a voice -> a video), voice.speak, captions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import autocut as ac
import gameplay_plan as gp
from autocut import AutocutError, log

STEPS = ("plan", "video")
MAX_READ = 420.0  # seconds of one stretch read closely
MAX_READ_ALL = 4200.0  # seconds read closely for one essay
MAX_PLAY = 14.0  # seconds: the longest stretch of the game's own lines played in one go
HARD_PLAY = 20.0  # seconds: what a played stretch may grow to when its lines are made whole (the times of lines are rough)
LANG = {"en": "English", "hi": "Hindi, written in Devanagari (names of people, places and games may stay in Latin letters)",
        "hinglish": "Hinglish (Hindi in Latin letters, the way people text)"}


def idea_block(idea: str, minutes: float, language: str, footage: str) -> str:
    return f"""## The idea
{idea.strip()}

## The video
About {minutes:g} minutes. The narration is in {LANG.get(language, LANG['en'])}. The game's own lines stay as they were said.
Footage exists ONLY of: {footage}. Anything about another game, a film or real history is told by the voice, over footage
of this game chosen to fit the words; never present it as footage of something else."""


def outline_prompt(idea: str, minutes: float, language: str, lib, research: dict) -> str:
    known = "\n\n".join(f"### {g}\n{str(r.get('summary', ''))[:1500]}\nCharacters: {json.dumps(r.get('characters', []), ensure_ascii=False)[:1500]}" for g, r in research.items())
    return f"""You are planning a narrated video essay for a gaming creator, made from their OWN recorded playthrough.
{idea_block(idea, minutes, language, ", ".join(lib.games))}

## What is known about the game(s)
{known or "(nothing looked up)"}

## The recordings (R1, R2 …): every row is a stretch, with how much action there is, what is on screen, and what is said
{chr(10).join(lib.digests())}

Lay out the argument as 3-5 chapters after a hook. An essay argues: each chapter makes ONE point and proves it with what is seen and
said in the recordings. Take a position; say where the idea itself is weaker than it sounds.
For each chapter name the stretches of the recordings worth reading closely to write it: where the scene that proves its point is
(the confrontation, the line that states a character's code, the choice, the consequence) and good pictures for what the voice will
say (a calm ride, a duel, a face). A stretch is at most {MAX_READ / 60:.0f} minutes; all stretches together at most {MAX_READ_ALL / 60:.0f} minutes.
Prefer stretches with lines said in them for proof, and say what you expect to find there.
"doubts": what the idea claims that these recordings do not show or that the story contradicts, for the creator to decide.

Answer with only this JSON:
{{"title": "the video's title", "thesis": "the position the essay takes, one sentence", "hook": "what the first 15 seconds do",
 "chapters": [{{"title": "...", "point": "the one point it makes", "stretches": [{{"r": 3, "start": "0:12:30", "end": "0:16:00", "why": "what should be there"}}]}}],
 "doubts": ["..."]}}"""


def script_prompt(idea: str, minutes: float, language: str, lib, outline: dict, details: list[str]) -> str:
    words = int(max(40.0, minutes * 60 * 0.85 - 20) * 2.2)  # (the game's own lines take the rest of the length)
    return f"""Write the narrated video essay below as beats, from the creator's own recordings.
{idea_block(idea, minutes, language, ", ".join(lib.games))}

## The outline
Title: {outline.get('title', '')}
Thesis: {outline.get('thesis', '')}
Hook: {outline.get('hook', '')}
{chr(10).join(f"{k}. {c['title']}: {c['point']}" for k, c in enumerate(outline['chapters'], 1))}

## The stretches of the recordings, read closely (every 5 seconds: the time, the action level, what is on screen; and every line said, with its time)
{chr(10).join(details)}

Rules:
- A beat is ONE of:
  {{"say": "one spoken line of the narration", "emotion": "Neutral|Curiosity|Tension|Sadness|Awe|Determination|Calmness|Anger|Joy", "show": [{{"r": 3, "start": "0:12:34", "end": "0:12:40", "what": "what is seen"}}]}}
  {{"play": {{"r": 3, "start": "0:14:02", "end": "0:14:09", "line": "the line(s) said there, as given above", "who": "who says it, if known"}}}}
- "say": the creator's voice. Short spoken sentences (at most ~22 words), one thought each, in the narration's language. About {words} words in all.
  Its "show" is 1 or 2 shots from the stretches above that fit exactly what the line says: the person, place or action it names, at a
  time where that is on screen (use the rows: combat for a fight, cutscene/dialogue for a face, exploring/riding for a calm thought).
  A shot's start is where it should begin; it is shown for as long as the line takes to say. Never the same seconds twice in the video.
  Avoid menus, maps and loading.
- "play": the game speaks for itself. Use it where a line said in the game proves the point better than narration could: 4 to 8 times
  in the video, each 3-{MAX_PLAY:.0f} seconds, starting at a line's start and ending at a line's end as given above. Set it up with the "say"
  before it and answer it with the "say" after it. Only lines that are in the stretches above, at their times.
- The first beats are the hook: the sharpest contrast or question, then the strongest "play" line within the first 20 seconds.
- Every claim about what happens in the game must be something the stretches above show or say, or something in the outline's
  thesis that is plainly opinion. Do not invent plot. What you could not support goes into "doubts".
- End on the question to the viewer, after stating the essay's own answer.

Answer with only this JSON:
{{"title": "...", "chapters": [{{"title": "...", "beats": [...]}}], "doubts": ["..."],
 "publish": {{"title": "...", "description": "2-4 sentences", "tags": ["..."]}}}}"""


def revise_prompt(essay: dict, notes: str, details: list[str]) -> str:
    return f"""The creator read the plan of their narrated video essay and wants changes. Make exactly those changes and nothing else: a
chapter no note is about stays as it is. The rules of the beats stay the same ("say" with its "show" shots, or "play" a line of the game
that is in the stretches below, at its time).

## The creator's notes
{notes.strip()}

## The essay now
{json.dumps({k: essay.get(k) for k in ("title", "thesis", "chapters", "doubts", "publish")}, ensure_ascii=False, indent=1)}

## The stretches of the recordings it may use (every 5 seconds: time, action, what is on screen; every line said with its time)
{chr(10).join(details)}

Answer with only the whole essay again as JSON in the same form, with one more field: "changed": one short sentence per change you made."""


# ---------------------------------------------------------------------------------------------
# Cleaning what Claude answers
# ---------------------------------------------------------------------------------------------


def clean_outline(raw: dict, lib) -> dict:
    chapters, read = [], 0.0
    for c in (raw.get("chapters") if isinstance(raw, dict) else None) or []:
        if not isinstance(c, dict) or not str(c.get("title", "")).strip():
            continue
        stretches = []
        for s in c.get("stretches") or []:
            try:
                n = lib.get(s.get("r"))["n"]
                a, b = gp.parse_t(s.get("start")), gp.parse_t(s.get("end"))
            except (AutocutError, AttributeError, TypeError, ValueError):
                continue
            a = max(0.0, min(a, lib.duration(n) - 5))
            b = min(max(b, a + 20), a + MAX_READ, lib.duration(n))
            if b - a < 10 or read + (b - a) > MAX_READ_ALL:
                continue
            read += b - a
            stretches.append({"r": n, "start": round(a), "end": round(b), "why": " ".join(str(s.get("why", "")).split())[:200]})
        chapters.append({"title": str(c["title"]).strip()[:120], "point": " ".join(str(c.get("point", "")).split())[:300], "stretches": stretches})
    if not chapters or not any(c["stretches"] for c in chapters):
        raise AutocutError("the outline came back without chapters or without stretches of the recordings to read")
    return {"title": str(raw.get("title", "")).strip()[:140] or "Video essay", "thesis": " ".join(str(raw.get("thesis", "")).split())[:400],
            "hook": " ".join(str(raw.get("hook", "")).split())[:300], "chapters": chapters[:6],
            "doubts": [" ".join(str(d).split())[:500] for d in raw.get("doubts") or [] if str(d).strip()][:8]}


def _shot(s, lib) -> dict | None:
    try:
        n = lib.get(s.get("r"))["n"]
        a = gp.parse_t(s.get("start"))
    except (AutocutError, AttributeError, TypeError, ValueError):
        return None
    a = max(0.0, min(a, lib.duration(n) - 2))
    return {"r": n, "start": round(a, 2), "what": " ".join(str(s.get("what", "")).split())[:160]}


def clean_essay(raw: dict, lib) -> dict:
    """The essay as the next steps can rely on it: every shot inside a recording, every played
    stretch moved to whole lines of the game and short enough, no empty beats."""
    chapters = []
    for c in (raw.get("chapters") if isinstance(raw, dict) else None) or []:
        if not isinstance(c, dict):
            continue
        beats = []
        for b in c.get("beats") or []:
            if not isinstance(b, dict):
                continue
            if isinstance(b.get("play"), dict):
                p = b["play"]
                try:
                    n = lib.get(p.get("r"))["n"]
                    a, e = gp.parse_t(p.get("start")), gp.parse_t(p.get("end"))
                except (AutocutError, TypeError, ValueError):
                    continue
                e = min(max(e, a + 1.5), a + MAX_PLAY)
                said = lib.lines(n, a, e, mostly=True) or lib.lines(n, a, e)[:1]
                if said:  # (from the first line's start to the last line's end: never half a line)
                    a, e = max(0.0, said[0]["start"] - 0.25), said[-1]["end"] + 0.35
                    while e - a > HARD_PLAY and len(said) > 1:
                        said = said[:-1]
                        e = said[-1]["end"] + 0.35
                e = min(e, a + HARD_PLAY, lib.duration(n))
                if e - a < 1.0:
                    continue
                beats.append({"play": {"r": n, "start": round(a, 2), "end": round(e, 2), "line": " ".join(d["text"].strip() for d in said) or " ".join(str(p.get("line", "")).split())[:300],
                                       "who": str(p.get("who", "")).strip()[:60], "heard": bool(said)}})
            elif str(b.get("say", "")).strip():
                shots = [s for s in (_shot(s, lib) for s in (b.get("show") if isinstance(b.get("show"), list) else [b.get("show")]) if isinstance(s, dict)) if s][:2]
                beats.append({"say": " ".join(str(b["say"]).split())[:400], "emotion": str(b.get("emotion", "Neutral")).strip().title()[:20] or "Neutral", "show": shots})
        if beats:
            chapters.append({"title": str(c.get("title", "")).strip()[:120] or f"Part {len(chapters) + 1}", "beats": beats})
    if not any("say" in b for c in chapters for b in c["beats"]):
        raise AutocutError("the essay came back without anything to say")
    # a line without a shot of its own continues the picture of the one before it
    last = None
    for c in chapters:
        for b in c["beats"]:
            if "say" in b:
                if not b["show"] and last:
                    b["show"] = [{**last, "start": round(last["start"] + 6.0, 2), "what": "(the same scene goes on)"}]
                if b["show"]:
                    last = b["show"][-1]
    pub = raw.get("publish") if isinstance(raw.get("publish"), dict) else {}
    return {"title": str(raw.get("title", "")).strip()[:140], "chapters": chapters,
            "doubts": [" ".join(str(d).split())[:500] for d in raw.get("doubts") or [] if str(d).strip()][:10],
            "changed": [" ".join(str(d).split())[:200] for d in raw.get("changed") or [] if str(d).strip()][:12],
            "publish": {"title": str(pub.get("title", "")).strip()[:140], "description": str(pub.get("description", "")).strip()[:1500],
                        "tags": [str(t).strip()[:40] for t in pub.get("tags") or [] if str(t).strip()][:20]}}


def says(essay: dict) -> list[dict]:
    return [b for c in essay["chapters"] for b in c["beats"] if "say" in b]


def said_sig(essay: dict) -> str:
    return hashlib.sha1(json.dumps([(b["say"], b.get("emotion")) for b in says(essay)], ensure_ascii=False).encode()).hexdigest()[:16]


def essay_md(essay: dict, lib=None) -> str:
    n_say, n_play = len(says(essay)), sum(1 for c in essay["chapters"] for b in c["beats"] if "play" in b)
    words = sum(len(b["say"].split()) for b in says(essay))
    name = (lambda r: Path(lib.get(r)["path"]).name) if lib else (lambda r: f"R{r}")
    out = [f"# {essay.get('title', '')}", "", f"**Idea:** {essay.get('idea', '')}", f"**Position:** {essay.get('thesis', '')}", "",
           f"{n_say} spoken lines ({words} words, about {words / 2.3 / 60:.1f} min of voice) and {n_play} moments where the game speaks.", ""]
    for k, c in enumerate(essay["chapters"], 1):
        out += [f"## {k}. {c['title']}", ""]
        for b in c["beats"]:
            if "play" in b:
                p = b["play"]
                out += [f"> **THE GAME** ({name(p['r'])} {gp.hms(p['start'])}–{gp.hms(p['end'])}, {p['end'] - p['start']:.0f} s){' · ' + p['who'] if p.get('who') else ''}: “{p['line']}”"
                        + ("" if p.get("heard", True) else "  \n> *(no line was heard at this time in the analysis: check it)*"), ""]
            else:
                out += [f"**You:** {b['say']}  ", "*over:* " + ("; ".join(f"{name(s['r'])} {gp.hms(s['start'])} {s['what']}" for s in b["show"]) or "(no shot chosen)"), ""]
    if essay.get("doubts"):
        out += ["## For you to decide or check", ""] + [f"- {d}" for d in essay["doubts"]] + [""]
    if essay.get("changed"):
        out += ["## Changed after your last notes", ""] + [f"- {d}" for d in essay["changed"]] + [""]
    out += ["---", "To change something: run it again with notes (\"chapter 2: open on the duel, not the ride\"). The times and lines can also be changed in essay.json."]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------
# From the essay and the spoken voice to pieces of video
# ---------------------------------------------------------------------------------------------


def voice_script(essay: dict, language: str) -> dict:
    segs = []
    for c, ch in enumerate(essay["chapters"]):
        mine = [b for b in ch["beats"] if "say" in b]
        for k, b in enumerate(mine):
            segs.append({"text": b["say"], "emotion": b.get("emotion", "Neutral"), "intensity": 0.65, "pace": "normal", "emphasis": [],
                         "pause_after": 0.8 if k == len(mine) - 1 else 0.35, "section": "hook" if c == 0 and k == 0 else "core", "visual": "", "chapter": c + 1})
    return {"title": essay.get("title", ""), "language": language, "segments": segs}


def build(essay: dict, segments: list[dict], voice_seconds: float, lib, wav: str, *, game_db: float = -24.0, fps: float = 30.0) -> dict:
    """The video as pieces. `segments`: the spoken lines' times in the voice file, one per "say" in
    order. A spoken line is shown over its shots for exactly as long as it is said (its pause
    included), with the game's sound at `game_db`; a played stretch is shown whole with its own
    sound. Returns {"pieces", "voice", "cues", "chapters", "seconds"}."""
    mine = says(essay)
    if len(segments) != len(mine):
        raise AutocutError(f"the voice has {len(segments)} lines and the essay {len(mine)}: speak it again")
    under = 0.0 if game_db <= -60 else 10 ** (game_db / 20)
    pieces, voice, cues, chapters, t, k = [], [], [], [], 0.0, 0
    used: dict[int, list[tuple[float, float]]] = {}

    def free(n: int, a: float, d: float) -> float:
        """The start nearest after `a` whose `d` seconds were not shown before and fit in the recording."""
        for _ in range(40):
            hit = next((e for s, e in used.get(n, []) if a < e and a + d > s), None)
            if hit is None:
                break
            a = hit + 0.5
        return max(0.0, min(a, lib.duration(n) - d - 0.1))

    for ch in essay["chapters"]:
        chapters.append({"title": ch["title"], "start": round(t, 2)})
        for b in ch["beats"]:
            if "play" in b:
                p = b["play"]
                d = p["end"] - p["start"]
                pieces.append({"src": lib.get(p["r"])["path"], "start": p["start"], "seconds": d, "gain": 1.0, "match": True, "kind": "play", "r": p["r"]})
                used.setdefault(p["r"], []).append((p["start"], p["end"]))
                for ln in lib.lines(p["r"], p["start"], p["end"], mostly=True):
                    cues.append({"start": round(t + max(0.0, ln["start"] - p["start"]), 2), "end": round(t + min(d, ln["end"] - p["start"]), 2), "text": ln["text"].strip(), "game": True})
                t += d
                continue
            seg = segments[k]
            a = float(seg["start"])
            e = float(segments[k + 1]["start"]) if k + 1 < len(segments) else float(voice_seconds)
            d = max(0.4, e - a)
            voice.append((wav, a, d, t))
            said_for = max(0.3, min(d, float(seg.get("end", e)) - a))
            cues.append({"start": round(t, 2), "end": round(t + said_for, 2), "text": b["say"], "game": False})
            shots = b["show"] or [{"r": lib.items[0]["n"], "start": 0.0, "what": ""}]
            if len(shots) > 1 and d < 3.2:
                shots = shots[:1]  # (too short a line for two pictures)
            each = d / len(shots)
            for s in shots:
                st = free(s["r"], lib.lit(s["r"], s["start"], each), each)
                pieces.append({"src": lib.get(s["r"])["path"], "start": round(st, 2), "seconds": each, "gain": under, "kind": "say", "r": s["r"]})
                used.setdefault(s["r"], []).append((st, st + each))
            t += d
            k += 1
    return {"pieces": pieces, "voice": voice, "cues": cues, "chapters": chapters, "seconds": t}


def cue_words(cues: list[dict]) -> list[dict]:
    """Caption words from the cues, each line's words spread over its time by their length."""
    out = []
    for c in cues:
        ws = c["text"].split()
        if not ws:
            continue
        total = sum(len(w) + 1 for w in ws)
        t, span = c["start"], max(0.2, c["end"] - c["start"])
        for w in ws:
            d = span * (len(w) + 1) / total
            out.append({"word": " " + w, "start": t, "end": t + d * 0.92})
            t += d
    return out


def clock(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def publish_text(essay: dict, chapters: list[dict]) -> str:
    pub = essay.get("publish") or {}
    out = [pub.get("title") or essay.get("title", ""), "", pub.get("description", ""), "", "Chapters"]
    out += [f"{clock(c['start'])} {c['title']}" for c in chapters]
    if pub.get("tags"):
        out += ["", "Tags: " + ", ".join(pub["tags"])]
    return "\n".join(out).strip() + "\n"


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def details_for(outline: dict, lib) -> list[str]:
    return [lib.detail(s["r"], s["start"], s["end"]) + (f"\n(expected here: {s['why']})" if s.get("why") else "") for c in outline["chapters"] for s in c["stretches"]]


def make(idea: str, plans: list[str], out_dir: str | None = None, *, minutes: float = 6.0, language: str = "en", voice: str | None = None, to: str = "video",
         notes: str = "", fresh: bool = False, model: str | None = None, tall: bool = False, frame_opacity: float = 0.35, frame_blur: float = 0.5,
         captions: str = "voice", music: str | None = None, game_db: float = -24.0, takes: int = 2, speech_pace: str = "reference",
         resolution: str = "1080p") -> dict:
    """`to`: "plan" stops when essay.md is written; "video" goes on. `notes`: what to change in the
    plan that is there. `captions`: "voice" (your lines; the game usually shows its own subtitles),
    "all" (the game's lines too), "none". `game_db`: how loud the game is under your voice."""
    import script_writer as sw
    from recordings import Library

    if to not in STEPS:
        raise AutocutError(f"go as far as one of: {', '.join(STEPS)}")
    if language not in LANG:
        raise AutocutError(f"the language is one of {', '.join(LANG)}")
    if captions not in ("voice", "all", "none"):
        raise AutocutError("captions: voice, all or none")
    if not 1 <= minutes <= 40:
        raise AutocutError("the length is 1 to 40 minutes")
    if to == "video" and not voice:
        raise AutocutError("choose a voice for the video (a cloned voice, or a macOS voice like mac:Rishi), or go only as far as the plan")
    ac.require_tools()
    folder = Path(out_dir).expanduser().resolve() if out_dir else None
    ej = folder / "essay.json" if folder else None
    essay = None
    if ej and ej.is_file() and not fresh:
        try:
            essay = json.loads(ej.read_text(encoding="utf-8"))
        except ValueError:
            essay = None
    idea = (idea or "").strip() or str((essay or {}).get("idea", ""))
    if not idea:
        raise AutocutError("say what the essay is about (a question, a comparison, an argument)")
    if essay is not None and essay.get("idea", "").strip() != idea:
        essay = None  # (another idea in this folder: planned anew)
    plans = [p for p in plans if str(p).strip()] or list((essay or {}).get("plans") or [])
    if not plans:
        raise AutocutError("choose the plans file of your recordings (plans.json, made by Plan Videos)")
    lib = Library(plans)
    if folder is None:
        folder = Path(plans[0]).expanduser().resolve().parent / "essays" / (sw.safe(idea)[:50] or "essay")
        ej = folder / "essay.json"
        if ej.is_file() and not fresh:
            try:
                old = json.loads(ej.read_text(encoding="utf-8"))
                essay = old if old.get("idea", "").strip() == idea else None
            except ValueError:
                essay = None
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ac.progress_plan({"outline": 0.5, "script": 0.5} if to == "plan" else {"outline": 0.08, "script": 0.1, "voice": 0.45, "video": 0.37})
    log(f"{len(lib.items)} analysed recording(s) of {', '.join(lib.games)}, {lib.seconds / 3600:.1f} h; nothing is analysed again")
    with ac.stage("outline"):
        if essay is None:
            research = lib.research()
            log("Claude is laying out the argument and choosing the stretches of the recordings to read closely")
            outline = clean_outline(sw.extract_json(sw.ask_claude(outline_prompt(idea, minutes, language, lib, research), model, timeout=900)), lib)
            log(f"outline: {outline['title']} · {len(outline['chapters'])} chapters, {sum(s['end'] - s['start'] for c in outline['chapters'] for s in c['stretches']) / 60:.0f} min of recordings to read")
        else:
            outline = essay["outline"]
            log(f"using the plan made {essay.get('made', '')}" + (" with your notes" if notes.strip() else ""))
    with ac.stage("script"):
        if essay is None:
            log("Claude is reading those stretches closely and writing the essay (your lines, their shots, the game's lines)")
            essay = clean_essay(sw.extract_json(sw.ask_claude(script_prompt(idea, minutes, language, lib, outline, details_for(outline, lib)), model, timeout=1200)), lib)
            essay.update(idea=idea, plans=[str(Path(p).expanduser().resolve()) for p in plans], language=language, minutes=minutes, outline=outline,
                         thesis=outline["thesis"], title=essay["title"] or outline["title"], doubts=outline["doubts"] + [d for d in essay["doubts"] if d not in outline["doubts"]],
                         made=time.strftime("%Y-%m-%d %H:%M"))
        elif notes.strip():
            log(f"your notes: {notes.strip()[:200]}")
            new = clean_essay(sw.extract_json(sw.ask_claude(revise_prompt(essay, notes, details_for(outline, lib)), model, timeout=1200)), lib)
            for c in new["changed"]:
                log(f"changed: {c}")
            essay = {**essay, **{k: v for k, v in new.items() if v or k == "changed"}, "noted": (essay.get("noted") or []) + [{"when": time.strftime("%Y-%m-%d %H:%M"), "text": notes.strip()[:1000]}]}
        else:
            essay = {**essay, **{k: v for k, v in clean_essay(essay, lib).items() if k == "chapters"}}  # (changed by hand: made safe again)
        ej.write_text(json.dumps(essay, ensure_ascii=False, indent=1), encoding="utf-8")
        (folder / "essay.md").write_text(essay_md(essay, lib), encoding="utf-8")
    n_play = sum(1 for c in essay["chapters"] for b in c["beats"] if "play" in b)
    log(f"essay: {essay['title']} · {len(says(essay))} spoken lines, {n_play} moments of the game's own")
    for d in essay.get("doubts", [])[:6]:
        log(f"  to check: {d[:200]}")
    outputs = [str(folder / "essay.md"), str(ej)]
    result = {"outputs": outputs, "out_dir": str(folder), "title": essay["title"], "lines": len(says(essay)), "plays": n_play, "video": None}
    if to == "plan":
        log(f"the plan is ready to read: {folder / 'essay.md'}")
        return result

    import captions as cap
    import pieces as pc
    import voice as vo

    name = sw.safe(essay["title"])[:60] or "essay"
    wav, vpath = folder / "voice.wav", folder / "voice-script.json"
    with ac.stage("voice"):
        sig = said_sig(essay) + f"|{voice}|{takes}|{speech_pace}"
        nar = wav.with_suffix(".narration.json")
        if not (wav.is_file() and nar.is_file() and essay.get("voiced") == sig):
            vpath.write_text(json.dumps(voice_script(essay, language), ensure_ascii=False, indent=1), encoding="utf-8")
            vo.speak(voice, str(vpath), str(wav), takes=takes, pace=speech_pace)
            essay["voiced"] = sig
            ej.write_text(json.dumps(essay, ensure_ascii=False, indent=1), encoding="utf-8")
        else:
            log("reusing the voice (the lines and the voice did not change)")
        narration = json.loads(nar.read_text(encoding="utf-8"))
    with ac.stage("video"):
        W, H = ac.sized(1920, 1080, resolution)
        TW, TH = ac.sized(1080, 1920, resolution)
        b = build(essay, narration["segments"], float(narration.get("duration") or ac.probe(str(wav)).duration), lib, str(wav), game_db=game_db)
        shown = [c for c in b["cues"] if captions == "all" or (captions == "voice" and not c["game"])]
        words = cue_words(shown)
        ass_path = None
        if words:
            CW, CH = (TW, TH) if tall else (W, H)
            ass = cap.build_ass(words, None, CW, CH, b["seconds"], "pop" if tall else "clean", "off")
            if any(re.search(r"[ऀ-ॿ]", w["word"]) for w in words):
                ass = ass.replace("Avenir Next Heavy", "Kohinoor Devanagari").replace("Avenir Next Demi Bold", "Kohinoor Devanagari")
            ass_path = folder / "captions.ass"
            ass_path.write_text(ass, encoding="utf-8")
        out = folder / f"{name}.mp4"
        pc.render(b["pieces"], b["voice"], str(out), W=W, H=H, ass=str(ass_path) if ass_path else None, music=music,
                  tall_frame=frame_opacity if tall else None, frame_blur=frame_blur, tall_size=(TW, TH))
        if ass_path:
            ass_path.unlink(missing_ok=True)
        out.with_suffix(".srt").write_text(ac.to_srt([{"start": c["start"], "end": c["end"], "text": c["text"]} for c in b["cues"]]), encoding="utf-8")
        (folder / "publish.txt").write_text(publish_text(essay, b["chapters"]), encoding="utf-8")
        (folder / "timeline.json").write_text(json.dumps({"pieces": b["pieces"], "chapters": b["chapters"], "cues": b["cues"]}, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"done in {time.time() - t0:.0f}s: {out}")
    return {**result, "outputs": [str(out), str(out.with_suffix(".srt")), str(folder / "publish.txt")] + outputs, "video": str(out), "seconds": round(b["seconds"], 1)}


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("idea", nargs="?", default="", help="what the essay is about: a question, a comparison, an argument (your own words, or a pasted suggestion)")
    p.add_argument("--plans", nargs="*", default=[], help="the plans file(s) of Plan Videos (plans.json) whose recordings the essay is made from")
    p.add_argument("--out-dir", help="the essay's folder (default: essays/<idea> next to the first plans file); give it again to go on with it")
    p.add_argument("--minutes", type=float, default=6.0)
    p.add_argument("--language", default="en", choices=list(LANG))
    p.add_argument("--voice", help="cloned voice name, or mac / mac:<voice>")
    p.add_argument("--to", default="video", choices=list(STEPS), help="plan: stop when essay.md is written; video: go on")
    p.add_argument("--notes", default="", help="what to change in the plan that is there")
    p.add_argument("--fresh", action="store_true", help="plan it anew")
    p.add_argument("--model")
    p.add_argument("--tall", action="store_true", help="deliver it tall (9:16): the wide picture across the middle, the same picture dimmed and blurred behind")
    p.add_argument("--frame-opacity", type=float, default=35)
    p.add_argument("--frame-blur", type=float, default=50)
    p.add_argument("--captions", default="voice", choices=["voice", "all", "none"])
    p.add_argument("--music")
    p.add_argument("--game-sound", type=float, default=-24.0, help="how loud the game is under your voice, in dB (-60: silent)")
    p.add_argument("--takes", type=int, default=2)
    p.add_argument("--speech-pace", default="reference", choices=["reference", "relaxed", "slow"])
    p.add_argument("--resolution", default="1080p", choices=list(ac.RESOLUTIONS))


def run_args(a) -> dict:
    if not 0 <= a.frame_opacity <= 100 or not 0 <= a.frame_blur <= 100:
        raise AutocutError("the opacity and the blur of the picture behind are 0 to 100 percent")
    return make(a.idea, a.plans, a.out_dir, minutes=a.minutes, language=a.language, voice=a.voice, to=a.to, notes=a.notes, fresh=a.fresh, model=a.model,
                tall=a.tall, frame_opacity=a.frame_opacity / 100, frame_blur=a.frame_blur / 100, captions=a.captions, music=a.music, game_db=a.game_sound,
                takes=a.takes, speech_pace=a.speech_pace, resolution=a.resolution)


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
    raise SystemExit(main())
