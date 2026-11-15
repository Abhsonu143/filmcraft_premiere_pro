"""plan_make: make the video of a plan from the gameplay planner (gameplay_plan.py).

    python ei_engine.py plan-make --plans ".../plans.json" --pick 6
    python ei_engine.py plan-make --plans ".../plans.json" --request "a short of the funniest death, in Hindi text"

A plan says which stretches of which recordings a video is made of, roughly (to the half minute).
Making it:

1. With `--request`, a new plan is written first: Claude reads what was asked, what is known about the
   game and what happens in every recording (the analysis kept next to each one), and plans exactly
   that video. It is added to `plans.json`.
2. The cuts: Claude reads each stretch of the plan in detail (every 5 seconds: the action, what is on
   screen, every line said with its time) and chooses the exact pieces to keep: a piece starts where a
   line or a scene starts and ends where one ends; menus, maps, loading, deaths and wandering are left
   out. No line is cut in the middle. The cuts are saved in the plan (`cuts`, editable: change the
   times in `plans.json` and make it again).
3. The video:
   - it opens with a hook: 2–3 seconds from inside it (the line or the moment that makes a viewer stay),
     then starts from its beginning. Optional (`--no-hook`), and saved in the plan as `hook_cut`.
   - Shorts and Reels: vertical (the middle of the picture over a blurred copy), the hook as text at
     the start, what is said as captions.
   - long videos: the pieces joined as they are (nothing is re-encoded, so hours take minutes; a piece
     starts on the recording's nearest keyframe before its time).
   Next to it: the captions (`.srt`), the edit for the timeline (`.otio`: every piece with its sound,
   so it can be changed by hand), and `publish.txt` (title, description with chapters, tags, thumbnail
   idea). The recordings are never changed.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

import autocut as ac
import gameplay_plan as gp
from autocut import AutocutError, log

VERTICAL = {"shorts", "reels"}
MAX_SECONDS = {"shorts": 59.0, "reels": 89.0}  # (a second under the platforms' limits: a file of exactly 60 s can be refused as a Short)
MIN_PIECE = 2.0  # seconds: nothing shorter is kept
HOOK = (1.5, 4.0)  # seconds: the shortest and longest teaser shown before the video starts
HOOK_LINE = 5.0  # a teaser that is a spoken line may be this long, so the line is never cut off
REACH = 60.0  # how far outside a stretch of the plan a cut may go (the plan's times are rough)


def load(plans_file: str) -> tuple[dict, Path]:
    p = Path(plans_file).expanduser()
    if p.is_dir():
        p = p / "plans.json"
    if not p.is_file():
        raise AutocutError(f"no plans file at {plans_file} (choose the plans.json made by Plan Videos)")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise AutocutError(f"{p.name} could not be read: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("plans"), list):
        raise AutocutError(f"{p.name} is not a plans file")
    return data, p


def pick_plan(plans: list[dict], pick: str) -> int:
    """Which plan: its number in the list (as in plans.md), or a part of its title."""
    pick = str(pick or "").strip()
    if not pick:
        raise AutocutError("say which plan: its number (as in plans.md) or a part of its title; or ask for a new video")
    if pick.isdigit():
        if not 1 <= int(pick) <= len(plans):
            raise AutocutError(f"there are {len(plans)} plans: {pick} is not one of them")
        return int(pick) - 1
    found = [k for k, p in enumerate(plans) if pick.lower() in str(p.get("title", "")).lower()]
    if not found:
        raise AutocutError(f"no plan has “{pick}” in its title")
    return found[0]


def analysis_of(recording: str) -> dict | None:
    """What the planner found in a recording (kept next to it); None when it is not there."""
    f = gp.sidecars(Path(recording))[0]
    if not f.is_file():
        return None
    try:
        a = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return a if isinstance(a, dict) else None


def detail(a: dict, start: float, end: float, step: int = 5) -> str:
    """A stretch of a recording as lines to read: every `step` seconds the action and what is on screen,
    and every line said, with its time."""
    action, _ = gp.per_second(a)
    said = [d for d in a.get("dialogue", []) if start <= d["start"] < end]
    cuts = [k["t"] for k in a.get("keyframes", []) if k.get("cut") and start <= k["t"] < end]
    lines = []
    for s in range(int(start // step * step), int(end), step):
        e = s + step
        act = float(action[max(0, s) : e].mean()) if len(action) > max(0, s) else 0.0
        tg = [t["top"][0] for t in a.get("tags", []) if s <= t["t"] < e and t.get("top")]
        row = f"{gp.hms(s)} act {act:.1f} [{','.join(sorted(set(tg), key=tg.count, reverse=True)[:2])}]"
        if any(s <= c < e for c in cuts):
            row += " scene-cut"
        lines.append(row)
        for d in said:
            if s <= d["start"] < e:
                lines.append(f"    {gp.hms(d['start'])}–{gp.hms(d['end'])} “{str(d['text']).strip()[:160]}”")
    return "\n".join(lines)


def cuts_prompt(plan: dict, game: str, blocks: list[str]) -> str:
    vertical = plan.get("platform") in VERTICAL
    limit = MAX_SECONDS.get(plan.get("platform"), 0)
    length = (f"- This is a short vertical video: the pieces together must be 15 to {limit - 7:.0f} seconds (a teaser of a few seconds is added in front). "
              "It must contain, complete, the line or moment its title and hook promise: the set-up and the payoff. When a long stretch with nothing said "
              "lies between them (a fight, riding), keep only its best seconds: make two or three pieces. Start on a strong second, not on a lead-in.\n"
              if vertical else
              f"- This is a long video (planned: {plan.get('length')}). Keep the story whole and in order; take out only what a viewer would skip. Aim for the planned length, give or take 15%.\n")
    return f"""You are the editor of a gaming video made from a creator's own recordings of "{game}".
The plan below says roughly which stretches to use. Read each stretch in detail and choose the exact pieces to keep.

## The plan
Title: {plan.get("title")}
For: {plan.get("platform")} · {plan.get("format")} · {plan.get("length")}
Hook: {plan.get("hook")}
Why it works: {plan.get("why")}
{('What the creator asked for: ' + str(plan.get("requested")) + chr(10)) if plan.get("requested") else ''}{corrections(plan)}
## The stretches (per 5 seconds: action 0-1, what is on screen, scene cuts; indented: every line said, with its time)
{chr(10).join(blocks)}

## Rules
{length}- A piece starts just before a line is said or on a scene cut, and ends just after a line ends. Never cut a line or an exchange in the middle.
- Leave out: menus, maps, skill and inventory screens, loading, deaths and repeated failed attempts, long riding or walking with nothing said
  and nothing happening. Keep: story dialogue, reveals, the build-up and the end of a fight, anything funny or beautiful.
- Use only times inside the stretches shown. Pieces of one stretch are in order and do not overlap. You may drop a whole stretch if it adds nothing.
- note: a few words on what happens in the piece (it becomes a chapter title in a long video).
- hook: the 2 to 3 seconds from inside your pieces that would stop someone scrolling: the line that shocks, the hit that ends the fight,
  the reveal. It is shown first as a teaser, then the video starts from its beginning, so it must not be the first seconds of the first piece.
Answer with only this JSON:
{{"cuts": [{{"part": 1, "start": "H:MM:SS", "end": "H:MM:SS", "note": "..."}}], "hook": {{"part": 1, "start": "H:MM:SS", "end": "H:MM:SS"}}}}"""


def corrections(plan: dict) -> str:
    """What the creator found wrong in earlier makes of this plan, for the editor to keep to."""
    notes = [str(x.get("note", "")).strip() for x in plan.get("fixes") or [] if isinstance(x, dict) and str(x.get("note", "")).strip()]
    return ("The creator found these wrong in an earlier make (keep them right): " + " | ".join(notes[-6:]) + "\n") if notes else ""


def keep_lines_whole(start: float, end: float, a: dict | None) -> tuple[float, float]:
    """Move a cut's ends so no line is cut in the middle: the start to just before the line it falls
    in, the end to just after."""
    if not a:
        return start, end
    for d in a.get("dialogue", []):
        if d["start"] - 0.2 < start < d["end"]:
            start = max(0.0, d["start"] - 0.4)
        if d["start"] < end < d["end"] + 0.2:
            end = d["end"] + 0.5
    return start, min(end, float(a.get("duration") or end))


def clean_cuts(raw, plan: dict, analyses: list[dict | None]) -> list[dict]:
    """The cuts Claude chose, kept only where they make sense: inside (or near) their stretch of the
    plan, in order, not overlapping, long enough. A stretch without usable cuts is kept whole."""
    parts = plan.get("parts") or []
    by: dict[int, list[dict]] = {}
    for c in raw if isinstance(raw, list) else []:
        if not isinstance(c, dict):
            continue
        try:
            k = int(c.get("part", 0)) - 1
        except (TypeError, ValueError):
            continue
        if not 0 <= k < len(parts):
            continue
        s, e = gp.parse_t(c.get("start")), gp.parse_t(c.get("end"))
        p = parts[k]
        lo, hi = float(p["start"]) - REACH, float(p["end"]) + REACH
        s, e = max(s, lo, 0.0), min(e, hi)
        s, e = keep_lines_whole(s, e, analyses[k])
        if e - s < MIN_PIECE:
            continue
        by.setdefault(k, []).append({"recording": p["recording"], "part": k + 1, "start": round(s, 2), "end": round(e, 2), "note": str(c.get("note", ""))[:120]})
    out = []
    for k in sorted(by):
        last = -1.0
        for c in sorted(by[k], key=lambda c: c["start"]):
            if c["start"] < last:  # overlapping the piece before it: starts where that one ended
                c["start"] = last
            if c["end"] - c["start"] >= MIN_PIECE:
                out.append(c)
                last = c["end"]
    return out


def whole_parts(plan: dict, analyses: list[dict | None]) -> list[dict]:
    """The plan's stretches as they are (no line cut in the middle): the cuts when Claude is not there."""
    out = []
    for k, p in enumerate(plan.get("parts") or []):
        s, e = keep_lines_whole(float(p["start"]), float(p["end"]), analyses[k])
        if e - s >= MIN_PIECE:
            out.append({"recording": p["recording"], "part": k + 1, "start": round(s, 2), "end": round(e, 2), "note": str(p.get("note", ""))[:120]})
    return out


def fit_short(cuts: list[dict], limit: float) -> list[dict]:
    """A Short or Reel is never over its limit: the last pieces give way."""
    out, total = [], 0.0
    for c in cuts:
        room = limit - total
        if room < MIN_PIECE:
            break
        d = min(c["end"] - c["start"], room)
        out.append({**c, "end": round(c["start"] + d, 2)})
        total += d
    return out


def choose_cuts(plan: dict, game: str, analyses: list[dict | None], writer: str = "claude") -> list[dict]:
    parts = plan.get("parts") or []
    if writer != "claude" or not any(analyses):
        log("the plan's stretches are used whole" + ("" if any(analyses) else " (the recordings' analysis files were not found next to them)"))
        return whole_parts(plan, analyses)
    total = sum(float(p["end"]) - float(p["start"]) for p in parts)
    step = 5 if total <= 2400 else 10
    blocks = []
    for k, (p, a) in enumerate(zip(parts, analyses), 1):
        head = f"### Stretch {k}: {gp.hms(p['start'])}–{gp.hms(p['end'])} of {Path(str(p['recording'])).name}: {p.get('note', '')}"
        pad = 30.0 if plan.get("platform") in VERTICAL else 15.0
        blocks.append(head + "\n" + (detail(a, max(0.0, float(p["start"]) - pad), float(p["end"]) + pad, step) if a else "(no detail: keep it whole)"))
    import script_writer as sw

    log(f"Claude is choosing the exact cuts ({total / 60:.0f} min of recording in {len(parts)} stretch(es))")
    try:
        answer = sw.extract_json(sw.ask_claude(cuts_prompt(plan, game, blocks), timeout=900))
        raw = answer.get("cuts")
        if isinstance(answer.get("hook"), dict):
            plan["hook_idea"] = answer["hook"]
    except (AutocutError, ValueError) as e:
        log(f"the cuts could not be chosen ({e}): the plan's stretches are used whole")
        return whole_parts(plan, analyses)
    cuts = clean_cuts(raw, plan, analyses)
    if not cuts:
        log("no usable cuts came back: the plan's stretches are used whole")
        return whole_parts(plan, analyses)
    return cuts


def hook_cut(plan: dict, cuts: list[dict], analyses: dict[str, dict | None], on_keys: bool) -> dict | None:
    """The teaser shown before the video starts: 2–3 seconds from inside the pieces. Claude's choice when
    it is a good one; else the most striking line said (or, with nothing said, the peak of the action).
    Never the opening seconds of the first piece (it would only play twice). `on_keys`: it starts on a
    keyframe (a video joined without re-encoding)."""
    if not cuts or sum(c["end"] - c["start"] for c in cuts) < 15:
        return None
    lo, hi = HOOK
    first = cuts[0]

    def whole_said(a: dict | None, s: float, e: float) -> tuple[float, float, bool]:
        """`s`–`e` moved to whole lines: from the start of the first line said in it, through the lines
        after it that still fit. A single line that is too long stops after a word (at a comma or a
        full stop when there is one). The last value: whether anything is said in it."""
        said = [d for d in sentences((a or {}).get("dialogue", [])) if d["start"] < e and d["end"] > s]
        if not said:
            return s, e, False
        s2, e2 = float(said[0]["start"]) - 0.15, float(said[0]["end"]) + 0.25
        for d in said[1:]:
            if float(d["end"]) + 0.25 - s2 > HOOK_LINE:
                break
            e2 = float(d["end"]) + 0.25
        if e2 - s2 > HOOK_LINE:
            ws = [w for w in said[0].get("words") or [] if float(w[2]) - s2 <= HOOK_LINE - 0.2]
            stops = [w for w in ws if str(w[0]).rstrip()[-1:] in ",.;:!?…" and float(w[2]) - s2 >= lo]
            e2 = (float((stops or ws)[-1][2]) + 0.2) if ws else s2 + HOOK_LINE
            broken.append(not stops)
        return s2, e2, True

    broken: list[bool] = []  # (whether the last line looked at had to stop in the middle of a phrase)

    def usable(rec: str, s: float, e: float, said: bool = False) -> dict | None:
        asked = s
        inside = [c for c in cuts if c["recording"] == rec and c["start"] - 0.5 <= s and e <= c["end"] + 0.5]
        if not inside or (inside[0] is first and s < first["start"] + 4.0):
            return None
        s, e = max(s, inside[0]["start"]), min(e, inside[0]["end"])
        if on_keys:
            keys = [k["t"] for k in (analyses.get(rec) or {}).get("keyframes", []) if inside[0]["start"] <= k["t"] <= s + 0.02]
            if not keys:
                return None
            s = float(keys[-1])
        e = min(max(e, s + lo), asked + (HOOK_LINE if said else hi), inside[0]["end"])
        return {"recording": rec, "part": inside[0].get("part"), "start": round(s, 2), "end": round(e, 2), "note": ""} if e - s >= lo else None

    idea = plan.pop("hook_idea", None)
    parts = plan.get("parts") or []
    second, best_said = None, False
    if isinstance(idea, dict):
        try:
            k = int(idea.get("part", 0)) - 1
        except (TypeError, ValueError):
            k = -1
        s, e = gp.parse_t(idea.get("start")), gp.parse_t(idea.get("end"))
        if 0 <= k < len(parts) and s >= 0 and e > s:
            s, e, said = whole_said(analyses.get(str(parts[k]["recording"])), s, e)  # (whole lines, not half of one)
            got = usable(str(parts[k]["recording"]), s, e, said)
            if got and not (broken and broken[-1]):
                return got
            second = got  # (a line that must stop mid-phrase: only when no whole line is found below)
    best, best_score = None, -1.0
    for c in cuts:
        a = analyses.get(c["recording"])
        if not a:
            continue
        action, _ = gp.per_second(a)
        lines = [d for d in sentences(a.get("dialogue", [])) if c["start"] <= d["start"] and d["end"] <= c["end"] and 1.0 <= d["end"] - d["start"] <= HOOK_LINE - 0.4]
        for d in lines:  # a short line with force: a question, a shout, action under it
            text = str(d.get("text", ""))
            act = float(action[int(d["start"]) : int(d["end"]) + 1].mean()) if len(action) > int(d["start"]) else 0.0
            score = 1.0 + act + (0.5 if "?" in text or "!" in text else 0.0) + (0.3 if 3 <= len(text.split()) <= 9 else 0.0)
            got = usable(c["recording"], d["start"] - 0.15, d["end"] + 0.25, True) if score > best_score else None
            if got:
                best, best_score, best_said = got, score, True
        if not lines and len(action) > int(c["start"]) + 1:  # nothing said: the peak of the action
            seg = action[int(c["start"]) : int(c["end"])]
            peak = int(c["start"]) + int(np.argmax(seg))
            got = usable(c["recording"], peak - 1.0, peak + 1.5) if float(seg.max()) > best_score else None
            if got:
                best, best_score, best_said = got, float(seg.max()), False
    return best if best_said or not second else second


def fix_prompt(plan: dict, game: str, blocks: list[str], note: str, captions: list[str]) -> str:
    now = "\n".join(f"- stretch {c.get('part')}: {gp.hms(c['start'])}–{gp.hms(c['end'])} ({c['end'] - c['start']:.0f} s): {c.get('note', '')}" for c in plan.get("cuts") or [])
    hook = plan.get("hook_cut")
    limit = MAX_SECONDS.get(plan.get("platform"), 0)
    return f"""You are the editor of a gaming video made from a creator's own recordings of "{game}".
The video below was made and the creator found something wrong with it. Correct that and nothing else.

## The video
Title: {plan.get("title")}
For: {plan.get("platform")} · {plan.get("format")} · {plan.get("length")}
Hook: {plan.get("hook")}
Its pieces now:
{now or "(none)"}
Its teaser now (shown first): {(gp.hms(hook["start"]) + "–" + gp.hms(hook["end"])) if isinstance(hook, dict) else "none"}
Its captions now:
{chr(10).join(captions) or "(none)"}

## What the creator says is wrong
{note}

## The recording around it (per 5 seconds: action 0-1, what is on screen, scene cuts; indented: every line said, with its time)
{chr(10).join(blocks)}

## Rules
- Change only what the creator's words ask for. Whatever they do not mention stays exactly as it is.
- cuts: give ALL the pieces again only when a piece must change (a time, an added or removed piece); otherwise null.
  A piece starts just before a line is said and ends just after a line ends; never inside a line. Use only times inside the stretches shown.
{f"  The pieces together stay under {limit - 7:.0f} seconds." + chr(10) if limit else ""}- hook: a new teaser of 2 to 3 seconds from inside the pieces when the teaser is what is wrong; "none" when the creator wants no teaser; otherwise null.
- captions: when words of a caption are wrong, the wrong words exactly as they stand in the captions above, and the right words.
  Only what the creator asks for, or a name they tell you is misspelled. Otherwise an empty list.
- did: one short plain sentence on what you changed, or why nothing could be changed.
Answer with only this JSON:
{{"cuts": null, "hook": null, "captions": [{{"wrong": "...", "right": "..."}}], "did": "..."}}"""


def apply_fix(plan: dict, game: str, analyses: list[dict | None], note: str, captions: list[str]) -> str:
    """What the creator says is wrong, corrected in the plan by Claude: its cuts, its teaser, words of
    its captions. The note is kept in the plan (`fixes`), so later makes keep to it. → what was done."""
    import script_writer as sw

    parts = plan.get("parts") or []
    blocks = []
    for k, (p, a) in enumerate(zip(parts, analyses), 1):
        head = f"### Stretch {k}: {gp.hms(p['start'])}–{gp.hms(p['end'])} of {Path(str(p['recording'])).name}: {p.get('note', '')}"
        blocks.append(head + "\n" + (detail(a, max(0.0, float(p["start"]) - 30.0), float(p["end"]) + 30.0, 5) if a else "(no detail)"))
    log("Claude is correcting what you found wrong")
    answer = sw.extract_json(sw.ask_claude(fix_prompt(plan, game, blocks, note, captions), timeout=900))
    if not isinstance(answer, dict):
        raise AutocutError("the correction could not be read")
    done = []
    if isinstance(answer.get("cuts"), list):
        cuts = clean_cuts(answer["cuts"], plan, analyses)
        if cuts:
            plan["cuts"], plan["times"] = cuts, "heard"
            done.append(f"{len(cuts)} piece(s) set again")
    h = answer.get("hook")
    if isinstance(h, dict):
        plan["hook_idea"] = h
        plan.pop("hook_cut", None)
        plan.pop("no_hook", None)
        done.append("a new teaser")
    elif isinstance(h, str) and h.strip().lower() == "none":
        plan["no_hook"] = True
        plan.pop("hook_cut", None)
        done.append("no teaser")
    n = add_caption_fixes(plan, [(str(c.get("wrong", "")), str(c.get("right", ""))) for c in answer.get("captions") or [] if isinstance(c, dict)])
    if n:
        done.append(f"{n} caption correction(s)")
    did = str(answer.get("did", "")).strip()[:300]
    plan.setdefault("fixes", []).append({"note": note[:500], "did": did, "when": time.strftime("%Y-%m-%d %H:%M")})
    return f"{did} ({', '.join(done) if done else 'nothing was changed'})"


def _tokens(text: str) -> list[str]:
    return [t for t in (re.sub(r"[^\w]+", "", w.lower()) for w in str(text).split()) if t]


def add_caption_fixes(plan: dict, pairs: list[tuple[str, str]]) -> int:
    """Caption corrections kept in the plan (`caption_fixes`: wrong words → right words), so they hold
    in every later make. A correction of the same wrong words replaces the earlier one."""
    fixes = [x for x in plan.get("caption_fixes") or [] if isinstance(x, dict)]
    n = 0
    for wrong, right in pairs:
        wrong, right = " ".join(wrong.split())[:200], " ".join(right.split())[:200]
        if not _tokens(wrong) or wrong == right:
            continue
        fixes = [x for x in fixes if _tokens(x.get("wrong", "")) != _tokens(wrong)] + [{"wrong": wrong, "right": right}]
        n += 1
    if fixes:
        plan["caption_fixes"] = fixes[-200:]
    return n


def parse_caption_fixes(text: str) -> list[tuple[str, str]]:
    """'wrong => right; wrong2 => right2' (also -> or →; one per line or after a semicolon)."""
    out = []
    for item in re.split(r"[;\n]+", text or ""):
        m = re.split(r"\s*(?:=>|->|→)\s*", item.strip(), maxsplit=1)
        if len(m) == 2 and m[0].strip():
            out.append((m[0].strip(), m[1].strip()))
    return out


def fix_captions(words: list[dict], fixes: list[dict]) -> tuple[list[dict], list[str]]:
    """The caption words with the plan's corrections made: every place the wrong words stand (compared
    without capitals and punctuation) shows the right words over the same time. Right words left empty
    take the words out. → (the words, the corrections that matched nothing)"""
    out, unused = [dict(w) for w in words], []
    for fx in fixes or []:
        wrong, right = _tokens(fx.get("wrong", "")), str(fx.get("right", "")).split()
        if not wrong:
            continue
        k, hit = 0, False
        while k + len(wrong) <= len(out):
            if [t for w in out[k:k + len(wrong)] for t in _tokens(w["word"])] != wrong or any(len(_tokens(w["word"])) != 1 for w in out[k:k + len(wrong)]):
                k += 1
                continue
            old = out[k:k + len(wrong)]
            s, e = old[0]["start"], old[-1]["end"]
            if len(right) == len(old):  # (word for word: each keeps its own time)
                new = [{**o, "word": r} for o, r in zip(old, right)]
            else:
                new = [{**sp, **({"est": True} if old[0].get("est") else {})} for sp in spread(right, s, e)] if right else []
            tail = re.search(r"[.!?…,;:]+$", str(old[-1]["word"]))
            if new and tail and not re.search(r"[^\w\s]$", new[-1]["word"]):
                new[-1]["word"] += tail.group(0)  # (the sentence still ends where it ended)
            out[k:k + len(wrong)] = new
            k += max(len(new), 1) if new else 0
            hit = True
        if not hit:
            unused.append(str(fx.get("wrong", "")))
    return out, unused


def request_plan(data: dict, text: str, writer: str = "claude") -> dict:
    """A new plan for exactly what was asked, from what happens in the recordings."""
    if writer != "claude":
        raise AutocutError("a requested video needs Claude to plan it")
    recs = [str(r) for r in data.get("recordings") or []]
    analyses = [{**a, "source": r} for r, a in ((r, analysis_of(r)) for r in recs) if a]  # (where the recording is now)
    if not analyses:
        raise AutocutError("the recordings' analysis files (.ei-analysis.json) were not found next to the recordings: run Plan Videos on them first")
    if len(analyses) < len(recs):
        log(f"{len(recs) - len(analyses)} recording(s) have no analysis next to them and are not looked at")
    game = str(data.get("game") or "this game")
    research = gp.research_game(game, writer) if game != "this game" else {"visual": {}}
    low = text.lower()
    platforms = ["shorts"] if "short" in low else ["reels"] if "reel" in low else ["youtube", "shorts", "reels"]
    total = sum(float(a.get("duration") or 0) for a in analyses)
    step = int(np.clip(total / 700, 30, 300) // 30 * 30) or 30
    digests = [gp.digest(i + 1, a, step) for i, a in enumerate(analyses)]
    focus = (f"Plan exactly ONE video, this one: {text}\n"
             "It is a specific request, not a suggestion: find the stretches of the recordings that deliver it. If something asked for is not in the "
             "recordings, use the closest thing that is, and say so in \"why\".")
    import script_writer as sw

    log(f"Claude is planning the video you asked for from {len(analyses)} recording(s), {total / 3600:.1f} h")
    r = sw.extract_json(sw.ask_claude(gp.plan_prompt(game, research, digests, platforms, 1, focus, []), timeout=900))
    new = gp.clean_plans(r.get("plans", []), analyses, platforms)
    if not new:
        raise AutocutError("no plan could be made for that request from these recordings")
    plan = new[0]
    plan["made"] = time.strftime("%Y-%m-%d %H:%M")
    plan["requested"] = text[:500]
    return plan


def hook_text(plan: dict) -> str | None:
    """The words to show at the start: what the hook puts in quotes as on-screen text."""
    hook = str(plan.get("hook") or "")
    m = re.search(r"[\"“]([^\"”]{3,70})[\"”]", hook) if re.search(r"text|screen|caption", hook, re.I) else None
    return m.group(1).strip() if m else None


LISTEN_PIECE = 600.0  # seconds listened to at once
LISTEN_PAD = 4.0  # seconds listened to before and after what is asked for


def _listen_worker(wavs: list[str], model: str, lang: str | None, hint: str = "") -> list[list[dict]]:
    """Runs in its own process: each sound heard word by word. A line's times are those of its
    words (without word times Whisper places a line seconds away from where it is said when there
    is music or silence around it)."""
    import mlx_whisper

    out = []
    for wav in wavs:
        try:
            kw = {"language": lang} if lang else {}
            if hint:  # (the game's names, so they are spelled right)
                kw["initial_prompt"] = hint
            r = mlx_whisper.transcribe(wav, path_or_hf_repo=model, word_timestamps=True, verbose=None, condition_on_previous_text=False, **kw)
        except Exception:  # noqa: BLE001
            out.append([])
            continue
        lines = []
        for sg in r.get("segments", []):
            words = [[str(w.get("word", "")).strip(), float(w.get("start", 0)), float(w.get("end", 0))] for w in sg.get("words", []) or [] if str(w.get("word", "")).strip()]
            text = str(sg.get("text", "")).strip()
            empty = sum(1 for w in words if w[2] - w[1] < 0.02)
            if not words or not text or float(sg.get("no_speech_prob", 0)) > 0.6 or float(sg.get("compression_ratio", 1)) > 2.6 or empty * 2 >= len(words):
                continue  # (music and game noise make Whisper imagine words: they have no length)
            lines.append({"start": words[0][1], "end": words[-1][2], "text": text, "words": words})
        out.append(lines)
    return out


def merge_spans(spans: list[tuple[float, float]], gap: float = 1.0) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        elif b > a:
            out.append((a, b))
    return out


def hear(rec: str, pieces: list[tuple[float, float]], lang: str | None, hint: str = "") -> list[list[dict]] | None:
    """The lines said in each piece of the recording ({start, end, text, words} in the recording's
    time). Heard pieces are kept in the cache folder, so each is listened to once. None: it failed."""
    store = ac.CACHE_DIR / "heard"
    out: list[list[dict] | None] = [None] * len(pieces)
    todo = []
    for k, (s, e) in enumerate(pieces):
        cache = store / f"{ac.file_key(rec, ac.DEFAULT_WHISPER, f'{s:.1f}-{e:.1f}', lang or '', hint)}.json"
        try:
            out[k] = json.loads(cache.read_text())
        except (OSError, ValueError):
            todo.append((k, s, e, cache))
    if todo:
        try:
            with tempfile.TemporaryDirectory(prefix="ei-heard-") as tmp:
                wavs = []
                for k, s, e, _ in todo:
                    wavs.append(str(Path(tmp) / f"p{k}.wav"))
                    ac.extract_audio(rec, wavs[-1], start=s, end=e)
                got = ac.run_isolated(_listen_worker, wavs, ac.DEFAULT_WHISPER, lang, hint)
            store.mkdir(parents=True, exist_ok=True)
            for (k, s, e, cache), lines in zip(todo, got):
                out[k] = []
                for ln in lines:
                    words = [[w[0], round(max(w[1], w[2] - 0.6) + s, 2), round(w[2] + s, 2)] for w in ln["words"]]  # (no word takes seconds: a long one is its end)
                    out[k].append({"start": words[0][1], "end": words[-1][2], "text": ln["text"], "words": words})
                cache.write_text(json.dumps(out[k], ensure_ascii=False), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            log(f"could not listen again ({type(e).__name__}: {e})")
            return None
    return [x or [] for x in out]


def same_line(a: str, b: str) -> bool:
    import difflib

    norm = lambda t: re.sub(r"[^\w ]+", "", str(t).lower()).split()  # noqa: E731
    return difflib.SequenceMatcher(None, norm(a), norm(b), autojunk=False).ratio() >= 0.6


def language_of(a: dict | None) -> str | None:
    said = " ".join(str(d.get("text", "")) for d in (a or {}).get("dialogue", [])[:200])
    return "en" if said and sum(1 for ch in said if ord(ch) < 128) > 0.95 * len(said) else None


def spelled_as(heard: dict, known: str) -> dict:
    """The heard line with the planner's spelling (it knew the game's names), when both have the same
    number of words; else as heard."""
    ws = str(known).split()
    if len(ws) != len(heard.get("words") or []):
        return heard
    return {**heard, "text": str(known).strip(), "words": [[w, x[1], x[2]] for w, x in zip(ws, heard["words"])]}


SUB_FPS = 3  # pictures a second looked at for the game's own subtitles
SUB_BAND = 0.4  # the lower part of the picture they are looked for in


def _read_pictures(files: list[str], names: list[str]) -> list[list[tuple[str, float, float, float, float]]]:
    """The text in each picture: (text, left, right, top, bottom) in picture fractions, top to bottom.
    Apple Vision, careful reading, told the game's names."""
    import objc
    import Vision
    from Foundation import NSURL

    out = []
    for p in files:
        found = []
        with objc.autorelease_pool():
            h = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(str(p)), None)
            rq = Vision.VNRecognizeTextRequest.alloc().init()
            rq.setRecognitionLevel_(0)  # accurate
            rq.setUsesLanguageCorrection_(True)
            if names:
                rq.setCustomWords_(names)
            ok = h.performRequests_error_([rq], None)
            if ok[0] if isinstance(ok, tuple) else ok:
                for r in rq.results() or []:
                    c = r.topCandidates_(1)
                    if not c or float(c[0].confidence()) < 0.4:
                        continue
                    b = r.boundingBox()
                    found.append((str(c[0].string()).strip(), float(b.origin.x), float(b.origin.x + b.size.width), float(1 - b.origin.y - b.size.height), float(1 - b.origin.y)))
        out.append(sorted(found, key=lambda x: x[3]))
    return out


def rows_of(found: list[tuple[str, float, float, float, float]]) -> list[tuple[str, float, float]]:
    """What was read, row by row: pieces of text on the same height are one row, left to right (a
    subtitle is sometimes read as two pieces). (text, centre x, bottom)."""
    rows: list[list[tuple]] = []
    for box in sorted(found, key=lambda b: b[4]):
        if rows and abs(rows[-1][0][4] - box[4]) < 0.035 and all(box[1] >= o[2] - 0.01 or box[2] <= o[1] + 0.01 for o in rows[-1]):
            rows[-1].append(box)
        else:
            rows.append([box])
    out = []
    for row in rows:
        row.sort(key=lambda b: b[1])
        text = row[0][0]
        for prev, b in zip(row, row[1:]):  # (pieces that touch are one word read in two)
            text += ("" if b[1] - prev[2] < 0.008 else " ") + b[0]
        out.append((text, (row[0][1] + row[-1][2]) / 2, max(b[4] for b in row)))
    return out


def same_reading(a: str, b: str) -> bool:
    """Two readings of the same subtitle: nearly the same text, or one a part of the other, or the
    same ending (text over a moving background is read in pieces)."""
    import difflib

    x, y = a.lower(), b.lower()
    if difflib.SequenceMatcher(None, x, y).ratio() >= 0.8:
        return True
    short, long = sorted((x.rstrip(".…!? "), y.rstrip(".…!? ")), key=len)
    if len(short) >= 6 and short in long:
        return True
    n = 0
    while n < len(short) and short[-1 - n] == long[-1 - n]:
        n += 1
    return n >= 8


def subtitle_events(frames: list[list[tuple[str, float, float, float, float]]], start: float, fps: float = SUB_FPS) -> list[dict]:
    """The game's subtitles from what was read in the pictures: text in the middle of the picture's
    width that stays for a moment. {start, end, text, y (its bottom edge)}. Each line of text is
    followed on its own (two people's lines can be shown at once); a line wrapped over two rows is
    joined again. Sounds in brackets ("(Chuckles)"), a speaker's name before the line and button
    prompts in capitals are not what is said and are left out."""
    import difflib

    done: list[dict] = []
    live: list[dict] = []
    for k, found in enumerate(frames):
        t = start + k / fps
        for text, x, bottom in rows_of(found):
            text, named = re.subn(r"^[A-Z][\w' ]{1,18}:\s+", "", text.strip().lstrip("•·*-–— "))
            text = re.sub(r"\s*[\(\[][^\)\]]*[\)\]]\s*", " ", text).strip()
            text = re.sub(r"\bl(?=\s|'|’|$)", "I", text)  # (a lone small L is a capital i read wrong)
            if abs(x - 0.5) >= 0.08 or not re.search(r"[^\W\d_]{2}", text) or (text.upper() == text and len(text.split()) >= 2):
                continue
            for e in live:
                if e["last"] < t and any(same_reading(v, text) for v in set(e["seen"])):
                    e["last"] = t
                    e["seen"].append(text)
                    break
            else:
                live.append({"start": t, "last": t, "text": text, "seen": [text], "y": bottom, "named": bool(named)})
        for e in [e for e in live if t - e["last"] > 1.5 / fps]:  # (one unread picture is not the end of a line)
            live.remove(e)
            done.append(e)
    done += live
    events = []
    for e in sorted(done, key=lambda e: (e["start"], e["y"])):
        if len(e["seen"]) < 3 or e["last"] - e["start"] > 15:  # (a real subtitle stays about a second or more)
            continue
        top = max(set(e["seen"]), key=lambda x: (e["seen"].count(x), len(x)))
        ev = {"start": round(e["start"], 2), "end": round(e["last"] + 1 / fps, 2), "text": top, "y": round(e["y"], 3), **({"named": True} if e.get("named") else {})}
        if e["seen"].count(top) * 2 < len(e["seen"]):  # read differently from picture to picture: not to be trusted as it is
            ev["variants"] = list(dict.fromkeys(e["seen"]))[:8]
        prev = events[-1] if events else None
        if prev and abs(prev["start"] - ev["start"]) < 1.5 / fps and abs(prev["end"] - ev["end"]) < 1.5 / fps and ev["y"] > prev["y"]:
            prev["text"] += " " + ev["text"]  # (the second row of the same subtitle)
            prev["y"] = ev["y"]
            if ev.get("variants") or prev.get("variants"):
                prev["variants"] = [prev["text"]]
        elif prev and prev["text"].lower() == ev["text"].lower() and ev["start"] - prev["end"] <= 0.7:
            prev["end"] = max(prev["end"], ev["end"])
        else:
            events.append(ev)
    # a subtitle read while it was still appearing ("I will av") is the start of the full one: left out
    return [e for e in events if not any(o is not e and len(o["text"]) > len(e["text"]) and o["text"].lower().startswith(e["text"].lower().rstrip(".…- "))
                                         and abs(o["start"] - e["start"]) < 3 for o in events)]


def read_subtitles(rec: str, spans: list[tuple[float, float]], names: list[str]) -> list[dict]:
    """The game's own subtitles in `spans` of the recording (read once, then remembered)."""
    store = ac.CACHE_DIR / "subtitles"
    out = []
    for s, e in spans:
        cache = store / f"{ac.file_key(rec, 'subs8', f'{s:.1f}-{e:.1f}', ','.join(names))}.json"
        try:
            out += json.loads(cache.read_text())
            continue
        except (OSError, ValueError):
            pass
        try:
            with tempfile.TemporaryDirectory(prefix="ei-subs-") as tmp:
                ac.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{s:.3f}", "-t", f"{e - s:.3f}", "-i", rec, "-an",
                        "-vf", f"fps={SUB_FPS},crop=iw:ih*{SUB_BAND}:0:ih*{1 - SUB_BAND},scale=1280:-2", "-q:v", "3", str(Path(tmp) / "f%05d.jpg")])
                files = sorted(str(p) for p in Path(tmp).glob("f*.jpg"))
                events = subtitle_events(_read_pictures(files, names), s)
            store.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")
            out += events
        except Exception as ex:  # noqa: BLE001 (the heard lines are used as they are)
            log(f"the game's subtitles could not be read ({type(ex).__name__}: {ex})")
    return sorted(out, key=lambda d: d["start"])


def words_for(text: str, heard: list[list], start: float, end: float, early: float | None = None, late: float | None = None) -> list[list]:
    """The words of `text` with times: those of the heard words where they are the same word, the
    others placed between their neighbours (all inside `start`–`end`). `early`, `late`: how far
    words before the first heard one, and after the last, may reach (a line heard only in part)."""
    import difflib

    norm = lambda w: re.sub(r"[^\w]+", "", str(w).lower())  # noqa: E731
    ws = text.split()
    times: list[tuple[float, float] | None] = [None] * len(ws)
    got = [h for h in heard if norm(h[0])]
    if len(got) == len(ws):  # word for word (a misheard name takes the time of what was heard in its place)
        return [[w, round(float(h[1]), 2), round(max(float(h[2]), float(h[1]) + 0.05), 2)] for w, h in zip(ws, got)]
    for blk in difflib.SequenceMatcher(None, [norm(w) for w in ws], [norm(h[0]) for h in got], autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            times[blk.a + k] = (float(got[blk.b + k][1]), float(got[blk.b + k][2]))
    hit = [k for k, t in enumerate(times) if t]
    if len(hit) * 2 < len(ws):  # heard too differently: evenly over the line
        per = (end - start) / max(1, len(ws))
        return [[w, round(start + k * per, 2), round(start + (k + 1) * per, 2)] for k, w in enumerate(ws)]
    k = 0
    while k < len(ws):
        if times[k]:
            k += 1
            continue
        j = k
        while j < len(ws) and not times[j]:
            j += 1
        a = times[k - 1][1] if k else max(start if early is None else early, (times[j][0] if j < len(ws) else end) - 0.6 * (j - k))
        b = times[j][0] if j < len(ws) else min(end if late is None else late, a + 0.6 * (j - k))
        b = max(b, a)
        per = (b - a) / (j - k)
        for m in range(k, j):
            times[m] = (a + (m - k) * per, a + (m - k + 1) * per)
        k = j
    return [[w, round(t[0], 2), round(max(t[1], t[0] + 0.05), 2)] for w, t in zip(ws, times)]


def use_subtitles(a: dict | None, subs: list[dict], spans: list[tuple[float, float]]) -> dict | None:
    """The analysis with the game's own subtitles as what is said: they are spelled right (names) and
    complete, where listening mishears and misses. A heard line said while a subtitle is shown takes
    the subtitle's words (keeping the heard word times); a subtitle nothing was heard for is added,
    when it sits where the confirmed subtitles sit (not a button prompt or a place name)."""
    if not a or not subs:
        return a
    lines = [dict(d) for d in a.get("dialogue", [])]
    inside = lambda t: any(s <= t < e for s, e in spans)  # noqa: E731
    loose = lambda x, y: same_line(x, y) or _ratio(x, y) >= 0.45  # noqa: E731
    used: set[int] = set()
    placed, confirmed, lags = [], [], []
    for sub in subs:
        near = [k for k, d in enumerate(lines) if k not in used and (d.get("read") or inside(float(d["start"]))) and d["start"] < sub["end"] + 1.5 and d["end"] > sub["start"] - 1.5]
        same = sorted((k for k in near if loose(lines[k]["text"], sub["text"])), key=lambda k: -_ratio(lines[k]["text"], sub["text"]))[:1]  # (the one most like it)
        if not same:  # heard as something else, but said while this subtitle was shown (and not another subtitle's own line)
            same = [k for k in near if min(lines[k]["end"], sub["end"]) - max(lines[k]["start"], sub["start"]) > 0.5 * (lines[k]["end"] - lines[k]["start"])
                    and not any(o is not sub and loose(lines[k]["text"], o["text"]) for o in subs)]
            sure = False
        else:
            sure = True
        if same:
            used.update(same)
            heard = [w for k in same for w in lines[k].get("words") or []]
            s0, e0 = min(lines[k]["start"] for k in same), max(lines[k]["end"] for k in same)
            if not sure:  # (the subtitle's own time is the better one)
                placed.append({"sub": sub, "sure": True})
                continue
            ws = words_for(sub["text"], heard, s0, e0, early=min(s0, sub["start"] - 0.4), late=max(e0, sub["end"] - 0.4))  # (only a part heard: the rest by the subtitle)
            placed.append({"start": ws[0][1], "end": ws[-1][2], "text": sub["text"], "words": ws, "read": True, **({"variants": sub["variants"]} if sub.get("variants") else {})})
            if sure:
                confirmed.append(sub["y"])
                lags.append(sub["start"] - ws[0][1])
        else:
            placed.append({"sub": sub})
    out = [d for k, d in enumerate(lines) if k not in used]
    if len(confirmed) >= 3:  # the game subtitles what is said: a line only heard, with no subtitle, is noise taken for words
        only = [d for d in out if inside(float(d["start"])) and not d.get("read")]
        if only:
            log(f"{len(only)} heard line(s) have no subtitle in a game that subtitles its lines: left out ({'; '.join(str(d['text'])[:30] for d in only[:4])})")
            out = [d for d in out if not (inside(float(d["start"])) and not d.get("read"))]
    lag = float(np.clip(np.median(lags), 0.0, 1.0)) if lags else 0.4  # (a subtitle comes up a little after the line starts)
    many_sentences = sum(1 for x in subs if re.search(r"[.!?…]$", x["text"]) and len(x["text"].split()) >= 3) >= 3
    added = 0
    for p in placed:
        if "sub" not in p:
            out.append(p)
            continue
        sub = p["sub"]
        # a subtitle nothing was heard for is taken when it is surely one: it names its speaker, or sits where
        # confirmed subtitles sit, or (nothing confirmed at all) it is one of several full sentences read
        known = p.get("sure") or sub.get("named") or any(abs(sub["y"] - y) <= 0.06 for y in confirmed) or (not confirmed and many_sentences and re.search(r"[.!?…]$", sub["text"]))
        if not known or not inside(sub["start"]):
            continue
        s0 = max(0.0, sub["start"] - lag)
        ws = words_for(sub["text"], [], s0, min(sub["end"] - lag, s0 + 0.45 * len(sub["text"].split()) + 0.6))
        out.append({"start": ws[0][1], "end": ws[-1][2], "text": sub["text"], "words": ws, "read": True, "est": True, **({"variants": sub["variants"]} if sub.get("variants") else {})})
        added += 1
    n = sum(1 for p in placed if "sub" not in p)
    if n or added:
        log(f"the game's own subtitles: {n} heard line(s) take their wording, {added} line(s) not heard right are taken from them")
    return {**a, "dialogue": sorted(out, key=lambda d: d["start"])}


def _ratio(a: str, b: str) -> float:
    import difflib

    norm = lambda t: re.sub(r"[^\w ]+", "", str(t).lower())  # noqa: E731
    return difflib.SequenceMatcher(None, norm(a), norm(b), autojunk=False).ratio()


def proofread(a: dict | None, spans: list[tuple[float, float]], game: str, names: list[str]) -> dict | None:
    """Claude reads the lines that were only heard (no subtitle to go by) and corrects misheard names
    and words: it knows the game. A correction that changes a line too much is not taken."""
    if not a:
        return a
    todo = [(k, d) for k, d in enumerate(a.get("dialogue", [])) if any(s <= d["start"] < e for s, e in spans)]
    if not todo:
        return a
    import script_writer as sw

    prompt = f"""These lines were heard by speech recognition, or read from the on-screen subtitles, in a recording of the game "{game}".
Speech recognition mishears names and some words; reading from the screen makes small slips (a small L for a capital i, a lost quote mark or letter).
Names in the game: {", ".join(names)}
Correct each line: fix misheard names (to the game's real characters and places) and words that are clearly wrong for this game.
Never add, remove or reorder words, and never complete a sentence that stops: each line keeps its number of words (except a READINGS line). A line that is fine stays exactly as it is. Lines before and after are context for each other.
A line marked READINGS was read differently from picture to picture (the text sits over a moving background): give the one line all its readings come from.
{chr(10).join(f"{n + 1}. {d['text']}" + (f"   READINGS: {' | '.join(d['variants'])}" if d.get("variants") else "") for n, (_, d) in enumerate(todo))}
Answer with only this JSON: {{"lines": ["...", "..."]}} with exactly {len(todo)} lines in the same order."""
    try:
        fixed = sw.extract_json(sw.ask_claude(prompt, timeout=300)).get("lines")
    except Exception as e:  # noqa: BLE001 (the lines stay as heard)
        log(f"the heard lines could not be proofread ({type(e).__name__})")
        return a
    if not isinstance(fixed, list) or len(fixed) != len(todo):
        return a
    lines = list(a["dialogue"])
    changed = 0
    for (k, d), new in zip(todo, fixed):
        new = " ".join(str(new).split())
        same_count = len(new.split()) == len(str(d["text"]).split())
        if new and d.get("variants"):  # put together from its readings: every word must be in one of them
            norm = lambda w: re.sub(r"[^\w]+", "", str(w).lower())  # noqa: E731
            known = {norm(w) for v in d["variants"] for w in v.split()}
            if all(norm(w) in known or not norm(w) for w in new.split()):
                lines[k] = {**{x: y for x, y in d.items() if x != "variants"}, "text": new, "words": words_for(new, [], float(d["start"]), float(d["end"]))}
                changed += new != d["text"]
            continue  # (not put right: it stays as it was read most often)
        if new and new != d["text"] and (_ratio(new, d["text"]) >= 0.85 and same_count if d.get("read") else _ratio(new, d["text"]) >= 0.6):  # (a subtitle: a slip in a word, never a word more or less)
            ws = words_for(new, d.get("words") or [], float(d["start"]), float(d["end"]))
            lines[k] = {**d, "text": new, "words": ws}
            changed += 1
    if changed:
        log(f"{changed} line(s) corrected (names, misheard words, reading slips)")
    return {**a, "dialogue": [d for d in lines if d]}


def sentences(lines: list[dict]) -> list[dict]:
    """Lines joined into what is said in one breath: a line that does not end its sentence and is
    followed within 2 s by the next belongs with it (a game shows a long sentence as two subtitles).
    {start, end, text, words}."""
    out: list[dict] = []
    for d in sorted(lines, key=lambda d: d["start"]):
        prev = out[-1] if out else None
        text = str(d.get("text", "")).strip()
        goes_on = not re.search(r"[.!?…\"”'’]$", prev["text"].rstrip().rstrip("-–—")) if prev else False
        if prev and len(prev["text"].split()) < 40 and ((goes_on and float(d["start"]) - prev["end"] <= 2.0) or (text[:1].islower() and float(d["start"]) - prev["end"] <= 3.0)):
            prev["end"] = max(prev["end"], float(d["end"]))
            prev["text"] += " " + str(d.get("text", "")).strip()
            prev["words"] = (prev.get("words") or []) + (d.get("words") or [])
        else:
            out.append({"start": float(d["start"]), "end": float(d["end"]), "text": str(d.get("text", "")).strip(), "words": list(d.get("words") or [])})
    return out


def whole_at_edges(cuts: list[dict], analyses: dict[str, dict | None], limit: float) -> list[dict]:
    """No piece starts or ends in the middle of a sentence: the piece grows to take the whole of it when
    the video stays within `limit` seconds and the next piece is not reached, else it stops before
    the line (or starts after it). Done once what is said is finally known."""
    cuts = [dict(c) for c in cuts]
    for k, c in enumerate(cuts):
        lines = sentences((analyses.get(c["recording"]) or {}).get("dialogue", []))
        same = [o for o in cuts if o is not c and o["recording"] == c["recording"]]
        nxt = min((o["start"] for o in same if o["start"] >= c["end"]), default=float("inf"))
        prv = max((o["end"] for o in same if o["end"] <= c["start"]), default=0.0)
        for d in lines:
            ds, de = float(d["start"]), float(d["end"])
            total = sum(o["end"] - o["start"] for o in cuts)
            if c["start"] <= ds < c["end"] < de + 0.15:  # the line runs past the end
                if de + 0.3 <= nxt and total + de + 0.3 - c["end"] <= limit:
                    c["end"] = round(de + 0.3, 2)
                else:  # stop before it, but not inside a sentence that is being said over its start (two people talking)
                    over = [float(o["end"]) for o in lines if o["start"] < ds < o["end"] and c["start"] <= o["start"]]
                    c["end"] = round(max([ds - 0.1] + [x + 0.2 for x in over if x + 0.2 <= c["end"]]), 2)
            elif c["start"] <= ds < c["start"] + 1.5 and d["text"][:1].islower() and de < c["end"]:  # the end of a sentence whose start is not there
                c["start"] = round(de + 0.1, 2)
            elif ds - 0.1 < c["start"] < de <= c["end"]:  # the line began before the start
                if ds - 0.2 >= prv and total + c["start"] - (ds - 0.2) <= limit:
                    c["start"] = round(max(0.0, ds - 0.2), 2)
                else:
                    c["start"] = round(de + 0.1, 2)
        inside = [d for d in lines if c["start"] <= d["start"] and d["end"] <= c["end"] + 0.35]
        if len(inside) >= 2 and not re.search(r"[.!?…\"”'’\-–—]$", inside[-1]["text"].rstrip()) and inside[-1]["start"] - 0.1 - c["start"] >= MIN_PIECE:
            c["end"] = round(inside[-1]["start"] - 0.1, 2)  # the last sentence does not end (its rest is not known): the piece stops before it
    return [c for c in cuts if c["end"] - c["start"] >= MIN_PIECE]


MENU_TEXTS = 5  # this many pieces of text in the picture at once is a map, a menu or a list, not the game


def trim_menus(cuts: list[dict], analyses: dict[str, dict | None]) -> list[dict]:
    """Pieces without the seconds at their start or end that show a map or a menu: the picture is full
    of text there and nobody speaks. Only the edges are looked at (a second at a time, 6 s each)."""
    try:
        import thumbnails as th
    except Exception:  # noqa: BLE001
        return cuts
    out = []
    for c in cuts:
        c = dict(c)
        said = [(float(d["start"]), float(d["end"])) for d in (analyses.get(c["recording"]) or {}).get("dialogue", [])]
        n = int(min(6, (c["end"] - c["start"]) // 2))
        try:
            with tempfile.TemporaryDirectory(prefix="ei-menu-") as tmp:
                times = [c["start"] + k + 0.5 for k in range(n)] + [c["end"] - k - 0.5 for k in range(n)]
                files = []
                for k, t in enumerate(times):
                    p = Path(tmp) / f"m{k:02d}.jpg"
                    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(c["recording"]), "-frames:v", "1", "-vf", "scale=960:-2", "-q:v", "4", str(p)], capture_output=True)
                    files.append(p)
                if not all(p.is_file() for p in files):
                    raise OSError("no picture")
                full = [len(boxes) >= MENU_TEXTS and not any(s < t + 0.5 and e > t - 0.5 for s, e in said) for (boxes, _), t in zip(th.text_and_turn(files), times)]
        except Exception as e:  # noqa: BLE001 (not knowing is not a reason to stop)
            log(f"could not look for maps and menus ({type(e).__name__})")
            out.append(c)
            continue
        lead = next((k for k in range(n) if not full[k]), n)
        tail = next((k for k in range(n) if not full[n + k]), n)
        if lead or tail:
            log(f"a map or menu at the {'start' if lead else 'end'} of a piece is left out ({lead + tail} s)")
            c["start"], c["end"] = round(c["start"] + lead, 2), round(c["end"] - tail, 2)
        if c["end"] - c["start"] >= MIN_PIECE:
            out.append(c)
    return out


def listen_closer(rec: str, a: dict | None, spans: list[tuple[float, float]], hint: str = "") -> dict | None:
    """Once more, only the chosen pieces: what is heard in them and was missed before is added (the
    same sound heard from another starting point gives lines the first listening lost). Nothing is
    taken away."""
    if not a or not spans:
        return a
    spans = merge_spans([(max(0.0, s - 0.5), e + 0.5) for s, e in spans], gap=0.0)
    got = hear(rec, spans, language_of(a), hint)
    if not got:
        return a
    lines = list(a.get("dialogue", []))
    new = [ln for piece in got for ln in piece if not any(x["start"] < ln["end"] and ln["start"] < x["end"] for x in lines)]
    if new:
        log(f"{len(new)} more line(s) heard in the chosen pieces")
    return {**a, "dialogue": sorted(lines + new, key=lambda d: d["start"])}


def check_captions(video: Path, cues: list[dict], lang: str | None, hint: str = "") -> dict | None:
    """The finished video listened to: is each caption shown when its words are said? {lines, found,
    worst (seconds), late (how many are more than 0.6 s off)}; None when it could not be checked."""
    norm = lambda w: re.sub(r"[^\w]+", "", str(w).lower())  # noqa: E731
    try:
        with tempfile.TemporaryDirectory(prefix="ei-check-") as tmp:
            wav = str(Path(tmp) / "made.wav")
            ac.extract_audio(str(video), wav)
            heard = ac.run_isolated(_listen_worker, [wav], ac.DEFAULT_WHISPER, lang, hint)[0]
    except Exception as e:  # noqa: BLE001
        log(f"the finished video could not be checked ({type(e).__name__})")
        return None
    words = [(norm(w[0]), float(w[1])) for ln in heard for w in ln["words"] if norm(w[0])]
    offs, off_lines, moves = [], [], []
    for c in cues:
        ws = [x for x in (norm(w) for w in str(c.get("text", "")).split()) if x]
        near = [t - float(c["start"]) for k, (w, t) in enumerate(words)
                if ws and w == ws[0] and abs(t - float(c["start"])) < 4 and (len(ws) < 2 or (k + 1 < len(words) and words[k + 1][0] == ws[1]))]
        if near:
            offs.append(min(near, key=abs))
            moves.append((float(c["start"]), float(c["end"]), offs[-1]))
            if abs(offs[-1]) > 0.6:
                off_lines.append(f"{float(c['start']):.1f}s “{str(c.get('text', ''))[:30]}” {offs[-1]:+.1f}s")
    return {"lines": len(cues), "found": len(offs), "worst": round(max((abs(o) for o in offs), default=0.0), 2), "late": len(off_lines), "which": off_lines, "moves": moves}


def moved_to_the_sound(words: list[dict], moves: list[tuple[float, float, float]]) -> tuple[list[dict], int]:
    """Caption lines whose time was only estimated from the game's subtitle, moved to where the check
    heard them in the finished video (when that is more than 0.4 s away). Lines timed from their
    own heard words are left alone. (the words, how many lines moved)"""
    out, n = [dict(w) for w in words], 0
    for s, e, off in moves:
        mine = [w for w in out if s - 0.01 <= w["start"] < e + 0.01]
        if mine and abs(off) > 0.4 and all(w.get("est") for w in mine):
            for w in mine:
                w["start"], w["end"] = round(max(0.0, w["start"] + off), 3), round(max(0.0, w["end"] + off), 3)
            n += 1
    out.sort(key=lambda w: w["start"])
    for prev, w in zip(out, out[1:]):
        if w["start"] < prev["end"]:
            prev["end"] = w["start"]
    return out, n


def listen_again(rec: str, a: dict | None, spans: list[tuple[float, float]], hint: str = "") -> dict | None:
    """The recording's analysis with what is said in `spans` heard again, word by word: the planner's
    lines are right in what is said but can be many seconds off in when (it listened without word
    times). A planner's line that was not heard this time is looked for once more around its rough
    time; when it is still not heard it is left out (a caption at the wrong moment is worse than none)."""
    if not a:
        return a
    dur = float(a.get("duration") or 0)
    spans = merge_spans([(max(0.0, s - LISTEN_PAD), min(dur, e + LISTEN_PAD) if dur else e + LISTEN_PAD) for s, e in spans])
    pieces = [(s + k * LISTEN_PIECE, min(e, s + (k + 1) * LISTEN_PIECE)) for s, e in spans for k in range(int((e - s) // LISTEN_PIECE) + 1) if s + k * LISTEN_PIECE < e - 0.5]
    lang = language_of(a)
    log(f"the exact time of each line: {sum(e - s for s, e in pieces) / 60:.1f} min of {Path(rec).name} (listened to once, then remembered)")
    got = hear(rec, pieces, lang, hint)
    if got is None:
        log("the planner's rough line times are used")
        return a
    lines = sorted((ln for piece in got for ln in piece), key=lambda d: d["start"])
    inside = lambda t: any(s <= t < e for s, e in spans)  # noqa: E731
    rough = [d for d in a.get("dialogue", []) if inside(float(d["start"]))]
    lost = [d for d in rough if not any(same_line(d["text"], ln["text"]) and abs(ln["start"] - float(d["start"])) < 45 for ln in lines)]
    if lost:
        around = [(max(0.0, float(d["start"]) - 12.0), float(d["end"]) + 12.0) for d in lost]
        again = hear(rec, around, lang, hint) or []
        found = 0
        for d, piece in zip(lost, again):
            for ln in piece:
                if same_line(d["text"], ln["text"]) and not any(x["start"] < ln["end"] and ln["start"] < x["end"] for x in lines):
                    lines.append(ln)
                    found += 1
                    break
        lines.sort(key=lambda d: d["start"])
        log(f"{len(lost)} line(s) were not heard the first time: {found} found on a second listen, {len(lost) - found} left out")
    moved = []
    for k, ln in enumerate(lines):  # (the planner's spelling of names, where the same line was heard)
        known = [d for d in rough if same_line(d["text"], ln["text"]) and abs(ln["start"] - float(d["start"])) < 45]
        if known:
            best = min(known, key=lambda d: abs(ln["start"] - float(d["start"])))
            moved.append(abs(ln["start"] - float(best["start"])))
            lines[k] = spelled_as(ln, best["text"])
    if moved:
        log(f"lines were up to {max(moved):.0f} s from where the planner had them")
    return {**a, "dialogue": sorted([d for d in a.get("dialogue", []) if not inside(float(d["start"]))] + lines, key=lambda d: d["start"]), "heard": spans}


def said_lines(cuts: list[dict], analyses: dict[str, dict | None]) -> list[dict]:
    """The lines said in the pieces: {text, rec, start, end (in the recording), at (where the line
    starts in the video)}."""
    out, t = [], 0.0
    for c in cuts:
        a = analyses.get(c["recording"])
        for d in (a or {}).get("dialogue", []):
            s, e = max(float(d["start"]), c["start"]), min(float(d["end"]), c["end"])
            text = str(d.get("text", "")).strip()
            if e - s < 0.2 or not text.split() or not c["start"] - 0.5 <= d["start"] < c["end"]:
                continue
            out.append({"text": text, "rec": c["recording"], "start": s, "end": e, "at": t + s - c["start"], "lo": c["start"], "hi": c["end"], "words": d.get("words"), "est": bool(d.get("est"))})
        t += c["end"] - c["start"]
    return out


def spread(words: list[str], start: float, end: float) -> list[dict]:
    per = (end - start) / max(1, len(words))
    return [{"word": w, "start": round(start + k * per, 3), "end": round(start + (k + 1) * per, 3)} for k, w in enumerate(words)]


def said_words(cuts: list[dict], analyses: dict[str, dict | None]) -> list[dict]:
    """What is said in the pieces, word by word, in the video's own time (a line's words are spread
    evenly over it: rough, the planner times a line only to a second or two)."""
    return [w for ln in said_lines(cuts, analyses) for w in spread(ln["text"].split(), ln["at"], ln["at"] + ln["end"] - ln["start"])]


def heard_words(cuts: list[dict], analyses: dict[str, dict | None]) -> list[dict]:
    """What is said in the pieces, each word at its time in the video. A line that was listened to
    again (`listen_again`) has its words' real times; any other line's words are spread over it."""
    out = []
    for ln in said_lines(cuts, analyses):
        shift = ln["at"] - ln["start"]
        if ln.get("words"):
            ws = [{"word": str(w[0]).strip(), "start": round(float(w[1]) + shift, 3), "end": round(float(w[2]) + shift, 3), **({"est": True} if ln.get("est") else {})}
                  for w in ln["words"] if ln["lo"] - 0.05 <= float(w[1]) and float(w[2]) <= ln["hi"] + 0.35 and str(w[0]).strip()]
            if len(ws) < len([w for w in ln["words"] if str(w[0]).strip()]):
                continue  # (a line the piece cuts off gets no caption: half a sentence on screen reads as a mistake)
        else:
            ws = spread(ln["text"].split(), ln["at"], ln["at"] + ln["end"] - ln["start"])
        for w in ws:  # (two people's lines can overlap: the later one waits for the earlier, no word is lost)
            if out and w["start"] < out[-1]["end"]:
                w["start"] = out[-1]["end"]
            w["end"] = round(max(w["end"], w["start"] + 0.12), 3)
            out.append(w)
    return out


def on_keyframes(cuts: list[dict], analyses: dict[str, dict | None]) -> list[dict]:
    """Pieces that are joined without re-encoding start on a keyframe: each start moves back to the
    recording's nearest one (never into the piece before it)."""
    out = []
    for c in cuts:
        keys = [k["t"] for k in (analyses.get(c["recording"]) or {}).get("keyframes", []) if k["t"] <= c["start"] + 0.02]
        start = float(keys[-1]) if keys else c["start"]
        if out and out[-1]["recording"] == c["recording"] and start < out[-1]["end"]:
            start = c["start"]
        out.append({**c, "start": round(start, 3)})
    return out


def picture_area(src: str, at: float, w: int, h: int) -> tuple[int, int, int, int]:
    """(width, height, x, y) of the picture without the black bars a cutscene puts around it."""
    r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-ss", f"{max(0.0, at):.2f}", "-i", src, "-frames:v", "12", "-vf", "fps=2,cropdetect=limit=24:round=2:reset=0", "-an", "-f", "null", "-"],
                       capture_output=True, text=True)
    found = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", r.stderr or "")
    if not found:
        return w, h, 0, 0
    cw, ch, x, y = (int(v) for v in found[-1])
    if y > 0.04 * h:  # bars above and below are the same height (the game's subtitles sit in the lower one)
        ch = min(ch, h - 2 * y)
    if cw < 0.5 * w or ch < 0.4 * h or x + cw > w or y + ch > h:  # (a dark scene, not bars)
        return w, h, 0, 0
    return cw, ch, x, y


def crop_for(area: tuple[int, int, int, int]) -> str:
    """The middle of the picture, as wide as a 5:4 frame of its height (what fills a phone well)."""
    aw, ah, x, y = area
    cw = min(aw, int(ah * 1.25)) // 2 * 2
    return f"crop={cw}:{ah // 2 * 2}:{x + (aw - cw) // 2}:{y}"


def game_shows_subtitles(cuts: list[dict], analyses: dict[str, dict | None]) -> bool:
    """Whether the game already prints what is said at the bottom of the picture (then captions of our
    own would say it twice). Looked at in a few frames where someone speaks."""
    moments = []
    for c in cuts:
        for d in (analyses.get(c["recording"]) or {}).get("dialogue", []):
            if c["start"] <= d["start"] and d["end"] <= c["end"] and d["end"] - d["start"] >= 1.0:
                moments.append((c["recording"], (d["start"] + d["end"]) / 2))
    moments = moments[:: max(1, len(moments) // 5)][:5]
    if len(moments) < 2:
        return False
    try:
        import thumbnails as th

        with tempfile.TemporaryDirectory(prefix="ei-subs-") as tmp:
            files = []
            for k, (rec, t) in enumerate(moments):
                f = Path(tmp) / f"f{k}.jpg"
                subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(rec), "-frames:v", "1", "-q:v", "3", str(f)], capture_output=True)
                if f.is_file():
                    files.append(f)
            if len(files) < 2:
                return False
            hits = sum(1 for boxes, _ in th.text_and_turn(files) if any(0.68 <= (b[1] + b[3]) / 2 <= 0.97 and 0.25 <= (b[0] + b[2]) / 2 <= 0.75 for b in boxes))
        return hits * 2 >= len(files)
    except Exception as e:  # noqa: BLE001 (not knowing is not a reason to stop)
        log(f"could not check for the game's own subtitles ({type(e).__name__})")
        return False


def join_long(cuts: list[dict], dst: Path) -> None:
    """The pieces one after another, as they are in the recordings (stream copy)."""
    with tempfile.TemporaryDirectory(prefix="ei-plan-", dir=str(dst.parent)) as tmp:
        files = []
        for k, c in enumerate(cuts):
            f = Path(tmp) / f"p{k:03d}.mp4"
            ac.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{c['start']:.3f}", "-t", f"{c['end'] - c['start']:.3f}", "-i", str(c["recording"]),
                    "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-avoid_negative_ts", "make_zero", str(f)])
            files.append(f)
            ac.stage_progress(0.9 * (k + 1) / len(cuts), f"piece {k + 1} of {len(cuts)}")
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{f.name}'\n" for f in files), encoding="utf-8")
        part = dst.with_name(dst.stem + ".part.mp4")
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", "-movflags", "+faststart", str(part)],
                       cwd=tmp, check=False, capture_output=True)
        if not part.is_file() or part.stat().st_size == 0:
            part.unlink(missing_ok=True)
            raise AutocutError("the pieces could not be joined")
        part.replace(dst)


def render_vertical(cuts: list[dict], dst: Path, ass: str | None, W: int = 1080, H: int = 1920, fps: int = 30) -> None:
    """The pieces as one vertical video: the middle of the picture, large, over a blurred copy of the
    whole picture; the hook and the captions burned in; one loudness."""
    with tempfile.TemporaryDirectory(prefix="ei-plan-") as tmp:
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        graph = []
        info = ac.probe(cuts[0]["recording"])
        for k, c in enumerate(cuts):
            fg = crop_for(picture_area(str(c["recording"]), c["start"] + min(2.0, (c["end"] - c["start"]) / 2), info.width, info.height))
            cmd += ["-ss", f"{c['start']:.3f}", "-t", f"{c['end'] - c['start']:.3f}", "-i", str(c["recording"])]
            graph.append(f"[{k}:v]fps={fps},setsar=1,split[s{k}a][s{k}b];"
                         f"[s{k}a]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,crop={W // 4}:{H // 4},boxblur=6:2,scale={W}:{H},eq=brightness=-0.08[bg{k}];"
                         f"[s{k}b]{fg},scale={W}:-2[fg{k}];[bg{k}][fg{k}]overlay=0:(H-h)/2,format=yuv420p[v{k}];"
                         f"[{k}:a]aresample=48000,aformat=channel_layouts=stereo[a{k}]")
        graph.append("".join(f"[v{k}][a{k}]" for k in range(len(cuts))) + f"concat=n={len(cuts)}:v=1:a=1[vc][ac]")
        if ass:
            (Path(tmp) / "c.ass").write_text(ass, encoding="utf-8")
            graph.append("[vc]ass=c.ass[vo]")
        graph.append("[ac]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[ao]")
        part = dst.with_name(dst.stem + ".part.mp4")
        cmd += ["-filter_complex", ";".join(graph), "-map", "[vo]" if ass else "[vc]", "-map", "[ao]", "-c:v", "h264_videotoolbox", "-b:v", ac.bitrate_for(W, H, 12.0),
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(part.resolve())]
        r = subprocess.run(cmd, cwd=tmp, capture_output=True)
        if r.returncode != 0 or not part.is_file():
            part.unlink(missing_ok=True)
            raise AutocutError(f"the video could not be made: {r.stderr.decode(errors='replace').strip()[-300:]}")
        part.replace(dst)


def write_otio(cuts: list[dict], name: str, W: int, H: int, fps: float, dst: Path) -> Path | None:
    """The edit for the timeline: every piece on V1 with its own sound on A1 (linked)."""
    try:
        import otio_export as oe

        b = oe._Builder(name, W, H, fps)
        v1, a1 = b.track("V1", "video"), b.track("A1", "audio")
        for c in cuts:
            b.link += 1
            v1.append(b.clip(c["recording"], c["start"], c["end"] - c["start"], link=b.link))
            a1.append(b.clip(c["recording"], c["start"], c["end"] - c["start"], link=b.link))
        b.otio.adapters.write_to_file(b.tl, str(dst))
        return dst
    except Exception as e:  # noqa: BLE001 (the video is made without it)
        log(f"the timeline file could not be written ({type(e).__name__}: {e})")
        return None


def publish_text(plan: dict, cuts: list[dict], long: bool) -> str:
    lines = [str(plan.get("title", "")), "", f"Hook: {plan.get('hook', '')}", f"Why: {plan.get('why', '')}", f"Thumbnail: {plan.get('thumbnail', '')}", ""]
    if long:
        lines.append("Chapters:")
        t, seen, last = 0.0, None, -99.0
        for c in cuts:  # (YouTube: chapters at least 10 s apart, the first at 0:00: it covers the teaser too)
            if c.get("note") and c["note"] != seen and t - last >= 10:
                at = 0.0 if seen is None else t
                stamp = f"{int(at // 3600)}:{int(at % 3600 // 60):02d}:{int(at % 60):02d}" if at >= 3600 else f"{int(at // 60)}:{int(at % 60):02d}"
                lines.append(f"{stamp} {c['note']}")
                seen, last = c["note"], at
            t += c["end"] - c["start"]
        lines.append("")
    lines += [" ".join(plan.get("tags") or []), ""]
    return "\n".join(lines)


def safe_name(title: str) -> str:
    s = re.sub(r"[^\w\s\-]+", "", title, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()[:70] or "video"


def read_srt(path: Path) -> list[dict]:
    cues = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8", errors="replace").strip()):
        m = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)[^\n]*\n(.*)", block, re.S)
        if m:
            g = [int(x) for x in m.groups()[:8]]
            cues.append({"start": g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000, "end": g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000, "text": " ".join(m.group(9).split())})
    return cues


def video_folder(pj: Path, idx: int, plan: dict, out_dir: str | None) -> tuple[Path, str]:
    name = f"{idx + 1:02d} {safe_name(str(plan.get('title', '')))}"
    return (Path(out_dir).expanduser() if out_dir else pj.parent / "videos") / name, name


def check_video(plans_file: str, *, pick: str, out_dir: str | None = None) -> dict:
    """The finished video of a plan, checked without making anything: its length and shape, each
    caption against where its words are heard in the video's own sound, how it ends. The result and
    every caption are written to `check.txt` next to the video (and to the log), to be read."""
    ac.require_tools()
    data, pj = load(plans_file)
    idx = pick_plan(data["plans"], pick)
    plan = data["plans"][idx]
    folder, name = video_folder(pj, idx, plan, out_dir)
    video = folder / f"{name}.mp4"
    if not video.is_file():
        raise AutocutError(f"plan {idx + 1} has no finished video yet ({video.name} is not in {folder.parent.name}): make it first")
    ac.progress_plan({"check": 1.0})
    with ac.stage("check"):
        info = ac.probe(str(video))
        srt = folder / f"{name}.srt"
        cues = read_srt(srt) if srt.is_file() else []
        vertical = plan.get("platform") in VERTICAL
        limit = MAX_SECONDS.get(plan.get("platform"))
        problems, good = [], []
        if limit and info.duration > limit + 0.6:
            problems.append(f"it is {info.duration:.0f} s long: over the {limit + 1:.0f} s limit of {plan.get('platform')}")
        else:
            good.append(f"{info.duration:.0f} s long")
        if vertical and info.width >= info.height:
            problems.append(f"it is not vertical ({info.width}x{info.height})")
        else:
            good.append(f"{info.width}x{info.height}")
        if cues and not re.search(r"[.!?…\"”'’\-–—]$", cues[-1]["text"]):
            problems.append(f"the last caption does not end a sentence: “{cues[-1]['text']}”")
        for a, b in zip(cues, cues[1:]):
            if _tokens(a["text"]) == _tokens(b["text"]) and len(_tokens(a["text"])) >= 2 and b["start"] - a["end"] < 3:
                problems.append(f"the same caption twice in a row at {gp.hms(b['start'])}: “{b['text']}”")
        shown = " " + " ".join(t for c in cues for t in _tokens(c["text"])) + " "
        still = [x["wrong"] for x in plan.get("caption_fixes") or [] if isinstance(x, dict) and _tokens(x.get("wrong", ""))
                 and f" {' '.join(_tokens(x['wrong']))} " in shown and f" {' '.join(_tokens(x['wrong']))} " not in f" {' '.join(_tokens(x.get('right', '')))} "]
        if still:
            problems.append("corrected caption words are still in the captions (make the video again): " + "; ".join(still[:5]))
        chk = None
        if cues:
            first = next((str(p["recording"]) for p in plan.get("parts") or []), "")
            chk = check_captions(video, cues, language_of(analysis_of(first)) if first else None)
        if chk:
            chk.pop("moves", None)
            if chk["late"]:
                problems.append(f"{chk['late']} of {chk['found']} caption lines heard are more than 0.6 s from the speech (worst {chk['worst']} s): {'; '.join(chk['which'][:6])}")
            elif chk["found"] * 2 < chk["lines"]:
                good.append(f"only {chk['found']} of {chk['lines']} caption lines could be heard in the video (quiet speech or loud music), those within {max(chk['worst'], 0.1)} s; "
                            "the others are not confirmed by sound")
            else:
                good.append(f"{chk['found']} of {chk['lines']} caption lines heard, all within {max(chk['worst'], 0.1)} s of the speech")
        elif not cues:
            good.append("no captions (nothing is said, or no caption file is next to the video)")
        lines = [f"Check of: {video.name}", time.strftime("%Y-%m-%d %H:%M"), ""]
        lines += ["PROBLEMS"] + [f"- {p}" for p in problems] + [""] if problems else ["No problems found by the check.", ""]
        lines += ["Checked"] + [f"- {g}" for g in good] + ["", "The captions, to read (time in the video, words):"]
        lines += [f"{gp.hms(c['start'])}  {c['text']}" for c in cues] or ["(none)"]
        lines += ["", "To correct something: Make Video from Plan → this plan → say what is wrong, or give a caption correction (wrong words => right words)."]
        out = folder / "check.txt"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for p in problems:
            log(f"CHECK: {p}")
        for g in good:
            log(f"checked: {g}")
        for c in cues[:80]:
            log(f"  {gp.hms(c['start'])}  {c['text']}")
    return {"outputs": [str(out)], "out_dir": str(folder), "plan": idx + 1, "title": plan.get("title"), "problems": problems, "checked": good, "check": chk, "captions": len(cues)}


def make(plans_file: str, *, pick: str = "", request: str = "", out_dir: str | None = None, writer: str = "claude", fresh: bool = False,
         captions: bool = True, cuts_only: bool = False, hook: bool = True, fix: str = "", caption_fix: str = "") -> dict:
    """The video of one plan (`pick`: its number or a part of its title), or of a new plan written for
    `request`. `fresh`: choose the cuts again. `cuts_only`: stop after the cuts are saved. `hook`: the video opens
    with a 2–3 second teaser from inside it (saved in the plan as `hook_cut`; delete it there to go without).
    `fix`: what is wrong with the video made before, in the creator's words: Claude corrects that in the plan
    and the video is made again. `caption_fix`: 'wrong words => right words' (several: one per line or after ';'),
    kept in the plan and made in every later make."""
    ac.require_tools()
    data, pj = load(plans_file)
    plans = data["plans"]
    game = str(data.get("game") or "this game")
    ac.progress_plan({"plan": 0.1, "cuts": 0.2, "video": 0.7})
    with ac.stage("plan"):
        if request.strip():
            plan = request_plan(data, request.strip(), writer)
            plans.append(plan)
            idx = len(plans) - 1
            log(f"new plan {idx + 1}: {plan['title']} ({plan['platform']}, {plan['length']})")
        else:
            idx = pick_plan(plans, pick)
            plan = plans[idx]
            log(f"plan {idx + 1}: {plan.get('title')} ({plan.get('platform')}, {plan.get('length')})")
        parts = plan.get("parts") or []
        if not parts:
            raise AutocutError("this plan has no parts")
        missing = sorted({Path(str(p["recording"])).name for p in parts if not Path(str(p["recording"])).is_file()})
        if missing:
            raise AutocutError(f"recording(s) not found: {', '.join(missing)} (is the disk connected?)")
        if caption_fix.strip():
            pairs = parse_caption_fixes(caption_fix)
            if not pairs:
                raise AutocutError("a caption correction is written as: wrong words => right words")
            log(f"{add_caption_fixes(plan, pairs)} caption correction(s) kept in the plan: " + "; ".join(f"“{w}” → “{r}”" for w, r in pairs[:6]))
    with ac.stage("cuts"):
        by_rec = {str(p["recording"]): analysis_of(str(p["recording"])) for p in parts}
        hint = ""
        if writer == "claude" and game != "this game":
            try:
                hint = gp.name_hint(gp.research_game(game, writer))
            except Exception:  # noqa: BLE001 (names are a help, not a need)
                hint = ""
        for rec in by_rec:  # what is said, at its real time (before anything is chosen by it)
            by_rec[rec] = listen_again(rec, by_rec[rec], [(float(p["start"]) - REACH / 2, float(p["end"]) + REACH / 2) for p in parts if str(p["recording"]) == rec])
        names = [n.strip() for n in hint.split(":", 1)[-1].split(",") if n.strip()]
        wide = {rec: merge_spans([(max(0.0, float(p["start"]) - REACH / 2), float(p["end"]) + REACH / 2) for p in parts if str(p["recording"]) == rec]) for rec in by_rec}
        if sum(e - s for sp in wide.values() for s, e in sp) <= 600:  # (longer stretches: only the chosen pieces are read, below)
            for rec in by_rec:
                by_rec[rec] = use_subtitles(by_rec[rec], read_subtitles(rec, wide[rec], names), wide[rec])
        if fix.strip():
            if writer != "claude":
                raise AutocutError("saying what is wrong needs Claude (the writer is set to rules)")
            old = video_folder(pj, idx, plan, out_dir)
            was = old[0] / f"{old[1]}.srt"
            shown = [f"{gp.hms(c['start'])}  {c['text']}" for c in read_srt(was)] if was.is_file() else []
            try:
                log("corrected: " + apply_fix(plan, game, [by_rec[str(p["recording"])] for p in parts], fix.strip(), shown))
            except ValueError as e:
                raise AutocutError(f"the correction could not be read ({e})") from e
            fresh = False  # (the corrected plan is what is made)
        if plan.get("no_hook"):
            hook = False
        if plan.get("cuts") and not fresh and plan.get("times") != "heard":
            log("the saved cuts were chosen from rough line times: they are chosen again")
            fresh = True
        if plan.get("cuts") and not fresh and all(Path(str(c.get("recording", ""))).is_file() for c in plan["cuts"]):
            cuts = [c for c in plan["cuts"] if float(c.get("end", 0)) - float(c.get("start", 0)) >= MIN_PIECE]
            log(f"using the {len(cuts)} cut(s) saved in the plan")
        else:
            cuts = choose_cuts(plan, game, [by_rec[str(p["recording"])] for p in parts], writer)
        vertical = plan.get("platform") in VERTICAL
        if vertical:
            cuts = fit_short(cuts, MAX_SECONDS[plan["platform"]] - (HOOK_LINE if hook else 0.0))
        else:
            cuts = on_keyframes(cuts, by_rec)
        if not cuts:
            raise AutocutError("no cuts were left to make a video from")
        if vertical:
            cuts = trim_menus(cuts, by_rec) or cuts
        if vertical:  # (captions are shown word by word: the pieces are heard once more for lines that were missed)
            for rec in by_rec:
                mine = merge_spans([(max(0.0, c["start"] - 1.0), c["end"] + 1.0) for c in cuts if c["recording"] == rec], gap=0.0)
                by_rec[rec] = listen_closer(rec, by_rec[rec], mine)  # (no names given to the listener: it repeats them where nothing is said)
                around = merge_spans([(max(0.0, s - 5.0), e + 5.0) for s, e in mine], gap=0.0)  # (a little further: how the last sentence ends)
                by_rec[rec] = use_subtitles(by_rec[rec], read_subtitles(rec, around, names), around)  # the game's own wording, where it shows it
                if writer == "claude":
                    by_rec[rec] = proofread(by_rec[rec], mine, game, names)
        teaser = None
        if hook:
            saved = plan.get("hook_cut") if not fresh else None
            if isinstance(saved, dict) and Path(str(saved.get("recording", ""))).is_file() and HOOK[0] <= float(saved.get("end", 0)) - float(saved.get("start", 0)) <= 10:
                teaser = saved
            else:
                teaser = hook_cut(plan, cuts, by_rec, not vertical)
            if teaser:
                plan["hook_cut"] = teaser
                log(f"hook: {teaser['end'] - teaser['start']:.1f}s from {gp.hms(teaser['start'])} of {Path(str(teaser['recording'])).name}, shown first")
            else:
                log("no hook: nothing in the pieces stands out enough to show first (or the video is very short)")
        plan.pop("hook_idea", None)
        if vertical:  # no piece starts or ends inside a sentence (now that the hook's length is known)
            cuts = whole_at_edges(cuts, by_rec, MAX_SECONDS[plan["platform"]] - (teaser["end"] - teaser["start"] if teaser else 0.0))
            if not cuts:
                raise AutocutError("no cuts were left to make a video from")
            if teaser and not any(c["recording"] == teaser["recording"] and c["start"] - 0.5 <= teaser["start"] and teaser["end"] <= c["end"] + 0.5 for c in cuts):
                teaser = None  # (what it showed is no longer in the video)
                plan.pop("hook_cut", None)
        plan["cuts"] = cuts
        plan["times"] = "heard"
        plan["cut_s"] = round(sum(c["end"] - c["start"] for c in cuts), 1)
        plan["cuts_made"] = time.strftime("%Y-%m-%d %H:%M")
        tmp = pj.with_name(pj.name + ".part")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(pj)
        gp.write_md(pj.parent, game, plans)
        planned = sum(float(p["end"]) - float(p["start"]) for p in parts)
        log(f"{len(cuts)} piece(s), {plan['cut_s'] / 60:.1f} min (the plan's stretches: {planned / 60:.1f} min)")
    folder, name = video_folder(pj, idx, plan, out_dir)
    res = {"outputs": [], "out_dir": str(folder), "plan": idx + 1, "title": plan.get("title"), "pieces": len(cuts), "seconds": plan["cut_s"], "plans": str(pj),
           "hook": bool(teaser)}
    if teaser:  # (from here on the teaser is simply the first piece)
        cuts = [teaser, *cuts]
    if cuts_only:
        return res
    with ac.stage("video"):
        folder.mkdir(parents=True, exist_ok=True)
        video = folder / f"{name}.mp4"
        info = ac.probe(cuts[0]["recording"])
        words, unused = fix_captions(heard_words(cuts, by_rec), plan.get("caption_fixes") or [])
        if plan.get("caption_fixes"):
            log(f"{len(plan['caption_fixes']) - len(unused)} of the plan's {len(plan['caption_fixes'])} caption correction(s) made"
                + (f" (not in this video's captions: {'; '.join(unused[:4])})" if unused else ""))
        if vertical:
            import captions as cp

            bars = any(picture_area(str(c["recording"]), c["start"] + min(2.0, (c["end"] - c["start"]) / 2), info.width, info.height)[3] > 0 for c in cuts)
            if captions and words and not bars and game_shows_subtitles(cuts, by_rec):  # (subtitles in a cutscene's black bar are cropped away with it)
                log("the game shows its own subtitles: no captions are added over them")
                captions = False
            length = sum(c["end"] - c["start"] for c in cuts)

            def render(ws: list[dict]) -> None:
                ass = cp.build_ass(ws if captions else [], hook_text(plan), 1080, 1920, length, "pop", "intro") if (captions and ws) or hook_text(plan) else None
                render_vertical(cuts, video, ass)

            render(words)
            W, H, fps = 1080, 1920, 30.0
            if words:  # the finished video is listened to: are the captions where the words are?
                lang = language_of(by_rec.get(str(cuts[0]["recording"])))
                chk = check_captions(video, ac.group_words(words, max_words=7, max_chars=38), lang)
                if chk and chk["moves"]:
                    words, n = moved_to_the_sound(words, chk["moves"])
                    if n:
                        log(f"{n} caption line(s) timed only by the game's subtitle were moved to where they are heard")
                        if captions:
                            render(words)
                        chk = check_captions(video, ac.group_words(words, max_words=7, max_chars=38), lang) or chk
                if chk:
                    chk.pop("moves", None)
                    res["check"] = chk
                    if chk["late"]:
                        log(f"CHECK: {chk['late']} of {chk['found']} checked caption lines are more than 0.6 s from the speech (worst {chk['worst']} s): {'; '.join(chk['which'][:4])}")
                    elif chk["found"] * 2 < chk["lines"]:
                        log(f"note: only {chk['found']} of {chk['lines']} caption lines could be checked by listening to the finished video (quiet speech or loud music); "
                            f"those are within {max(chk['worst'], 0.1)} s, the others follow the game's subtitles and are not confirmed by sound")
                    else:
                        log(f"checked: {chk['found']} of {chk['lines']} caption lines heard in the finished video, all within {max(chk['worst'], 0.1)} s of the speech")
        else:
            join_long(cuts, video)
            W, H, fps = info.width, info.height, float(info.fps or 30.0)
        outputs = [str(video)]
        if words:
            srt = folder / f"{name}.srt"
            srt.write_text(ac.to_srt(ac.group_words(words, max_words=7, max_chars=38)), encoding="utf-8")
            outputs.append(str(srt))
            res["srt"] = str(srt)
        otio = write_otio(cuts, name, W, H, fps, folder / f"{name}.otio")
        if otio:
            outputs.append(str(otio))
            res["otio"] = str(otio)
        pub = folder / "publish.txt"
        pub.write_text(publish_text(plan, cuts, not vertical), encoding="utf-8")
        outputs.append(str(pub))
    log(f"made: {video.name} ({plan['cut_s'] / 60:.1f} min, {len(cuts)} piece(s))")
    res["outputs"] = outputs
    return res


def add_args(p) -> None:
    p.add_argument("--plans", required=True, help="the plans.json made by Plan Videos (or its folder)")
    p.add_argument("--pick", default="", help="which plan: its number (as in plans.md) or a part of its title")
    p.add_argument("--request", default="", help="instead of a plan from the list: say what video you want; it is planned from the recordings, then made")
    p.add_argument("--out-dir", help="where the video's folder goes (default: 'videos' next to plans.json)")
    p.add_argument("--writer", default="claude", choices=["claude", "rules"], help="rules: no Claude, the plan's stretches are used whole")
    p.add_argument("--fresh", action="store_true", help="choose the cuts again (drops the cuts saved in the plan)")
    p.add_argument("--no-captions", action="store_true")
    p.add_argument("--cuts-only", action="store_true", help="choose and save the cuts, make no video")
    p.add_argument("--no-hook", action="store_true", help="do not open with a 2–3 second teaser from inside the video")
    p.add_argument("--fix", default="", help="what is wrong with the video made before, in your own words: that is corrected and the video made again")
    p.add_argument("--caption-fix", default="", help="'wrong words => right words' (several: after ';'): kept in the plan, made in every later make")
    p.add_argument("--check", action="store_true", help="make nothing: check the plan's finished video (length, shape, captions against its own sound) and write check.txt")


def run_args(a) -> dict:
    if a.check:
        return check_video(a.plans, pick=a.pick, out_dir=a.out_dir)
    return make(a.plans, pick=a.pick, request=a.request, out_dir=a.out_dir, writer=a.writer, fresh=a.fresh, captions=not a.no_captions, cuts_only=a.cuts_only, hook=not a.no_hook,
                fix=a.fix, caption_fix=a.caption_fix)
