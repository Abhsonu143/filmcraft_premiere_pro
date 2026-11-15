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
MAX_SECONDS = {"shorts": 60.0, "reels": 90.0}
MIN_PIECE = 2.0  # seconds: nothing shorter is kept
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
    length = (f"- This is a short vertical video: the pieces together must be 15 to {limit:.0f} seconds. Start on the strongest second (the hook), not on a lead-in.\n"
              if vertical else
              f"- This is a long video (planned: {plan.get('length')}). Keep the story whole and in order; take out only what a viewer would skip. Aim for the planned length, give or take 15%.\n")
    return f"""You are the editor of a gaming video made from a creator's own recordings of "{game}".
The plan below says roughly which stretches to use. Read each stretch in detail and choose the exact pieces to keep.

## The plan
Title: {plan.get("title")}
For: {plan.get("platform")} · {plan.get("format")} · {plan.get("length")}
Hook: {plan.get("hook")}
Why it works: {plan.get("why")}
{('What the creator asked for: ' + str(plan.get("requested")) + chr(10)) if plan.get("requested") else ''}
## The stretches (per 5 seconds: action 0-1, what is on screen, scene cuts; indented: every line said, with its time)
{chr(10).join(blocks)}

## Rules
{length}- A piece starts just before a line is said or on a scene cut, and ends just after a line ends. Never cut a line or an exchange in the middle.
- Leave out: menus, maps, skill and inventory screens, loading, deaths and repeated failed attempts, long riding or walking with nothing said
  and nothing happening. Keep: story dialogue, reveals, the build-up and the end of a fight, anything funny or beautiful.
- Use only times inside the stretches shown. Pieces of one stretch are in order and do not overlap. You may drop a whole stretch if it adds nothing.
- note: a few words on what happens in the piece (it becomes a chapter title in a long video).
Answer with only this JSON:
{{"cuts": [{{"part": 1, "start": "H:MM:SS", "end": "H:MM:SS", "note": "..."}}]}}"""


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
        raw = sw.extract_json(sw.ask_claude(cuts_prompt(plan, game, blocks), timeout=900)).get("cuts")
    except (AutocutError, ValueError) as e:
        log(f"the cuts could not be chosen ({e}): the plan's stretches are used whole")
        return whole_parts(plan, analyses)
    cuts = clean_cuts(raw, plan, analyses)
    if not cuts:
        log("no usable cuts came back: the plan's stretches are used whole")
        return whole_parts(plan, analyses)
    return cuts


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


def said_words(cuts: list[dict], analyses: dict[str, dict | None]) -> list[dict]:
    """What is said in the pieces, word by word, in the video's own time (a line's words are spread
    evenly over it)."""
    out, t = [], 0.0
    for c in cuts:
        a = analyses.get(c["recording"])
        for d in (a or {}).get("dialogue", []):
            s, e = max(float(d["start"]), c["start"]), min(float(d["end"]), c["end"])
            ws = str(d.get("text", "")).split()
            if e - s < 0.2 or not ws or not c["start"] - 0.5 <= d["start"] < c["end"]:
                continue
            per = (e - s) / len(ws)
            out += [{"word": w, "start": round(t + s - c["start"] + k * per, 3), "end": round(t + s - c["start"] + (k + 1) * per, 3)} for k, w in enumerate(ws)]
        t += c["end"] - c["start"]
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
        for c in cuts:  # (YouTube: chapters at least 10 s apart, the first at 0:00)
            if c.get("note") and c["note"] != seen and t - last >= 10:
                at = f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}" if t >= 3600 else f"{int(t // 60)}:{int(t % 60):02d}"
                lines.append(f"{at} {c['note']}")
                seen, last = c["note"], t
            t += c["end"] - c["start"]
        lines.append("")
    lines += [" ".join(plan.get("tags") or []), ""]
    return "\n".join(lines)


def safe_name(title: str) -> str:
    s = re.sub(r"[^\w\s\-]+", "", title, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()[:70] or "video"


def make(plans_file: str, *, pick: str = "", request: str = "", out_dir: str | None = None, writer: str = "claude", fresh: bool = False,
         captions: bool = True, cuts_only: bool = False) -> dict:
    """The video of one plan (`pick`: its number or a part of its title), or of a new plan written for
    `request`. `fresh`: choose the cuts again. `cuts_only`: stop after the cuts are saved."""
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
    with ac.stage("cuts"):
        by_rec = {str(p["recording"]): analysis_of(str(p["recording"])) for p in parts}
        if plan.get("cuts") and not fresh and all(Path(str(c.get("recording", ""))).is_file() for c in plan["cuts"]):
            cuts = [c for c in plan["cuts"] if float(c.get("end", 0)) - float(c.get("start", 0)) >= MIN_PIECE]
            log(f"using the {len(cuts)} cut(s) saved in the plan")
        else:
            cuts = choose_cuts(plan, game, [by_rec[str(p["recording"])] for p in parts], writer)
        vertical = plan.get("platform") in VERTICAL
        if vertical:
            cuts = fit_short(cuts, MAX_SECONDS[plan["platform"]])
        else:
            cuts = on_keyframes(cuts, by_rec)
        if not cuts:
            raise AutocutError("no cuts were left to make a video from")
        plan["cuts"] = cuts
        plan["cut_s"] = round(sum(c["end"] - c["start"] for c in cuts), 1)
        plan["cuts_made"] = time.strftime("%Y-%m-%d %H:%M")
        tmp = pj.with_name(pj.name + ".part")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(pj)
        gp.write_md(pj.parent, game, plans)
        planned = sum(float(p["end"]) - float(p["start"]) for p in parts)
        log(f"{len(cuts)} piece(s), {plan['cut_s'] / 60:.1f} min (the plan's stretches: {planned / 60:.1f} min)")
    name = f"{idx + 1:02d} {safe_name(str(plan.get('title', '')))}"
    folder = (Path(out_dir).expanduser() if out_dir else pj.parent / "videos") / name
    res = {"outputs": [], "out_dir": str(folder), "plan": idx + 1, "title": plan.get("title"), "pieces": len(cuts), "seconds": plan["cut_s"], "plans": str(pj)}
    if cuts_only:
        return res
    with ac.stage("video"):
        folder.mkdir(parents=True, exist_ok=True)
        video = folder / f"{name}.mp4"
        info = ac.probe(cuts[0]["recording"])
        words = said_words(cuts, by_rec)
        if vertical:
            import captions as cp

            if captions and words and game_shows_subtitles(cuts, by_rec):
                log("the game shows its own subtitles: no captions are added over them")
                captions = False
            ass = cp.build_ass(words if captions else [], hook_text(plan), 1080, 1920, plan["cut_s"], "pop", "intro") if (captions and words) or hook_text(plan) else None
            render_vertical(cuts, video, ass)
            W, H, fps = 1080, 1920, 30.0
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


def run_args(a) -> dict:
    return make(a.plans, pick=a.pick, request=a.request, out_dir=a.out_dir, writer=a.writer, fresh=a.fresh, captions=not a.no_captions, cuts_only=a.cuts_only)
