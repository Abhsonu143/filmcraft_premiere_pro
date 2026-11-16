#!/usr/bin/env python3
"""fix_log: what was corrected in one video, round by round, kept in the video's folder.

A unit for any tool that makes a video again from the creator's notes. Each round is kept
(`fixes.json`, and `fixes.md` to read): what the creator said was wrong, what was changed for it,
and what could not be done. The next round is given all of it, so a correction is never undone by
a later one, a note that comes back ("still too quiet") is known to have been tried already, and
what is still open is not forgotten.

    fix_log.block(folder)                       the earlier rounds, as text for a prompt ("" when none)
    fix_log.text(rounds)                        the same for rounds a tool keeps itself
    fix_log.add(folder, notes, changed, open)   one more round
"""

from __future__ import annotations

import json
import time
from pathlib import Path

FILE, READ = "fixes.json", "fixes.md"
MOST = 40  # rounds kept


def rounds(folder) -> list[dict]:
    try:
        data = json.loads((Path(folder) / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []


def _lines(v, most: int = 20) -> list[str]:
    if isinstance(v, str):
        v = v.splitlines()
    return [" ".join(str(x).split())[:500] for x in (v or []) if str(x).strip()][:most]


def add(folder, notes, changed, still_open=None, tool: str = "") -> list[dict]:
    """Keeps one more round. Returns all rounds."""
    folder = Path(folder)
    all_ = rounds(folder)
    all_.append({"round": (all_[-1].get("round", len(all_)) + 1) if all_ else 1, "when": time.strftime("%Y-%m-%d %H:%M"), "tool": tool,
                 "notes": _lines(notes), "changed": _lines(changed), "open": _lines(still_open)})
    all_ = all_[-MOST:]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / FILE).write_text(json.dumps(all_, ensure_ascii=False, indent=1), encoding="utf-8")
    (folder / READ).write_text(as_md(all_), encoding="utf-8")
    return all_


def as_md(all_: list[dict]) -> str:
    out = ["# Corrections of this video", ""]
    for r in all_:
        out += [f"## Round {r.get('round', '?')} · {r.get('when', '')}", "", "**You said:**"] + [f"- {n}" for n in r.get("notes", [])]
        out += ["", "**Changed:**"] + ([f"- {c}" for c in r.get("changed", [])] or ["- (nothing)"])
        if r.get("open"):
            out += ["", "**Still open:**"] + [f"- {o}" for o in r["open"]]
        out.append("")
    return "\n".join(out)


def still_open(folder) -> list[str]:
    """What the last round left undone."""
    all_ = rounds(folder)
    return list(all_[-1].get("open", [])) if all_ else []


def keep(folder, all_: list[dict]) -> None:
    """Writes rounds a tool keeps elsewhere (in its plan) into the video's folder too, to read."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / FILE).write_text(json.dumps(all_[-MOST:], ensure_ascii=False, indent=1), encoding="utf-8")
    (folder / READ).write_text(as_md(all_[-MOST:]), encoding="utf-8")


def block(folder, last: int = 8) -> str:
    """The earlier rounds as a section of a prompt ("" when there were none)."""
    return text(rounds(folder), last)


def text(all_: list[dict], last: int = 8) -> str:
    """Rounds as a section of a prompt ("" when there are none)."""
    all_ = all_[-last:]
    if not all_:
        return ""
    out = ["## Earlier rounds of corrections on this same video (oldest first)",
           "These were asked and done before. Keep every one of them: a new note never undoes an earlier correction unless it says so.",
           "If a new note is about something already corrected (\"still\", \"again\", \"not fixed\"), what was tried did not work: do it",
           "another way or more strongly, and say what you did differently. What is listed as still open stays open unless you can do it now.", ""]
    for r in all_:
        out.append(f"Round {r.get('round', '?')} ({r.get('when', '')})")
        out += [f"  asked: {n}" for n in r.get("notes", [])]
        out += [f"  changed: {c}" for c in r.get("changed", [])]
        out += [f"  still open: {o}" for o in r.get("open", [])]
    return "\n".join(out) + "\n"
