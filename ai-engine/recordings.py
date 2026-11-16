#!/usr/bin/env python3
"""recordings: your analysed recordings as one library that any tool can read.

Plan Videos analyses each recording once and keeps what it found next to it (what is said and
when, the action second by second, what is on screen). This unit opens one or more of its plans
files (so several games can be used together) and answers what the tools need, without ever
analysing again:

    lib = Library([".../Ghost of Yotei - plans/plans.json"])
    lib.digests()            every recording in rough steps, for planning
    lib.detail(3, 600, 900)  a stretch every 5 seconds with every line said, for choosing cuts
    lib.lines(3, 600, 900)   the lines said there
    lib.whole(3, 612.3, 618) the same span moved so that no line is cut in the middle
    lib.lit(3, 600, 5)       where a 5 second shot can start so that it is not a black screen

A recording is named by its number R1, R2 … through all the plans files in the order given.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import gameplay_plan as gp
import plan_make as pm
from autocut import AutocutError, log

BLACK = 12.0  # a frame's brightness (0-255) below which nothing can be seen


class Library:
    def __init__(self, plans_files: list[str]):
        self.items: list[dict] = []  # {"n", "game", "path", "analysis"}
        self.games: list[str] = []
        missing = 0
        for pf in plans_files:
            data, _ = pm.load(pf)
            game = str(data.get("game") or "this game")
            if game not in self.games:
                self.games.append(game)
            for r in data.get("recordings") or []:
                a = pm.analysis_of(str(r))
                if not a:
                    missing += 1
                    continue
                if not Path(str(r)).is_file():
                    missing += 1
                    continue
                self.items.append({"n": len(self.items) + 1, "game": game, "path": str(r), "analysis": {**a, "source": str(r)}})
        if missing:
            log(f"{missing} recording(s) are not there now, or have no analysis next to them: they are not used")
        if not self.items:
            raise AutocutError("none of the recordings of these plans files is there with its analysis (is the disk connected? run Plan Videos on them first)")

    @property
    def seconds(self) -> float:
        return sum(float(i["analysis"].get("duration") or 0) for i in self.items)

    def get(self, n) -> dict:
        try:
            k = int(str(n).strip().lstrip("Rr"))
        except ValueError as e:
            raise AutocutError(f"{n} is not a recording number") from e
        if not 1 <= k <= len(self.items):
            raise AutocutError(f"there are {len(self.items)} recordings: R{k} is not one of them")
        return self.items[k - 1]

    def duration(self, n) -> float:
        return float(self.get(n)["analysis"].get("duration") or 0)

    def digests(self, most_rows: int = 700) -> list[str]:
        """Every recording as lines a planner can read. The step is chosen so that all of them
        together stay readable (30 seconds to 5 minutes a row)."""
        step = int(np.clip(self.seconds / most_rows, 30, 300) // 30 * 30) or 30
        many = len(self.games) > 1
        return [gp.digest(i["n"], i["analysis"], step).replace("## R", f"## [{i['game']}] R" if many else "## R", 1) for i in self.items]

    def detail(self, n, start: float, end: float, step: int = 5) -> str:
        i = self.get(n)
        end = min(float(end), self.duration(n))
        return f"### R{i['n']} ({i['game']}) {gp.hms(start)}–{gp.hms(end)}\n" + pm.detail(i["analysis"], max(0.0, float(start)), end, step)

    def lines(self, n, start: float, end: float, mostly: bool = False) -> list[dict]:
        """The lines said in the span: every line that touches it, or with `mostly` only those whose
        middle is in it (so a span padded around its lines finds the same lines again)."""
        said = self.get(n)["analysis"].get("dialogue", [])
        if mostly:
            return [d for d in said if start <= (d["start"] + d["end"]) / 2 <= end]
        return [d for d in said if d["end"] > start and d["start"] < end]

    def whole(self, n, start: float, end: float) -> tuple[float, float]:
        """The span with no line cut in the middle, inside the recording."""
        a = self.get(n)["analysis"]
        s, e = pm.keep_lines_whole(max(0.0, float(start)), float(end), a)
        return s, min(e, float(a.get("duration") or e))

    def lit(self, n, start: float, seconds: float, reach: float = 30.0) -> float:
        """A start at or soon after `start` whose `seconds` are not a black screen (a fade, a loading
        screen), as far as the analysis saw; `start` itself when there is none within `reach`."""
        kf = self.get(n)["analysis"].get("keyframes") or []
        a = float(start)
        while a <= start + reach:
            if not any(float(k.get("bright", 255)) < BLACK for k in kf if a - 1.0 <= float(k.get("t", -9)) <= a + seconds + 1.0):
                return a
            a += 2.0
        return float(start)

    def research(self) -> dict[str, dict]:
        """What is known about each game (looked up once a week and kept)."""
        return {g: gp.research_game(g) for g in self.games if g != "this game"}
