#!/usr/bin/env python3
"""explain_run: the code of an explainer's script is run, and the output shown is the real one.

Optional (`explain --run-code`). Python only: a bug scene's code and its fix, a terminal that runs a
file shown in the same chapter (`python3 f.py`), and interpreter sessions (`>>> …`). Each runs for
at most 10 seconds in a temporary folder inside the macOS sandbox: no network, and nothing can be
written outside that folder. Where the real output differs from what the script shows, the scene
gets the real output and the difference is listed for the creator (the spoken lines of that
chapter may then need a look). What cannot be run here is said so, never guessed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT = 10
MAX_ROWS = 12
PROFILE = ('(version 1)(deny default)(allow process-fork)(allow process-exec)(allow file-read*)(allow file-write* (subpath "{tmp}"))'
           "(allow sysctl-read)(allow mach-lookup)(allow signal (target self))")
SESSION = r"""
import code, contextlib, io, json, sys
con, out, buf = code.InteractiveConsole({"__name__": "__main__"}), [], io.StringIO()
for line in json.load(open(sys.argv[1])):
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        more = con.push(line)
    if not more:
        out.append(buf.getvalue())
        buf = io.StringIO()
    else:
        out.append(None)
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    con.push("")
if out and buf.getvalue():
    out[-1] = (out[-1] or "") + buf.getvalue()
json.dump(out, open(sys.argv[2], "w"))
"""


def sandboxed(args: list[str], tmp: Path) -> tuple[int, str, str] | None:
    """(exit code, output, errors) of Python run on `args` inside the sandbox; None when it cannot be run here."""
    box = shutil.which("sandbox-exec") or "/usr/bin/sandbox-exec"
    if not Path(box).is_file():
        return None
    try:
        p = subprocess.run([box, "-p", PROFILE.format(tmp=str(tmp.resolve())), sys.executable, "-I", *args], cwd=tmp, capture_output=True, text=True, timeout=TIMEOUT,
                           env={"PATH": "/usr/bin:/bin", "HOME": str(tmp), "TMPDIR": str(tmp), "PYTHONDONTWRITEBYTECODE": "1"})
    except subprocess.TimeoutExpired:
        return (124, "", f"(still running after {TIMEOUT} seconds: stopped)")
    except OSError:
        return None
    return p.returncode, p.stdout[-20000:], p.stderr[-20000:]


def short_error(err: str) -> list[str]:
    """A traceback as a viewer needs it: its first line and what was raised (paths differ on every machine)."""
    lines = [ln.rstrip() for ln in err.splitlines() if ln.strip()]
    if not lines:
        return []
    if lines[0].startswith("Traceback") and len(lines) > 2:
        where = next((ln.strip() for ln in lines[1:] if ln.lstrip().startswith("File ")), "")
        m = re.search(r'File ".*?([^/"]+)", (line \d+)', where)
        return [lines[0]] + ([f'  File "{m.group(1)}", {m.group(2)}'] if m else []) + [lines[-1]]
    return lines[-4:]


def run_file(files: dict[str, list[str]], name: str) -> list[dict] | None:
    """The rows a terminal shows after `python3 name`, really run."""
    with tempfile.TemporaryDirectory(prefix="ei-run-") as t:
        tmp = Path(t)
        for fname, lines in files.items():
            if re.fullmatch(r"[\w.\-]+\.py", fname):
                (tmp / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")
        if not (tmp / name).is_file():
            return None
        got = sandboxed([name], tmp)
    if got is None:
        return None
    _, out, err = got
    return [{"t": "out", "text": ln.rstrip()[:200]} for ln in out.splitlines() if ln.strip()] + [{"t": "err", "text": ln[:200]} for ln in short_error(err)]


def run_session(inputs: list[str]) -> list[list[str]] | None:
    """What an interpreter prints after each input line of a `>>>` session (an empty list while a statement is still open)."""
    with tempfile.TemporaryDirectory(prefix="ei-run-") as t:
        tmp = Path(t)
        (tmp / "session.py").write_text(SESSION, encoding="utf-8")
        (tmp / "in.json").write_text(json.dumps(inputs), encoding="utf-8")
        got = sandboxed(["session.py", "in.json", "out.json"], tmp)
        try:
            outs = json.loads((tmp / "out.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    if got is None:
        return None
    res = []
    for o in outs:
        lines = [ln.rstrip() for ln in (o or "").splitlines() if ln.strip()]
        res.append(short_error("\n".join(lines)) if lines and lines[0].startswith("Traceback") else lines)
    return res


def _said(rows: list[dict]) -> list[str]:
    return [" ".join(str(r.get("text", "")).split()) for r in rows if r.get("t") != "in" and str(r.get("text", "")).strip()]


def same(shown: list[dict], real: list[dict]) -> bool:
    """Whether a terminal shows what really came out. An error is the same error when what was
    raised is the same (the lines of a traceback in between are not compared)."""
    a, b = _said(shown), _said(real)
    if any(r.get("t") == "err" for r in real) and a and b:
        plain = lambda rows: [x for x in rows if not x.startswith(("Traceback", "File "))]  # noqa: E731
        return plain(a) == plain(b) or (a[-1] == b[-1] and plain(a)[:-1] == plain(b)[:-1])
    return a == b


def _command(rows: list[dict]) -> tuple[int, str] | None:
    for i, r in enumerate(rows):
        m = re.fullmatch(r"\s*(?:\$\s+)?python3?\s+([\w.\-]+\.py)\s*", str(r.get("text", "")))
        if m and (r.get("t") == "in" or i == 0):
            return i, m.group(1)
    return None


def check_script(script: dict) -> list[dict]:
    """Runs what can be run in the script's scenes. Scenes whose output was not the real one get the
    real one. Returns what was found, in the form of the script's `checks`."""
    found: list[dict] = []
    version = f"Python {sys.version_info[0]}.{sys.version_info[1]}"
    ran = 0

    def differs(where: str, what: str, shown: list[dict], real: list[dict]) -> None:
        found.append({"where": where, "problem": f"run with {version}, {what} prints something else than the script showed. The scene now shows the real output: "
                                                  "read this chapter's spoken lines against it.",
                      "was": " / ".join(_said(shown))[:300] or "(nothing)", "now": " / ".join(_said(real))[:300] or "(nothing)", "corrected": True})

    for c, ch in enumerate(script.get("chapters", []), 1):
        files: dict[str, list[str]] = {}
        for k, sc in enumerate(ch.get("scenes", []), 1):
            where, kind = f"{c}.s{k}", sc.get("type")
            if kind == "code" and str(sc.get("title", "")).endswith(".py"):
                files[sc["title"]] = sc.get("lines") or []
            elif kind == "debug":
                cmd = _command(sc.get("run") or [])
                name = cmd[1] if cmd else sc.get("title") if str(sc.get("title", "")).endswith(".py") else None
                if not name or not re.fullmatch(r"[\w.\-]+\.py", str(name)):
                    continue
                for code_key, rows_key, what in (("code", "run", "the code"), ("fixed", "passes", "the fixed code")):
                    if not sc.get(code_key):
                        continue
                    real = run_file({**files, name: sc[code_key]}, name)
                    if real is None:
                        found.append({"where": where, "problem": f"{what} could not be run here (no sandbox): its output is as Claude wrote it", "corrected": False})
                        break
                    ran += 1
                    shown = sc.get(rows_key) or []
                    if not same(shown, real):
                        differs(where, what, shown, real)
                        ok = rows_key == "passes" and not any(r["t"] == "err" for r in real)
                        sc[rows_key] = ([r for r in shown if r.get("t") == "in"][:1] + [{**r, "t": "ok"} if ok else r for r in real])[:MAX_ROWS]
                    if rows_key == "passes" and any(r["t"] == "err" for r in real):
                        found.append({"where": where, "problem": f"the FIXED code still fails when run with {version}: {real[-1]['text'][:160]}", "corrected": False})
                files[name] = sc.get("fixed") or sc.get("code") or []
            elif kind == "terminal":
                rows = sc.get("lines") or []
                if any(re.match(r"\s*>>> ", str(r.get("text", ""))) for r in rows):
                    ins = [re.sub(r"^\s*(>>>|\.\.\.) ?", "", str(r["text"])) for r in rows if re.match(r"\s*(>>>|\.\.\.)( |$)", str(r.get("text", "")))]
                    outs = run_session(ins)
                    if outs is None or len(outs) != len(ins):
                        found.append({"where": where, "problem": "this interpreter session could not be run here: its output is as Claude wrote it", "corrected": False})
                        continue
                    ran += 1
                    real, n = [], 0
                    for r in rows:
                        if re.match(r"\s*(>>>|\.\.\.)( |$)", str(r.get("text", ""))):
                            real.append({"t": "in", "text": r["text"]})
                            real += [{"t": "err" if re.search(r"Error|Traceback|Exception|^\s+File ", ln) else "out", "text": ln[:200]} for ln in outs[n]]
                            n += 1
                    if not same(rows, real):
                        differs(where, "this session", rows, real)
                        sc["lines"] = real[:MAX_ROWS + len(ins)]
                elif (cmd := _command(rows)) and cmd[1] in files:
                    real = run_file(files, cmd[1])
                    if real is None:
                        found.append({"where": where, "problem": f"{cmd[1]} could not be run here (no sandbox): its output is as Claude wrote it", "corrected": False})
                        continue
                    ran += 1
                    shown = rows[cmd[0] + 1:]
                    if not same(shown, real):
                        differs(where, cmd[1], shown, real)
                        sc["lines"] = (rows[: cmd[0] + 1] + real)[:MAX_ROWS]
    script["ran"] = {"runs": ran, "with": version}
    return found
