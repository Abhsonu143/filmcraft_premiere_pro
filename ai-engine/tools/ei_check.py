#!/usr/bin/env python3
"""ei_check: is an EI AI job healthy, slow, stuck, or starved of memory?

    python3 ei_check.py            report once
    python3 ei_check.py --watch 30 report every 30 s (Ctrl+C to stop)
    python3 ei_check.py --clean    also stop leftover EI model workers (never a running job)

What it looks at (read-only unless --clean):
  - Memory: free RAM, swap in use, memory pressure. A voice model needs ~3.4 GB; with heavy swap every
    step slows down several times.
  - Each running AI job (an `ei_engine.py` process) with its process group: model workers, ffmpeg,
    their memory and CPU.
  - Work in progress: voice takes and rendered shots being saved under `.ei-work/` folders in
    ~/Movies/EI, how fast they come and when the last one was saved (a job that saved nothing for
    10 minutes is probably stuck).
  - Two jobs in a model-heavy step at the same time (two voice models do not fit in 8 GB).
  - Leftovers: EI model workers whose job is gone (parent = launchd), still holding memory. --clean
    stops them; it never touches a process that belongs to a running job.
  - The heaviest other apps (closing one frees memory for the voice model).

Only the Python standard library and macOS tools (ps, sysctl, vm_stat, footprint) are used.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

MOVIES = Path.home() / "Movies" / "EI"
STUCK_AFTER = 600  # seconds without a saved take/shot while a job runs


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def processes() -> list[dict]:
    out = []
    for line in sh(["ps", "-axww", "-o", "pid=,ppid=,pgid=,pcpu=,rss=,etime=,command="]).splitlines():
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        pid, ppid, pgid, cpu, rss, etime, cmd = parts
        try:
            out.append({"pid": int(pid), "ppid": int(ppid), "pgid": int(pgid), "cpu": float(cpu), "rss_mb": int(rss) / 1024,
                        "etime": etime, "cmd": cmd})
        except ValueError:
            continue
    return out


def footprint_mb(pid: int) -> float | None:
    """Real memory use (includes what is swapped out; RSS hides it)."""
    m = re.search(r"Footprint:\s*([\d.]+)\s*(KB|MB|GB)", sh(["footprint", "-p", str(pid)]))
    if not m:
        return None
    return float(m.group(1)) * {"KB": 1 / 1024, "MB": 1.0, "GB": 1024.0}[m.group(2)]


def memory() -> dict:
    total = int(sh(["sysctl", "-n", "hw.memsize"]).strip() or 0) / 2**30
    swap = sh(["sysctl", "-n", "vm.swapusage"])
    used = re.search(r"used = ([\d.]+)M", swap)
    free_pct = re.search(r"free percentage: (\d+)%", sh(["memory_pressure"]))
    return {"ram_gb": total, "swap_gb": float(used.group(1)) / 1024 if used else 0.0, "free_pct": int(free_pct.group(1)) if free_pct else None}


def describe(cmd: str) -> str:
    """The job's command, readable: `topic gta 6 … --voice x` -> `topic: gta 6 …`."""
    rest = cmd.split("ei_engine.py", 1)[-1].strip()
    sub = rest.split(" ", 1)[0]
    what = rest[len(sub):].split(" --", 1)[0].strip()
    return f"{sub}: {what[:70]}" if what else sub


def role(cmd: str) -> str:
    if "ei_engine.py" in cmd:
        return "engine"
    if "resource_tracker" in cmd:
        return "helper"
    if "multiprocessing" in cmd:
        return "model worker"
    if "ffmpeg" in cmd:
        return "ffmpeg"
    return Path(cmd.split(" ", 1)[0]).name[:20]


def work_activity(since_s: float = 7200) -> list[dict]:
    """Folders with recent `.ei-work` activity: voice takes and rendered shots."""
    out = []
    now = time.time()
    if not MOVIES.is_dir():
        return out
    for work in MOVIES.rglob(".ei-work"):
        for job in [p for p in work.iterdir() if p.is_dir()]:
            files = [f for f in job.rglob("*") if f.is_file() and f.suffix in (".wav", ".mp4")]
            if not files:
                continue
            times = sorted(f.stat().st_mtime for f in files)
            if now - times[-1] > since_s:
                continue
            gaps = [b - a for a, b in zip(times, times[1:])][-15:]
            kind = "voice takes" if any(f.suffix == ".wav" for f in files) else "rendered shots"
            out.append({"folder": str(work.parent), "job": job.name, "kind": kind, "count": len(files), "last_ago": now - times[-1],
                        "per_item": sorted(gaps)[len(gaps) // 2] if gaps else None})
    return out


def report(clean: bool) -> None:
    procs = processes()
    by_pid = {p["pid"]: p for p in procs}
    mem = memory()
    print(time.strftime("== EI check %H:%M:%S"))
    warn = []
    free = f", {mem['free_pct']}% free" if mem["free_pct"] is not None else ""
    print(f"memory: {mem['ram_gb']:.0f} GB RAM{free}, swap in use {mem['swap_gb']:.1f} GB")
    if mem["swap_gb"] > 4 or (mem["free_pct"] is not None and mem["free_pct"] < 15):
        warn.append("memory is very tight (heavy swap): AI steps run several times slower; close heavy apps or run one job at a time")

    ei_app = [p for p in procs if p["cmd"].startswith("/Applications/EI.app/Contents/MacOS/")]
    print(f"EI app: {'running (pid ' + str(ei_app[0]['pid']) + ')' if ei_app else 'not running'}")

    engines = [p for p in procs if "ei_engine.py" in p["cmd"] and "multiprocessing" not in p["cmd"]]
    groups = {e["pgid"] for e in engines}
    heavy_jobs = 0
    if not engines:
        print("AI jobs: none running")
    for e in engines:
        members = [p for p in procs if p["pgid"] == e["pgid"]]
        print(f"\njob {e['pid']} (running {e['etime']}): {describe(e['cmd'])}")
        model_mb = 0.0
        for m in members:
            fp = footprint_mb(m["pid"]) if role(m["cmd"]) in ("engine", "model worker") else None
            mb = fp if fp is not None else m["rss_mb"]
            if role(m["cmd"]) == "model worker":
                model_mb += mb
            print(f"   {role(m['cmd']):13s} pid {m['pid']:<6} cpu {m['cpu']:5.1f}%  memory {mb:7.0f} MB  up {m['etime']}")
        if model_mb > 1500:
            heavy_jobs += 1
    if heavy_jobs > 1:
        warn.append(f"{heavy_jobs} jobs hold a large model at the same time (two voice models do not fit in 8 GB): pause one in AI ▸ AI Jobs")

    acts = work_activity()
    if acts:
        print("\nwork in progress:")
    for a in acts:
        rate = f", one every {a['per_item']:.0f} s" if a["per_item"] else ""
        print(f"   {Path(a['folder']).name[:60]}: {a['count']} {a['kind']}{rate}, last saved {a['last_ago']:.0f} s ago")
        if engines and a["last_ago"] > STUCK_AFTER:
            warn.append(f"nothing saved for {a['last_ago'] / 60:.0f} min in {Path(a['folder']).name[:50]}: the job may be stuck (see its log in AI Jobs; Pause/Resume, or Cancel and Retry: finished takes are kept)")

    # leftovers: Python model workers / helpers whose parent is gone and that belong to no running job
    leftovers = [p for p in procs if "multiprocessing" in p["cmd"] and "Python" in p["cmd"] and p["ppid"] == 1 and p["pgid"] not in groups]
    stray_ffmpeg = [p for p in procs if Path(p["cmd"].split(" ", 1)[0]).name == "ffmpeg" and p["ppid"] == 1 and p["pgid"] not in groups]
    if leftovers or stray_ffmpeg:
        print("\nleftovers (their job is gone):")
    for p in leftovers + stray_ffmpeg:
        fp = footprint_mb(p["pid"])
        print(f"   {role(p['cmd']):13s} pid {p['pid']:<6} memory {fp if fp is not None else p['rss_mb']:7.0f} MB  up {p['etime']}")
    if leftovers and not clean:
        warn.append(f"{len(leftovers)} leftover model process(es) hold memory: run with --clean to stop them")
    if clean:
        for p in leftovers:
            try:
                os.kill(p["pid"], signal.SIGTERM)
                print(f"   stopped leftover {p['pid']}")
            except OSError as e:
                print(f"   could not stop {p['pid']}: {e}")
    if stray_ffmpeg:
        warn.append("an ffmpeg without a parent is running: if it is not yours, stop it in Activity Monitor")

    others = sorted((p for p in procs if p["pgid"] not in groups and p not in leftovers and "/Applications/EI.app" not in p["cmd"]),
                    key=lambda p: -p["rss_mb"])[:5]
    print("\nheaviest other apps: " + ", ".join(f"{Path(p['cmd'].split(' ', 1)[0]).name[:24]} {p['rss_mb']:.0f} MB" for p in others))
    print()
    for w in warn:
        print(f"⚠️  {w}")
    if not warn:
        print("✔ nothing looks wrong")
    _ = by_pid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", type=float, help="repeat every N seconds")
    ap.add_argument("--clean", action="store_true", help="stop leftover EI model workers (never a running job)")
    a = ap.parse_args()
    try:
        while True:
            report(a.clean)
            if not a.watch:
                return 0
            time.sleep(max(5.0, a.watch))
            print()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
