#!/usr/bin/env python3
"""Time each stage of a command and record peak memory while it runs.

    python scripts/bench.py -- python autocut.py plan big.mp4 --captions

Peak memory is reported two ways:
  tree RSS   - resident memory of the command and all its child processes (ffmpeg, model workers)
  system     - how much the Mac's total used memory rose above the starting point. This also
               catches GPU (Metal/MLX) buffers, which live in unified memory but not always in RSS.
Stage timings are read from the "[autocut] ... in N s" log lines the tools print.
"""
import subprocess
import sys
import threading
import time

import psutil


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    base_used = psutil.virtual_memory().total - psutil.virtual_memory().available
    peak = {"tree": 0, "sys": 0, "swap0": psutil.swap_memory().used, "swap": 0}
    t0 = time.time()
    proc = subprocess.Popen(argv, stderr=subprocess.PIPE, text=True)
    done = threading.Event()

    def sample():
        p = psutil.Process(proc.pid)
        while not done.is_set():
            try:
                procs = [p] + p.children(recursive=True)
                rss = 0
                for q in procs:
                    try:
                        rss += q.memory_info().rss
                    except psutil.Error:
                        pass
                peak["tree"] = max(peak["tree"], rss)
            except psutil.Error:
                pass
            vm = psutil.virtual_memory()
            peak["sys"] = max(peak["sys"], vm.total - vm.available - base_used)
            peak["swap"] = max(peak["swap"], psutil.swap_memory().used - peak["swap0"])
            time.sleep(0.2)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    for line in proc.stderr:
        if line.startswith("[autocut]"):
            print(f"  {time.time() - t0:7.1f}s  {line.rstrip()}", flush=True)
    rc = proc.wait()
    done.set()
    th.join()
    mb = 1024 * 1024
    print(f"exit={rc} wall={time.time() - t0:.1f}s peak_tree_rss={peak['tree'] / mb:.0f}MB "
          f"peak_system_increase={peak['sys'] / mb:.0f}MB swap_increase={peak['swap'] / mb:.0f}MB")
    return rc


if __name__ == "__main__":
    sys.exit(main())
