#!/usr/bin/env python3
"""split_video: cut a long video into clips of a fixed length.

    python split_video.py talk.mp4 --out-dir clips/ --clip-len 60
    python split_video.py talk.mp4 --out-dir clips/ --clip-len 30 --quality exact

Quality modes:
  same    (default) original quality, nothing re-encoded. Fast. Each cut lands on the first
          keyframe at or after the boundary, so clips can be a little longer than --clip-len
          (by up to one keyframe interval: usually under 2 s for phone/camera footage).
  exact   frame-exact lengths. Re-encodes with the hardware encoder at the source's own
          bitrate, so quality stays very close to the original.
  high | medium | small   re-encode at a chosen quality (exact lengths, smaller files).

The original file is never modified. Clips are named <name>_part001.<ext>, …
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import autocut as ac
from autocut import AutocutError, log

QUALITY_BITRATE = {"high": "16M", "medium": "8M", "small": "4M"}
COPY_EXT = {".mp4", ".mov", ".m4v", ".mkv"}


def segment_count(duration: float, clip_len: float, min_last: float = 1.0) -> int:
    """Number of clips. A tail shorter than `min_last` seconds is folded into the previous clip."""
    if duration <= 0 or clip_len <= 0:
        return 0
    n = max(1, math.ceil(duration / clip_len))
    if n > 1 and duration - (n - 1) * clip_len < min_last:
        n -= 1
    return n


def split_points(duration: float, clip_len: float, min_last: float = 1.0) -> list[float]:
    """Cut times (excluding 0 and the end)."""
    n = segment_count(duration, clip_len, min_last)
    return [clip_len * i for i in range(1, n)]


def source_bitrate(path: str) -> int:
    """Video bitrate in bit/s (stream value, else estimated from the file size)."""
    proc = ac.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=bit_rate:format=bit_rate,size,duration", "-of", "json", path])
    data = json.loads(proc.stdout or "{}")
    for s in data.get("streams", []):
        try:
            if int(s.get("bit_rate", 0)) > 0:
                return int(s["bit_rate"])
        except ValueError:
            pass
    fmt = data.get("format", {})
    try:
        return max(1_000_000, int(fmt.get("bit_rate", 0)))
    except ValueError:
        return 8_000_000


def split(src: str, out_dir: str, clip_len: float = 60.0, quality: str = "same", codec: str = "h264", min_last: float = 1.0) -> list[str]:
    ac.require_tools()
    src = str(Path(src).resolve())
    info = ac.probe(src)
    if not info.has_video and not info.has_audio:
        raise AutocutError("no audio or video stream in the file")
    if clip_len < 1:
        raise AutocutError("clip length must be at least 1 second")
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    ext = Path(src).suffix.lower() if quality == "same" and Path(src).suffix.lower() in COPY_EXT else ".mp4"
    pattern = str(out / f"{stem}_part%03d{ext}")
    points = split_points(info.duration, clip_len, min_last)
    n = len(points) + 1
    log(f"splitting {info.duration:.1f}s into {n} clips of ~{clip_len:g}s ({quality})")
    for old in out.glob(f"{stem}_part[0-9][0-9][0-9]{ext}"):
        old.unlink()  # earlier run with other settings: never mix stale parts in
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    times = ",".join(f"{t:.3f}" for t in points)
    if quality == "same":
        cmd += ["-i", src, "-map", "0:v:0?", "-map", "0:a?", "-c", "copy"]
    else:
        bitrate = str(source_bitrate(src)) if quality == "exact" else QUALITY_BITRATE.get(quality)
        if not bitrate:
            raise AutocutError(f"unknown quality {quality!r}")
        cmd += ["-hwaccel", "videotoolbox", "-i", src, "-map", "0:v:0?", "-map", "0:a?"]
        cmd += ac.video_encoder_args(codec, bitrate) + ["-c:a", "aac", "-b:a", "192k"]
        if points:
            cmd += ["-force_key_frames", times]
    cmd += ["-f", "segment", "-reset_timestamps", "1", "-segment_start_number", "1"]
    if points:
        cmd += ["-segment_times", times]
    else:
        cmd += ["-segment_time", str(max(info.duration, clip_len) + 1)]
    cmd += ["-segment_format", "mov" if ext in (".mov", ".m4v") else ("matroska" if ext == ".mkv" else "mp4"), pattern]
    t0 = time.time()
    ac.run(cmd)
    files = sorted(str(p) for p in out.glob(f"{stem}_part[0-9][0-9][0-9]{ext}"))
    log(f"wrote {len(files)} clips to {out} in {time.time() - t0:.1f}s")
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--out-dir", help="output folder (default: <name>_clips next to the video)")
    ap.add_argument("--clip-len", type=float, default=60.0, help="seconds per clip (default 60)")
    ap.add_argument("--quality", choices=["same", "exact", "high", "medium", "small"], default="same")
    ap.add_argument("--codec", choices=["h264", "hevc"], default="h264", help="encoder when re-encoding")
    args = ap.parse_args(argv)
    out_dir = args.out_dir or str(Path(args.input).with_name(Path(args.input).stem + "_clips"))
    try:
        files = split(args.input, out_dir, args.clip_len, args.quality, args.codec)
    except AutocutError as e:
        log(f"error: {e}")
        return 1
    print(json.dumps({"outputs": files, "out_dir": str(Path(out_dir).resolve())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
