#!/usr/bin/env python3
"""extract_audio: save the sound of videos as audio files.

    python extract_audio.py talk.mp4 --format mp3 --quality high --out-dir audio/
    python extract_audio.py a.mov b.mp4 --format wav --start 12 --end 95.5
    python extract_audio.py clip.mp4 --format m4a --quality original     # no re-encode when possible

Formats: mp3, m4a (AAC), wav, aiff, flac, alac (Apple Lossless, .m4a), opus, ogg (Vorbis).
Quality: original (copy the audio untouched when the format allows, else best), best, high,
medium, small. Lossless formats (wav, aiff, flac, alac) are always full quality.
Options: a time range, mono, a sample rate, loudness normalisation (-14 LUFS), which audio track.
The source is never changed; an existing output gets " 2", " 3" … instead of being overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import autocut as ac
from autocut import AutocutError, log

FORMATS = {
    # name: (extension, encoder, lossless, codecs a stream copy accepts)
    "mp3": (".mp3", "libmp3lame", False, {"mp3"}),
    "m4a": (".m4a", "aac_at", False, {"aac"}),
    "wav": (".wav", "pcm_s16le", True, {"pcm_s16le"}),
    "aiff": (".aiff", "pcm_s16be", True, {"pcm_s16be"}),
    "flac": (".flac", "flac", True, {"flac"}),
    "alac": (".m4a", "alac", True, {"alac"}),
    "opus": (".opus", "libopus", False, {"opus"}),
    "ogg": (".ogg", "libvorbis", False, {"vorbis"}),
}
BITRATES = {  # kbit/s for lossy formats
    "mp3": {"best": 320, "high": 256, "medium": 192, "small": 128},
    "m4a": {"best": 320, "high": 256, "medium": 192, "small": 128},
    "opus": {"best": 256, "high": 160, "medium": 128, "small": 96},
    "ogg": {"best": 320, "high": 256, "medium": 192, "small": 128},
}


def audio_streams(path: str) -> list[dict]:
    proc = ac.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index,codec_name,sample_rate,channels,bit_rate:stream_tags=language,title", "-of", "json", path])
    return json.loads(proc.stdout or "{}").get("streams", [])


def free_name(folder: Path, stem: str, ext: str) -> Path:
    p = folder / f"{stem}{ext}"
    i = 2
    while p.exists():
        p = folder / f"{stem} {i}{ext}"
        i += 1
    return p


def build_command(src: str, dst: Path, fmt: str, quality: str, stream: dict, *, start: float | None, end: float | None,
                  mono: bool, rate: int | None, normalize: bool, track: int) -> list[str]:
    ext, enc, lossless, copyable = FORMATS[fmt]
    filters = []
    if normalize:
        filters.append("loudnorm=I=-14:TP=-1.5:LRA=11")
    copy = quality == "original" and not filters and not mono and not rate and stream.get("codec_name") in copyable
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", src, "-vn", "-sn", "-dn", "-map", f"0:a:{track}", "-map_metadata", "0"]
    if copy:
        cmd += ["-c:a", "copy"]
    else:
        if fmt == "wav" and quality in ("best", "original"):
            enc = "pcm_s24le" if (stream.get("codec_name") or "").startswith("pcm_s24") else enc
        cmd += ["-c:a", enc]
        if not lossless:
            q = "best" if quality == "original" else quality
            cmd += ["-b:a", f"{BITRATES[fmt][q]}k"]
        if filters:
            cmd += ["-af", ",".join(filters)]
        if mono:
            cmd += ["-ac", "1"]
        out_rate = rate or (48000 if fmt == "opus" else None)
        if normalize and not out_rate:
            out_rate = int(stream.get("sample_rate") or 48000)  # loudnorm resamples internally; keep the rate
        if fmt == "opus" and out_rate not in (8000, 12000, 16000, 24000, 48000):
            out_rate = 48000
        if out_rate:
            cmd += ["-ar", str(out_rate)]
    if fmt in ("m4a", "alac"):
        cmd += ["-movflags", "+faststart"]
    return cmd + [str(dst)]


def extract(inputs: list[str], out_dir: str | None, fmt: str = "mp3", quality: str = "high", *, start: float | None = None,
            end: float | None = None, mono: bool = False, rate: int | None = None, normalize: bool = False, track: int = 0) -> list[str]:
    ac.require_tools()
    if fmt not in FORMATS:
        raise AutocutError(f"unknown format {fmt!r} ({', '.join(FORMATS)})")
    if quality not in ("original", "best", "high", "medium", "small"):
        raise AutocutError(f"unknown quality {quality!r}")
    if start is not None and end is not None and end <= start:
        raise AutocutError("the end must be after the start")
    if rate is not None and not (8000 <= rate <= 192000):
        raise AutocutError("sample rate must be between 8000 and 192000 Hz")
    import voiceover_cut as vc

    files = []
    for it in inputs:  # folders: every video in them
        files += vc.collect_clips([it]) if Path(it).is_dir() else [it]
    if not files:
        raise AutocutError("no video files given")
    outs = []
    for src in files:
        src = str(Path(src).resolve())
        info = ac.probe(src)
        streams = audio_streams(src)
        if not streams:
            raise AutocutError(f"{Path(src).name} has no sound")
        if not (0 <= track < len(streams)):
            raise AutocutError(f"{Path(src).name} has {len(streams)} audio track(s); track {track + 1} does not exist")
        if start is not None and start >= info.duration:
            raise AutocutError(f"{Path(src).name} is only {info.duration:.1f}s long")
        folder = Path(out_dir).resolve() if out_dir else Path(src).parent
        folder.mkdir(parents=True, exist_ok=True)
        ext = FORMATS[fmt][0]
        dst = free_name(folder, Path(src).stem, ext)
        if dst.resolve() == Path(src).resolve():
            raise AutocutError("refusing to overwrite the source")
        tmp = dst.with_name(dst.stem + ".part" + ext)
        ac.run(build_command(src, tmp, fmt, quality, streams[track], start=start, end=end, mono=mono, rate=rate, normalize=normalize, track=track))
        tmp.replace(dst)
        log(f"saved {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")
        outs.append(str(dst))
    return outs


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("inputs", nargs="+", help="video (or audio) files")
    p.add_argument("--out-dir", help="folder for the audio files (default: next to each video)")
    p.add_argument("--format", default="mp3", choices=sorted(FORMATS))
    p.add_argument("--quality", default="high", choices=["original", "best", "high", "medium", "small"])
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.add_argument("--mono", action="store_true")
    p.add_argument("--rate", type=int, help="sample rate in Hz (default: keep)")
    p.add_argument("--normalize", action="store_true", help="loudness to -14 LUFS (streaming level)")
    p.add_argument("--track", type=int, default=1, help="audio track number (1 = first)")


def run_args(a) -> dict:
    outs = extract(a.inputs, a.out_dir, a.format, a.quality, start=a.start, end=a.end, mono=a.mono, rate=a.rate, normalize=a.normalize, track=a.track - 1)
    return {"outputs": outs, "out_dir": str(Path(outs[0]).parent) if outs else ""}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    try:
        print(json.dumps(run_args(ap.parse_args(argv))))
    except AutocutError as e:
        log(f"error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
