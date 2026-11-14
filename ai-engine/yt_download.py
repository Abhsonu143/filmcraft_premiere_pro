"""yt_download: a YouTube playlist (or one video) downloaded in the highest quality, or only its audio.

    python yt_download.py "https://youtube.com/playlist?list=…" --out-dir ~/Music/bgm
    python yt_download.py URL --audio-only --audio-format original
    python yt_download.py URL --items 1-20

  - Video: the best video and the best audio there are, merged (mp4 by default; mkv keeps any codec).
  - Audio only: the best audio stream, kept as it is ("original", no quality lost) or converted
    (mp3 320k, m4a, flac, wav).
  - A playlist goes into a folder named after it (`<out>/<playlist name>/001 - <title>.<ext>`); one
    video goes straight into the output folder.
  - Running it again downloads only what is new or missing (`.downloaded.txt` in the folder), so a
    stopped download continues and a playlist can be kept in sync. Unavailable videos (deleted,
    private, blocked here) are skipped and listed; the rest still download.
  - Uses EI's own yt-dlp (kept current with pip) and ffmpeg.

For your own use: respect the creators' licences (NCS music, for example, needs a credit when used
in a video).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import autocut as ac
from autocut import AutocutError, log

AUDIO_FORMATS = ("original", "mp3", "m4a", "flac", "wav", "opus")
CONTAINERS = ("mp4", "mkv")
ITEM_RE = re.compile(r"\[download\] Downloading (?:item|video) (\d+) of (\d+)")
PCT_RE = re.compile(r"\[download\]\s+([\d.]+)% of")
FILE_RE = re.compile(r"^EI_FILE (.+)$")


def js_runtime() -> list[str]:
    """YouTube needs a JavaScript runtime to unlock every format (without one, yt-dlp may miss the
    best ones): deno, node or bun, wherever it is installed (apps started from the Finder do not
    see nvm's PATH)."""
    import shutil

    for name in ("deno", "node", "bun"):
        found = shutil.which(name)
        if not found:
            cands = sorted(Path.home().glob(f".nvm/versions/node/*/bin/{name}"), reverse=True) if name == "node" else []
            cands += [Path(p) for p in (f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}", str(Path.home() / f".{name}/bin/{name}"))]
            found = next((str(c) for c in cands if Path(c).is_file()), None)
        if found:
            return ["--js-runtimes", f"{name}:{found}"]
    return []


def is_playlist(url: str) -> bool:
    """A playlist link (…/playlist?list=…, or a video opened from a playlist)."""
    return "list=" in url


def build_cmd(url: str, out_dir: Path, *, audio_only: bool, audio_format: str = "original", container: str = "mp4",
              items: str | None = None) -> list[str]:
    import script_writer as sw

    if audio_format not in AUDIO_FORMATS:
        raise AutocutError(f"audio format must be one of {', '.join(AUDIO_FORMATS)}")
    if container not in CONTAINERS:
        raise AutocutError(f"video container must be one of {', '.join(CONTAINERS)}")
    if items and not re.fullmatch(r"[\d,\-: ]+", items):
        raise AutocutError("items must look like 1-20 or 1,3,5")
    # a playlist: a folder named after it, files numbered in playlist order; one video: no folder
    playlist = is_playlist(url)
    template = str(out_dir / ("%(playlist_title)s/%(playlist_index)03d - %(title)s.%(ext)s" if playlist else "%(title)s.%(ext)s"))
    cmd = sw.yt_dlp_cmd() + js_runtime() + [
        "--ignore-errors", "--no-overwrites", "--newline", "--no-colors", "--windows-filenames",
        "--download-archive", str(out_dir / ".downloaded.txt"),
        "--yes-playlist" if playlist else "--no-playlist",
        "-o", template, "--print", "after_move:EI_FILE %(filepath)s",
        "--embed-metadata",
    ]
    if items:
        cmd += ["--playlist-items", items.replace(" ", "")]
    if audio_only:
        cmd += ["-f", "bestaudio/best", "-x"]
        if audio_format == "original":
            cmd += ["--audio-quality", "0"]  # (keeps the stream as it is: no re-encode)
        else:
            cmd += ["--audio-format", audio_format, "--audio-quality", "0"]
            if audio_format in ("mp3", "m4a", "flac", "opus"):
                cmd += ["--embed-thumbnail"]
    else:
        # the best video and audio, whatever their resolution; merged into the chosen container
        cmd += ["-f", "bv*+ba/b", "--merge-output-format", container]
        if container == "mp4":
            cmd += ["--embed-thumbnail"]
    return cmd + [url]


def download(url: str, out_dir: str, *, audio_only: bool = False, audio_format: str = "original", container: str = "mp4",
             items: str | None = None) -> dict:
    if not re.match(r"https?://(www\.|m\.|music\.)?(youtube\.com|youtu\.be)/", url.strip()):
        raise AutocutError("give a YouTube link (a playlist or a video)")
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(url.strip(), out, audio_only=audio_only, audio_format=audio_format, container=container, items=items)
    log(f"downloading {'audio only' if audio_only else 'video'} ({audio_format if audio_only else container}, highest quality) into {out}")
    t0 = time.time()
    files: list[str] = []
    skipped: list[str] = []
    if not js_runtime():
        log("no JavaScript runtime found (deno or node): YouTube may hide some of the best formats; installing deno (brew install deno) fixes it")
    k, n = 0, 1
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        m = ITEM_RE.search(line)
        if m:
            k, n = int(m.group(1)), int(m.group(2))
            log(f"progress {k - 1}/{n} items")
            continue
        m = PCT_RE.search(line)
        if m:
            try:
                ac.stage_progress(min(1.0, ((k - 1 if k else 0) + float(m.group(1)) / 100) / max(1, n)), f"item {max(1, k)} of {n}")
            except ValueError:
                pass
            continue
        m = FILE_RE.match(line)
        if m:
            files.append(m.group(1))
            log(f"saved {Path(m.group(1)).name}")
            continue
        if line.startswith("ERROR:"):
            skipped.append(line[6:].strip()[:160])
            log(f"skipped: {line[6:].strip()[:160]}")
        elif "has already been recorded in the archive" in line:
            pass  # (downloaded before)
        elif line and not line.startswith("[download]"):
            log(line[:200])
    code = proc.wait()
    if code != 0 and not files and not skipped:
        raise AutocutError("yt-dlp could not download this link (see the log)")
    folder = str(Path(files[0]).parent) if files else str(out)
    log(f"done in {time.time() - t0:.0f}s: {len(files)} new file(s)" + (f", {len(skipped)} unavailable" if skipped else "")
        + " (already downloaded ones are skipped)")
    (Path(folder) / "skipped.txt").write_text("\n".join(skipped) + "\n", encoding="utf-8") if skipped else None
    return {"outputs": files, "out_dir": folder, "downloaded": len(files), "skipped": skipped}


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("url", help="a YouTube playlist or video link")
    p.add_argument("--out-dir", default=str(Path.home() / "Movies" / "EI" / "Downloads"))
    p.add_argument("--audio-only", action="store_true", help="only the audio")
    p.add_argument("--audio-format", default="original", choices=list(AUDIO_FORMATS), help="original: the best stream as it is (no re-encode)")
    p.add_argument("--container", default="mp4", choices=list(CONTAINERS), help="video: mp4 (plays everywhere) or mkv (keeps any codec)")
    p.add_argument("--items", help="only these playlist items, e.g. 1-20 or 1,4,9")


def run_args(a) -> dict:
    return download(a.url, a.out_dir, audio_only=a.audio_only, audio_format=a.audio_format, container=a.container, items=a.items)


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
    sys.exit(main())
