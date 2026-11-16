"""yt_download: a YouTube playlist (or one video) downloaded in the highest quality, or only its audio.

    python yt_download.py "https://youtube.com/playlist?list=…" --out-dir ~/Music/bgm
    python yt_download.py URL --audio-only --audio-format original
    python yt_download.py URL --items 1-20

  - Video: the best video and the best audio there are, merged (mp4 by default; mkv keeps any codec).
  - Audio only: the best audio stream, kept as it is ("original", no quality lost) or converted
    (mp3 320k, m4a, flac, wav).
  - A playlist goes into a folder named after it (`<out>/<playlist name>/001 - <title>.<ext>`); one
    video goes straight into the output folder.
  - Several videos at once (`--parallel N`, 1 to 8; 3 is a good default on an 8 GB Mac: downloads
    wait on the network, not the CPU, and each one takes about 100 MB). Converting (mp3, m4a, flac,
    wav) and merging video use ffmpeg, so those take some CPU per download.
  - Running it again skips what is already in the folder (the same file name, or a file of this
    video recorded in `.downloaded.txt`), so a stopped download continues and a playlist can be kept
    in sync. `--redo` downloads everything again and replaces the files. Unavailable videos
    (deleted, private, blocked here) are skipped and listed; the rest still download.
  - Uses EI's own yt-dlp (kept current with pip) and ffmpeg.

Subtitles only (`--subs --sub-langs en hi`): the captions of every video, cleaned into text you can
use, for example to learn a creator's talking style for your own scripts:
  - real captions only: the creator's own subtitles, or YouTube's automatic captions in the video's
    own language; never a machine translation;
  - cleaned: YouTube's rolling duplicates, [Music]/[Applause] tags, >> markers and stray symbols
    removed; automatic captions (no punctuation, some misheard words) are polished by Claude
    ("--polish claude": punctuation, sentences, obvious mishearings fixed, the speaker's own words,
    slang and Hinglish kept) or by rules (pauses become sentence ends);
  - per video `<NNN - title>.<lang>.txt` (clean text) and `.srt` (clean timed captions), plus one
    `all.<lang>.txt` for the playlist;
  - `--learn-style NAME` also learns the talking style from them (AI ▸ Write Voiceover Script ▸ Style).

For your own use: respect the creators' licences (NCS music, for example, needs a credit when used
in a video).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import autocut as ac
from autocut import AutocutError, log

AUDIO_FORMATS = ("original", "mp3", "m4a", "flac", "wav", "opus")
CONTAINERS = ("mp4", "mkv")
ITEM_RE = re.compile(r"\[download\] Downloading (?:item|video) (\d+) of (\d+)")
PCT_RE = re.compile(r"\[download\]\s+([\d.]+)% of")
FILE_RE = re.compile(r"^EI_FILE (.+)$")
MAX_PARALLEL = 8
AUDIO_EXTS = (".opus", ".m4a", ".mp3", ".flac", ".wav", ".ogg", ".aac", ".webm")
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov")
VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
ARCHIVE = ".downloaded.txt"


def clamp_parallel(n) -> int:
    try:
        return max(1, min(MAX_PARALLEL, int(n)))
    except (TypeError, ValueError):
        return 1


def js_runtime() -> list[str]:
    import script_writer as sw

    return sw.js_runtime()


def is_playlist(url: str) -> bool:
    """A playlist or a channel (…/playlist?list=…, a video opened from a playlist, youtube.com/@name,
    /channel/…, /c/…, a channel's /videos or /shorts tab)."""
    return "list=" in url or bool(re.search(r"youtube\.com/(@[^/?]+|channel/[^/?]+|c/[^/?]+|user/[^/?]+)(/(videos|shorts|streams))?/?($|\?)", url))


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
    cmd = sw.yt_dlp_cmd() + [
        "--ignore-errors", "--no-overwrites", "--newline", "--no-colors", "--windows-filenames",
        "--download-archive", str(out_dir / ".downloaded.txt"),
        "--yes-playlist" if playlist else "--no-playlist",
        "-o", template, "--print", "after_move:EI_FILE %(filepath)s",
        "--embed-metadata",
    ]
    if items:
        cmd += ["--playlist-items", items.replace(" ", "")]
    return cmd + media_args(audio_only=audio_only, audio_format=audio_format, container=container) + [url]


def media_args(*, audio_only: bool, audio_format: str, container: str) -> list[str]:
    if audio_only:
        cmd = ["-f", "bestaudio/best", "-x"]
        if audio_format == "original":
            return cmd + ["--audio-quality", "0"]  # (keeps the stream as it is: no re-encode)
        cmd += ["--audio-format", audio_format, "--audio-quality", "0"]
        return cmd + (["--embed-thumbnail"] if audio_format in ("mp3", "m4a", "flac", "opus") else [])
    # the best video and audio, whatever their resolution; merged into the chosen container
    return ["-f", "bv*+ba/b", "--merge-output-format", container] + (["--embed-thumbnail"] if container == "mp4" else [])


def plan_items(url: str, out: Path, items: str | None) -> list[dict] | None:
    """The videos of a link with the exact path yt-dlp would give each (without its extension):
    [{id, url, index, stem: Path}]. None when the link is not a plain list of videos (a channel with
    tabs, for example), so the caller lets one yt-dlp handle it."""
    import script_writer as sw

    playlist = is_playlist(url)
    template = "%(playlist_title)s/%(playlist_index)03d - %(title)s.%(ext)s" if playlist else "%(title)s.%(ext)s"
    cmd = sw.yt_dlp_cmd() + ["--flat-playlist", "--ignore-errors", "--windows-filenames", "-o", template,
                             "--print", "%(id)s\t%(playlist_index|0)s\t%(filename)s"]
    if items:
        cmd += ["--playlist-items", items.replace(" ", "")]
    cmd += ["--yes-playlist" if playlist else "--no-playlist", url]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return None
    vids = []
    for line in res.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        vid, idx, name = parts
        if not VIDEO_ID.fullmatch(vid):
            return None  # (an entry that is itself a list)
        stem = name[:-3] if name.endswith(".NA") else str(Path(name).with_suffix(""))
        vids.append({"id": vid, "url": f"https://www.youtube.com/watch?v={vid}", "index": int(idx) if idx.isdigit() else 0,
                     "stem": out / stem})
    return vids or None


def read_archive(out: Path) -> set[str]:
    try:
        return {l.split()[-1] for l in (out / ARCHIVE).read_text(encoding="utf-8").splitlines() if l.strip()}
    except OSError:
        return set()


def wanted_exts(audio_only: bool, audio_format: str, container: str) -> tuple[str, ...]:
    if audio_only:
        return AUDIO_EXTS if audio_format == "original" else (f".{audio_format}",)
    return (f".{container}",)


def present(v: dict, exts: tuple[str, ...], archived: set[str]) -> Path | None:
    """The file of this video already in the folder: the same name, or (for a video recorded as
    downloaded whose title has changed since) the same playlist number."""
    stem: Path = v["stem"]
    for e in exts:
        f = stem.with_name(stem.name + e)
        if f.is_file() and f.stat().st_size > 0:
            return f
    if v["id"] in archived and v["index"] and stem.parent.is_dir():
        for f in stem.parent.glob(f"{v['index']:03d} - *"):
            if f.suffix.lower() in exts and f.stat().st_size > 0:
                return f
    return None


def yt_template(stem: Path) -> str:
    return str(stem).replace("%", "%%") + ".%(ext)s"


def download(url: str, out_dir: str, *, audio_only: bool = False, audio_format: str = "original", container: str = "mp4",
             items: str | None = None, parallel: int = 3, redo: bool = False) -> dict:
    if not re.match(r"https?://(www\.|m\.|music\.)?(youtube\.com|youtu\.be)/", url.strip()):
        raise AutocutError("give a YouTube link (a playlist or a video)")
    if audio_format not in AUDIO_FORMATS:
        raise AutocutError(f"audio format must be one of {', '.join(AUDIO_FORMATS)}")
    if container not in CONTAINERS:
        raise AutocutError(f"video container must be one of {', '.join(CONTAINERS)}")
    if items and not re.fullmatch(r"[\d,\-: ]+", items):
        raise AutocutError("items must look like 1-20 or 1,3,5")
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    parallel = clamp_parallel(parallel)
    if not js_runtime():
        log("no JavaScript runtime found (deno or node): YouTube may hide some of the best formats; installing deno (brew install deno) fixes it")
    vids = plan_items(url.strip(), out, items)
    if vids is None:
        if redo:
            log("this link is not a plain list of videos: downloading it as one job (it cannot be redone file by file)")
        return download_one(url, out, audio_only=audio_only, audio_format=audio_format, container=container, items=items)
    return download_parallel(vids, out, audio_only=audio_only, audio_format=audio_format, container=container,
                             parallel=parallel, redo=redo)


def download_parallel(vids: list[dict], out: Path, *, audio_only: bool, audio_format: str, container: str, parallel: int,
                      redo: bool) -> dict:
    import script_writer as sw

    t0 = time.time()
    exts = wanted_exts(audio_only, audio_format, container)
    archived = read_archive(out)
    todo, already = [], []
    for v in vids:
        f = None if redo else present(v, exts, archived)
        (already if f else todo).append(v)
    what = "audio only" if audio_only else "video"
    log(f"downloading {what} ({audio_format if audio_only else container}, highest quality) into {out}: {len(vids)} video(s), "
        + (f"{len(already)} already there (skipped), " if already else "") + (f"{len(todo)} to download, "
        + (f"{min(parallel, len(todo))} at a time" if todo else "")) + (" (redoing all, files are replaced)" if redo else ""))
    base = sw.yt_dlp_cmd() + ["--ignore-errors", "--newline", "--no-colors", "--windows-filenames", "--no-playlist",
                              "--force-overwrites" if redo else "--no-overwrites", "--print", "after_move:EI_FILE %(filepath)s",
                              "--embed-metadata"] + media_args(audio_only=audio_only, audio_format=audio_format, container=container)
    lock = threading.Lock()
    files: list[str] = []
    skipped: list[str] = []
    part: dict[str, float] = {}
    done = [0]

    def report(detail: str) -> None:
        frac = (done[0] + sum(part.values())) / max(1, len(todo))
        ac.stage_progress(min(1.0, frac), detail)

    def one(v: dict) -> None:
        v["stem"].parent.mkdir(parents=True, exist_ok=True)
        cmd = base + ["-o", yt_template(v["stem"]), v["url"]]
        got: list[str] = []
        errors: list[str] = []
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        except OSError as e:
            errors.append(str(e))
            proc = None
        if proc is not None and proc.stdout is not None:
            for line in proc.stdout:
                line = line.rstrip()
                m = PCT_RE.search(line)
                if m:
                    try:
                        pct = float(m.group(1)) / 100
                    except ValueError:
                        continue
                    with lock:
                        part[v["id"]] = min(0.99, pct)
                        report(f"{done[0]} of {len(todo)} done, {len(part)} downloading")
                    continue
                m = FILE_RE.match(line)
                if m:
                    got.append(m.group(1))
                elif line.startswith("ERROR:"):
                    errors.append(line[6:].strip()[:160])
            proc.wait()
        with lock:
            part.pop(v["id"], None)
            done[0] += 1
            if got:
                files.extend(got)
                try:
                    with open(out / ARCHIVE, "a", encoding="utf-8") as fh:
                        fh.write(f"youtube {v['id']}\n")
                except OSError:
                    pass
                log(f"saved {Path(got[-1]).name}")
            else:
                why = errors[-1] if errors else "nothing was downloaded"
                skipped.append(f"{v['stem'].name}: {why}"[:200])
                log(f"skipped: {v['stem'].name[:80]}: {why}")
            report(f"{done[0]} of {len(todo)} done")

    if todo:
        with ThreadPoolExecutor(max_workers=min(parallel, len(todo))) as pool:
            list(pool.map(one, todo))
    folder = str(vids[0]["stem"].parent) if vids else str(out)
    if skipped:
        Path(folder).mkdir(parents=True, exist_ok=True)
        (Path(folder) / "skipped.txt").write_text("\n".join(skipped) + "\n", encoding="utf-8")
    if todo and not files and len(skipped) == len(todo) and not already:
        raise AutocutError("yt-dlp could not download this link (see the log)")
    log(f"done in {time.time() - t0:.0f}s: {len(files)} new file(s)" + (f", {len(already)} already there" if already else "")
        + (f", {len(skipped)} unavailable (skipped.txt)" if skipped else ""))
    return {"outputs": files, "out_dir": folder, "downloaded": len(files), "already": len(already), "skipped": skipped}


def download_one(url: str, out: Path, *, audio_only: bool, audio_format: str, container: str, items: str | None) -> dict:
    """One yt-dlp for the whole link (a channel, or when the list could not be read first)."""
    cmd = build_cmd(url.strip(), out, audio_only=audio_only, audio_format=audio_format, container=container, items=items)
    log(f"downloading {'audio only' if audio_only else 'video'} ({audio_format if audio_only else container}, highest quality) into {out}")
    t0 = time.time()
    files: list[str] = []
    skipped: list[str] = []
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


# ---------------------------------------------------------------------------------------------
# Subtitles only
# ---------------------------------------------------------------------------------------------

SUB_LANGS = ("en", "hi")
_REGIONAL = {"en": ("en", "en-IN", "en-US", "en-GB", "en-CA", "en-AU"), "hi": ("hi", "hi-IN", "hi-Latn")}


def pick_lang(info: dict, lang: str) -> tuple[str, bool] | None:
    """(track code, automatic?) for one language: the creator's own subtitles, else YouTube's
    automatic captions when the video is spoken in that language (a machine translation never)."""
    subs = info.get("subtitles") or {}
    for code in _REGIONAL.get(lang, (lang,)):
        if code in subs:
            return code, False
    auto = info.get("automatic_captions") or {}
    spoken = (info.get("language") or "").split("-")[0]
    if f"{lang}-orig" in auto:
        return f"{lang}-orig", True
    if spoken == lang and lang in auto:
        return lang, True
    return None


_TAGS = re.compile(r"\[(?:music|applause|laughter|laughs|cheering|inaudible|silence|noise|♪+)[^\]]*\]|\((?:music|applause|laughs?)\)|[♪♫]+", re.I)


def devanagari_share(text: str) -> float:
    letters = re.findall(r"[A-Za-zऀ-ॿ]", text)
    return sum(1 for c in letters if "ऀ" <= c <= "ॿ") / max(1, len(letters))


def looks_like(text: str, lang: str) -> bool:
    """A track really is in its language (a "Hindi" track sometimes holds an English translation)."""
    d = devanagari_share(text)
    return d >= 0.3 if lang == "hi" else d < 0.3


def clean_cues(cues: list[dict]) -> list[dict]:
    """Captions without tags, speaker markers and stray symbols (rolling duplicates are already
    gone: script_writer.parse_timed)."""
    out = []
    for c in cues:
        t = _TAGS.sub(" ", c["text"])
        t = re.sub(r"^\s*(>>|-|–)\s*", "", t)
        t = re.sub(r"&nbsp;|&amp;", lambda m: "&" if "amp" in m.group(0) else " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if t and re.search(r"[\wऀ-ॿ]", t):
            out.append({**c, "text": t})
    return out


def sentences_by_pauses(cues: list[dict], gap: float = 0.7) -> str:
    """Rules for unpunctuated captions: a pause is a sentence end, sentences start with a capital,
    "i" is "I"; Hindi sentences end with a danda."""
    out, cur = [], []
    for k, c in enumerate(cues):
        cur.append(c["text"])
        nxt = cues[k + 1] if k + 1 < len(cues) else None
        if nxt is None or nxt["start"] - c["end"] >= gap:
            sent = re.sub(r"\s+", " ", " ".join(cur)).strip()
            cur = []
            if not sent:
                continue
            sent = re.sub(r"\bi\b", "I", sent)
            sent = sent[:1].upper() + sent[1:]
            if not re.search(r"[.!?।]$", sent):
                sent += "।" if re.search(r"[ऀ-ॿ]", sent) and not re.search(r"[A-Za-z]{3,}\s*$", sent) else "."
            out.append(sent)
    return " ".join(out)


def is_punctuated(text: str) -> bool:
    """Human subtitles have sentences; YouTube's automatic captions mostly do not."""
    words = max(1, len(text.split()))
    return len(re.findall(r"[.!?।]", text)) / words > 1 / 40


def polish_claude(text: str, lang: str, title: str) -> str | None:
    """Claude makes an automatic transcript readable without changing how the person speaks.
    None when it fails or changes too much (it must not summarise or rewrite)."""
    import script_writer as sw

    prompt = (f"This is YouTube's automatic transcript of the video “{title}” ({'Hindi/Hinglish' if lang == 'hi' else 'English'}). "
              "It has no punctuation and some misheard words.\n\nReturn the same speech as clean text: add punctuation, sentence "
              "breaks and paragraphs (a blank line between topics), fix capital letters, and fix only words that are clearly misheard "
              "from the context. Keep the speaker's own words, slang, fillers, repetitions and Hinglish exactly as spoken: do not "
              "paraphrase, shorten, summarise or translate. Hindi stays in Devanagari and English words stay in English. "
              "Reply with only the text.\n\nTRANSCRIPT:\n" + text)
    try:
        out = sw.ask_claude(prompt, None, timeout=600).strip()
    except AutocutError as e:
        log(f"polishing with Claude failed ({e}); using rules")
        return None
    n_in, n_out = len(text.split()), len(out.split())
    if not out or not (0.85 * n_in <= n_out <= 1.15 * n_in):
        log(f"Claude's text changed too much ({n_out} words for {n_in}); using rules")
        return None
    return out


def video_list(url: str, items: str | None) -> list[dict]:
    """The videos of a link: [{url, title, index, playlist}] (one entry for a video link)."""
    import script_writer as sw

    cmd = sw.yt_dlp_cmd() + ["--flat-playlist", "--ignore-errors", "--print", "%(playlist_index|0)s\t%(playlist_title|)s\t%(id)s\t%(title)s"]
    if items:
        cmd += ["--playlist-items", items.replace(" ", "")]
    cmd += ["--yes-playlist" if is_playlist(url) else "--no-playlist", url]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
    vids = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 4 and parts[2] and parts[2] != "NA":
            idx = int(parts[0]) if parts[0].isdigit() else 0
            vids.append({"url": f"https://www.youtube.com/watch?v={parts[2]}", "title": parts[3], "index": idx, "playlist": parts[1]})
    return vids


def safe_name(t: str) -> str:
    t = re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", t)
    return re.sub(r"\s+", " ", t).strip()[:120] or "video"


def download_subs(url: str, out_dir: str, *, langs: list[str], items: str | None = None, polish: str = "claude",
                  learn_style: str | None = None, parallel: int = 3, redo: bool = False) -> dict:
    import script_writer as sw

    langs = [l for l in langs if l in SUB_LANGS]
    if not langs:
        raise AutocutError(f"choose a subtitle language ({', '.join(SUB_LANGS)})")
    if polish not in ("claude", "rules"):
        raise AutocutError("polish must be claude or rules")
    vids = video_list(url, items)
    if not vids:
        raise AutocutError("no videos found at this link")
    playlist = vids[0]["playlist"] if is_playlist(url) and vids[0]["playlist"] else ""
    folder = Path(out_dir).expanduser().resolve() / safe_name(playlist) if playlist else Path(out_dir).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    parallel = clamp_parallel(parallel)
    log(f"subtitles ({', '.join(langs)}) of {len(vids)} video(s) into {folder}, {min(parallel, len(vids))} at a time"
        + (" (redoing all)" if redo else ""))
    t0 = time.time()
    files: list[str] = []
    missing: list[str] = []
    base = sw.yt_dlp_cmd() + ["--no-playlist", "--quiet", "--no-warnings"]
    mpath = folder / "subtitles.json"
    try:
        manifest: dict = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    this_run: list[str] = []
    lock = threading.RLock()
    finished = [0]

    def say(msg: str) -> None:
        with lock:  # (whole lines, whichever worker writes)
            log(msg)
    for v in vids:
        stem = (f"{v['index']:03d} - " if playlist and v["index"] else "") + safe_name(v["title"])
        v["stem"] = stem
        this_run += [f"{stem}.{l}.txt" for l in langs]

    def subs_of(k: int, v: dict, tmp: str) -> None:
        try:
            subs_one(k, v, tmp)
        finally:
            with lock:
                finished[0] += 1
                log(f"progress {finished[0]}/{len(vids)} videos")

    def subs_one(k: int, v: dict, tmp: str) -> None:
        stem = v["stem"]
        todo = [l for l in langs if redo or not (folder / f"{stem}.{l}.txt").is_file()]
        if not todo:
            return  # (done on an earlier run)
        try:
            proc = subprocess.run(base + ["-J", "--skip-download", v["url"]], capture_output=True, text=True, timeout=180)
            info = json.loads(proc.stdout)
        except (ValueError, OSError, subprocess.TimeoutExpired):
            with lock:
                missing.append(f"{v['title']}: unavailable")
            say(f"skipped (unavailable): {v['title'][:70]}")
            return
        if not (info.get("subtitles") or info.get("automatic_captions")):
            with lock:
                missing.append(f"{v['title']}: this video has no subtitles at all")
            say(f"no subtitles at all: {v['title'][:70]}")
            return
        spoken = (info.get("language") or "").split("-")[0] or next(
            (k_.split("-")[0] for k_ in (info.get("automatic_captions") or {}) if k_.endswith("-orig")), "")
        for lang in todo:
            tr = pick_lang(info, lang)
            if tr is None:
                with lock:
                    missing.append(f"{v['title']}: no {lang} subtitles")
                continue
            code, is_auto = tr
            cues = fetch_cues(base, v["url"], code, is_auto, Path(tmp) / f"{k}-{lang}")
            if cues and not looks_like(" ".join(c["text"] for c in cues), lang):
                # labelled "Hindi" but written in English (or the other way round): the real speech instead
                auto = info.get("automatic_captions") or {}
                alt = f"{lang}-orig" if f"{lang}-orig" in auto else (lang if spoken == lang and lang in auto else None)
                say(f"the {code} subtitles of “{v['title'][:50]}” are not in {lang}" + ("; using the automatic captions of the speech" if alt else ""))
                cues, code, is_auto = (fetch_cues(base, v["url"], alt, True, Path(tmp) / f"{k}-{lang}-alt"), alt, True) if alt else ([], code, is_auto)
            if cues is None or not cues:
                with lock:
                    missing.append(f"{v['title']}: " + (f"{lang} subtitles could not be downloaded" if cues is None else f"no real {lang} subtitles"))
                continue
            raw = " ".join(c["text"] for c in cues)
            if is_punctuated(raw):
                text = re.sub(r"\s+", " ", raw).strip()
            else:
                text = (polish_claude(raw, lang, v["title"]) if polish == "claude" else None) or sentences_by_pauses(cues)
            (folder / f"{stem}.{lang}.srt").write_text(ac.to_srt(cues_as_words(cues)), encoding="utf-8")
            (folder / f"{stem}.{lang}.txt").write_text(f"{v['title']}\n{v['url']}\n\n{text.strip()}\n", encoding="utf-8")
            with lock:
                files.append(str(folder / f"{stem}.{lang}.txt"))
                # what this text is: the person's own words, or a translation of them
                manifest[f"{stem}.{lang}.txt"] = {"url": v["url"], "lang": lang, "track": code, "automatic": is_auto,
                                                   "spoken": (not spoken) or spoken == lang}
            say(f"subtitles ({lang}{', automatic, polished' if is_auto else ''}): {stem[:70]}")

    with tempfile.TemporaryDirectory(prefix="ei-subs-") as tmp:
        with ThreadPoolExecutor(max_workers=min(parallel, len(vids))) as pool:
            list(pool.map(lambda kv: subs_of(kv[0], kv[1], tmp), enumerate(vids, 1)))
    # one text per language with everything (the input for a style)
    alls = []
    for lang in langs:
        parts = sorted(p for p in folder.glob(f"*.{lang}.txt") if not p.name.startswith("all."))
        if parts:
            body = "\n\n".join(p.read_text(encoding="utf-8").split("\n", 3)[-1].strip() for p in parts)
            allp = folder / f"all.{lang}.txt"
            allp.write_text(body + "\n", encoding="utf-8")
            alls.append(allp)
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    if missing:  # (added to what earlier runs found missing)
        mf = folder / "subtitles-missing.txt"
        old = mf.read_text(encoding="utf-8").splitlines() if mf.is_file() else []
        mf.write_text("\n".join(dict.fromkeys(old + missing)) + "\n", encoding="utf-8")
    res = {"outputs": files, "out_dir": str(folder), "downloaded": len(files), "skipped": missing}
    if learn_style and learn_style.strip():
        # only this download, and only the words as spoken (a translation is not the person's style)
        names = [n for n in this_run if (folder / n).is_file() and manifest.get(n, {}).get("spoken", True)]
        srcs = [{"name": n, "cues": [], "text": re.sub(r"\s+", " ", (folder / n).read_text(encoding="utf-8").split("\n", 3)[-1]).strip()} for n in names]
        srcs = [x for x in srcs if x["text"]]
        if srcs:
            prof = sw.analyse_style(srcs)
            sp = sw.save_style(learn_style.strip(), prof)
            res["style"] = str(sp)
            log(f"style “{learn_style.strip()}” learned from {len(srcs)} transcript(s), {prof['words']} words, {prof['language']}: "
                "choose it in AI ▸ Write Voiceover Script or Video from Topic ▸ Style")
    log(f"done in {time.time() - t0:.0f}s: {len(files)} subtitle file(s)" + (f", {len(missing)} missing (subtitles-missing.txt)" if missing else ""))
    return res


def fetch_cues(base: list[str], url: str, code: str, is_auto: bool, d: Path) -> list[dict] | None:
    """Cleaned cues of one caption track (None when it could not be downloaded)."""
    import script_writer as sw

    d.mkdir(parents=True, exist_ok=True)
    cmd = base + ["--skip-download", "--write-auto-subs" if is_auto else "--write-subs", "--sub-langs", code, "--sub-format", "vtt/srt/best",
                  "-o", str(d / "s.%(ext)s"), url]
    for attempt in range(3):
        subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        got = next(iter(sorted(d.glob("s.*.vtt")) + sorted(d.glob("s.*.srt"))), None)
        if got:
            return clean_cues(sw.parse_timed(got.read_text(encoding="utf-8", errors="replace")))
        time.sleep(4 * (attempt + 1))  # YouTube rate limit
    return None


def cues_as_words(cues: list[dict]) -> list[dict]:
    """Cues in the shape ac.to_srt expects (groups of words with times)."""
    return [{"start": c["start"], "end": c["end"], "text": c["text"]} for c in cues]


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("url", help="a YouTube playlist or video link")
    p.add_argument("--out-dir", default=str(Path.home() / "Movies" / "EI" / "Downloads"))
    p.add_argument("--audio-only", action="store_true", help="only the audio")
    p.add_argument("--audio-format", default="original", choices=list(AUDIO_FORMATS), help="original: the best stream as it is (no re-encode)")
    p.add_argument("--container", default="mp4", choices=list(CONTAINERS), help="video: mp4 (plays everywhere) or mkv (keeps any codec)")
    p.add_argument("--items", help="only these playlist items, e.g. 1-20 or 1,4,9")
    p.add_argument("--subs", action="store_true", help="only the subtitles (cleaned text and .srt), no video or audio")
    p.add_argument("--sub-langs", nargs="+", default=["en", "hi"], choices=list(SUB_LANGS))
    p.add_argument("--polish", default="claude", choices=["claude", "rules"], help="automatic captions: Claude adds punctuation and fixes mishearings")
    p.add_argument("--learn-style", help="also learn a talking style with this name from the subtitles")
    p.add_argument("--parallel", type=int, default=3, help=f"videos downloaded at the same time (1 to {MAX_PARALLEL})")
    p.add_argument("--redo", action="store_true", help="download everything again (replacing the files) instead of skipping what is already there")


def run_args(a) -> dict:
    if a.subs:
        return download_subs(a.url, a.out_dir, langs=a.sub_langs, items=a.items, polish=a.polish, learn_style=a.learn_style,
                             parallel=a.parallel, redo=a.redo)
    return download(a.url, a.out_dir, audio_only=a.audio_only, audio_format=a.audio_format, container=a.container, items=a.items,
                    parallel=a.parallel, redo=a.redo)


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
