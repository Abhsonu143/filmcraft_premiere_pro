#!/usr/bin/env python3
"""autocut — talking-head auto editor (jump cuts, captions, ducked music).

Pipeline:  clip -> silence detection -> speech segments -> plan.json (editable) -> render.

    python autocut.py plan   talk.mp4 --preset short --captions --music bed.mp3
    python autocut.py render talk.plan.json
    python autocut.py auto   talk.mp4 --preset long
    python autocut.py export talk.plan.json          # -> talk.otio (+ .srt) for FilmCraft

The AI only writes a plan. Edit `segments` in the plan by hand, then `render` again.
Original media is never modified; everything is written next to the plan or in the cache.

This module also holds the helpers `voiceover_cut.py` imports (probe, ffmpeg runner, SRT,
transcription, cache, presets).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

PLAN_VERSION = 1
CACHE_DIR = Path(os.environ.get("AUTOCUT_CACHE", Path.home() / ".cache" / "autocut"))
DEFAULT_WHISPER = "mlx-community/whisper-small-mlx"

# Output formats. `fps` is the timeline rate; every cut is snapped to its frame grid.
PRESETS: dict[str, dict] = {
    "short": {"width": 1080, "height": 1920, "fps": 30, "video_bitrate": "10M", "caption_style": "big"},
    "long": {"width": 1920, "height": 1080, "fps": 30, "video_bitrate": "12M", "caption_style": "bottom"},
    "travel": {"width": 1920, "height": 1080, "fps": 30, "video_bitrate": "14M", "caption_style": "bottom"},
}

# libass styles for the `subtitles` filter. Sizes are in the script's 384x288 reference space.
CAPTION_STYLES = {
    "big": "FontName=Helvetica,FontSize=22,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
    "BorderStyle=1,Outline=3,Shadow=0,Alignment=2,MarginV=70",
    "bottom": "FontName=Helvetica,FontSize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
    "BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginV=20",
}


class AutocutError(Exception):
    """A user-facing error: printed without a traceback."""


def log(msg: str) -> None:
    print(f"[autocut] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------
# ffmpeg / ffprobe
# ---------------------------------------------------------------------------------------------


def require_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise AutocutError(f"{tool} not found. Install it with: brew install ffmpeg")


def ffmpeg_major() -> int:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-version"], capture_output=True, text=True).stdout
    m = re.search(r"ffmpeg version n?(\d+)", out)
    return int(m.group(1)) if m else 0


def run(cmd: list[str], *, quiet: bool = True) -> subprocess.CompletedProcess:
    """Run a command; raise AutocutError with the tail of stderr when it fails."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise AutocutError(f"command failed ({proc.returncode}): {' '.join(cmd[:6])} ...\n{tail}")
    if not quiet and proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return proc


def filter_script_args(script_path: str) -> list[str]:
    """`-filter_complex_script` is deprecated in FFmpeg 7; there the `-/opt file` form replaces it."""
    if ffmpeg_major() >= 7:
        return ["-/filter_complex", script_path]
    return ["-filter_complex_script", script_path]


@dataclass
class MediaInfo:
    path: str
    duration: float
    has_video: bool
    has_audio: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0
    rotation: int = 0
    vfr: bool = False

    @property
    def display_size(self) -> tuple[int, int]:
        """Width/height after applying the rotation tag (phones record portrait as rotated landscape)."""
        if self.rotation % 180 == 90:
            return self.height, self.width
        return self.width, self.height


def _rate(s: str | None) -> float:
    try:
        f = Fraction(s or "0")
        return float(f) if f > 0 else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path: str | Path) -> MediaInfo:
    path = str(path)
    if not os.path.isfile(path):
        raise AutocutError(f"file not found: {path}")
    proc = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    dur = 0.0
    for src in (data.get("format", {}), v or {}, a or {}):
        try:
            dur = max(dur, float(src.get("duration", 0) or 0))
        except ValueError:
            pass
    info = MediaInfo(path=path, duration=dur, has_video=v is not None, has_audio=a is not None)
    if v:
        info.width, info.height = int(v.get("width", 0)), int(v.get("height", 0))
        r, avg = _rate(v.get("r_frame_rate")), _rate(v.get("avg_frame_rate"))
        info.fps = avg or r
        # Phones often record variable frame rate: r_frame_rate (max) and avg differ noticeably.
        info.vfr = bool(r and avg and abs(r - avg) / r > 0.01)
        rot = v.get("tags", {}).get("rotate")
        for sd in v.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = sd["rotation"]
        try:
            info.rotation = int(float(rot or 0)) % 360
        except ValueError:
            info.rotation = 0
    return info


# ---------------------------------------------------------------------------------------------
# Silence -> speech segments (pure logic, unit-tested)
# ---------------------------------------------------------------------------------------------

_SIL_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SIL_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def parse_silencedetect(stderr: str, duration: float) -> list[tuple[float, float]]:
    """Silence intervals from ffmpeg `silencedetect` output. An unterminated silence runs to the end."""
    out: list[tuple[float, float]] = []
    start: float | None = None
    for line in stderr.splitlines():
        if (m := _SIL_START.search(line)) is not None:
            start = max(0.0, float(m.group(1)))
        elif (m := _SIL_END.search(line)) is not None and start is not None:
            out.append((start, min(float(m.group(1)), duration)))
            start = None
    if start is not None and start < duration:
        out.append((start, duration))
    return out


def speech_segments(
    silences: list[tuple[float, float]],
    duration: float,
    *,
    pad: float = 0.12,
    min_segment: float = 0.30,
    merge_gap: float = 0.15,
) -> list[tuple[float, float]]:
    """Invert silences into speech, pad each side, merge overlaps/near-touching, drop blips."""
    if duration <= 0:
        return []
    speech: list[tuple[float, float]] = []
    cur = 0.0
    for s, e in sorted(silences):
        if s > cur:
            speech.append((cur, s))
        cur = max(cur, e)
    if cur < duration:
        speech.append((cur, duration))
    padded = [(max(0.0, s - pad), min(duration, e + pad)) for s, e in speech if e - s > 0]
    merged: list[tuple[float, float]] = []
    for s, e in padded:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= min_segment]


def snap_segments(segments: list[tuple[float, float]], fps: float) -> list[tuple[float, float]]:
    """Snap to the frame grid so video (frames) and audio (samples) cut at the same instant; no drift."""
    out: list[tuple[float, float]] = []
    for s, e in segments:
        fs, fe = round(s * fps), round(e * fps)
        if out and fs <= round(out[-1][1] * fps):
            fs = round(out[-1][1] * fps)
        if fe > fs:
            out.append((fs / fps, fe / fps))
    return out


def output_duration(segments: list[tuple[float, float]]) -> float:
    return sum(e - s for s, e in segments)


def map_to_output(t: float, segments: list[tuple[float, float]]) -> float | None:
    """Source time -> time on the cut timeline, or None when `t` was cut out."""
    acc = 0.0
    for s, e in segments:
        if s <= t <= e:
            return acc + (t - s)
        acc += e - s
    return None


def remap_words(words: list[dict], segments: list[tuple[float, float]]) -> list[dict]:
    """Move word timings from source time onto the cut timeline. Words fully cut out are dropped;
    words straddling a cut are clamped to the kept part."""
    out = []
    acc = 0.0
    bounds = []
    for s, e in segments:
        bounds.append((s, e, acc))
        acc += e - s
    for w in words:
        ws, we = float(w["start"]), float(w["end"])
        for s, e, off in bounds:
            lo, hi = max(ws, s), min(we, e)
            if hi > lo or (ws == we and s <= ws <= e):
                out.append({"word": w["word"], "start": off + (lo - s), "end": off + (max(hi, lo) - s)})
                break
    return out


# ---------------------------------------------------------------------------------------------
# Captions / SRT (pure logic, unit-tested)
# ---------------------------------------------------------------------------------------------


def srt_time(t: float) -> str:
    ms = max(0, int(round(t * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def group_words(words: list[dict], *, max_words: int = 6, max_chars: int = 32, max_gap: float = 0.6) -> list[dict]:
    """Group words into caption cues; a cue breaks on size, a pause, or sentence punctuation."""
    cues: list[dict] = []
    cur: list[dict] = []

    def flush() -> None:
        if cur:
            text = " ".join(w["word"].strip() for w in cur).strip()
            if text:
                cues.append({"start": cur[0]["start"], "end": cur[-1]["end"], "text": text})
            cur.clear()

    for w in words:
        word = w["word"].strip()
        if not word:
            continue
        if cur:
            length = len(" ".join(x["word"].strip() for x in cur)) + 1 + len(word)
            if len(cur) >= max_words or length > max_chars or w["start"] - cur[-1]["end"] > max_gap:
                flush()
        cur.append(w)
        if word[-1] in ".?!":
            flush()
    flush()
    # Never overlap: a cue ends no later than the next one starts.
    for a, b in zip(cues, cues[1:]):
        a["end"] = min(a["end"], b["start"])
    return [c for c in cues if c["end"] > c["start"]]


def to_srt(cues: list[dict]) -> str:
    blocks = [f"{i}\n{srt_time(c['start'])} --> {srt_time(c['end'])}\n{c['text']}\n" for i, c in enumerate(cues, 1)]
    return "\n".join(blocks)


def subtitles_filter(srt_path: str, style: str) -> str:
    """The `subtitles` filter with a path escaped for filtergraph syntax."""
    p = srt_path.replace("\\", "\\\\\\\\").replace(":", "\\\\:").replace("'", "\\\\\\'")
    s = CAPTION_STYLES.get(style, CAPTION_STYLES["bottom"]).replace(",", "\\,")
    return f"subtitles=filename={p}:force_style={s}"


# ---------------------------------------------------------------------------------------------
# Cache + transcription (models run in a child process, so memory is returned when it exits)
# ---------------------------------------------------------------------------------------------


def file_key(path: str | Path, *extra: str) -> str:
    st = os.stat(path)
    raw = "|".join([str(Path(path).resolve()), str(st.st_size), str(int(st.st_mtime)), *extra])
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def extract_audio(src: str, dst: str, *, start: float | None = None, end: float | None = None) -> None:
    """16 kHz mono WAV, the format Whisper/VAD want."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", src, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst]
    run(cmd)


def _transcribe_worker(wav: str, model: str, prompt: str | None, language: str | None) -> dict:
    import mlx_whisper  # imported here: only the child process loads the model

    kwargs = {"path_or_hf_repo": model, "word_timestamps": True, "verbose": None}
    if prompt:
        kwargs["initial_prompt"] = prompt
    if language:
        kwargs["language"] = language
    res = mlx_whisper.transcribe(wav, **kwargs)
    words = []
    for seg in res.get("segments", []):
        for w in seg.get("words", []) or []:
            words.append({"word": w["word"], "start": float(w["start"]), "end": float(w["end"])})
    segs = [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()} for s in res.get("segments", [])]
    return {"language": res.get("language"), "segments": segs, "words": words}


def run_isolated(fn, *args):
    """Run `fn(*args)` in a fresh process (one heavy model at a time; memory freed on exit)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(fn, *args).result()


def transcribe(src: str, *, model: str = DEFAULT_WHISPER, prompt: str | None = None, language: str | None = None) -> dict:
    """Word-level transcript of `src`, cached by file identity + model + prompt."""
    key = file_key(src, model, prompt or "", language or "")
    cache = CACHE_DIR / "transcripts" / f"{key}.json"
    if cache.is_file():
        log(f"transcript cache hit ({cache.name})")
        return json.loads(cache.read_text())
    try:
        import mlx_whisper  # noqa: F401
    except ImportError as e:
        raise AutocutError("mlx-whisper is not installed: pip install -r requirements.txt") from e
    with tempfile.TemporaryDirectory(prefix="autocut-") as tmp:
        wav = os.path.join(tmp, "audio.wav")
        extract_audio(src, wav)
        log(f"transcribing with {model} (the first use downloads it)")
        t0 = time.time()
        result = run_isolated(_transcribe_worker, wav, model, prompt, language)
        log(f"transcribed in {time.time() - t0:.1f}s, {len(result['words'])} words")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result))
    return result


# ---------------------------------------------------------------------------------------------
# Video filters shared by both modes
# ---------------------------------------------------------------------------------------------


def cover_filter(width: int, height: int, fps: float) -> str:
    """Constant frame rate, scale to cover the frame, centre crop (16:9 -> 9:16 for shorts)."""
    return (
        f"fps={fps:g},scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1,format=yuv420p"
    )


def rate_arg(fps: float) -> str:
    """Exact rational rate for ffmpeg (29.97 -> 30000/1001)."""
    return str(Fraction(fps).limit_denominator(1001))


def video_encoder_args(codec: str, bitrate: str) -> list[str]:
    if codec == "hevc":
        return ["-c:v", "hevc_videotoolbox", "-b:v", bitrate, "-tag:v", "hvc1"]
    if codec == "h264":
        return ["-c:v", "h264_videotoolbox", "-b:v", bitrate]
    if codec == "x264":  # software fallback (slower, uses more CPU)
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    raise AutocutError(f"unknown codec {codec!r} (hevc, h264, x264)")


def music_mix_filter(voice: str, music: str, out: str, duration: float, music_db: float = -16.0) -> str:
    """Music under the voice, ducked by a sidechain compressor keyed on the voice, then loudness-normalised."""
    return (
        f"[{voice}]asplit=2[v_main][v_key];"
        f"[{music}]atrim=0:{duration:.3f},asetpts=PTS-STARTPTS,volume={music_db}dB,"
        f"afade=t=out:st={max(0.0, duration - 2):.3f}:d=2[m_bed];"
        f"[m_bed][v_key]sidechaincompress=threshold=0.02:ratio=10:attack=15:release=350[m_duck];"
        f"[v_main][m_duck]amix=inputs=2:duration=first:normalize=0,"
        f"loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[{out}]"
    )


# ---------------------------------------------------------------------------------------------
# plan / render / export
# ---------------------------------------------------------------------------------------------


def default_plan_path(src: str, suffix: str = ".plan.json") -> Path:
    p = Path(src)
    return p.with_name(p.stem + suffix)


def detect_silences(src: str, duration: float, noise_db: float, min_silence: float) -> list[tuple[float, float]]:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", src, "-vn", "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AutocutError(f"silencedetect failed:\n{proc.stderr[-1500:]}")
    return parse_silencedetect(proc.stderr, duration)


def make_plan(args: argparse.Namespace) -> dict:
    require_tools()
    src = str(Path(args.input).resolve())
    info = probe(src)
    if not info.has_audio:
        raise AutocutError("this clip has no audio track, so silence cutting cannot work. Use voiceover_cut.py (narration) instead.")
    if not info.has_video:
        raise AutocutError("this file has no video stream.")
    preset = dict(PRESETS[args.preset])
    fps = float(args.fps or preset["fps"])
    t0 = time.time()
    silences = detect_silences(src, info.duration, args.silence_db, args.min_silence)
    segs = speech_segments(silences, info.duration, pad=args.pad, min_segment=args.min_segment, merge_gap=args.merge_gap)
    segs = snap_segments(segs, fps)
    if not segs:
        raise AutocutError("no speech found. Try a lower --silence-db (e.g. -40) or check the audio.")
    log(f"silence detection {time.time() - t0:.1f}s: kept {len(segs)} segments, {output_duration(segs):.1f}s of {info.duration:.1f}s")
    plan = {
        "version": PLAN_VERSION,
        "mode": "autocut",
        "input": src,
        "preset": args.preset,
        "width": preset["width"],
        "height": preset["height"],
        "fps": fps,
        "video_bitrate": preset["video_bitrate"],
        "source": {"duration": info.duration, "width": info.width, "height": info.height, "fps": info.fps, "vfr": info.vfr, "rotation": info.rotation},
        "settings": {"silence_db": args.silence_db, "min_silence": args.min_silence, "pad": args.pad, "min_segment": args.min_segment},
        "segments": [{"start": round(s, 6), "end": round(e, 6)} for s, e in segs],
        "music": {"path": str(Path(args.music).resolve()), "db": args.music_db} if args.music else None,
        "captions": None,
    }
    if args.captions:
        tr = transcribe(src, model=args.whisper_model, language=args.language)
        plan["captions"] = {"style": preset["caption_style"], "words": tr["words"]}
    return plan


def plan_segments(plan: dict) -> list[tuple[float, float]]:
    segs = [(float(s["start"]), float(s["end"])) for s in plan.get("segments", [])]
    segs = [(s, e) for s, e in segs if e > s]
    if not segs:
        raise AutocutError("the plan has no segments")
    return segs


def write_captions(plan: dict, segs: list[tuple[float, float]], path: Path) -> bool:
    cap = plan.get("captions")
    if not cap or not cap.get("words"):
        return False
    style = cap.get("style", "bottom")
    big = style == "big"
    cues = group_words(remap_words(cap["words"], segs), max_words=3 if big else 7, max_chars=18 if big else 38)
    path.write_text(to_srt(cues), encoding="utf-8")
    return bool(cues)


def build_autocut_graph(plan: dict, segs: list[tuple[float, float]], srt: str | None, has_music: bool) -> str:
    """One pass: select the kept ranges (no per-segment split buffers, so memory stays flat)."""
    fps = float(plan["fps"])
    # Segments sit on the frame grid. Half a frame of slack on both ends makes the test immune to
    # rounding: frame k is kept when s <= k/fps < e.
    half = 0.5 / fps
    keep = "+".join(f"between(t,{s - half:.6f},{e - half:.6f})" for s, e in segs)
    dur = output_duration(segs)
    # aselect keeps or drops whole audio frames, so first re-chunk audio into one-video-frame
    # packets: then audio is cut on exactly the same instants as the picture.
    spf = max(1, round(48000 / fps))
    v = (
        f"[0:v]fps={fps:g},select='{keep}',setpts=N/FRAME_RATE/TB,"
        f"scale={plan['width']}:{plan['height']}:force_original_aspect_ratio=increase,"
        f"crop={plan['width']}:{plan['height']},setsar=1,format=yuv420p"
    )
    if srt:
        v += "," + subtitles_filter(srt, plan["captions"].get("style", "bottom"))
    graph = [v + "[vout]"]
    graph.append(
        f"[0:a]aresample=48000,asetnsamples=n={spf}:p=0,aselect='{keep}',asetpts=N/SR/TB,atrim=0:{dur:.6f}[voice]"
    )
    if has_music:
        music_db = float((plan.get("music") or {}).get("db", -16))
        graph.append(music_mix_filter("voice", "1:a", "aout", dur, music_db))
    else:
        graph.append("[voice]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]")
    return ";\n".join(graph)


def render_plan(plan_path: Path, *, out: str | None, codec: str) -> Path:
    require_tools()
    plan = json.loads(plan_path.read_text())
    segs = plan_segments(plan)
    src = plan["input"]
    if not os.path.isfile(src):
        raise AutocutError(f"source clip missing: {src}")
    out_path = Path(out) if out else plan_path.with_name(plan_path.name.replace(".plan.json", "") + ".cut.mp4")
    if out_path.resolve() == Path(src).resolve():
        raise AutocutError("refusing to overwrite the original clip")
    music = (plan.get("music") or {}).get("path")
    if music and not os.path.isfile(music):
        log(f"music file missing, rendering without music: {music}")
        music = None
    with tempfile.TemporaryDirectory(prefix="autocut-") as tmp:
        srt = os.path.join(tmp, "captions.srt")
        has_caps = write_captions(plan, segs, Path(srt))
        if has_caps:
            shutil.copy(srt, out_path.with_suffix(".srt"))
        script = os.path.join(tmp, "graph.txt")
        Path(script).write_text(build_autocut_graph(plan, segs, srt if has_caps else None, bool(music)))
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-hwaccel", "videotoolbox", "-i", src]
        if music:
            cmd += ["-stream_loop", "-1", "-i", music]
        cmd += filter_script_args(script)
        # cfr at the timeline rate: ffmpeg's default sync otherwise drops a few selected frames.
        cmd += ["-map", "[vout]", "-map", "[aout]", "-fps_mode", "cfr", "-r", rate_arg(float(plan["fps"]))]
        cmd += video_encoder_args(codec, plan.get("video_bitrate", "10M"))
        cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{output_duration(segs):.6f}", str(out_path)]
        t0 = time.time()
        run(cmd)
        log(f"rendered {out_path} in {time.time() - t0:.1f}s")
    return out_path


def save_plan(plan: dict, path: Path) -> None:
    path.write_text(json.dumps(plan, indent=2))
    log(f"plan written: {path}  (edit it if you like, then run render)")


def add_plan_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="talking-head clip (needs an audio track with speech)")
    p.add_argument("--preset", choices=sorted(PRESETS), default="long")
    p.add_argument("--fps", type=float, help="timeline frame rate (default from preset)")
    p.add_argument("--music", help="background music file (ducked under the voice)")
    p.add_argument("--music-db", type=float, default=-16.0, help="music level before ducking")
    p.add_argument("--captions", action="store_true", help="transcribe with Whisper and burn word captions")
    p.add_argument("--whisper-model", default=DEFAULT_WHISPER)
    p.add_argument("--language", help="force a language code, e.g. en, hi")
    p.add_argument("--silence-db", type=float, default=-32.0, help="below this level counts as silence")
    p.add_argument("--min-silence", type=float, default=0.5, help="shortest pause (s) that gets cut")
    p.add_argument("--pad", type=float, default=0.12, help="seconds kept around each speech part")
    p.add_argument("--min-segment", type=float, default=0.3)
    p.add_argument("--merge-gap", type=float, default=0.15)
    p.add_argument("-o", "--plan", help="plan file (default <name>.plan.json)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_plan = sub.add_parser("plan", help="analyse a clip and write an editable plan")
    add_plan_args(p_plan)
    p_render = sub.add_parser("render", help="render a plan")
    p_render.add_argument("plan")
    p_render.add_argument("-o", "--out")
    p_render.add_argument("--codec", choices=["hevc", "h264", "x264"], default="hevc")
    p_auto = sub.add_parser("auto", help="plan + render")
    add_plan_args(p_auto)
    p_auto.add_argument("--out")
    p_auto.add_argument("--codec", choices=["hevc", "h264", "x264"], default="hevc")
    p_exp = sub.add_parser("export", help="write an OpenTimelineIO timeline for FilmCraft / Resolve")
    p_exp.add_argument("plan")
    p_exp.add_argument("-o", "--out", help="output .otio (default next to the plan)")
    args = ap.parse_args(argv)
    try:
        if args.cmd in ("plan", "auto"):
            plan = make_plan(args)
            path = Path(args.plan) if args.plan else default_plan_path(args.input)
            save_plan(plan, path)
            if args.cmd == "auto":
                render_plan(path, out=args.out, codec=args.codec)
        elif args.cmd == "render":
            render_plan(Path(args.plan), out=args.out, codec=args.codec)
        elif args.cmd == "export":
            import otio_export

            otio_export.export_plan_file(Path(args.plan), Path(args.out) if args.out else None)
    except AutocutError as e:
        log(f"error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
