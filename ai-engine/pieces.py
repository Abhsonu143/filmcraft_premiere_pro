#!/usr/bin/env python3
"""pieces: a video put together from pieces of recordings, with a voice over it.

A unit for any tool that has decided WHAT to show and only needs it made:

    render(pieces, voice, out, W=1920, H=1080)

`pieces`: in order, each {"src": a recording, "start": seconds, "seconds": how long it is shown,
"gain": how loud its own sound is (1.0 as recorded, 0.08 quietly under a voice, 0 silent),
"match": true to bring it to about the voice's loudness (for a line of the recording that must be heard)}.
`voice`: [(a wav, where in it the part starts, how long the part is, when it is heard in the video)].

- The picture is cut frame-exactly (a piece's length is a whole number of frames, counted along the
  whole video, so nothing drifts), two pieces at a time, with the hardware encoder.
- The recordings' own sound and the voice are written straight to disk piece by piece (an hour of
  video never sits in memory); the recordings' sound and the music duck under the voice.
- Optional: captions (an .ass file), music, and a wide video delivered tall with itself dimmed and
  blurred behind (narrate.tall_frame_filter).
The recordings are only read.
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import autocut as ac
from autocut import AutocutError, log

SR = 48000
FADE = 0.03  # seconds: each piece's sound fades in and out, so no cut clicks


def frames_of(pieces: list[dict], fps: float) -> list[int]:
    """How many frames each piece gets: its end is rounded on the whole video's clock."""
    out, t = [], 0.0
    for p in pieces:
        a = int(round(t * fps))
        t += max(0.0, float(p["seconds"]))
        out.append(max(1, int(round(t * fps)) - a))
    return out


def _cut_picture(job: tuple) -> bool:
    src, start, frames, dst, W, H, fps = job
    vf = f"fps={fps},scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p"
    proc = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", src, "-an", "-vf", vf, "-frames:v", str(frames),
                           *ac.video_encoder_args("h264", ac.bitrate_for(W, H)), "-video_track_timescale", "30000", str(dst)], capture_output=True)
    return proc.returncode == 0 and Path(dst).is_file()


def _sound(src: str, start: float, seconds: float) -> np.ndarray:
    """A stretch of a file's sound as 16-bit stereo samples (empty when it has none)."""
    proc = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", src, "-vn",
                           "-ac", "2", "-ar", str(SR), "-f", "s16le", "-"], capture_output=True)
    return np.frombuffer(proc.stdout, dtype=np.int16).reshape(-1, 2) if proc.returncode == 0 and len(proc.stdout) >= 4 else np.zeros((0, 2), np.int16)


def _fit(x: np.ndarray, n: int, gain: float = 1.0) -> np.ndarray:
    """Exactly n samples, at `gain`, faded at both ends."""
    out = np.zeros((n, 2), np.float32)
    k = min(n, len(x))
    out[:k] = x[:k].astype(np.float32) * gain
    f = min(int(FADE * SR), k // 2)
    if f > 0:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)[:, None]
        out[:f] *= ramp
        out[k - f:k] *= ramp[::-1]
    return np.clip(out, -32768, 32767).astype(np.int16)


def _level(x: np.ndarray) -> float:
    """How loud the heard part of a sound is (RMS of its louder half-seconds, 0-32768; 0 when silent)."""
    if len(x) < SR // 2:
        return 0.0
    m = x[: len(x) // (SR // 2) * (SR // 2)].astype(np.float32).mean(axis=1).reshape(-1, SR // 2)
    r = np.sqrt((m ** 2).mean(axis=1))
    r = r[r > 60]
    return float(np.sqrt((np.sort(r)[len(r) // 2:] ** 2).mean())) if len(r) else 0.0


def sound_tracks(pieces: list[dict], frames: list[int], fps: float, voice: list[tuple], game_wav: Path, voice_wav: Path) -> float:
    """Writes the recordings' own sound and the voice as two tracks of the video's exact length. Returns that length.
    A piece with "match" is brought to about the voice's loudness (a line of the game is then heard as well as yours)."""
    total = sum(frames) / fps
    heard = []
    with wave.open(str(voice_wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        at = 0
        for src, start, seconds, when in sorted(voice, key=lambda v: v[3]):
            a = int(round(float(when) * SR))
            if a > at:
                w.writeframes(np.zeros((a - at, 2), np.int16).tobytes())
                at = a
            n = int(round(float(seconds) * SR)) - max(0, at - a)
            if n <= 0:
                continue
            x = _fit(_sound(str(src), float(start) + max(0, at - a) / SR, n / SR + 0.05), n)
            heard.append(_level(x))
            w.writeframes(x.tobytes())
            at += n
        end = int(round(total * SR))
        if end > at:
            w.writeframes(np.zeros((end - at, 2), np.int16).tobytes())
    heard = [h for h in heard if h > 0]
    voice_level = float(np.median(heard)) if heard else 0.0
    with wave.open(str(game_wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        for p, fr in zip(pieces, frames):
            n = int(round(fr / fps * SR))
            gain = float(p.get("gain", 1.0))
            x = _sound(p["src"], float(p["start"]), fr / fps + 0.05) if gain > 0.001 else np.zeros((0, 2), np.int16)
            if p.get("match") and voice_level > 0:
                own = _level(x)
                if own > 0:
                    gain *= float(np.clip(0.8 * voice_level / own, 0.5, 4.0))
            w.writeframes(_fit(x, n, gain).tobytes())
    return total


def render(pieces: list[dict], voice: list[tuple], out: str, *, W: int = 1920, H: int = 1080, fps: float = 30.0, ass: str | None = None,
           music: str | None = None, music_db: float = -22.0, tall_frame: float | None = None, frame_blur: float = 0.5, tall_size: tuple[int, int] = (1080, 1920),
           workers: int = 2) -> dict:
    """Makes the video. `ass`: captions laid out for the frame that is delivered (the tall one with
    `tall_frame`). Returns {"seconds", "pieces"}."""
    if not pieces:
        raise AutocutError("there is nothing to show")
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    frames = frames_of(pieces, fps)
    with tempfile.TemporaryDirectory(prefix="ei-pieces-", dir=str(outp.parent)) as tmp:
        t = Path(tmp)
        jobs = [(p["src"], float(p["start"]), fr, str(t / f"p{k:04d}.mp4"), W, H, fps) for k, (p, fr) in enumerate(zip(pieces, frames))]
        ok = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for k, good in enumerate(ex.map(_cut_picture, jobs), 1):
                ok.append(good)
                ac.stage_progress(0.6 * k / len(jobs))
        bad = [k for k, good in enumerate(ok) if not good]
        if bad:
            p = pieces[bad[0]]
            raise AutocutError(f"{len(bad)} piece(s) could not be cut from the recordings (the first: {Path(p['src']).name} at {p['start']:.0f} s)")
        (t / "list.txt").write_text("".join(f"file '{j[3]}'\n" for j in jobs), encoding="utf-8")
        ac.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(t / "list.txt"), "-c", "copy", str(t / "video.mp4")])
        total = sound_tracks(pieces, frames, fps, voice, t / "game.wav", t / "voice.wav")
        ac.stage_progress(0.75)
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(t / "video.mp4"), "-i", str(t / "game.wav"), "-i", str(t / "voice.wav")]
        graph = ["[2:a]asplit=2[v1][v2]"]
        bed = "[1:a]"
        if music:
            import music as mu

            m = mu.analyse(music)
            cmd += ["-ss", f"{mu.choose_window(m, total):.3f}", "-t", f"{total + 0.5:.3f}", "-i", str(Path(music).resolve())]
            graph.append(f"[3:a]aresample={SR},apad,atrim=0:{total:.4f},afade=t=in:d=1.0,afade=t=out:st={max(0, total - 2.5):.4f}:d=2.5,volume={10 ** (music_db / 20):.4f}[mus]")
            graph.append("[1:a][mus]amix=inputs=2:normalize=0[bedmix]")
            bed = "[bedmix]"
        graph.append(f"{bed}[v2]sidechaincompress=threshold=0.03:ratio=5:attack=15:release=400[duck]")
        graph.append(f"[v1][duck]amix=inputs=2:normalize=0:duration=first,loudnorm=I=-14:TP=-1.5:LRA=11,aresample={SR},atrim=0:{total:.4f}[ao]")
        framed = tall_frame is not None and W > H
        OW, OH = tall_size if framed else (W, H)
        e = str(ass).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'") if ass else ""
        if framed:
            import narrate as nr

            graph.append(nr.tall_frame_filter(W, H, OW, OH, tall_frame, frame_blur) + (f";[framed]subtitles=filename='{e}'[vo]" if ass else ";[framed]null[vo]"))
        elif ass:
            graph.append(f"[0:v]subtitles=filename='{e}'[vo]")
        vmap = "[vo]" if framed or ass else "0:v"
        g = t / "graph.txt"
        g.write_text(";\n".join(graph), encoding="utf-8")
        partial = outp.with_name(outp.stem + ".part.mp4")
        cmd += ac.filter_script_args(str(g)) + ["-map", vmap, "-map", "[ao]"]
        cmd += [*ac.video_encoder_args("h264", ac.bitrate_for(OW, OH))] if vmap != "0:v" else ["-c:v", "copy"]
        ac.run(cmd + ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(partial)])
        partial.replace(outp)
    ac.stage_progress(1.0)
    log(f"made {outp.name}: {total:.0f} s from {len(pieces)} piece(s)")
    return {"seconds": total, "pieces": len(pieces)}
