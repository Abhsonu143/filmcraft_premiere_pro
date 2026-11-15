"""Repair a finished video whose voice has a sound in its pauses.

Before the room-tone check in `voice.py`, a cloned voice without a quiet moment in its recordings got a
piece of a word looped into every pause between passages (a burst every few seconds). The spoken takes are
gone once a video is made, so the voice cannot simply be put together again. But what was put into each
pause is known exactly: it is in the voice file. So:

- the folder is copied (the original is never changed);
- in the copy's voice file the pauses are made silent;
- in the copy's video, per pause, the burst is found in the sound track, a short filter from burst to
  mix is fitted (the mix chain's gain and tone), and the fit is subtracted. The picture is not re-encoded.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

import autocut as ac
from autocut import AutocutError, log

QUIET_DB = -45.0  # a pause louder than this (20 ms frames) has something in it
TAPS = 48  # the fitted filter reaches this many samples each way
MIN_MATCH = 0.5  # how clearly the burst must show in the mix to be subtracted there


def _db(v: np.ndarray) -> float:
    return 20 * float(np.log10(max(1e-9, float(np.sqrt(np.mean(np.square(v, dtype=np.float64)))) if len(v) else 1e-9)))


def _peak_frame_db(v: np.ndarray, hop: int) -> float:
    n = len(v) // hop
    if n == 0:
        return -180.0
    r = np.sqrt((v[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(axis=1))
    return 20 * float(np.log10(max(1e-9, float(r.max()))))


def find_video(folder: Path) -> tuple[Path, Path | None]:
    """The narration of a topic/voiceover folder and the video made from it (None: audio only)."""
    cands = sorted(folder.glob("*.narration.json"))
    if not cands:
        raise AutocutError(f"{folder.name} has no narration (.narration.json): choose a folder made by Video from Topic, a voiceover edit or Speak")
    nar = cands[0]
    base = nar.name[: -len(".narration.json")]
    video = folder / f"{base}.mp4"
    return nar, video if video.is_file() else None


def pauses(narration: dict, timeline: dict | None, voice_seconds: float) -> list[tuple[float, float, float]]:
    """(start, end, shift) of every pause between spoken pieces, in the voice file's time; `shift` is
    how much later it is heard in the video (a cold open put in front moves everything)."""
    segs = [s for s in narration.get("segments") or [] if float(s.get("end", 0)) > float(s.get("start", 0))]
    norm = lambda t: " ".join(str(t).split()).lower()  # noqa: E731
    where = {}
    for ln in (timeline or {}).get("lines") or []:  # (a line repeated as the cold open: its later place is the real one)
        where[norm(ln.get("text", ""))] = float(ln.get("start", 0))
    shift_of = lambda s: where.get(norm(s.get("text", "")), float(s["start"])) - float(s["start"])  # noqa: E731
    out = []
    for a, b in zip(segs, segs[1:]):
        pa, pb = (a.get("check") or {}).get("passage"), (b.get("check") or {}).get("passage")
        if (pa is None or pa != pb) and float(b["start"]) - float(a["end"]) > 0.05:
            out.append((float(a["end"]), float(b["start"]), shift_of(b)))
    if segs and voice_seconds - float(segs[-1]["end"]) > 0.05:
        out.append((float(segs[-1]["end"]), voice_seconds, shift_of(segs[-1])))
    return out


def dirty(voice: np.ndarray, sr: int, gaps: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """The pauses that are not quiet."""
    edge = int(0.01 * sr)
    return [g for g in gaps if _peak_frame_db(voice[int(g[0] * sr) + edge : max(int(g[0] * sr) + edge, int(g[1] * sr) - edge)], sr // 50) > QUIET_DB]


def clean_voice(voice: np.ndarray, sr: int, gaps: list[tuple[float, float, float]]) -> tuple[np.ndarray, list[np.ndarray]]:
    """The voice with those pauses silent (4 ms fades at their edges), and what was taken out of each."""
    out, bursts = voice.copy(), []
    fade = max(1, int(0.004 * sr))
    for g0, g1, _ in gaps:
        i0, i1 = int(g0 * sr), min(len(voice), int(g1 * sr))
        if i1 - i0 <= 2 * fade:
            bursts.append(np.zeros(0))
            continue
        env = np.zeros(i1 - i0)
        env[:fade] = np.linspace(1, 0, fade)
        env[-fade:] = np.linspace(0, 1, fade)
        bursts.append(voice[i0:i1] * (1 - env))
        out[i0:i1] = voice[i0:i1] * env
    return out, bursts


def subtract(mix: np.ndarray, sr: int, gaps: list[tuple[float, float, float]], bursts: list[np.ndarray]) -> dict:
    """Take each burst out of the mix (in place). Returns how many pauses were repaired and the levels."""
    before, after, left = [], [], 0
    span = int(0.4 * sr)
    for (g0, _, shift), burst in zip(gaps, bursts):
        if len(burst) < 4 * TAPS or not burst.any():
            continue
        m0 = int((g0 + shift) * sr)
        lo = max(0, m0 - span)
        ref = mix[lo : m0 + span + len(burst)].mean(axis=1, dtype=np.float64)
        if len(ref) < len(burst) + 10:
            left += 1
            continue
        corr = np.correlate(ref, burst, mode="valid")
        k = int(np.argmax(np.abs(corr)))
        start = lo + k
        match = abs(float(corr[k])) / (float(np.linalg.norm(burst)) * float(np.linalg.norm(ref[k : k + len(burst)])) + 1e-12)
        if start + len(burst) + TAPS > len(mix):  # the pause at the very end: as much of it as the video still has
            burst = burst[: max(0, len(mix) - start - TAPS)]
        seg = mix[start - TAPS : start + len(burst) + TAPS]
        if match < MIN_MATCH or start - TAPS < 0 or len(burst) < 4 * TAPS or len(seg) < len(burst) + 2 * TAPS:
            left += 1  # not found here (under a loud effect, or already clean): this pause is left as it is
            continue
        padded = np.concatenate([np.zeros(2 * TAPS), burst, np.zeros(2 * TAPS)])
        cols = np.stack([padded[j : j + len(seg)] for j in range(2 * TAPS, -1, -1)], axis=1)
        before.append(_db(seg))
        for c in range(mix.shape[1]):
            h, *_ = np.linalg.lstsq(cols, seg[:, c].astype(np.float64), rcond=None)
            seg[:, c] = seg[:, c] - (cols @ h).astype(mix.dtype)
        after.append(_db(seg))
    return {"repaired": len(after), "left": left, "before_db": round(float(np.median(before)), 1) if before else None,
            "after_db": round(float(np.median(after)), 1) if after else None, "worst_after_db": round(max(after), 1) if after else None}


def _read_voice(path: Path) -> tuple[int, np.ndarray]:
    try:
        with wave.open(str(path)) as w:
            if w.getsampwidth() != 2:
                raise AutocutError("the voice file is not 16-bit")
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
            return w.getframerate(), x.reshape(-1, w.getnchannels()).mean(axis=1)
    except (wave.Error, EOFError, OSError) as e:
        raise AutocutError(f"the voice file could not be read: {e}") from e


def _copy_folder(src: Path) -> Path:
    dst = src.with_name(src.name + " - fixed")
    k = 2
    while dst.exists():
        dst = src.with_name(f"{src.name} - fixed {k}")
        k += 1
    # a copy-on-write clone where the disk can (instant, no extra space until something changes)
    if subprocess.run(["cp", "-cR", str(src), str(dst)], capture_output=True).returncode != 0:
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
    return dst


def repair(folder: str, *, force: bool = False) -> dict:
    src = Path(folder).expanduser().resolve()
    if src.is_file():
        src = src.parent
    if not src.is_dir():
        raise AutocutError(f"no such folder: {folder}")
    ac.progress_plan({"check": 0.1, "copy": 0.1, "sound": 0.8})
    with ac.stage("check"):
        nar_path, video = find_video(src)
        base = nar_path.name[: -len(".narration.json")]
        voice_path = src / f"{base}.wav"
        if not voice_path.is_file():
            raise AutocutError(f"the voice file {voice_path.name} is not in the folder")
        narration = json.loads(nar_path.read_text(encoding="utf-8"))
        tl_path = src / f"{base}.narrate.json"
        timeline = json.loads(tl_path.read_text(encoding="utf-8")) if tl_path.is_file() else None
        sr, voice = _read_voice(voice_path)
        every = pauses(narration, timeline, len(voice) / sr)
        gaps = every if force else dirty(voice, sr, every)
        log(f"pause repair: {len(gaps)} of {len(every)} pauses have a sound in them")
        if not gaps:
            return {"outputs": [], "repaired": 0, "pauses": len(every), "message": "The pauses of this voice are quiet: nothing to repair, nothing was copied."}
    with ac.stage("copy"):
        dst = _copy_folder(src)
        log(f"pause repair: working in a copy, {dst.name}")
    with ac.stage("sound"):
        fixed, bursts = clean_voice(voice, sr, gaps)
        with wave.open(str(dst / voice_path.name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(fixed, -1, 1) * 32767).astype(np.int16).tobytes())
        outputs, stats = [str(dst / voice_path.name)], {"repaired": len(gaps), "left": 0}
        if video is not None:
            with tempfile.TemporaryDirectory(prefix="ei-pauses-") as tmp:
                raw, out_raw, tmp_video = Path(tmp) / "mix.f32", Path(tmp) / "fixed.f32", dst / f"{base}.repairing.mp4"
                ac.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vn", "-ac", "2", "-ar", str(sr), "-f", "f32le", str(raw)])
                ac.stage_progress(0.25)
                mix = np.fromfile(raw, np.float32).reshape(-1, 2)
                stats = subtract(mix, sr, gaps, bursts)
                ac.stage_progress(0.7)
                mix.tofile(out_raw)
                del mix
                ac.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", str(out_raw),
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "224k", "-movflags", "+faststart", str(tmp_video)])
                tmp_video.replace(dst / video.name)
            outputs.insert(0, str(dst / video.name))
            log(f"pause repair: {stats['repaired']} pauses repaired, {stats['left']} left as they were; level in them {stats['before_db']} dB -> {stats['after_db']} dB")
    (dst / "pause-repair.json").write_text(json.dumps({"from": str(src), "pauses": len(every), "with_sound": len(gaps), **stats}, indent=1), encoding="utf-8")
    return {"outputs": outputs, "folder": str(dst), "pauses": len(every), "with_sound": len(gaps), **stats}


def add_args(p) -> None:
    p.add_argument("--folder", required=True, help="a folder made by Video from Topic, a voiceover edit or Speak (or a file in it)")
    p.add_argument("--force", action="store_true", help="treat every pause as one with a sound in it")


def run_args(a) -> dict:
    res = repair(a.folder, force=a.force)
    if res.get("message"):
        log(res["message"])
    return res
