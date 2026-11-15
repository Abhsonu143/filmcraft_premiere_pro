#!/usr/bin/env python3
"""voice: clone your voice (Hindi and English) and speak scripts with it. Everything runs locally.

Enrolment (making a voice)
  * Record: EI shows sentences to read (voice_prompts.py: every sound of the language and the
    deliveries a video needs: calm, excited, curious, serious, warm, suspense …). Each take is
    checked: was the right sentence read (Whisper), is it loud enough, clipped, noisy?
  * Or give existing recordings/videos of your voice: the speech is cut into clean phrases
    (3–12 s), music/noise/overlapping parts are scored down, every phrase is transcribed.
  Each phrase becomes a reference with its language, delivery (emotion), pitch, energy, pace and
  quality. Saved as a voice profile in `~/Library/Application Support/EI/voices/<name>/`.

Speaking (text -> your voice)
  For every line of a script (script_writer.py JSON, or plain text):
  1. the reference that fits best is chosen: same language, same delivery (an excited line is
     cloned from an excited take of yours), good quality;
  2. the clone model generates several takes (the delivery set from the line's emotion/energy);
  3. each take is checked and the best kept: it must say the words (Whisper, character error
     rate), sound like you (speaker-verification similarity to your whole profile), and have a
     natural length; a bad line is regenerated;
  4. lines are joined with the script's pauses, levelled, and given your microphone's room tone,
     so it sounds like one recording, not stitched sentences.
  Output: `<name>.wav` plus `<name>.narration.json` (each line's time, text, emotion and visual
  hint) for the voiceover editor.

Model: Chatterbox Multilingual v3 (Resemble AI, MIT licence, ~2.7 GB download, ~4.8 GB while
speaking): 23 languages incl. Hindi, with an emotion-intensity control. Runs on its own: other
models are unloaded first.

Use it only for your own voice (or with the speaker's consent).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

import numpy as np

import autocut as ac
import emotions as em
from autocut import AutocutError, log

SR = 24000
# Measured on an 8 GB M3 (Hindi + English, FLEURS speakers): Chatterbox Multilingual v3 kept Hindi
# intelligible (Whisper CER 0.14–0.29) and English exact, with speaker similarity 0.70–0.83;
# OmniVoice garbled Hindi (CER ~0.8) and needed 5.6 GB, so it is not offered.
ENGINES = {
    "chatterbox": {"repo": "mlx-community/chatterbox-multilingual-v3", "license": "MIT", "label": "Chatterbox Multilingual v3"},
}
CHECK_WHISPER = {"hi": "mlx-community/whisper-large-v3-turbo", "en": ac.DEFAULT_WHISPER}  # Hindi needs the bigger ear
DEFAULT_ENGINE = "chatterbox"
EMOTION_ENERGY = {"calm": 0.25, "neutral": 0.4, "warm": 0.4, "story": 0.45, "serious": 0.45, "curious": 0.5, "suspense": 0.4, "cta": 0.6, "hook": 0.65, "excited": 0.85}


def voices_dir() -> Path:
    base = Path(os.environ["EI_SETTINGS"]).parent if os.environ.get("EI_SETTINGS") else Path.home() / "Library" / "Application Support" / "EI"
    return base / "voices"


def slug(name: str) -> str:
    return re.sub(r"[^\w\-]+", "-", name.strip()).strip("-")[:60] or "voice"


# ---------------------------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------------------------


def read_wav(path: str | Path) -> np.ndarray:
    """Mono float32 at SR (any input format, via ffmpeg)."""
    proc = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                          capture_output=True)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def write_wav(path: str | Path, y: np.ndarray, sr: int = SR) -> None:
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


def frame_db(y: np.ndarray, hop: int = SR // 50) -> np.ndarray:
    n = len(y) // hop
    if n == 0:
        return np.zeros(0)
    r = np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    return 20 * np.log10(np.maximum(r, 1e-6))


ROOM_TONE_MAX_DB = -45.0  # louder than this is not a quiet room: it is speech, breath or music


def room_tone(clips: list[np.ndarray]) -> np.ndarray:
    """The quietest 0.2 s of each clip, kept only where it really is quiet. Clips that are speech from
    end to end (short, tightly cut references) give nothing: a "quietest" bit of them is a piece of a word."""
    tone = []
    for y in clips:
        db = frame_db(y)
        if len(db) > 10:
            i = int(np.argmin(np.convolve(db, np.ones(10) / 10, mode="valid")))
            if float(db[i : i + 10].max()) <= ROOM_TONE_MAX_DB:
                tone.append(y[i * (SR // 50) : (i + 10) * (SR // 50)])
    return np.concatenate(tone) if tone else np.zeros(0, np.float32)


def pause_fill(tone: np.ndarray, n: int) -> np.ndarray:
    """`n` samples for a pause between passages: the room tone if it is one, else silence.
    (A saved room tone that is not quiet, from before the check above, must never be heard.)"""
    if n <= 0:
        return np.zeros(0, np.float32)
    db = frame_db(tone)
    if len(tone) == 0 or len(db) == 0 or float(db.max()) > ROOM_TONE_MAX_DB:
        return np.zeros(n, np.float32)
    return (np.tile(tone, int(np.ceil(n / len(tone))))[:n] * 0.9).astype(np.float32)


def pitch_track(y: np.ndarray, sr: int = SR, hop: int = 480, lo: float = 70, hi: float = 400) -> np.ndarray:
    """F0 per 20 ms frame (Hz, 0 = unvoiced): normalised autocorrelation, cheap and robust enough
    for a speaker's median pitch and range."""
    win = int(sr * 0.04)
    out = []
    lag_lo, lag_hi = int(sr / hi), int(sr / lo)
    for i in range(0, max(0, len(y) - win), hop):
        f = y[i : i + win] - y[i : i + win].mean()
        e = float((f * f).sum())
        if e < 1e-4:
            out.append(0.0)
            continue
        ac_ = np.correlate(f, f, mode="full")[win - 1 :]
        seg = ac_[lag_lo : min(lag_hi, len(ac_) - 1)]
        if len(seg) == 0:
            out.append(0.0)
            continue
        k = int(np.argmax(seg))
        out.append(sr / (lag_lo + k) if seg[k] / ac_[0] > 0.45 else 0.0)
    return np.asarray(out)


def features(y: np.ndarray, words: int = 0) -> dict:
    db = frame_db(y)
    if len(db) == 0:
        return {"dur": 0.0}
    speech = db > (np.percentile(db, 95) - 30)
    noise = float(np.percentile(db, 10))
    level = float(np.median(db[speech])) if speech.any() else -60.0
    f0 = pitch_track(y)
    voiced = f0[f0 > 0]
    dur = len(y) / SR
    return {
        "dur": round(dur, 2),
        "level_db": round(level, 1),
        "snr_db": round(level - noise, 1),
        "clipped": round(float((np.abs(y) > 0.985).mean()), 5),
        "pitch_hz": round(float(np.median(voiced)), 1) if len(voiced) > 5 else None,
        "pitch_range": round(float(np.percentile(voiced, 90) / max(1.0, np.percentile(voiced, 10))), 2) if len(voiced) > 5 else None,
        "energy_var": round(float(db[speech].std()) if speech.any() else 0.0, 2),
        "words_per_s": round(words / max(0.5, float(speech.sum()) * 0.02), 2) if words else None,
    }


def trim_silence(y: np.ndarray, pad: float = 0.08, thr_below_peak: float = 38) -> np.ndarray:
    db = frame_db(y)
    if len(db) == 0:
        return y
    on = np.nonzero(db > db.max() - thr_below_peak)[0]
    if len(on) == 0:
        return y
    hop = SR // 50
    a = max(0, on[0] * hop - int(pad * SR))
    b = min(len(y), (on[-1] + 1) * hop + int(pad * SR))
    return y[a:b]


def clean_file(src: str, dst: Path, denoise: bool | None = None) -> None:
    """Mono 24 kHz, rumble removed, consistent level (originals untouched). Noise reduction only
    when the recording needs it (SNR under 30 dB): on a clean recording it only adds a faint
    artificial sound that the clone would copy."""
    base = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", str(SR)]
    ac.run(base + ["-af", "highpass=f=70,loudnorm=I=-20:TP=-2:LRA=11", "-c:a", "pcm_s16le", str(dst)])
    if denoise is None:
        denoise = features(read_wav(dst)).get("snr_db", 99) < 30
    if denoise:
        ac.run(base + ["-af", "highpass=f=70,afftdn=nr=8:nf=-45,loudnorm=I=-20:TP=-2:LRA=11", "-c:a", "pcm_s16le", str(dst)])


def norm_text(t: str) -> str:
    t = unicodedata.normalize("NFC", t.lower())
    return re.sub(r"[^\w]", "", t)


def cer(hyp: str, ref: str) -> float:
    a, b = norm_text(hyp), norm_text(ref)
    if not b:
        return 0.0 if not a else 1.0
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
            prev, d[j] = d[j], cur
    return d[len(b)] / len(b)


HINDI_NUMBERS = set("एक दो तीन चार पांच पाँच छह छः सात आठ नौ दस ग्यारह बारह पंद्रह बीस पच्चीस तीस चालीस पचास साठ सत्तर अस्सी नब्बे सौ हज़ार हजार लाख करोड़".split())


def _sim(a: str, b: str) -> float:
    return 1 - cer(a, b) if b else 0.0


def speech_error(heard: str, text: str) -> float:
    """How much of the line was not said (0 = all of it).
    Hindi lines: Whisper writes English words in Devanagari ("battery" -> "बैटरी") and numbers as digits,
    so only the line's Hindi words are checked, each fuzzily against what was heard (spelling-tolerant).
    English lines: character error rate."""
    hindi = [w for w in re.findall(r"[\u0900-\u097F]+", text) if len(w) > 1]
    if re.search(r"\d", heard):  # "तीस" heard as "30"
        hindi = [w for w in hindi if w not in HINDI_NUMBERS]
    if not hindi:
        return cer(heard, text)
    hw = re.findall(r"[\u0900-\u097F]+", heard)
    if not hw:
        return 1.0
    found = sum(1 for w in hindi if max(_sim(h, w) for h in hw) >= 0.6)
    return 1 - found / len(hindi)


def detect_lang(text: str) -> str:
    return "hi" if re.search(r"[ऀ-ॿ]", text) else "en"


# ---------------------------------------------------------------------------------------------
# Microphone
# ---------------------------------------------------------------------------------------------


def list_mics() -> list[str]:
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""], capture_output=True, text=True)
    mics, on = [], False
    for line in proc.stderr.splitlines():
        if "audio devices" in line.lower():
            on = True
            continue
        m = re.search(r"\[(\d+)\] (.+)$", line)
        if on and m:
            mics.append(f"{m.group(1)}: {m.group(2).strip()}")
    return mics


MIC_WAIT = ("No sound from the microphone yet. If macOS asked whether EI may use the microphone, click Allow, then press "
            "Record again. Otherwise open System Settings ▸ Privacy & Security ▸ Microphone and turn on EI.")
MIC_DENIED = ("EI is not allowed to use the microphone (macOS gives it silence). Open System Settings ▸ Privacy & Security ▸ "
              "Microphone, turn on EI, then quit and reopen EI.")


def detect_speech(dbs: list[float], start_margin: float = 12.0, floor_db: float = -55.0) -> int | None:
    """Index of the first 50-ms chunk that is clearly speech: louder than the room by `start_margin` dB.
    The room level is the quiet end (20th percentile) of what was heard, so talking right away
    (before any silence) is still recognised."""
    if not dbs:
        return None
    room = float(np.percentile(dbs, 20))
    for i, d in enumerate(dbs):
        if d > max(floor_db, room + start_margin):
            return i
    return None


def quiet_after(dbs: list[float], quiet: float, dt: float) -> float:
    """Seconds of quiet so far after speech. Quiet = the last 0.25 s (smoothed) is near the room
    level, well below the speech level; a short noise blip does not restart the count (hysteresis),
    so a fan or street noise does not keep the recording going."""
    room = float(np.percentile(dbs, 20))
    loud = float(np.percentile(dbs, 90))
    thr = max(room + 6, min(loud - 15, room + 12), -60.0)
    level = float(np.mean(dbs[-5:]))
    if level < thr:
        return quiet + dt
    if level > thr + 4:
        return 0.0
    return quiet


def record(out: str, max_s: float = 30.0, stop_after: float = 1.4, mic: str = "0", wait_s: float = 12.0, stop_file: str | None = None) -> dict:
    """Record from the microphone until you stop talking (stop_after seconds of quiet after speech),
    until `stop_file` appears (the app's Stop button), or max_s. A take with too little speech is
    still saved (with a warning) so it can be played back. Returns level/noise/clipping checks."""
    rate = 48000
    stop = Path(stop_file) if stop_file else None
    if stop and stop.exists():
        stop.unlink()
    proc = subprocess.Popen(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "avfoundation", "-i", f":{mic}",
                             "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    chunk = rate // 20  # 50 ms
    buf: list[np.ndarray] = []
    dbs: list[float] = []
    started, quiet, t, peak = False, 0.0, 0.0, 0.0
    next_report = 0.5
    log("recording: speak now")
    import os
    import select

    parent = os.getppid()
    fd = proc.stdout.fileno() if proc.stdout else -1
    pending = b""
    waited = 0.0
    try:
        while t < max_s:
            # never block on the microphone: macOS sends nothing while its permission question is open
            ready, _, _ = select.select([fd], [], [], 0.25) if fd >= 0 else ([], [], [])
            if os.getppid() != parent:
                raise AutocutError("EI closed: recording stopped")
            if stop and stop.exists() and not buf:
                break
            if not ready:
                waited += 0.25
                if waited >= 5.0 and not buf:
                    raise AutocutError(MIC_WAIT)
                if waited >= 5.0:
                    break  # the microphone went away mid-take: keep what was recorded
                continue
            waited = 0.0
            data = os.read(fd, chunk * 2)
            if not data:
                break
            pending += data
            if len(pending) < chunk * 2:
                continue
            raw, pending = pending[: chunk * 2], pending[chunk * 2 :]
            x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
            buf.append(x)
            t += len(x) / rate
            peak = max(peak, float(np.abs(x).max()) if len(x) else 0.0)
            db = 20 * math.log10(max(1e-6, float(np.sqrt((x * x).mean()))))
            dbs.append(db)
            if t >= 1.5 and peak == 0.0:
                raise AutocutError(MIC_DENIED)  # exact digital silence: permission, not a quiet room
            if not started:
                i = detect_speech(dbs) if t > 0.25 else None
                if i is not None:
                    started = True
                    log(f"hearing you ({db:.0f} dB)")
                elif t > wait_s:
                    break
            else:
                quiet = quiet_after(dbs, quiet, len(x) / rate)
                if quiet >= stop_after:
                    break
            if t >= next_report:  # a live level for the app
                log(f"{'hearing you' if started else 'listening'} ({db:.0f} dB)")
                next_report += 0.5
            if stop and stop.exists():
                log("stopped")
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        if stop and stop.exists():
            stop.unlink()
    err = (proc.stderr.read().decode(errors="replace") if proc.stderr else "").strip()
    if not buf:
        raise AutocutError(f"no sound from the microphone{(': ' + err[:160]) if err else ''}. {MIC_DENIED}")
    y = np.concatenate(buf)
    if peak == 0.0:
        raise AutocutError(MIC_DENIED)
    issues = []
    if started:
        y = trim_silence(y, pad=0.15)
    else:
        issues.append("no clear speech heard: check the microphone choice, speak closer, then Record Again")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_wav(out, y, rate)
    f = features(np.interp(np.arange(0, len(y), rate / SR), np.arange(len(y)), y).astype(np.float32))
    if f.get("clipped", 0) > 0.001:
        issues.append("too loud (clipping): move back from the microphone a little")
    if started and f.get("level_db", -60) < -38:
        issues.append("quiet: move closer to the microphone")
    if started and f.get("snr_db", 99) < 25:
        issues.append("background noise: a quieter room helps the clone a lot")
    log(f"recorded {len(y) / rate:.1f}s" + (f" ({'; '.join(issues)})" if issues else ""))
    return {"outputs": [out], "file": out, "seconds": round(len(y) / rate, 2), "features": f, "issues": issues, "speech": started}


# ---------------------------------------------------------------------------------------------
# Speaker similarity (ECAPA speaker verification, independent of the clone model)
# ---------------------------------------------------------------------------------------------


def _embed_worker(paths: list[str]) -> list[list[float] | None]:
    import numpy as np
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ac.CACHE_DIR / "ecapa"), run_opts={"device": "cpu"})
    out = []
    for p in paths:
        raw = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", p, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                             capture_output=True).stdout
        if len(raw) < 16000 * 4 * 0.5:
            out.append(None)
            continue
        w = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())[None, :]
        with torch.no_grad():
            e = enc.encode_batch(w).squeeze()
        out.append((e / e.norm()).tolist())
    return out


def embed(paths: list[str]) -> list[np.ndarray | None]:
    try:
        import speechbrain  # noqa: F401
    except ImportError:
        return [None] * len(paths)
    try:
        res = ac.run_isolated(_embed_worker, paths)
    except Exception as e:
        log(f"speaker check unavailable ({type(e).__name__})")
        return [None] * len(paths)
    return [np.asarray(r) if r is not None else None for r in res]


def _asr_worker(items: list[tuple[str, str]], models: dict[str, str]) -> list[str]:
    import mlx_whisper

    return [mlx_whisper.transcribe(p, path_or_hf_repo=models.get(lang, models["en"]), language=lang, condition_on_previous_text=False)["text"]
            for p, lang in items]


def asr(items: list[tuple[str, str]], models: dict[str, str] | None = None) -> list[str]:
    """What Whisper hears in each file (Hindi checked with large-v3-turbo: small mishears Hindi)."""
    if not items:
        return []
    return ac.run_isolated(_asr_worker, items, models or CHECK_WHISPER)


# ---------------------------------------------------------------------------------------------
# Enrolment
# ---------------------------------------------------------------------------------------------


def emotion_from(f: dict, base: dict, text: str) -> str:
    """Rough delivery of a phrase from how it differs from the speaker's usual way of talking."""
    if text.rstrip().endswith("?"):
        return "curious"
    e = f.get("level_db", 0) - base.get("level_db", 0)
    p = (f.get("pitch_range") or 1.5) / max(1.0, base.get("pitch_range") or 1.5)
    r = (f.get("words_per_s") or 2.5) / max(0.5, base.get("words_per_s") or 2.5)
    if e > 3 and p > 1.1:
        return "excited"
    if e < -2.5 and r < 0.95:
        return "calm"
    if r > 1.12:
        return "story"
    return "neutral"


def phrase_cuts(words: list[dict], lo: float = 3.0, hi: float = 12.0, gap: float = 0.25) -> list[tuple[float, float, str]]:
    """Clean phrases for references: whole phrases 3–12 s split at pauses."""
    out = []
    cur: list[dict] = []
    for w in words:
        if cur and (w["start"] - cur[-1]["end"] > gap and cur[-1]["end"] - cur[0]["start"] >= lo or w["end"] - cur[0]["start"] > hi):
            if cur[-1]["end"] - cur[0]["start"] >= lo:
                out.append((cur[0]["start"], cur[-1]["end"], " ".join(x["word"].strip() for x in cur)))
            cur = []
        cur.append(w)
    if cur and cur[-1]["end"] - cur[0]["start"] >= lo:
        out.append((cur[0]["start"], cur[-1]["end"], " ".join(x["word"].strip() for x in cur)))
    return out


def ref_quality(f: dict, cer_: float | None = None) -> float:
    q = 1.0
    q -= max(0.0, 30 - f.get("snr_db", 30)) * 0.03
    q -= min(0.5, f.get("clipped", 0) * 200)
    d = f.get("dur", 0)
    q -= 0.0 if 5 <= d <= 11 else 0.15 if 3 <= d < 5 or 11 < d <= 14 else 0.4
    if cer_ is not None:
        q -= min(0.6, cer_ * 1.5)
    return round(max(0.0, q), 3)


SEPARATE_BELOW = 20.0  # dB: a recording whose voice is less than this above the background has music under it
SEPARATE_PIECE = 10.0  # seconds separated at a time (with 1 s of overlap), so memory stays ~0.5 GB


def _separate_worker(items: list[tuple[str, str]]) -> list[bool]:
    """Runs in its own process: the voice of each recording without its music (Hybrid Demucs, MIT,
    via torchaudio), 10 seconds at a time with crossfaded overlaps."""
    import os
    import subprocess

    import numpy as np
    import soundfile as sf
    import torch

    try:
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())  # (the model downloads once)
    except ImportError:
        pass
    from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS as bundle

    model = bundle.get_model().eval()
    sr = bundle.sample_rate
    vi = model.sources.index("vocals")
    piece, lap = int(SEPARATE_PIECE * sr), sr
    out = []
    for k, (src, dst) in enumerate(items):
        raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", src, "-vn", "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout
        x = np.frombuffer(raw[: len(raw) // 8 * 8], np.float32).reshape(-1, 2).T
        n = x.shape[1]
        if n < sr:
            out.append(False)
            continue
        voc = np.zeros((2, n), np.float32)
        weight = np.zeros(n, np.float32)
        a = 0
        while a < n:
            b = min(n, a + piece + lap)
            seg = torch.from_numpy(x[:, a:b].copy())
            m, sd = seg.mean(), seg.std().clamp(min=1e-6)
            with torch.no_grad():
                y = (model(((seg - m) / sd)[None])[0, vi] * sd + m).numpy()
            w = np.ones(b - a, np.float32)
            if a > 0:
                w[: min(lap, b - a)] = np.linspace(0, 1, min(lap, b - a), dtype=np.float32)
            if b < n:
                w[-lap:] = np.minimum(w[-lap:], np.linspace(1, 0, lap, dtype=np.float32))
            voc[:, a:b] += y * w
            weight[a:b] += w
            a += piece
        sf.write(dst, (voc / np.maximum(weight, 1e-6)).T, sr)
        out.append(True)
        ac.stage_progress((k + 1) / len(items), f"took the music out of {k + 1}/{len(items)}")
    return out


def music_under(src: str, tmp: Path) -> float:
    """How far (dB) the voice sits above the rest of a recording (low: music under it)."""
    probe = tmp / "probe.wav"
    clean_file(src, probe, denoise=False)
    return float(features(read_wav(probe)).get("snr_db", 99))


def learn_style(name: str, files: list[str] | None) -> dict:
    """The talking style of the same recordings (pace, sentence length, hooks, fillers, endings),
    saved as a style of the same name for the script writer. Free talking only: read prompts are
    someone else's sentences."""
    import script_writer as sw

    if not files:
        raise AutocutError("a talking style is learned from recordings of free talking: add recordings (audio or video)")
    srcs = sw.load_sources(list(files))
    if not srcs:
        raise AutocutError("no speech found in the recordings to learn a talking style from")
    prof = sw.analyse_style(srcs)
    path = sw.save_style(name, prof)
    log(f"talking style '{name}': {prof['words']} words from {len(srcs)} recording(s), {prof['language']}, {prof['words_per_minute'] or '?'} words a minute")
    return {"style": str(path), "style_profile": {k: prof.get(k) for k in ("language", "words", "words_per_minute", "avg_sentence_words")}}


def merge_style(res: dict, st: dict) -> dict:
    out = {**res, **st}
    out["outputs"] = [*res.get("outputs", []), st["style"]]
    out.setdefault("out_dir", str(Path(st["style"]).parent))
    return out


def enroll(name: str, files: list[str] | None = None, takes: list[dict] | None = None, languages: list[str] | None = None,
           engine: str = DEFAULT_ENGINE, add: bool = False, separate: str = "always") -> dict:
    """Build (or extend) a voice profile from recordings of your voice.
    files: any audio/video of you talking.  takes: [{file, text, lang, emotion}] read from the prompts."""
    ac.require_tools()
    if engine not in ENGINES:
        raise AutocutError(f"engine must be one of {', '.join(ENGINES)}")
    d = voices_dir() / slug(name)
    refs_dir = d / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    pf = d / "profile.json"
    prof = json.loads(pf.read_text(encoding="utf-8")) if (add and pf.is_file()) else {"name": name, "refs": []}
    if not add:
        for old in refs_dir.glob("*.wav"):
            old.unlink()
        prof["refs"] = []
    t0 = time.time()
    new: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="ei-voice-") as tmp:
        k = len(prof["refs"])
        for tk in takes or []:  # read prompts: one take = one reference, checked against the text
            if not Path(tk["file"]).is_file():
                continue
            k += 1
            dst = refs_dir / f"r{k:03d}.wav"
            clean_file(tk["file"], dst)
            y = trim_silence(read_wav(dst))
            write_wav(dst, y)
            new.append({"file": dst.name, "text": tk["text"], "lang": tk.get("lang") or detect_lang(tk["text"]), "emotion": tk.get("emotion", "neutral"),
                        "source": Path(tk["file"]).name, "kind": "read"})
        for src in files or []:
            if not Path(src).is_file():
                raise AutocutError(f"not found: {src}")
        # recordings with music under the voice (YouTube videos, Shorts): the voice is taken out of
        # them first, or most phrases fail the quality check and the clone copies the music
        voiced = {}
        if separate != "never" and files:
            todo = [s for s in files if separate == "always" or music_under(s, Path(tmp)) < SEPARATE_BELOW]
            if todo:
                log(f"taking the voice out of {len(todo)} of {len(files)} recording(s) (music, game sound and noise are left behind; once per recording)")
                pairs = [(s, str(Path(tmp) / f"v{i}.wav")) for i, s in enumerate(todo)]
                done = ac.run_isolated(_separate_worker, pairs)
                voiced = {s: d for (s, d), ok in zip(pairs, done) if ok}
        for src in files or []:  # free recordings: cut into phrases
            clean = Path(tmp) / f"c{len(new)}.wav"
            clean_file(voiced.get(src, src), clean)
            tr = ac.transcribe(str(clean), language=None)
            lang = tr.get("language") or "en"
            y = read_wav(clean)
            for a, b, text in phrase_cuts(tr.get("words", [])):
                seg = trim_silence(y[int(max(0, a - 0.1) * SR) : int((b + 0.15) * SR)])
                if len(seg) < 2.5 * SR:
                    continue
                k += 1
                dst = refs_dir / f"r{k:03d}.wav"
                write_wav(dst, seg)
                new.append({"file": dst.name, "text": text, "lang": "hi" if lang == "hi" or detect_lang(text) == "hi" else "en",
                            "emotion": None, "source": Path(src).name, "kind": "free"})
        if not new and not prof["refs"]:
            raise AutocutError("no usable speech found (need at least a few seconds of clear talking)")
        log(f"{len(new)} reference phrases: checking what was said and how")
        heard = asr([(str(refs_dir / r["file"]), r["lang"]) for r in new]) if new else []
    for r, h in zip(new, heard):
        y = read_wav(refs_dir / r["file"])
        f = features(y, len(r["text"].split()))
        r["features"] = f
        r["heard"] = h.strip()
        r["cer"] = round(cer(h, r["text"]), 3) if r["kind"] == "read" else None
        r["quality"] = ref_quality(f, r["cer"])
    allrefs = prof["refs"] + new
    base = {k2: float(np.median([r["features"].get(k2) or 0 for r in allrefs])) for k2 in ("level_db", "pitch_range", "words_per_s")}
    for r in new:
        if not r.get("emotion"):
            r["emotion"] = emotion_from(r["features"], base, r["text"])
    embs = embed([str(refs_dir / r["file"]) for r in allrefs])
    good = [e for e, r in zip(embs, allrefs) if e is not None and r["quality"] >= 0.5]
    if good:
        c = np.mean(good, axis=0)
        c /= np.linalg.norm(c) + 1e-9
        for r, e in zip(allrefs, embs):
            r["similarity"] = round(float(e @ c), 3) if e is not None else None
            if r["similarity"] is not None and r["similarity"] < 0.45:
                r["quality"] = round(r["quality"] * 0.3, 3)  # someone else talking, or heavy music
        np.save(d / "centroid.npy", c)
    langs = sorted({r["lang"] for r in allrefs if r["quality"] >= 0.4})
    stats = {}
    for lg in langs:
        rs = [r for r in allrefs if r["lang"] == lg and r["quality"] >= 0.4]
        stats[lg] = {"refs": len(rs), "minutes": round(sum(r["features"]["dur"] for r in rs) / 60, 1),
                     "pitch_hz": round(float(np.median([r["features"]["pitch_hz"] for r in rs if r["features"].get("pitch_hz")] or [0])), 1),
                     "words_per_s": round(float(np.median([r["features"]["words_per_s"] for r in rs if r["features"].get("words_per_s")] or [0])), 2),
                     "emotions": sorted({r["emotion"] for r in rs})}
    # room tone: the quietest stretch of the best references, used between lines
    tone = room_tone([read_wav(refs_dir / r["file"]) for r in sorted(allrefs, key=lambda r: -r["quality"])[:6]])
    if len(tone):
        write_wav(d / "roomtone.wav", tone)
    else:
        (d / "roomtone.wav").unlink(missing_ok=True)  # no quiet moment in these recordings: pauses are silent
    prof.update({"name": name, "engine": prof.get("engine", engine) if add else engine, "languages": languages or langs, "stats": stats,
                 "refs": allrefs, "updated": time.strftime("%Y-%m-%d %H:%M"), "version": 1})
    pf.write_text(json.dumps(prof, ensure_ascii=False, indent=1), encoding="utf-8")
    usable = sum(1 for r in allrefs if r["quality"] >= 0.4)
    tips = []
    for lg in languages or langs:
        m = stats.get(lg, {}).get("minutes", 0)
        if m < 2:
            tips.append(f"{lg}: only {m} min of good speech; 3–10 min makes a much more faithful voice")
    log(f"voice '{name}': {usable} good references ({', '.join(f'{lg} {s['minutes']} min' for lg, s in stats.items())}) in {time.time() - t0:.0f}s")
    for t in tips:
        log("tip: " + t)
    return {"outputs": [str(pf)], "voice": name, "profile": str(pf), "out_dir": str(d), "stats": stats, "tips": tips}


def load_profile(name: str) -> tuple[dict, Path]:
    d = voices_dir() / slug(name)
    pf = d / "profile.json"
    if not pf.is_file():
        raise AutocutError(f"voice '{name}' not found (make it with AI ▸ Voice Clone)")
    return json.loads(pf.read_text(encoding="utf-8")), d


def list_voices() -> list[dict]:
    d = voices_dir()
    out = []
    for pf in sorted(d.glob("*/profile.json")) if d.is_dir() else []:
        try:
            p = json.loads(pf.read_text(encoding="utf-8"))
        except ValueError:
            continue
        out.append({"name": p.get("name", pf.parent.name), "languages": p.get("languages", []), "stats": p.get("stats", {}), "engine": p.get("engine")})
    return out


# ---------------------------------------------------------------------------------------------
# Speaking
# ---------------------------------------------------------------------------------------------


def choose_ref(prof: dict, lang: str, emotion: str, energy: float, used: dict[str, int]) -> dict:
    """The reference that best fits a line: same language (else any), same delivery, good, fresh."""
    refs = [r for r in prof["refs"] if r.get("quality", 0) >= 0.4] or prof["refs"]
    if not refs:
        raise AutocutError("the voice has no usable references")
    lvl = np.median([r["features"].get("level_db", -20) for r in refs])

    def score(r: dict) -> float:
        s = r["quality"] + (0.15 * (r.get("similarity") or 0.7))
        s += 0.6 if r["lang"] == lang else 0.0
        s += 0.45 if r.get("emotion") == emotion else 0.0
        e_ref = float(np.clip(0.5 + (r["features"].get("level_db", lvl) - lvl) / 12, 0, 1))
        s -= 0.25 * abs(e_ref - energy)
        d = r["features"].get("dur", 6)
        s -= 0.1 if d < 4 else 0.0
        s -= 0.08 * used.get(r["file"], 0)  # spread across references: more natural variation
        return s

    return max(refs, key=score)


def line_energy(seg: dict) -> float:
    """How energetic a line is (0–1): its own energy/intensity, else from its emotion."""
    emo = seg.get("emotion", "neutral")
    if "energy" in seg:
        return float(seg["energy"])
    if emo in EMOTION_ENERGY:
        return EMOTION_ENERGY[emo]
    e = em.info(emo)
    return float(0.2 + 0.6 * e["arousal"] * (0.6 + 0.4 * float(seg.get("intensity", 0.6))))


def engine_params(seg: dict, engine: str) -> dict:
    if engine == "chatterbox" and seg.get("emotion") and seg.get("emotion") not in EMOTION_ENERGY:
        # the 42 emotions: their own delivery (expressiveness, pacing) at the line's intensity
        return em.voice_params(seg["emotion"], float(seg.get("intensity", 0.6)))
    energy = float(seg.get("energy", EMOTION_ENERGY.get(seg.get("emotion", "neutral"), 0.45)))
    if engine == "chatterbox":
        # one voice sample for the whole narration: the delivery comes from exaggeration (emotion
        # intensity) and cfg_weight (lower = freer, faster, more natural pacing), not from
        # switching samples
        exag = 0.35 + 0.4 * energy  # calm ~0.45 … excited ~0.7
        cfg = 0.3 if (seg.get("pace") == "fast" or energy >= 0.7) else 0.5 if seg.get("pace") != "slow" else 0.55
        return {"exaggeration": round(exag, 2), "cfg_weight": cfg, "temperature": 0.8}
    return {}


_MODELS: dict = {}  # (in a worker process) the clone model, loaded once for every round


def take_limits(text: str) -> tuple[float, float]:
    """(longest sensible audio in s, longest sensible generation time in s) for a passage: a take
    that runs past these is a runaway (the model repeating or babbling), not a slow take."""
    expected = max(1.0, len(text) / 13.0)  # ~13 characters per second of speech
    return max(8.0, 2.5 * expected + 3.0), max(120.0, 25.0 * expected)


def _speak_worker(jobs: list[dict], engine: str, repo: str) -> list[dict]:
    """In its own process: load the clone model once (kept for later rounds), generate every take.
    A take is written only when it is complete; a runaway take is dropped (its passage is retried)."""
    import time as _time

    import numpy as np
    from mlx_audio.tts.utils import load_model

    if repo not in _MODELS:
        _MODELS[repo] = load_model(repo)
    model = _MODELS[repo]
    out = []
    for k, j in enumerate(jobs):
        ac.progress_count(k, len(jobs), "takes")
        kw = {"text": j["text"], "ref_audio": j["ref"]}
        kw.update(lang_code=j["lang"], **j.get("params", {}))
        max_audio, max_time = take_limits(j["text"])
        try:
            import mlx.core as mx

            mx.random.seed(int(j.get("seed", 0)))
            sr = int(getattr(model, "sample_rate", 24000))
            chunks, got, t0, runaway = [], 0.0, _time.time(), ""
            for r in model.generate(**kw):
                a = np.array(r.audio, dtype=np.float32)
                chunks.append(a)
                got += len(a) / sr
                if got > max_audio:
                    runaway = f"runaway take ({got:.0f}s of audio for {len(j['text'])} characters)"
                    break
                if _time.time() - t0 > max_time:
                    runaway = f"take took over {max_time:.0f}s"
                    break
            if runaway:
                out.append({"out": j["out"], "ok": False, "error": runaway})
                continue
            y = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
            tmp = j["out"] + ".part.wav"
            write_wav(tmp, y, sr)
            os.replace(tmp, j["out"])  # (a take on disk is always whole: a retry can trust it)
            out.append({"out": j["out"], "ok": True})
        except Exception as e:  # one bad line must not lose the rest
            out.append({"out": j["out"], "ok": False, "error": f"{type(e).__name__}: {e}"[:200]})
    ac.progress_count(len(jobs), len(jobs), "takes")
    return out


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
# English number words as a Hindi speaker says them, in Devanagari (what the Hindi voice reads best)
DEVA_NUMBERS = {
    "zero": "ज़ीरो", "one": "वन", "two": "टू", "three": "थ्री", "four": "फ़ोर", "five": "फ़ाइव", "six": "सिक्स", "seven": "सेवन", "eight": "एट",
    "nine": "नाइन", "ten": "टेन", "eleven": "इलेवन", "twelve": "ट्वेल्व", "thirteen": "थर्टीन", "fourteen": "फ़ोर्टीन", "fifteen": "फ़िफ़्टीन",
    "sixteen": "सिक्सटीन", "seventeen": "सेवनटीन", "eighteen": "एटीन", "nineteen": "नाइनटीन", "twenty": "ट्वेंटी", "thirty": "थर्टी",
    "forty": "फ़ोर्टी", "fifty": "फ़िफ़्टी", "sixty": "सिक्सटी", "seventy": "सेवेंटी", "eighty": "एटी", "ninety": "नाइंटी", "hundred": "हंड्रेड",
    "thousand": "थाउज़ेंड", "million": "मिलियन", "billion": "बिलियन", "point": "पॉइंट", "percent": "परसेंट",
}


def number_words(n: int) -> str:
    """An integer in English words ("two hundred twelve", "one million five hundred thousand")."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + number_words(n % 100))
    for size, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= size:
            rest = n % size
            return number_words(n // size) + " " + name + ("" if rest == 0 else " " + number_words(rest))
    return str(n)


def year_words(n: int) -> str:
    """A year as it is said: 2026 "twenty twenty-six", 2005 "two thousand five", 1999 "nineteen ninety-nine"."""
    if 2000 <= n <= 2009:
        return number_words(n)
    hi_, lo = divmod(n, 100)
    return number_words(hi_) + " " + ("hundred" if lo == 0 else ("oh " + _ONES[lo] if lo < 10 else number_words(lo)))


def english_numbers(text: str, devanagari: bool = True) -> str:
    """Digits as English number words, even in a Hindi line (the creator wants numbers said in
    English): in Devanagari letters for the Hindi voice, or plain English."""
    def say(m: re.Match) -> str:
        whole, frac, pct = m.group(1), m.group(2), m.group(3)
        digits = whole.replace(",", "")
        if not digits.isdigit() or len(digits) > 12:
            return m.group(0)
        n = int(digits)
        words = year_words(n) if (len(digits) == 4 and "," not in whole and 1100 <= n <= 2099 and not frac and not pct) else number_words(n)
        if frac:
            words += " point " + " ".join(_ONES[int(c)] for c in frac[:4])
        if pct:
            words += " percent"
        if devanagari:
            words = " ".join(DEVA_NUMBERS.get(w, w) for w in words.split())
        return words

    return re.sub(r"(?<![\w.])(\d[\d,]*\d|\d)(?:\.(\d+))?(?:\s*(%))?(?![A-Za-z0-9])", say, text)


def script_segments(script: str) -> tuple[list[dict], str]:
    """Segments from a script_writer JSON, or plain text (one line/sentence per segment)."""
    p = Path(script)
    if p.is_file() and p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        segs = [s for s in data.get("segments", []) if str(s.get("text", "")).strip()]
        lang = data.get("language", "en")
        return segs, ("hi" if lang in ("hi", "hinglish") else "en")
    text = p.read_text(encoding="utf-8") if p.is_file() else script
    parts = [x.strip() for x in re.split(r"(?<=[.!?।])\s+|\n+", text) if x.strip()]
    segs = []
    for x in parts:
        # "[Suspense 0.8] text", "[awe] text" or plain text
        m = re.match(r"\[\s*([A-Za-z]+)(?:\s+([0-9.]+))?\s*\]\s*(.+)", x)
        if m:
            seg = {"text": m.group(3), "emotion": em.resolve(m.group(1)), "pause_after": 0.35}
            if m.group(2):
                try:
                    seg["intensity"] = min(1.0, max(0.0, float(m.group(2))))
                except ValueError:
                    pass
            segs.append(seg)
        else:
            segs.append({"text": x, "emotion": "Neutral", "pause_after": 0.35})
    return segs, ("hi" if detect_lang(text) == "hi" else "en")


MAC_VOICES = {"hi": "Lekha", "en": "Rishi"}  # built into macOS (Indian Hindi / Indian English)


def list_mac_voices() -> list[str]:
    """macOS voices for Hindi and Indian/US/UK English (quick, offline, not a clone)."""
    proc = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
    out = []
    for line in proc.stdout.splitlines():
        m = re.match(r"^(.+?)\s+(hi_IN|en_IN|en_US|en_GB)\s+#", line)
        if m and not re.search(r"Bad News|Bahh|Bells|Boing|Bubbles|Cellos|Wobble|Good News|Jester|Organ|Superstar|Trinoids|Whisper|Zarvox|Hysterical|Bells", m.group(1)):
            out.append(f"{m.group(1).strip()} ({m.group(2)})")
    return out


def speak_mac(voice: str, script: str, out: str, language: str | None = None) -> dict:
    """A script read by a macOS voice ("mac", "mac:Lekha", "mac:Rishi"…): same outputs as speak()."""
    segs, default_lang = script_segments(script)
    if not segs:
        raise AutocutError("the script is empty")
    name = voice.split(":", 1)[1].strip() if ":" in voice else ""
    outp = Path(out).with_suffix(".wav")
    outp.parent.mkdir(parents=True, exist_ok=True)
    parts, timeline, t = [], [], 0.0
    with tempfile.TemporaryDirectory(prefix="ei-say-") as tmp:
        for i, s in enumerate(segs):
            lang = language or ("hi" if detect_lang(s["text"]) == "hi" else default_lang)
            v = name or MAC_VOICES.get(lang, "Rishi")
            rate = {"slow": 165, "fast": 205}.get(s.get("pace", "normal"), 185)
            aiff = Path(tmp) / f"s{i}.aiff"
            subprocess.run(["say", "-v", v, "-r", str(rate), "-o", str(aiff), s["text"]], capture_output=True, check=False)
            y = trim_silence(read_wav(aiff), pad=0.04) if aiff.is_file() else np.zeros(int(0.5 * SR), np.float32)
            parts.append(y)
            dur = len(y) / SR
            timeline.append({"start": round(t, 3), "end": round(t + dur, 3), "text": s["text"], "emotion": s.get("emotion", "neutral"),
                             "section": s.get("section"), "visual": s.get("visual", ""), "lang": lang, "check": {"cer": None, "sim": None, "heard": "", "ref": f"mac:{v}"}})
            t += dur
            gap = float(s.get("pause_after", 0.35)) if i < len(segs) - 1 else 0.6
            parts.append(np.zeros(int(gap * SR), np.float32))
            t += gap
        raw = Path(tmp) / "raw.wav"
        write_wav(raw, np.concatenate(parts))
        ac.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw), "-af", "loudnorm=I=-16:TP=-1.5:LRA=9", "-ar", "48000", str(outp)])
    nj = outp.with_suffix(".narration.json")
    nj.write_text(json.dumps({"version": 1, "voice": voice, "engine": "macos", "duration": round(t, 3), "segments": timeline}, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"narration {t:.1f}s with the macOS voice {name or 'Lekha/Rishi'}")
    return {"outputs": [str(outp)], "narration": str(nj), "out_dir": str(outp.parent), "duration": round(t, 2)}


def speak(voice_name: str, script: str, out: str, *, engine: str | None = None, takes: int = 2, language: str | None = None,
          whisper_model: str | None = None) -> dict:
    """Script -> narration in the cloned voice (wav + narration.json). `mac` / `mac:<name>` uses a macOS voice."""
    if voice_name == "mac" or voice_name.startswith("mac:"):
        return speak_mac(voice_name, script, out, language)
    ac.require_tools()
    prof, d = load_profile(voice_name)
    eng = engine or prof.get("engine") or DEFAULT_ENGINE
    if eng not in ENGINES:
        raise AutocutError(f"engine must be one of {', '.join(ENGINES)}")
    try:
        import mlx_audio  # noqa: F401
    except ImportError as e:
        raise AutocutError("the voice model library is not installed (AI ▸ AI Engine Setup ▸ Install)") from e
    segs, default_lang = script_segments(script)
    if not segs:
        raise AutocutError("the script is empty")
    for sg in segs:  # numbers in a Hindi line are said in English ("थर्टी एफ़पीएस", never "तीस")
        if (language or ("hi" if detect_lang(sg["text"]) == "hi" else default_lang)) == "hi" and re.search(r"\d", sg["text"]):
            sg["text"] = english_numbers(sg["text"], devanagari=True)
    takes = int(np.clip(takes, 1, 4))
    outp = Path(out)
    if outp.suffix.lower() != ".wav":
        outp = outp.with_suffix(".wav")
    outp.parent.mkdir(parents=True, exist_ok=True)
    centroid = np.load(d / "centroid.npy") if (d / "centroid.npy").is_file() else None
    t0 = time.time()
    plan = []
    for i, sg in enumerate(segs):
        lang = language or ("hi" if detect_lang(sg["text"]) == "hi" else default_lang)
        emo = sg.get("emotion", "neutral")
        plan.append({"i": i, "lang": lang, "emotion": emo, "energy": line_energy(sg)})
    chunks = group_lines(segs, plan)
    anchors = {lg: anchor_ref(prof, d, lg) for lg in {c["lang"] for c in chunks}}
    for lg, (path, refs) in anchors.items():
        log(f"voice sample ({lg}): {', '.join(refs)} ({len(read_wav(path)) / SR:.1f}s), the same for every line")
    # takes are kept next to the output until the narration is done: a stopped or failed run
    # continues where it was (only the missing takes are spoken again)
    work = outp.parent / ".ei-work" / outp.stem / "takes"
    work.mkdir(parents=True, exist_ok=True)
    anchor_keys = {lg: hashlib.sha1(Path(a[0]).read_bytes()).hexdigest()[:12] for lg, a in anchors.items()}
    ac.progress_plan({"first takes": 0.62, "checking": 0.13, "better takes": 0.2, "assembling": 0.05})
    # the clone model stays loaded across rounds when the checks are light (English: Whisper small);
    # Hindi is checked with the 1.6 GB Whisper, so then the clone model is unloaded for the checks
    langs = {c["lang"] for c in chunks}
    check_models = {lg: (whisper_model or CHECK_WHISPER.get(lg, CHECK_WHISPER["en"])) for lg in langs}
    keep_loaded = all("large" not in m for m in check_models.values())
    iso = ac.Isolated("voice model", stall=300.0)
    with iso, tempfile.TemporaryDirectory(prefix="ei-speak-") as tmp:
        cands: dict[int, list[dict]] = {}
        # 1 take of every passage; more takes only where the first is unclear or sounds less like
        # you than the others (best of two where it matters); a last retry for what is still wrong
        rounds = [("first takes", list(range(len(chunks))), 1)]
        for round_ in range(3):
            if round_ >= len(rounds):
                break
            stage_name, pending, n_takes = rounds[round_]
            per = n_takes if isinstance(n_takes, dict) else {ci: n_takes for ci in pending}
            jobs, cached = [], []
            for ci in pending:
                c = chunks[ci]
                params = engine_params({"energy": c["energy"], "pace": c["pace"], "emotion": c.get("emotion"), "intensity": c.get("intensity", 0.6)}, eng)
                for t in range(per.get(ci, 1)):
                    seed = 1000 * round_ + 17 * ci + t
                    key = hashlib.sha1(json.dumps([eng, c["text"], c["lang"], params, seed, anchor_keys[c["lang"]]], ensure_ascii=False).encode()).hexdigest()[:20]
                    j = {"chunk": ci, "text": c["text"], "lang": c["lang"], "ref": anchors[c["lang"]][0], "params": params, "seed": seed,
                         "out": str(work / f"{key}.wav"), "meta": str(work / f"{key}.json")}
                    (cached if Path(j["out"]).is_file() and Path(j["meta"]).is_file() else jobs).append(j)
            log(f"speaking {len(pending)} passage(s) ({sum(len(chunks[ci]['lines']) for ci in pending)} lines), {sum(per.get(ci, 1) for ci in pending)} take(s) with {ENGINES[eng]['label']}"
                + (f", {len(cached)} take(s) kept from before" if cached else "") + (f" ({stage_name})" if round_ else ""))
            with ac.stage(stage_name if stage_name != "retry" else "better takes"):
                res = []
                for attempt in range(2):  # a frozen worker is restarted once; finished takes are kept
                    todo = [j for j in jobs if not Path(j["out"]).is_file()]
                    if not todo:
                        break
                    try:
                        res += iso.run(_speak_worker, todo, eng, ENGINES[eng]["repo"])
                        break
                    except ac.Stalled:
                        log("the voice model froze: starting it again (finished takes are kept)" if attempt == 0 else "the voice model froze again")
                if not keep_loaded:
                    iso.close()  # (room for the big Whisper)
            errors = {r["out"]: r.get("error") for r in res if not r.get("ok")}
            ok = [j for j in jobs if Path(j["out"]).is_file()]
            for j in jobs:
                if not Path(j["out"]).is_file():
                    log(f"passage {j['chunk'] + 1}: {errors.get(j['out']) or 'not spoken'}")
            with ac.stage("checking") if round_ == 0 else contextlib.nullcontext():
                if ok:
                    heard = asr([(j["out"], j["lang"]) for j in ok], {"en": whisper_model, "hi": whisper_model} if whisper_model else None)
                    embs = embed([j["out"] for j in ok]) if centroid is not None else [None] * len(ok)
                    for j, h, e in zip(ok, heard, embs):
                        y = read_wav(j["out"])
                        cps = len(norm_text(chunks[j["chunk"]]["text"])) / max(0.3, len(y) / SR)
                        meta = {"cer": round(speech_error(h, chunks[j["chunk"]]["text"]), 3),
                                "sim": None if e is None or centroid is None else round(float(e @ centroid), 3),
                                "emb": None if e is None else [float(x) for x in e], "heard": h.strip(), "cps": round(cps, 1)}
                        Path(j["meta"]).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
                        cached.append(j)
                for j in cached:
                    try:
                        meta = json.loads(Path(j["meta"]).read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    emb = np.asarray(meta["emb"], dtype=np.float32) if meta.get("emb") is not None else None
                    cands.setdefault(j["chunk"], []).append({"file": j["out"], "cer": meta["cer"], "sim": meta["sim"], "emb": emb,
                                                             "heard": meta.get("heard", ""), "cps": meta.get("cps", 0.0)})
            best = pick_takes(chunks, cands)
            sims = [b["sim"] for b in best.values() if b.get("sim") is not None]
            med = float(np.median(sims)) if sims else None
            bad = [ci for ci in range(len(chunks)) if ci not in best or best[ci]["cer"] > 0.35 or (best[ci]["sim"] is not None and best[ci]["sim"] < 0.5)]
            weak = [ci for ci in range(len(chunks)) if ci in best and ci not in bad and len(cands.get(ci, [])) < takes
                    and (best[ci]["cer"] > 0.12 or (med is not None and best[ci]["sim"] is not None and best[ci]["sim"] < med - 0.05))]
            if round_ == 0 and (bad or weak) and takes >= 1:
                # a weak passage gets one more take (best of two); a wrong one two fresh takes
                rounds.append(("better takes", sorted(set(bad) | set(weak)), {**{ci: max(1, takes - 1) for ci in weak}, **{ci: max(2, takes - 1) for ci in bad}}))
                log(f"{len(chunks) - len(bad) - len(weak)} of {len(chunks)} passage(s) are good at the first take; "
                    f"{len(weak)} get another take, {len(bad)} are spoken again")
            elif round_ == 1 and bad:
                rounds.append(("retry", bad, 2))
            elif round_ == 0:
                log(f"all {len(chunks)} passage(s) are good at the first take")
        with ac.stage("better takes"):  # (done, or not needed)
            pass
        best = pick_takes(chunks, cands)
        missing = [ci for ci in range(len(chunks)) if ci not in best]
        if missing:
            raise AutocutError(f"{len(missing)} passage(s) could not be spoken")
        # assemble: passages joined by the script's pauses (your room tone), one gentle level for all
        tone = read_wav(d / "roomtone.wav") if (d / "roomtone.wav").is_file() else np.zeros(0, np.float32)
        parts, timeline, t = [], [], 0.0
        fade = int(0.012 * SR)
        for ci, c in enumerate(chunks):
            y = trim_silence(read_wav(best[ci]["file"]), pad=0.05)
            if len(y) > 2 * fade:
                y[:fade] *= np.linspace(0, 1, fade)
                y[-fade:] *= np.linspace(1, 0, fade)
            bounds = split_points(y, [segs[i]["text"] for i in c["lines"]])
            for k, i in enumerate(c["lines"]):
                a_, b_ = bounds[k], bounds[k + 1]
                timeline.append({"start": round(t + a_, 3), "end": round(t + b_, 3), "text": segs[i]["text"], "emotion": segs[i].get("emotion", "neutral"),
                                 "intensity": float(segs[i].get("intensity", 0.6)),
                                 "section": segs[i].get("section"), "visual": segs[i].get("visual", ""), "lang": c["lang"],
                                 "check": {"cer": best[ci]["cer"], "sim": best[ci]["sim"], "heard": best[ci]["heard"], "ref": "anchor", "passage": ci}})
            parts.append(y)
            t += len(y) / SR
            last = c["lines"][-1]
            # the script's pause, or the dramatic pause the next line's emotion asks for (before a reveal, awe, grief)
            gap = max(float(segs[last].get("pause_after", 0.35)), pause_before(segs[chunks[ci + 1]["lines"][0]])) if ci < len(chunks) - 1 else 0.6
            n = int(gap * SR)
            parts.append(pause_fill(tone, n))
            t += gap
        raw = Path(tmp) / "raw.wav"
        write_wav(raw, np.concatenate(parts))
        ac.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", str(outp)])
    worst = sorted(timeline, key=lambda x: -(x["check"]["cer"] or 0))[:1]
    narration = {"version": 2, "voice": voice_name, "engine": eng, "script": str(script) if Path(str(script)).is_file() else None,
                 "duration": round(t, 3), "segments": timeline}
    nj = outp.with_suffix(".narration.json")
    nj.write_text(json.dumps(narration, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.rmtree(work.parent, ignore_errors=True)  # the narration is done: the kept takes are no longer needed
    try:
        work.parent.parent.rmdir()  # (.ei-work, when nothing else is in it)
    except OSError:
        pass
    sims = [best[ci]["sim"] for ci in best if best[ci]["sim"] is not None]
    log(f"narration {t:.1f}s, {len(segs)} lines in {len(chunks)} passages in {time.time() - t0:.0f}s"
        + (f"; voice match {np.mean(sims):.2f} (lowest {min(sims):.2f})" if sims else "")
        + (f"; least clear: “{worst[0]['text'][:50]}”" if worst and (worst[0]['check']['cer'] or 0) > 0.2 else ""))
    return {"outputs": [str(outp)], "narration": str(nj), "out_dir": str(outp.parent), "duration": round(t, 2)}


def pause_before(seg: dict) -> float:
    """Silence wanted before a line: its emotion's pause (Surprise, Awe, Grief …) at its intensity."""
    e = em.info(seg.get("emotion"))
    return float(e["pause_before"]) * (0.5 + 0.5 * float(seg.get("intensity", 0.6)))


def group_lines(segs: list[dict], plan: list[dict], max_chars: int = 240, max_lines: int = 3) -> list[dict]:
    """Consecutive lines spoken together (2–3 sentences), so the voice flows from one sentence into
    the next like a real read; a long pause in the script, a language change or ~240 characters
    starts a new passage."""
    out: list[dict] = []
    cur: list[int] = []

    def flush() -> None:
        if cur:
            lang = plan[cur[0]]["lang"]
            paces = [segs[i].get("pace", "normal") for i in cur]
            top = max(cur, key=lambda i: float(segs[i].get("intensity", 0.6)))
            out.append({"lines": list(cur), "lang": lang, "text": " ".join(segs[i]["text"].strip() for i in cur),
                        "energy": float(np.mean([plan[i]["energy"] for i in cur])), "pace": max(set(paces), key=paces.count),
                        "emotion": segs[top].get("emotion"), "intensity": float(segs[top].get("intensity", 0.6))})
            cur.clear()

    for i, sg in enumerate(segs):
        if cur:
            chars = sum(len(segs[k]["text"]) for k in cur) + len(sg["text"])
            prev = segs[cur[-1]]
            if (plan[i]["lang"] != plan[cur[0]]["lang"] or chars > max_chars or len(cur) >= max_lines
                    or float(prev.get("pause_after", 0.35)) >= 0.8 or abs(plan[i]["energy"] - plan[cur[-1]]["energy"]) > 0.35
                    or em.family(sg.get("emotion")) != em.family(prev.get("emotion")) or pause_before(sg) >= 0.2):
                flush()
        cur.append(i)
    flush()
    return out


def anchor_ref(prof: dict, d: Path, lang: str, target: float = 9.0) -> tuple[str, list[str]]:
    """One voice sample for the whole narration: the most typical clean take(s) of this language
    (closest to your voice fingerprint, a natural delivery), about 9 s (what the model listens to)."""
    refs = [r for r in prof["refs"] if r.get("quality", 0) >= 0.6]
    pool = [r for r in refs if r["lang"] == lang] or refs or prof["refs"]
    if not pool:
        raise AutocutError("the voice has no usable references")

    def score(r: dict) -> float:
        s = (r.get("similarity") or 0.8) * 2 + r.get("quality", 0.5)
        s += 0.25 if r.get("emotion") in ("neutral", "warm", "story", "calm", "natural") else 0.0
        s -= 0.3 if r["features"].get("dur", 0) < 5 else 0.0
        return s

    ranked = sorted(pool, key=score, reverse=True)
    chosen = [ranked[0]]
    y = read_wav(d / "refs" / ranked[0]["file"])
    for r in ranked[1:]:
        if len(y) / SR >= target:
            break
        y = np.concatenate([y, np.zeros(int(0.25 * SR), np.float32), read_wav(d / "refs" / r["file"])])
        chosen.append(r)
    y = y[: int(10 * SR)]
    path = d / f"anchor_{lang}.wav"
    write_wav(path, y)
    return str(path), [r["file"] for r in chosen]


def pick_takes(chunks: list[dict], cands: dict[int, list[dict]]) -> dict[int, dict]:
    """Best take per passage, in order: says the words, sounds like you, and like the passage
    before it (so the narration stays one person)."""
    best: dict[int, dict] = {}
    prev = None
    for ci in range(len(chunks)):
        cs = cands.get(ci)
        if not cs:
            prev = None
            continue

        def score(c: dict) -> float:
            s = (1 - min(1.0, c["cer"])) + 1.5 * (c["sim"] if c["sim"] is not None else 0.6)
            if prev is not None and c["emb"] is not None and prev.get("emb") is not None:
                s += 0.6 * float(c["emb"] @ prev["emb"])
            s -= 0.4 if not 6 <= c["cps"] <= 22 else 0.0
            return s

        best[ci] = max(cs, key=score)
        prev = best[ci]
    return best


def split_points(y: np.ndarray, texts: list[str], search: float = 0.6) -> list[float]:
    """Where each line starts and ends inside a passage: proportional to the text, moved to the
    quietest moment nearby (the breath between sentences)."""
    dur = len(y) / SR
    if len(texts) <= 1:
        return [0.0, dur]
    w = np.asarray([max(1, len(norm_text(t))) for t in texts], dtype=float)
    cum = np.cumsum(w) / w.sum() * dur
    db = frame_db(y)  # 20 ms frames
    pts = [0.0]
    for c in cum[:-1]:
        lo, hi = int(max(pts[-1] + 0.3, c - search) * 50), int(min(dur - 0.3, c + search) * 50)
        if hi > lo + 2 and hi <= len(db):
            sm = np.convolve(db, np.ones(3) / 3, mode="same")
            c = (lo + int(np.argmin(sm[lo:hi]))) / 50
        pts.append(float(c))
    pts.append(dur)
    return pts


def tempo(y: np.ndarray, factor: float) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="ei-tempo-") as tmp:
        a, b = Path(tmp) / "a.wav", Path(tmp) / "b.wav"
        write_wav(a, y)
        ac.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(a), "-af", f"atempo={factor:.3f}", str(b)])
        return read_wav(b)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def add_args(sub) -> None:
    p = sub.add_parser("voice-prompts", help="the sentences to read for a voice")
    p.add_argument("--languages", nargs="+", default=["hi", "mix", "en"], choices=["hi", "mix", "en"], help="sets in reading order (mix = Hinglish)")
    p = sub.add_parser("voice-record", help="record one take from the microphone (stops when you stop talking)")
    p.add_argument("--out", required=True)
    p.add_argument("--max", type=float, default=30.0)
    p.add_argument("--mic", default="0")
    p.add_argument("--stop-file", help="stop (and keep the take) when this file appears")
    sub.add_parser("voice-mics", help="list microphones")
    p = sub.add_parser("voice-enroll", help="make or extend a voice from recordings")
    p.add_argument("--name", required=True)
    p.add_argument("--files", nargs="*", default=[], help="audio/video files of your voice (free talking)")
    p.add_argument("--takes", help="JSON file: [{file, text, lang, emotion}] of read prompts")
    p.add_argument("--languages", nargs="*")
    p.add_argument("--engine", default=DEFAULT_ENGINE, choices=list(ENGINES))
    p.add_argument("--add", action="store_true", help="add to an existing voice instead of replacing its references")
    p.add_argument("--make", default="voice", choices=["voice", "style", "both"],
                   help="clone the voice, learn the talking style (saved as a style of the same name), or both")
    p.add_argument("--separate", default="always", choices=["auto", "always", "never"],
                   help="take the voice out of music, game sound and noise first (always: every recording; auto: only when there is music under the voice)")
    sub.add_parser("voice-list", help="saved voices")
    p = sub.add_parser("speak", help="speak a script (or text) with a cloned voice")
    p.add_argument("--voice", required=True)
    p.add_argument("--script", required=True, help=".script.json from script_writer, a .txt, or the text itself")
    p.add_argument("--out", required=True)
    p.add_argument("--engine", choices=list(ENGINES))
    p.add_argument("--takes", type=int, default=2, help="takes per line (best one kept): 1 fast … 4 best")
    p.add_argument("--language", choices=["en", "hi"])
    import score

    score.add_sound_args(p)


def run_args(a) -> dict:
    if a.cmd == "voice-prompts":
        import voice_prompts

        return {"prompts": voice_prompts.prompts(a.languages)}
    if a.cmd == "voice-mics":
        return {"mics": list_mics()}
    if a.cmd == "voice-record":
        return record(a.out, a.max, mic=a.mic, stop_file=a.stop_file)
    if a.cmd == "voice-list":
        return {"voices": list_voices(), "mac": list_mac_voices()}
    if a.cmd == "voice-enroll":
        takes = json.loads(Path(a.takes).read_text(encoding="utf-8")) if a.takes else None
        res: dict = {"outputs": []}
        if a.make in ("voice", "both"):
            res = enroll(a.name, a.files, takes, a.languages, a.engine, a.add, a.separate)
        if a.make in ("style", "both"):
            res = merge_style(res, learn_style(a.name, a.files))
        return res
    if a.cmd == "speak":
        import score

        opts = score.options(a)
        res = speak(a.voice, a.script, a.out, engine=a.engine, takes=a.takes, language=a.language)
        if opts:  # the voice alone stays; the scored version is written next to it
            wav = res["outputs"][0]
            scored = str(Path(wav).with_name(Path(wav).stem + " - scored.wav"))
            nar = json.loads(Path(res["narration"]).read_text(encoding="utf-8"))
            sc = score.score_narration(wav, nar, scored, opts)
            res["outputs"] = [scored, wav]
            res["credits"] = sc["credits"]
            log("credits: " + ("; ".join(sc["credits"]) or "none needed"))
        return res
    raise AutocutError(f"unknown command {a.cmd}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_args(sub)
    try:
        print(json.dumps(run_args(ap.parse_args(argv)), ensure_ascii=False))
    except AutocutError as e:
        log(f"error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
