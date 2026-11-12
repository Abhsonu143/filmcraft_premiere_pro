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


def clean_file(src: str, dst: Path) -> None:
    """Mono 24 kHz, rumble removed, gentle noise reduction, consistent level (originals untouched)."""
    ac.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", str(SR),
            "-af", "highpass=f=70,afftdn=nr=8:nf=-45,loudnorm=I=-20:TP=-2:LRA=11", "-c:a", "pcm_s16le", str(dst)])


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


def record(out: str, max_s: float = 30.0, stop_after: float = 1.4, mic: str = "0", wait_s: float = 8.0) -> dict:
    """Record from the microphone until you stop talking (stop_after seconds of silence after
    speech), at most max_s. Returns level/noise/clipping checks."""
    rate = 48000
    proc = subprocess.Popen(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "avfoundation", "-i", f":{mic}",
                             "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    chunk = rate // 20  # 50 ms
    buf: list[np.ndarray] = []
    noise: list[float] = []
    started, quiet, t = False, 0.0, 0.0
    log("recording: speak now")
    try:
        while t < max_s:
            raw = proc.stdout.read(chunk * 2) if proc.stdout else b""
            if not raw:
                break
            x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
            buf.append(x)
            t += len(x) / rate
            db = 20 * math.log10(max(1e-6, float(np.sqrt((x * x).mean()))))
            if not started:
                noise.append(db)
                floor = float(np.median(noise[-40:])) if noise else -60.0
                if t > 0.3 and db > max(-45.0, floor + 15):
                    started = True
                    log("hearing you")
                elif t > wait_s:
                    break
            else:
                floor = float(np.median(noise)) if noise else -60.0
                quiet = quiet + len(x) / rate if db < max(-50.0, floor + 8) else 0.0
                if quiet >= stop_after:
                    break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    err = (proc.stderr.read().decode(errors="replace") if proc.stderr else "").strip()
    if not buf:
        raise AutocutError(f"no sound from the microphone{(': ' + err[:160]) if err else ''} (System Settings ▸ Privacy ▸ Microphone: allow EI)")
    y = np.concatenate(buf)
    if not started:
        raise AutocutError("nothing was heard: check the microphone and speak a little louder")
    y = trim_silence(y, pad=0.15)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_wav(out, y, rate)
    f = features(np.interp(np.arange(0, len(y), rate / SR), np.arange(len(y)), y).astype(np.float32))
    issues = []
    if f.get("clipped", 0) > 0.001:
        issues.append("too loud (clipping): move back from the microphone a little")
    if f.get("level_db", -60) < -38:
        issues.append("quiet: move closer to the microphone")
    if f.get("snr_db", 99) < 25:
        issues.append("background noise: a quieter room helps the clone a lot")
    log(f"recorded {len(y) / rate:.1f}s" + (f" ({'; '.join(issues)})" if issues else ""))
    return {"outputs": [out], "file": out, "features": f, "issues": issues}


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


def enroll(name: str, files: list[str] | None = None, takes: list[dict] | None = None, languages: list[str] | None = None,
           engine: str = DEFAULT_ENGINE, add: bool = False) -> dict:
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
        for src in files or []:  # free recordings: cut into phrases
            if not Path(src).is_file():
                raise AutocutError(f"not found: {src}")
            clean = Path(tmp) / f"c{len(new)}.wav"
            clean_file(src, clean)
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
    tone = []
    for r in sorted(allrefs, key=lambda r: -r["quality"])[:6]:
        y = read_wav(refs_dir / r["file"])
        db = frame_db(y)
        if len(db) > 10:
            i = int(np.argmin(np.convolve(db, np.ones(10) / 10, mode="valid")))
            tone.append(y[i * (SR // 50) : (i + 10) * (SR // 50)])
    if tone:
        write_wav(d / "roomtone.wav", np.concatenate(tone))
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


def engine_params(seg: dict, engine: str) -> dict:
    energy = float(seg.get("energy", EMOTION_ENERGY.get(seg.get("emotion", "neutral"), 0.45)))
    if engine == "chatterbox":
        exag = 0.25 + 0.6 * energy  # calm 0.4 … excited 0.75
        cfg = 0.5 if seg.get("pace") != "fast" else 0.35
        if seg.get("pace") == "slow":
            cfg = 0.6
        return {"exaggeration": round(exag, 2), "cfg_weight": cfg, "temperature": 0.75}
    return {}


def _speak_worker(jobs: list[dict], engine: str, repo: str) -> list[dict]:
    """In its own process: load the clone model once, generate every take."""
    import numpy as np
    from mlx_audio.tts.utils import load_model

    model = load_model(repo)
    out = []
    for j in jobs:
        kw = {"text": j["text"], "ref_audio": j["ref"]}
        kw.update(lang_code=j["lang"], **j.get("params", {}))
        try:
            import mlx.core as mx

            mx.random.seed(int(j.get("seed", 0)))
            chunks = [np.array(r.audio, dtype=np.float32) for r in model.generate(**kw)]
            y = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
            sr = int(getattr(model, "sample_rate", 24000))
            write_wav(j["out"], y, sr)
            out.append({"out": j["out"], "ok": True})
        except Exception as e:  # one bad line must not lose the rest
            out.append({"out": j["out"], "ok": False, "error": f"{type(e).__name__}: {e}"[:200]})
    return out


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
        m = re.match(r"\[(\w+)\]\s*(.+)", x)
        emo, t = (m.group(1).lower(), m.group(2)) if m else ("neutral", x)
        segs.append({"text": t, "emotion": emo, "pause_after": 0.35})
    return segs, ("hi" if detect_lang(text) == "hi" else "en")


def speak(voice_name: str, script: str, out: str, *, engine: str | None = None, takes: int = 2, language: str | None = None,
          whisper_model: str | None = None) -> dict:
    """Script -> narration in the cloned voice (wav + narration.json)."""
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
    takes = int(np.clip(takes, 1, 4))
    outp = Path(out)
    if outp.suffix.lower() != ".wav":
        outp = outp.with_suffix(".wav")
    outp.parent.mkdir(parents=True, exist_ok=True)
    centroid = np.load(d / "centroid.npy") if (d / "centroid.npy").is_file() else None
    used: dict[str, int] = {}
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="ei-speak-") as tmp:
        plan = []
        for i, s in enumerate(segs):
            lang = language or ("hi" if detect_lang(s["text"]) == "hi" else default_lang)
            emo = s.get("emotion", "neutral")
            energy = float(s.get("energy", EMOTION_ENERGY.get(emo, 0.45)))
            plan.append({"i": i, "lang": lang, "emotion": emo, "energy": energy})
        best: dict[int, dict] = {}
        pending = list(range(len(segs)))
        for round_ in range(2):  # a second round regenerates lines that came out wrong
            jobs = []
            for i in pending:
                pl = plan[i]
                ref = choose_ref(prof, pl["lang"], pl["emotion"], pl["energy"], used)
                used[ref["file"]] = used.get(ref["file"], 0) + 1
                for t in range(takes):
                    jobs.append({"seg": i, "text": segs[i]["text"], "lang": pl["lang"], "ref": str(d / "refs" / ref["file"]), "ref_text": ref["text"],
                                 "params": engine_params({**segs[i], "energy": pl["energy"]}, eng), "seed": 1000 * round_ + 17 * i + t,
                                 "out": str(Path(tmp) / f"s{i:03d}_r{round_}_t{t}.wav"), "ref_file": ref["file"]})
            log(f"speaking {len(pending)} line(s) × {takes} take(s) with {ENGINES[eng]['label']}" + (" (retry)" if round_ else ""))
            res = ac.run_isolated(_speak_worker, [{k: v for k, v in j.items() if k != "ref_file"} for j in jobs], eng, ENGINES[eng]["repo"])
            ok = [j for j, r in zip(jobs, res) if r.get("ok")]
            for j, r in zip(jobs, res):
                if not r.get("ok"):
                    log(f"line {j['seg'] + 1}: {r.get('error')}")
            if not ok:
                break
            heard = asr([(j["out"], j["lang"]) for j in ok], {"en": whisper_model, "hi": whisper_model} if whisper_model else None)
            embs = embed([j["out"] for j in ok]) if centroid is not None else [None] * len(ok)
            for j, h, e in zip(ok, heard, embs):
                y = read_wav(j["out"])
                dur = len(y) / SR
                chars = len(norm_text(segs[j["seg"]]["text"]))
                cps = chars / max(0.3, dur)
                c = speech_error(h, segs[j["seg"]]["text"])
                sim = float(e @ centroid) if (e is not None and centroid is not None) else None
                score = (1 - min(1.0, c)) * 1.0 + (sim if sim is not None else 0.6) * 0.8 - (0.4 if not 6 <= cps <= 22 else 0.0)
                cand = {"file": j["out"], "cer": round(c, 3), "sim": None if sim is None else round(sim, 3), "score": score, "heard": h.strip(), "ref": j["ref_file"], "cps": round(cps, 1)}
                if j["seg"] not in best or score > best[j["seg"]]["score"]:
                    best[j["seg"]] = cand
            pending = [i for i in range(len(segs)) if i not in best or best[i]["cer"] > 0.35 or (best[i]["sim"] is not None and best[i]["sim"] < 0.5)]
            if not pending:
                break
        missing = [i for i in range(len(segs)) if i not in best]
        if missing:
            raise AutocutError(f"{len(missing)} line(s) could not be spoken")
        # assemble: lines + pauses filled with room tone, levelled to one recording
        tone = read_wav(d / "roomtone.wav") if (d / "roomtone.wav").is_file() else np.zeros(SR, np.float32)
        if len(tone) == 0:
            tone = np.zeros(SR, np.float32)
        parts, timeline, t = [], [], 0.0
        fade = int(0.012 * SR)
        for i, s in enumerate(segs):
            y = trim_silence(read_wav(best[i]["file"]), pad=0.05)
            if len(y) > 2 * fade:
                y[:fade] *= np.linspace(0, 1, fade)
                y[-fade:] *= np.linspace(1, 0, fade)
            pace = s.get("pace", "normal")
            if pace in ("slow", "fast"):  # small tempo touch (pitch kept)
                y = tempo(y, 0.94 if pace == "slow" else 1.05)
            parts.append(y)
            dur = len(y) / SR
            timeline.append({"start": round(t, 3), "end": round(t + dur, 3), "text": s["text"], "emotion": s.get("emotion", "neutral"),
                             "section": s.get("section"), "visual": s.get("visual", ""), "lang": plan[i]["lang"],
                             "check": {k: best[i][k] for k in ("cer", "sim", "heard", "ref")}})
            t += dur
            gap = float(s.get("pause_after", 0.35)) if i < len(segs) - 1 else 0.6
            n = int(gap * SR)
            reps = int(np.ceil(n / max(1, len(tone))))
            parts.append(np.tile(tone, reps)[:n] * 0.9)
            t += gap
        y = np.concatenate(parts)
        raw = Path(tmp) / "raw.wav"
        write_wav(raw, y)
        ac.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
                "-af", "acompressor=threshold=-20dB:ratio=2:attack=10:release=150,loudnorm=I=-16:TP=-1.5:LRA=9", "-ar", "48000", str(outp)])
    worst = sorted(timeline, key=lambda x: -(x["check"]["cer"] or 0))[:3]
    narration = {"version": 1, "voice": voice_name, "engine": eng, "script": str(script) if Path(str(script)).is_file() else None,
                 "duration": round(t, 3), "segments": timeline}
    nj = outp.with_suffix(".narration.json")
    nj.write_text(json.dumps(narration, ensure_ascii=False, indent=1), encoding="utf-8")
    sims = [x["check"]["sim"] for x in timeline if x["check"]["sim"] is not None]
    log(f"narration {t:.1f}s, {len(segs)} lines in {time.time() - t0:.0f}s" + (f"; voice match {np.mean(sims):.2f}" if sims else "")
        + (f"; least clear line: “{worst[0]['text'][:50]}” (heard “{worst[0]['check']['heard'][:50]}”)" if worst and (worst[0]['check']['cer'] or 0) > 0.2 else ""))
    return {"outputs": [str(outp)], "narration": str(nj), "out_dir": str(outp.parent), "duration": round(t, 2)}


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
    p.add_argument("--languages", nargs="+", default=["en", "hi"])
    p = sub.add_parser("voice-record", help="record one take from the microphone (stops when you stop talking)")
    p.add_argument("--out", required=True)
    p.add_argument("--max", type=float, default=30.0)
    p.add_argument("--mic", default="0")
    sub.add_parser("voice-mics", help="list microphones")
    p = sub.add_parser("voice-enroll", help="make or extend a voice from recordings")
    p.add_argument("--name", required=True)
    p.add_argument("--files", nargs="*", default=[], help="audio/video files of your voice (free talking)")
    p.add_argument("--takes", help="JSON file: [{file, text, lang, emotion}] of read prompts")
    p.add_argument("--languages", nargs="*")
    p.add_argument("--engine", default=DEFAULT_ENGINE, choices=list(ENGINES))
    p.add_argument("--add", action="store_true", help="add to an existing voice instead of replacing its references")
    sub.add_parser("voice-list", help="saved voices")
    p = sub.add_parser("speak", help="speak a script (or text) with a cloned voice")
    p.add_argument("--voice", required=True)
    p.add_argument("--script", required=True, help=".script.json from script_writer, a .txt, or the text itself")
    p.add_argument("--out", required=True)
    p.add_argument("--engine", choices=list(ENGINES))
    p.add_argument("--takes", type=int, default=2, help="takes per line (best one kept): 1 fast … 4 best")
    p.add_argument("--language", choices=["en", "hi"])


def run_args(a) -> dict:
    if a.cmd == "voice-prompts":
        import voice_prompts

        return {"prompts": voice_prompts.prompts(a.languages)}
    if a.cmd == "voice-mics":
        return {"mics": list_mics()}
    if a.cmd == "voice-record":
        return record(a.out, a.max, mic=a.mic)
    if a.cmd == "voice-list":
        return {"voices": list_voices()}
    if a.cmd == "voice-enroll":
        takes = json.loads(Path(a.takes).read_text(encoding="utf-8")) if a.takes else None
        return enroll(a.name, a.files, takes, a.languages, a.engine, a.add)
    if a.cmd == "speak":
        return speak(a.voice, a.script, a.out, engine=a.engine, takes=a.takes, language=a.language)
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
