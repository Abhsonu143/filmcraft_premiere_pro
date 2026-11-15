"""gameplay_plan: plan what to publish from a list of gameplay recordings.

    python gameplay_plan.py REC1.mov REC2.mov ... --game "Ghost of Yotei" [--platforms youtube shorts reels]
                            [--count 12] [--focus "story recap, boss fights"] [--out-dir D] [--analyse-only]

Recordings are given in playing order. For each one, ONCE (the results are kept next to it):
  - `<rec>.ei.srt`: what is said (the game's dialogue), Whisper small, 10-minute pieces;
  - `<rec>.ei-analysis.json` (+ `.ei-clip.npy`): every second: how loud and how much is moving;
    every 4 s: what is on screen (CLIP: combat, cutscene, exploring, menu, scenery… plus things this
    game has, from the research); scene changes; the strongest action peaks, story stretches and
    quiet scenic stretches.
  The analysis goes one recording at a time and, inside a recording, 10 minutes at a time (only the
  keyframes are decoded: PS5 recordings have one every ~2 s), so memory stays low however long the
  file is. A stopped analysis continues from the last finished piece.

Then (cached per game for a week): Claude researches the game on the web: story, characters, bosses,
places, and which videos about it do well (formats, topics, titles).

Then Claude plans videos from all of it: long YouTube videos (episodes, story recaps, boss fights,
lore), Shorts and Reels (single moments), each with a title, hook, why it works, and the exact parts
of which recordings to use, and the EI tool that makes it. `plans.json` + `plans.md` in the output
folder; running again adds new plans and never repeats earlier ones. You can also give the
`.ei-analysis.json` / `.srt` files instead of the recordings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import autocut as ac
from autocut import AutocutError, log

VERSION = "gp1"
CHUNK = 600.0  # seconds of recording analysed at a time
TAG_EVERY = 4.0  # seconds between "what is on screen" samples
WHISPER = ac.DEFAULT_WHISPER
VIDEO_EXT = {".mov", ".mp4", ".m4v", ".mkv", ".webm", ".avi", ".mts"}
BASE_PROMPTS = {
    "combat": "a video game sword fight or combat with enemies",
    "boss": "a video game boss fight against a large powerful enemy",
    "cutscene": "a cinematic video game cutscene with characters talking, close-up faces",
    "exploring": "exploring an open world landscape in a video game, riding or walking",
    "scenic": "a beautiful scenic landscape view in a video game, sunset, snow or mountains",
    "stealth": "sneaking or hiding in tall grass in a video game, stealth",
    "menu": "a video game menu, inventory, map or skill screen with text",
    "death": "a video game death or game over screen",
    "town": "a village or town with people in a video game",
    "night": "a dark night scene in a video game",
}


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


def sidecars(rec: Path) -> tuple[Path, Path, Path]:
    return rec.with_name(rec.stem + ".ei-analysis.json"), rec.with_name(rec.stem + ".ei.srt"), rec.with_name(rec.stem + ".ei-clip.npy")


def resolve_inputs(items: list[str]) -> list[dict]:
    """Recordings in the order given: {video (or None), analysis path, srt path}. Folders give their
    recordings sorted by name; analysis or .srt files stand for their recording."""
    out: list[dict] = []
    seen: set[str] = set()
    for it in items:
        p = Path(it).expanduser()
        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT and not f.name.startswith("."))
        elif p.is_file():
            files = [p]
        else:
            raise AutocutError(f"not found: {it}")
        for f in files:
            name = f.name
            if name.endswith(".ei-analysis.json"):
                a = json.loads(f.read_text(encoding="utf-8"))
                video = Path(a.get("source", "")) if a.get("source") else None
                key = str(f.resolve())[: -len(".ei-analysis.json")]
                entry = {"video": video if video and video.is_file() else None, "analysis": f, "srt": f.with_name(name.replace(".ei-analysis.json", ".ei.srt"))}
            elif f.suffix.lower() == ".srt":
                stem = name[:-len(".ei.srt")] if name.endswith(".ei.srt") else f.stem
                ana = f.with_name(stem + ".ei-analysis.json")
                vid = next((f.with_name(stem + e) for e in (".mov", ".mp4", ".MOV", ".MP4", ".mkv") if f.with_name(stem + e).is_file()), None)
                key = str(f.resolve().with_name(stem))
                entry = {"video": vid, "analysis": ana if ana.is_file() else None, "srt": f}
            elif f.suffix.lower() in VIDEO_EXT:
                ana, srt, _ = sidecars(f)
                key = str(f.resolve().with_suffix(""))
                entry = {"video": f.resolve(), "analysis": ana, "srt": srt}
            else:
                continue
            # (one recording given several ways, as its video, analysis and .srt, counts once)
            if key not in seen:
                seen.add(key)
                out.append(entry)
    if not out:
        raise AutocutError("no recordings, analyses or .srt files in what was given")
    return out


# ---------------------------------------------------------------------------------------------
# Analysing one recording (in pieces)
# ---------------------------------------------------------------------------------------------


def _whisper_worker(wavs: list[str], model: str, hint: str = "") -> list[list[dict]]:
    """Runs in its own process: Whisper once, the pieces one after another. `hint`: the game's
    names (characters, places), so they are spelled right."""
    import mlx_whisper

    out = []
    for w in wavs:
        try:
            kw = {"initial_prompt": hint} if hint else {}
            r = mlx_whisper.transcribe(w, path_or_hf_repo=model, word_timestamps=False, verbose=None, condition_on_previous_text=False, **kw)
        except Exception:
            out.append([])
            continue
        segs = []
        for s in r.get("segments", []):
            text = str(s.get("text", "")).strip()
            if float(s.get("no_speech_prob", 0)) > 0.6 or float(s.get("compression_ratio", 1)) > 2.6 or len(text.split()) < 2:
                continue  # (music and game noise make Whisper imagine words)
            segs.append({"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2), "text": text})
        out.append(segs)
    return out


def _clip_worker(images: list[str], prompts: list[str]) -> dict:
    """Runs in its own process: CLIP once; image embeddings (float16) and prompt similarities."""
    import open_clip
    import torch
    from PIL import Image

    import voiceover_cut as vc

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, pre = open_clip.create_model_and_transforms(vc.CLIP_MODEL[0], pretrained=vc.CLIP_MODEL[1], device=dev)
    model.eval()
    tok = open_clip.get_tokenizer(vc.CLIP_MODEL[0])
    embs = []
    with torch.no_grad():
        for i in range(0, len(images), 32):
            batch = torch.stack([pre(Image.open(p).convert("RGB")) for p in images[i:i + 32]]).to(dev)
            e = model.encode_image(batch)
            embs.append((e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy())
        t = model.encode_text(tok(prompts).to(dev))
        t = (t / t.norm(dim=-1, keepdim=True)).float().cpu().numpy()
    E = np.concatenate(embs) if embs else np.zeros((0, 512), np.float32)
    return {"emb": E.astype(np.float16).tolist(), "sims": (E @ t.T).round(4).tolist()}


def audio_levels(wav: Path, sr: int = 16000) -> list[float]:
    """Loudness (RMS, 0–1) per second of a mono 16-bit wav."""
    import wave

    with wave.open(str(wav), "rb") as w:
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    n = len(x) // sr
    if n == 0:
        return []
    r = np.sqrt((x[: n * sr].reshape(n, sr) ** 2).mean(axis=1))
    return [round(float(v), 4) for v in r]


def video_signals(video: Path, start: float, dur: float, work: Path, tag: str) -> tuple[list[dict], list[str]]:
    """Keyframes of one piece: per keyframe {t, motion, bright, cut}; small JPEGs every TAG_EVERY s for CLIP."""
    import cv2

    w, h = 320, 180
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-skip_frame", "nokey", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-i", str(video), "-an", "-vf", f"scale={w}:{h}", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    # keyframe times come from ffprobe (the decoded stream above has no clock of its own)
    pts = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey", "-show_entries", "frame=pts_time",
                          "-of", "csv=p=0", "-read_intervals", f"{start:.3f}%+{dur:.3f}", str(video)], capture_output=True, text=True).stdout
    times = [float(x.strip(",")) for x in pts.split() if x.strip(",").replace(".", "", 1).isdigit()]
    times = [t for t in times if start - 0.5 <= t < start + dur]
    rows, imgs = [], []
    prev, last_tag = None, -1e9
    k = 0
    try:
        while True:
            buf = proc.stdout.read(w * h * 3) if proc.stdout else b""
            if len(buf) < w * h * 3:
                break
            f = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            t = times[k] if k < len(times) else start + k * 2.0
            k += 1
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            motion = float(np.abs(g - prev).mean()) if prev is not None else 0.0
            hist = cv2.calcHist([f], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3).ravel()
            hist /= max(1.0, hist.sum())
            rows.append({"t": round(t, 2), "motion": round(motion, 2), "bright": round(float(g.mean()), 1), "_h": hist})
            prev = g
            if t - last_tag >= TAG_EVERY - 0.01:
                p = work / f"{tag}-{len(imgs):05d}.jpg"
                cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 85])
                imgs.append(str(p))
                last_tag = t
    finally:
        proc.kill()
        proc.wait()
    # scene changes: the colour make-up changes a lot from one keyframe to the next
    for a, b in zip(rows, rows[1:]):
        b["cut"] = bool(0.5 * np.abs(a["_h"] - b["_h"]).sum() > 0.45)
    for r in rows:
        r.pop("_h", None)
        r.setdefault("cut", False)
    return rows, imgs


def name_hint(research: dict) -> str:
    """The game's names for Whisper (characters, places, bosses), at most ~200 characters."""
    names = [c.get("name", "") for c in research.get("characters") or [] if isinstance(c, dict)]
    names += [str(x) for x in (research.get("places") or [])[:8]] + [str(x) for x in (research.get("bosses") or [])[:6]]
    names = [re.sub(r"\s*\(.*?\)", "", n).strip() for n in names if n]
    hint = ", ".join(dict.fromkeys(n for n in names if 1 < len(n) < 30))
    return (f"{research.get('game', '')}: " + hint)[:220] if hint else ""


def analyse(entry: dict, prompts: dict[str, str], hint: str = "") -> dict:
    """The analysis of one recording (made once; continued if it was stopped)."""
    video = entry["video"]
    apath = entry["analysis"] or (sidecars(video)[0] if video else None)
    if apath and Path(apath).is_file():
        a = json.loads(Path(apath).read_text(encoding="utf-8"))
        if a.get("complete") and (video is None or a.get("key") == ac.file_key(str(video), VERSION)):
            log(f"{Path(apath).name}: analysed before, reused")
            return a
    if video is None:  # only an .srt: dialogue without picture signals
        segs = srt_segments(entry["srt"]) if entry.get("srt") and Path(entry["srt"]).is_file() else []
        return {"source": "", "name": Path(entry["srt"]).stem, "duration": segs[-1]["end"] if segs else 0, "dialogue": segs, "seconds": [],
                "keyframes": [], "tags": [], "prompts": list(prompts), "complete": True}
    info = ac.probe(str(video))
    dur = float(info.duration)
    key = ac.file_key(str(video), VERSION)
    a = None
    if apath.is_file():
        try:
            old = json.loads(apath.read_text(encoding="utf-8"))
            if old.get("key") == key and list(old.get("prompts", [])) == list(prompts):
                a = old  # (a stopped analysis: continue)
        except ValueError:
            pass
    if a is None:
        a = {"version": VERSION, "key": key, "source": str(video), "name": video.stem, "duration": round(dur, 2), "prompts": list(prompts),
             "audio_until": 0.0, "video_until": 0.0, "dialogue": [], "loud": [], "keyframes": [], "tags": [], "complete": False}
    _, srt_path, clip_path = sidecars(video)
    names = list(prompts)
    texts = [prompts[n] for n in names]
    embs = list(np.load(clip_path).astype(np.float16)) if clip_path.is_file() and a.get("video_until", 0) > 0 else []
    n_chunks = max(1, math.ceil(dur / CHUNK))
    log(f"{video.name}: {dur / 60:.0f} min, in {n_chunks} piece(s) of {CHUNK / 60:.0f} min: first what is said, then what is on screen")

    def save():
        Path(apath).write_text(json.dumps(a, ensure_ascii=False), encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="ei-gp-") as tmp:
        work = Path(tmp)
        # 1. the sound, piece by piece (Whisper loaded once for the whole recording)
        if a["audio_until"] < dur - 0.5:
            with ac.Isolated("whisper") as iso:
                while a["audio_until"] < dur - 0.5:
                    s0 = a["audio_until"]
                    d = min(CHUNK, dur - s0)
                    t0 = time.time()
                    wav = work / "a.wav"
                    ac.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{s0:.3f}", "-t", f"{d:.3f}", "-i", str(video),
                            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])
                    a["loud"] += audio_levels(wav)
                    segs = iso.run(_whisper_worker, [str(wav)], WHISPER, hint)[0]
                    a["dialogue"] += [{"start": round(x["start"] + s0, 2), "end": round(x["end"] + s0, 2), "text": x["text"]} for x in segs]
                    a["audio_until"] = round(s0 + d, 2)
                    save()
                    ac.stage_progress(0.5 * a["audio_until"] / dur, f"{video.name}: listening, {a['audio_until'] / 60:.0f} of {dur / 60:.0f} min")
                    log(f"{video.name}: heard {a['audio_until'] / 60:.0f}/{dur / 60:.0f} min ({len(segs)} lines) in {time.time() - t0:.0f}s")
        # 2. the picture, piece by piece (only keyframes; CLIP loaded once)
        if a["video_until"] < dur - 0.5:
            with ac.Isolated("clip") as iso:
                while a["video_until"] < dur - 0.5:
                    s0 = a["video_until"]
                    d = min(CHUNK, dur - s0)
                    t0 = time.time()
                    rows, imgs = video_signals(video, s0, d, work, f"c{int(s0)}")
                    clip = iso.run(_clip_worker, imgs, texts) if imgs else {"emb": [], "sims": []}
                    a["keyframes"] += rows
                    for k, sims in enumerate(clip["sims"]):
                        order = np.argsort(sims)[::-1][:3]
                        a["tags"].append({"t": round(s0 + k * TAG_EVERY, 1), "top": [names[int(i)] for i in order],
                                          "sim": [round(float(sims[int(i)]), 3) for i in order]})
                    embs += [np.float16(e) for e in np.asarray(clip["emb"], dtype=np.float16)]
                    for f in imgs:
                        Path(f).unlink(missing_ok=True)
                    a["video_until"] = round(s0 + d, 2)
                    np.save(clip_path, np.asarray(embs, dtype=np.float16).reshape(-1, 512))
                    save()
                    ac.stage_progress(0.5 + 0.5 * a["video_until"] / dur, f"{video.name}: watching, {a['video_until'] / 60:.0f} of {dur / 60:.0f} min")
                    log(f"{video.name}: watched {a['video_until'] / 60:.0f}/{dur / 60:.0f} min ({len(rows)} keyframes) in {time.time() - t0:.0f}s")
    a["moments"] = moments(a)
    a["complete"] = True
    Path(apath).write_text(json.dumps(a, ensure_ascii=False), encoding="utf-8")
    srt_path.write_text(ac.to_srt([{"start": x["start"], "end": x["end"], "text": x["text"]} for x in a["dialogue"]]), encoding="utf-8")
    log(f"{video.name}: analysis and {srt_path.name} saved next to the recording")
    return a


def srt_segments(path: Path) -> list[dict]:
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    out = []

    def secs(x: str) -> float:
        h, m, rest = x.strip().replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest)

    for block in re.split(r"\n\s*\n", txt.strip()):
        lines = block.strip().splitlines()
        m = next((re.match(r"(\S+)\s*-->\s*(\S+)", l) for l in lines if "-->" in l), None)
        if not m:
            continue
        text = " ".join(l for l in lines[lines.index(next(l for l in lines if "-->" in l)) + 1:]).strip()
        try:
            out.append({"start": secs(m.group(1)), "end": secs(m.group(2)), "text": text})
        except ValueError:
            continue
    return out


def per_second(a: dict) -> tuple[np.ndarray, np.ndarray]:
    """Action (0–1) and dialogue (0/1) per second."""
    n = int(math.ceil(a.get("duration") or 0))
    loud = np.zeros(n, np.float32)
    lv = np.asarray(a.get("loud") or [], np.float32)[:n]
    loud[: len(lv)] = lv
    motion = np.zeros(n, np.float32)
    for r in a.get("keyframes", []):
        i = int(r["t"])
        if 0 <= i < n:
            motion[i:i + 2] = r["motion"]
    talk = np.zeros(n, np.float32)
    for d in a.get("dialogue", []):
        talk[int(d["start"]):int(math.ceil(d["end"]))] = 1
    def norm(x):
        if not len(x) or float(x.max()) <= 0:
            return x
        lo, hi = np.percentile(x, 10), np.percentile(x, 98)
        return np.clip((x - lo) / max(1e-6, hi - lo), 0, 1)
    k = np.ones(9) / 9
    action = np.convolve(0.55 * norm(motion) + 0.45 * norm(loud), k, mode="same") if n else np.zeros(0)
    return action, talk


def moments(a: dict, n: int = 12) -> dict:
    """The strongest action peaks, the longest story (dialogue) stretches and quiet scenic stretches."""
    action, talk = per_second(a)
    out: dict = {"action": [], "story": [], "calm": []}
    if not len(action):
        return out
    taken = np.zeros(len(action), bool)
    for i in np.argsort(-action):
        if len(out["action"]) >= n or action[i] < 0.55:
            break
        if taken[max(0, i - 20):i + 20].any():
            continue
        s, e = max(0, i - 12), min(len(action), i + 18)
        taken[s:e] = True
        out["action"].append({"start": int(s), "end": int(e), "score": round(float(action[i]), 2)})
    # story: runs of dialogue (gaps under 8 s joined), longest first
    runs, s0, last = [], None, -99
    for d in a.get("dialogue", []):
        if s0 is None or d["start"] - last > 8:
            if s0 is not None:
                runs.append((s0, last))
            s0 = d["start"]
        last = d["end"]
    if s0 is not None:
        runs.append((s0, last))
    for s, e in sorted(runs, key=lambda r: -(r[1] - r[0]))[:n]:
        if e - s >= 20:
            out["story"].append({"start": int(s), "end": int(e), "lines": sum(1 for d in a["dialogue"] if s <= d["start"] <= e)})
    # calm and scenic: low action, no dialogue, tagged scenic/exploring, at least 20 s
    scenic = np.zeros(len(action), bool)
    for t in a.get("tags", []):
        if t["top"] and t["top"][0] in ("scenic", "exploring"):
            i = int(t["t"])
            scenic[max(0, i - 2):i + int(TAG_EVERY) + 2] = True
    run = None
    for i in range(len(action)):
        ok = bool(action[i] < 0.3 and not talk[i] and scenic[i])
        if ok and run is None:
            run = i
        if (not ok or i == len(action) - 1) and run is not None:
            if i - run >= 20:
                out["calm"].append({"start": int(run), "end": int(i)})
            run = None
    out["calm"] = sorted(out["calm"], key=lambda r: -(r["end"] - r["start"]))[:n]
    return out


# ---------------------------------------------------------------------------------------------
# The game, researched once a week
# ---------------------------------------------------------------------------------------------


def research_game(game: str, writer: str = "claude", fresh: bool = False) -> dict:
    slug = re.sub(r"[^a-z0-9]+", "-", game.lower()).strip("-") or "game"
    f = ac.CACHE_DIR / "games" / f"{slug}.json"
    if f.is_file() and not fresh and time.time() - f.stat().st_mtime < 7 * 86400:
        log(f"research on {game}: from {time.strftime('%d %b', time.localtime(f.stat().st_mtime))} (refreshed weekly)")
        return json.loads(f.read_text(encoding="utf-8"))
    if writer != "claude":
        return {"game": game, "summary": "", "characters": [], "formats": [], "visual": {}}
    import script_writer as sw
    import topic_video as tv

    log(f"Claude is researching {game} on the web (story, characters, bosses, what videos about it do well)")
    prompt = f"""Research the video game "{game}" for a creator who has recorded their own playthrough and wants to publish videos
about it on YouTube (long videos and Shorts) and Instagram Reels, for Indian and global audiences. Today is {time.strftime('%d %B %Y')}.
Use web search: the game's story and acts, main characters (names and how to recognise them), bosses, notable places, weapons,
mechanics, collectibles, secrets and famous moments; and which videos about this game do well right now on YouTube and
Instagram (formats, topics, title patterns, video lengths, what people search for).
Answer with only this JSON:
{{"game": "{game}", "summary": "3-5 sentences", "acts": ["story acts or chapters in order"],
 "characters": [{{"name": "...", "role": "...", "look": "how to recognise them on screen"}}],
 "bosses": ["..."], "places": ["..."], "mechanics": ["..."], "famous_moments": ["..."],
 "formats": [{{"format": "e.g. no-commentary walkthrough part N", "why": "...", "length": "e.g. 20-40 min", "platform": "youtube|shorts|reels"}}],
 "title_patterns": ["5-8 examples of title patterns that work for this game"],
 "searches": ["what people search about this game"],
 "visual": {{"key": "a short English description of a scene typical for this game, for a vision model to recognise (6-10 of them)"}}}}"""
    try:
        r = sw.extract_json(tv.ask_claude_research(prompt, None))
    except AutocutError as e:
        log(f"research failed ({e}); planning from the recordings alone")
        return {"game": game, "summary": "", "characters": [], "formats": [], "visual": {}}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


def vision_prompts(research: dict) -> dict[str, str]:
    p = dict(BASE_PROMPTS)
    for k, v in (research.get("visual") or {}).items():
        k2 = re.sub(r"[^a-z0-9]+", "_", str(k).lower()).strip("_")[:24]
        if k2 and isinstance(v, str) and len(p) < 24:
            p.setdefault(k2, v[:160])
    return p


# ---------------------------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------------------------


def hms(t: float) -> str:
    t = int(max(0, t))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


def digest(idx: int, a: dict, step: int) -> str:
    """One recording as lines the planner can read: every `step` seconds, the action level, what is on
    screen, and what is said; plus its strongest moments."""
    action, _ = per_second(a)
    n = int(math.ceil(a.get("duration") or 0))
    lines = [f"## R{idx}: {a.get('name')} ({hms(n)})"]
    tags = a.get("tags", [])
    for s in range(0, max(n, 1), step):
        e = min(n, s + step)
        act = float(action[s:e].mean()) if len(action) and e > s else 0.0
        tg = [t["top"][0] for t in tags if s <= t["t"] < e and t["top"]]
        common = ",".join(sorted(set(tg), key=tg.count, reverse=True)[:3])
        said = " ".join(d["text"] for d in a.get("dialogue", []) if s <= d["start"] < e)
        said = (said[:220] + "…") if len(said) > 220 else said
        cuts = sum(1 for k in a.get("keyframes", []) if s <= k["t"] < e and k.get("cut"))
        lines.append(f"{hms(s)} act {act:.1f} [{common}] cuts {cuts}" + (f" “{said}”" if said else ""))
    m = a.get("moments") or moments(a)
    for kind in ("action", "story", "calm"):
        if m.get(kind):
            lines.append(f"{kind} peaks: " + ", ".join(f"{hms(x['start'])}-{hms(x['end'])}" for x in m[kind][:8]))
    return "\n".join(lines)


def plan_prompt(game: str, research: dict, digests: list[str], platforms: list[str], count: int, focus: str, earlier: list[str]) -> str:
    plat = {"youtube": "YouTube long videos", "shorts": "YouTube Shorts (under 60 s, vertical)", "reels": "Instagram Reels (under 90 s, vertical)",
            "instagram": "Instagram feed videos"}
    want = ", ".join(plat.get(p, p) for p in platforms)
    return f"""You plan what a gaming creator should publish from their own recordings of "{game}" (recorded on PS5, in playing order).
They post for {want}. Plan {count} videos, the strongest first.

## What is known about the game (researched online)
{json.dumps({k: research.get(k) for k in ("summary", "acts", "characters", "bosses", "places", "famous_moments", "formats", "title_patterns", "searches") if research.get(k)}, ensure_ascii=False)[:9000]}

## The recordings (per stretch: action level 0-1, what is on screen, scene cuts, what is said; then their strongest moments)
{chr(10).join(digests)}

{('## What the creator wants\n' + focus + '\n') if focus else ''}{('## Already planned (do not repeat these)\n' + chr(10).join('- ' + t for t in earlier) + '\n') if earlier else ''}
## Rules
- Use only what is really in the recordings: every video lists its parts as recording + start + end (H:MM:SS, from the lines above).
- Mix formats that work for this game: story episodes or recaps (in order), boss fights, best moments, lore/character explainers,
  "things you missed", scenic/relaxing, funny or epic Shorts, cinematic edits to music. Match what does well (formats above).
- Long videos: 8-40 min, built from whole stretches in order. Shorts/Reels: 15-60 s, one moment with a hook in the first second.
- Titles: specific, curiosity, the game's name first for search; no lies. Hooks: the first line or on-screen text.
- edit_tool: which EI tool makes it: "talk" (cut silences, long videos), "sections" (continuous clips from a long video),
  "clips" (short highlight clips), "music" (montage to music), "topic" (voiceover explainer with a script).
Answer with only this JSON:
{{"plans": [{{"title": "...", "platform": "youtube|shorts|reels|instagram", "format": "...", "length": "e.g. 12 min",
  "hook": "...", "why": "why it will work, under 20 words", "edit_tool": "talk|sections|clips|music|topic",
  "parts": [{{"recording": "R1", "start": "0:12:30", "end": "0:20:10", "note": "what happens"}}],
  "thumbnail": "one-line idea", "tags": ["#..."]}}]}}"""


def parse_t(s) -> float:
    try:
        p = [float(x) for x in str(s).split(":")]
    except ValueError:
        return -1.0
    v = 0.0
    for x in p:
        v = v * 60 + x
    return v


def clean_plans(raw: list, analyses: list[dict]) -> list[dict]:
    out = []
    for p in raw if isinstance(raw, list) else []:
        if not isinstance(p, dict) or not str(p.get("title", "")).strip():
            continue
        parts = []
        for x in p.get("parts") or []:
            m = re.match(r"R(\d+)", str(x.get("recording", "")))
            if not m:
                continue
            i = int(m.group(1)) - 1
            if not 0 <= i < len(analyses):
                continue
            s, e = parse_t(x.get("start")), parse_t(x.get("end"))
            dur = float(analyses[i].get("duration") or 0)
            if s < 0 or e <= s or s >= dur:
                continue
            parts.append({"recording": analyses[i].get("source") or analyses[i].get("name"), "r": i + 1, "start": round(s, 1), "end": round(min(e, dur), 1),
                          "note": str(x.get("note", ""))[:200]})
        if not parts:
            continue
        out.append({"title": str(p["title"]).strip()[:120], "platform": str(p.get("platform", "youtube")), "format": str(p.get("format", ""))[:80],
                    "length": str(p.get("length", ""))[:30], "hook": str(p.get("hook", ""))[:200], "why": str(p.get("why", ""))[:200],
                    "edit_tool": p.get("edit_tool") if p.get("edit_tool") in ("talk", "sections", "clips", "music", "topic") else "sections",
                    "parts": parts, "thumbnail": str(p.get("thumbnail", ""))[:200], "tags": [str(t) for t in (p.get("tags") or [])][:8],
                    "total_s": round(sum(x["end"] - x["start"] for x in parts), 1)})
    return out


def write_md(folder: Path, game: str, plans: list[dict]) -> Path:
    lines = [f"# Video plans: {game}", "", f"{len(plans)} plan(s), strongest first in each batch. Times are positions in each recording.", ""]
    for k, p in enumerate(plans, 1):
        lines += [f"## {k}. {p['title']}", f"*{p['platform']} · {p['format']} · {p['length']} · edit with: {p['edit_tool']}*", "",
                  f"**Hook:** {p['hook']}", f"**Why:** {p['why']}", f"**Thumbnail:** {p['thumbnail']}", "", "Parts:"]
        for x in p["parts"]:
            lines.append(f"- R{x['r']} {Path(str(x['recording'])).name}  {hms(x['start'])}–{hms(x['end'])}  {x['note']}")
        if p["tags"]:
            lines += ["", " ".join(p["tags"])]
        lines.append("")
    f = folder / "plans.md"
    f.write_text("\n".join(lines), encoding="utf-8")
    return f


def make(items: list[str], game: str, *, platforms: list[str] | None = None, count: int = 12, focus: str = "", out_dir: str | None = None,
         analyse_only: bool = False, writer: str = "claude", fresh_research: bool = False) -> dict:
    ac.require_tools()
    entries = resolve_inputs(items)
    game = (game or "").strip() or "this game"
    platforms = platforms or ["youtube", "shorts", "reels"]
    count = max(1, min(40, int(count)))
    ac.progress_plan({"research": 0.04, "analysis": 0.86, "planning": 0.1})
    with ac.stage("research"):
        research = research_game(game, writer, fresh_research) if game != "this game" else {"visual": {}}
    prompts = vision_prompts(research)
    hint = name_hint(research)
    analyses = []
    with ac.stage("analysis"):
        for k, e in enumerate(entries, 1):
            log(f"recording {k}/{len(entries)}: {(e['video'] or e['srt'] or e['analysis']).name}")
            analyses.append(analyse(e, prompts, hint))
            ac.stage_progress(k / len(entries), f"{k} of {len(entries)} recordings analysed")
    first = entries[0]["video"] or entries[0]["srt"] or entries[0]["analysis"]
    folder = Path(out_dir).expanduser().resolve() if out_dir else Path(first).parent / f"{re.sub(r'[^A-Za-z0-9 ]+', '', game).strip() or 'Game'} - plans"
    folder.mkdir(parents=True, exist_ok=True)
    res = {"outputs": [], "out_dir": str(folder), "analyses": [str(e["analysis"] or sidecars(e["video"])[0]) for e in entries]}
    if analyse_only:
        log(f"analysed {len(analyses)} recording(s); plans can be made from them any time")
        return res
    with ac.stage("planning"):
        pj = folder / "plans.json"
        try:
            earlier = json.loads(pj.read_text(encoding="utf-8")).get("plans", []) if pj.is_file() else []
        except ValueError:
            earlier = []
        total = sum(float(a.get("duration") or 0) for a in analyses)
        step = int(np.clip(total / 700, 30, 300) // 30 * 30) or 30  # (about 700 lines in all)
        digests = [digest(i + 1, a, step) for i, a in enumerate(analyses)]
        if writer != "claude":
            raise AutocutError("planning needs Claude (the analysis is saved; plan again when Claude is available)")
        import script_writer as sw

        log(f"Claude is planning {count} video(s) from {len(analyses)} recording(s), {total / 3600:.1f} h ({step} s per line)")
        r = sw.extract_json(sw.ask_claude(plan_prompt(game, research, digests, platforms, count, focus, [p["title"] for p in earlier]), timeout=900))
        new = clean_plans(r.get("plans", []), analyses)
        for p in new:
            p["made"] = time.strftime("%Y-%m-%d %H:%M")
        plans = earlier + new
        pj.write_text(json.dumps({"game": game, "recordings": [a.get("source") or a.get("name") for a in analyses], "plans": plans}, ensure_ascii=False, indent=1),
                      encoding="utf-8")
        md = write_md(folder, game, plans)
    for p in new[:5]:
        log(f"plan: {p['title']} ({p['platform']}, {p['length']})")
    log(f"{len(new)} new plan(s), {len(plans)} in all: {md}")
    res.update(outputs=[str(md), str(pj)], plans=len(plans), new=len(new))
    return res


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("items", nargs="+", help="recordings (in playing order), folders, or their .ei-analysis.json / .srt files")
    p.add_argument("--game", default="", help="the game's name (for the research)")
    p.add_argument("--platforms", nargs="+", default=["youtube", "shorts", "reels"], choices=["youtube", "shorts", "reels", "instagram"])
    p.add_argument("--count", type=int, default=12, help="how many new plans")
    p.add_argument("--focus", default="", help="what you want: e.g. story recap episodes, boss fights, Hindi audience")
    p.add_argument("--out-dir")
    p.add_argument("--analyse-only", action="store_true", help="only analyse (plans later)")
    p.add_argument("--writer", default="claude", choices=["claude", "rules"])
    p.add_argument("--fresh-research", action="store_true", help="research the game again (otherwise weekly)")


def run_args(a) -> dict:
    return make(a.items, a.game, platforms=a.platforms, count=a.count, focus=a.focus, out_dir=a.out_dir, analyse_only=a.analyse_only,
                writer=a.writer, fresh_research=a.fresh_research)


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
