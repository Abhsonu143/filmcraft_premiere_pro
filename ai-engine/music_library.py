"""music_library: music for EI's sound design, sorted by emotional family, from three sources.

    internet  free music found by mood on Openverse (Jamendo, ccMixter, Freesound …) and kept in
              a local library with its credit. Licence "monetized" (default) takes only CC0, CC BY
              and CC BY-SA (fine for monetized videos, with credit); "personal" also takes the
              non-commercial licences.
    folder    your own music: each track is sorted into a family from how it sounds (major or minor,
              tempo, loudness, brightness).
    ai        a score composed on this Mac by MusicGen small (Meta, weights CC BY-NC 4.0: personal
              use only), one cue per family and length, cached.

Library: ~/Library/Application Support/EI/music-library/ (index.json lists every track: family,
source, title, author, licence, page). Downloads happen only for the families a video needs.
"""

from __future__ import annotations

import json
import os
import re
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

import autocut as ac
import emotions as em
from autocut import AutocutError, log

AUDIO_EXT = {".mp3", ".m4a", ".wav", ".aiff", ".aif", ".flac", ".ogg", ".opus", ".aac"}
SOURCES = ("internet", "folder", "ai")
SAFE_LICENCES = ("cc0", "by", "by-sa", "pdm")
PER_FAMILY = 4  # internet tracks kept per family
GOOD_FIT = 0.25  # how well a track must suit a mood (CLAP probability) to be used for it


def library_dir() -> Path:
    base = Path(os.environ["EI_SETTINGS"]).parent if os.environ.get("EI_SETTINGS") else Path.home() / "Library" / "Application Support" / "EI"
    return base / "music-library"


def load_index(lib: Path | None = None) -> dict:
    p = (lib or library_dir()) / "index.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"tracks": []}


def save_index(idx: dict, lib: Path | None = None) -> None:
    lib = lib or library_dir()
    lib.mkdir(parents=True, exist_ok=True)
    tmp = lib / "index.json.part"
    tmp.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(lib / "index.json")


# ---------------------------------------------------------------------------------------------
# How a track sounds -> family (folder source; also checks downloads)
# ---------------------------------------------------------------------------------------------

_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def mode_score(chroma: np.ndarray) -> float:
    """+1 clearly major … -1 clearly minor (Krumhansl key profiles, best key)."""
    c = (chroma - chroma.mean()) / (chroma.std() + 1e-9)

    def best(profile: np.ndarray) -> float:
        p = (profile - profile.mean()) / profile.std()
        return max(float(np.dot(c, np.roll(p, k))) / 12 for k in range(12))

    return float(np.clip((best(_MAJOR) - best(_MINOR)) * 4, -1, 1))


def features(path: str, seconds: float = 90.0) -> dict:
    """Tempo, loudness, brightness, onset density and major/minor of (the middle of) a track."""
    import montage as mt

    dur = ac.probe(path).duration
    start = max(0.0, dur / 2 - seconds / 2)
    sr = 22050
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{seconds:.2f}", "-i", path, "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True).stdout
    y = np.frombuffer(raw, np.float32)
    if len(y) < sr * 3:
        raise AutocutError(f"could not read {Path(path).name}")
    n, hop = 4096, 2048
    fr = np.lib.stride_tricks.sliding_window_view(y, n)[::hop] * np.hanning(n)
    mag = np.abs(np.fft.rfft(fr, axis=1))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    ok = (freqs > 60) & (freqs < 2000)
    pc = (np.round(12 * np.log2(freqs[ok] / 440.0)) % 12).astype(int)
    chroma = np.zeros(12)
    np.add.at(chroma, pc, (mag[:, ok] ** 2).sum(axis=0))
    centroid = float((mag * freqs).sum() / (mag.sum() + 1e-9))
    rms_db = float(20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-9))
    env = mt.onset_envelope(y.astype(np.float32))
    period = mt.estimate_period(env) if len(env) > 16 else 0
    fps = 22050 / mt.HOP if hasattr(mt, "HOP") else 43.0
    tempo = float(60 * fps / period) if period else 0.0
    flux = np.maximum(0, np.diff(mag, axis=0)).sum(axis=1)
    density = float((flux > flux.mean() + flux.std()).mean())
    return {"tempo": round(tempo, 1), "loud_db": round(rms_db, 1), "centroid": round(centroid), "density": round(density, 3),
            "mode": round(mode_score(chroma), 3), "duration": round(dur, 1)}


CLAP_MODEL = "laion/clap-htsat-unfused"
VOCAL_TEXTS = ["instrumental music without vocals", "a song with a singer singing lyrics"]


def _emb(x):
    """The embedding tensor (transformers 5 returns an output object, older versions a tensor)."""
    for k in ("text_embeds", "audio_embeds", "pooler_output"):
        v = getattr(x, k, None)
        if v is not None:
            return v
    return x


def _clap_worker(paths: list[str], texts: list[str]) -> list[list[float] | None]:
    """In its own process: how well each track matches each mood description (CLAP, softmax)."""
    import subprocess

    import numpy as np
    import torch
    from transformers import ClapModel, ClapProcessor

    model = ClapModel.from_pretrained(CLAP_MODEL).eval()
    proc = ClapProcessor.from_pretrained(CLAP_MODEL)
    with torch.no_grad():
        ti = proc(text=texts, return_tensors="pt", padding=True)
        temb = _emb(model.get_text_features(**ti))
        temb = temb / temb.norm(dim=-1, keepdim=True)
        out = []
        for path in paths:
            try:
                dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                                           capture_output=True, text=True).stdout.strip() or 0)
                clips = []
                for frac in (0.2, 0.45, 0.7):  # three 10 s moments of the track
                    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(0.0, dur * frac - 5):.2f}", "-t", "10", "-i", path,
                                          "-ac", "1", "-ar", "48000", "-f", "f32le", "-"], capture_output=True).stdout
                    y = np.frombuffer(raw, np.float32)
                    if len(y) > 48000:
                        clips.append(y)
                if not clips:
                    out.append(None)
                    continue
                ai = proc(audio=clips, sampling_rate=48000, return_tensors="pt")
                aemb = _emb(model.get_audio_features(**ai))
                aemb = aemb / aemb.norm(dim=-1, keepdim=True)
                logits = 30.0 * aemb.mean(dim=0, keepdim=True) @ temb.T  # soft: a track can suit two moods
                k = len(texts) - len(VOCAL_TEXTS)
                moods = torch.softmax(logits[:, :k], dim=-1)[0].tolist()
                vocals = torch.softmax(logits[:, k:], dim=-1)[0].tolist()
                out.append(moods + [vocals[1]])
            except Exception:  # one unreadable track must not lose the others
                out.append(None)
    return out


def listen(paths: list[str]) -> list[dict | None]:
    """Mood of each track by listening (CLAP): {family: probability}, cached; None when CLAP is unavailable."""
    fams = list(em.FAMILIES)
    res: list[dict | None] = [None] * len(paths)
    todo = []
    for i, pth in enumerate(paths):
        cache = ac.CACHE_DIR / "mood" / f"{ac.file_key(pth, 'clap-v3')}.json"
        if cache.is_file():
            res[i] = json.loads(cache.read_text())
        else:
            todo.append(i)
    if todo:
        log(f"listening to {len(todo)} track(s) for their mood (CLAP; the first use downloads it, ~600 MB)")
        try:
            probs = ac.run_isolated(_clap_worker, [paths[i] for i in todo], [em.FAMILIES[f]["listen"] for f in fams] + VOCAL_TEXTS)
        except Exception as e:
            log(f"mood listening unavailable ({type(e).__name__}); using the sound features")
            return res
        for i, pr in zip(todo, probs):
            if pr is None:
                continue
            d = {f: round(float(x), 4) for f, x in zip(fams, pr)}
            d["_vocals"] = round(float(pr[-1]), 4)  # singing under a narration clashes
            cache = ac.CACHE_DIR / "mood" / f"{ac.file_key(paths[i], 'clap-v3')}.json"
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(d))
            res[i] = d
    return res


def fit(track: dict, fam: str) -> float:
    """How well a track suits a family (0–1): its CLAP probability, else the feature distance."""
    moods = track.get("moods")
    if moods:
        return float(moods.get(fam, 0.0)) * (1.0 - 0.8 * float(moods.get("_vocals", 0.0)))
    return float(max(0.0, 1.0 - em.family_distance(fam, track["family"]) / 1.5)) * 0.3


def classify(f: dict) -> tuple[str, float, float]:
    """(family, valence, arousal) from features (fallback when CLAP cannot listen)."""
    tempo = f.get("tempo") or 100.0
    if tempo <= 0:
        tempo = 100.0
    while tempo > 170:
        tempo /= 2
    arousal = np.clip(0.35 * (tempo - 60) / 100 + 0.35 * (f["loud_db"] + 30) / 20 + 0.3 * min(1.0, f["density"] * 4), 0, 1)
    valence = np.clip(0.65 * f["mode"] + 0.35 * (f["centroid"] - 1500) / 1500, -1, 1)
    return em.nearest_family(float(valence), float(arousal)), round(float(valence), 2), round(float(arousal), 2)


def scan_folder(folder: str) -> list[dict]:
    """Your own music, each track sorted into a family (cached by file)."""
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise AutocutError(f"music folder not found: {folder}")
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXT or p.name.startswith("."):
            continue
        key = ac.file_key(str(p), "mood-v1")
        cache = ac.CACHE_DIR / "mood" / f"{key}.json"
        try:
            if cache.is_file():
                rec = json.loads(cache.read_text())
            else:
                f = features(str(p))
                fam, v, a = classify(f)
                rec = {"file": str(p), "family": fam, "valence": v, "arousal": a, "source": "folder", "title": p.stem, "author": "",
                       "license": "your music", "page": "", **f}
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(rec))
            out.append(rec)
        except (AutocutError, OSError, ValueError) as e:
            log(f"skipped {p.name} ({e})")
    for rec, moods in zip(out, listen([r["file"] for r in out])):
        if moods:
            rec["moods"] = moods
            rec["family"] = max((k for k in moods if not k.startswith("_")), key=moods.get)
    log(f"{len(out)} track(s) in your music folder: " + ", ".join(f"{k} {v}" for k, v in sorted(_count(out).items())))
    return out


def _count(tracks: list[dict]) -> dict[str, int]:
    c: dict[str, int] = {}
    for t in tracks:
        c[t["family"]] = c.get(t["family"], 0) + 1
    return c


# ---------------------------------------------------------------------------------------------
# Internet (Openverse)
# ---------------------------------------------------------------------------------------------


def _ctx() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def openverse(query: str, licence: str = "monetized", page_size: int = 20, page: int = 1) -> list[dict]:
    page_size = min(20, page_size)  # the most the API allows without an account
    q = {"q": query, "page_size": str(page_size), "category": "music", "page": str(page)}
    if licence == "monetized":
        q["license_type"] = "commercial,modification"
    url = "https://api.openverse.org/v1/audio/?" + urllib.parse.urlencode(q)
    data = None
    for wait in (0, 3, 10, 25):  # the free API limits anonymous searches: wait and try again
        if wait:
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "EI video editor (personal use)"})
            data = json.load(urllib.request.urlopen(req, context=_ctx(), timeout=30))
            break
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                log(f"music search failed for “{query}” (HTTP {e.code})")
                return []
        except (OSError, ValueError) as e:
            log(f"music search failed for “{query}” ({type(e).__name__})")
            return []
    if data is None:
        log(f"music search for “{query}” is rate-limited right now; using the library")
        return []
    out = []
    for r in data.get("results", []):
        lic = str(r.get("license", "")).lower()
        dur = (r.get("duration") or 0) / 1000
        if licence == "monetized" and lic not in SAFE_LICENCES:
            continue
        if not (40 <= dur <= 420) or not r.get("url"):
            continue
        out.append({"id": r.get("id"), "url": r["url"], "title": r.get("title") or "untitled", "author": r.get("creator") or "",
                    "license": f"CC {lic.upper()} {r.get('license_version') or ''}".strip() if lic not in ("cc0", "pdm") else lic.upper(),
                    "page": r.get("foreign_landing_url") or "", "provider": r.get("source") or "", "duration": dur, "code": lic})
    return out


def _download(r: dict, folder: Path) -> Path | None:
    dst = folder / f"{re.sub(r'[^A-Za-z0-9]+', '-', r['title'])[:50].strip('-')}-{str(r['id'])[:8]}.mp3"
    part = dst.with_suffix(".part")
    try:
        req = urllib.request.Request(r["url"], headers={"User-Agent": "EI video editor (personal use)"})
        with urllib.request.urlopen(req, context=_ctx(), timeout=90) as resp, open(part, "wb") as fh:
            fh.write(resp.read(40_000_000))
        part.replace(dst)
        return dst
    except OSError as e:
        log(f"skipped “{r['title']}” ({type(e).__name__})")
        part.unlink(missing_ok=True)
        return None


def fetch_family(fam: str, licence: str = "monetized", want: int = PER_FAMILY, lib: Path | None = None, batch: int = 6) -> list[dict]:
    """Make sure the library has `want` internet tracks that sound like this family; returns them.
    Searches by mood words, downloads a few at a time, listens to them (CLAP) and files every track
    under the mood it really has, so a miss for one mood is a find for another."""
    lib = lib or library_dir()
    idx = load_index(lib)
    unheard = [t for t in idx["tracks"] if "_vocals" not in (t.get("moods") or {}) and Path(t["file"]).is_file()]
    if unheard:  # tracks from before EI could listen: listen once and remember
        for t, moods in zip(unheard, listen([t["file"] for t in unheard])):
            t["moods"] = moods
            if moods:
                t["family"] = max((k for k in moods if not k.startswith("_")), key=moods.get)
        save_index(idx, lib)

    def suits(t: dict) -> bool:  # by what it sounds like (whatever it was searched for)
        return fit(t, fam) >= GOOD_FIT if t.get("moods") else t["family"] == fam

    def usable(t: dict) -> bool:
        return t["source"] == "internet" and (licence == "personal" or bool(t.get("safe"))) and Path(t["file"]).is_file()

    have = [t for t in idx["tracks"] if usable(t) and suits(t)]
    if len(have) >= want:
        return have
    seen = {t.get("id") for t in idx["tracks"]}
    folder = lib / "tracks"
    folder.mkdir(parents=True, exist_ok=True)
    tried = 0
    for query, page in [(q, pg) for pg in (1, 2) for q in em.FAMILIES[fam]["search"]]:
        if len(have) >= want or tried >= 30:
            break
        fresh = [r for r in openverse(query, licence, page_size=20, page=page) if r["id"] not in seen]
        while fresh and len(have) < want and tried < 30:
            group, fresh = fresh[:batch], fresh[batch:]
            got = []
            for r in group:
                seen.add(r["id"])
                tried += 1
                dst = _download(r, folder)
                if dst is None:
                    continue
                try:
                    got.append((r, dst, features(str(dst))))
                except AutocutError as e:
                    log(f"skipped “{r['title']}” ({e})")
                    dst.unlink(missing_ok=True)
            heard = listen([str(d) for _, d, _ in got]) if got else []
            for (r, dst, f), moods in zip(got, heard):
                fam_heard, v, a = classify(f)
                if moods:
                    fam_heard = max((k for k in moods if not k.startswith("_")), key=moods.get)
                elif em.family_distance(fam, fam_heard) <= 1.25:
                    fam_heard = fam
                rec = {"file": str(dst), "family": fam_heard, "searched": fam, "valence": v, "arousal": a, "source": "internet", "moods": moods,
                       "safe": r["code"] in SAFE_LICENCES, **{k: r[k] for k in ("id", "title", "author", "license", "page", "provider")}, **f}
                idx["tracks"].append(rec)
                if usable(rec) and suits(rec):
                    have.append(rec)
                    log(f"music for {fam}: “{r['title']}” by {r['author'] or 'unknown'} ({r['license']})")
                else:
                    log(f"“{r['title']}” sounds {fam_heard}, not {fam}: kept in the library for {fam_heard}")
            save_index(idx, lib)
    return have


# ---------------------------------------------------------------------------------------------
# AI score (MusicGen small, in its own process)
# ---------------------------------------------------------------------------------------------

AI_MODEL = "facebook/musicgen-small"


def _musicgen_worker(jobs: list[dict]) -> list[dict]:
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(AI_MODEL)
    model = MusicgenForConditionalGeneration.from_pretrained(AI_MODEL).to(dev)
    sr = model.config.audio_encoder.sampling_rate
    out = []
    for j in jobs:
        torch.manual_seed(int(j.get("seed", 0)))
        inputs = proc(text=[j["prompt"]], padding=True, return_tensors="pt").to(dev)
        tokens = int(j["seconds"] * 50) + 8
        with torch.no_grad():
            audio = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=tokens)
        y = audio[0, 0].float().cpu().numpy()
        y = y / (np.abs(y).max() + 1e-9) * 0.9
        sf.write(j["out"], y, sr)
        out.append({"ok": True, "out": j["out"]})
    return out


def ai_cue(fam: str, seconds: float, seed: int = 0, lib: Path | None = None) -> dict:
    """A cue composed for a family (at most 30 s; longer acts loop it), cached."""
    lib = lib or library_dir()
    seconds = float(np.clip(seconds, 8, 30))
    seconds = float(np.ceil(seconds / 5) * 5)
    dst = lib / "ai" / f"{fam}-{int(seconds)}s-{seed}.wav"
    idx = load_index(lib)
    for t in idx["tracks"]:
        if t["file"] == str(dst) and dst.is_file():
            return t
    dst.parent.mkdir(parents=True, exist_ok=True)
    prompt = em.FAMILIES[fam]["prompt"] + ", instrumental, no vocals, high quality film score"
    log(f"composing {seconds:.0f} s of {fam} music on this Mac (MusicGen small, the first use downloads it, ~1.2 GB)")
    t0 = time.time()
    try:
        res = ac.run_isolated(_musicgen_worker, [{"prompt": prompt, "seconds": seconds, "seed": seed, "out": str(dst)}])
    except Exception as e:
        raise AutocutError(f"the AI score could not be made ({type(e).__name__}: {e})") from e
    if not res or not res[0].get("ok") or not dst.is_file():
        raise AutocutError("the AI score could not be made")
    log(f"composed in {time.time() - t0:.0f}s")
    f = features(str(dst), seconds)
    rec = {"file": str(dst), "family": fam, "valence": em.FAMILIES[fam]["valence"], "arousal": em.FAMILIES[fam]["arousal"], "source": "ai",
           "title": f"{fam} cue (AI)", "author": "MusicGen small (Meta)", "license": "CC BY-NC 4.0 model output, personal use", "page": "", **f}
    idx["tracks"].append(rec)
    save_index(idx, lib)
    return rec


# ---------------------------------------------------------------------------------------------
# Choosing
# ---------------------------------------------------------------------------------------------


def candidates(fam: str, source: str, *, folder: str | None = None, licence: str = "monetized", length: float = 30.0,
               seed: int = 0, lib: Path | None = None) -> list[dict]:
    """Tracks for a family from a source, best first (nearest families when the family has none)."""
    if source == "internet":
        pool = fetch_family(fam, licence, lib=lib)
        if not pool:  # offline or nothing found: whatever the library already has
            pool = [t for t in load_index(lib)["tracks"] if t["source"] == "internet" and Path(t["file"]).is_file()]
    elif source == "folder":
        if not folder:
            raise AutocutError("choose your music folder")
        pool = scan_folder(folder)
    elif source == "ai":
        pool = [ai_cue(fam, length, seed, lib)]
    else:
        raise AutocutError(f"unknown music source {source!r} ({', '.join(SOURCES)})")
    return sorted(pool, key=lambda t: -fit(t, fam))
