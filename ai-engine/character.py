"""character: an animated 3D presenter (head to stomach) that speaks a voiceover, for any EI video.

    python character.py --voice narration.wav --narration narration.json --video video.mp4 \
        --character Seed-san --look studio --position bottom-right --size 0.3 --out video-with-character.mp4

What makes it read as a person rather than a puppet:

  lips        every word's sounds become mouth shapes (aa, ih, ou, ee, oh, closed lips on m/b/p), in
              English and Hindi, blended into each other and opened by how loud the voice really is
              at that instant, so the mouth closes in every pause.
  face        each script line's emotion (the same 43 the voice and the music use) becomes a mix of
              expressions that swells at the start of the line and eases, never a frozen mask;
              brows lift on stressed words; blinks come at phrase ends, sometimes double.
  body        it breathes, shifts its weight, and its head never stops moving a little; it nods on
              the stressed words, tilts and turns anew with every line, leans in when intense and
              sinks when sad. Hair and cloth swing behind the movement (vrm.Rig).
  hands       gestures are chosen from what is being said (a wave on a greeting, a raised finger on
              a number or "important", a shrug on a question, a hand on the chest on "I", an open
              palm while explaining, a stop on "no", both arms wide on "huge", a fist on a win…),
              timed so the stroke lands on the stressed word, held through the phrase and let go;
              never the same one twice running, smaller when calm and larger when excited.
  smoothing   arms and head are driven through a spring, so they overshoot a little and settle.

Everything decided is written to `<out>.character.json` (the plan: mouth, emotions, gestures,
nods, blinks). Edit it and run again with `--plan` to change a gesture or an expression.

Looks: `studio` (the character in a small studio or room, shown in a circle over the video),
`cutout` (no background, standing in a corner of the video) and `green` (a separate clip on a
green screen; the video can still get the cut-out). The character alone is also saved as a clip.

No AI model is used: drawing is on the GPU (vrm.py), about ten times faster than real time.
Other tools add a character with `add_character_args(parser)`, `options(args)` and `add_to_video`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import autocut as ac
import vrm
from autocut import AutocutError, log

LOOKS = ("studio", "cutout", "green")
POSITIONS = {"bottom-right": (1.0, 1.0), "bottom-left": (0.0, 1.0), "top-right": (1.0, 0.0), "top-left": (0.0, 0.0), "bottom-center": (0.5, 1.0)}
BACKGROUNDS = ("studio", "room", "color")
ENERGY = {"calm": 0.35, "normal": 0.65, "energetic": 1.0}
GREEN = (0, 177, 64)
VOWELS = ("aa", "ih", "ou", "ee", "oh")
MAX_SECONDS = 3 * 3600.0


def characters_dir() -> Path:
    import os

    base = Path(os.environ["EI_SETTINGS"]).parent if os.environ.get("EI_SETTINGS") else Path.home() / "Library" / "Application Support" / "EI"
    return base / "characters"


def list_characters() -> list[dict]:
    out = []
    d = characters_dir()
    for p in sorted(d.glob("*/character.json")) if d.is_dir() else []:
        try:
            prof = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (p.parent / str(prof.get("file", "model.vrm"))).is_file():
            out.append({"name": p.parent.name, **prof})
    return out


def add_character(path: str, name: str | None = None) -> dict:
    """Copy a VRM file into EI's characters (your own from VRoid Studio, or any VRM you may use)."""
    m = vrm.Model(path)  # refuses what is not a usable character
    name = re.sub(r"[^\w\- ]+", "", name or m.name or Path(path).stem).strip()[:40] or "character"
    folder = characters_dir() / name
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, folder / "model.vrm")
    meta = m.meta
    prof = {"file": "model.vrm", "title": m.name, "author": ", ".join(meta.get("authors") or [str(meta.get("author") or "")]).strip(", "),
            "licence": str(meta.get("licenseUrl") or meta.get("licenseName") or meta.get("otherLicenseUrl") or ""),
            "credit": str(meta.get("creditNotation") or ""), "tints": {}, "zoom": 1.0}
    (folder / "character.json").write_text(json.dumps(prof, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"character “{name}” added ({m.name}; {len(m.shapes)} face shapes, {len(m.bones)} bones)")
    return {"name": name, **prof}


def resolve_character(name: str | None) -> tuple[str, dict]:
    """(VRM path, profile) for a character name, a .vrm path, or the first character when none is named."""
    if name and Path(name).expanduser().is_file():
        return str(Path(name).expanduser().resolve()), {"tints": {}, "zoom": 1.0}
    have = list_characters()
    if name:
        for c in have:
            if c["name"].lower() == name.lower():
                return str(characters_dir() / c["name"] / c["file"]), c
        raise AutocutError(f"no character named “{name}” (have: {', '.join(c['name'] for c in have) or 'none'}). Add one with: character --add your.vrm")
    if not have:
        raise AutocutError("no character yet: make one in VRoid Studio (free), export it as VRM, and add it with: character --add your.vrm")
    return str(characters_dir() / have[0]["name"] / have[0]["file"]), have[0]


# ---------------------------------------------------------------------------------------------
# Speech: loudness and mouth shapes
# ---------------------------------------------------------------------------------------------


def loudness(wav: str, fps: float) -> np.ndarray:
    """How loud the voice is in each video frame, 0–1 (1 = its usual loud level)."""
    import footage as fo

    sr = 16000
    hop = max(1, int(round(sr / fps)))
    parts, carry = [], np.zeros(0, np.float32)
    for y in fo.iter_audio(wav, sr, 120.0):
        y = np.concatenate([carry, y])
        n = len(y) // hop
        if n:
            parts.append(np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(axis=1)))
        carry = y[n * hop :]
    rms = np.concatenate(parts) if parts else np.zeros(1, np.float32)
    top = float(np.percentile(rms, 95)) if len(rms) else 0.0
    if top < 1e-5:
        return np.zeros(len(rms), np.float32)
    return np.clip((rms / top - 0.06) / 0.94, 0.0, 1.0).astype(np.float32)


LATIN = {"a": "aa", "e": "ee", "i": "ih", "y": "ih", "o": "oh", "u": "ou"}
PAIRS = {"oo": "ou", "ee": "ee", "ea": "ee", "ou": "oh", "ow": "oh", "ai": "ee", "ay": "ee", "au": "oh", "aw": "oh", "ie": "ee", "oa": "oh"}
DEVA_VOWEL = {"अ": "aa", "आ": "aa", "ा": "aa", "इ": "ih", "ई": "ih", "ि": "ih", "ी": "ih", "उ": "ou", "ऊ": "ou", "ु": "ou", "ू": "ou",
              "ए": "ee", "ऐ": "ee", "े": "ee", "ै": "ee", "ओ": "oh", "औ": "oh", "ो": "oh", "ौ": "oh", "ऑ": "oh", "ॉ": "oh", "ृ": "ih"}
DEVA_LIPS = set("पफबभम")


def mouth_units(word: str) -> list[tuple[str, float]]:
    """The mouth shapes of one word, in order: (shape, opening 0–1). shape "closed" = lips together
    (m, b, p), "rest" = a consonant that only narrows the mouth."""
    w = word.strip().lower()
    out: list[tuple[str, float]] = []

    def add(shape: str, amount: float) -> None:
        if out and out[-1][0] == shape and shape in ("rest", "closed"):
            return
        out.append((shape, amount))

    if re.search(r"[ऀ-ॿ]", w):
        chars = list(w)
        for k, ch in enumerate(chars):
            if ch in DEVA_VOWEL:
                add(DEVA_VOWEL[ch], 1.0)
            elif "क" <= ch <= "ह" or ch in "क़ख़ग़ज़ड़ढ़फ़य़":
                add("closed" if ch in DEVA_LIPS else "rest", 0.0 if ch in DEVA_LIPS else 0.25)
                nxt = chars[k + 1] if k + 1 < len(chars) else ""
                if nxt not in DEVA_VOWEL and nxt != "्" and k + 1 < len(chars):  # the vowel every consonant carries
                    add("aa", 0.6)
            elif ch in "ंँ":
                add("closed", 0.0)
        return out or [("aa", 0.6)]
    w = re.sub(r"[^a-z0-9]", "", w)
    if not w:
        return []
    if w.isdigit():
        return [("aa", 0.8), ("rest", 0.25)] * min(3, len(w))
    if len(w) > 3 and w.endswith("e") and w[-2] not in "aeiou":
        w = w[:-1]  # a silent e
    k = 0
    while k < len(w):
        two = w[k : k + 2]
        ch = w[k]
        if two in PAIRS:
            add(PAIRS[two], 1.0)
            k += 2
            continue
        if ch in LATIN:
            add(LATIN[ch], 1.0)
        elif ch in "mbp":
            add("closed", 0.0)
        elif ch in "fv":
            add("ih", 0.3)
        elif ch == "w":
            add("ou", 0.6)
        else:
            add("rest", 0.25)
        k += 1
    return out


def mouth_keys(words: list[dict]) -> list[list]:
    """[time, shape, opening] through the words (time = the middle of each sound)."""
    keys: list[list] = []
    for w in words:
        units = mouth_units(str(w.get("word", "")))
        a, b = float(w["start"]), float(w["end"])
        if not units or b <= a:
            continue
        step = (b - a) / len(units)
        for k, (shape, amount) in enumerate(units):
            keys.append([round(a + (k + 0.5) * step, 3), shape, amount])
    return keys


# ---------------------------------------------------------------------------------------------
# Emotion -> face and body
# ---------------------------------------------------------------------------------------------

# family -> face mix, and how the body carries it: head pitch (+ = down), lean forward, how big gestures are
FACE = {
    "joy": ({"happy": 0.7}, {"pitch": -2, "lean": 1, "amp": 1.0}),
    "euphoria": ({"happy": 0.85, "surprised": 0.2}, {"pitch": -4, "lean": 3, "amp": 1.25}),
    "love": ({"happy": 0.45, "relaxed": 0.4}, {"pitch": 2, "lean": 1, "amp": 0.7, "tilt": 5}),
    "nostalgia": ({"relaxed": 0.45, "sad": 0.25}, {"pitch": 3, "lean": -1, "amp": 0.6, "tilt": 4}),
    "sorrow": ({"sad": 0.8}, {"pitch": 9, "lean": -2, "amp": 0.45}),
    "fear": ({"surprised": 0.55, "sad": 0.3}, {"pitch": 1, "lean": -3, "amp": 0.8}),
    "tension": ({"angry": 0.3, "surprised": 0.15}, {"pitch": 3, "lean": 4, "amp": 0.85}),
    "mystery": ({"relaxed": 0.2, "surprised": 0.25}, {"pitch": 1, "lean": 3, "amp": 0.75, "tilt": 6}),
    "wonder": ({"surprised": 0.6, "happy": 0.3}, {"pitch": -5, "lean": 1, "amp": 1.1}),
    "calm": ({"relaxed": 0.5}, {"pitch": 1, "lean": 0, "amp": 0.6}),
    "heroic": ({"happy": 0.4, "angry": 0.15}, {"pitch": -4, "lean": 2, "amp": 1.2}),
    "anger": ({"angry": 0.85}, {"pitch": 5, "lean": 5, "amp": 1.15}),
}
NEUTRAL = ({"happy": 0.18}, {"pitch": 0, "lean": 0, "amp": 0.8})


def feeling(emotion: str | None, intensity: float) -> tuple[dict, dict]:
    """(face mix, body) for a script line's emotion and intensity."""
    import emotions as em

    e = em.EMOTIONS.get(str(emotion or ""))
    face, body = NEUTRAL if e is None or emotion == "Neutral" else FACE.get(e["family"], NEUTRAL)
    k = 0.45 + 0.55 * float(min(1.0, max(0.0, intensity)))
    return {n: round(v * k, 3) for n, v in face.items()}, {"pitch": body.get("pitch", 0) * k, "lean": body.get("lean", 0) * k,
                                                             "tilt": body.get("tilt", 0) * k, "amp": 0.7 + (body.get("amp", 1.0) - 0.7) * k}


# ---------------------------------------------------------------------------------------------
# Gestures
# ---------------------------------------------------------------------------------------------

# An arm, in degrees: down (0 = straight out to the side, 78 = hanging), fwd (swung forward), twist
# (upper arm turned inward), elbow (bend), turn (forearm: - palm up), wrist (+ hand tipped up),
# curl (fingers, 0 open – 1 fist), index (the index finger alone; -1 = like the others).
ARM = ("down", "fwd", "twist", "elbow", "turn", "wrist", "curl", "index")
REST = dict(down=76, fwd=4, twist=10, elbow=16, turn=0, wrist=0, curl=0.35, index=-1)
READY = dict(down=70, fwd=10, twist=38, elbow=72, turn=20, wrist=4, curl=0.3, index=-1)
# kind -> pose of the gesturing arm ("a"), of the other one ("o", else it stays as it is), both
# arms alike ("both"), what the head and shoulders add, and a movement while it is held
GESTURES: dict[str, dict] = {
    "explain": {"a": dict(down=64, fwd=36, twist=8, elbow=86, turn=-78, wrist=-6, curl=0.1, index=-1)},
    "present": {"a": dict(down=52, fwd=18, twist=-28, elbow=66, turn=-72, wrist=-4, curl=0.08, index=-1), "head": {"yaw": 9}},
    "open_both": {"both": dict(down=56, fwd=28, twist=-12, elbow=88, turn=78, wrist=-8, curl=0.06, index=-1), "head": {"pitch": -3}},
    "point_up": {"a": dict(down=58, fwd=42, twist=18, elbow=128, turn=10, wrist=6, curl=0.92, index=0.0), "head": {"pitch": -2}, "move": "bounce"},
    "chest": {"a": dict(down=56, fwd=30, twist=74, elbow=138, turn=-10, wrist=8, curl=0.14, index=-1), "head": {"roll": 3, "pitch": 2}},
    "stop": {"a": dict(down=50, fwd=52, twist=10, elbow=84, turn=-25, wrist=62, curl=0.0, index=-1), "head": {"yaw": -4}, "move": "shake"},
    "fist": {"a": dict(down=60, fwd=34, twist=16, elbow=122, turn=30, wrist=4, curl=1.0, index=-1), "head": {"pitch": 3}, "move": "pump"},
    "think": {"a": dict(down=58, fwd=58, twist=58, elbow=160, turn=20, wrist=14, curl=0.62, index=0.25), "head": {"roll": 6, "pitch": 3, "yaw": 5}},
    "shrug": {"both": dict(down=62, fwd=16, twist=-30, elbow=92, turn=-72, wrist=-12, curl=0.1, index=-1), "head": {"roll": 7, "pitch": -2}, "shoulders": 9},
    "wave": {"a": dict(down=44, fwd=24, twist=-62, elbow=122, turn=-70, wrist=10, curl=0.0, index=-1), "head": {"roll": -4}, "move": "wave"},
    "namaste": {"both": dict(down=70, fwd=34, twist=31, elbow=120, turn=-90, wrist=30, curl=0.0, index=-1), "head": {"pitch": 10}, "bow": 7},
    "beat": {"both": dict(down=66, fwd=18, twist=30, elbow=88, turn=45, wrist=0, curl=0.2, index=-1)},
}
# words that call for a gesture (English, Hindi, Hinglish); looked up in the phrase being spoken
CUES: list[tuple[str, str]] = [
    ("namaste", r"\b(namaste|namaskar|thank you|thanks|dhanyavaad|shukriya)\b|नमस्ते|नमस्कार|धन्यवाद|शुक्रिया"),
    ("wave", r"\b(hi|hello|hey|welcome|bye|goodbye|see you|dosto|doston)\b|हेलो|दोस्तों|स्वागत|अलविदा"),
    ("stop", r"\b(no|not|never|stop|don'?t|can'?t|won'?t|ban|banned|refused|wait|mat|nahi|nahin|ruko)\b|नहीं|मत|रुको|कभी नहीं"),
    ("point_up", r"\b(one|first|number|important|remember|listen|note|secret|key|only|rule|sabse|zaroori|yaad|pehla|pehle|ek)\b|\d|सबसे|ज़रूरी|जरूरी|याद|पहला|पहले|एक"),
    ("shrug", r"\b(why|how|maybe|perhaps|who knows|no idea|don'?t know|guess|kya|kyun|kyon|kaise|pata nahi|shayad)\b|क्या|क्यों|कैसे|पता नहीं|शायद"),
    ("open_both", r"\b(big|huge|massive|giant|all|every|everyone|everything|whole|entire|world|million|billion|bada|pura|poora|sab|sabhi|duniya)\b|बड़ा|पूरा|सब|सभी|दुनिया|करोड़|लाख"),
    ("chest", r"\b(i|i'?m|me|my|mine|myself|honestly|personally|main|mera|meri|mujhe|maine)\b|मैं|मेरा|मेरी|मुझे|मैंने"),
    ("fist", r"\b(win|won|wins|record|finally|yes|power|strong|best|victory|champion|jeet|jeeta|zabardast)\b|जीत|ज़बरदस्त|जबरदस्त|रिकॉर्ड"),
    ("think", r"\b(think|imagine|wonder|hmm|suppose|what if|consider|socho|sochiye|lagta)\b|सोचो|सोचिए|लगता"),
    ("present", r"\b(look|see|this|here|watch|check|these|dekho|dekhiye|yeh|ye|yahan)\b|देखो|देखिए|यह|ये|यहाँ|यहां"),
]
FAMILY_GESTURE = {"wonder": "open_both", "euphoria": "open_both", "heroic": "fist", "anger": "fist", "mystery": "think", "love": "chest",
                  "sorrow": "chest", "fear": "stop"}
FALLBACK = ["explain", "beat", "present", "open_both", "explain"]


def split_phrases(words: list[dict], gap: float = 0.28) -> list[list[dict]]:
    out, cur = [], []
    for w in words:
        if cur and (float(w["start"]) - float(cur[-1]["end"]) > gap):
            out.append(cur)
            cur = []
        cur.append(w)
        if str(w.get("word", "")).strip()[-1:] in ",;:.?!—–।":
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def stressed(words: list[dict], loud: np.ndarray, fps: float, min_gap: float = 0.45) -> list[dict]:
    """The words the voice leans on: louder than their neighbours, or long content words."""
    if not words:
        return []
    peak = []
    for w in words:
        a, b = int(float(w["start"]) * fps), max(int(float(w["start"]) * fps) + 1, int(float(w["end"]) * fps))
        seg = loud[a:b]
        peak.append(float(seg.max()) if len(seg) else 0.0)
    out, last = [], -9.0
    for k, w in enumerate(words):
        near = peak[max(0, k - 3) : k + 4]
        base = (sum(near) - peak[k]) / max(1, len(near) - 1)
        text = str(w.get("word", "")).strip()
        strong = peak[k] > 0.25 and (peak[k] >= 1.15 * base or len(re.sub(r"\W", "", text)) >= 7 or text[-1:] in "!?")
        if strong and float(w["start"]) - last >= min_gap:
            t = float(w["start"]) + min(0.12, (float(w["end"]) - float(w["start"])) * 0.35)
            out.append({"t": round(t, 3), "word": text, "power": round(min(1.0, peak[k]), 3)})
            last = float(w["start"])
    return out


def plan_gestures(spans: list[dict], words: list[dict], stress: list[dict], energy: float, side: str, rng: np.random.Generator) -> list[dict]:
    """Which gesture, when and with which hand. `side`: the arm that points towards the middle of
    the picture ("l" or "r", the character's own)."""
    import emotions as em

    out: list[dict] = []
    last_kind, last_t, turn = None, -9.0, int(rng.integers(0, len(FALLBACK)))
    recent: list[str] = []  # a gesture comes back only after two others
    min_gap = 2.6 - 1.2 * energy  # calm: a gesture every few seconds; energetic: almost every phrase
    for si, sp in enumerate(spans):
        fam = (em.EMOTIONS.get(str(sp.get("emotion") or "")) or {}).get("family")
        mine = [w for w in words if sp["start"] - 0.05 <= float(w["start"]) < sp["end"]]
        for ph in split_phrases(mine):
            a, b = float(ph[0]["start"]), float(ph[-1]["end"])
            if b - a < 0.35:
                continue
            text = " ".join(str(w.get("word", "")).strip() for w in ph).lower()
            kind, at = None, None
            for name, pat in CUES:
                mt = re.search(pat, text)
                if not mt or name in recent[-2:]:
                    continue
                if name in ("wave", "namaste") and not (si == 0 or si == len(spans) - 1):
                    continue  # greetings open or close the video
                # the stroke lands on the cue word
                pos, acc = mt.start(), 0
                for w in ph:
                    acc += len(str(w.get("word", "")).strip()) + 1
                    if acc > pos:
                        at = float(w["start"]) + 0.05
                        break
                kind = name
                break
            here = [s for s in stress if a <= s["t"] <= b]
            cue = kind is not None
            if kind is None:
                if not here or rng.random() > 0.45 + 0.5 * energy:
                    continue
                if text.rstrip().endswith("?") and last_kind != "shrug":
                    kind = "shrug"
                elif fam in FAMILY_GESTURE and FAMILY_GESTURE[fam] != last_kind and rng.random() < 0.5:
                    kind = FAMILY_GESTURE[fam]
                else:
                    kind = FALLBACK[turn % len(FALLBACK)]
                    turn += 1
                    if kind == last_kind:
                        kind = FALLBACK[turn % len(FALLBACK)]
                        turn += 1
            if at is None:
                at = max(here, key=lambda s: s["power"])["t"] if here else a + 0.15
            if at - last_t < (1.3 if cue else min_gap):
                continue
            hold = float(np.clip(b - at, 0.4, 1.9 if kind not in ("wave", "namaste", "think") else 2.4))
            g = GESTURES[kind]
            arm = "both" if "both" in g else (side if kind in ("present", "stop", "wave") else ("r" if (len(out) % 3) else "l"))
            out.append({"t": round(at, 3), "kind": kind, "arm": arm, "hold": round(hold, 2), "size": round(float(np.clip(0.75 + 0.3 * energy + 0.1 * rng.standard_normal(), 0.5, 1.25)), 2)})
            last_kind, last_t = kind, at + hold
            recent.append(kind)
    return out


def plan_blinks(total: float, spans: list[dict], gestures: list[dict], rng: np.random.Generator) -> list[float]:
    out, t = [], 0.6 + float(rng.random())
    marks = sorted([sp["end"] + 0.1 for sp in spans] + [g["t"] - 0.25 for g in gestures])
    while t < total:
        out.append(round(t, 3))
        if rng.random() < 0.12:
            out.append(round(t + 0.28, 3))  # a double blink
        nxt = t + float(np.clip(rng.normal(3.4, 1.1), 1.4, 6.5))
        near = [m for m in marks if t + 1.2 < m < nxt + 0.6]
        t = near[0] if near and rng.random() < 0.6 else nxt  # people blink at the end of a thought
    return out


def spans_from(narration: dict, timeline: dict | None) -> list[dict]:
    """Where each line is heard in the video. Without `timeline` (the edit's own lines, e.g. after a
    cold open was put in front) the video's time is the narration's."""
    segs = [s for s in narration.get("segments") or [] if float(s.get("end", 0)) > float(s.get("start", 0))]
    if not segs:
        raise AutocutError("the narration has no lines")
    base = [{"start": float(s["start"]), "end": float(s["end"]), "src": float(s["start"]), "text": str(s.get("text", "")),
             "emotion": s.get("emotion"), "intensity": float(s.get("intensity") or 0.6)} for s in segs]
    lines = [ln for ln in (timeline or {}).get("lines") or [] if str(ln.get("text", "")).strip()]
    if not lines:
        return base
    by_text = {re.sub(r"\W+", " ", b["text"]).strip().lower(): b for b in base}
    out = []
    for ln in lines:
        b = by_text.get(re.sub(r"\W+", " ", str(ln["text"])).strip().lower())
        if b is None:
            continue
        a = float(ln["start"])
        e = min(float(ln["end"]), a + (b["end"] - b["src"]))
        out.append({**b, "start": a, "end": e, "emotion": ln.get("emotion") or b["emotion"], "intensity": float(ln.get("intensity") or b["intensity"])})
    return out or base


def make_plan(spans: list[dict], words: list[dict], loud: np.ndarray, fps: float, total: float, *, energy: float = 0.65,
              gestures: bool = True, side: str = "r", seed: int = 0) -> dict:
    """The whole performance, in video time. `words`/`loud` are in the voice file's own time; each
    span says where its part of the voice is heard."""
    rng = np.random.default_rng(seed)
    vwords: list[dict] = []
    vloud = np.zeros(max(1, int(math.ceil(total * fps))), np.float32)
    for sp in spans:
        shift = sp["start"] - sp["src"]
        src_end = sp["src"] + (sp["end"] - sp["start"])
        for w in words:
            if sp["src"] - 0.05 <= float(w["start"]) < src_end:
                vwords.append({"word": w.get("word", ""), "start": float(w["start"]) + shift, "end": min(float(w["end"]) + shift, sp["end"] + 0.2)})
        a, b = int(sp["src"] * fps), int(src_end * fps)
        seg = loud[a:b]
        k = int(sp["start"] * fps)
        n = max(0, min(len(seg), len(vloud) - k))
        if n and k >= 0:
            vloud[k : k + n] = np.maximum(vloud[k : k + n], seg[:n])
    vwords.sort(key=lambda w: w["start"])
    stress = stressed(vwords, vloud, fps)
    feelings = []
    for sp in spans:
        face, body = feeling(sp.get("emotion"), sp.get("intensity", 0.6))
        feelings.append({"start": round(sp["start"], 3), "end": round(sp["end"], 3), "emotion": sp.get("emotion") or "Neutral", "face": face, "body": body,
                         # every line the head settles somewhere new
                         "head": {"yaw": round(float(rng.normal(0, 4.5)), 1), "roll": round(float(rng.normal(0, 2.5)), 1)}})
    gs = plan_gestures(spans, vwords, stress, energy, side, rng) if gestures else []
    return {"version": 1, "fps": fps, "duration": round(total, 3), "energy": energy, "seed": seed, "mouth": mouth_keys(vwords),
            "loud": [round(float(x), 3) for x in vloud], "feelings": feelings, "gestures": gs,
            "nods": [{"t": s["t"], "power": s["power"]} for s in stress], "blinks": plan_blinks(total, spans, gs, rng)}


# ---------------------------------------------------------------------------------------------
# Animation: the plan at time t -> bone rotations and face shapes
# ---------------------------------------------------------------------------------------------


def ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * x * (x * (6 * x - 15) + 10)


def wander(t: float, k: int) -> float:
    """Smooth, never-repeating drift in -1..1 (a few slow sines that do not line up)."""
    return (math.sin(t * 0.61 + k * 1.7) + 0.6 * math.sin(t * 1.13 + k * 2.9) + 0.35 * math.sin(t * 2.07 + k * 4.3)) / 1.95


def pulse(dt: float, attack: float, decay: float) -> float:
    """0 → 1 quickly, then fading: a nod, a beat, a brow lift."""
    if dt < 0:
        return 0.0
    return ease(dt / attack) if dt < attack else math.exp(-(dt - attack) / decay)


class Spring:
    """A value that follows its target like a limb: it lags, overshoots a little and settles."""

    def __init__(self, n: int, speed: float, damping: float):
        self.x = None
        self.v = np.zeros(n)
        self.w, self.z = speed, damping

    def step(self, target: np.ndarray, dt: float) -> np.ndarray:
        if self.x is None:
            self.x = target.astype(np.float64).copy()
            return self.x
        for _ in range(2):  # two half steps keep it stable at 24–30 fps
            h = dt / 2
            self.v += (self.w * self.w * (target - self.x) - 2 * self.z * self.w * self.v) * h
            self.x = self.x + self.v * h
        return self.x


class Animator:
    """Steps through a plan frame by frame (in order: springs and hair carry over)."""

    def __init__(self, plan: dict, model: vrm.Model):
        self.p = plan
        self.fps = float(plan["fps"])
        self.total = float(plan["duration"])
        n = max(1, int(math.ceil(self.total * self.fps)))
        self.shapes_have = set(model.shapes)
        self.bones = set(model.bones)
        loud = np.asarray(plan.get("loud") or [0.0], np.float32)
        self.loud = np.pad(loud, (0, max(0, n - len(loud))))[:n]
        # the mouth opens fast and closes a little slower
        gate = np.zeros(n, np.float32)
        g = 0.0
        for i in range(n):
            x = float(self.loud[i]) ** 0.6
            g = x if x > g else g + (x - g) * 0.55
            gate[i] = g
        self.gate = gate
        # each sound spreads into its neighbours (the mouth is already forming the next one)
        self.mouth = {v: np.zeros(n, np.float32) for v in VOWELS}
        keys = plan.get("mouth") or []
        for k, (t, shape, amount) in enumerate(keys):
            nxt = keys[k + 1][0] if k + 1 < len(keys) else t + 0.12
            prv = keys[k - 1][0] if k else t - 0.12
            width = float(np.clip(max(nxt - t, t - prv), 0.06, 0.16))
            if shape not in self.mouth:
                continue
            a, b = max(0, int((t - width) * self.fps)), min(n, int((t + width) * self.fps) + 2)
            for i in range(a, b):
                d = abs(i / self.fps - t) / width
                if d < 1:
                    self.mouth[shape][i] = max(self.mouth[shape][i], float(amount) * (0.5 + 0.5 * math.cos(math.pi * d)))
        closed = np.ones(n, np.float32)  # lips meet on m, b, p even when the voice is loud
        for t, shape, _ in keys:
            if shape == "closed":
                for i in range(max(0, int((t - 0.05) * self.fps)), min(n, int((t + 0.05) * self.fps) + 1)):
                    closed[i] = min(closed[i], abs(i / self.fps - t) / 0.05)
        self.closed = closed
        self.arms = Spring(2 * len(ARM), 15.0, 0.72)
        self.head = Spring(6, 11.0, 0.8)
        self.talk = 0.0

    # ---- pieces ---------------------------------------------------------------------------

    def _feeling(self, t: float) -> tuple[dict, dict, dict, float]:
        """Face mix, body and head offsets at t (lines blend into each other), and how far into
        its line t is (0–1)."""
        fs = self.p.get("feelings") or []
        face: dict[str, float] = {}
        body = {"pitch": 0.0, "lean": 0.0, "tilt": 0.0, "amp": 0.8}
        head = {"yaw": 0.0, "roll": 0.0}
        weight_sum, into = 0.0, 0.0
        for f in fs:
            fade = 0.45
            w = ease((t - (f["start"] - fade)) / fade) * (1.0 - 0.6 * ease((t - f["end"]) / 1.2))
            if t > f["end"] + 1.2:
                w = 0.4 * (1.0 - ease((t - f["end"] - 1.2) / 2.0))  # the feeling lingers, then lets go
            if w <= 1e-3:
                continue
            # it swells as the line starts and eases to 70%: a held face looks like a mask
            x = (t - f["start"]) / max(0.2, f["end"] - f["start"])
            swell = 0.7 + 0.3 * math.exp(-max(0.0, x) * 2.2)
            for k, v in f["face"].items():
                face[k] = face.get(k, 0.0) + v * w * swell
            for k in ("pitch", "lean", "tilt"):
                body[k] += float(f["body"].get(k, 0.0)) * w
            body["amp"] += (float(f["body"].get("amp", 0.8)) - 0.8) * w
            for k in head:
                head[k] += float((f.get("head") or {}).get(k, 0.0)) * w
            if w > weight_sum:
                weight_sum, into = w, float(np.clip(x, 0.0, 1.0))
        return {k: min(1.0, v) for k, v in face.items()}, body, head, into

    def _arm_targets(self, t: float, amp: float) -> tuple[np.ndarray, dict, float, float]:
        """Both arms' numbers at t, head extras, shoulder lift and bow from the gestures in play."""
        base = {k: REST[k] + (READY[k] - REST[k]) * self.talk for k in ARM}
        arms = {"l": dict(base), "r": dict(base)}
        head = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        shoulders = bow = 0.0
        for g in self.p.get("gestures") or []:
            spec = GESTURES.get(g.get("kind"))
            if spec is None:
                continue
            t0, hold = float(g["t"]), float(g.get("hold", 0.8))
            prep, back = 0.34, 0.55
            if t < t0 - prep or t > t0 + hold + back:
                continue
            e = ease((t - (t0 - prep)) / prep) if t < t0 else (1.0 if t < t0 + hold else 1.0 - ease((t - t0 - hold) / back))
            size = float(g.get("size", 1.0)) * amp
            dt = t - t0
            for side in ("l", "r"):
                tgt = spec.get("both") if g.get("arm") == "both" else (spec.get("a") if g.get("arm") == side else spec.get("o"))
                if tgt is None:
                    continue
                cur = arms[side]
                for k in ARM:
                    goal = READY[k] + (tgt[k] - READY[k]) * (min(1.15, size) if k not in ("curl", "index") else 1.0)
                    if k == "index":
                        goal, start = (tgt["curl"] if tgt[k] < 0 else tgt[k]), (cur["curl"] if cur[k] < 0 else cur[k])
                        cur[k] = start + (goal - start) * e
                    else:
                        cur[k] = cur[k] + (goal - cur[k]) * e
                move = spec.get("move")
                if move and dt > 0 and e > 0.5:
                    if move == "wave":
                        cur["twist"] += 16 * math.sin(dt * 2 * math.pi * 2.3) * e
                    elif move == "bounce":
                        cur["elbow"] -= 9 * pulse(dt, 0.08, 0.2) + 5 * pulse(dt - 0.45, 0.08, 0.2)
                    elif move == "pump":
                        cur["elbow"] += 14 * pulse(dt, 0.07, 0.18)
                        cur["fwd"] += 8 * pulse(dt, 0.07, 0.18)
                    elif move == "shake":
                        cur["twist"] += 5 * math.sin(dt * 2 * math.pi * 2.6) * math.exp(-dt * 1.6)
            for k in head:
                head[k] += float((spec.get("head") or {}).get(k, 0.0)) * e * (1 if g.get("arm") != "l" or k == "pitch" else -1)
            if spec.get("move") == "shake" and dt > 0:
                head["yaw"] += 7 * math.sin(dt * 2 * math.pi * 2.2) * math.exp(-dt * 1.8) * e
            shoulders += float(spec.get("shoulders", 0.0)) * e * (1.0 if dt < 0.5 else max(0.35, 1.0 - (dt - 0.5)))
            bow += float(spec.get("bow", 0.0)) * e
        vec = np.array([arms[s][k] for s in ("l", "r") for k in ARM], np.float64)
        return vec, head, shoulders, bow

    # ---- one frame -------------------------------------------------------------------------

    def frame(self, t: float) -> tuple[dict, dict, dict]:
        """(pose, face shapes, lift) at time t."""
        dt = 1.0 / self.fps
        i = int(min(len(self.loud) - 1, max(0, round(t * self.fps))))
        energy = float(self.p.get("energy", 0.65))
        face, body, line_head, _ = self._feeling(t)
        # is it in the middle of talking (arms come up) or in a pause (they sink)?
        speaking = any(f["start"] - 0.25 <= t <= f["end"] + 0.5 for f in self.p.get("feelings") or [])
        self.talk += ((1.0 if speaking else 0.0) - self.talk) * min(1.0, dt * (3.0 if speaking else 1.2))
        amp = body["amp"] * (0.75 + 0.4 * energy)
        vec, g_head, shoulders, bow = self._arm_targets(t, amp)
        # beats and nods on the stressed words
        nod = brow = beat = 0.0
        for k, s in enumerate(self.p.get("nods") or []):
            d = t - float(s["t"])
            if -0.1 < d < 1.2:
                p = pulse(d + 0.06, 0.12, 0.26) * float(s.get("power", 0.6))
                nod += p * (1.0 if k % 3 else -0.5)  # mostly down, sometimes a lift of the chin
                brow += p if k % 2 == 0 else 0.0
                beat += pulse(d + 0.03, 0.09, 0.2) * float(s.get("power", 0.6))
        n = len(ARM)
        for off in (0, n):
            vec[off + ARM.index("elbow")] -= 7.0 * beat * amp * self.talk
            vec[off + ARM.index("wrist")] -= 5.0 * beat * amp * self.talk
        # never perfectly still, and never the two arms alike
        vec[ARM.index("fwd")] += 1.6 * wander(t * 0.7, 11)
        vec[n + ARM.index("fwd")] += 1.6 * wander(t * 0.7, 12)
        vec[ARM.index("elbow")] += 2.2 * wander(t * 0.8, 13)
        vec[n + ARM.index("elbow")] += 2.2 * wander(t * 0.8, 14)
        arms = self.arms.step(vec, dt)
        live = 0.55 + 0.6 * energy
        head_t = np.array([
            body["pitch"] + g_head["pitch"] + 4.2 * nod * live + 1.8 * wander(t, 1) * live + bow * 0.6,
            line_head["yaw"] + g_head["yaw"] + 4.5 * wander(t * 0.8, 2) * live,
            line_head["roll"] + body["tilt"] * (1 if line_head["roll"] >= 0 else -1) + g_head["roll"] + 1.8 * wander(t * 0.9, 3) * live,
            body["lean"] + bow + 1.0 * math.sin(t * 2 * math.pi * 0.24),  # the chest: lean + breathing
            1.6 * wander(t * 0.5, 4) * live,  # the spine turning
            1.4 * wander(t * 0.33, 5),  # weight shifting from side to side
        ])
        hp, hy, hr, lean, turn, sway = self.head.step(head_t, dt)
        breath = math.sin(t * 2 * math.pi * 0.24)
        pose = {
            "head": vrm.euler(hp * 0.65, hy * 0.6, hr * 0.7), "neck": vrm.euler(hp * 0.35, hy * 0.4, hr * 0.3),
            "chest": vrm.euler(lean * 0.6 - 0.7 * breath, turn * 0.5, sway * 0.5), "spine": vrm.euler(lean * 0.4, turn * 0.5, sway * 0.6),
            "upperChest": vrm.euler(-0.5 * breath, 0, 0), "hips": vrm.euler(0, turn * -0.3, -sway * 0.5),
        }
        for s, side in enumerate(("l", "r")):
            pose.update(arm_pose(side, dict(zip(ARM, arms[s * n : (s + 1) * n])), shoulders + 0.9 * breath))
        lift = {"hips": [0.004 * sway, 0.002 * breath, 0.0]}
        # ---- the face
        shapes: dict[str, float] = {}
        gate = float(self.gate[i]) * float(self.closed[i])
        smile = face.get("happy", 0.0)
        total = sum(float(self.mouth[v][i]) for v in VOWELS)
        scale = (0.72 - 0.2 * smile) * gate / max(1.0, total)
        for v in VOWELS:
            shapes[v] = float(self.mouth[v][i]) * scale
        if total < 0.05 and gate > 0.15:  # the voice is heard but no sound was placed here: follow the loudness
            shapes["aa"] = 0.5 * gate
        talking = min(1.0, gate * 1.5)
        for k, v in face.items():
            shapes[k] = v * (1.0 - 0.3 * talking if k in ("happy", "angry") else 1.0)
        shapes["surprised"] = min(1.0, shapes.get("surprised", 0.0) + 0.22 * min(1.0, brow))
        blink = 0.0
        for b in self.p.get("blinks") or []:
            d = t - float(b)
            if -0.02 <= d < 0.26:
                blink = max(blink, ease(d / 0.07) if d < 0.07 else (1.0 if d < 0.1 else 1.0 - ease((d - 0.1) / 0.15)))
        shapes["blink"] = max(blink, 0.25 * face.get("sad", 0.0), 0.2 * face.get("relaxed", 0.0)) * (1.0 - 0.7 * shapes.get("surprised", 0.0) if blink < 0.5 else 1.0)
        # the eyes: on the viewer, with small glances away
        gx = 0.5 * wander(t * 0.9, 21) + (0.5 if math.sin(t * 0.37 + 2.0) > 0.93 else 0.0)
        gy = 0.4 * wander(t * 0.7, 22)
        if {"lookLeft", "lookRight"} <= self.shapes_have:
            shapes["lookLeft" if gx > 0 else "lookRight"] = min(0.35, abs(gx) * 0.35)
            shapes["lookUp" if gy > 0 else "lookDown"] = min(0.25, abs(gy) * 0.25)
        elif "leftEye" in self.bones:
            pose["leftEye"] = pose["rightEye"] = vrm.euler(-3.0 * gy, 5.0 * gx, 0)
        return pose, shapes, lift


def arm_pose(side: str, a: dict, shoulder_lift: float = 0.0) -> dict:
    """Bone rotations of one arm from its numbers (see ARM)."""
    s = 1.0 if side == "l" else -1.0
    name = "left" if side == "l" else "right"
    rot = vrm.rot
    out = {
        f"{name}Shoulder": rot("z", s * shoulder_lift),
        # turns about the left-right axis are the same for both arms; the others mirror
        f"{name}UpperArm": rot("x", -a["fwd"]) @ rot("z", -s * a["down"]) @ rot("x", a["twist"]),
        f"{name}LowerArm": rot("y", -s * a["elbow"]) @ rot("x", -a["turn"]),
        f"{name}Hand": rot("z", s * a["wrist"]),
    }
    curl = float(np.clip(a["curl"], 0.0, 1.0))
    index = curl if a["index"] < 0 else float(np.clip(a["index"], 0.0, 1.0))
    for finger, c in (("Index", index), ("Middle", curl), ("Ring", curl * 1.05), ("Little", curl * 1.1)):
        for joint, share in (("Proximal", 70), ("Intermediate", 85), ("Distal", 60)):
            out[f"{name}{finger}{joint}"] = rot("z", -s * min(1.0, c) * share)
    for joint, share in (("Metacarpal", 18), ("Proximal", 30), ("Distal", 35)):
        out[f"{name}Thumb{joint}"] = rot("y", s * (0.25 + 0.75 * curl) * share)
    return out


# ---------------------------------------------------------------------------------------------
# Pictures: backgrounds, the circle, frames
# ---------------------------------------------------------------------------------------------


def hex_rgb(s: str | None, default: tuple[int, int, int]) -> tuple[int, int, int]:
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(s or "").strip())
    return tuple(int(m.group(1)[i : i + 2], 16) for i in (0, 2, 4)) if m else default


def backdrop(kind: str, size: int, color: str | None = None):
    """What is behind the character in the studio look: `studio` (a lit wall), `room` (a soft,
    out-of-focus room), `color`, or the path of your own picture. A PIL RGB image, size x size."""
    from PIL import Image, ImageDraw, ImageFilter

    if kind not in BACKGROUNDS and Path(str(kind)).expanduser().is_file():
        try:
            im = Image.open(Path(kind).expanduser()).convert("RGB")
        except Exception as e:  # noqa: BLE001
            raise AutocutError(f"could not open the background picture: {e}") from None
        k = max(size / im.width, size / im.height)
        im = im.resize((max(size, round(im.width * k)), max(size, round(im.height * k))))
        x, y = (im.width - size) // 2, (im.height - size) // 2
        return im.crop((x, y, x + size, y + size))
    base = np.array(hex_rgb(color, (46, 58, 92) if kind != "room" else (196, 170, 140)), np.float32)
    if kind == "color":
        return Image.new("RGB", (size, size), tuple(int(v) for v in base))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    if kind == "room":
        wall = base[None, None, :] * (1.05 - 0.35 * yy[..., None])
        im = Image.fromarray(np.clip(wall, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        dark = tuple(int(v * 0.45) for v in base)
        d.rectangle([size * 0.04, size * 0.2, size * 0.34, size * 0.24], fill=dark)  # a shelf with things on it
        for k, c in enumerate(((170, 70, 60), (60, 110, 150), (220, 190, 90), (70, 130, 90))):
            d.rectangle([size * (0.06 + 0.065 * k), size * 0.1, size * (0.11 + 0.065 * k), size * 0.2], fill=c)
        d.rectangle([size * 0.68, size * 0.08, size * 0.97, size * 0.46], fill=(235, 240, 250))  # a window
        d.ellipse([size * 0.74, size * 0.5, size * 0.98, size * 0.86], fill=(60, 120, 70))  # a plant
        d.rectangle([0, size * 0.9, size, size], fill=tuple(int(v * 0.6) for v in base))
        return im.filter(ImageFilter.GaussianBlur(size / 38))
    # studio: a pool of light on the wall behind the head, darker towards the edges
    r = np.sqrt((xx - 0.5) ** 2 + (yy - 0.36) ** 2)
    glow = np.clip(1.35 - 1.6 * r, 0.32, 1.45)[..., None]
    return Image.fromarray(np.clip(base[None, None, :] * glow, 0, 255).astype(np.uint8))


class Stage:
    """Turns each drawn frame into what goes over the video (`over`: premultiplied RGBA) and the
    character's own clip (`clip`: RGB on its background or on green)."""

    def __init__(self, look: str, size: int, background: str = "studio", bg_color: str | None = None, border_color: str | None = None,
                 border: float = 0.035):
        from PIL import Image, ImageDraw, ImageFilter

        self.look, self.size = look, size
        self.green = np.array(GREEN, np.uint16)[None, None, :]
        if look == "studio":
            self.bg = np.asarray(backdrop(background, size, bg_color), np.uint16)
            self.pad = int(size * 0.07)
            c = self.canvas = size + 2 * self.pad
            big = 3
            m = Image.new("L", (c * big, c * big), 0)
            ImageDraw.Draw(m).ellipse([self.pad * big, self.pad * big, (self.pad + size) * big - 1, (self.pad + size) * big - 1], fill=255)
            disc = np.asarray(m.resize((c, c), Image.LANCZOS), np.float32) / 255.0
            bw = max(0.0, min(0.12, border)) * size
            inner = Image.new("L", (c * big, c * big), 0)
            ImageDraw.Draw(inner).ellipse([(self.pad + bw) * big, (self.pad + bw) * big, (self.pad + size - bw) * big - 1, (self.pad + size - bw) * big - 1], fill=255)
            hole = np.asarray(inner.resize((c, c), Image.LANCZOS), np.float32) / 255.0
            ring = np.clip(disc - hole, 0.0, 1.0)
            shadow = np.asarray(m.resize((c, c), Image.LANCZOS).filter(ImageFilter.GaussianBlur(self.pad * 0.45)), np.float32) / 255.0
            shadow = np.roll(shadow, int(self.pad * 0.35), axis=0) * 0.5
            alpha = disc + shadow * (1.0 - disc)
            self.hole = hole[..., None]
            self.ring = (ring[..., None] * np.array(hex_rgb(border_color, (255, 255, 255)), np.float32)[None, None, :])
            self.alpha = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        else:
            self.pad, self.canvas = 0, size
            fade = np.ones(size, np.float32)
            n = max(1, int(size * 0.05))
            fade[-n:] = np.linspace(1.0, 0.0, n)  # the cut at the stomach fades out instead of ending in a line
            self.fade = fade[:, None, None]

    def frame(self, rgba: bytes) -> tuple[bytes, bytes]:
        s = self.size
        f = np.frombuffer(rgba, np.uint8).reshape(s, s, 4)[::-1]  # top row first
        rgb, a = f[..., :3].astype(np.uint16), f[..., 3:4].astype(np.uint16)
        if self.look == "studio":
            flat = rgb + self.bg * (255 - a) // 255
            c = self.canvas
            out = np.zeros((c, c, 4), np.uint8)
            inside = np.zeros((c, c, 3), np.float32)
            inside[self.pad : self.pad + s, self.pad : self.pad + s] = flat
            out[..., :3] = np.clip(inside * self.hole + self.ring, 0, 255).astype(np.uint8)
            out[..., 3] = self.alpha
            return out.tobytes(), flat.astype(np.uint8).tobytes()
        over = (f.astype(np.float32) * self.fade).astype(np.uint8)
        return over.tobytes(), (rgb + self.green * (255 - a) // 255).astype(np.uint8).tobytes()


def place(W: int, H: int, canvas: int, pad: int, position: str, x: float | None, y: float | None, flush_bottom: bool) -> tuple[int, int]:
    """Top-left corner of the overlay. x, y (0–1): where its centre goes, instead of a corner."""
    margin = int(0.03 * min(W, H))
    if x is not None and y is not None:
        return int(round(float(np.clip(x, 0, 1)) * W - canvas / 2)), int(round(float(np.clip(y, 0, 1)) * H - canvas / 2))
    px, py = POSITIONS.get(position, POSITIONS["bottom-right"])
    left = int(round((margin - pad) + px * (W - canvas + 2 * pad - 2 * margin)))
    top = int(round((margin - pad) + py * (H - canvas + 2 * pad - 2 * margin)))
    if flush_bottom and py == 1.0:
        top = H - canvas  # a cut-out stands on the bottom edge
    return left, top


# ---------------------------------------------------------------------------------------------
# Making it
# ---------------------------------------------------------------------------------------------

DEFAULTS = {"character": None, "look": "studio", "position": "bottom-right", "size": 0.3, "x": None, "y": None, "background": "studio",
            "bg_color": None, "border_color": None, "energy": "normal", "gestures": True, "green_overlay": True, "tints": None, "zoom": None,
            "seed": 0, "detail": 720}


def words_of(voice: str, spans: list[dict], language: str | None, listen: bool = True) -> list[dict]:
    """When each word is said (in the voice file's time): heard by Whisper (kept in the cache), or,
    without it, the line's words spread over the line."""
    if listen:
        try:
            got = ac.transcribe(voice, language=language).get("words") or []
            if got:
                return got
        except Exception as e:  # noqa: BLE001 (the lips then follow the script and the loudness)
            log(f"could not listen for word timings ({e}); spreading each line's words over it")
    out = []
    for sp in spans:
        ws = str(sp.get("text", "")).split()
        if not ws:
            continue
        weights = [len(w) + 1.5 for w in ws]
        length, acc = sp["end"] - sp["start"], 0.0
        for w, k in zip(ws, weights):
            a = sp["src"] + length * acc / sum(weights)
            acc += k
            out.append({"word": w, "start": round(a, 3), "end": round(sp["src"] + length * acc / sum(weights) - 0.02, 3)})
    return out


def render(plan: dict, vrm_path: str, opts: dict, *, video: str | None, voice: str | None, out: Path, clip_out: Path, W: int, H: int) -> dict:
    """Draw every frame once and send it two ways: over the video (when there is one and the look
    asks for it) and into the character's own clip."""
    fps, total = float(plan["fps"]), float(plan["duration"])
    frames = max(1, int(round(total * fps)))
    look = opts["look"]
    R = int(np.clip(int(opts.get("detail") or 720) // 2 * 2, 240, 1440))
    model = vrm.Model(vrm_path)
    tints = {k: [float(x) for x in v][:3] for k, v in (opts.get("tints") or {}).items() if isinstance(v, (list, tuple)) and len(v) >= 3}
    rend = vrm.Renderer(model, R, tints=tints)
    rend.set_camera(zoom=float(opts.get("zoom") or 1.0))
    rig = vrm.Rig(model)
    anim = Animator(plan, model)
    stage = Stage(look, R, opts.get("background") or "studio", opts.get("bg_color"), opts.get("border_color"))
    procs: list[subprocess.Popen] = []
    over = clip = None
    overlay = video is not None and (look != "green" or opts.get("green_overlay", True))
    try:
        if overlay:
            scale = float(np.clip(opts.get("size") or 0.3, 0.08, 0.9)) * min(W, H) / R
            canvas = max(2, int(round(stage.canvas * scale)) // 2 * 2)
            x, y = place(W, H, canvas, int(round(stage.pad * scale)), opts.get("position") or "bottom-right", opts.get("x"), opts.get("y"), look != "studio")
            cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", video, "-f", "rawvideo", "-pix_fmt", "rgba",
                   "-s", f"{stage.canvas}x{stage.canvas}", "-r", ac.rate_arg(fps), "-i", "-", "-filter_complex",
                   f"[1:v]scale={canvas}:{canvas}:flags=bicubic,format=rgba[c];[0:v][c]overlay=x={x}:y={y}:alpha=premultiplied:eof_action=pass[v]",
                   "-map", "[v]", "-map", "0:a?", "-c:a", "copy", "-c:v", "h264_videotoolbox", "-b:v", ac.bitrate_for(W, H), "-profile:v", "high",
                   "-movflags", "+faststart", str(out.with_suffix(".part.mp4"))]
            over = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            procs.append(over)
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{R}x{R}", "-r", ac.rate_arg(fps), "-i", "-"]
        if voice and video is None:
            cmd += ["-i", voice, "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-c:v", "h264_videotoolbox", "-b:v", ac.bitrate_for(R, R), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(clip_out.with_suffix(".part.mp4"))]
        clip = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append(clip)
        t0 = time.time()
        dt = 1.0 / fps
        for k in range(-int(fps * 0.6), frames):  # a moment of settling first, so the hair does not start mid-swing
            pose, shapes, lift = anim.frame(max(0.0, k * dt))
            world = rig.step(pose, lift, dt)
            if k < 0:
                continue
            a, b = stage.frame(rend.render(shapes=shapes, world=world))
            try:
                if over is not None and over.stdin:
                    over.stdin.write(a)
                if clip.stdin:
                    clip.stdin.write(b)
            except BrokenPipeError:
                break
            if k % max(1, int(fps * 5)) == 0:
                ac.stage_progress(k / frames, f"character: {k / fps:.0f} of {total:.0f} s")
        for p in procs:
            if p.stdin:
                p.stdin.close()
        for p in procs:
            err = (p.stderr.read().decode(errors="replace") if p.stderr else "").strip()
            if p.wait() != 0:
                raise AutocutError(f"ffmpeg failed while adding the character: {err[-400:]}")
        took = time.time() - t0
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        rend.close()
    clip_out.with_suffix(".part.mp4").replace(clip_out)
    if overlay:
        out.with_suffix(".part.mp4").replace(out)
    log(f"character drawn: {frames} frames in {took:.0f}s ({total / max(took, 1e-3):.1f}x real time)")
    return {"video": str(out) if overlay else None, "clip": str(clip_out), "seconds": round(took, 1)}


def add_to_video(video: str | None, voice: str, narration: dict | str, out: str, opts: dict | None = None, *, timeline: dict | str | None = None,
                 language: str | None = None, plan_path: str | None = None, listen: bool = True) -> dict:
    """The one call other tools use: voice + its lines (+ the finished video) -> the video with
    the character, the character's own clip, and the plan.
    `timeline`: the edit's plan when it moved the lines (narrate's .narrate.json)."""
    ac.require_tools()
    o = {**DEFAULTS, **{k: v for k, v in (opts or {}).items() if v is not None}}
    if o["look"] not in LOOKS:
        raise AutocutError(f"look must be one of {', '.join(LOOKS)}")
    if not Path(voice).is_file():
        raise AutocutError(f"voice file not found: {voice}")
    outp = Path(out).expanduser().resolve()
    if video and outp == Path(video).resolve():
        raise AutocutError("the output must not be the video itself")
    vrm_path, prof = resolve_character(o["character"])
    o["tints"] = {**(prof.get("tints") or {}), **(o["tints"] or {})}
    o["zoom"] = o["zoom"] or prof.get("zoom") or 1.0
    load = lambda x: json.loads(Path(x).read_text(encoding="utf-8")) if isinstance(x, str) else x  # noqa: E731
    try:
        narration, timeline = load(narration), load(timeline) if timeline else None
    except (OSError, ValueError) as e:
        raise AutocutError(f"could not read the narration: {e}") from None
    if video:
        info = ac.probe(video)
        if not info.has_video:
            raise AutocutError("the video has no picture")
        W, H = info.display_size
        fps, total = float(np.clip(info.fps or 30.0, 12.0, 60.0)), float(info.duration)
    else:
        W = H = int(o["detail"])
        fps, total = 30.0, float(ac.probe(voice).duration) + 0.5
    if not 0.5 <= total <= MAX_SECONDS:
        raise AutocutError("the video must be between half a second and three hours long")
    outp.parent.mkdir(parents=True, exist_ok=True)
    pj = outp.with_suffix(".character.json")
    if plan_path:
        try:
            plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
            plan["fps"], plan["duration"] = float(plan.get("fps") or fps), float(plan.get("duration") or total)
        except (OSError, ValueError, TypeError) as e:
            raise AutocutError(f"could not read the character plan: {e}") from None
        log(f"using your plan: {plan_path}")
    else:
        spans = spans_from(narration, timeline)
        loud = loudness(voice, fps)
        words = words_of(voice, spans, language, listen)
        px = POSITIONS.get(o["position"], (1.0, 1.0))[0] if o["x"] is None else float(o["x"])
        seed = int(hashlib.sha1((narration.get("script") or voice).encode()).hexdigest()[:6], 16) + int(o["seed"] or 0)
        plan = make_plan(spans, words, loud, fps, total, energy=ENERGY.get(str(o["energy"]), 0.65), gestures=bool(o["gestures"]),
                         side="r" if px >= 0.5 else "l", seed=seed)
        kinds: dict[str, int] = {}
        for g in plan["gestures"]:
            kinds[g["kind"]] = kinds.get(g["kind"], 0) + 1
        log(f"performance: {len(plan['feelings'])} lines, {len(plan['mouth'])} mouth shapes, {len(plan['gestures'])} gestures "
            f"({', '.join(f'{k} x{v}' for k, v in sorted(kinds.items())) or 'none'}), {len(plan['nods'])} nods, {len(plan['blinks'])} blinks")
    if not plan_path or Path(plan_path).resolve() != pj:
        pj.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    clip_out = outp.with_name(outp.stem + (".character.green.mp4" if o["look"] != "studio" else ".character.mp4"))
    res = render(plan, vrm_path, o, video=video, voice=voice, out=outp, clip_out=clip_out, W=W, H=H)
    credit = None
    if prof.get("credit") == "required" or prof.get("author"):
        credit = f"Character: {prof.get('title') or Path(vrm_path).stem}" + (f" by {prof['author']}" if prof.get("author") else "") + (f" ({prof['licence']})" if prof.get("licence") else "")
    return {"outputs": [x for x in (res["video"], res["clip"], str(pj)) if x], "video": res["video"], "clip": res["clip"], "plan": str(pj),
            "gestures": len(plan.get("gestures") or []), "credit": credit}


def add_character_args(p) -> None:
    """The character options, for any tool that makes a narrated video."""
    g = p.add_argument_group("animated character (optional)")
    g.add_argument("--character", help="add an animated presenter: a character's name, or a .vrm file")
    g.add_argument("--character-look", default="studio", choices=list(LOOKS), help="studio: in a circle with a studio behind; cutout: no background; green: a separate green-screen clip")
    g.add_argument("--character-position", default="bottom-right", choices=list(POSITIONS))
    g.add_argument("--character-x", type=float, help="centre, 0 (left) – 1 (right); with --character-y, instead of a corner")
    g.add_argument("--character-y", type=float, help="centre, 0 (top) – 1 (bottom)")
    g.add_argument("--character-size", type=float, default=0.3, help="width as a share of the video's shorter side (0.08–0.9)")
    g.add_argument("--character-background", default="studio", help="studio look: studio, room, color, or a picture file")
    g.add_argument("--character-bg-color", help="background colour, e.g. #2e3a5c")
    g.add_argument("--character-border-color", help="circle border colour, e.g. #ffffff")
    g.add_argument("--character-energy", default="normal", choices=list(ENERGY), help="how much it moves and gestures")
    g.add_argument("--character-no-gestures", action="store_true")
    g.add_argument("--character-no-cutout", action="store_true", help="green look: leave the video itself without the character")
    g.add_argument("--character-seed", type=int, default=0, help="another number = another performance of the same lines")


def options(a, always: bool = False) -> dict | None:
    """Character options from parsed arguments, or None when no character was asked for
    (`always`: the first character then)."""
    if not getattr(a, "character", None) and not always:
        return None
    return {"character": a.character, "look": a.character_look, "position": a.character_position, "x": a.character_x, "y": a.character_y,
            "size": a.character_size, "background": a.character_background, "bg_color": a.character_bg_color, "border_color": a.character_border_color,
            "energy": a.character_energy, "gestures": not a.character_no_gestures, "green_overlay": not a.character_no_cutout, "seed": a.character_seed}


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--add", metavar="VRM", help="add a character from a .vrm file (your own from VRoid Studio, or one you may use)")
    p.add_argument("--name", help="with --add: the character's name")
    p.add_argument("--list", action="store_true", help="list your characters")
    p.add_argument("--voice", help="the spoken audio")
    p.add_argument("--narration", help="the voice's lines with their emotions (voice.speak's .narration.json)")
    p.add_argument("--timeline", help="the edit's plan, when it moved the lines (.narrate.json)")
    p.add_argument("--video", help="the video to put the character on (without it: only the character's clip)")
    p.add_argument("--out", help="the video with the character")
    p.add_argument("--plan", help="a character plan to perform as it is (after editing it)")
    p.add_argument("--language")
    add_character_args(p)


def run_args(a) -> dict:
    if a.list:
        cs = list_characters()
        for c in cs:
            log(f"{c['name']}: {c.get('title') or ''} {('by ' + c['author']) if c.get('author') else ''}")
        return {"characters": cs}
    if a.add:
        return {"character": add_character(a.add, a.name)}
    if not (a.voice and a.narration and a.out):
        raise AutocutError("give --voice, --narration and --out (and --video to put the character on a video)")
    return add_to_video(a.video, a.voice, a.narration, a.out, options(a, always=True), timeline=a.timeline, language=a.language, plan_path=a.plan)


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
