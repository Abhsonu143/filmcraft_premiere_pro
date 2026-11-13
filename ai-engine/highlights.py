#!/usr/bin/env python3
"""highlights: turn one long video into N short clips, each edited from several pieces.

    python highlights.py long.mp4 --count 5 --out-dir clips/
    python highlights.py long.mp4 --count 3 --min 10 --max 60 --target 48 --aspect vertical

Like Kapwing's Repurpose Studio / AI Clip Maker, but local and free:

  1. Transcript (Whisper on MLX, cached) -> sentences, with filler words ("um", "uh") and long
     pauses marked for removal.
  2. Topic sections: where the vocabulary changes (TextTiling-style lexical cohesion) and where
     the speaker pauses longest.
  3. Every sentence is scored: hooks (questions, numbers, "secret", "mistake", "never" …), how
     energetically it was said (loudness against the speaker's average), and a penalty for fillers.
     When a small local language model is installed (Qwen 2.5 1.5B on MLX), it also rates how well
     each section stands alone and writes a short title; otherwise titles come from the best line.
  4. Each clip is assembled, not cut: the best sentences of a section (or of neighbouring
     sections) are kept in order to reach the target length (10–60 s, mostly 45–50 s), weak lines
     are dropped, fillers and pauses inside sentences are cut out, and optionally the strongest
     line plays first as a teaser hook. One clip is typically 6–25 pieces.
  5. Picture: reframed to the chosen shape following the speaker's face (Apple Vision), with an
     alternating zoom on jump cuts so cuts inside a shot read as intentional. Big word-by-word
     captions (the spoken word highlighted) and the title at the top.

Videos with little speech (gameplay, travel) use loudness, motion and scene changes instead:
each clip is built from the most intense moments of a stretch of the video.

Clips never share material. Output: `<name> - 01 - <title>.mp4` … plus `<name>.clips.json` (every
clip's pieces, editable) and `.srt` captions per clip.
"""

from __future__ import annotations

import argparse
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

FILLERS = {"um", "uh", "erm", "er", "ah", "hmm", "mm", "uhm", "umm"}
HOOK_WORDS = {
    "secret", "mistake", "mistakes", "never", "always", "best", "worst", "biggest", "most", "number", "tip", "tips",
    "why", "how", "what", "truth", "nobody", "everyone", "stop", "free", "money", "dollars", "percent", "crazy",
    "amazing", "heart", "lost", "almost", "first", "only", "hidden", "changed", "important", "favorite", "favourite",
}
STOP = set("""a an the and or but so to of in on at for with by from as is are was were be been it its this that these those
i you he she we they me my your our their them us him her his do does did not no yes just very really can will would
could should have has had there here then than about into out up down over again also too more some any all one
what when where who which how why if because while okay oh well yeah like get got go going""".split())
ASPECTS = {"vertical": (1080, 1920), "square": (1080, 1080), "horizontal": (1920, 1080)}


# ---------------------------------------------------------------------------------------------
# Transcript -> sentences (pure logic, unit-tested)
# ---------------------------------------------------------------------------------------------


def norm_word(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower())


def sentences_from_words(words: list[dict], max_len: float = 14.0) -> list[dict]:
    """Group words into sentences on . ? ! (and on long pauses); very long sentences split at commas."""
    out: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        t = w["word"].strip()
        if not t:
            continue
        if cur and (w["start"] - cur[-1]["end"] > 1.2 or (w["end"] - cur[0]["start"] > max_len and cur[-1]["word"].strip().endswith(","))):
            out.append(cur)
            cur = []
        cur.append(w)
        if t[-1] in ".?!":
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    sents = []
    for ws in out:
        text = " ".join(x["word"].strip() for x in ws)
        sents.append({"start": ws[0]["start"], "end": ws[-1]["end"], "text": text, "words": ws})
    return sents


def keep_spans(sent: dict, pad: float = 0.06, max_gap: float = 0.3) -> list[tuple[float, float]]:
    """The parts of a sentence worth keeping: fillers removed, inner pauses tightened. The small
    padding around kept words never reaches into a removed filler."""
    spans: list[list[float]] = []
    removed: list[tuple[float, float]] = []
    for w in sent["words"]:
        a, b = float(w["start"]), float(w["end"])
        if norm_word(w["word"]) in FILLERS:
            removed.append((a, b))
            continue
        if spans and a - spans[-1][1] <= max_gap and not any(spans[-1][1] <= ra < a for ra, _ in removed):
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    out = []
    for a, b in spans:
        lo = max([rb for _, rb in removed if rb <= a] + [a - pad])
        hi = min([ra for ra, _ in removed if ra >= b] + [b + pad])
        out.append((max(0.0, lo), hi))
    return [(a, b) for a, b in out if b > a]


def span_len(spans) -> float:
    return sum(b - a for a, b in spans)


def tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-zA-Z']+", text.lower()):
        if w in STOP or len(w) < 3:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


def topic_boundaries(sents: list[dict], window: int = 3, min_size: int = 3) -> list[int]:
    """Sentence indices where a new topic starts (TextTiling-style, plus long pauses)."""
    n = len(sents)
    if n < 2 * min_size:
        return [0]
    toks = [tokens(s["text"]) for s in sents]
    vocab = {w: i for i, w in enumerate(sorted({w for t in toks for w in t}))}
    vecs = np.zeros((n, max(1, len(vocab))), np.float32)
    for i, t in enumerate(toks):
        for w in t:
            vecs[i, vocab[w]] += 1
    gaps = np.zeros(n)
    for i in range(1, n):
        a = vecs[max(0, i - window) : i].sum(axis=0)
        b = vecs[i : i + window].sum(axis=0)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        sim = float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
        pause = sents[i]["start"] - sents[i - 1]["end"]
        gaps[i] = (1 - sim) + min(1.0, max(0.0, pause - 0.6)) * 0.8
    cuts = [0]
    thr = float(gaps[1:].mean() + 0.4 * gaps[1:].std())
    for i in np.argsort(-gaps):
        i = int(i)
        if i == 0 or gaps[i] < thr:
            continue
        if all(abs(i - c) >= min_size for c in cuts) and n - i >= min_size:
            cuts.append(i)
    return sorted(cuts)


CHANNEL_TALK = re.compile(r"welcome back|subscribe|like and|comment below|in the comments|thanks for watching|see you (in|on) the next|link in the|sponsor|grab a coffee|that's it for today", re.I)


def sentence_score(s: dict, loud: float = 0.0) -> float:
    words = [norm_word(w["word"]) for w in s["words"]]
    real = [w for w in words if w and w not in FILLERS]
    if not real:
        return 0.0
    sc = 0.3
    t = s["text"]
    if t.rstrip().endswith("?"):
        sc += 0.35
    if re.search(r"\d", t) or any(w in ("one", "two", "three", "ten", "hundred", "thousand", "eleven", "twenty", "thirty", "fifty") for w in real):
        sc += 0.2
    sc += min(0.45, 0.15 * sum(1 for w in real if w in HOOK_WORDS))
    if "!" in t:
        sc += 0.1
    sc += 0.25 * float(np.clip(loud, -1, 1))
    dur = s["end"] - s["start"]
    if dur < 1.0:
        sc -= 0.15
    sc -= 0.3 * (len(words) - len(real)) / max(1, len(words))
    if CHANNEL_TALK.search(t):
        sc -= 0.6  # channel housekeeping makes a poor short
    return float(np.clip(sc, 0, 1.5))


# ---------------------------------------------------------------------------------------------
# Building clips from sentences (pure logic, unit-tested)
# ---------------------------------------------------------------------------------------------


def assemble(sents: list[dict], lo: int, hi: int, target: float, mn: float, mx: float, used: set[int], starts: frozenset = frozenset()) -> dict | None:
    """Best clip from sentences lo..hi-1: a run of sentences in order with weak ones dropped, total
    length near `target` within [mn, mx]. Returns {idx: [...], dur, value} or None."""
    best = None
    cand = [i for i in range(lo, hi) if i not in used and sents[i]["dur"] > 0.2]
    for ai, a in enumerate(cand):
        total, chosen = 0.0, []
        for b in cand[ai:]:
            if b != a and b - chosen[-1] > 3:
                break  # don't bridge a big hole (already-used material)
            chosen.append(b)
            total += sents[b]["dur"]
            if total < mn:
                continue
            if total > min(mx, target * 1.35) / 0.7:
                break  # never drop more than ~30% of a run to fit
            keep = list(chosen)
            dur = total
            # drop the weakest inner sentences until the length fits
            while dur > target + 3 and len(keep) > 2:
                inner = keep[1:-1] or keep[1:]
                worst = min(inner, key=lambda i: sents[i]["score"])
                keep.remove(worst)
                dur -= sents[worst]["dur"]
            if not (mn <= dur <= mx):
                continue
            if (len(chosen) - len(keep)) / max(1, len(chosen)) > 0.35:
                continue
            mean = float(np.mean([sents[i]["score"] for i in keep]))
            hook = sents[keep[0]]["score"]
            ending = 0.15 if sents[keep[-1]]["text"].rstrip().endswith((".", "!")) else -0.1
            dropped = (len(chosen) - len(keep)) / max(1, len(chosen))
            # length matters: most clips should sit near the target; being short costs more than long
            rel = (dur - target) / max(target, 1)
            closeness = 1 - min(1.0, abs(rel) * (2.5 if rel < 0 else 1.2))
            crossed = sum(1 for b in starts if keep[0] < b <= keep[-1])  # topic changes inside the clip
            value = mean + 0.4 * hook + ending + 1.2 * closeness - 0.35 * dropped - 0.25 * crossed
            if best is None or value > best["value"]:
                best = {"idx": keep, "dur": dur, "value": value}
    return best


def plan_clips(sents: list[dict], bounds: list[int], count: int, target: float, mn: float, mx: float, section_bonus: dict[int, float] | None = None) -> list[dict]:
    """Pick `count` non-overlapping clips; each grows over neighbouring sections when one is short."""
    n = len(sents)
    sections = [(b, bounds[k + 1] if k + 1 < len(bounds) else n) for k, b in enumerate(bounds)]
    used: set[int] = set()
    clips = []
    for _ in range(count):
        best = None
        for k, (lo, hi) in enumerate(sections):
            for span in (0, 1, 2):  # this section, or with its neighbours when it is short
                l2 = sections[max(0, k - span)][0] if span == 2 else lo
                h2 = sections[min(len(sections) - 1, k + span)][1] if span else hi
                c = assemble(sents, l2, h2, target, mn, mx, used, frozenset(bounds))
                if c is None:
                    continue
                c["value"] += (section_bonus or {}).get(k, 0.0) - 0.1 * span
                c["section"] = k
                if best is None or c["value"] > best["value"]:
                    best = c
                break_ok = c["dur"] >= target * 0.8
                if break_ok:
                    break
        if best is None:
            break
        used.update(range(min(best["idx"]), max(best["idx"]) + 1))
        clips.append(best)
    clips.sort(key=lambda c: min(c["idx"]))
    return clips


def pieces_for(sents: list[dict], idx: list[int], hook_first: bool) -> tuple[list[tuple[float, float]], int | None]:
    """Source time pieces in playing order (optionally the strongest line first as a teaser)."""
    teaser = None
    if hook_first and len(idx) >= 4:
        later = idx[len(idx) // 2 :]
        top = max(later, key=lambda i: sents[i]["score"])
        if sents[top]["score"] >= 0.75 and sents[top]["dur"] <= 6.0 and sents[top]["score"] > sents[idx[0]]["score"] + 0.15:
            teaser = top
    pieces: list[tuple[float, float]] = []
    order = ([teaser] if teaser is not None else []) + idx
    for i in order:
        pieces += keep_spans(sents[i])
    return merge_close(pieces), teaser


def merge_close(pieces: list[tuple[float, float]], gap: float = 0.12) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in pieces:
        if out and 0 <= a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


# ---------------------------------------------------------------------------------------------
# Audio loudness, faces, LLM
# ---------------------------------------------------------------------------------------------


def loudness_per_sentence(src: str, sents: list[dict]) -> list[float]:
    import montage as mt

    y = mt.decode_mono(src, 8000)
    if len(y) == 0:
        return [0.0] * len(sents)
    db = []
    for s in sents:
        a, b = int(s["start"] * 8000), int(s["end"] * 8000)
        seg = y[a:b]
        db.append(20 * math.log10(max(1e-5, float(np.sqrt(np.mean(seg**2))))) if len(seg) else -60.0)
    arr = np.asarray(db)
    mu_, sd = float(np.median(arr)), float(arr.std() + 1e-6)
    return [float((d - mu_) / sd) for d in db]


def llm_judge(texts: list[str], model: str | None) -> list[tuple[str | None, float | None]]:
    """(title, standalone score 0–1) per transcript excerpt from the local model, or Nones."""
    import llm

    prompts = [
        "You edit short social videos. Read this excerpt of a video transcript and reply in exactly two lines:\n"
        "TITLE: <a catchy title of at most 7 words, no hashtags, no quotes, no emojis>\n"
        "SCORE: <1-10: how well this works as a standalone short clip with a strong hook>\n\n"
        f"Transcript:\n{t[:1800]}"
        for t in texts
    ]
    outs = llm.ask(prompts, model, 60)
    if outs is None:
        return [(None, None)] * len(texts)
    res = []
    for o in outs:
        m = re.search(r"TITLE:\s*(.+)", o)
        sc = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", o)
        res.append((llm.clean_title(m.group(1)) if m else None, min(1.0, float(sc.group(1)) / 10) if sc else None))
    return res


def llm_titles(texts: list[str], model: str | None) -> list[str | None]:
    """A short title for each finished clip, from its own words (or, for gameplay, what is on screen)."""
    import llm

    prompts = [
        "Write a catchy title for this short video clip. Use at most 7 words. Reply with the title only: "
        f"no quotes, no hashtags, no emojis.\n\n{t[:1500]}"
        for t in texts
    ]
    outs = llm.ask(prompts, model, 24)
    return [llm.clean_title(o) for o in outs] if outs else [None] * len(texts)


GAME_TAGS = [
    "a sword fight", "a boss battle", "riding a horse", "a cinematic cutscene with characters talking", "exploring a village",
    "fire and burning buildings", "a snowy mountain landscape", "a battle at night", "shooting a bow and arrow",
    "a game menu or map screen", "a training duel", "a scenic landscape", "an explosion", "fighting many enemies",
    "a close-up of a character's face", "a car race", "a gunfight", "a sports match",
]


def _tag_worker(requests: list[tuple[str, float]], prompts: list[str]) -> list[list[float]]:
    import numpy as np
    import open_clip
    import torch
    from PIL import Image

    import voiceover_cut as vc

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, pre = open_clip.create_model_and_transforms(vc.CLIP_MODEL[0], pretrained=vc.CLIP_MODEL[1], device=device)
    model.eval()
    tok = open_clip.get_tokenizer(vc.CLIP_MODEL[0])
    imgs = []
    for path, t in requests:
        proc = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1", "-vf", "scale=224:224:force_original_aspect_ratio=increase,crop=224:224", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
        buf = proc.stdout[: 224 * 224 * 3]
        imgs.append(np.frombuffer(buf, dtype=np.uint8).reshape(224, 224, 3) if len(buf) == 224 * 224 * 3 else np.zeros((224, 224, 3), np.uint8))
    with torch.no_grad():
        im = model.encode_image(torch.stack([pre(Image.fromarray(i)) for i in imgs]).to(device))
        tx = model.encode_text(tok(prompts).to(device))
        im = im / im.norm(dim=-1, keepdim=True)
        tx = tx / tx.norm(dim=-1, keepdim=True)
        return (100 * im @ tx.T).softmax(dim=-1).float().cpu().numpy().tolist()


def visual_tags(src: str, clips: list[dict], per_clip: int = 4) -> list[list[str]]:
    """The two most likely things on screen in each clip (CLIP, zero-shot), or [] without CLIP."""
    try:
        import open_clip  # noqa: F401
    except ImportError:
        return [[] for _ in clips]
    reqs, owner = [], []
    for k, c in enumerate(clips):
        ps = c["pieces"]
        for j in range(per_clip):
            a, b = ps[min(len(ps) - 1, j * len(ps) // per_clip)]
            reqs.append((src, (a + b) / 2))
            owner.append(k)
    try:
        probs = ac.run_isolated(_tag_worker, reqs, GAME_TAGS)
    except Exception as e:
        log(f"no visual tags ({type(e).__name__})")
        return [[] for _ in clips]
    out = []
    for k in range(len(clips)):
        rows = np.asarray([p for p, o in zip(probs, owner) if o == k])
        mean = rows.mean(axis=0) if len(rows) else np.zeros(len(GAME_TAGS))
        out.append([GAME_TAGS[i] for i in np.argsort(-mean)[:2] if mean[i] > 0.08])
    return out


def tidy_title(t: str | None) -> str | None:
    if not t:
        return None
    words = t.split()
    out = []
    for w in words:  # "I I", "Ah Ah Ah"
        if out and re.sub(r"\W", "", w).lower() == re.sub(r"\W", "", out[-1]).lower():
            continue
        out.append(w)
    while out and re.sub(r"\W", "", out[0]).lower() in ("ah", "oh", "uh", "um", "hey", "so", "and", "but"):
        out.pop(0)
    t = " ".join(out).strip(" -–—:,")
    return t[:1].upper() + t[1:] if t else None


def fallback_title(sents: list[dict], idx: list[int]) -> str:
    best = max(idx, key=lambda i: sents[i]["score"])
    words = [w for w in re.findall(r"[\w'%$]+", sents[best]["text"])][:7]
    t = " ".join(words)
    return t[:1].upper() + t[1:] if t else "Highlight"


# ---------------------------------------------------------------------------------------------
# Captions (see captions.py)
# ---------------------------------------------------------------------------------------------


def build_ass(words_out: list[dict], title: str | None, W: int, H: int, total: float, style: str = "pop", title_mode: str = "always") -> str:
    import captions

    return captions.build_ass(words_out, title, W, H, total, style, title_mode)


def remap_to_pieces(words: list[dict], pieces: list[tuple[float, float]]) -> list[dict]:
    import captions

    fps = 1000.0
    return captions.remap_to_shots([w for w in words if norm_word(w["word"]) not in FILLERS],
                                   [{"in": a, "frames": round((b - a) * fps), "speed": 1.0} for a, b in pieces], fps)


# ---------------------------------------------------------------------------------------------
# Shots, framing and rendering one clip
# ---------------------------------------------------------------------------------------------

FRAMINGS = ("auto", "smart", "center", "blur", "bars")


def talk_shots(pieces: list[tuple[float, float]], teaser: bool) -> list[dict]:
    """Shots for a talk clip: jump cuts alternate between a wide and a closer framing (so a cut
    inside one take reads as intentional); a teaser is followed by a flash cut."""
    shots: list[dict] = []
    z = 1.0
    prev_end = None
    teaser_done = not teaser
    for k, (a, b) in enumerate(pieces):
        sh: dict = {"in": a, "len": b - a, "speed": 1.0, "role": "talk"}
        if not teaser_done and k > 0 and a < pieces[k - 1][0]:
            teaser_done = True
            sh["transition_in"] = {"type": "fadewhite", "frames": 6}
            for prev in shots:
                prev["role"] = "hook"
            z = 1.0
        elif prev_end is not None and 0 <= a - prev_end < 4.0:
            z = 1.12 if z == 1.0 else 1.0
        if z != 1.0:
            sh["zoom"] = {"from": z, "to": z}
        elif b - a >= 4.0:
            sh["zoom"] = {"from": 1.0, "to": 1.05}  # a slow push-in keeps a long take alive
        shots.append(sh)
        prev_end = b
    if shots:
        shots[-1]["fade_out_s"] = 0.3
    return shots


def finalise_shots(shots: list[dict], fps: float) -> list[dict]:
    """Exact frame counts (the edit never drifts) and source lengths that match them."""
    out = []
    for s in shots:
        frames = max(1, int(round(s["len"] * fps)))
        sh = {k: v for k, v in s.items() if k != "len"}
        sh["frames"] = frames
        if s.get("fade_out_s"):
            sh["fade_out"] = max(1, int(round(float(s["fade_out_s"]) * fps)))
            sh.pop("fade_out_s", None)
        out.append(sh)
    return out


def frame_shots(src: str, shots: list[dict], src_w: int, src_h: int, W: int, H: int, fps: float, framing: str, talk: bool) -> str:
    """Adds framing to each shot (`track`/`active` for a true fill, `fit` for blur/bars). Returns
    the framing actually used."""
    import reframe

    same_shape = abs(W / H - src_w / max(src_h, 1)) < 0.03
    mode = "smart" if framing == "auto" else framing
    if mode in ("blur", "bars"):
        for s in shots:
            s["fit"] = mode
        return mode
    if mode == "center" or same_shape:
        for s in shots:
            s["focus"] = [0.5, 0.5]
        if same_shape:
            return "same shape"
        return mode
    ranges = [(float(s["in"]), float(s["in"]) + s["frames"] / fps * float(s.get("speed", 1.0))) for s in shots]
    frames = reframe.reframe_pieces(src, ranges, src_w, src_h, W, H, talk)
    for s, fr in zip(shots, frames):
        track, active = reframe.track_for_shot(fr, float(s["in"]), float(s.get("speed", 1.0)), s["frames"] / fps)
        if track:
            s["track"] = track
        if active:
            s["active"] = active
    return "smart"


def build_audio(src: str, shots: list[dict], fps: float, dst: Path, music: dict | None, fade: float) -> None:
    """The clip's sound: each shot's own source audio (slow motion pitched down, like a film
    slow-mo), tiny fades at every cut so nothing clicks, optional music ducked under it, then
    loudness for social platforms (-14 LUFS)."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    parts, labels = [], []
    total = 0.0
    for k, s in enumerate(shots):
        d = s["frames"] / fps
        sp = float(s.get("speed", 1.0))
        src_len = d * sp
        cmd += ["-ss", f"{float(s['in']):.3f}", "-t", f"{src_len + 0.25:.3f}", "-i", src]
        chain = f"[{k}:a]aresample=48000,asetpts=PTS-STARTPTS,atrim=0:{src_len:.4f}"
        if abs(sp - 1.0) > 1e-3:
            chain += f",asetrate={48000 * sp:.0f},aresample=48000"
        fd = min(fade, d / 3)
        chain += f",apad,atrim=0:{d:.4f},afade=t=in:d={fd:.4f},afade=t=out:st={max(0.0, d - fd):.4f}:d={fd:.4f}[a{k}]"
        parts.append(chain)
        labels.append(f"[a{k}]")
        total += d
    parts.append(f"{''.join(labels)}concat=n={len(shots)}:v=0:a=1[mix]")
    out = "[mix]"
    if music:
        m = len(shots)
        cmd += ["-ss", f"{float(music.get('start', 0.0)):.3f}", "-t", f"{total + 0.5:.3f}", "-i", music["path"]]
        vol = float(music.get("volume", 0.35))
        parts.append(f"[{m}:a]aresample=48000,apad,atrim=0:{total:.4f},afade=t=in:d=0.4,afade=t=out:st={max(0.0, total - 1.2):.4f}:d=1.2,volume={vol:.3f}[mus]")
        parts.append("[mix]asplit=2[g1][g2]")
        parts.append("[mus][g2]sidechaincompress=threshold=0.05:ratio=4:attack=20:release=400[duck]")
        parts.append("[g1][duck]amix=inputs=2:normalize=0:duration=first[mm]")
        out = "[mm]"
    parts.append(f"{out}loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000,atrim=0:{total:.4f}[ao]")
    with tempfile.TemporaryDirectory(prefix="ei-aud-") as tmp:
        g = Path(tmp) / "a.txt"
        g.write_text(";\n".join(parts))
        ac.run(cmd + ac.filter_script_args(str(g)) + ["-map", "[ao]", "-c:a", "pcm_s16le", str(dst)])


def render_edit(src: str, clip: dict, W: int, H: int, fps: float, dst: Path, ass_text: str | None, fade: float,
                music: dict | None = None, workers: int = 2) -> None:
    """Shots (effects, framing) -> picture via render.py; sound per shot; captions burned in last."""
    import render

    shots = clip["shots"]
    total = sum(s["frames"] for s in shots)
    with tempfile.TemporaryDirectory(prefix="ei-clip-") as tmp:
        plan = {"version": 2, "mode": "clip", "width": W, "height": H, "fps": fps, "look": clip.get("look", "natural"),
                "shots": [{**s, "clip": src} for s in shots]}
        plan_path = Path(tmp) / "clip.json"
        plan_path.write_text(json.dumps(plan))
        video = Path(tmp) / "video.mp4"
        render.render(plan_path, str(video), workers=workers)
        audio = Path(tmp) / "audio.wav"
        build_audio(src, shots, fps, audio, music, fade)
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-i", str(audio)]
        if ass_text:
            ass = Path(tmp) / "caps.ass"
            ass.write_text(ass_text, encoding="utf-8")
            e = str(ass).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
            cmd += ["-vf", f"subtitles=filename='{e}'", "-c:v", "h264_videotoolbox", "-b:v", "14M", "-profile:v", "high"]
        else:
            cmd += ["-c:v", "copy"]
        tmp_out = dst.with_name(dst.stem + ".part.mp4")
        cmd += ["-map", "0:v", "-map", "1:a", "-frames:v", str(total), "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(tmp_out)]
        ac.run(cmd)
        tmp_out.replace(dst)


# ---------------------------------------------------------------------------------------------
# Action mode (gameplay, sport, vlogs with little talk): see clip_edit.py
# ---------------------------------------------------------------------------------------------


def action_clips(src: str, duration: float, words: list[dict], count: int, target: float, mn: float, mx: float,
                 hook: bool = True, src_fps: float = 30.0, pace: str = "medium") -> list[dict]:
    import clip_edit as ce

    sig = ce.signals(src, duration)
    score = ce.interest(sig, ce.speech_mask(words, len(sig["loud"])))
    pf = ce.PACE.get(pace, 1.0)
    clips = []
    for s0, s1 in ce.stretches(score, count, target, duration):
        c = ce.edit_clip(ce.moments(score, sig, words, s0, s1, pf), s0, s1, target, mn, mx, hook=hook, src_fps=src_fps, words=words, pace=pf)
        if c is None:
            continue
        dur = sum(s["len"] for s in c["shots"])
        if dur < mn:
            continue
        text = " ".join(w["word"].strip() for w in words if any(a <= w["start"] < b for a, b in c["pieces"]))
        seg = score[int(s0 * ce.HZ) : int(s1 * ce.HZ)]
        c.update({"value": round(float(seg.mean()) if len(seg) else 0.0, 3), "title": None, "text": text})
        log(f"stretch {s0:.0f}–{s1:.0f}s: {ce.describe(c)}")
        clips.append(c)
    clips.sort(key=lambda c: c["pieces"][0][0])
    return clips


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def safe_name(t: str) -> str:
    t = re.sub(r"[^\w\s\-'&]", "", t).strip()
    return re.sub(r"\s+", " ", t)[:60] or "clip"


def make_clips(src: str, out_dir: str, count: int = 5, mn: float = 10, mx: float = 60, target: float = 48, aspect: str = "vertical",
               captions: str = "auto", title: str = "off", hook: bool = True, model: str | None = None, language: str | None = None,
               content: str = "auto", framing: str = "auto", whisper_model: str = ac.DEFAULT_WHISPER, plan_only: bool = False,
               caption_style: str = "pop", music: str | None = None, music_volume: float = 0.35, pace: str = "medium", workers: int = 2) -> dict:
    import captions as cap
    import clip_edit as ce

    ac.require_tools()
    src = str(Path(src).resolve())
    info = ac.probe(src)
    if not info.has_video:
        raise AutocutError("the file has no video")
    if not (1 <= count <= 50):
        raise AutocutError("number of clips must be 1–50")
    if not (3 <= mn <= mx <= 600):
        raise AutocutError("clip lengths must satisfy 3 ≤ min ≤ max ≤ 600 seconds")
    framing = {"crop": "smart", "fit": "blur"}.get(framing, framing)
    if framing not in FRAMINGS:
        raise AutocutError(f"unknown framing {framing!r} ({', '.join(FRAMINGS)})")
    title = {"on": "always", "true": "always", "false": "off"}.get(str(title).lower(), str(title).lower())
    if title not in cap.TITLE_MODES:
        raise AutocutError(f"unknown title mode {title!r} ({', '.join(cap.TITLE_MODES)})")
    if music and not Path(music).is_file():
        raise AutocutError(f"music file not found: {music}")
    target = float(np.clip(target, mn, mx))
    if info.duration < mn:
        raise AutocutError(f"the video ({info.duration:.0f}s) is shorter than the minimum clip length")
    dw, dh = info.display_size
    if aspect == "same":
        sc = min(1.0, 1920 / max(dw, dh, 1))
        W, H = max(2, int(dw * sc) // 2 * 2), max(2, int(dh * sc) // 2 * 2)
    else:
        W, H = ASPECTS[aspect]
    fps = 30.0
    t0 = time.time()
    words: list[dict] = []
    if info.has_audio:
        words = ac.transcribe(src, model=whisper_model, language=language)["words"]
    speech = sum(w["end"] - w["start"] for w in words) / max(1.0, info.duration)
    mode = content if content in ("talk", "gameplay") else ("talk" if speech >= 0.4 else "gameplay")
    log(f"speech covers {speech * 100:.0f}% of the video: {'talk' if mode == 'talk' else 'action / gameplay'} mode")
    clips: list[dict] = []
    if mode == "talk" and words:
        sents = sentences_from_words(words)
        loud = loudness_per_sentence(src, sents)
        for s_, l in zip(sents, loud):
            s_["score"] = sentence_score(s_, l)
            s_["spans"] = keep_spans(s_)
            s_["dur"] = span_len(s_["spans"])
        bounds = topic_boundaries(sents)
        log(f"{len(sents)} sentences in {len(bounds)} topic sections")
        texts = [" ".join(x["text"] for x in sents[b : (bounds[k + 1] if k + 1 < len(bounds) else len(sents))]) for k, b in enumerate(bounds)]
        judged = llm_judge(texts, model)
        bonus = {k: 0.6 * (scv - 0.5) for k, (_, scv) in enumerate(judged) if scv is not None}
        for c in plan_clips(sents, bounds, count, target, mn, mx, bonus):
            pieces, teaser = pieces_for(sents, c["idx"], hook)
            clips.append({"pieces": pieces, "shots": talk_shots(pieces, teaser is not None), "sentences": c["idx"], "teaser": teaser,
                          "value": round(c["value"], 3), "title": fallback_title(sents, c["idx"]), "text": " ".join(sents[i]["text"] for i in c["idx"])})
        fade = 0.012
    else:
        clips = action_clips(src, info.duration, words, count, target, mn, mx, hook=hook, src_fps=info.fps or 30.0, pace=pace)
        fade = 0.03
    if clips:
        if mode == "talk":
            prompts = [f"Clip transcript:\n{c['text']}" for c in clips]
        else:
            tags = visual_tags(src, clips)
            for c, t in zip(clips, tags):
                c["tags"] = t
            prompts = [
                "This is a highlight clip from a gameplay video for YouTube Shorts. "
                f"On screen: {', '.join(c.get('tags') or ['gameplay action'])}. "
                f"Dialogue heard (partial, may be garbled): {c['text'][:600] or '(none)'}\n"
                "Describe the moment (a fight, a duel, a chase, a story twist) in an exciting way. "
                "Do not quote the dialogue word for word."
                for c in clips
            ]
        titles = [tidy_title(t) for t in llm_titles(prompts, model)]
        seen: set[str] = set()
        dup = []
        for k, t in enumerate(titles):
            if t and t.lower() in seen:
                dup.append(k)
            elif t:
                seen.add(t.lower())
        if dup:  # ask again for clips whose title repeats another one
            avoid = "; ".join(sorted(seen))
            again = llm_titles([prompts[k] + f"\nThe title must be different from these: {avoid}" for k in dup], model)
            for k, t in zip(dup, again):
                t = tidy_title(t)
                titles[k] = t if t and t.lower() not in seen else None
                if titles[k]:
                    seen.add(titles[k].lower())
        for c, t in zip(clips, titles):
            if t:
                c["title"] = t
            elif mode != "talk" and not c.get("title"):
                c["title"] = " ".join(w.capitalize() for w in (c.get("tags") or ["Highlight"])[0].replace("a ", "", 1).split()) or None
    if len(clips) < count:
        log(f"only {len(clips)} clip(s) of {mn:g}–{mx:g}s fit without reusing material")
    mus = None
    if music:
        import music as mu

        mus = {"path": str(Path(music).resolve()), "volume": music_volume, "analysis": mu.analyse(str(Path(music).resolve()))}
    used_framing = framing
    for c in clips:
        if mus:  # cuts onto the beat
            length = sum(s["len"] for s in c["shots"])
            start = mu.choose_window(mus["analysis"], length)
            beats = [b - start for b in mus["analysis"]["beats"] if start <= b < start + length + 1]
            ce.snap_to_beats(c["shots"], beats)
            c["music_start"] = round(start, 3)
        c["shots"] = finalise_shots(c["shots"], fps)
        used_framing = frame_shots(src, c["shots"], dw, dh, W, H, fps, framing, mode == "talk")
        c["framing"] = used_framing
        c["duration"] = round(sum(s["frames"] for s in c["shots"]) / fps, 2)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    plan = {"version": 3, "mode": "highlights", "content": mode, "input": src, "width": W, "height": H, "fps": fps, "count": count,
            "min": mn, "max": mx, "target": target, "captions": captions, "caption_style": caption_style, "title": title,
            "framing": used_framing, "music": mus and {"path": mus["path"], "volume": music_volume}, "clips": clips}
    plan_path = out / f"{stem}.clips.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    log(f"planned {len(clips)} clips in {time.time() - t0:.0f}s ({used_framing} framing): "
        + ", ".join(f"{c['duration']:.0f}s/{len(c['shots'])} shots" for c in clips))
    outputs = []
    if not plan_only:
        clean = cap.clean_words([w for w in words if norm_word(w["word"]) not in FILLERS]) if words else []
        for k, c in enumerate(clips, 1):
            name = f"{stem} - {k:02d}" + (f" - {safe_name(c['title'])}" if c.get("title") else "")
            dst = out / f"{name}.mp4"
            wout = cap.remap_to_shots(clean, c["shots"], fps) if clean else []
            caps_on = captions == "on" or (captions == "auto" and cap.coverage(wout, c["duration"]) >= 0.2)
            show_title = c.get("title") if title != "off" else None
            ass = cap.build_ass(wout if caps_on else [], show_title, W, H, c["duration"], caption_style, title) if ((caps_on and wout) or show_title) else None
            t1 = time.time()
            m = {"path": mus["path"], "volume": music_volume, "start": c.get("music_start", 0.0)} if mus else None
            render_edit(src, c, W, H, fps, dst, ass, fade, m, workers)
            if wout:
                dst.with_suffix(".srt").write_text(ac.to_srt(ac.group_words(wout, max_words=7, max_chars=40)), encoding="utf-8")
            log(f"progress {k}/{len(clips)} clips: {dst.name} ({c['duration']:.0f}s, {len(c['shots'])} shots, "
                f"captions {'on' if caps_on and wout else 'off'}) in {time.time() - t1:.0f}s")
            outputs.append(str(dst))
    return {"outputs": outputs, "out_dir": str(out), "plan": str(plan_path),
            "clips": [{"title": c.get("title"), "duration": c["duration"], "shots": len(c["shots"])} for c in clips]}


def add_args(p: argparse.ArgumentParser) -> None:
    import captions as cap

    p.add_argument("input")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--count", type=int, default=5, help="how many clips")
    p.add_argument("--min", type=float, default=10.0, help="shortest clip (s)")
    p.add_argument("--max", type=float, default=60.0, help="longest clip (s)")
    p.add_argument("--target", type=float, default=48.0, help="typical clip length (s)")
    p.add_argument("--aspect", default="vertical", choices=["vertical", "square", "horizontal", "same"])
    p.add_argument("--content", default="auto", choices=["auto", "talk", "gameplay"], help="talk: built from sentences; gameplay: from action moments")
    p.add_argument("--framing", default="auto", choices=[*FRAMINGS, "crop", "fit"],
                   help="smart: fill the shape following the action/faces; center: fill, centred crop; blur: whole picture over a blurred copy; bars: whole picture with black bars")
    p.add_argument("--captions", default="auto", choices=["auto", "on", "off"], help="auto: when the clip has clear speech")
    p.add_argument("--caption-style", default="pop", choices=list(cap.STYLES))
    p.add_argument("--title", default="off", choices=list(cap.TITLE_MODES), help="title card in the video (the title is always in the file name)")
    p.add_argument("--no-title", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-hook", action="store_true", help="don't open with a teaser of the best moment")
    p.add_argument("--pace", default="medium", choices=list(__import__("clip_edit").PACE))
    p.add_argument("--music", help="optional music (audio or video file): cuts move onto its beats, it plays under the clip's sound")
    p.add_argument("--music-volume", type=float, default=0.35)
    p.add_argument("--llm", default="default", help="language model key (see llm.py), 'default' or 'none'")
    p.add_argument("--language")
    p.add_argument("--plan-only", action="store_true")


def run_args(a) -> dict:
    return make_clips(a.input, a.out_dir, a.count, a.min, a.max, a.target, a.aspect, captions=a.captions,
                      title="off" if a.no_title else a.title, hook=not a.no_hook, model=a.llm, language=a.language, content=a.content,
                      framing=a.framing, plan_only=a.plan_only, caption_style=a.caption_style, music=a.music, music_volume=a.music_volume,
                      pace=a.pace, workers=getattr(a, "workers", 2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    try:
        print(json.dumps(run_args(ap.parse_args(argv))))
    except AutocutError as e:
        log(f"error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
