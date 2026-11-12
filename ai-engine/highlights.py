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


def face_centres(src: str, ranges: list[tuple[float, float]]) -> list[tuple[float, float] | None]:
    """Normalised face centre (x, y) per time range via Apple Vision; None when no face."""
    try:
        import Vision
        from Foundation import NSURL
    except ImportError:
        return [None] * len(ranges)
    out = []
    with tempfile.TemporaryDirectory(prefix="ei-faces-") as tmp:
        for k, (a, b) in enumerate(ranges):
            xs = []
            n = max(1, min(4, int((b - a) / 0.8)))
            for j in range(n):
                t = a + (b - a) * (j + 0.5) / n
                img = Path(tmp) / f"f{k}_{j}.jpg"
                subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", src, "-frames:v", "1", "-vf", "scale=640:-2", str(img)], capture_output=True)
                if not img.is_file():
                    continue
                req = Vision.VNDetectFaceRectanglesRequest.alloc().init()
                handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(str(img)), None)
                ok = handler.performRequests_error_([req], None)
                if not (ok[0] if isinstance(ok, tuple) else ok):
                    continue
                faces = req.results() or []
                if faces:
                    f = max(faces, key=lambda r: r.boundingBox().size.width * r.boundingBox().size.height)
                    bb = f.boundingBox()
                    xs.append((bb.origin.x + bb.size.width / 2, 1 - (bb.origin.y + bb.size.height / 2)))
            out.append((float(np.median([x for x, _ in xs])), float(np.median([y for _, y in xs]))) if xs else None)
    return out


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
# Captions (ASS: word-by-word highlight + title)
# ---------------------------------------------------------------------------------------------


def ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_escape(t: str) -> str:
    return t.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def build_ass(words_out: list[dict], title: str | None, W: int, H: int, total: float) -> str:
    big = H > W
    fs = int(H * (0.05 if big else 0.06))
    tfs = int(H * (0.034 if big else 0.05))
    margin_v = int(H * (0.24 if big else 0.08))
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Helvetica Neue,{fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,{max(3, fs // 12)},2,2,60,60,{margin_v},1
Style: Title,Helvetica Neue,{tfs},&H00FFFFFF,&H00FFFFFF,&H00000000,&HB4000000,1,0,0,0,100,100,0,0,3,{max(8, tfs // 3)},0,8,80,80,{int(H * 0.08)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    if title:
        ev.append(f"Dialogue: 1,{ass_time(0)},{ass_time(total)},Title,,0,0,0,,{ass_escape(title)}")
    cues = ac.group_words(words_out, max_words=3 if big else 6, max_chars=18 if big else 34, max_gap=0.5)
    for cue in cues:
        ws = [w for w in words_out if cue["start"] - 1e-3 <= w["start"] < cue["end"] + 1e-3]
        if not ws:
            ev.append(f"Dialogue: 0,{ass_time(cue['start'])},{ass_time(cue['end'])},Cap,,0,0,0,,{ass_escape(cue['text'].upper())}")
            continue
        for k, w in enumerate(ws):
            a = w["start"] if k else cue["start"]
            b = ws[k + 1]["start"] if k + 1 < len(ws) else cue["end"]
            if b <= a:
                continue
            parts = []
            for j, x in enumerate(ws):
                t = ass_escape(x["word"].strip().upper())
                parts.append("{\\c&H00E6FF&}" + t + "{\\c&HFFFFFF&}" if j == k else t)
            ev.append(f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Cap,,0,0,0,,{' '.join(parts)}")
    return head + "\n".join(ev) + "\n"


def remap_to_pieces(words: list[dict], pieces: list[tuple[float, float]]) -> list[dict]:
    out = []
    acc = 0.0
    for a, b in pieces:
        for w in words:
            if norm_word(w["word"]) in FILLERS:
                continue
            ws, we = float(w["start"]), float(w["end"])
            if ws >= a - 0.02 and ws < b:
                out.append({"word": w["word"], "start": acc + max(0.0, ws - a), "end": acc + min(b, we) - a})
        acc += b - a
    out.sort(key=lambda w: w["start"])
    return out


# ---------------------------------------------------------------------------------------------
# Rendering one clip
# ---------------------------------------------------------------------------------------------


def piece_video_chain(k: int, a: float, b: float, f, z: float, W: int, H: int, fps: float, framing: str) -> str:
    frames = max(1, round((b - a) * fps))
    head = f"[{k}:v]setpts=PTS-STARTPTS,fps={fps:g}"
    tail = f"setsar=1,format=yuv420p,trim=end_frame={frames},setpts=PTS-STARTPTS[v{k}]"
    if framing == "fit":
        # the whole frame, over a blurred, darkened copy of itself filling the shape
        return (
            f"{head},split=2[bg{k}][fg{k}];"
            f"[bg{k}]scale={W // 8}:{H // 8}:force_original_aspect_ratio=increase,crop={W // 8}:{H // 8},boxblur=6:2,"
            f"scale={W}:{H},eq=brightness=-0.12:saturation=1.15[bb{k}];"
            f"[fg{k}]scale='if(gt(a,{W}/{H}),{W},-2)':'if(gt(a,{W}/{H}),-2,{H})':flags=lanczos,scale=trunc(iw*{z:.3f}/2)*2:trunc(ih*{z:.3f}/2)*2,"
            f"crop='min(iw,{W})':'min(ih,{H})'[ff{k}];"
            f"[bb{k}][ff{k}]overlay=x='(W-w)/2':y='(H-h)/2',{tail}"
        )
    fx = f[0] if f else 0.5
    fy = f[1] if f else 0.4
    cw = f"min(iw,ih*{W}/{H})/{z:.3f}"
    ch = f"min(ih,iw*{H}/{W})/{z:.3f}"
    return (
        f"{head},crop=w='{cw}':h='{ch}':x='max(0,min(iw-ow,iw*{fx:.3f}-ow/2))':y='max(0,min(ih-oh,ih*{fy:.3f}-oh/2))',"
        f"scale={W}:{H}:flags=lanczos,{tail}"
    )


def render_clip(src: str, clip: dict, W: int, H: int, fps: float, dst: Path, ass_text: str | None, fade: float = 0.015) -> None:
    pieces = clip["pieces"]
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    for a, b in pieces:
        cmd += ["-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", src]
    parts, labels = [], []
    for k, ((a, b), f, z) in enumerate(zip(pieces, clip["focus"], clip["zooms"])):
        parts.append(piece_video_chain(k, a, b, f, z, W, H, fps, clip.get("framing", "crop")))
        d = max(1, round((b - a) * fps)) / fps
        fd = min(fade, d / 3)
        parts.append(f"[{k}:a]aresample=48000,asetpts=PTS-STARTPTS,atrim=0:{d:.4f},afade=t=in:d={fd:.4f},afade=t=out:st={max(0.0, d - fd):.4f}:d={fd:.4f}[a{k}]")
        labels.append(f"[v{k}][a{k}]")
    parts.append(f"{''.join(labels)}concat=n={len(pieces)}:v=1:a=1[vc][ac]")
    vout = "[vc]"
    with tempfile.TemporaryDirectory(prefix="ei-clip-") as tmp:
        if ass_text:
            ass = Path(tmp) / "caps.ass"
            ass.write_text(ass_text, encoding="utf-8")
            esc = str(ass).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
            parts.append(f"[vc]subtitles=filename='{esc}'[vs]")
            vout = "[vs]"
        parts.append("[ac]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[ao]")
        graph = Path(tmp) / "g.txt"
        graph.write_text(";\n".join(parts))
        cmd += ac.filter_script_args(str(graph))
        cmd += ["-map", vout, "-map", "[ao]", "-fps_mode", "cfr", "-r", ac.rate_arg(fps), "-c:v", "h264_videotoolbox", "-b:v", "10M",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(dst)]
        ac.run(cmd)


# ---------------------------------------------------------------------------------------------
# Action mode (gameplay, sport, vlogs with little talk)
# ---------------------------------------------------------------------------------------------


def action_signals(src: str, duration: float, words: list[dict]) -> dict:
    """Per-second loudness, attacks, motion, scene changes, brightness and dialogue (cached)."""
    import footage as fo
    import montage as mt

    key = ac.file_key(src, "action-v1")
    cache = ac.CACHE_DIR / "action" / f"{key}.json"
    if cache.is_file():
        sig = json.loads(cache.read_text())
    else:
        t0 = time.time()
        log("measuring action across the video (loudness, motion, scene changes); cached afterwards")
        a = fo.analyse(src)
        secs = max(1, int(duration))
        mot = np.asarray(a["motion"], dtype=np.float32)
        bright = np.asarray(a["bright"], dtype=np.float32)
        per = lambda v: [float(v[int(i * fo.FPS) : int((i + 1) * fo.FPS)].mean()) if int(i * fo.FPS) < len(v) else 0.0 for i in range(secs)]  # noqa: E731
        med = float(np.median(mot)) if len(mot) else 1.0
        cut_flags = (mot > max(30.0, 5 * med)).astype(np.float32)
        y = mt.decode_mono(src, 8000)
        hop = 400  # 50 ms
        nb = len(y) // hop
        r = np.sqrt((y[: nb * hop].reshape(nb, hop) ** 2).mean(axis=1)) if nb else np.zeros(0)
        db = 20 * np.log10(np.maximum(r, 1e-5))
        rise = np.maximum(0, np.diff(db, prepend=db[:1] if len(db) else 0))
        loud = [float(db[i * 20 : (i + 1) * 20].mean()) if i * 20 < len(db) else -80.0 for i in range(secs)]
        attacks = [float((rise[i * 20 : (i + 1) * 20] > 6).sum()) if i * 20 < len(rise) else 0.0 for i in range(secs)]
        sig = {"loud": loud, "attacks": attacks, "motion": per(mot), "cuts": per(cut_flags), "bright": per(bright)}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(sig))
        log(f"action signals in {time.time() - t0:.0f}s")
    secs = len(sig["loud"])
    speech = np.zeros(secs)
    for w in words:
        for i in range(int(w["start"]), min(secs, int(w["end"]) + 1)):
            speech[i] = 1
    sig["speech"] = speech.tolist()
    return sig


def interest_curve(sig: dict) -> np.ndarray:
    def z(v):
        v = np.asarray(v, dtype=np.float64)
        return (v - np.median(v)) / (v.std() + 1e-6)

    loud, mot = np.asarray(sig["loud"]), np.asarray(sig["motion"])
    s = 0.35 * z(loud) + 0.25 * z(sig["attacks"]) + 0.25 * z(np.log1p(mot)) + 0.1 * z(sig["cuts"]) + 0.15 * np.asarray(sig["speech"])
    still = (mot < np.percentile(mot, 15)) & (loud < np.percentile(loud, 40))
    s[still] -= 1.0  # menus, maps, loading screens
    s[np.asarray(sig["bright"]) < 14] -= 2.0  # black
    return np.convolve(s, np.ones(3) / 3, mode="same")


def boundary_near(t: float, lo: float, hi: float, sig: dict, words: list[dict], prefer_late: bool) -> float:
    """A natural cut point in [lo, hi]: a scene change, else the quietest second; never inside a line of dialogue."""
    secs = len(sig["loud"])
    cand = [i for i in range(max(0, int(lo)), min(secs - 1, int(hi)) + 1)]
    if not cand:
        return t
    cuts = [i for i in cand if sig["cuts"][i] > 0]
    if cuts:
        b = float(max(cuts) if prefer_late else min(cuts))
    else:
        b = float(min(cand, key=lambda i: sig["loud"][i]))
    for w in words:  # don't cut a word in half
        if w["start"] < b < w["end"]:
            b = w["end"] + 0.05 if prefer_late else w["start"] - 0.05
    return b


def action_clips(src: str, duration: float, words: list[dict], count: int, target: float, mn: float, mx: float) -> list[dict]:
    sig = action_signals(src, duration, words)
    score = interest_curve(sig)
    secs = len(score)
    # Hotspots: where a target-long stretch is most intense. Each clip is built around one
    # (one fight, one scene), and hotspots keep apart so clips never share material.
    R = int(np.clip(2.2 * target, 30, max(30, duration / max(count, 1))))
    win = max(3, int(target))
    smooth = np.convolve(score, np.ones(win) / win, mode="same")
    used = np.zeros(secs, bool)
    clips = []
    for _ in range(count):
        order = np.argsort(-smooth)
        best = None
        for c in order:
            c = int(c)
            s0, s1 = max(0, c - R // 2), min(secs, c + R // 2)
            if s1 - s0 >= mn and not used[s0:s1].any():
                best = (s0, s1)
                break
        if best is None:
            break
        s0, s1 = best
        used[max(0, s0 - 3) : min(secs, s1 + 3)] = True
        bv = float(smooth[(s0 + s1) // 2])
        seg = score[s0:s1]
        pieces: list[tuple[float, float]] = []
        total = 0.0
        goal = min(mx, max(mn, target))
        for i in np.argsort(-seg):
            t = s0 + int(i)
            if any(a - 2 <= t <= b + 2 for a, b in pieces):
                continue
            span = float(np.clip(5 + 7 * (seg[i] - np.median(seg)) / (np.ptp(seg) + 1e-6), 4, 12))
            a0 = boundary_near(t, t - span / 2 - 2, t - 2, sig, words, prefer_late=False)
            b0 = boundary_near(t, t + 2, t + span / 2 + 2, sig, words, prefer_late=True)
            a0, b0 = max(float(s0), a0), min(float(s1), b0)
            if b0 - a0 < 2.5 or any(not (b0 <= a or a0 >= b) for a, b in pieces):
                continue
            pieces.append((a0, b0))
            total += b0 - a0
            if total >= goal - 2:
                break
        pieces.sort()
        if total > mx and pieces:  # trim the last piece to fit
            a0, b0 = pieces[-1]
            pieces[-1] = (a0, max(a0 + 2.5, b0 - (total - mx)))
        if sum(b - a for a, b in pieces) >= mn:
            text = " ".join(w["word"].strip() for w in words if any(a <= w["start"] < b for a, b in pieces))
            clips.append({"pieces": pieces, "value": round(bv, 3), "title": None, "text": text})
        elif len(pieces) == 0:
            continue
    clips.sort(key=lambda c: c["pieces"][0][0])
    return clips


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def safe_name(t: str) -> str:
    t = re.sub(r"[^\w\s\-'&]", "", t).strip()
    return re.sub(r"\s+", " ", t)[:60] or "clip"


def make_clips(src: str, out_dir: str, count: int = 5, mn: float = 10, mx: float = 60, target: float = 48, aspect: str = "vertical",
               captions: str = "auto", title: bool = True, hook: bool = True, model: str | None = None, language: str | None = None,
               content: str = "auto", framing: str = "auto", whisper_model: str = ac.DEFAULT_WHISPER, plan_only: bool = False) -> dict:
    ac.require_tools()
    src = str(Path(src).resolve())
    info = ac.probe(src)
    if not info.has_video:
        raise AutocutError("the file has no video")
    if not (1 <= count <= 50):
        raise AutocutError("number of clips must be 1–50")
    if not (3 <= mn <= mx <= 600):
        raise AutocutError("clip lengths must satisfy 3 ≤ min ≤ max ≤ 600 seconds")
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
            clips.append({"pieces": pieces, "sentences": c["idx"], "teaser": teaser, "value": round(c["value"], 3),
                          "title": fallback_title(sents, c["idx"]), "text": " ".join(sents[i]["text"] for i in c["idx"])})
        fade = 0.015
    else:
        clips = action_clips(src, info.duration, words, count, target, mn, mx)
        fade = 0.12
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
        for c, t in zip(clips, llm_titles(prompts, model)):
            t = tidy_title(t)
            if t:
                c["title"] = t
            elif mode != "talk":
                c["title"] = " ".join(w.capitalize() for w in (c.get("tags") or ["Highlight"])[0].replace("a ", "", 1).split()) or None
    if len(clips) < count:
        log(f"only {len(clips)} clip(s) of {mn:g}–{mx:g}s fit without reusing material")
    for c in clips:
        want_faces = framing == "crop" or (framing == "auto" and mode == "talk")
        faces = face_centres(src, c["pieces"]) if (want_faces and (W / H) < (dw / max(dh, 1)) - 0.05) else [None] * len(c["pieces"])
        found = sum(1 for f in faces if f) >= max(1, len(faces) // 3)
        same_shape = abs(W / H - dw / max(dh, 1)) < 0.05
        if framing in ("crop", "fit"):
            c["framing"] = framing
        elif mode == "gameplay":
            c["framing"] = "crop" if same_shape else "fit"  # keep the HUD and the game's subtitles
        else:
            c["framing"] = "crop" if found or same_shape else "fit"
        last = None
        for k, f in enumerate(faces):
            faces[k] = f if f else last
            last = faces[k]
        c["focus"] = faces
        z, zooms, prev_end = (1.3 if c["framing"] == "fit" and mode == "gameplay" and H > W else 1.0), [], None
        for a, b in c["pieces"]:
            if c["framing"] == "crop" and prev_end is not None and 0 < a - prev_end < 4.0:
                z = 1.12 if z == 1.0 else 1.0  # alternate framing on a jump cut
            zooms.append(z)
            prev_end = b
        c["zooms"] = zooms
        c["duration"] = round(sum(b - a for a, b in c["pieces"]), 2)
    caps = (mode == "talk") if captions == "auto" else captions == "on"
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    plan = {"version": 2, "mode": "highlights", "content": mode, "input": src, "width": W, "height": H, "fps": fps, "count": count,
            "min": mn, "max": mx, "target": target, "captions": caps, "title": title, "clips": clips}
    plan_path = out / f"{stem}.clips.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    log(f"planned {len(clips)} clips in {time.time() - t0:.0f}s: " + ", ".join(f"{c['duration']:.0f}s/{len(c['pieces'])}" for c in clips))
    outputs = []
    if not plan_only:
        for k, c in enumerate(clips, 1):
            name = f"{stem} - {k:02d}" + (f" - {safe_name(c['title'])}" if c.get("title") else "")
            dst = out / f"{name}.mp4"
            wout = remap_to_pieces(words, c["pieces"]) if words else []
            show_title = c.get("title") if title else None
            ass = build_ass(wout if caps else [], show_title, W, H, c["duration"]) if ((caps and wout) or show_title) else None
            t1 = time.time()
            render_clip(src, c, W, H, fps, dst, ass, fade)
            if wout:
                dst.with_suffix(".srt").write_text(ac.to_srt(ac.group_words(wout, max_words=7, max_chars=40)), encoding="utf-8")
            log(f"progress {k}/{len(clips)} clips: {dst.name} ({c['duration']:.0f}s, {len(c['pieces'])} pieces) in {time.time() - t1:.0f}s")
            outputs.append(str(dst))
    return {"outputs": outputs, "out_dir": str(out), "plan": str(plan_path),
            "clips": [{"title": c.get("title"), "duration": c["duration"], "pieces": len(c["pieces"])} for c in clips]}


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--count", type=int, default=5, help="how many clips")
    p.add_argument("--min", type=float, default=10.0, help="shortest clip (s)")
    p.add_argument("--max", type=float, default=60.0, help="longest clip (s)")
    p.add_argument("--target", type=float, default=48.0, help="typical clip length (s)")
    p.add_argument("--aspect", default="vertical", choices=["vertical", "square", "horizontal", "same"])
    p.add_argument("--content", default="auto", choices=["auto", "talk", "gameplay"], help="talk: built from sentences; gameplay: from action moments")
    p.add_argument("--framing", default="auto", choices=["auto", "crop", "fit"], help="crop (follow faces) or fit over a blurred background")
    p.add_argument("--captions", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--no-title", action="store_true")
    p.add_argument("--no-hook", action="store_true", help="don't open with the strongest line")
    p.add_argument("--llm", default="default", help="language model key (see llm.py), 'default' or 'none'")
    p.add_argument("--language")
    p.add_argument("--plan-only", action="store_true")


def run_args(a) -> dict:
    return make_clips(a.input, a.out_dir, a.count, a.min, a.max, a.target, a.aspect, captions=a.captions, title=not a.no_title,
                      hook=not a.no_hook, model=a.llm, language=a.language, content=a.content, framing=a.framing, plan_only=a.plan_only)


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
