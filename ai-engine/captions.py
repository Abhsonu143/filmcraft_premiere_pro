"""captions: burned-in captions and title cards for short social clips (ASS, rendered by libass).

Caption styles
  pop     big heavy uppercase words, 1–3 at a time on vertical, the spoken word highlighted and
          slightly larger (the TikTok / Shorts look)
  clean   sentence case, medium weight, up to ~6 words, white with a soft outline
  boxed   sentence case on a dark box (readable over anything)

Placement keeps clear of the platform buttons and description (bottom ~25% and the right edge
on TikTok/Reels/Shorts): vertical captions sit about two thirds down, square a little lower,
horizontal near the bottom as usual.

Title card (optional): off, intro (the first 3 s, popping in and fading out) or always.

Whisper on game audio and music sometimes "hears" words that aren't there ("Thank you.",
"you you you"); those are dropped before captions are made.
"""

from __future__ import annotations

import re

import autocut as ac

STYLES = ("pop", "clean", "boxed")
TITLE_MODES = ("off", "intro", "always")
INTERJECTIONS = {"ah", "aah", "oh", "uh", "um", "hmm", "huh", "ha", "hah", "argh", "ugh", "hey", "ow", "whoa", "mm", "eh"}
HALLUCINATIONS = re.compile(r"^(thank you\.?|thanks for watching!?|you|bye\.?|\.+|subtitles by.*|♪+)$", re.I)


def norm(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower())


def clean_words(words: list[dict]) -> list[dict]:
    """Drop words a viewer would read as mistakes: runs of the same word, stretched words,
    isolated stock phrases in silence, symbols only."""
    out: list[dict] = []
    run = 0
    for i, w in enumerate(words):
        t = w["word"].strip()
        n = norm(t)
        if not n:
            continue
        if w["end"] - w["start"] > 2.5:
            continue
        same = bool(out) and norm(out[-1]["word"]) == n
        run = run + 1 if same else 0
        if run >= 2 or (same and w["start"] - out[-1]["end"] < 0.4 and len(n) <= 4):
            continue  # "the the": a stutter
        out.append(dict(w))
    out = drop_loops(out)
    # remove "lonely" stock phrases: 1–3 words alone with > 2 s of silence around them
    keep = []
    for i, w in enumerate(out):
        prev_gap = w["start"] - out[i - 1]["end"] if i else 99.0
        next_gap = out[i + 1]["start"] - w["end"] if i + 1 < len(out) else 99.0
        if prev_gap > 2 and next_gap > 2 and HALLUCINATIONS.match(w["word"].strip()):
            continue
        if prev_gap > 0.6 and next_gap > 0.6 and norm(w["word"]) in INTERJECTIONS:
            continue  # a lone grunt or shout reads as noise on screen
        keep.append(w)
    return keep


def drop_loops(words: list[dict], min_repeats: int = 3) -> list[dict]:
    """Whisper loops ("we are we are we are"): a phrase of 1–4 words repeated 3+ times in a row
    is kept once."""
    keys = [norm(w["word"]) for w in words]
    out: list[dict] = []
    i = 0
    while i < len(words):
        skipped = False
        for n in (4, 3, 2, 1):
            seq = keys[i : i + n]
            if len(seq) < n:
                continue
            reps = 1
            while keys[i + reps * n : i + (reps + 1) * n] == seq:
                reps += 1
            if reps >= min_repeats:
                out += words[i : i + n]
                i += reps * n
                skipped = True
                break
        if not skipped:
            out.append(words[i])
            i += 1
    return out


def coverage(words: list[dict], total: float) -> float:
    return sum(max(0.0, w["end"] - w["start"]) for w in words) / max(total, 1e-3)


def ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def esc(t: str) -> str:
    return t.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def _display(word: str, style: str) -> str:
    w = word.strip()
    if style == "pop":
        w = w.upper().rstrip(",.;:")
    return esc(w)


def layout(W: int, H: int, style: str) -> dict:
    vertical, square = H > W * 1.2, abs(W - H) < W * 0.2
    base = min(W, H)
    if style == "pop":
        fs = int(base * (0.085 if vertical else 0.075 if square else 0.062))
        words, chars = (3, 16) if vertical else (4, 22) if square else (6, 34)
    else:
        fs = int(base * (0.062 if vertical else 0.058 if square else 0.05))
        words, chars = (5, 26) if vertical else (6, 32) if square else (9, 48)
    margin_v = int(H * (0.30 if vertical else 0.13 if square else 0.08))
    return {"fs": fs, "words": words, "chars": chars, "margin_v": margin_v, "vertical": vertical}


def header(W: int, H: int, style: str) -> str:
    L = layout(W, H, style)
    fs = L["fs"]
    if style == "pop":
        cap = f"Style: Cap,Avenir Next Heavy,{fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,1,0,1,{max(4, fs // 8)},{max(2, fs // 22)},2,{W // 12},{W // 12},{L['margin_v']},1"
    elif style == "clean":
        cap = f"Style: Cap,Avenir Next Demi Bold,{fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,{max(2, fs // 16)},{max(1, fs // 30)},2,{W // 12},{W // 12},{L['margin_v']},1"
    else:
        cap = f"Style: Cap,Avenir Next Demi Bold,{fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H40000000,1,0,0,0,100,100,0,0,3,{max(6, fs // 4)},0,2,{W // 12},{W // 12},{L['margin_v']},1"
    tfs = int(min(W, H) * (0.075 if L["vertical"] else 0.06))
    title = f"Style: Title,Avenir Next Heavy,{tfs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,1,0,0,0,100,100,0,0,1,{max(4, tfs // 9)},{max(3, tfs // 16)},8,{W // 10},{W // 10},{int(H * (0.16 if L['vertical'] else 0.08))},1"
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{cap}
{title}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def cues_for(words: list[dict], W: int, H: int, style: str) -> list[dict]:
    L = layout(W, H, style)
    cues = ac.group_words(words, max_words=L["words"], max_chars=L["chars"], max_gap=0.45)
    # hold each cue a little into the pause after it (and at least 0.4 s), never into the next
    for i, c in enumerate(cues):
        nxt = cues[i + 1]["start"] if i + 1 < len(cues) else c["end"] + 0.6
        c["end"] = min(nxt, max(c["end"] + 0.25, c["start"] + 0.4))
    return cues


HIGHLIGHT = "&H3DE1FF&"  # warm yellow (ASS colours are BGR)


def build_ass(words: list[dict], title: str | None, W: int, H: int, total: float, style: str = "pop", title_mode: str = "intro") -> str:
    style = style if style in STYLES else "pop"
    ev: list[str] = []
    if title and title_mode != "off":
        end = min(total, 3.0) if title_mode == "intro" else total
        anim = "{\\fad(120,350)\\fscx70\\fscy70\\t(0,180,\\fscx100\\fscy100)}" if title_mode == "intro" else "{\\fad(150,0)}"
        ev.append(f"Dialogue: 1,{ass_time(0)},{ass_time(end)},Title,,0,0,0,,{anim}{esc(title)}")
    for cue in cues_for(words, W, H, style):
        ws = [w for w in words if cue["start"] - 1e-3 <= w["start"] < cue["end"] - 1e-3 and w["word"].strip()]
        if not ws:
            continue
        pop_in = "{\\fscx92\\fscy92\\t(0,90,\\fscx100\\fscy100)}"
        if style != "pop":
            ev.append(f"Dialogue: 0,{ass_time(cue['start'])},{ass_time(cue['end'])},Cap,,0,0,0,,{' '.join(_display(w['word'], style) for w in ws)}")
            continue
        for k, w in enumerate(ws):
            a = cue["start"] if k == 0 else w["start"]
            b = ws[k + 1]["start"] if k + 1 < len(ws) else cue["end"]
            if b - a < 0.02:
                continue
            parts = []
            for j, x in enumerate(ws):
                t = _display(x["word"], style)
                parts.append(f"{{\\c{HIGHLIGHT}\\fscx108\\fscy108}}{t}{{\\c&HFFFFFF&\\fscx100\\fscy100}}" if j == k else t)
            ev.append(f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Cap,,0,0,0,,{pop_in if k == 0 else ''}{' '.join(parts)}")
    return header(W, H, style) + "\n".join(ev) + "\n"


def remap_to_shots(words: list[dict], shots: list[dict], fps: float) -> list[dict]:
    """Words onto the edited timeline (shots: in, frames, speed); words cut off by an edit go."""
    out = []
    acc = 0.0
    for s in shots:
        d = s["frames"] / fps
        sp = float(s.get("speed", 1.0))
        a, b = float(s["in"]), float(s["in"]) + d * sp
        for w in words:
            ws, we = float(w["start"]), float(w["end"])
            if ws >= a - 0.02 and ws < b - 0.05:
                out.append({"word": w["word"], "start": acc + max(0.0, ws - a) / sp, "end": acc + (min(b, we) - a) / sp})
        acc += d
    out.sort(key=lambda w: w["start"])
    return out
