#!/usr/bin/env python3
"""explain_draw: the pictures of an explainer video, drawn from the script's words.

A scene (a flow, code, a bug and its fix, a terminal, a marked passage of a source page, a quote,
an example, a comparison, a chart, a timeline, a card) is drawn at a `step`: the same scene with
more of it on screen, so it builds up while the voice talks. Everything is laid out for the frame
it is given, so the same scene works wide (16:9) and tall (9:16, for a clip posted alone).

Fonts are the Mac's own (nothing is shipped). Looks: clean (paper, documentary), dark (code),
viral (bold, word-by-word captions, a small push on every change).
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

LOOKS = {
    "clean": {"bg": (246, 243, 235), "panel": (255, 255, 253), "ink": (26, 28, 34), "soft": (112, 114, 122), "accent": (190, 58, 40), "mark": (255, 229, 92),
              "line": (212, 206, 192), "ghost": (232, 228, 218), "code_bg": (252, 251, 247), "code_ink": (36, 38, 46), "band": (255, 240, 170),
              "syntax": {"kw": (160, 40, 130), "str": (24, 120, 70), "com": (140, 140, 140), "num": (170, 90, 10)},
              "captions": "sentence", "punch": 0.0, "fade": 0.35},
    "dark": {"bg": (16, 18, 24), "panel": (26, 29, 38), "ink": (232, 234, 240), "soft": (140, 146, 160), "accent": (96, 196, 255), "mark": (120, 104, 20),
             "line": (52, 58, 74), "ghost": (30, 33, 43), "code_bg": (22, 24, 32), "code_ink": (222, 226, 236), "band": (44, 60, 92),
             "syntax": {"kw": (198, 142, 255), "str": (150, 220, 140), "com": (110, 118, 134), "num": (255, 180, 100)},
             "captions": "sentence", "punch": 0.0, "fade": 0.3},
    "viral": {"bg": (12, 10, 28), "panel": (28, 24, 56), "ink": (255, 255, 255), "soft": (176, 172, 210), "accent": (255, 214, 10), "mark": (150, 112, 0),
              "line": (70, 62, 120), "ghost": (22, 19, 46), "code_bg": (20, 17, 42), "code_ink": (240, 240, 250), "band": (72, 56, 130),
              "syntax": {"kw": (255, 120, 190), "str": (130, 240, 170), "com": (130, 126, 170), "num": (255, 200, 90)},
              "captions": "words", "punch": 0.035, "fade": 0.16},
}
RED, GREEN = (222, 72, 60), (52, 168, 96)
TERM = {"bg": (20, 22, 28), "in": (236, 238, 244), "out": (170, 176, 190), "err": (255, 120, 108), "ok": (120, 220, 150)}
FONTS = {
    "sans": ("/System/Library/Fonts/Avenir Next.ttc", 5), "bold": ("/System/Library/Fonts/Avenir Next.ttc", 0), "heavy": ("/System/Library/Fonts/Avenir Next.ttc", 8),
    "mono": ("/System/Library/Fonts/Menlo.ttc", 0), "monob": ("/System/Library/Fonts/Menlo.ttc", 1),
    "serif": ("/System/Library/Fonts/Supplemental/Georgia.ttf", 0), "serifb": ("/System/Library/Fonts/Supplemental/Georgia Bold.ttf", 0),
    "serifi": ("/System/Library/Fonts/Supplemental/Georgia Italic.ttf", 0),
}
DEVA = ("/System/Library/Fonts/Kohinoor.ttc", {"sans": 1, "bold": 3, "heavy": 3})
KEYWORDS = set("def class return if elif else for while in not and or is None True False import from as with try except finally raise lambda yield pass break continue "
               "function const let var new this null undefined true false async await export default switch case throw catch typeof "
               "int float str bool void static public private fn mut impl struct enum match use pub self".split())
TOKEN = re.compile(r"(?P<com>#.*$|//.*$)|(?P<str>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')|(?P<num>\b\d[\d_.]*\b)|(?P<word>[A-Za-z_]\w*)|(?P<other>\s+|.)")


def is_deva(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", str(text)))


_SHAPED: dict = {}


def shaped(text: str, size: float, bold: bool, fill: tuple):
    """(picture, width) of a line of Hindi drawn by macOS, which joins the letters as they are written
    (the drawing library here places Devanagari letter by letter, which breaks every joined letter).
    None where macOS's text drawing is not there."""
    key = (text, int(size), bold, tuple(fill[:3]))
    if key in _SHAPED:
        return _SHAPED[key]
    try:
        import AppKit
        import Foundation
        from PIL import Image

        font = AppKit.NSFont.fontWithName_size_("KohinoorDevanagari-Bold" if bold else "KohinoorDevanagari-Medium", size) or AppKit.NSFont.systemFontOfSize_(size)
        colour = AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(fill[0] / 255, fill[1] / 255, fill[2] / 255, 1.0)
        line = AppKit.NSAttributedString.alloc().initWithString_attributes_(text, {AppKit.NSFontAttributeName: font, AppKit.NSForegroundColorAttributeName: colour})
        sz = line.size()
        w, h = max(1, int(sz.width) + 4), max(1, int(sz.height) + 2)
        rep = AppKit.NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
            None, w, h, 8, 4, True, False, AppKit.NSCalibratedRGBColorSpace, w * 4, 32)
        ctx = AppKit.NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep)
        AppKit.NSGraphicsContext.saveGraphicsState()
        AppKit.NSGraphicsContext.setCurrentContext_(ctx)
        line.drawAtPoint_(Foundation.NSMakePoint(1, 1))
        ctx.flushGraphics()
        AppKit.NSGraphicsContext.restoreGraphicsState()
        data = rep.bitmapData()
        got = (Image.frombuffer("RGBA", (w, h), bytes(data.tobytes() if hasattr(data, "tobytes") else data[: w * h * 4]), "raw", "RGBa", 0, 1).copy(), float(sz.width))
    except Exception:  # noqa: BLE001 (no macOS text drawing: the plain way is used)
        got = None
    if len(_SHAPED) > 400:
        _SHAPED.clear()
    _SHAPED[key] = got
    return got


class Draw:
    """The drawing surface of a picture. Hindi text goes through macOS (see `shaped`); everything else is the plain drawing library."""

    def __init__(self, img):
        from PIL import ImageDraw

        self.img, self.d = img, ImageDraw.Draw(img)

    def __getattr__(self, name):
        return getattr(self.d, name)

    @staticmethod
    def _how(font) -> tuple[float, bool]:
        return float(getattr(font, "size", 24)), "Kohinoor" in str(getattr(font, "path", "")) and getattr(font, "index", 0) == 3

    def textlength(self, text, font=None, **kw):
        got = shaped(str(text), *self._how(font), (0, 0, 0)) if is_deva(text) else None
        return got[1] if got else self.d.textlength(text, font=font, **kw)

    def text(self, xy, text, font=None, fill=None, stroke_width=0, stroke_fill=None, **kw):
        fill = tuple(fill[:3]) if isinstance(fill, (tuple, list)) else (255, 255, 255)
        got = shaped(str(text), *self._how(font), fill) if is_deva(text) else None
        if not got:
            return self.d.text(xy, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill, **kw)
        x, y = int(xy[0]) - 1, int(xy[1])
        put = (lambda im, at: self.img.alpha_composite(im, at)) if self.img.mode == "RGBA" else (lambda im, at: self.img.paste(im, at, im))
        if stroke_width:
            edge = shaped(str(text), *self._how(font), tuple((stroke_fill or (0, 0, 0))[:3]))[0]
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-0.7, -0.7), (0.7, -0.7), (-0.7, 0.7), (0.7, 0.7)):
                put(edge, (max(0, x + int(dx * stroke_width)), max(0, y + int(dy * stroke_width))))
        put(got[0], (max(0, x), max(0, y)))


class Pen:
    """One frame size and look: fonts, measures, the areas of the frame."""

    def __init__(self, W: int, H: int, look: str = "clean", captions: str | None = None):
        self.W, self.H = int(W), int(H)
        self.k = LOOKS.get(look, LOOKS["clean"])
        self.captions = captions if captions in ("sentence", "words", "none") else self.k["captions"]
        self.u = min(self.W, self.H) / 1080.0
        self.tall = self.H > self.W
        self.m = int(self.W * (0.055 if not self.tall else 0.06))
        self.top = int(self.H * (0.125 if not self.tall else 0.13))
        low = 0.06 if self.captions == "none" else (0.19 if not self.tall else 0.30)
        self.bottom = int(self.H * (1 - low))
        self._fonts: dict = {}

    def px(self, v: float) -> int:
        return max(1, int(round(v * self.u)))

    def font(self, kind: str, size: float, text: str = ""):
        from PIL import ImageFont

        path, index = FONTS.get(kind, FONTS["sans"])
        if text and is_deva(text) and kind in DEVA[1]:
            path, index = DEVA[0], DEVA[1][kind]
        key = (path, index, int(size))
        if key not in self._fonts:
            try:
                self._fonts[key] = ImageFont.truetype(path, max(8, int(size)), index=index) if Path(path).is_file() else ImageFont.load_default()
            except OSError:
                self._fonts[key] = ImageFont.load_default()
        return self._fonts[key]


def wrap(d, text: str, font, maxw: float) -> list[str]:
    out = []
    for para in str(text).split("\n"):
        cur = ""
        for w in para.split():
            trial = (cur + " " + w).strip()
            if not cur or d.textlength(trial, font=font) <= maxw:
                cur = trial
            else:
                out.append(cur)
                cur = w
        out.append(cur)
    return out or [""]


def fit_text(pen: Pen, d, text: str, kind: str, maxw: float, maxh: float, size: float, least: float, lh: float = 1.28):
    """(font, lines, line height): the largest size from `size` down at which the text fits the box."""
    s = size
    while True:
        f = pen.font(kind, s, text)
        lines = wrap(d, text, f, maxw)
        if (len(lines) * s * lh <= maxh and all(d.textlength(ln, font=f) <= maxw for ln in lines)) or s <= least:
            return f, lines, int(s * lh)
        s = max(least, s * 0.93)


def panel(d, box, fill, outline=None, r: int = 18, width: int = 2) -> None:
    d.rounded_rectangle([int(v) for v in box], radius=r, fill=fill, outline=outline, width=width)


def window(pen: Pen, d, box, title: str, fill, bar, ink) -> int:
    """A window with three dots and a title; returns the y where its content starts."""
    x0, y0, x1, y1 = box
    bh = pen.px(54)
    panel(d, box, fill, outline=bar, r=pen.px(18))
    d.rounded_rectangle([x0, y0, x1, y0 + bh], radius=pen.px(18), fill=bar)
    d.rectangle([x0, y0 + bh // 2, x1, y0 + bh], fill=bar)
    for i, c in enumerate(((236, 106, 94), (244, 191, 79), (98, 197, 84))):
        cx = x0 + pen.px(30) + i * pen.px(26)
        d.ellipse([cx - pen.px(8), y0 + bh // 2 - pen.px(8), cx + pen.px(8), y0 + bh // 2 + pen.px(8)], fill=c)
    if title:
        f = pen.font("sans", pen.px(24), title)
        t = title if d.textlength(title, font=f) < (x1 - x0) * 0.7 else title[: max(8, int(len(title) * (x1 - x0) * 0.7 / max(1.0, d.textlength(title, font=f)))) - 1] + "…"
        d.text(((x0 + x1) / 2 - d.textlength(t, font=f) / 2, y0 + bh / 2 - pen.px(15)), t, font=f, fill=ink)
    return y0 + bh


def header(pen: Pen, d, title: str) -> None:
    title = title.replace("`", "")
    if not title:
        return
    y = int(pen.H * (0.05 if not pen.tall else 0.06))
    s = pen.px(30)
    d.rectangle([pen.m, y + pen.px(4), pen.m + pen.px(10), y + s + pen.px(4)], fill=pen.k["accent"])
    f = pen.font("bold", s, title)
    t = title
    while len(t) > 8 and d.textlength(t, font=f) > pen.W - 2 * pen.m - pen.px(30):
        t = t[:-2].rstrip() + "…"
    d.text((pen.m + pen.px(24), y), t, font=f, fill=pen.k["soft"])


def credit(pen: Pen, d, fact: dict | None) -> None:
    if not fact or not fact.get("source"):
        return
    host = urlparse(str(fact.get("url", ""))).netloc.removeprefix("www.")
    text = f"Source: {fact['source']}" + (f"  ·  {host}" if host and host.lower() not in fact["source"].lower() else "")
    f = pen.font("sans", pen.px(24), text)
    while len(text) > 12 and d.textlength(text, font=f) > pen.W - 2 * pen.m:
        text = text[:-2].rstrip() + "…"
    d.text((pen.m, pen.bottom - pen.px(30)), text, font=f, fill=pen.k["soft"])


def code_lines(pen: Pen, d, box, lines: list[str], *, shown: int = 0, focus=(), band=None, ink=None, numbers: bool = True, syntax: dict | None = None,
               dim=None) -> None:
    """Code in a box: as large as fits, the `focus` lines on a band (the others dimmed), `shown` lines visible (0: all)."""
    x0, y0, x1, y1 = box
    lines = [str(ln).replace("\t", "    ").rstrip() for ln in lines][:40] or [""]
    n = len(lines)
    longest = max(len(ln) for ln in lines) + (4 if numbers else 0)
    size = min(pen.px(48 if n > 6 else 62), (x1 - x0) / max(12, longest) / 0.602, (y1 - y0) / (n * 1.42))
    size = max(pen.px(17), size)
    f = pen.font("mono", size)
    cw = d.textlength("M", font=f)
    lh = size * 1.42
    cols = max(8, int((x1 - x0) / cw) - (4 if numbers else 0))
    focus = {int(v) for v in focus if isinstance(v, (int, float))}
    for i, ln in enumerate(lines, 1):
        y = y0 + (i - 1) * lh
        if y + lh > y1 + 2 or (shown and i > shown):
            break
        if i in focus and band:
            d.rounded_rectangle([x0 - pen.px(10), y - 1, x1 + pen.px(10), y + lh - 2], radius=pen.px(6), fill=band)
        faded = bool(focus) and i not in focus
        x = x0
        if numbers:
            d.text((x, y + lh * 0.12), f"{i:>2}", font=f, fill=dim or pen.k["soft"])
            x += cw * 4
        text = ln if len(ln) <= cols else ln[: cols - 1] + "…"
        for m in TOKEN.finditer(text):
            kind, tok = m.lastgroup, m.group()
            colour = ink or pen.k["code_ink"]
            if syntax and kind in ("com", "str", "num"):
                colour = syntax[kind]
            elif syntax and kind == "word" and tok in KEYWORDS:
                colour = syntax["kw"]
            if faded:
                colour = tuple(int(c * 0.5 + b * 0.5) for c, b in zip(colour, pen.k["code_bg"]))
            if tok.strip():
                d.text((x, y + lh * 0.12), tok, font=f, fill=colour)
            x += cw * len(tok)


def term_lines(pen: Pen, d, box, rows: list[dict], shown: int = 0, focus=()) -> None:
    x0, y0, x1, y1 = box
    rows = rows[:30] or [{"t": "out", "text": ""}]
    longest = max(len(str(r.get("text", ""))) for r in rows)
    size = max(pen.px(17), min(pen.px(44), (x1 - x0) / max(12, longest) / 0.602, (y1 - y0) / (len(rows) * 1.45)))
    f = pen.font("mono", size)
    cw, lh = d.textlength("M", font=f), size * 1.45
    cols = max(8, int((x1 - x0) / cw))
    focus = {int(v) for v in focus if isinstance(v, (int, float))}
    for i, r in enumerate(rows, 1):
        y = y0 + (i - 1) * lh
        if y + lh > y1 + 2 or (shown and i > shown):
            break
        if i in focus:
            d.rounded_rectangle([x0 - pen.px(10), y - 1, x1 + pen.px(10), y + lh - 2], radius=pen.px(6), fill=(52, 58, 76))
        text = str(r.get("text", ""))
        d.text((x0, y + lh * 0.12), text if len(text) <= cols else text[: cols - 1] + "…", font=f, fill=TERM.get(r.get("t"), TERM["out"]))


def paragraph(pen: Pen, d, x: float, y: float, maxw: float, runs: list[tuple[str, bool, tuple]], font, lh: float, mark=None) -> tuple[float, list]:
    """Text in runs (text, marked, colour) wrapped as one paragraph. Marked words get the marker
    behind them when `mark` is a colour. Returns (the y below, the marked boxes line by line)."""
    words = [(w, flag, col) for text, flag, col in runs for w in str(text).split()]
    space = d.textlength(" ", font=font)
    rows, cur, width = [], [], 0.0
    for w, flag, col in words:
        ww = d.textlength(w, font=font)
        if cur and width + space + ww > maxw:
            rows.append(cur)
            cur, width = [], 0.0
        cur.append((w, flag, col, width + (space if cur else 0), ww))
        width += (space if len(cur) > 1 else 0) + ww
    if cur:
        rows.append(cur)
    marks = []
    for r, row in enumerate(rows):
        yy = y + r * lh
        flagged = [(wx, ww) for _, flag, _, wx, ww in row if flag]
        if flagged:
            box = [int(x + flagged[0][0] - pen.px(6)), int(yy + lh * 0.06), int(x + flagged[-1][0] + flagged[-1][1] + pen.px(6)), int(yy + lh * 0.94)]
            marks.append(box)
            if mark:
                d.rounded_rectangle(box, radius=pen.px(5), fill=mark)
        for w, _, col, wx, _ in row:
            d.text((x + wx, yy + lh * 0.1), w, font=font, fill=col)
    return y + len(rows) * lh, marks


def _clip(text: str, n: int, end: bool) -> str:
    """About `n` characters of text from its end (`end`) or start, cut at a word, with an ellipsis."""
    text = " ".join(str(text).split())
    if len(text) <= n:
        return text
    return ("… " + text[-n:].split(" ", 1)[-1]) if end else (text[:n].rsplit(" ", 1)[0] + " …")


# ---------------------------------------------------------------------------------------------
# the scenes
# ---------------------------------------------------------------------------------------------


def s_card(pen, d, sc, step, focus, fact):
    text, sub = str(sc.get("text", "")), str(sc.get("sub", ""))
    maxw, maxh = pen.W - 2 * pen.m - pen.px(40), (pen.bottom - pen.top) * (0.62 if sub else 0.8)
    f, lines, lh = fit_text(pen, d, text, "heavy" if pen.k["punch"] else "bold", maxw, maxh, pen.px(92 if len(text) < 60 else 68), pen.px(34))
    fs, subs, slh = fit_text(pen, d, sub, "sans", maxw, (pen.bottom - pen.top) * 0.25, pen.px(40), pen.px(24)) if sub else (None, [], 0)
    total = lh * len(lines) + (slh * len(subs) + pen.px(28) if sub else 0)
    y = pen.top + (pen.bottom - pen.top - total) / 2
    d.rectangle([pen.m, y + pen.px(6), pen.m + pen.px(12), y + lh * len(lines) - pen.px(6)], fill=pen.k["accent"])
    for ln in lines:
        d.text((pen.m + pen.px(40), y), ln, font=f, fill=pen.k["ink"])
        y += lh
    y += pen.px(28)
    for ln in subs:
        d.text((pen.m + pen.px(40), y), ln, font=fs, fill=pen.k["soft"])
        y += slh
    return []


def _title(pen, d, title: str) -> int:
    """A scene's own title line; returns the y below it."""
    if not title:
        return pen.top
    f, lines, lh = fit_text(pen, d, title, "bold", pen.W - 2 * pen.m, pen.px(120), pen.px(46), pen.px(30))
    for i, ln in enumerate(lines[:2]):
        d.text((pen.m, pen.top + i * lh), ln, font=f, fill=pen.k["ink"])
    return pen.top + lh * min(2, len(lines)) + pen.px(22)


def s_flow(pen, d, sc, step, focus, fact):
    boxes = [str(b) for b in sc.get("boxes") or []][:8] or ["…"]
    n = len(boxes)
    y0 = _title(pen, d, str(sc.get("title", "")))
    cols = 1 if pen.tall else min(n, 4)
    rows = -(-n // cols)
    gap = pen.px(54)
    bw = (pen.W - 2 * pen.m - gap * (cols - 1)) / cols
    bh = min(pen.px(230 if not pen.tall else 150), (pen.bottom - pen.px(40) - y0 - gap * (rows - 1)) / rows)
    top = y0 + max(0, (pen.bottom - pen.px(40) - y0 - (bh * rows + gap * (rows - 1))) / 2)
    lit_one = bool(step)
    step = n if not step else min(n, step)
    centres = []
    for i in range(n):
        r, c = divmod(i, cols)
        if r % 2:  # every second row runs back, so the arrow at a row's end goes straight down
            c = cols - 1 - c
        centres.append((pen.m + c * (bw + gap) + bw / 2, top + r * (bh + gap) + bh / 2))
    for i, text in enumerate(boxes):
        cx, cy = centres[i]
        box = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
        if i >= step:
            panel(d, box, pen.k["ghost"], r=pen.px(20))
            continue
        lit = lit_one and i == step - 1
        panel(d, box, pen.k["panel"], outline=pen.k["accent"] if lit else pen.k["line"], r=pen.px(20), width=pen.px(5 if lit else 2))
        d.text((box[0] + pen.px(16), box[1] + pen.px(10)), str(i + 1), font=pen.font("bold", pen.px(24)), fill=pen.k["accent"])
        f, lines, lh = fit_text(pen, d, text, "bold" if lit else "sans", bw - pen.px(44), bh - pen.px(50), pen.px(38), pen.px(20), 1.22)
        ty = cy - lh * len(lines) / 2 + pen.px(8)
        for ln in lines:
            d.text((cx - d.textlength(ln, font=f) / 2, ty), ln, font=f, fill=pen.k["ink"])
            ty += lh
        if i > 0:
            px_, py_ = centres[i - 1]
            a = pen.px(12)
            if abs(py_ - cy) < 2:  # same row
                s = 1 if cx > px_ else -1
                xa, xb = px_ + s * bw / 2 + s * pen.px(6), cx - s * bw / 2 - s * pen.px(6)
                d.line([xa, cy, xb, cy], fill=pen.k["soft"], width=pen.px(4))
                d.polygon([(xb, cy), (xb - s * a, cy - a), (xb - s * a, cy + a)], fill=pen.k["soft"])
            else:
                ya, yb = py_ + bh / 2 + pen.px(6), cy - bh / 2 - pen.px(6)
                d.line([cx, ya, cx, yb], fill=pen.k["soft"], width=pen.px(4))
                d.polygon([(cx, yb), (cx - a, yb - a), (cx + a, yb - a)], fill=pen.k["soft"])
    return []


def s_code(pen, d, sc, step, focus, fact):
    box = [pen.m, pen.top, pen.W - pen.m, pen.bottom - pen.px(20)]
    y = window(pen, d, box, str(sc.get("title", "")), pen.k["code_bg"], pen.k["line"], pen.k["soft"])
    code_lines(pen, d, [box[0] + pen.px(34), y + pen.px(26), box[2] - pen.px(34), box[3] - pen.px(20)], sc.get("lines") or [], shown=step or 0, focus=focus,
               band=pen.k["band"], syntax=pen.k["syntax"])
    return []


def s_terminal(pen, d, sc, step, focus, fact):
    box = [pen.m, pen.top, pen.W - pen.m, pen.bottom - pen.px(20)]
    y = window(pen, d, box, str(sc.get("title", "") or "Terminal"), TERM["bg"], (44, 48, 60), (170, 176, 190))
    term_lines(pen, d, [box[0] + pen.px(34), y + pen.px(26), box[2] - pen.px(34), box[3] - pen.px(20)], sc.get("lines") or [], shown=step or 0, focus=focus)
    return []


def s_debug(pen, d, sc, step, focus, fact):
    """1 the code and the failing run · 2 the cause pointed at · 3 the fix · 4 the passing run."""
    step = min(4, max(1, step or 4))
    fixed = step >= 3 and sc.get("fixed")
    code = sc.get("fixed") if fixed else sc.get("code") or []
    run = (sc.get("passes") if step >= 4 else sc.get("run")) or []
    gap = pen.px(26)
    if pen.tall:
        split = pen.top + (pen.bottom - pen.px(20) - pen.top) * 0.58
        a, b = [pen.m, pen.top, pen.W - pen.m, split - gap / 2], [pen.m, split + gap / 2, pen.W - pen.m, pen.bottom - pen.px(20)]
    else:
        split = pen.m + (pen.W - 2 * pen.m) * 0.56
        a, b = [pen.m, pen.top, split - gap / 2, pen.bottom - pen.px(20)], [split + gap / 2, pen.top, pen.W - pen.m, pen.bottom - pen.px(20)]
    tag, colour = {1: ("the run", pen.k["soft"]), 2: ("the cause", RED), 3: ("the fix", GREEN), 4: ("it passes", GREEN)}[step]
    y = window(pen, d, a, str(sc.get("title", "")), pen.k["code_bg"], pen.k["line"], pen.k["soft"])
    mix = lambda c: tuple(int(v * 0.22 + w * 0.78) for v, w in zip(c, pen.k["code_bg"]))  # noqa: E731
    lit = (sc.get("changed") or []) if fixed else (sc.get("cause") or []) if step == 2 else []
    code_lines(pen, d, [a[0] + pen.px(30), y + pen.px(24), a[2] - pen.px(30), a[3] - pen.px(60)], code, focus=lit, band=mix(GREEN if fixed else RED), syntax=pen.k["syntax"])
    f = pen.font("bold", pen.px(26))
    tw = d.textlength(tag, font=f)
    d.rounded_rectangle([a[2] - tw - pen.px(52), a[3] - pen.px(56), a[2] - pen.px(16), a[3] - pen.px(14)], radius=pen.px(21), fill=colour)
    d.text((a[2] - tw - pen.px(34), a[3] - pen.px(52)), tag, font=f, fill=(255, 255, 255))
    y = window(pen, d, b, "Run", TERM["bg"], (44, 48, 60), (170, 176, 190))
    rows = [{"t": r.get("t") if step != 3 else "out", "text": r.get("text", "")} for r in run]
    term_lines(pen, d, [b[0] + pen.px(28), y + pen.px(24), b[2] - pen.px(28), b[3] - pen.px(20)], rows)
    return []


def s_article(pen, d, sc, step, focus, fact):
    """The passage as it stands in the source page; from step 2 the sentence is marked."""
    fact = fact or {}
    ctx = fact.get("context") or {}
    exact = ctx.get("exact") or fact.get("quote") or str(sc.get("text", ""))
    url = str(fact.get("url", ""))
    paper, ink, grey = (255, 255, 252), (24, 24, 28), (118, 118, 124)
    box = [pen.m, pen.top, pen.W - pen.m, pen.bottom - pen.px(44)]
    shown = re.sub(r"^https?://(www\.)?", "", url).rstrip("/")
    y = window(pen, d, box, shown[:90], paper, (226, 224, 218), (96, 96, 104))
    x, maxw = box[0] + pen.px(56), box[2] - box[0] - pen.px(112)
    y += pen.px(30)
    d.text((x, y), str(fact.get("source", "")).upper()[:70], font=pen.font("bold", pen.px(22)), fill=pen.k["accent"] if sum(pen.k["accent"]) < 500 else (176, 52, 36))
    y += pen.px(40)
    title = str(ctx.get("title") or "")
    if title:
        tf, tl, tlh = fit_text(pen, d, title, "serifb", maxw, pen.px(110), pen.px(42), pen.px(28), 1.2)
        for ln in tl[:2]:
            d.text((x, y), ln, font=tf, fill=ink)
            y += tlh
    if fact.get("date"):
        d.text((x, y + pen.px(4)), str(fact["date"])[:40], font=pen.font("serifi", pen.px(24)), fill=grey)
        y += pen.px(36)
    y += pen.px(14)
    d.line([x, y, x + maxw, y], fill=(226, 224, 218), width=2)
    y += pen.px(22)
    room = box[3] - pen.px(28) - y
    marks: list = []
    for size, around in ((40, 260), (36, 240), (32, 220), (28, 180), (25, 120), (22, 60), (22, 0)):
        f = pen.font("serif", pen.px(size))
        lh = pen.px(size) * 1.5
        runs = [(_clip(ctx.get("before", ""), around, True), False, grey), (exact, True, ink), (_clip(ctx.get("after", ""), around, False), False, grey)]
        from PIL import Image

        probe = Draw(Image.new("RGB", (8, 8)))
        end, _ = paragraph(pen, probe, 0, 0, maxw, runs, f, lh)
        if end <= room or (size, around) == (22, 0):
            _, marks = paragraph(pen, d, x, y, maxw, runs, f, lh, mark=(255, 232, 96) if (step or 2) >= 2 else None)
            break
    return marks if (step or 2) >= 2 else []


def s_quote(pen, d, sc, step, focus, fact):
    fact = fact or {}
    text = (fact.get("context") or {}).get("exact") or fact.get("quote") or str(sc.get("text", ""))
    who = str(sc.get("who") or fact.get("source") or "")
    x = pen.m + pen.px(70)
    maxw = pen.W - pen.m - x
    d.text((pen.m - pen.px(6), pen.top - pen.px(40)), "“", font=pen.font("serifb", pen.px(190)), fill=pen.k["accent"])
    size = pen.px(64)
    while True:
        f = pen.font("serif", size, text)
        lh = size * 1.42
        if len(wrap(d, text, f, maxw)) * lh <= (pen.bottom - pen.top) * 0.66 or size <= pen.px(30):
            break
        size *= 0.93
    y = pen.top + max(pen.px(70), ((pen.bottom - pen.top) * 0.8 - len(wrap(d, text, f, maxw)) * lh) / 2)
    end, marks = paragraph(pen, d, x, y, maxw, [(text, True, pen.k["ink"])], f, lh, mark=pen.k["mark"] if (step or 1) >= 2 else None)
    if who:
        d.text((x, end + pen.px(26)), "— " + who[:90], font=pen.font("bold", pen.px(30), who), fill=pen.k["soft"])
    return marks if (step or 1) >= 2 else []


def s_example(pen, d, sc, step, focus, fact):
    rows = [str(r) for r in sc.get("lines") or []][:7] or [""]
    y = _title(pen, d, str(sc.get("title", "")))
    lit_one = bool(step)
    step = len(rows) if not step else min(len(rows), step)
    each = (pen.bottom - pen.px(30) - y) / len(rows)
    for i, text in enumerate(rows):
        if i >= step:
            break
        lit = lit_one and i == step - 1
        box = [pen.m, y + i * each, pen.W - pen.m, y + (i + 1) * each - pen.px(16)]
        panel(d, box, pen.k["panel"], outline=pen.k["accent"] if lit else pen.k["line"], r=pen.px(16), width=pen.px(4 if lit else 2))
        d.text((box[0] + pen.px(24), (box[1] + box[3]) / 2 - pen.px(24)), str(i + 1), font=pen.font("heavy", pen.px(40)), fill=pen.k["accent"])
        mono = bool(re.search(r"[=(){}\[\]]|^\s*(\$|>>>)", text))
        f, lines, lh = fit_text(pen, d, text, "mono" if mono else "sans", box[2] - box[0] - pen.px(130), box[3] - box[1] - pen.px(20), pen.px(40), pen.px(20), 1.25)
        ty = (box[1] + box[3]) / 2 - lh * len(lines) / 2
        for ln in lines:
            d.text((box[0] + pen.px(96), ty), ln, font=f, fill=pen.k["ink"])
            ty += lh
    return []


def s_compare(pen, d, sc, step, focus, fact):
    y = _title(pen, d, str(sc.get("title", "")))
    gap = pen.px(30)
    sides = [sc.get("left") or {}, sc.get("right") or {}]
    step = 2 if not step else min(2, step)
    for i, side in enumerate(sides):
        if pen.tall:
            h = (pen.bottom - pen.px(30) - y - gap) / 2
            box = [pen.m, y + i * (h + gap), pen.W - pen.m, y + i * (h + gap) + h]
        else:
            w = (pen.W - 2 * pen.m - gap) / 2
            box = [pen.m + i * (w + gap), y, pen.m + i * (w + gap) + w, pen.bottom - pen.px(30)]
        if i >= step:
            panel(d, box, pen.k["ghost"], r=pen.px(20))
            continue
        colour = (RED, GREEN)[i] if sc.get("good_bad") else pen.k["accent"] if i == step - 1 else pen.k["line"]
        panel(d, box, pen.k["panel"], outline=colour, r=pen.px(20), width=pen.px(4))
        head = str(side.get("title", ""))
        hf, hl, hlh = fit_text(pen, d, head, "bold", box[2] - box[0] - pen.px(60), pen.px(100), pen.px(40), pen.px(26), 1.2)
        ty = box[1] + pen.px(24)
        for ln in hl[:2]:
            d.text((box[0] + pen.px(30), ty), ln, font=hf, fill=colour if colour != pen.k["line"] else pen.k["ink"])
            ty += hlh
        ty += pen.px(14)
        rows = [str(r) for r in side.get("lines") or []][:8]
        if side.get("code"):
            code_lines(pen, d, [box[0] + pen.px(30), ty, box[2] - pen.px(30), box[3] - pen.px(20)], rows, numbers=False, syntax=pen.k["syntax"])
            continue
        body = "\n".join("•  " + r for r in rows)
        f, lines, lh = fit_text(pen, d, body, "sans", box[2] - box[0] - pen.px(60), box[3] - ty - pen.px(20), pen.px(36), pen.px(20), 1.3)
        for ln in lines:
            d.text((box[0] + pen.px(30), ty), ln, font=f, fill=pen.k["ink"])
            ty += lh
    return []


def s_chart(pen, d, sc, step, focus, fact):
    bars = []
    for b in sc.get("bars") or []:
        try:
            bars.append((str(b.get("label", "")), float(b.get("value"))))
        except (TypeError, ValueError, AttributeError):
            continue
    bars = bars[:8]
    y = _title(pen, d, str(sc.get("title", "")))
    if not bars:
        return []
    top = max(abs(v) for _, v in bars) or 1.0
    unit = str(sc.get("unit", ""))
    step = len(bars) if not step else min(len(bars), step)
    each = min(pen.px(120), (pen.bottom - pen.px(60) - y) / len(bars))
    lf = pen.font("sans", min(pen.px(32), each * 0.34))
    lw = min((pen.W - 2 * pen.m) * 0.34, max(d.textlength(lb, font=lf) for lb, _ in bars) + pen.px(24))
    x0, room = pen.m + lw, pen.W - 2 * pen.m - lw - pen.px(190)
    for i, (label, value) in enumerate(bars):
        if i >= step:
            break
        yy = y + i * each
        while len(label) > 4 and d.textlength(label, font=lf) > lw - pen.px(16):
            label = label[:-2] + "…"
        d.text((pen.m, yy + each * 0.3), label, font=lf, fill=pen.k["ink"])
        w = max(pen.px(6), room * abs(value) / top)
        d.rounded_rectangle([x0, yy + each * 0.18, x0 + w, yy + each * 0.82], radius=pen.px(8), fill=pen.k["accent"] if i == step - 1 else pen.k["soft"])
        d.text((x0 + w + pen.px(14), yy + each * 0.28), f"{value:g}{unit}", font=pen.font("bold", min(pen.px(34), each * 0.36)), fill=pen.k["ink"])
    return []


def s_timeline(pen, d, sc, step, focus, fact):
    pts = [p for p in sc.get("points") or [] if isinstance(p, dict)][:7]
    y = _title(pen, d, str(sc.get("title", "")))
    if not pts:
        return []
    step = len(pts) if not step else min(len(pts), step)
    each = (pen.bottom - pen.px(40) - y) / len(pts)
    x = pen.m + pen.px(26)
    d.line([x, y + each * 0.5, x, y + each * (step - 0.5)], fill=pen.k["line"], width=pen.px(4))
    for i, p in enumerate(pts[:step]):
        cy = y + each * (i + 0.5)
        lit = i == step - 1
        r = pen.px(15 if lit else 11)
        d.ellipse([x - r, cy - r, x + r, cy + r], fill=pen.k["accent"] if lit else pen.k["soft"])
        when = str(p.get("when", ""))
        wf = pen.font("heavy", min(pen.px(40), each * 0.34))
        d.text((x + pen.px(40), cy - each * 0.36), when, font=wf, fill=pen.k["accent"])
        f, lines, lh = fit_text(pen, d, str(p.get("what", "")), "sans", pen.W - pen.m - x - pen.px(50), each * 0.5, pen.px(34), pen.px(20), 1.25)
        ty = cy - each * 0.36 + min(pen.px(40), each * 0.34) * 1.3
        for ln in lines[:2]:
            d.text((x + pen.px(40), ty), ln, font=f, fill=pen.k["ink"])
            ty += lh
    return []


def s_footage(pen, d, sc, step, focus, fact):
    """A photo filling the picture area, what it is on a dark band, its credit underneath."""
    from PIL import Image, ImageOps

    box = [pen.m, pen.top, pen.W - pen.m, pen.bottom - pen.px(44)]
    w, h = int(box[2] - box[0]), int(box[3] - box[1])
    try:
        photo = ImageOps.fit(ImageOps.exif_transpose(Image.open(sc.get("file", ""))).convert("RGB"), (w, h), Image.LANCZOS, centering=(0.5, 0.4))
    except (OSError, ValueError):
        return s_card(pen, d, {"text": sc.get("caption") or sc.get("search", ""), "sub": ""}, step, focus, fact)
    mask = Image.new("L", (w, h), 0)
    Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=pen.px(22), fill=255)
    d.img.paste(photo, (int(box[0]), int(box[1])), mask)
    cap = str(sc.get("caption", ""))
    if cap:
        f, lines, lh = fit_text(pen, d, cap, "bold", w - pen.px(60), pen.px(130), pen.px(42), pen.px(26), 1.2)
        bh = lh * len(lines[:2]) + pen.px(28)
        band = Image.new("RGBA", (w, bh), (0, 0, 0, 150))
        d.img.paste(band, (int(box[0]), int(box[3]) - bh), Image.composite(band.getchannel("A"), Image.new("L", (w, bh), 0), mask.crop((0, h - bh, w, h))))
        for i, ln in enumerate(lines[:2]):
            d.text((box[0] + pen.px(30), box[3] - bh + pen.px(12) + i * lh), ln, font=f, fill=(255, 255, 255))
    credit_ = "Photo: " + str(sc.get("credit", ""))
    cf = pen.font("sans", pen.px(22), credit_)
    while len(credit_) > 12 and d.textlength(credit_, font=cf) > pen.W - 2 * pen.m:
        credit_ = credit_[:-2].rstrip() + "…"
    d.text((pen.m, pen.bottom - pen.px(30)), credit_, font=cf, fill=pen.k["soft"])
    return []


SCENES = {"footage": s_footage, "card": s_card, "flow": s_flow, "code": s_code, "terminal": s_terminal, "debug": s_debug, "article": s_article, "quote": s_quote, "example": s_example,
          "compare": s_compare, "chart": s_chart, "timeline": s_timeline}


def draw_scene(pen: Pen, scene: dict, step: int = 0, focus=(), *, title: str = "", fact: dict | None = None):
    """(the frame, the boxes of what was just marked). `step`: how far the scene has built up (0: all of it)."""
    from PIL import Image

    img = Image.new("RGB", (pen.W, pen.H), pen.k["bg"])
    d = Draw(img)
    header(pen, d, title)
    fn = SCENES.get(scene.get("type"), s_card)
    marks = fn(pen, d, scene, int(step or 0), focus or (), fact) or []
    if scene.get("type") in ("article", "quote", "chart") or (fact and scene.get("type") == "card"):
        credit(pen, d, fact)
    return img, marks


def draw_caption(pen: Pen, words: list[str], active: int):
    """(picture with see-through background, where its top-left goes) of a caption: a line in a
    dark box (`sentence`), or big words with the one being said in the accent colour (`words`)."""
    from PIL import Image

    text = " ".join(words)
    if not text or pen.captions == "none":
        return None, (0, 0)
    probe = Draw(Image.new("RGB", (8, 8)))
    big = pen.captions == "words"
    size = pen.px(74 if big else 40)
    f = pen.font("heavy" if big else "bold", size, text)
    maxw = pen.W - 2 * pen.m - pen.px(60)
    space = probe.textlength(" ", font=f)
    rows, cur, width = [], [], 0.0
    for i, w in enumerate(words):
        ww = probe.textlength(w, font=f)
        if cur and width + space + ww > maxw:
            rows.append((cur, width))
            cur, width = [], 0.0
        width += (space if cur else 0) + ww
        cur.append((i, w, ww))
    rows.append((cur, width))
    lh = int(size * 1.3)
    pad = pen.px(22)
    w, h = int(max(r[1] for r in rows) + 2 * pad), lh * len(rows) + 2 * pad
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = Draw(img)
    if not big:
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=pen.px(16), fill=(0, 0, 0, 170))
    for r, (row, rw) in enumerate(rows):
        x = (w - rw) / 2
        for i, word, ww in row:
            colour = pen.k["accent"] if big and i == active else (255, 255, 255)
            d.text((x, pad + r * lh), word, font=f, fill=colour, stroke_width=pen.px(6) if big else 0, stroke_fill=(0, 0, 0))
            x += ww + space
    zone = (pen.bottom + pen.H * (0.94 if not pen.tall else 0.86)) / 2  # (tall: the lowest part of a Short is covered by the app's own buttons)
    return img, (int((pen.W - w) / 2), int(min(pen.H - h - pen.px(10), max(pen.bottom + pen.px(6), zone - h / 2))))
