#!/usr/bin/env python3
"""topic_cards: the text screens of a topic video (a fact with its source), each drawn differently.

Six designs: a headline on a dark gradient, the source's page in a browser window with the key
words marked, a pull quote on paper, the fact's number drawn huge, a newspaper clipping lying on
a desk, and a news band. The design follows the fact (one with a number can become the number
card), the colours follow the source (the same publisher keeps its colour), and the cards of one
video never repeat a design before every other was used. Hindi is drawn with its letters joined.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import explain_draw as xd

DESIGNS = ("headline", "page", "quote", "number", "clipping", "band")
ACCENTS = ((255, 196, 61), (255, 107, 87), (64, 196, 160), (96, 176, 255), (214, 130, 255), (255, 140, 60), (120, 214, 90))
DARKS = ((18, 24, 48), (30, 16, 40), (12, 34, 38), (36, 20, 18), (20, 22, 28))
PAPERS = ((246, 243, 235), (240, 244, 248), (250, 244, 232), (238, 242, 236))
NUMBER = re.compile(r"(?:[₹$€£]\s?)?\d[\d,.]*\d|\d")
UNIT = re.compile(r"\s?(%|percent|per cent|crore|lakh|million|billion|trillion|km|kg|°C|minutes?|hours?|days?|years?|times|people|मिनट|घंटे|दिन|साल|प्रतिशत|करोड़|लाख|अरब|किमी|किलोमीटर)", re.I)


def _n(seed: str) -> int:
    return int(hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8], 16)


def site_of(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", url or "").split("/")[0]


def big_number(text: str) -> tuple[str, str] | None:
    """(the number with its unit, the whole sentence) when a fact has a number worth showing
    huge: one with a unit or a currency, or a large count; never a bare year, a day of the month or
    the number in a name, and not when the sentence is nothing but the number."""
    best = None
    for m in NUMBER.finditer(text):
        raw = m.group(0)
        unit = UNIT.match(text, m.end())
        digits = re.sub(r"\D", "", raw)
        named = m.start() > 0 and text[m.start() - 1] in "-–/"  # (Luna-25, 24/7: part of a name)
        if not unit and raw[0] not in "₹$€£" and (re.fullmatch(r"(1[5-9]|20)\d\d", digits) or len(digits) < 3 or named):
            continue
        end = unit.end() if unit else m.end()
        score = (2 if unit or raw[0] in "₹$€£" else 0) + min(len(digits), 6) / 6
        if best is None or score > best[0]:
            best = (score, m.start(), end)
    if best is None:
        return None
    num = text[best[1]:best[2]].strip()
    return (num, text) if len(text) - len(num) >= 12 and len(num) <= 14 else None


def key_phrase(text: str) -> tuple[str, str, str]:
    """(before, the words worth marking, after): the part with the number, or the longest clause."""
    m = NUMBER.search(text)
    words = text.split()
    if len(words) < 6:
        return "", text, ""
    if m:
        k = len(text[:m.start()].split())
        a, b = max(0, k - 2), min(len(words), k + 4)
    else:
        parts = [p for p in re.split(r"(?<=[,;:—–])\s", text) if p.strip()]
        best = max(parts, key=len) if len(parts) > 1 else " ".join(words[: max(4, len(words) // 2)])
        k = len(text[:text.find(best)].split())
        a, b = k, min(len(words), k + min(len(best.split()), 9))
    return " ".join(words[:a]), " ".join(words[a:b]), " ".join(words[b:])


def design_for(text: str, n: int, seed: str, used: list[str] | None = None) -> str:
    """Which design the n-th card of a video gets."""
    used = used or []
    order = [d for d in DESIGNS if d != "number"]
    k = _n(seed) % len(order)
    order = order[k:] + order[:k]
    if big_number(text) and "number" not in used[-2:]:
        return "number"
    unused = [d for d in order if d not in used]
    if unused:
        return unused[0]
    return min(order, key=lambda d: max(i for i, u in enumerate(used) if u == d))  # (all were used: the one longest ago)


def _gradient(img, top: tuple, bottom: tuple) -> None:
    from PIL import Image

    W, H = img.size
    col = Image.new("RGB", (1, H))
    col.putdata([tuple(int(top[c] + (bottom[c] - top[c]) * y / max(1, H - 1)) for c in range(3)) for y in range(H)])
    img.paste(col.resize((W, H)))


def _lines(d, x: float, y: float, lines: list[str], font, lh: int, fill: tuple) -> float:
    for ln in lines:
        d.text((x, y), ln, font=font, fill=fill)
        y += lh
    return y


def _headline(pen, img, d, text, source, site, accent, dark, paper):
    _gradient(img, tuple(min(255, c + 22) for c in dark), tuple(c // 2 for c in dark))
    x, maxw = pen.W * 0.09, pen.W * 0.82
    f, lines, lh = xd.fit_text(pen, d, text, "bold", maxw, pen.H * 0.56, pen.px(82 if len(text) < 70 else 64), pen.px(34), 1.25)
    y = (pen.H - lh * len(lines)) / 2 - lh / 2
    d.rectangle([x - pen.px(26), y + pen.px(6), x - pen.px(14), y + lh * len(lines)], fill=accent)
    y = _lines(d, x, y, lines, f, lh, (255, 255, 255))
    if source:
        d.text((x, y + lh * 0.4), f"Source: {source}", font=pen.font("sans", pen.px(34), source), fill=(184, 192, 212))


def _page(pen, img, d, text, source, site, accent, dark, paper):
    img.paste(tuple(max(0, c - 22) for c in paper), [0, 0, pen.W, pen.H])
    from PIL import Image

    probe = xd.Draw(Image.new("RGB", (8, 8)))
    before, key, after = key_phrase(text)
    runs = [(before, False, (24, 24, 28)), (key, True, (24, 24, 28)), (after, False, (24, 24, 28))]
    x0, x1 = pen.W * 0.07, pen.W * 0.93
    maxw = x1 - x0 - pen.px(112)
    head = pen.px(54 + 34) + (pen.px(78) if source else 0)
    most = pen.H * (0.62 if pen.tall else 0.8)
    for size in (66, 58, 50, 44, 38, 32, 27):  # (the window is as tall as what it says, in the middle of the frame)
        f = pen.font("serifb", pen.px(size), text)
        lh = pen.px(size) * 1.42
        body = xd.paragraph(pen, probe, 0, 0, maxw, runs, f, lh)[0]
        if head + body + pen.px(60) <= most or size == 27:
            break
    tall = min(most, head + body + pen.px(60))
    box = [x0, (pen.H - tall) / 2, x1, (pen.H + tall) / 2]
    y = xd.window(pen, d, box, (site or source)[:80], (255, 255, 252), (226, 224, 218), (96, 96, 104))
    x = box[0] + pen.px(56)
    y += pen.px(34)
    ink_accent = accent if sum(accent) < 520 else tuple(int(c * 0.62) for c in accent)
    if source:
        d.text((x, y), source.upper()[:60], font=pen.font("bold", pen.px(26), source), fill=ink_accent)
        y += pen.px(48)
        d.line([x, y, x + maxw, y], fill=(226, 224, 218), width=max(2, pen.px(2)))
        y += pen.px(30)
    xd.paragraph(pen, d, x, y, maxw, runs, f, lh, mark=tuple(int(255 - (255 - c) * 0.55) for c in accent))


def _quote(pen, img, d, text, source, site, accent, dark, paper):
    img.paste(paper, [0, 0, pen.W, pen.H])
    ink_accent = accent if sum(accent) < 520 else tuple(int(c * 0.62) for c in accent)
    x, maxw = pen.W * 0.12, pen.W * 0.76
    f, lines, lh = xd.fit_text(pen, d, text, "serif", maxw, pen.H * 0.5, pen.px(70), pen.px(32), 1.36)
    y = (pen.H - lh * len(lines)) / 2 - pen.px(30)
    d.text((x - pen.px(10), y - pen.px(190)), "“", font=pen.font("serifb", pen.px(260)), fill=ink_accent)
    y = _lines(d, x, y, lines, f, lh, (28, 28, 34))
    if source:
        d.line([x, y + pen.px(34), x + pen.px(70), y + pen.px(34)], fill=ink_accent, width=max(3, pen.px(4)))
        d.text((x + pen.px(90), y + pen.px(12)), source[:60], font=pen.font("bold", pen.px(34), source), fill=(90, 90, 98))


def _number(pen, img, d, text, source, site, accent, dark, paper):
    num, rest = big_number(text) or ("", text)
    _gradient(img, dark, tuple(c // 3 for c in dark))
    x, maxw = pen.W * 0.09, pen.W * 0.82
    num = num.replace("₹", "Rs ")  # (the heavy face has no rupee sign)
    nf, nl, nlh = xd.fit_text(pen, d, num, "heavy", maxw, pen.H * 0.34, pen.px(300), pen.px(90), 1.0)
    f, lines, lh = xd.fit_text(pen, d, rest, "bold", maxw, pen.H * 0.3, pen.px(56), pen.px(30), 1.3)
    y = (pen.H - nlh * len(nl) - lh * len(lines) - pen.px(60)) / 2
    y = _lines(d, x, y, nl, nf, nlh, accent) + pen.px(26)
    d.rectangle([x, y, x + pen.px(120), y + pen.px(8)], fill=accent)
    y = _lines(d, x, y + pen.px(34), lines, f, lh, (240, 242, 248))
    if source:
        d.text((x, y + lh * 0.35), f"Source: {source}", font=pen.font("sans", pen.px(32), source), fill=(170, 178, 198))


def _clipping(pen, img, d, text, source, site, accent, dark, paper):
    from PIL import Image

    _gradient(img, tuple(int(c * 1.5) + 14 for c in dark), dark)
    w, h = int(pen.W * (0.8 if pen.tall else 0.66)), int(pen.H * (0.5 if pen.tall else 0.7))
    sheet = Image.new("RGBA", (w, h), (252, 250, 243, 255))
    sd = xd.Draw(sheet)
    pad = pen.px(48)
    name = (source or site or "NEWS").upper()[:40]
    mf, ml, mlh = xd.fit_text(pen, sd, name, "serifb", w - 2 * pad, pen.px(90), pen.px(54), pen.px(26), 1.1)
    sd.text(((w - sd.textlength(ml[0], font=mf)) / 2, pad * 0.7), ml[0], font=mf, fill=(22, 22, 26))
    y = pad * 0.7 + mlh + pen.px(10)
    sd.line([pad, y, w - pad, y], fill=(22, 22, 26), width=max(3, pen.px(4)))
    sd.line([pad, y + pen.px(9), w - pad, y + pen.px(9)], fill=(22, 22, 26), width=1)
    y += pen.px(34)
    f, lines, lh = xd.fit_text(pen, sd, text, "serifb", w - 2 * pad, h - y - pad * 1.6, pen.px(58), pen.px(26), 1.3)
    y = _lines(sd, pad, y, lines, f, lh, (22, 22, 26))
    for k in range(3):  # (the grey lines of the article's body, running off the cut edge)
        yy = y + pen.px(26) + k * pen.px(22)
        if yy < h - pad * 0.6:
            sd.line([pad, yy, w - pad - (pen.px(160) if k == 2 else 0), yy], fill=(206, 202, 192), width=max(3, pen.px(6)))
    tilt = -2.2 if _n(text) % 2 else 1.8
    turned = sheet.rotate(tilt, expand=True, resample=Image.BICUBIC)
    shadow = Image.new("RGBA", turned.size, (0, 0, 0, 0))
    shadow.paste((0, 0, 0, 110), mask=turned.getchannel("A"))
    px, py = (pen.W - turned.width) // 2, (pen.H - turned.height) // 2
    img.paste(shadow, (px + pen.px(14), py + pen.px(18)), shadow)
    img.paste(turned, (px, py), turned)
    d.rectangle([px + turned.width * 0.42, py - pen.px(8), px + turned.width * 0.58, py + pen.px(30)], fill=tuple(int(255 - (255 - c) * 0.6) for c in accent))


def _band(pen, img, d, text, source, site, accent, dark, paper):
    _gradient(img, tuple(c // 2 + 6 for c in dark), tuple(min(255, c + 16) for c in dark))
    x, maxw = pen.W * 0.08, pen.W * 0.84
    on_accent = (16, 16, 20) if sum(accent) > 380 else (255, 255, 255)
    label = (source or site or "REPORT").upper()[:36]
    lf = pen.font("heavy", pen.px(34), label)
    lw = d.textlength(label, font=lf) + pen.px(44)
    f, lines, lh = xd.fit_text(pen, d, text, "bold", maxw - pen.px(40), pen.H * 0.42, pen.px(68), pen.px(32), 1.28)
    top = (pen.H - lh * len(lines)) / 2 - pen.px(50)
    d.rectangle([x, top - pen.px(64), x + lw, top - pen.px(6)], fill=accent)
    d.text((x + pen.px(22), top - pen.px(58)), label, font=lf, fill=on_accent)
    d.rectangle([x, top - pen.px(6), x + maxw, top + lh * len(lines) + pen.px(40)], fill=(250, 250, 252))
    d.rectangle([x, top - pen.px(6), x + pen.px(12), top + lh * len(lines) + pen.px(40)], fill=accent)
    _lines(d, x + pen.px(40), top + pen.px(14), lines, f, lh, (18, 18, 24))
    if site:
        d.text((x, top + lh * len(lines) + pen.px(58)), site[:60], font=pen.font("sans", pen.px(28)), fill=(176, 184, 204))


DRAW = {"headline": _headline, "page": _page, "quote": _quote, "number": _number, "clipping": _clipping, "band": _band}


def make_card(text: str, source: str, dst: Path, W: int, H: int, *, url: str = "", design: str | None = None, seed: str = "") -> str:
    """Draws one text screen and saves it. Returns the design used. The colours come from the
    source's name, so one publisher looks the same through a video and two publishers do not."""
    from PIL import Image

    text, source = " ".join(str(text).replace("₹", "Rs ").split()), " ".join(str(source or "").split())
    design = design if design in DRAW else design_for(text, 0, seed or text)
    if design == "number" and not big_number(text):
        design = "headline"
    who = _n(source or site_of(url) or seed or text)
    pen = xd.Pen(W, H, "clean", "none")
    img = Image.new("RGB", (W, H))
    d = xd.Draw(img)
    DRAW[design](pen, img, d, text, source, site_of(url), ACCENTS[who % len(ACCENTS)], DARKS[(who // 7) % len(DARKS)], PAPERS[(who // 5) % len(PAPERS)])
    img.save(dst)
    return design
