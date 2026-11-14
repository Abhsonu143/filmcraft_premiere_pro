"""thumbnails: thumbnails, titles, hooks and hashtags for a video, for YouTube and Instagram, in
English and Hindi ("packaging": what makes someone stop scrolling and click).

    python thumbnails.py --clips CLIP... --topic "GTA 6 will only be 30 FPS" [--hook "…" | --find-hook]
                         [--platforms youtube shorts instagram reels] [--languages en hi] [--variants 3]

What it makes, in `<out>/<topic> - package/`:
  - thumbnails/<platform>-<language>-<n>.jpg: the best frames of the clips (sharp, a big face with
    an emotion or the most striking moment, on topic), cropped for the platform with the subject
    on a third, made to pop (colour, contrast, a darker blurred background, vignette), with 2–4 huge
    words in a heavy outlined font placed away from faces and out of the platform's buttons and
    badges (YouTube's duration badge, the Shorts/Reels captions, the Instagram grid crop). n = 1 is
    the strongest; the variants differ in frame, words and style so you can choose (or A/B test with
    YouTube's Test & Compare).
  - preview.jpg: every thumbnail at the size people really see it (home feed, phone).
  - <platform>-<language>.txt: titles (best first), the thumbnail words, the opening hook, the
    description or caption, and hashtags; hashtags-youtube.txt and hashtags-instagram.txt with both
    languages; package.json with everything (for posting).

The words are written by Claude (from your hook or one found online with --find-hook, the topic's
most-viewed titles and your hook library) following what makes packaging work: a curiosity gap the
video really closes, the title and the thumbnail words telling one story without repeating each
other, the keyword first, short. Hindi is natural spoken Hindi in Devanagari (names like GTA 6 stay
as they are). Without Claude, simple rules make a usable set.

Fonts: the Mac's own (Impact, Arial Black; Kohinoor Devanagari, ITF Devanagari for Hindi), drawn by
libass (ffmpeg), which shapes Hindi correctly.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import autocut as ac
from autocut import AutocutError, log

PLATFORMS: dict[str, dict] = {
    # safe: where words may go (fractions of the picture); avoid: covered by the app's own badges
    "youtube": {"label": "YouTube video", "size": (1280, 720), "safe": (0.04, 0.05, 0.96, 0.95), "avoid": [(0.78, 0.80, 1.0, 1.0)]},
    "shorts": {"label": "YouTube Shorts", "size": (1080, 1920), "safe": (0.06, 0.10, 0.94, 0.66), "avoid": [(0.0, 0.70, 1.0, 1.0)]},
    "instagram": {"label": "Instagram video (feed)", "size": (1080, 1350), "safe": (0.06, 0.06, 0.94, 0.90), "avoid": []},
    # Reels: the profile grid shows the middle 3:4, and the caption covers the bottom
    "reels": {"label": "Instagram Reels", "size": (1080, 1920), "safe": (0.06, 0.15, 0.94, 0.66), "avoid": [(0.0, 0.72, 1.0, 1.0)]},
}
LANGUAGES = {"en": "English", "hi": "Hindi"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tif", ".tiff", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mts", ".m2ts"}
FONTS = {"en": ["Impact", "Arial Black"], "hi": ["Kohinoor Devanagari", "ITF Devanagari"]}
STYLES = ("pop", "box", "glow")
YELLOW, RED, WHITE, BLACK = (255, 230, 0), (230, 30, 40), (255, 255, 255), (0, 0, 0)

# what CLIP looks for in a thumbnail frame (and against)
GOOD = ["an eye-catching YouTube thumbnail", "a close-up of a face with a strong emotion, shocked, excited or angry",
        "an intense dramatic action moment", "a vivid colourful cinematic shot, sharp and well lit"]
BAD = ["a blurry frame", "a dark underexposed frame", "a frame full of on-screen text, subtitles or a menu",
       "a black screen, a logo or a title card"]


# ---------------------------------------------------------------------------------------------
# Choosing frames
# ---------------------------------------------------------------------------------------------


def collect(items: list[str]) -> list[str]:
    out: list[str] = []
    for it in items:
        p = Path(it).expanduser()
        if p.is_dir():
            out += sorted(str(f.resolve()) for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT | IMAGE_EXT and not f.name.startswith("."))
        elif p.is_file():
            out.append(str(p.resolve()))
        else:
            raise AutocutError(f"not found: {it}")
    seen: set[str] = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def zscore(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-6) if len(a) else a


def frame_at(src: str, t: float, dst: Path, width: int | None = 960) -> bool:
    vf = ["-vf", f"scale={width}:-2"] if width else []
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    cmd += (["-ss", f"{max(0.0, t):.3f}"] if Path(src).suffix.lower() not in IMAGE_EXT else []) + ["-i", src, "-frames:v", "1"] + vf + ["-q:v", "2", str(dst)]
    return subprocess.run(cmd, capture_output=True).returncode == 0 and dst.is_file()


def measures(path: Path) -> dict:
    """Sharpness, brightness, colourfulness and a tiny fingerprint (for telling frames apart)."""
    import cv2

    im = cv2.imread(str(path))
    if im is None:
        return {"sharp": 0.0, "bright": 0.0, "colour": 0.0, "print": np.zeros(64)}
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    b, gr, r = (im[..., i].astype(np.float32) for i in range(3))
    rg, yb = r - gr, 0.5 * (r + gr) - b
    colour = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    small = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return {"sharp": float(cv2.Laplacian(g, cv2.CV_64F).var()), "bright": float(g.mean()), "colour": colour,
            "print": (small - small.mean()) / (small.std() + 1e-6)}


def detect(paths: list[Path]) -> list[dict]:
    """Faces, people, where the eye goes and on-screen text (Apple Vision), per image; empty where
    unavailable."""
    empty = {"faces": [], "humans": [], "objects": [], "attention": [], "text": []}
    try:
        import reframe

        out = reframe._vision_boxes([str(p) for p in paths])
    except Exception as e:  # (Vision is macOS only; frames are still chosen without it)
        log(f"no face detection ({e.__class__.__name__}): frames are chosen by look and topic only")
        return [dict(empty) for _ in paths]
    try:
        for d, (t, yaws) in zip(out, text_and_turn(paths)):
            d["text"] = t
            d["yaw"] = yaws
    except Exception:  # (older macOS: no text or direction check)
        for d in out:
            d.setdefault("text", [])
            d.setdefault("yaw", [])
    return out


def text_and_turn(paths: list[Path]) -> list[tuple[list[tuple], list[tuple[float, float]]]]:
    """Words already in the picture (subtitles, menus, logos) as boxes in picture fractions, and how
    far each face turns from the camera ((area, |yaw| in radians), biggest first)."""
    import objc
    import Vision
    from Foundation import NSURL

    res = []
    for p in paths:
        with objc.autorelease_pool():
            h = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(str(p)), None)
            rq = Vision.VNRecognizeTextRequest.alloc().init()
            rq.setRecognitionLevel_(1)  # fast
            fq = Vision.VNDetectFaceRectanglesRequest.alloc().init()
            try:
                fq.setRevision_(3)  # (gives yaw)
            except Exception:
                pass
            ok = h.performRequests_error_([rq, fq], None)
            boxes, yaws = [], []
            if ok[0] if isinstance(ok, tuple) else ok:
                for r in rq.results() or []:
                    b = r.boundingBox()
                    if float(r.confidence()) >= 0.4:
                        boxes.append((b.origin.x, 1 - b.origin.y - b.size.height, b.origin.x + b.size.width, 1 - b.origin.y))
                for r in fq.results() or []:
                    b = r.boundingBox()
                    y = r.yaw()
                    yaws.append((float(b.size.width * b.size.height), abs(float(y.floatValue())) if y is not None else 0.0))
            yaws.sort(key=lambda x: -x[0])
            res.append((boxes, yaws))
    return res


def pick_frames(paths: list[str], topic: str, n: int, work: Path) -> list[dict]:
    """The n best thumbnail frames: [{src, t, path (full size), det, score}], different from each other."""
    import voiceover_cut as vc

    videos = [p for p in paths if Path(p).suffix.lower() not in IMAGE_EXT]
    images = [p for p in paths if Path(p).suffix.lower() in IMAGE_EXT]
    cands: list[dict] = [{"src": p, "t": 0.0, "clip": 0.0} for p in images]
    if videos:
        texts = [topic[:200] or "a video"] + GOOD + BAD
        samples, sims = vc.clip_similarities(videos, texts)
        if samples:
            s = np.asarray(sims, dtype=np.float64)
            on_topic = zscore(s[0]) if topic else np.zeros(s.shape[1])
            good = zscore(s[1:1 + len(GOOD)].mean(axis=0))
            bad = zscore(s[1 + len(GOOD):].max(axis=0))
            score = 1.0 * on_topic + 0.9 * good - 0.8 * bad
            face_like = zscore(s[2])  # "a close-up of a face with a strong emotion"
            dur = {v: max(0.0, ac.probe(v).duration) for v in videos}
            # the best overall, with the most face-like moments always among them
            top_faces = list(np.argsort(-(face_like - 0.5 * bad))[: max(20, 6 * n)])
            order = top_faces[:: 2] + [k for k in np.argsort(-score) if k not in set(top_faces[:: 2])]
            per_clip: dict[int, list[float]] = {}
            for k in order:
                ci, t = samples[int(k)]
                src = videos[ci]
                if t < 0.4 or t > dur[src] - 0.4:
                    continue  # (fades in and out)
                ts = per_clip.setdefault(ci, [])
                if len(ts) >= 4 or any(abs(t - u) < 3.0 for u in ts):
                    continue
                ts.append(t)
                cands.append({"src": src, "t": float(t) + 0.5, "clip": float(score[int(k)])})
                if len(cands) >= max(70, 14 * n):
                    break
    if not cands:
        raise AutocutError("no frames to choose from (give video clips or images)")
    log(f"looking closely at {len(cands)} candidate frames")
    keep = []
    for i, c in enumerate(cands):
        f = work / f"cand-{i:03d}.jpg"
        if frame_at(c["src"], c["t"], f):
            c["small"] = f
            keep.append(c)
    dets = detect([c["small"] for c in keep])
    for c, d in zip(keep, dets):
        c["det"] = d
        c.update(measures(c["small"]))
    if not keep:
        raise AutocutError("could not read frames from the clips")
    sharp = zscore([np.log1p(c["sharp"]) for c in keep])
    colour = zscore([c["colour"] for c in keep])
    for c, sh, co in zip(keep, sharp, colour):
        faces = sorted(c["det"]["faces"], key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
        face = 0.0
        if faces:
            area = (faces[0][2] - faces[0][0]) * (faces[0][3] - faces[0][1])
            # a big, clear face is what makes people click; one or two read best
            face = 2.4 * min(1.0, area / 0.03) + 1.4 * min(1.0, area / 0.10) + (0.3 if len(faces) <= 2 else -0.2)
            yaw = c["det"].get("yaw") or []
            if yaw:  # a face turned away (a profile) says much less than one looking at us
                face *= 1.0 - 0.6 * min(1.0, yaw[0][1] / 0.9)
        text = sum((b[2] - b[0]) * (b[3] - b[1]) for b in c["det"].get("text", []))
        c["has_text"] = text > 0.002
        dark = max(0.0, (75 - c["bright"]) / 60) + max(0.0, (c["bright"] - 215) / 40)
        c["score"] = 0.5 * c["clip"] + 0.6 * sh + 0.35 * co + face - 1.5 * dark - (3.0 if c["has_text"] else 0.0) - 20 * text
    keep.sort(key=lambda c: -c["score"])
    chosen: list[dict] = []
    for c in keep:
        if any(np.dot(c["print"], o["print"]) / 64 > 0.85 for o in chosen):
            continue  # (nearly the same picture)
        chosen.append(c)
        if len(chosen) >= n:
            break
    for i, c in enumerate(chosen):
        full = work / f"frame-{i}.png"
        if not frame_at(c["src"], c["t"], full, width=None):
            full = c["small"]
        c["path"] = full
        log(f"frame {i + 1}: {Path(c['src']).name[:60]} at {c['t']:.1f}s" + (f", {len(c['det']['faces'])} face(s)" if c["det"]["faces"] else ""))
    return chosen


# ---------------------------------------------------------------------------------------------
# Composing a thumbnail
# ---------------------------------------------------------------------------------------------


def subject_box(det: dict) -> tuple[float, float, float, float] | None:
    """What the thumbnail is about, in picture fractions: the biggest face with its head and
    shoulders, else a person, else where the eye goes."""
    faces = sorted(det.get("faces", []), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    if faces:
        x0, y0, x1, y1 = faces[0][:4]
        w, h = x1 - x0, y1 - y0
        return (max(0.0, x0 - 0.6 * w), max(0.0, y0 - 0.45 * h), min(1.0, x1 + 0.6 * w), min(1.0, y1 + 1.2 * h))
    for key in ("humans", "objects", "attention"):
        bs = sorted(det.get(key, []), key=lambda b: -b[4] * (b[2] - b[0]) * (b[3] - b[1]))
        if bs:
            return tuple(float(v) for v in bs[0][:4])  # type: ignore[return-value]
    return None


def crop_window(iw: int, ih: int, W: int, H: int, subj, text_side: str) -> tuple[int, int, int, int]:
    """The crop (x, y, w, h) of the source for a W×H thumbnail: as large as fits, zoomed in on a small
    subject, with the subject on the third away from the words."""
    aspect = W / H
    cw, ch = (iw, int(iw / aspect)) if iw / ih < aspect else (int(ih * aspect), ih)
    if subj:
        sw_, sh_ = (subj[2] - subj[0]) * iw, (subj[3] - subj[1]) * ih
        # a small subject: zoom so it fills about half the height (never past 1.6x, never below the output size)
        z = min(1.6, max(1.0, 0.55 * ch / max(1.0, sh_)))
        z = min(z, max(1.0, cw / (W * 0.75)))  # (at most a mild enlargement)
        cw, ch = int(cw / z), int(ch / z)
        sx, sy = (subj[0] + subj[2]) / 2 * iw, (subj[1] + subj[3]) / 2 * ih
        if aspect >= 1:
            fx = {"left": 0.66, "right": 0.34}.get(text_side, 0.5)
            fy = 0.45
        else:
            fx, fy = 0.5, {"top": 0.62, "bottom": 0.38}.get(text_side, 0.5)
        x, y = sx - fx * cw, sy - fy * ch
    else:
        x, y = (iw - cw) / 2, (ih - ch) / 2
    x = int(min(max(0, x), iw - cw))
    y = int(min(max(0, y), ih - ch))
    return x, y, cw, ch


def big_face(det: dict, iw: int, ih: int, W: int, H: int) -> bool:
    """A face big enough to carry a vertical crop on its own (from a landscape frame)."""
    faces = det.get("faces", [])
    return any((b[3] - b[1]) * ih >= 0.22 * ih for b in faces)


def framed_layout(src, subj, platform: str):
    """A vertical cover from a landscape frame without enlarging it: the frame big in the middle
    (a little zoomed on the subject) over a blurred, darkened copy of itself, with the space above
    it for the words. Returns (image, subject box in pixels, text region)."""
    from PIL import Image, ImageEnhance, ImageFilter

    W, H = PLATFORMS[platform]["size"]
    iw, ih = src.size
    # the background: the frame filling the picture, blurred and dark
    k = max(W / iw, H / ih)
    bg = src.resize((int(iw * k) + 1, int(ih * k) + 1), Image.LANCZOS)
    bx, by = (bg.width - W) // 2, (bg.height - H) // 2
    bg = bg.crop((bx, by, bx + W, by + H)).filter(ImageFilter.GaussianBlur(W / 28))
    bg = ImageEnhance.Brightness(bg).enhance(0.45)
    # the frame: about half the height, at the bottom; the words get the space above it
    fa = min(iw / ih, max(1.0, W / (0.56 * H)))
    fw, fh = (int(ih * fa), ih) if iw / ih > fa else (iw, int(iw / fa))
    cx = ((subj[0] + subj[2]) / 2 * iw) if subj else iw / 2
    cy = ((subj[1] + subj[3]) / 2 * ih) if subj else ih / 2
    x = int(min(max(0, cx - fw / 2), iw - fw))
    y = int(min(max(0, cy - fh / 2), ih - fh))
    fg = src.crop((x, y, x + fw, y + fh)).resize((W, int(W / fa)), Image.LANCZOS)
    s = PLATFORMS[platform]["safe"]
    top = H - int(W / fa) - int(0.035 * H)
    fg = enhance(fg, None, "none")
    bg.paste(fg, (0, top))
    img = bg
    subj_px = None
    if subj:
        sx0, sy0 = (subj[0] * iw - x) / fw * W, (subj[1] * ih - y) / fh * fg.height + top
        sx1, sy1 = (subj[2] * iw - x) / fw * W, (subj[3] * ih - y) / fh * fg.height + top
        subj_px = (int(sx0), int(sy0), int(sx1), int(sy1))
    region = (int(s[0] * W), int(s[1] * H), int(s[2] * W), top - int(0.015 * H))
    return img, subj_px, region


def text_side(subj, W: int, H: int) -> str:
    if W >= H:
        return "left" if not subj or (subj[0] + subj[2]) / 2 >= 0.5 else "right"
    return "top"


def enhance(img, subj_px: tuple[int, int, int, int] | None, side: str):
    """Make it pop: colour and contrast, a crisp subject on a slightly darker, softer background, a
    vignette, and a shade behind the words."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

    W, H = img.size
    img = ImageOps.autocontrast(img, cutoff=0.6)
    img = ImageEnhance.Color(img).enhance(1.3)
    img = ImageEnhance.Contrast(img).enhance(1.12)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=90, threshold=2))
    if subj_px:
        # depth: the background a little blurred and darker, the subject untouched
        x0, y0, x1, y1 = subj_px
        mask = Image.new("L", (W, H), 0)
        pad_x, pad_y = (x1 - x0) * 0.25, (y1 - y0) * 0.2
        ImageDraw.Draw(mask).ellipse((x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(max(W, H) / 30))
        back = ImageEnhance.Brightness(img.filter(ImageFilter.GaussianBlur(max(W, H) / 500))).enhance(0.82)
        img = Image.composite(img, back, mask)
    # vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
    vig = np.clip(1.0 - 0.28 * np.clip(r - 0.55, 0, None) ** 1.6, 0.0, 1.0)
    # shade behind the words: a soft dark gradient on their side
    if side == "left":
        shade = np.clip(1 - xx / (W * 0.62), 0, 1)
    elif side == "right":
        shade = np.clip((xx - W * 0.38) / (W * 0.62), 0, 1)
    elif side == "top":
        shade = np.clip(1 - yy / (H * 0.55), 0, 1)
    elif side == "bottom":
        shade = np.clip((yy - H * 0.45) / (H * 0.55), 0, 1)
    else:
        shade = np.zeros_like(xx)
    k = vig * (1 - 0.42 * shade**1.3)
    arr = (np.asarray(img, dtype=np.float32) * k[..., None]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def lines_of(text: str, max_lines: int = 3) -> list[str]:
    """Balanced lines: short words together, at most max_lines, the longest line as short as possible."""
    words_ = text.split()
    if len(words_) <= 1:
        return words_
    best, best_w = [text], len(text)
    n = len(words_)
    for k in range(1, min(max_lines, n)):
        # all ways to cut n words into k+1 lines
        import itertools

        for cuts in itertools.combinations(range(1, n), k):
            parts = [" ".join(words_[a:b]) for a, b in zip((0,) + cuts, cuts + (n,))]
            width = max(len(p) for p in parts) + 1.5 * k  # (each extra line costs a little)
            if width < best_w:
                best, best_w = parts, width
    return best


def ass_color(rgb: tuple[int, int, int], alpha: int = 0) -> str:
    r, g, b = rgb
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def ass_escape(t: str) -> str:
    return t.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def strip_unsupported(t: str) -> str:
    """Thumbnail words without emoji and symbols the heavy fonts lack."""
    t = re.sub(r"[\U0001F000-\U0001FFFF☀-➿️‍]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def render_text(lines: list[str], highlight: str, lang: str, style: str, size: tuple[int, int], fs: float, work: Path, tag: str):
    """The words drawn on a transparent W×H layer (libass via ffmpeg): an RGBA image."""
    from PIL import Image

    W, H = size
    font = FONTS["hi" if lang == "hi" else "en"][0 if style != "box" or lang == "hi" else 1]
    bold = 1 if lang == "hi" else 0
    hl = {w.lower() for w in highlight.split()} if highlight else set()
    outline = max(4, round(fs * 0.085))
    shadow = max(3, round(fs * 0.05))
    events = []
    lh = fs * (1.18 if lang == "hi" else 1.02)
    for i, line in enumerate(lines):
        y = 10 + i * lh
        if style == "box":
            # each line on its own colour slab (the highlighted one red, the rest yellow)
            is_hl = bool(hl) and any(w.lower().strip("?!.,") in hl for w in line.split())
            fg, slab = (WHITE, RED) if is_hl else (BLACK, YELLOW)
            tags = f"\\pos(20,{y:.0f})\\an7\\bord{max(6, round(fs * 0.14))}\\shad0\\c{ass_color(fg)}\\3c{ass_color(slab)}\\4c{ass_color(slab)}"
            events.append((3, tags, ass_escape(line)))
        else:
            parts = []
            for w in line.split(" "):
                is_hl = w.lower().strip("?!.,") in hl
                col = YELLOW if is_hl else WHITE
                parts.append(f"{{\\c{ass_color(col)}}}{ass_escape(w)}")
            if style == "glow":
                tags = f"\\pos(20,{y:.0f})\\an7\\bord{outline + 6}\\blur{max(4, outline)}\\shad0\\3c{ass_color((255, 40, 40))}"
                events.append((1, tags, " ".join(parts)))
                tags2 = f"\\pos(20,{y:.0f})\\an7\\bord{max(3, outline // 2)}\\shad0\\3c{ass_color(BLACK)}"
                events.append((1, tags2, " ".join(parts)))
                continue
            tags = f"\\pos(20,{y:.0f})\\an7\\bord{outline}\\shad{shadow}\\3c{ass_color(BLACK)}\\4c{ass_color(BLACK, 0x60)}"
            events.append((1, tags, " ".join(parts)))
    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {W}\nPlayResY: {H}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
        "StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
    )
    styles = ""
    for bs in (1, 3):
        styles += (f"Style: s{bs},{font},{fs:.0f},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,{bold},0,0,0,100,100,"
                   f"{-1 if lang != 'hi' else 0},0,{bs},4,2,7,0,0,0,1\n")
    ev = "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    for bs, tags, text in events:
        ev += f"Dialogue: 0,0:00:00.00,0:00:05.00,s{bs},,0,0,0,,{{{tags}}}{text}\n"
    ass = work / f"{tag}.ass"
    ass.write_text(head + styles + ev, encoding="utf-8")
    png = work / f"{tag}.png"
    esc = str(ass).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c=black@0.0:s={W}x{H},format=rgba",
           "-vf", f"ass=filename='{esc}':alpha=1", "-frames:v", "1", str(png)]
    if subprocess.run(cmd, capture_output=True).returncode != 0 or not png.is_file():
        raise AutocutError("could not draw the thumbnail words (ffmpeg with libass is needed)")
    return Image.open(png).convert("RGBA")


def text_region(platform: str, side: str, W: int, H: int) -> tuple[int, int, int, int]:
    """Where the words go, in pixels (inside the platform's safe area, on the side away from the subject)."""
    s = PLATFORMS[platform]["safe"]
    x0, y0, x1, y1 = s[0] * W, s[1] * H, s[2] * W, s[3] * H
    if W >= H:
        if side == "left":
            x1 = W * 0.58
        else:
            x0 = W * 0.42
    else:
        y1 = y0 + (y1 - y0) * 0.5
    return int(x0), int(y0), int(x1), int(y1)


def fit_words(text: str, highlight: str, lang: str, style: str, region: tuple[int, int, int, int], canvas: tuple[int, int], work: Path, tag: str):
    """The words as large as the region allows (1–3 lines): (layer cropped to the ink, its size)."""
    rw, rh = region[2] - region[0], region[3] - region[1]
    lines = lines_of(text, 2)
    fs = min(rh / max(1, len(lines)) * 0.86, canvas[1] * (0.26 if canvas[0] >= canvas[1] else 0.16))
    layer = None
    for _ in range(4):
        layer = render_text(lines, highlight, lang, style, (max(canvas[0], rw + 200), max(canvas[1], rh + 200)), fs, work, tag)
        bb = layer.getbbox()
        if not bb:
            break
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        k = min(rw / w, rh / h)
        if 0.97 <= k <= 1.05 or (k > 1 and fs >= canvas[1] * (0.26 if canvas[0] >= canvas[1] else 0.16)):
            break
        fs *= k * 0.98
    if layer is None or not layer.getbbox():
        raise AutocutError("the thumbnail words came out empty")
    return layer.crop(layer.getbbox())


def compose(frame: dict, platform: str, text: str, highlight: str, lang: str, style: str, out: Path, work: Path) -> dict:
    """One thumbnail: (crop, pop, words placed away from faces); returns notes on how it turned out."""
    from PIL import Image

    W, H = PLATFORMS[platform]["size"]
    src = Image.open(frame["path"]).convert("RGB")
    iw, ih = src.size
    subj = subject_box(frame["det"])
    side = text_side(subj, W, H)
    framed = W < H and iw > ih  # (a vertical crop of a landscape frame is enlarged and soft)
    if framed:
        img, subj_px, region = framed_layout(src, subj, platform)
        cw = W
    else:
        x, y, cw, ch = crop_window(iw, ih, W, H, subj, side)
        img = src.crop((x, y, x + cw, y + ch)).resize((W, H), Image.LANCZOS)
        subj_px = None
        if subj:
            subj_px = (int((subj[0] * iw - x) / cw * W), int((subj[1] * ih - y) / ch * H), int((subj[2] * iw - x) / cw * W), int((subj[3] * ih - y) / ch * H))
        img = enhance(img, subj_px, side)
        region = None
    notes: dict = {"frame": f"{Path(frame['src']).name} @ {frame['t']:.1f}s", "style": style, "layout": "framed" if framed else "crop",
                   "upscaled": cw < W * 0.7}
    text = strip_unsupported(text)
    if text:
        region = region or text_region(platform, side, W, H)
        layer = fit_words(text, highlight, lang, style, region, (W, H), work, f"{platform}-{lang}-{style}")
        lw, lh = layer.size
        rx0, ry0, rx1, ry1 = region
        if W >= H:
            px = rx0 if side == "left" else rx1 - lw
            py = ry0 + (ry1 - ry0 - lh) // 2
        else:
            px = (W - lw) // 2
            py = ry0 + (int((ry1 - ry0 - lh) * 0.5) if framed else int((ry1 - ry0 - lh) * 0.35))
        for a in PLATFORMS[platform]["avoid"]:  # out of the badge: move up (it is always at the bottom)
            ax0, ay0, ax1, ay1 = a[0] * W, a[1] * H, a[2] * W, a[3] * H
            if px < ax1 and px + lw > ax0 and py + lh > ay0 and py < ay1:
                py = max(int(PLATFORMS[platform]["safe"][1] * H), int(ay0 - lh - 0.02 * H))
        base = img.convert("RGBA")
        base.alpha_composite(layer, (max(0, px), max(0, py)))
        img = base.convert("RGB")
        box = (px, py, px + lw, py + lh)
        notes["words"] = text
        notes["text_box"] = box
        if subj_px:
            fx0, fy0, fx1, fy1 = subj_px
            inter = max(0, min(box[2], fx1) - max(box[0], fx0)) * max(0, min(box[3], fy1) - max(box[1], fy0))
            notes["covers_subject"] = round(inter / max(1, (fx1 - fx0) * (fy1 - fy0)), 2)
        for a in PLATFORMS[platform]["avoid"]:
            ax0, ay0, ax1, ay1 = a[0] * W, a[1] * H, a[2] * W, a[3] * H
            if box[0] < ax1 and box[2] > ax0 and box[1] < ay1 and box[3] > ay0:
                notes["under_badge"] = True
        notes["text_height"] = round(lh / H, 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "JPEG", quality=92, optimize=True, progressive=True)
    return notes


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def preview_sheet(items: list[tuple[str, Path]], out: Path) -> None:
    """Every thumbnail at the size people see it: landscape at YouTube's home-feed size, vertical at
    a phone's grid size."""
    from PIL import Image, ImageDraw, ImageFont

    tiles = []
    for label, p in items:
        im = Image.open(p).convert("RGB")
        w = 360 if im.width >= im.height else 180
        tiles.append((label, im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)))
    if not tiles:
        return
    cols = 4
    cell_w, cell_h = 380, 360
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    except OSError:
        font = ImageFont.load_default()
    for i, (label, im) in enumerate(tiles):
        cx, cy = (i % cols) * cell_w + 10, (i // cols) * cell_h + 10
        sheet.paste(im, (cx, cy))
        d.text((cx, cy + im.height + 6), label, fill=(220, 220, 220), font=font)
    sheet.save(out, "JPEG", quality=88)


# ---------------------------------------------------------------------------------------------
# The words: titles, thumbnail words, hooks, descriptions, hashtags
# ---------------------------------------------------------------------------------------------


def copy_prompt(topic: str, hook: str, langs: list[str], platforms: list[str], niche: str, notes: str, tops: list[dict], library: str, online: bool) -> str:
    top = "\n".join(f"- {v['title']} ({v.get('views', 0):,} views)" for v in tops[:12]) or "(none found)"
    want = ", ".join(PLATFORMS[p]["label"] for p in platforms)
    langs_txt = " and ".join(f'"{l}" ({LANGUAGES[l]})' for l in langs)
    research = ("\nBefore writing, search the web for what is hot about this topic right now (news, the angle people argue about, the "
                "searches rising) and for the hashtags creators use on it this week on YouTube and Instagram. Use what you find.\n") if online else ""
    return f"""Write the packaging for a video: what makes someone stop scrolling and click. It is for {want}.

Topic: {topic}
Niche: {niche or "work it out from the topic"}
{f'The creator\'s hook (build on it, keep its meaning): {hook}' if hook else 'No hook given: find the strongest one for this topic.'}
{f'About the video: {notes}' if notes else ''}
{research}
The most-viewed YouTube videos on this topic (what already wins clicks):
{top}
{library}
## Rules (follow all)
- A curiosity gap the video really closes: promise something specific (a number, a stake, a surprise, a conflict). Never a lie.
- Never promise anything the topic and notes above do not back up (no "I'll show you a secret setting" unless it is in the notes).
- Title and thumbnail words tell ONE story together and never repeat each other: the thumbnail adds the emotion or the stakes
  (e.g. title "GTA 6 Will Only Run at 30 FPS on Consoles" + thumbnail "IT'S OVER?").
- YouTube title: keyword first, 40–60 characters, at most one word in capitals, no emoji, no hashtags in it.
- Shorts title: under 40 characters, a punchy line, then 1–2 hashtags.
- Thumbnail words: 2–4 words, at most 18 characters, readable on a phone; give the one word to colour as "highlight".
- Hook: the first spoken/on-screen line of the video (under 12 words) that makes leaving feel like missing out.
- Instagram caption: the first line is a hook under 90 characters (it is all people see before "more"), then 1–2 short lines
  of value, then a call to action (comment a word, save, share, follow for part 2).
- YouTube hashtags: 3–5, the first three show above the title; specific ones first, one broad one; YouTube tags: 10–15 search
  phrases people really type.
- Instagram hashtags: the best 5 (Instagram now limits posts to 5), then 10 spares; a mix of specific and broad, no banned or spammy tags.
- Hindi ("hi"): natural spoken Hindi as Indian creators write it, in Devanagari; names, games and tech words stay as they are
  (GTA 6, FPS, PS5); never stiff textbook Hindi. Hindi hashtags may mix Hindi and English words people search.
- Rank every list best first and say in a few words why the first works.

Languages: {langs_txt}. Answer with only this JSON:
{{"languages": {{"<code>": {{
  "titles": ["5 YouTube titles, best first"],
  "why": "why title 1 works, under 15 words",
  "thumbnail": [{{"text": "2-4 words", "highlight": "the word to colour"}}, "5 of them, best first, each fitting title 1"],
  "hooks": ["3 opening lines"],
  "description": "a YouTube description: 2-3 lines with the keyword, then a line with the hashtags",
  "youtube_tags": ["10-15 search phrases"],
  "youtube_hashtags": ["#3-5"],
  "shorts_titles": ["3 Shorts titles"],
  "shorts_hashtags": ["#2-4"],
  "instagram_captions": ["2 full captions"],
  "instagram_hashtags": ["#best 5"],
  "instagram_spare_hashtags": ["#10 more"]
}}}}}}
"""


def tag_of(words_: str) -> str:
    w = re.sub(r"[^\w]+", "", words_, flags=re.UNICODE)
    return f"#{w}" if w else ""


def hashtags(xs) -> list[str]:
    out = []
    for x in xs or []:
        t = str(x).strip()
        t = t if t.startswith("#") else tag_of(t)
        t = "#" + re.sub(r"[^\w]", "", t[1:], flags=re.UNICODE)
        if len(t) > 1 and t.lower() not in {o.lower() for o in out}:
            out.append(t)
    return out


def rules_copy(topic: str, hook: str, lang: str) -> dict:
    """A usable set without a language model: the hook or the topic's strongest words."""
    words_ = [w for w in re.findall(r"[\w'’]+", topic) if len(w) > 1]
    key = " ".join(words_[:4]) or "Watch this"
    thumb = " ".join((hook or topic).split()[:3]).upper() if lang == "en" else " ".join((hook or topic).split()[:3])
    tags = hashtags(["".join(w.capitalize() for w in words_[:2])] + [w for w in words_ if len(w) > 3][:4])
    title = topic if len(topic) <= 60 else topic[:57].rstrip() + "…"
    return {"titles": [title, f"{key}: what nobody tells you", f"The truth about {key}"], "why": "made by rules (no language model)",
            "thumbnail": [{"text": thumb, "highlight": thumb.split()[-1] if thumb else ""}], "hooks": [hook or f"Here's what's really going on with {key}."],
            "description": f"{topic}\n\n{' '.join(tags[:3])}", "youtube_tags": [topic] + words_[:10], "youtube_hashtags": tags[:4],
            "shorts_titles": [title[:40]], "shorts_hashtags": tags[:2] + ["#shorts"], "instagram_captions": [f"{hook or title}\n\n{' '.join(tags[:5])}"],
            "instagram_hashtags": tags[:5], "instagram_spare_hashtags": tags[5:]}


def clean_copy(c: dict, lang: str) -> dict:
    """The model's answer, checked: lists are lists, hashtags are hashtags, lengths within limits."""
    def strs(k, n=20):
        v = c.get(k) or []
        return [re.sub(r"\s+", " ", str(x)).strip() for x in (v if isinstance(v, list) else [v]) if str(x).strip()][:n]

    thumbs = []
    for t in c.get("thumbnail") or []:
        if isinstance(t, dict) and str(t.get("text", "")).strip():
            thumbs.append({"text": strip_unsupported(str(t["text"]))[:40], "highlight": str(t.get("highlight", "")).strip()})
        elif isinstance(t, str) and t.strip():
            thumbs.append({"text": strip_unsupported(t)[:40], "highlight": ""})
    return {
        "titles": [t[:100] for t in strs("titles", 6)], "why": str(c.get("why", ""))[:200], "thumbnail": thumbs[:6], "hooks": strs("hooks", 4),
        "description": str(c.get("description", "")).strip()[:4500], "youtube_tags": strs("youtube_tags", 20),
        "youtube_hashtags": hashtags(strs("youtube_hashtags", 8))[:5], "shorts_titles": [t[:100] for t in strs("shorts_titles", 4)],
        "shorts_hashtags": hashtags(strs("shorts_hashtags", 6))[:4], "instagram_captions": [str(x).strip()[:2200] for x in (c.get("instagram_captions") or [])][:3],
        "instagram_hashtags": hashtags(strs("instagram_hashtags", 8))[:5], "instagram_spare_hashtags": hashtags(strs("instagram_spare_hashtags", 15))[:12],
    }


def write_copy(topic: str, hook: str, langs: list[str], platforms: list[str], *, niche: str = "", notes: str = "", find_hook: bool = False,
               writer: str = "claude", model: str | None = None, fresh: bool = False) -> dict:
    import hashlib

    import hooks as hk

    # the same request again (for new frames or another platform) reuses the words for a day
    key = hashlib.sha1(json.dumps([topic, hook, sorted(langs), sorted(platforms), niche, notes, find_hook, writer, model]).encode()).hexdigest()[:16]
    cache = ac.CACHE_DIR / "package" / f"{key}.json"
    if not fresh and cache.is_file() and time.time() - cache.stat().st_mtime < 86400:
        try:
            got = json.loads(cache.read_text(encoding="utf-8"))
            if all(l in got for l in langs):
                log("using the titles, words and hashtags written for this topic earlier today (--fresh-words to write new ones)")
                return got
        except ValueError:
            pass
    tops = hk.youtube_top(topic) if topic else []
    if tops:
        log(f"{len(tops)} most-viewed videos on the topic read")
    if find_hook and not hook:
        try:
            found = hk.find_for_topic(topic, niche, language=langs[0], n=6, model=model)
            if found:
                notes = (notes + "\nHooks found online for this topic (strongest first): " + " | ".join(h["text"] for h in found[:6])).strip()
        except AutocutError as e:
            log(f"hook search failed ({e}); writing without it")
    lib = hk.prompt_block(hk.suggest(topic, niche, language=langs[0], n=6))
    out: dict = {}
    if writer == "claude":
        import script_writer as sw
        import topic_video as tv

        prompt = copy_prompt(topic, hook, langs, platforms, niche, notes, tops, lib, online=find_hook)
        log("Claude is writing the titles, thumbnail words, hooks and hashtags" + (" (with a web search)" if find_hook else ""))
        try:
            text = tv.ask_claude_research(prompt, model) if find_hook else sw.ask_claude(prompt, model)
            res = sw.extract_json(text).get("languages", {})
            for l in langs:
                if isinstance(res.get(l), dict):
                    out[l] = clean_copy(res[l], l)
        except AutocutError as e:
            log(f"Claude could not write the packaging ({e}); using simple rules")
    for l in langs:
        if l not in out or not out[l]["thumbnail"] or not out[l]["titles"]:
            out[l] = clean_copy(rules_copy(topic, hook, l), l)
        if hook and l == "en" and not any(hook.lower() == h.lower() for h in out[l]["hooks"]):
            out[l]["hooks"].insert(0, hook)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def txt_for(platform: str, lang: str, c: dict) -> str:
    L = LANGUAGES[lang]
    th = "\n".join(f"  {i}. {t['text']}" + (f"   (colour: {t['highlight']})" if t.get("highlight") else "") for i, t in enumerate(c["thumbnail"], 1))
    hooks = "\n".join(f"  {i}. {h}" for i, h in enumerate(c["hooks"], 1))
    if platform in ("youtube", "shorts"):
        titles = c["titles"] if platform == "youtube" else c["shorts_titles"] or c["titles"]
        tags = c["youtube_hashtags"] if platform == "youtube" else c["shorts_hashtags"]
        body = (f"{PLATFORMS[platform]['label']} — {L}\n\nTITLE (best first)\n" + "\n".join(f"  {i}. {t}" for i, t in enumerate(titles, 1))
                + (f"\n  why 1 works: {c['why']}" if c.get("why") else "")
                + f"\n\nTHUMBNAIL WORDS\n{th}\n\nOPENING HOOK\n{hooks}\n\nDESCRIPTION\n{c['description']}\n\nHASHTAGS (the first three show above the title)\n{' '.join(tags)}\n")
        if platform == "youtube":
            body += f"\nTAGS (YouTube Studio ▸ Tags)\n{', '.join(c['youtube_tags'])}\n"
        return body
    caps = "\n\n---\n\n".join(c["instagram_captions"]) or (c["hooks"][0] if c["hooks"] else "")
    return (f"{PLATFORMS[platform]['label']} — {L}\n\nCOVER WORDS\n{th}\n\nOPENING HOOK\n{hooks}\n\nCAPTION (the first line is all people see before “more”)\n{caps}\n\n"
            f"HASHTAGS (the best 5)\n{' '.join(c['instagram_hashtags'])}\n\nSPARE HASHTAGS\n{' '.join(c['instagram_spare_hashtags'])}\n")


# ---------------------------------------------------------------------------------------------
# All of it
# ---------------------------------------------------------------------------------------------


def make(clips: list[str], topic: str, *, hook: str = "", find_hook: bool = False, platforms: list[str] | None = None, langs: list[str] | None = None,
         variants: int = 3, out_dir: str | None = None, niche: str = "", notes: str = "", writer: str = "claude", model: str | None = None,
         fresh_words: bool = False, style: str = "studio") -> dict:
    ac.require_tools()
    platforms = [p for p in (platforms or ["youtube"]) if p in PLATFORMS] or ["youtube"]
    langs = [l for l in (langs or ["en"]) if l in LANGUAGES] or ["en"]
    variants = max(1, min(6, int(variants)))
    topic = re.sub(r"\s+", " ", topic or "").strip()
    hook = re.sub(r"\s+", " ", hook or "").strip()
    paths = collect(clips)
    if not paths:
        raise AutocutError("give video clips or images to take the thumbnails from")
    if not topic:
        topic = Path(paths[0]).stem
    import yt_download as yd

    base = Path(out_dir).expanduser().resolve() if out_dir else Path(paths[0]).parent
    folder = base / f"{yd.safe_name(topic)[:70]} - package"
    (folder / "thumbnails").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ac.progress_plan({"words": 0.35, "frames": 0.4, "thumbnails": 0.25})
    with ac.stage("words"):
        copy = write_copy(topic, hook, langs, platforms, niche=niche, notes=notes, find_hook=find_hook, writer=writer, model=model, fresh=fresh_words)
    made: list[dict] = []
    sheet: list[tuple[str, Path]] = []
    with tempfile.TemporaryDirectory(prefix="ei-thumbs-") as tmp:
        work = Path(tmp)
        with ac.stage("frames"):
            # the studio's art director chooses among more frames than it needs
            frames = pick_frames(paths, f"{topic}. {hook}".strip(". "), max(8, 3 * variants) if style == "studio" else variants, work)
        credits: list[str] = []
        if style == "studio":
            import thumb_studio as ts

            with ac.stage("thumbnails"):
                made, credits, concepts = ts.make(frames, topic, hook, copy, {p: PLATFORMS[p]["size"] for p in platforms}, langs, variants, folder, work,
                                                  niche=niche, writer=writer, model=model)
            sheet = [(f"{PLATFORMS[m['platform']]['label']} · {LANGUAGES[m['language']]} · {m['variant']}", Path(m["path"])) for m in made]
        with ac.stage("thumbnails") if style != "studio" else _nothing():
            total = len(platforms) * len(langs) * variants
            k = 0
            for p in (platforms if style != "studio" else []):
                for l in langs:
                    words_ = copy[l]["thumbnail"] or [{"text": hook or topic, "highlight": ""}]
                    for v in range(variants):
                        frame = frames[v % len(frames)]
                        w = words_[v % len(words_)]
                        look = STYLES[v % len(STYLES)]
                        out = folder / "thumbnails" / f"{p}-{l}-{v + 1}.jpg"
                        notes_ = compose(frame, p, w["text"], w.get("highlight", ""), l, look, out, work)
                        made.append({"platform": p, "language": l, "variant": v + 1, "path": str(out), **{k_: v_ for k_, v_ in notes_.items() if k_ != "text_box"}})
                        sheet.append((f"{PLATFORMS[p]['label']} · {LANGUAGES[l]} · {v + 1}", out))
                        k += 1
                        ac.stage_progress(k / max(1, total), f"{k} of {total} thumbnails")
    preview_sheet(sheet, folder / "preview.jpg")
    for p in platforms:
        for l in langs:
            (folder / f"{p}-{l}.txt").write_text(txt_for(p, l, copy[l]), encoding="utf-8")
    if any(p in ("youtube", "shorts") for p in platforms):
        (folder / "hashtags-youtube.txt").write_text("\n\n".join(
            f"{LANGUAGES[l]}\nVideo: {' '.join(copy[l]['youtube_hashtags'])}\nShorts: {' '.join(copy[l]['shorts_hashtags'])}\nTags: {', '.join(copy[l]['youtube_tags'])}"
            for l in langs) + "\n", encoding="utf-8")
    if any(p in ("instagram", "reels") for p in platforms):
        (folder / "hashtags-instagram.txt").write_text("\n\n".join(
            f"{LANGUAGES[l]}\nBest 5: {' '.join(copy[l]['instagram_hashtags'])}\nSpares: {' '.join(copy[l]['instagram_spare_hashtags'])}" for l in langs) + "\n",
            encoding="utf-8")
    issues = [f"{Path(m['path']).name}: the words cover part of the subject" for m in made if m.get("covers_subject", 0) > 0.15]
    issues += [f"{Path(m['path']).name}: the words reach the platform's badge area" for m in made if m.get("under_badge")]
    issues += [f"{Path(m['path']).name}: the frame was enlarged (the source is small; a sharper clip looks better)" for m in made if m.get("upscaled")]
    if credits:
        (folder / "credits.txt").write_text("Pictures used in the thumbnails (openly licensed; credit them where the licence asks):\n"
                                            + "\n".join(f"- {c}" for c in dict.fromkeys(credits)) + "\n", encoding="utf-8")
    package = {"version": 1, "style": style, "credits": list(dict.fromkeys(credits)), "topic": topic, "hook": hook, "platforms": platforms, "languages": langs, "copy": copy, "thumbnails": made,
               "frames": [{"src": f["src"], "t": round(f["t"], 2), "score": round(float(f["score"]), 2)} for f in frames], "issues": issues,
               "made": time.strftime("%Y-%m-%d %H:%M")}
    (folder / "package.json").write_text(json.dumps(package, ensure_ascii=False, indent=1), encoding="utf-8")
    for i in issues:
        log(f"check: {i}")
    best = {l: copy[l]["titles"][0] for l in langs if copy[l]["titles"]}
    log(f"done in {time.time() - t0:.0f}s: {len(made)} thumbnails, titles and hashtags in {folder}" + "".join(f"; {LANGUAGES[l]}: “{t}”" for l, t in best.items()))
    return {"outputs": [m["path"] for m in made], "out_dir": str(folder), "preview": str(folder / "preview.jpg"), "package": str(folder / "package.json"),
            "titles": best, "issues": issues}


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--clips", nargs="+", required=True, help="video clips, images or folders to take the thumbnail frames from")
    p.add_argument("--topic", default="", help="what the video is about")
    p.add_argument("--hook", default="", help="your hook (the thumbnail words and titles build on it)")
    p.add_argument("--find-hook", action="store_true", help="search online for a hot hook and this week's hashtags")
    p.add_argument("--platforms", nargs="+", default=["youtube"], choices=list(PLATFORMS))
    p.add_argument("--languages", nargs="+", default=["en"], choices=list(LANGUAGES))
    p.add_argument("--variants", type=int, default=3, help="thumbnails per platform and language (1 to 6)")
    p.add_argument("--niche", default="")
    p.add_argument("--notes", default="", help="anything about the video the words should know")
    p.add_argument("--writer", default="claude", choices=["claude", "rules"])
    p.add_argument("--model", default=None)
    p.add_argument("--fresh-words", action="store_true", help="write new words even if this topic was done earlier today")
    p.add_argument("--style", default="studio", choices=["studio", "simple"],
                   help="studio: designed from scratch (art director, generated backgrounds, cut-outs, props, pro lettering); simple: a frame with words")
    p.add_argument("--out-dir", default=None)


def run_args(a) -> dict:
    return make(a.clips, a.topic, hook=a.hook, find_hook=a.find_hook, platforms=a.platforms, langs=a.languages, variants=a.variants,
                out_dir=a.out_dir, niche=a.niche, notes=a.notes, writer=a.writer, model=a.model, fresh_words=a.fresh_words, style=a.style)


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
