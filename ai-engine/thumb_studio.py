"""thumb_studio: professional thumbnails built from scratch, not a frame with words on it.

An art director (Claude, looking at the best frames of the clips) plans a few different concepts:
the layout (the hero on one side, versus, a reaction pointing at an object, a huge number), who or
what is the subject, the 2–3 words and a small kicker, the colours, a background to generate, props
to find and arrows or badges. Then each concept is built in layers:

  1. background: generated from scratch by a local image model (LCM Dreamshaper v7, MIT licence,
     ~5 s an image on an M3) or a blurred, graded frame; light rays and a glow behind the subject;
  2. the subject cut out of the clip (Apple Vision; burned-in subtitles removed), enlarged, with a
     white outline, a coloured outer glow, a rim light and a drop shadow;
  3. props found online (Wikimedia Commons, Openverse: openly licensed, credited in credits.txt),
     cut out the same way, tilted, with an arrow or a ring pointing at them; badges;
  4. the words: Anton (English) or Mukta ExtraBold (Hindi), both SIL Open Font License, downloaded
     once into EI's cache, drawn by libass (correct Hindi shaping) and built up as a gradient fill, a
     thick dark stroke, a 3-D extrusion and a soft shadow, slightly tilted, the key word in the accent
     colour, a kicker on a colour slab above.

Then a critic (Claude, looking at all concepts at the size people see them) ranks them; the best
becomes variant 1. Without Claude, rules choose the concepts and the order.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

import autocut as ac
from autocut import AutocutError, log

FONT_FILES = {
    "Anton-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",
    "Mukta-ExtraBold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/mukta/Mukta-ExtraBold.ttf",
}
FONT_FAMILY = {"en": ("Anton", "Impact"), "hi": ("Mukta ExtraBold", "Kohinoor Devanagari")}
IMAGE_MODEL = "SimianLuo/LCM_Dreamshaper_v7"
LAYOUTS = ("hero-right", "hero-left", "versus", "reaction-object", "big-number")
UA = "EI-thumbnails/1.0 (a personal video editor)"
_FONT_TRIED: set[str] = set()


def _ctx():
    """Certificates from certifi: python.org's Python does not use the macOS keychain."""
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch(url: str, timeout: int = 40) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as r:
        return r.read()


def luma(c: tuple[int, int, int]) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def bright(c: tuple[int, int, int], fallback=(255, 212, 0)) -> tuple[int, int, int]:
    """A colour that reads on a dark background (a dark one gives way to yellow)."""
    return c if luma(c) >= 120 else fallback


def hexrgb(h: str, default=(255, 200, 0)) -> tuple[int, int, int]:
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(h or "").strip())
    if not m:
        return default
    v = m.group(1)
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


# ---------------------------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------------------------


def fonts_dir() -> Path:
    d = ac.CACHE_DIR / "fonts"
    d.mkdir(parents=True, exist_ok=True)
    for name, url in FONT_FILES.items():
        f = d / name
        if (f.is_file() and f.stat().st_size > 10000) or name in _FONT_TRIED:
            continue
        _FONT_TRIED.add(name)  # (one try per run)
        try:
            data = fetch(url, 60)
            if len(data) > 10000:
                f.write_bytes(data)
                log(f"font {name} downloaded (SIL Open Font License)")
        except OSError as e:
            log(f"font {name} could not be downloaded ({e}); using the Mac's own")
    return d


def font_family(lang: str) -> str:
    d = fonts_dir()
    want, fallback = FONT_FAMILY["hi" if lang == "hi" else "en"]
    file = "Mukta-ExtraBold.ttf" if lang == "hi" else "Anton-Regular.ttf"
    return want if (d / file).is_file() else fallback


# ---------------------------------------------------------------------------------------------
# Cut-outs (Apple Vision)
# ---------------------------------------------------------------------------------------------


def cutout(src: Path, text_boxes: list | None = None) -> np.ndarray | None:
    """The foreground (people, animals, objects) as a 0–1 mask the size of the image, without any
    words already in the picture; None when nothing stands out."""
    import objc
    import Quartz
    import Vision
    from Foundation import NSURL
    from PIL import Image

    out = src.with_suffix(".mask.png")
    with objc.autorelease_pool():
        h = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(str(src)), None)
        rq = Vision.VNGenerateForegroundInstanceMaskRequest.alloc().init()
        ok = h.performRequests_error_([rq], None)
        res = rq.results() or []
        if not (ok[0] if isinstance(ok, tuple) else ok) or not res:
            return None
        obs = res[0]
        buf, _err = obs.generateScaledMaskForImageForInstances_fromRequestHandler_error_(obs.allInstances(), h, None)
        if buf is None:
            return None
        ci = Quartz.CIImage.imageWithCVPixelBuffer_(buf)
        Quartz.CIContext.context().writePNGRepresentationOfImage_toURL_format_colorSpace_options_error_(
            ci, NSURL.fileURLWithPath_(str(out)), Quartz.kCIFormatL8, Quartz.CGColorSpaceCreateDeviceGray(), None, None)
    if not out.is_file():
        return None
    size = Image.open(src).size
    m = np.asarray(Image.open(out).convert("L").resize(size), dtype=np.float32) / 255.0
    W, H = size
    for b in text_boxes or []:  # (subtitles and logos are "foreground" too: not here)
        x0, y0, x1, y1 = int(b[0] * W) - 6, int(b[1] * H) - 6, int(b[2] * W) + 6, int(b[3] * H) + 6
        m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 0
    if m.mean() < 0.01:
        return None
    return m


def bbox(mask: np.ndarray, thr: float = 0.5) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > thr)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def subject_layer(frame_path: Path, mask: np.ndarray):
    """The cut-out as an RGBA image cropped to it, graded to pop."""
    from PIL import Image, ImageEnhance, ImageFilter

    im = Image.open(frame_path).convert("RGB")
    g = np.asarray(im.convert("L"), dtype=np.float32)
    inside = g[mask > 0.5]
    lum = float(inside.mean()) if inside.size else 128.0
    if lum < 95:  # a backlit subject would be a silhouette: lift the shadows
        gamma = max(0.45, min(1.0, math.log(110 / 255) / math.log(max(lum, 8) / 255)))
        im = im.point(lambda v: int(255 * (v / 255) ** gamma))
    im = ImageEnhance.Contrast(im).enhance(1.15)
    im = ImageEnhance.Color(im).enhance(1.25)
    im = ImageEnhance.Brightness(im).enhance(1.08)
    im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=110, threshold=2))
    a = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    rgba = im.convert("RGBA")
    rgba.putalpha(a)
    b = bbox(mask)
    return rgba.crop(b) if b else rgba


# ---------------------------------------------------------------------------------------------
# Props from openly licensed images
# ---------------------------------------------------------------------------------------------


def http_json(url: str, timeout: int = 30) -> dict:
    return json.loads(fetch(url, timeout))


def search_images(query: str, n: int = 6) -> list[dict]:
    """Openly licensed pictures: Wikimedia Commons first, then Openverse. [{url, title, creator, licence, page}]"""
    out: list[dict] = []
    try:
        q = urllib.parse.urlencode({"action": "query", "generator": "search", "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": 6, "gsrlimit": 12,
                                    "prop": "imageinfo", "iiprop": "url|size|extmetadata", "iiurlwidth": 1200, "format": "json"})
        d = http_json(f"https://commons.wikimedia.org/w/api.php?{q}")
        pages = sorted((d.get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 99))
        for p in pages:
            ii = (p.get("imageinfo") or [{}])[0]
            meta = ii.get("extmetadata", {})
            lic = (meta.get("LicenseShortName") or {}).get("value", "")
            if not ii.get("thumburl") or not re.search(r"CC0|Public domain|CC BY", lic, re.I) or re.search(r"\bNC\b|ND", lic):
                continue
            creator = re.sub(r"<[^>]+>", "", (meta.get("Artist") or {}).get("value", "")).strip()[:80]
            out.append({"url": ii["thumburl"], "title": (p.get("title") or "").removeprefix("File:"), "creator": creator, "licence": lic,
                        "page": ii.get("descriptionurl", "")})
    except (OSError, ValueError) as e:
        log(f"Wikimedia Commons search failed ({e})")
    if len(out) < n:
        try:
            q = urllib.parse.urlencode({"q": query, "page_size": 20, "license_type": "commercial,modification"})
            d = http_json(f"https://api.openverse.org/v1/images/?{q}")
            for r in d.get("results", []):
                out.append({"url": r.get("url"), "title": r.get("title") or "", "creator": r.get("creator") or "",
                            "licence": f"{r.get('license', '').upper()} {r.get('license_version', '')}".strip(), "page": r.get("foreign_landing_url") or ""})
        except (OSError, ValueError) as e:
            log(f"Openverse search failed ({e})")
    return out[: n * 2]


GENERIC = {"a", "an", "the", "of", "and", "photo", "image", "picture", "png", "jpg", "console", "device", "isolated", "object"}


def matches(query: str, title: str) -> bool:
    """The picture is of what was asked: every number of the query in its title (a PS3 is not a
    PS5) and most of its words."""
    tok = lambda t: re.findall(r"[a-z]+|\d+", t.lower().replace("_", " "))  # noqa: E731
    q, t = [w for w in tok(query) if w not in GENERIC], set(tok(title))
    if not q:
        return True
    nums = [w for w in q if w.isdigit()]
    if any(n not in t for n in nums):
        return False
    return sum(w in t for w in q) >= max(1, math.ceil(len(q) * 0.6))


def find_prop(query: str, work: Path, tag: str) -> dict | None:
    """The best cut-out-able picture for a prop: {layer (RGBA), credit}."""
    from PIL import Image

    for k, r in enumerate(r for r in search_images(query) if matches(query, r["title"])):
        f = work / f"prop-{tag}-{k}.jpg"
        try:
            f.write_bytes(fetch(r["url"]))
            im = Image.open(f).convert("RGB")
            if min(im.size) < 300:
                continue
            im.save(f.with_suffix(".png"))
        except (OSError, ValueError):
            continue
        m = cutout(f.with_suffix(".png"))
        if m is None:
            continue
        cover = float(m.mean())
        b = bbox(m)
        if not b or cover < 0.08 or cover > 0.85 or (b[2] - b[0]) < 200:
            continue
        layer = subject_layer(f.with_suffix(".png"), m)
        log(f"prop “{query}”: {r['title'][:60]} ({r['licence']})")
        return {"layer": layer, "credit": f"{r['title']} — {r['creator'] or 'unknown'} — {r['licence']} — {r['page']}"}
    log(f"prop “{query}”: nothing usable found")
    return None


# ---------------------------------------------------------------------------------------------
# Backgrounds generated from scratch
# ---------------------------------------------------------------------------------------------


def _gen_worker(jobs: list[tuple[str, int, int, str, int]]) -> list[str]:
    """Runs in its own process: the image model once, every background."""
    import torch
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(IMAGE_MODEL, torch_dtype=torch.float16, safety_checker=None)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe.to(dev)
    pipe.enable_attention_slicing()
    done = []
    for prompt, w, h, out, seed in jobs:
        g = torch.Generator("cpu").manual_seed(seed)
        im = pipe(prompt, num_inference_steps=6, guidance_scale=8.0, width=w, height=h, generator=g).images[0]
        im.save(out)
        done.append(out)
    return done


def gen_size(W: int, H: int) -> tuple[int, int]:
    """The model's size for a W×H picture (about 330k pixels, multiples of 8)."""
    a = W / H
    h = math.sqrt(331776 / a)
    return int(round(a * h / 8) * 8), int(round(h / 8) * 8)


def generate_backgrounds(jobs: list[tuple[str, int, int, str, int]]) -> list[str]:
    if not jobs:
        return []
    log(f"generating {len(jobs)} background(s) with the local image model (first use downloads it, ~4 GB)")
    t0 = time.time()
    try:
        res = ac.run_isolated(_gen_worker, jobs)
    except Exception as e:  # (no model, no memory: frames are used instead)
        log(f"could not generate backgrounds ({e.__class__.__name__}: {str(e)[:160]}); using the clips' frames")
        return []
    log(f"backgrounds in {time.time() - t0:.0f}s")
    return list(res)


# ---------------------------------------------------------------------------------------------
# The art director and the critic (Claude, looking at the pictures)
# ---------------------------------------------------------------------------------------------


def contact_sheet(frames: list[dict], out: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    cols = 4
    tw, th = 480, 270
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Black.ttf", 40)
    except OSError:
        font = ImageFont.load_default()
    for i, f in enumerate(frames):
        im = Image.open(f["path"]).convert("RGB")
        im.thumbnail((tw, th))
        x, y = (i % cols) * tw, (i // cols) * th
        sheet.paste(im, (x, y))
        d.rectangle((x, y, x + 64, y + 52), fill=(230, 30, 40))
        d.text((x + 12, y + 2), str(i + 1), fill=(255, 255, 255), font=font)
    sheet.save(out, "JPEG", quality=85)


def ask_claude_seeing(prompt: str, folder: Path, model: str | None = None, timeout: int = 600) -> str:
    """Claude Code with the Read tool only, in the folder that holds the pictures it should look at."""
    import shutil

    exe = shutil.which("claude") or next((p for p in ("/opt/homebrew/bin/claude", "/usr/local/bin/claude", str(Path.home() / ".local/bin/claude")) if Path(p).is_file()), None)
    if not exe:
        raise AutocutError("Claude Code is not installed")
    cmd = [exe, "-p", "--output-format", "json", "--allowedTools", "Read", "--system-prompt",
           "You are a top YouTube thumbnail art director for Indian creators (English and Hindi audiences). Look at the images you are given with the Read tool. Output only the requested JSON."]
    if model:
        cmd += ["--model", model]
    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=folder)
    except subprocess.TimeoutExpired as e:
        raise AutocutError("Claude took too long") from e
    try:
        res = json.loads(proc.stdout)
    except ValueError as e:
        raise AutocutError(f"Claude Code did not answer ({(proc.stderr or proc.stdout).strip()[:200]})") from e
    if res.get("is_error"):
        raise AutocutError(f"Claude: {str(res.get('result'))[:200]}")
    return str(res.get("result", ""))


def director_prompt(topic: str, hook: str, titles: dict, words: dict, langs: list[str], n: int, nframes: int, niche: str) -> str:
    wl = "\n".join(f"- {l}: title “{titles.get(l, '')}”; word ideas: " + " | ".join(w["text"] for w in words.get(l, [])[:5]) for l in langs)
    return f"""Plan {n} DIFFERENT thumbnail concepts for this video. Open frames.jpg first (Read tool): it has {nframes} numbered frames from the creator's clips.

Topic: {topic}
Niche: {niche or "work it out"}
Hook: {hook or "(none given)"}
{wl}

What wins clicks: ONE clear focal point; a big, expressive face (shock, fear, anger, awe) or a striking subject; extreme contrast
and saturated complementary colours; 2-3 huge words (never the title repeated) that add emotion or stakes; a curiosity gap; arrows,
circles, badges or a "VS" only when they tell the story; clean background with depth; nothing important in the bottom-right corner.

Layouts: hero-right (subject on the right, words left), hero-left, versus (two subjects, VS between), reaction-object (a reacting
face + an object/prop with an arrow at it), big-number (a huge number like "30 FPS" is the hero, subject beside it).
Use different layouts, subjects and colours across the concepts. Pick subjects whose frame number shows a clear, large person or
object that would cut out well (not tiny, not blurry, not full of on-screen text).

Backgrounds are generated from scratch by an image model from your prompt: describe a dramatic scene or atmosphere that fits
(lighting, colours, place), NO people, NO text, NO logos (the subject is added on top). Or use "frame" to blur a frame.
Props are searched on Wikimedia Commons (real objects: "PlayStation 5 console", "gaming PC", "Mumbai skyline"); at most one.

Words: "main" 1-3 words (max 14 characters, in CAPITALS for English), "highlight" = the word(s) to colour, "kicker" = 1-2 small
words above (a name or context like "GTA 6", may be empty). Hindi: natural spoken Hindi in Devanagari (names and tech words stay
English), same idea not a literal translation.

Answer with only this JSON:
{{"concepts": [{{"layout": "hero-right", "subject": 3, "second": null, "background": {{"type": "generate", "prompt": "…", "frame": null}},
  "palette": ["#FFD400", "#E01E37"], "props": [{{"query": "…"}}], "arrow": false, "ring": false, "badge": "",
  "words": {{"en": {{"kicker": "GTA 6", "main": "30 FPS?!", "highlight": "30 FPS"}}, "hi": {{"kicker": "GTA 6", "main": "सिर्फ 30 FPS?", "highlight": "30 FPS"}}}},
  "why": "under 15 words"}}]}}
"""


def rule_concepts(frames: list[dict], words: dict, langs: list[str], n: int, topic: str) -> list[dict]:
    out = []
    palettes = [["#FFD400", "#E01E37"], ["#00E5FF", "#FF2D95"], ["#FF7A00", "#1E90FF"]]
    for k in range(n):
        w = {}
        for l in langs:
            ws = words.get(l) or [{"text": topic[:14].upper(), "highlight": ""}]
            t = ws[k % len(ws)]
            w[l] = {"kicker": "", "main": t["text"], "highlight": t.get("highlight", "")}
        out.append({"layout": ("hero-right", "hero-left", "big-number")[k % 3], "subject": k % max(1, len(frames)) + 1, "second": None,
                    "background": {"type": "generate", "prompt": f"dramatic cinematic background for {topic}, moody lighting, high contrast, no people, no text"},
                    "palette": palettes[k % 3], "props": [], "arrow": False, "ring": False, "badge": "", "words": w, "why": "rules"})
    return out


def clean_concepts(raw: list, nframes: int, langs: list[str], fallback: list[dict]) -> list[dict]:
    out = []
    for k, c in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(c, dict):
            continue
        fb = fallback[k % len(fallback)]

        def frame_no(v):
            try:
                i = int(v)
            except (TypeError, ValueError):
                return None
            return i if 1 <= i <= nframes else None

        bg = c.get("background") if isinstance(c.get("background"), dict) else {}
        words = {}
        for l in langs:
            w = (c.get("words") or {}).get(l) if isinstance(c.get("words"), dict) else None
            if not isinstance(w, dict) or not str(w.get("main", "")).strip():
                w = fb["words"][l]
            words[l] = {"kicker": str(w.get("kicker", ""))[:24], "main": str(w.get("main", ""))[:40], "highlight": str(w.get("highlight", ""))[:30]}
        props = [p for p in (c.get("props") or []) if isinstance(p, dict) and str(p.get("query", "")).strip()][:1]
        out.append({
            "layout": c.get("layout") if c.get("layout") in LAYOUTS else fb["layout"],
            "subject": frame_no(c.get("subject")) or fb["subject"], "second": frame_no(c.get("second")),
            "background": {"type": "frame" if bg.get("type") == "frame" else "generate", "prompt": str(bg.get("prompt") or fb["background"]["prompt"])[:400],
                           "frame": frame_no(bg.get("frame"))},
            "palette": [p for p in (c.get("palette") or []) if re.fullmatch(r"#?[0-9a-fA-F]{6}", str(p))][:3] or fb["palette"],
            "props": props, "arrow": bool(c.get("arrow")), "ring": bool(c.get("ring")), "badge": str(c.get("badge") or "")[:16],
            "words": words, "why": str(c.get("why", ""))[:160]})
    return out or fallback


def plan(frames: list[dict], topic: str, hook: str, copy: dict, langs: list[str], n: int, niche: str, work: Path,
         writer: str = "claude", model: str | None = None) -> list[dict]:
    words = {l: copy.get(l, {}).get("thumbnail", []) for l in langs}
    titles = {l: (copy.get(l, {}).get("titles") or [""])[0] for l in langs}
    fallback = rule_concepts(frames, words, langs, n, topic)
    if writer != "claude":
        return fallback
    contact_sheet(frames, work / "frames.jpg")
    log("the art director (Claude) is looking at the frames and planning the concepts")
    try:
        import script_writer as sw

        raw = sw.extract_json(ask_claude_seeing(director_prompt(topic, hook, titles, words, langs, n, len(frames), niche), work, model)).get("concepts", [])
        got = clean_concepts(raw, len(frames), langs, fallback)
        for c in got:
            log(f"concept: {c['layout']}, frame {c['subject']}, “{c['words'][langs[0]]['main']}” — {c['why']}")
        return got[:n] if len(got) >= n else got + fallback[len(got):n]
    except AutocutError as e:
        log(f"the art director could not plan ({e}); using rules")
        return fallback


def critique(images: list[Path], work: Path, model: str | None = None) -> list[int]:
    """The concepts ranked by how likely they get the click (indexes into images, best first)."""
    from PIL import Image

    names = []
    for i, p in enumerate(images):
        im = Image.open(p).convert("RGB")
        im.thumbnail((640, 640))
        f = work / f"concept-{i + 1}.jpg"
        im.save(f, "JPEG", quality=88)
        names.append(f.name)
    prompt = (f"Open these thumbnails with the Read tool: {', '.join(names)}. They are concepts for one video, seen at feed size. Rank them by how "
              "likely a viewer clicks (focal point, emotion, readability of the words at a glance, contrast, curiosity). Answer only JSON: "
              '{"ranking": [best concept number, ...], "notes": {"1": "one short line"}}')
    try:
        import script_writer as sw

        res = sw.extract_json(ask_claude_seeing(prompt, work, model, timeout=300))
        order = [int(x) - 1 for x in res.get("ranking", []) if str(x).isdigit() and 1 <= int(x) <= len(images)]
        order = list(dict.fromkeys(order)) + [i for i in range(len(images)) if i not in order]
        for i in order:
            note = (res.get("notes") or {}).get(str(i + 1))
            if note:
                log(f"critic on concept {i + 1}: {note}")
        return order
    except (AutocutError, ValueError) as e:
        log(f"the critic could not rank ({e}); keeping the planned order")
        return list(range(len(images)))


# ---------------------------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------------------------


def dilate(a, px: int):
    from PIL import ImageFilter

    px = max(1, int(px))
    out = a
    while px > 0:  # (MaxFilter sizes must be odd and small: grow in steps)
        k = min(px, 6)
        out = out.filter(ImageFilter.MaxFilter(2 * k + 1))
        px -= k
    return out


def styled_subject(layer, size_h: int, palette: list, max_scale: float = 1.7):
    """The cut-out at a height, with outline, glow and rim light: (RGBA, its size)."""
    from PIL import Image, ImageChops, ImageFilter

    w, h = layer.size
    k = min(size_h / max(1, h), max_scale)
    layer = layer.resize((max(1, int(w * k)), max(1, int(h * k))), Image.LANCZOS)
    pad = int(0.08 * layer.height) + 20
    canvas = Image.new("RGBA", (layer.width + 2 * pad, layer.height + 2 * pad), (0, 0, 0, 0))
    a = Image.new("L", canvas.size, 0)
    a.paste(layer.getchannel("A"), (pad, pad))
    glow_c = hexrgb(palette[0] if palette else "", (255, 210, 0))
    rim_c = hexrgb(palette[1] if len(palette) > 1 else "", (0, 220, 255))
    stroke = max(4, layer.height // 90)
    glow = dilate(a, stroke * 3).filter(ImageFilter.GaussianBlur(stroke * 4))
    canvas.paste(Image.new("RGBA", canvas.size, (*glow_c, 255)), (0, 0), glow.point(lambda v: int(v * 0.85)))
    outline = dilate(a, stroke)
    canvas.paste(Image.new("RGBA", canvas.size, (255, 255, 255, 255)), (0, 0), outline)
    canvas.alpha_composite(layer, (pad, pad))
    # rim light: the cut-out's edge in the second colour, on the side away from the words
    inner = a.filter(ImageFilter.MinFilter(9))
    rim = ImageChops.subtract(a, inner).filter(ImageFilter.GaussianBlur(3))
    canvas.paste(Image.new("RGBA", canvas.size, (*rim_c, 255)), (0, 0), rim.point(lambda v: int(v * 0.55)))
    return canvas


def background(W: int, H: int, gen: str | None, frame: Path | None, palette: list, focus: tuple[float, float]):
    """Generated (or a blurred frame) background with depth: graded, light rays and a glow behind the subject, vignette."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

    if gen and Path(gen).is_file():
        bg = Image.open(gen).convert("RGB")
        blur = W / 640
    elif frame and Path(frame).is_file():
        bg = Image.open(frame).convert("RGB")
        blur = W / 90
    else:
        bg = Image.new("RGB", (W, H), (20, 20, 30))
        blur = 0
    k = max(W / bg.width, H / bg.height)
    bg = bg.resize((int(bg.width * k) + 1, int(bg.height * k) + 1), Image.LANCZOS)
    x, y = (bg.width - W) // 2, (bg.height - H) // 2
    bg = bg.crop((x, y, x + W, y + H))
    if blur:
        bg = bg.filter(ImageFilter.GaussianBlur(blur))
    bg = ImageEnhance.Contrast(ImageEnhance.Color(bg).enhance(1.25)).enhance(1.1)
    bg = ImageEnhance.Brightness(bg).enhance(0.72)
    c1 = hexrgb(palette[0] if palette else "", (255, 200, 0))
    fx, fy = focus[0] * W, focus[1] * H
    # light rays from behind the subject
    rays = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(rays)
    R = math.hypot(W, H)
    for i in range(18):
        a0 = i * (2 * math.pi / 18)
        a1 = a0 + math.pi / 30
        d.polygon([(fx, fy), (fx + R * math.cos(a0), fy + R * math.sin(a0)), (fx + R * math.cos(a1), fy + R * math.sin(a1))], fill=70)
    rays = rays.filter(ImageFilter.GaussianBlur(W / 120))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dist = np.hypot((xx - fx) / W, (yy - fy) / H)
    glow = np.clip(1 - dist / 0.55, 0, 1) ** 1.6
    ray = np.asarray(rays, dtype=np.float32) / 255 * np.clip(1 - dist / 0.9, 0, 1)
    arr = np.asarray(bg, dtype=np.float32)
    light = (glow * 0.55 + ray * 0.6)[..., None] * np.asarray(c1, dtype=np.float32)[None, None, :]
    arr = arr + light
    vig = np.clip(1.0 - 0.45 * np.clip(np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)) - 0.45, 0, None) ** 1.5, 0.35, 1.0)
    arr = arr * vig[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8)).convert("RGBA")


def ass_text(lines: list[tuple[str, int]], hl: set[str], lang: str, fs: float, work: Path, tag: str, size: tuple[int, int], colors: dict):
    """Fill-only words (no outline: that is built after) on a transparent layer: RGBA."""
    from PIL import Image

    W, H = size
    font = font_family(lang)
    ev = []
    y = 10.0
    for text, rel in lines:
        size_px = fs * rel / 100
        parts = []
        for w in text.split(" "):
            key = re.sub(r"[?!.,:;]+$", "", w).lower()
            col = colors["hl"] if key and key in hl else colors["fill"]
            parts.append(f"{{\\c&H00{col[2]:02X}{col[1]:02X}{col[0]:02X}&}}{w.replace('{', '(').replace('}', ')')}")
        ev.append(f"Dialogue: 0,0:00:00.00,0:00:05.00,s,,0,0,0,,{{\\pos(20,{y:.0f})\\an7\\fs{size_px:.0f}\\bord0\\shad0}}{' '.join(parts)}")
        y += size_px * (1.22 if lang == "hi" else 1.0)
    ass = ("[Script Info]\nScriptType: v4.00+\n" f"PlayResX: {W}\nPlayResY: {H}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
           "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
           "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
           f"Style: s,{font},{fs:.0f},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,{1 if lang != 'hi' else 0},0,1,0,0,7,0,0,0,1\n\n"
           "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n" + "\n".join(ev) + "\n")
    f = work / f"{tag}.ass"
    f.write_text(ass, encoding="utf-8")
    png = work / f"{tag}.png"
    esc = lambda p: str(p).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")  # noqa: E731
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c=black@0.0:s={W}x{H},format=rgba",
           "-vf", f"ass=filename='{esc(f)}':fontsdir='{esc(fonts_dir())}':alpha=1", "-frames:v", "1", str(png)]
    if subprocess.run(cmd, capture_output=True).returncode != 0 or not png.is_file():
        raise AutocutError("could not draw the words (ffmpeg with libass is needed)")
    im = Image.open(png).convert("RGBA")
    b = im.getbbox()
    return im.crop(b) if b else None


def pro_words(words: dict, lang: str, palette: list, box_w: int, box_h: int, work: Path, tag: str, vertical: bool):
    """The words as a finished layer: kicker slab + main words with gradient fill, stroke, 3-D depth
    and shadow, tilted; sized to fit box_w × box_h."""
    from PIL import Image, ImageDraw, ImageFilter

    main = re.sub(r"\s+", " ", str(words.get("main", ""))).strip()
    kicker = re.sub(r"\s+", " ", str(words.get("kicker", ""))).strip()
    if not main:
        return None
    hl = {re.sub(r"[?!.,:;]+$", "", w).lower() for w in str(words.get("highlight", "")).split()}
    accent = bright(hexrgb(palette[0] if palette else "", (255, 212, 0)))
    second = hexrgb(palette[1] if len(palette) > 1 else "", (224, 30, 55))
    if luma(second) > 200:  # (a white slab would hide white words)
        second = (224, 30, 55)
    ws = main.split()
    # one line or two (each split tried): whichever lets the words be biggest in this box
    options = [[(main, 100)]] + [[(" ".join(ws[:i]), 100), (" ".join(ws[i:]), 100)] for i in range(1, len(ws))]
    fits = []
    for opt in options:
        probe_fs = box_h / len(opt) * 0.8
        t = ass_text(opt, hl, lang, probe_fs, work, tag + "-probe", (int(box_w * 3) + 400, int(box_h * 3) + 400), {"fill": (255, 255, 255), "hl": accent})
        if t is not None:
            fits.append((probe_fs * min(box_w / t.width, box_h / t.height), opt))
    top = max((f for f, _ in fits), default=0.0)
    # one line unless two are clearly bigger; among near-equal splits, the most balanced
    near = [(f, o) for f, o in fits if f >= top * 0.92]
    one = [x for x in near if len(x[1]) == 1]
    pick = one[0] if one else min(near, key=lambda x: abs(len(x[1][0][0]) - len(x[1][-1][0])), default=(0.0, options[0]))
    lines = pick[1]
    fs = pick[0] * 0.97 if pick[0] else box_h / len(lines) * 0.8
    fill = None
    for _ in range(5):
        fill = ass_text(lines, hl, lang, fs, work, tag, (int(box_w * 2.2) + 400, int(box_h * 2.2) + 400), {"fill": (255, 255, 255), "hl": accent})
        if fill is None:
            return None
        k = min(box_w / fill.width, box_h / fill.height)
        if 0.95 <= k <= 1.04:
            break
        fs *= min(k, 1.6) * 0.99
    if fill is None:
        return None
    # gradient: lighter at the top of each glyph, deeper at the bottom
    w, h = fill.size
    grad = np.linspace(1.08, 0.82, h, dtype=np.float32)[:, None, None]
    arr = np.asarray(fill, dtype=np.float32)
    arr[..., :3] = np.clip(arr[..., :3] * grad, 0, 255)
    fill = Image.fromarray(arr.astype(np.uint8))
    a = fill.getchannel("A")
    stroke = max(5, int(fs * 0.075))
    depth = max(4, int(fs * 0.06))
    pad = stroke * 2 + depth * 2 + 30
    W2, H2 = w + 2 * pad, h + 2 * pad
    canvas = Image.new("RGBA", (W2, H2), (0, 0, 0, 0))
    A = Image.new("L", (W2, H2), 0)
    A.paste(a, (pad, pad))
    outer = dilate(A, stroke)
    # soft shadow, then 3-D extrusion (the stroke shape pushed down-right in a dark tone)
    shadow = outer.filter(ImageFilter.GaussianBlur(stroke * 1.4))
    canvas.paste(Image.new("RGBA", (W2, H2), (0, 0, 0, 255)), (depth + 4, depth + 6), shadow.point(lambda v: int(v * 0.75)))
    dark = tuple(int(c * 0.35) for c in second)
    for i in range(depth, 0, -1):
        canvas.paste(Image.new("RGBA", (W2, H2), (*dark, 255)), (i, i), outer)
    canvas.paste(Image.new("RGBA", (W2, H2), (0, 0, 0, 255)), (0, 0), outer)
    canvas.alpha_composite(fill, (pad, pad))
    if kicker:
        ink = (0, 0, 0) if luma(second) > 150 else (255, 255, 255)
        kf = ass_text([(kicker.upper() if lang != "hi" else kicker, 100)], set(), lang, fs * 0.36, work, tag + "-k", (W2 * 2, int(fs) + 200),
                      {"fill": ink, "hl": ink})
        if kf is not None:
            px, py = int(fs * 0.12), int(fs * 0.06)
            slab = Image.new("RGBA", (kf.width + 2 * px, kf.height + 2 * py), (*second, 255))
            slab.alpha_composite(kf, (px, py))
            out = Image.new("RGBA", (max(W2, slab.width + pad), H2 + slab.height + 6), (0, 0, 0, 0))
            out.alpha_composite(slab, (pad, 0))
            out.alpha_composite(canvas, (0, slab.height + 6 - pad // 2) if slab.height + 6 - pad // 2 >= 0 else (0, 0))
            canvas = out
    canvas = canvas.rotate(3.5 if not vertical else 2.5, resample=Image.BICUBIC, expand=True)
    b = canvas.getbbox()
    canvas = canvas.crop(b) if b else canvas
    k = min(box_w / canvas.width, box_h * 1.35 / canvas.height, 1.0)
    if k < 1:
        canvas = canvas.resize((int(canvas.width * k), int(canvas.height * k)), Image.LANCZOS)
    return canvas


def arrow(size: int, color: tuple[int, int, int]):
    """A fat curved arrow pointing right (RGBA), with a dark outline."""
    from PIL import Image, ImageDraw

    s = size
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    pts = []
    for t in np.linspace(0, 1, 30):  # the shaft: an arc
        x = 0.08 * s + t * 0.62 * s
        y = 0.75 * s - math.sin(t * math.pi * 0.85) * 0.35 * s
        pts.append((x, y))
    d = ImageDraw.Draw(im)
    tip = pts[-1]
    head = [(tip[0] - 0.05 * s, tip[1] - 0.17 * s), (tip[0] + 0.24 * s, tip[1] + 0.02 * s), (tip[0] - 0.02 * s, tip[1] + 0.19 * s)]
    for width, col in ((int(0.16 * s), (0, 0, 0)), (int(0.1 * s), color)):
        d.line(pts, fill=(*col, 255), width=width, joint="curve")
        grow = 0.02 * s if col == (0, 0, 0) else 0
        h2 = [(head[0][0] - grow, head[0][1] - grow), (head[1][0] + grow * 1.5, head[1][1]), (head[2][0] - grow, head[2][1] + grow)]
        d.polygon(h2, fill=(*col, 255))
    return im


def badge(text: str, h: int, color: tuple[int, int, int], work: Path, tag: str):
    from PIL import Image, ImageDraw

    if luma(color) > 200:  # (white words need a coloured pill)
        color = (224, 30, 55)
    ink = (0, 0, 0) if luma(color) > 150 else (255, 255, 255)
    t = ass_text([(text.upper(), 100)], set(), "en", h * 0.62, work, tag, (h * 12, h * 3), {"fill": ink, "hl": ink})
    if t is None:
        return None
    px = int(h * 0.3)
    im = Image.new("RGBA", (t.width + 2 * px, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, im.width - 1, h - 1), radius=h // 5, fill=(*color, 255), outline=(255, 255, 255, 255), width=max(3, h // 18))
    im.alpha_composite(t, (px, (h - t.height) // 2))
    return im.rotate(-4, resample=Image.BICUBIC, expand=True)


def ring(w: int, h: int, color: tuple[int, int, int]):
    from PIL import Image, ImageDraw

    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    lw = max(6, w // 30)
    d.ellipse((lw, lw, w - lw, h - lw), outline=(0, 0, 0, 255), width=lw + 6)
    d.ellipse((lw, lw, w - lw, h - lw), outline=(*color, 255), width=lw)
    return im


def compose(concept: dict, assets: dict, platform: str, lang: str, size: tuple[int, int], out: Path, work: Path, gen_bg: str | None) -> dict:
    """One finished thumbnail for a concept."""
    from PIL import Image, ImageEnhance, ImageFilter

    W, H = size
    vertical = H > W
    pal = concept["palette"]
    layout = concept["layout"]
    subj = assets.get("subject")
    sec = assets.get("second") if layout == "versus" else None
    prop = assets.get("prop")
    right = layout != "hero-left"
    # where the subject stands, and where the light comes from
    if vertical:
        focus = (0.5, 0.66)
    elif layout == "versus":
        focus = (0.5, 0.55)
    else:
        focus = (0.7 if right else 0.3, 0.5)
    canvas = background(W, H, gen_bg, assets.get("frame_bg"), pal, focus)
    notes: dict = {"layout": layout}
    safe_right = W * 0.78 if platform == "youtube" else W  # (the duration badge)

    def place(layer, x, y):
        canvas.alpha_composite(layer, (int(x), int(y)))

    # the subject(s)
    if subj is not None:
        sh = int(H * (0.98 if not vertical else 0.52)) if layout != "big-number" else int(H * (0.9 if not vertical else 0.52))
        s1 = styled_subject(subj, sh, pal)
        if vertical:
            x = (W - s1.width) / 2
            y = H - s1.height + int(H * 0.02)
        elif layout == "versus":
            s1 = styled_subject(subj, int(H * 0.9), pal)
            x, y = -s1.width * 0.08, H - s1.height + H * 0.06
        else:
            x = (W - s1.width * 0.92) if right else -s1.width * 0.08
            y = H - s1.height + H * 0.06
        place(s1, x, y)
        notes["subject_box"] = (int(x), int(y), int(x + s1.width), int(y + s1.height))
        if sec is not None:
            s2 = styled_subject(sec, int(H * 0.9), pal[::-1] if len(pal) > 1 else pal).transpose(Image.FLIP_LEFT_RIGHT)
            place(s2, W - s2.width * 0.92, H - s2.height + H * 0.06)
            vs = badge("VS", int(H * 0.2), hexrgb(pal[1] if len(pal) > 1 else "", (224, 30, 55)), work, f"vs-{platform}-{lang}")
            if vs is not None:
                place(vs, (W - vs.width) / 2, H * 0.42)
    # the prop, with an arrow or a ring
    if prop is not None and layout in ("reaction-object", "hero-right", "hero-left", "big-number"):
        ph = int(H * (0.38 if not vertical else 0.22))
        p1 = styled_subject(prop, ph, pal[::-1] if len(pal) > 1 else pal, max_scale=2.5).rotate(-8, resample=Image.BICUBIC, expand=True)
        if vertical:
            px_, py_ = min(W * 0.56, W * 0.97 - p1.width), H * 0.36
        else:
            px_ = (W * 0.06) if right else (W * 0.94 - p1.width)
            py_ = H * 0.52
        place(p1, px_, py_)
        if concept.get("ring"):
            rg = ring(int(p1.width * 1.15), int(p1.height * 1.15), hexrgb(pal[1] if len(pal) > 1 else "", (224, 30, 55)))
            place(rg, px_ - p1.width * 0.07, py_ - p1.height * 0.07)
        if concept.get("arrow"):
            ar = arrow(int(H * (0.26 if not vertical else 0.14)), hexrgb(pal[1] if len(pal) > 1 else "", (224, 30, 55)))
            if not right and not vertical:
                ar = ar.transpose(Image.FLIP_LEFT_RIGHT)
            ax = (px_ + p1.width * 0.6) if right or vertical else (px_ - ar.width * 0.6)
            ax = min(max(ax, W * 0.02), W * 0.98 - ar.width)  # (inside the picture)
            place(ar, ax, max(H * 0.02, py_ - ar.height * 0.75))
    # the words
    if vertical:
        box = (int(W * 0.06), int(H * 0.11), int(W * 0.94), int(H * 0.34))
    elif layout == "versus":
        box = (int(W * 0.18), int(H * 0.04), int(W * 0.82), int(H * 0.36))
    elif layout == "big-number":
        box = (int(W * 0.04), int(H * 0.08), int(W * 0.62) if right else int(safe_right), int(H * 0.78))
        if not right:
            box = (int(W * 0.38), box[1], int(safe_right) - 10, box[3])
    else:
        box = (int(W * 0.04), int(H * 0.06), int(W * 0.56), int(H * 0.62)) if right else (int(W * 0.44), int(H * 0.06), int(safe_right) - 10, int(H * 0.62))
    if prop is not None and not vertical and layout != "big-number":
        box = (box[0], box[1], box[2], min(box[3], int(H * 0.5)))
    words = concept["words"].get(lang) or next(iter(concept["words"].values()))
    tl = pro_words(words, lang, pal, box[2] - box[0], box[3] - box[1], work, f"w-{platform}-{lang}-{out.stem}", vertical)
    if tl is not None:
        tx = box[0] + ((box[2] - box[0] - tl.width) // 2 if vertical or layout == "versus" else (0 if right else box[2] - box[0] - tl.width))
        ty = box[1] + max(0, (box[3] - box[1] - tl.height) // (2 if not vertical else 3))
        place(tl, tx, ty)
        notes["words_box"] = (int(tx), int(ty), int(tx + tl.width), int(ty + tl.height))
        notes["words_height"] = round(tl.height / H, 2)
    if concept.get("badge") and layout != "versus":
        bd = badge(concept["badge"], int(H * (0.15 if not vertical else 0.065)), hexrgb(pal[1] if len(pal) > 1 else "", (224, 30, 55)), work, f"b-{platform}-{lang}-{out.stem}")
        if bd is not None:
            # right under the words (never over them, never in the duration badge's corner)
            wb = notes.get("words_box") or (box[0], box[1], box[2], box[3])
            bx = (wb[0] + wb[2] - bd.width) / 2 if vertical else (wb[0] if right else max(wb[0], min(wb[2], safe_right) - bd.width))
            by = wb[3] + H * 0.02
            limit = H * (0.78 if platform == "youtube" and bx + bd.width > W * 0.78 else 0.96)
            if by + bd.height <= limit:
                place(bd, bx, by)
    img = canvas.convert("RGB")
    img = ImageEnhance.Color(img).enhance(1.08).filter(ImageFilter.UnsharpMask(radius=1.5, percent=60, threshold=2))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "JPEG", quality=92, optimize=True, progressive=True)
    return notes


# ---------------------------------------------------------------------------------------------
# All of it
# ---------------------------------------------------------------------------------------------


def make(frames: list[dict], topic: str, hook: str, copy: dict, platforms: dict, langs: list[str], variants: int, folder: Path, work: Path,
         niche: str = "", writer: str = "claude", model: str | None = None) -> tuple[list[dict], list[str], list[dict]]:
    """Every thumbnail: ([{platform, language, variant, path, …}], credits, concepts)."""
    concepts = plan(frames, topic, hook, copy, langs, variants, niche, work, writer, model)
    # cut-outs of the chosen subjects (once per frame)
    cuts: dict[int, object] = {}

    def subject_of(no: int | None):
        if not no or no > len(frames):
            return None
        if no not in cuts:
            f = frames[no - 1]
            m = cutout(Path(f["path"]), f["det"].get("text"))
            cuts[no] = subject_layer(Path(f["path"]), m) if m is not None and 0.02 < float(m.mean()) < 0.9 else None
            if cuts[no] is None:
                log(f"frame {no}: nothing clear to cut out")
        return cuts[no]

    credits: list[str] = []
    assets: list[dict] = []
    for k, c in enumerate(concepts):
        a: dict = {"subject": subject_of(c["subject"]), "second": subject_of(c.get("second")) if c["layout"] == "versus" else None}
        if a["subject"] is None:  # the best frame that does cut out
            for no in range(1, len(frames) + 1):
                if subject_of(no) is not None:
                    a["subject"] = cuts[no]
                    break
        bgf = c["background"].get("frame") or c["subject"]
        a["frame_bg"] = Path(frames[bgf - 1]["path"]) if bgf and bgf <= len(frames) else None
        a["prop"] = None
        for p in c["props"]:
            got = find_prop(p["query"], work, f"{k}")
            if got:
                a["prop"] = got["layer"]
                credits.append(got["credit"])
        assets.append(a)
    # backgrounds: one per concept and shape
    shapes = {s for s in platforms.values()}
    jobs, where = [], {}
    for k, c in enumerate(concepts):
        if c["background"]["type"] != "generate":
            continue
        for (W, H) in shapes:
            gw, gh = gen_size(W, H)
            out = str(work / f"bg-{k}-{W}x{H}.png")
            jobs.append((c["background"]["prompt"] + ", cinematic, dramatic lighting, highly detailed, no text, no people, no watermark", gw, gh, out, 1000 + k))
            where[(k, W, H)] = out
    made_bgs = set(generate_backgrounds(jobs))
    # the critic ranks the concepts on the first shape and language
    first_p = next(iter(platforms))
    trial = []
    for k, c in enumerate(concepts):
        W, H = platforms[first_p]
        f = work / f"trial-{k}.jpg"
        bg = where.get((k, W, H))
        compose(c, assets[k], first_p, langs[0], (W, H), f, work, bg if bg in made_bgs else None)
        trial.append(f)
    order = critique(trial, work, model) if writer == "claude" and len(trial) > 1 else list(range(len(trial)))
    made = []
    total = len(platforms) * len(langs) * len(concepts)
    n = 0
    for p, (W, H) in platforms.items():
        for l in langs:
            for rank, k in enumerate(order, 1):
                c = concepts[k]
                out = folder / "thumbnails" / f"{p}-{l}-{rank}.jpg"
                bg = where.get((k, W, H))
                notes = compose(c, assets[k], p, l, (W, H), out, work, bg if bg in made_bgs else None)
                made.append({"platform": p, "language": l, "variant": rank, "path": str(out), "concept": k + 1, "words": c["words"].get(l, {}),
                             "why": c.get("why", ""), **{x: v for x, v in notes.items() if x not in ("subject_box", "words_box")}})
                n += 1
                ac.stage_progress(n / max(1, total), f"{n} of {total} thumbnails")
    return made, credits, [concepts[k] for k in order]
