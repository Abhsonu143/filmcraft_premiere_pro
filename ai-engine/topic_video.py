#!/usr/bin/env python3
"""topic_video: a topic (or a link to a post/article) -> researched script -> voiceover -> edited video.

    python topic_video.py "GTA 6 release" --seconds 60 --voice test1
    python topic_video.py https://example.com/article --seconds 45 --script-only
    python topic_video.py "Chandrayaan-3 landing" --language hinglish --aspect vertical --voice mac:Lekha

1. Research. Claude (through the Claude Code app you are logged into; it can search and read the web)
   finds the facts with their source links, the angle, related articles, image search ideas and
   primary video sources (official trailers, agency footage). With a link, that page is read
   first. Offline (`--writer local`): only the given link is read, and Qwen writes from it.
2. Script. Written from those facts only, in your talking style (script_writer.py), sized to the
   length and to the voice's pace; every line carries an emotion and a shot idea.
3. Media, from the sources you allow:
     free      Wikimedia Commons and Openverse images/videos with open licences (no "no
               derivatives" licences); licence and author recorded for the credits
     youtube   short sections of the YouTube videos the research found. Opt-in: only where you
               have the right to use them (e.g. commentary or news); every source is listed.
     footage   your own clips/folders
     cards     headline and fact cards made from the research (always available)
   Images become slow push-in shots. Every item remembers which line it was found for.
4. Voice: a cloned voice, a macOS voice (`mac`, `mac:Lekha`, `mac:Rishi` … quick, no clone), or
   none (`--script-only`: research + script + sources only).
5. Edit: the voiceover editor (narrate.py) chooses the shot for each line, preferring the media
   found for that line; captions and music optional.

Output folder: script (.script.json + .txt), sources.md (facts with links, media with licences),
description.txt (ready to paste, with credits), media/, narration .wav and the video.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import autocut as ac
from autocut import AutocutError, log

UA = "EI/0.1 (local video tool; personal use)"
ASPECTS = {"horizontal": (1920, 1080), "vertical": (1080, 1920), "square": (1080, 1080)}
SOURCES = ("free", "youtube", "cards")


# ---------------------------------------------------------------------------------------------
# Web helpers
# ---------------------------------------------------------------------------------------------


def _ssl_context():
    """Certificates from certifi: python.org's Python does not use the macOS keychain."""
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:  # noqa: S310 (http/https only, checked by callers)
        return r.read(25_000_000)


def page_text(url: str, limit: int = 12000) -> tuple[str, str]:
    """(title, readable text) of a web page: scripts/styles/navigation removed."""
    if not re.match(r"https?://", url):
        raise AutocutError(f"not a web link: {url}")
    raw = http_get(url).decode("utf-8", errors="replace")
    title = html.unescape(re.search(r"<title[^>]*>(.*?)</title>", raw, re.S | re.I).group(1)).strip() if re.search(r"<title", raw, re.I) else url
    body = re.sub(r"(?is)<(script|style|nav|header|footer|aside|form|noscript)[^>]*>.*?</\1>", " ", raw)
    paras = [html.unescape(re.sub(r"<[^>]+>", " ", p)) for p in re.findall(r"(?is)<(?:p|h1|h2|h3|li)[^>]*>(.*?)</(?:p|h1|h2|h3|li)>", body)]
    text = "\n".join(re.sub(r"\s+", " ", p).strip() for p in paras if len(p.split()) >= 6)
    return re.sub(r"\s+", " ", title)[:200], text[:limit]


# ---------------------------------------------------------------------------------------------
# 1. Research
# ---------------------------------------------------------------------------------------------


def hints_block(hints: str) -> str:
    """The creator's hints: what they are looking for and how the video should come out."""
    h = re.sub(r"[ \t]+", " ", hints or "").strip()
    if not h:
        return ""
    return ("\n\n## The creator's hints (follow them: they say what this video should focus on, what to find out and how it should come out)\n"
            + "\n".join(f"- {l.strip(' -•')}" for l in h.splitlines() if l.strip()) + "\n")


def research_prompt(topic: str, seconds: float, language: str, page: tuple[str, str] | None, hints: str = "") -> str:
    ref = ""
    if page:
        ref = f"\n\nThe creator gave this reference post/article. Read it (fetch the link for more if useful):\nTitle: {page[0]}\n{page[1][:6000]}\n"
    return f"""Research this topic for a {seconds:.0f}-second video voiceover (language of the video: {language}). Today is {time.strftime('%d %B %Y')}.
Topic: {topic}{ref}{hints_block(hints)}
Use web search to get current, accurate information from reliable sources (official sites, major news, primary sources).
Prefer the latest facts; note dates. Never invent facts.

Reply with only a JSON object:
{{"title_idea": "...", "angle": "what makes this interesting now, one sentence",
 "summary": "3-5 sentences",
 "facts": [{{"fact": "one short verifiable fact", "url": "source link", "source": "publisher name"}}],
 "articles": [{{"title": "...", "url": "...", "publisher": "..."}}],
 "youtube": [{{"url": "https://www.youtube.com/watch?v=...", "title": "...", "channel": "...", "why": "official trailer / news report / agency footage"}}],
 "image_queries": ["2-4 word English searches for free-licensed photos of the real subjects: people, places, objects"],
 "keywords": ["names and terms"]}}
Give 6-12 facts, up to 5 articles, up to 4 YouTube videos (official channels first), 6-10 image queries."""


def research(topic: str, seconds: float, language: str, writer: str = "claude", model: str | None = None, hints: str = "") -> dict:
    page = None
    url = topic.strip() if re.match(r"https?://", topic.strip()) else None
    if url:
        try:
            page = page_text(url)
            log(f"reference: {page[0][:80]}")
        except Exception as e:  # unreachable page: research the link anyway
            log(f"could not read the link ({type(e).__name__}); researching from the link itself")
    t0 = time.time()
    if writer == "claude":
        import script_writer as sw

        log("Claude is researching the topic on the web")
        raw = ask_claude_research(research_prompt(topic, seconds, language, page, hints), model)
        try:
            data = sw.extract_json(raw)
        except AutocutError:
            data = {}
    else:
        if not page:
            raise AutocutError("offline research needs a link to read (or choose Claude, which can search the web)")
        data = {"title_idea": page[0], "summary": page[1][:600],
                "facts": [{"fact": s.strip(), "url": url, "source": urllib.parse.urlparse(url).netloc}
                          for s in re.split(r"(?<=[.!?])\s+", page[1]) if 8 <= len(s.split()) <= 40][:12],
                "articles": [{"title": page[0], "url": url, "publisher": urllib.parse.urlparse(url).netloc}], "youtube": [],
                "image_queries": [], "keywords": []}
    if url and page and not any(a.get("url") == url for a in data.get("articles", [])):
        data.setdefault("articles", []).insert(0, {"title": page[0], "url": url, "publisher": urllib.parse.urlparse(url).netloc})
    data["topic"] = topic
    data["facts"] = [f for f in data.get("facts", []) if isinstance(f, dict) and f.get("fact")][:15]
    if not data["facts"]:
        raise AutocutError("the research found no usable facts")
    log(f"research: {len(data['facts'])} facts, {len(data.get('articles', []))} articles, {len(data.get('youtube', []))} videos in {time.time() - t0:.0f}s")
    return data


def ask_claude_research(prompt: str, model: str | None) -> str:
    """Claude Code with web search allowed (only search/read tools; runs in an empty folder)."""
    import shutil
    import tempfile

    exe = shutil.which("claude") or next((p for p in ("/opt/homebrew/bin/claude", "/usr/local/bin/claude", str(Path.home() / ".local/bin/claude")) if Path(p).is_file()), None)
    if not exe:
        raise AutocutError("Claude Code is not installed (needed for web research): https://claude.com/claude-code")
    cmd = [exe, "-p", "--output-format", "json", "--allowedTools", "WebSearch,WebFetch",
           "--system-prompt", "You are a careful research assistant for a video creator. Verify facts against sources. Output only the requested JSON."]
    if model:
        cmd += ["--model", model]
    with tempfile.TemporaryDirectory(prefix="ei-research-") as tmp:
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=900, cwd=tmp)
        except subprocess.TimeoutExpired as e:
            raise AutocutError("the research took too long") from e
    try:
        res = json.loads(proc.stdout)
    except ValueError as e:
        raise AutocutError(f"Claude Code did not answer ({(proc.stderr or proc.stdout).strip()[:200]}); is it logged in?") from e
    if res.get("is_error"):
        raise AutocutError(f"Claude: {str(res.get('result'))[:200]}")
    return str(res.get("result", ""))


def facts_notes(data: dict) -> str:
    lines = [f"- {f['fact']} (source: {f.get('source') or urllib.parse.urlparse(f.get('url', '')).netloc})" for f in data["facts"]]
    angle = f"Angle: {data.get('angle')}\n" if data.get("angle") else ""
    return (f"{angle}Use ONLY these researched facts (do not add facts that are not here; you may leave some out):\n" + "\n".join(lines)
            + "\nSay where something comes from when it matters (\"according to Rockstar\")."
            + "\nThis is a researched video: do not invent personal experiences, memories or opinions presented as facts; ask the viewer instead.")


# ---------------------------------------------------------------------------------------------
# 3. Media
# ---------------------------------------------------------------------------------------------


def strip_tags(t: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t or ""))).strip()


def commons_search(query: str, limit: int = 6, video: bool = False) -> list[dict]:
    """Free-licensed files on Wikimedia Commons."""
    q = f"{query} filetype:{'video' if video else 'bitmap'}"
    url = ("https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=search&gsrnamespace=6"
           f"&gsrlimit={limit}&gsrsearch={urllib.parse.quote(q)}&prop=imageinfo&iiprop=url|mime|size|extmetadata&iiurlwidth=1920")
    try:
        d = json.loads(http_get(url, 20))
    except Exception:
        return []
    out = []
    for p in (d.get("query", {}).get("pages", {}) or {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        m = ii.get("extmetadata", {})
        lic = (m.get("LicenseShortName", {}) or {}).get("value", "")
        if not lic or "nd" in lic.lower().replace("-", " ").split():
            continue
        w = int(ii.get("width") or 0)
        if not video and w < 900:
            continue
        out.append({"kind": "video" if video else "image", "url": ii.get("url") if video else (ii.get("thumburl") or ii.get("url")),
                    "page": ii.get("descriptionurl", ""), "title": (p.get("title") or "").replace("File:", ""), "license": lic,
                    "author": strip_tags((m.get("Artist", {}) or {}).get("value", ""))[:80], "provider": "Wikimedia Commons", "width": w})
    return out


def openverse_search(query: str, limit: int = 6) -> list[dict]:
    """Openly licensed images (Flickr, museums …) via Openverse; licences that forbid edits excluded."""
    url = f"https://api.openverse.org/v1/images/?q={urllib.parse.quote(query)}&page_size={limit}&license_type=modification&size=large"
    try:
        d = json.loads(http_get(url, 20))
    except Exception:
        return []
    out = []
    for r in d.get("results", []):
        if (r.get("width") or 0) and r["width"] < 900:
            continue
        if not r.get("url"):
            continue
        # (Openverse sends missing fields as null: every one may be empty)
        out.append({"kind": "image", "url": r["url"], "page": r.get("foreign_landing_url") or "", "title": (r.get("title") or "")[:100],
                    "license": f"CC {(r.get('license') or '').upper()} {r.get('license_version') or ''}".strip(), "author": r.get("creator") or "",
                    "provider": r.get("source") or "Openverse", "width": r.get("width") or 0})
    return out


def download(url: str, dst: Path) -> bool:
    try:
        data = http_get(url, 60)
    except Exception:
        return False
    if len(data) < 5000:
        return False
    dst.write_bytes(data)
    return True


def image_to_clip(img: Path, dst: Path, W: int, H: int, seconds: float = 6.0, push_in: bool = True) -> bool:
    """A still image as a slow push-in (or pull-out) shot, filling the frame."""
    fps = 30
    n = int(seconds * fps)
    z = f"min(1+0.10*on/{n},1.10)" if push_in else f"max(1.10-0.10*on/{n},1.0)"
    k = 2 if W * H <= 1920 * 1080 else 1.5  # push-in headroom (less at 4K: the frame is already big)
    sw, sh = int(W * k) // 2 * 2, int(H * k) // 2 * 2
    vf = (f"scale={sw}:{sh}:force_original_aspect_ratio=increase,crop={sw}:{sh},"
          f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={n}:s={W}x{H}:fps={fps},format=yuv420p")
    proc = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(img), "-vf", vf, "-frames:v", str(n),
                           "-c:v", "h264_videotoolbox", "-b:v", ac.bitrate_for(W, H, 12.0), str(dst)], capture_output=True)
    return proc.returncode == 0 and dst.is_file()


def font_for(text: str, bold: bool = True) -> str | None:
    cands = (["/System/Library/Fonts/Kohinoor.ttc", "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc"] if re.search(r"[ऀ-ॿ]", text)
             else ["/System/Library/Fonts/Avenir Next.ttc", "/System/Library/Fonts/Helvetica.ttc"])
    return next((c for c in cands if Path(c).is_file()), None)


def make_card(headline: str, source: str, dst: Path, W: int, H: int) -> bool:
    """A headline/fact card: big text on a dark gradient, the source underneath."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    for y in range(H):  # dark blue -> near black
        k = y / H
        d.line([(0, y), (W, y)], fill=(int(18 + 10 * (1 - k)), int(24 + 14 * (1 - k)), int(48 + 30 * (1 - k))))
    fp = font_for(headline)
    size = int(min(W, H) * (0.075 if len(headline) < 70 else 0.058))
    deva = bool(re.search(r"[\u0900-\u097F]", headline))
    # faces inside the .ttc files: Avenir Next 0 = Bold, 5 = Medium; Kohinoor Devanagari 3 = Bold, 1 = Medium
    bold, medium = (3, 1) if deva else (0, 5)
    try:
        font = ImageFont.truetype(fp, size, index=bold if fp and fp.endswith(".ttc") else 0) if fp else ImageFont.load_default()
        small = ImageFont.truetype(fp, int(size * 0.45), index=medium if fp and fp.endswith(".ttc") else 0) if fp else ImageFont.load_default()
    except OSError:
        font = small = ImageFont.load_default()
    words, lines, cur = headline.split(), [], ""
    maxw = W * 0.82
    for w in words:
        trial = (cur + " " + w).strip()
        if d.textlength(trial, font=font) <= maxw:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    lh = int(size * 1.25)
    y = (H - lh * len(lines)) // 2 - lh // 2
    d.rectangle([W * 0.09 - 14, y - 10, W * 0.09 - 6, y + lh * len(lines)], fill=(255, 196, 61))
    for ln in lines:
        d.text((W * 0.09, y), ln, font=font, fill=(255, 255, 255))
        y += lh
    if source:
        d.text((W * 0.09, y + lh // 2), f"Source: {source}", font=small, fill=(180, 190, 210))
    img.save(dst)
    return True


def youtube_sections(url: str, dst_dir: Path, tag: str, n: int = 3, seconds: float = 8.0) -> list[dict]:
    """A few short sections spread through a YouTube video (only those sections are downloaded)."""
    import script_writer as sw

    base = sw.yt_dlp_cmd() + ["--no-playlist", "--quiet", "--no-warnings"]
    try:
        info = json.loads(subprocess.run(base + ["-J", "--skip-download", url], capture_output=True, text=True, timeout=120).stdout)
    except ValueError:
        return []
    dur = float(info.get("duration") or 0)
    if dur < 15:
        return []
    out = []
    for k in range(n):
        a = dur * (0.15 + 0.7 * k / max(1, n - 1)) if n > 1 else dur * 0.3
        a = min(max(2.0, a), max(2.0, dur - seconds - 1))
        dst = dst_dir / f"{tag}_yt{k}.mp4"
        cmd = base + ["-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b", "--download-sections", f"*{a:.1f}-{a + seconds:.1f}",
                      "--force-keyframes-at-cuts", "--merge-output-format", "mp4", "-o", str(dst), url]
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if dst.is_file() and dst.stat().st_size > 50_000:
            out.append({"kind": "video", "file": str(dst), "url": url, "page": url, "title": info.get("title", ""), "license": "YouTube (your responsibility)",
                        "author": info.get("channel") or info.get("uploader") or "", "provider": "YouTube", "at": round(a, 1)})
    return out


def collect_media(script: dict, data: dict, folder: Path, sources: list[str], W: int, H: int, per_line: int = 2) -> tuple[list[dict], dict[str, list[int]]]:
    """Downloads/creates media for the script. Returns (items, hints: clip path -> line indices)."""
    import narrate as nr

    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / "originals"
    raw.mkdir(exist_ok=True)
    items: list[dict] = []
    hints: dict[str, list[int]] = {}
    seen: set[str] = set()
    segs = script["segments"]
    topic_words = " ".join(data.get("keywords", [])[:2])
    t0 = time.time()
    if "free" in sources:
        # the researcher's short searches for the real subjects first, each tied to the line that
        # mentions it; then a short search per line still without media
        def words(t: str) -> set[str]:
            return {w for w in re.findall(r"[\w-]{3,}", t.lower()) if w not in {"the", "and", "with", "from", "this", "that"}}

        seg_words = [words(s["text"] + " " + (s.get("visual") or "")) for s in segs]
        queries: list[tuple[int | None, str]] = []
        for q in data.get("image_queries", [])[:10]:
            qw = words(q)
            best = max(range(len(segs)), key=lambda i: len(qw & seg_words[i])) if segs else None
            queries.append((best if best is not None and qw & seg_words[best] else None, q))
        covered = {i for i, _ in queries if i is not None}
        for i, s in enumerate(segs):
            if i not in covered:
                subj = nr.subject_only(s.get("visual") or s["text"]).split()
                queries.append((i, " ".join(([topic_words] if topic_words else []) + subj[:3])))
        queries = [(i, " ".join(q.split()[:6])) for i, q in queries if q.strip()]

        def search(q: str) -> list[dict]:
            out: list[dict] = []
            for fn in (commons_search, openverse_search):  # (one odd answer never stops the video)
                try:
                    out += fn(q, 5)
                except Exception as e:
                    log(f"picture search “{q[:40]}” ({fn.__name__.split('_')[0]}): skipped ({e.__class__.__name__})")
            return out

        # all searches at once (the picture sites answer slowly, not heavily)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=6) as ex:
            results = list(ex.map(search, [q for _, q in queries]))
        log(f"searched {len(queries)} subjects in {time.time() - t0:.0f}s")
        for qi, ((i, q), found) in enumerate(zip(queries, results), 1):
            log(f"progress {qi}/{len(queries)} media")
            if i is not None and topic_words and len(found) < 2:
                found += commons_search(f"{topic_words} {q}", 4)
            got = 0
            for c in found:
                if not c.get("url") or c["url"] in seen:
                    continue
                seen.add(c["url"])
                ext = Path(urllib.parse.urlparse(c["url"]).path).suffix.lower() or ".jpg"
                k = len(items)
                uid = hashlib.sha1(c["url"].encode()).hexdigest()[:10]  # the same picture keeps its files: a retry reuses them
                src = raw / f"m{uid}{ext}"
                if not (src.is_file() and src.stat().st_size >= 5000) and not download(c["url"], src):
                    continue
                clip = folder / f"m{uid}_{W}x{H}_{'line' + str(i) if i is not None else 'topic'}.mp4"
                ok = clip.is_file() or (image_to_clip(src, clip, W, H, push_in=k % 2 == 0) if c["kind"] == "image" else _normalise_video(src, clip))
                if not ok:
                    continue
                items.append({**c, "file": str(clip), "original": str(src), "query": q, "line": i})
                if i is not None:
                    hints.setdefault(str(clip.resolve()), []).append(i)
                got += 1
                if got >= (per_line if i is not None else 1):
                    break
        log(f"free-licensed media: {len(items)} items in {time.time() - t0:.0f}s")
    if "youtube" in sources:
        for k, v in enumerate(data.get("youtube", [])[:3]):
            if not re.match(r"https?://(www\.)?(youtube\.com|youtu\.be)/", str(v.get("url", ""))):
                continue
            got = youtube_sections(v["url"], folder, f"v{k}")
            items += got
            log(f"YouTube: {len(got)} section(s) from {v.get('title', v['url'])[:60]}")
    if "cards" in sources:
        facts = data.get("facts", [])
        for k, f in enumerate(facts[:4]):
            png = raw / f"card{k}.png"
            clip = folder / f"card{k}.mp4"
            if make_card(f["fact"], f.get("source") or urllib.parse.urlparse(f.get("url", "")).netloc, png, W, H) and image_to_clip(png, clip, W, H, 5.0, k % 2 == 0):
                items.append({"kind": "card", "file": str(clip), "title": f["fact"], "page": f.get("url", ""), "license": "made by EI",
                              "author": "", "provider": f.get("source", "")})
                # a fact card belongs to the line that says the fact (shared words)
                fw = set(re.findall(r"\w{4,}", f["fact"].lower()))
                best = max(range(len(segs)), key=lambda i: len(fw & set(re.findall(r"\w{4,}", segs[i]["text"].lower())))) if segs else None
                if best is not None:
                    hints.setdefault(str(clip.resolve()), []).append(best)
    return items, hints


def _normalise_video(src: Path, dst: Path) -> bool:
    proc = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-t", "20", "-an", "-vf", "fps=30,format=yuv420p",
                           "-c:v", "h264_videotoolbox", "-b:v", "12M", str(dst)], capture_output=True)
    return proc.returncode == 0 and dst.is_file()


# ---------------------------------------------------------------------------------------------
# Output documents
# ---------------------------------------------------------------------------------------------


def sources_md(data: dict, items: list[dict], script: dict) -> str:
    out = [f"# {script.get('title', data.get('topic'))}", "", f"Topic: {data.get('topic')}", ""]
    if data.get("summary"):
        out += ["## Summary", "", data["summary"], ""]
    out += ["## Facts and sources", ""]
    out += [f"- {f['fact']} — [{f.get('source') or urllib.parse.urlparse(f.get('url', '')).netloc}]({f.get('url', '')})" for f in data.get("facts", [])]
    if data.get("articles"):
        out += ["", "## Articles", ""] + [f"- [{a.get('title')}]({a.get('url')}) — {a.get('publisher', '')}" for a in data["articles"]]
    if data.get("youtube"):
        out += ["", "## Videos found", ""] + [f"- [{v.get('title')}]({v.get('url')}) — {v.get('channel', '')}: {v.get('why', '')}" for v in data["youtube"]]
    if items:
        out += ["", "## Media used in the edit", "", "| File | Kind | Licence | Author | Source |", "|---|---|---|---|---|"]
        out += [f"| {Path(i['file']).name} | {i['kind']} | {i.get('license', '')} | {i.get('author', '')} | {i.get('page') or i.get('url', '')} |" for i in items]
    return "\n".join(out) + "\n"


def description_txt(data: dict, items: list[dict], script: dict) -> str:
    lines = [script.get("title", ""), "", data.get("summary", ""), "", "Sources:"]
    lines += [f"• {a.get('publisher') or a.get('title')}: {a.get('url')}" for a in data.get("articles", [])[:5]]
    credits = [i for i in items if i["kind"] in ("image", "video") and i.get("license")]
    if credits:
        lines += ["", "Credits:"] + [f"• {i.get('title', '')[:60]}, {i.get('author') or 'unknown'}, {i.get('license')} ({i.get('provider')}): {i.get('page') or i.get('url')}" for i in credits]
    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


class State:
    """What a topic folder already has (progress.json): a stage whose inputs did not change is
    not run again, so a stopped or failed video continues where it was."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self.data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    @staticmethod
    def sig(*parts) -> str:
        h = hashlib.sha1()
        for x in parts:
            h.update(x if isinstance(x, bytes) else json.dumps(x, sort_keys=True, default=str).encode())
        return h.hexdigest()[:16]

    def done(self, stage: str, sig: str) -> bool:
        return self.data.get(stage) == sig

    def mark(self, stage: str, sig: str) -> None:
        self.data[stage] = sig
        tmp = self.path.with_suffix(".json.part")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        tmp.replace(self.path)


def make(topic: str, out_dir: str | None = None, *, seconds: float = 60, language: str = "en", aspect: str = "horizontal", voice: str | None = None,
         script_only: bool = False, style: str | None = None, writer: str = "claude", model: str | None = None, tone: str = "", niche: str = "",
         sources: list[str] | None = None, footage: list[str] | None = None, music: str | None = None, captions: bool = True,
         caption_style: str | None = None, takes: int = 2, pace: str = "medium", fresh: bool = False, sound: dict | None = None,
         resolution: str = "1080p", hook_line: str | None = None, find_hook: bool = False, hints: str = "") -> dict:
    import script_writer as sw

    hints = (hints or "").strip()
    ac.require_tools()
    if not topic.strip():
        raise AutocutError("give a topic or a link")
    if aspect not in ASPECTS:
        raise AutocutError(f"aspect must be one of {', '.join(ASPECTS)}")
    if not script_only and not voice:
        raise AutocutError("choose a voice for the video (a cloned voice or a macOS voice like mac:Rishi), or make the script only")
    sources = [s for s in (sources if sources is not None else ["free", "cards"]) if s in SOURCES]
    W, H = ac.sized(*ASPECTS[aspect], resolution)
    t0 = time.time()
    name = sw.safe(re.sub(r"https?://(www\.)?", "", topic))[:50] or "topic"
    folder = Path(out_dir).expanduser().resolve() if out_dir else Path.home() / "Movies" / "EI" / "Topics" / name
    folder.mkdir(parents=True, exist_ok=True)
    state = State(folder / "progress.json")
    ac.progress_plan({"research": 0.08, "hook": 0.02, "script": 0.03} if script_only else
                     {"research": 0.06, "hook": 0.02, "script": 0.02, "voice": 0.42, "media": 0.13, "edit": 0.35})
    # 1. research (kept in the folder: running the same topic again reuses it unless asked to refresh)
    rj = folder / "research.json"
    data = None
    if rj.is_file() and not fresh:
        try:
            old = json.loads(rj.read_text(encoding="utf-8"))
            if old.get("topic") == topic and old.get("facts") and (old.get("hints") or "") == hints:
                data = old
                log(f"reusing the research from {time.strftime('%d %b %H:%M', time.localtime(rj.stat().st_mtime))} (fresh research: --fresh)")
        except ValueError:
            data = None
    with ac.stage("research"):
        if data is None:
            if hints:
                log(f"your hints: {hints[:160]}")
            data = research(topic, seconds, language, writer, model, hints)
            data["hints"] = hints
            data.setdefault("topic", topic)
            rj.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    # 2. script (sized to the voice's pace when there is one)
    clone = voice if (voice and not voice.startswith("mac") and not script_only) else None
    # 2. script: kept when it was written with these same settings (a retry does not rewrite it)
    script_path = folder / f"{name}.script.json"
    script_sig = state.sig(rj.read_bytes(), seconds, language, tone, niche, style, writer, hook_line, find_hook, clone, hints)
    found = []
    if state.done("script", script_sig) and script_path.is_file() and not fresh:
        log(f"reusing the script from {time.strftime('%d %b %H:%M', time.localtime(script_path.stat().st_mtime))}")
        with ac.stage("hook"), ac.stage("script"):
            pass
    else:
        with ac.stage("hook"):
            if hook_line and hook_line.strip():
                log(f"hook: your line “{hook_line.strip()[:80]}”")
            elif find_hook:  # hooks for exactly this topic, looked up online now
                import hooks as hk

                # search with your own words (a link: the researched topic), not the script's title idea
                query = data.get("topic") or data.get("title_idea") or topic if re.match(r"https?://", topic.strip()) else topic
                found = hk.find_for_topic(query, niche, language, facts=facts_notes(data)[:3000], model=model if writer == "claude" else None)
        with ac.stage("script"):
            res = sw.write_script(data.get("title_idea") or topic, str(folder / name), niche=niche, seconds=seconds, language=language, tone=tone,
                                  notes=(hints_block(hints).strip() + "\n\n" if hints else "") + facts_notes(data), style=style, backend=writer, model=model if writer == "claude" else None, pace_voice=clone,
                                  hook_line=hook_line, found_hooks=found)
        script_path = Path(res["script"])
        state.mark("script", script_sig)
    script = json.loads(script_path.read_text(encoding="utf-8"))
    outputs = [str(script_path), str(script_path.with_name(script_path.name.replace(".script.json", ".txt")))]
    items: list[dict] = []
    video_out = None
    if not script_only:
        import threading

        import voice as vo

        # 3 + 4. voice and media at the same time (media needs only the script, not the voice)
        media_sig = state.sig(script_path.read_bytes(), sources, W, H)
        media_json = folder / "media" / "media.json"
        got: dict = {}

        def media_job() -> None:
            try:
                with ac.stage("media"):
                    if state.done("media", media_sig) and media_json.is_file():
                        saved = json.loads(media_json.read_text(encoding="utf-8"))
                        if all(Path(i["file"]).is_file() for i in saved["items"]):
                            log(f"reusing {len(saved['items'])} media item(s)")
                            got["media"] = (saved["items"], saved["hints"])
                            return
                    its, hs = collect_media(script, data, folder / "media", sources, W, H)
                    media_json.write_text(json.dumps({"items": its, "hints": hs}, ensure_ascii=False, indent=1), encoding="utf-8")
                    state.mark("media", media_sig)
                    got["media"] = (its, hs)
            except Exception as e:  # reported after the voice is done
                got["error"] = e

        mt = threading.Thread(target=media_job, name="ei-topic-media", daemon=True)
        mt.start()
        wav = folder / f"{name}.wav"
        voice_sig = state.sig(script_path.read_bytes(), voice, takes)
        with ac.stage("voice"):
            if state.done("voice", voice_sig) and wav.is_file() and wav.with_suffix(".narration.json").is_file():
                log("reusing the narration (the script and voice did not change)")
                sp = {"outputs": [str(wav)], "narration": str(wav.with_suffix(".narration.json"))}
            else:
                sp = vo.speak(voice, str(script_path), str(wav), takes=takes)
                state.mark("voice", voice_sig)
        outputs.append(sp["outputs"][0])
        mt.join()
        if "error" in got:
            raise got["error"]
        items, hints = got["media"]
        clips = [i["file"] for i in items] + list(footage or [])
        if not clips:
            raise AutocutError("no media was found or allowed for this topic: allow more sources or add your own footage")
        # 5. edit
        import narrate as nr

        video_out = folder / f"{name}.mp4"
        stills = {str(Path(i["file"]).resolve()) for i in items if i["kind"] == "image"}
        cards = {str(Path(i["file"]).resolve()) for i in items if i["kind"] == "card"}
        with ac.stage("edit"):
            nr.make(sp["outputs"][0], clips, str(video_out), narration=sp["narration"], aspect=aspect, pace=pace, captions=captions,
                    caption_style=caption_style or ("pop" if aspect == "vertical" else "clean"), music=music, describe="none", hints=hints, stills=stills, cards=cards,
                    ambience_db=-60 if all(i["kind"] != "video" for i in items) else -30, sound=sound, resolution=resolution, niche=niche,
                    # your own hook opens the video: no cold open before it, and it is the on-screen hook too
                    hook="text" if hook_line and hook_line.strip() else "auto", screen_text=hook_line)
        state.mark("edit", "done")
        outputs.insert(0, str(video_out))
    (folder / "sources.md").write_text(sources_md(data, items, script), encoding="utf-8")
    desc = description_txt(data, items, script)
    music_credits = video_out.with_suffix(".credits.txt") if video_out else None
    if music_credits and music_credits.is_file():  # the score's free music must be credited too
        desc = desc.rstrip() + "\n" + music_credits.read_text(encoding="utf-8")
    (folder / "description.txt").write_text(desc, encoding="utf-8")
    outputs += [str(folder / "sources.md"), str(folder / "description.txt")]
    log(f"done in {time.time() - t0:.0f}s: {folder}")
    return {"outputs": outputs, "out_dir": str(folder), "script": str(script_path), "video": str(video_out) if video_out else None,
            "media": len(items), "facts": len(data["facts"])}


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("topic", help="a topic, or a link to a post/article")
    p.add_argument("--out-dir")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--language", default="en", choices=["en", "hi", "hinglish"])
    p.add_argument("--aspect", default="horizontal", choices=list(ASPECTS))
    p.add_argument("--resolution", default="1080p", choices=list(ac.RESOLUTIONS), help="1080p, or 4k in any shape")
    p.add_argument("--voice", help="cloned voice name, or mac / mac:<voice> (e.g. mac:Rishi, mac:Lekha)")
    p.add_argument("--script-only", action="store_true", help="research + script + sources only (no voice, no video)")
    p.add_argument("--style")
    p.add_argument("--writer", default="claude", choices=["claude", "local"])
    p.add_argument("--model")
    p.add_argument("--tone", default="")
    p.add_argument("--niche", default="")
    p.add_argument("--sources", nargs="*", default=["free", "cards"], help=f"media sources: {', '.join(SOURCES)}")
    p.add_argument("--footage", nargs="*", default=[], help="your own clips/folders")
    p.add_argument("--music")
    p.add_argument("--no-captions", action="store_true")
    p.add_argument("--caption-style", choices=["pop", "clean", "boxed"])
    p.add_argument("--takes", type=int, default=2)
    p.add_argument("--pace", default="medium", choices=["slow", "medium", "fast"])
    p.add_argument("--fresh", action="store_true", help="research again even if this folder has research for the topic")
    p.add_argument("--hook-line", help="your own first line, spoken word for word (the rest of the script follows from it)")
    p.add_argument("--find-hook", action="store_true", help="look online for hooks that work for this exact topic first")
    p.add_argument("--hints", default="", help="what you are looking for: angles, facts to find, what to leave out, how it should come out (one per line)")
    import score

    score.add_sound_args(p)


def run_args(a) -> dict:
    footage = None
    if a.footage:
        import voiceover_cut as vc

        footage = vc.collect_clips(a.footage)
    return make(a.topic, a.out_dir, seconds=a.seconds, language=a.language, aspect=a.aspect, voice=a.voice, script_only=a.script_only, style=a.style,
                writer=a.writer, model=a.model, tone=a.tone, niche=a.niche, sources=a.sources, footage=footage, music=a.music,
                captions=not a.no_captions, caption_style=a.caption_style, takes=a.takes, pace=a.pace, fresh=a.fresh,
                sound=__import__("score").options(a), resolution=a.resolution, hook_line=a.hook_line, find_hook=a.find_hook,
                hints=a.hints or "")


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
