#!/usr/bin/env python3
"""explain_media: the real pictures and video of an explainer, found while it is planned.

The plan names what should really be seen (a place, a machine, an event, a person in public
life) and where a drawn picture is better (code, logic, numbers). For each thing to be seen:
1. candidates are collected: your own files first, then free-licensed photos and videos
   (Wikimedia Commons, Openverse);
2. Claude LOOKS at them (a sheet of small pictures per thing) and takes one only when it truly
   shows that thing: a wrong picture is worse than none;
3. what was taken is kept in the video's folder with its credit and page; what was not found is
   listed in the plan, so you can give a file of your own or say to do without.
The script then uses only what was found: full screen, or dimmed behind a card.
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import autocut as ac
from autocut import AutocutError, log

IMAGES = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".tif", ".tiff", ".bmp")
VIDEOS = (".mp4", ".mov", ".m4v", ".webm", ".ogv", ".mkv", ".avi")
MAX_VIDEO_BYTES = 160_000_000
CLIP_SECONDS = 14.0  # the most of one video that is kept
TILE = 480


def commons(query: str, video: bool, limit: int = 5) -> list[dict]:
    """Free-licensed files on Wikimedia Commons, each with a small preview picture (a frame, for a video)."""
    import topic_video as tv

    q = f"{query} filetype:{'video' if video else 'bitmap'}"
    url = ("https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=search&gsrnamespace=6"
           f"&gsrlimit={limit}&gsrsearch={urllib.parse.quote(q)}&prop=imageinfo&iiprop=url|mime|size|extmetadata&iiurlwidth=1920")
    try:
        d = json.loads(tv.http_get(url, 20))
    except Exception:  # noqa: BLE001 (no answer: nothing found)
        return []
    out = []
    for p in sorted((d.get("query", {}).get("pages", {}) or {}).values(), key=lambda p: p.get("index", 0)):
        ii = (p.get("imageinfo") or [{}])[0]
        m = ii.get("extmetadata", {})
        lic = (m.get("LicenseShortName", {}) or {}).get("value", "")
        if not lic or "nd" in lic.lower().replace("-", " ").split() or not ii.get("url"):
            continue
        if (not video and int(ii.get("width") or 0) < 900) or (video and (int(ii.get("size") or 0) > MAX_VIDEO_BYTES * 4 or float(ii.get("duration") or 0) < 2)):
            continue
        out.append({"kind": "video" if video else "photo", "url": ii["url"] if video else (ii.get("thumburl") or ii["url"]), "preview": ii.get("thumburl") or "",
                    "page": ii.get("descriptionurl", ""), "title": (p.get("title") or "").replace("File:", "")[:90], "license": lic,
                    "author": tv.strip_tags((m.get("Artist", {}) or {}).get("value", ""))[:60], "provider": "Wikimedia Commons",
                    "about": tv.strip_tags((m.get("ImageDescription", {}) or {}).get("value", ""))[:200], "bytes": int(ii.get("size") or 0), "seconds": float(ii.get("duration") or 0)})
    return out


def nasa(query: str, video: bool, limit: int = 3) -> list[dict]:
    """NASA's image and video library (its own material is in the public domain; no key needed)."""
    import topic_video as tv

    try:
        d = json.loads(tv.http_get(f"https://images-api.nasa.gov/search?q={urllib.parse.quote(query)}&media_type={'video' if video else 'image'}&page_size={limit * 2}", 20))
    except Exception:  # noqa: BLE001
        return []
    out = []
    for it in (d.get("collection", {}).get("items") or []):
        meta = (it.get("data") or [{}])[0]
        prev = next((ln.get("href", "") for ln in it.get("links") or [] if ln.get("render") == "image" or str(ln.get("href", "")).endswith(".jpg")), "")
        nid = str(meta.get("nasa_id", ""))
        if not nid or not prev or (meta.get("photographer") and "NASA" not in str(meta.get("photographer")) and "NASA" not in str(meta.get("secondary_creator", "NASA"))):
            continue
        out.append({"kind": "video" if video else "photo", "nasa": nid, "url": "", "preview": prev.replace(" ", "%20"), "page": f"https://images.nasa.gov/details/{urllib.parse.quote(nid)}",
                    "title": str(meta.get("title", ""))[:90], "license": "public domain", "author": "NASA" + (f"/{meta['center']}" if meta.get("center") else ""),
                    "provider": "NASA Image and Video Library", "about": " ".join(str(meta.get("description", "")).split())[:200], "bytes": 0})
        if len(out) >= limit:
            break
    return out


def nasa_file(nid: str, video: bool) -> str:
    """The address of a NASA item's file: a medium-size video, or a large picture."""
    import topic_video as tv

    try:
        files = [str(i.get("href", "")) for i in json.loads(tv.http_get(f"https://images-api.nasa.gov/asset/{urllib.parse.quote(nid)}", 20)).get("collection", {}).get("items", [])]
    except Exception:  # noqa: BLE001
        return ""
    for end in (("~medium.mp4", "~mobile.mp4", "~small.mp4", "~orig.mp4") if video else ("~large.jpg", "~medium.jpg", "~orig.jpg")):
        hit = next((f for f in files if f.endswith(end)), "")
        if hit:
            return hit.replace("http://", "https://").replace(" ", "%20")
    return ""


def candidates(need: dict) -> list[dict]:
    import topic_video as tv

    q, video = need["search"], need.get("kind") == "video"
    found = (commons(q, True, 3) + nasa(q, True, 2) if video else []) + commons(q, False, 4 if video else 5) + nasa(q, False, 2)
    if len(found) < 4:
        found += [{**o, "kind": "photo", "preview": o["url"], "about": ""} for o in tv.openverse_search(q, 4)]
    return found[:8]


def fetch(url: str, dst: Path, cap: int = MAX_VIDEO_BYTES) -> bool:
    """A file from the web saved in pieces (a video is never held in memory); False when it fails or is larger than `cap`."""
    import topic_video as tv

    try:
        req = urllib.request.Request(url, headers={"User-Agent": tv.UA})
        with urllib.request.urlopen(req, timeout=60, context=tv._ssl_context()) as r, open(dst, "wb") as f:
            got = 0
            while chunk := r.read(1 << 18):
                got += len(chunk)
                if got > cap:
                    raise OSError("too large")
                f.write(chunk)
        return got > 3000
    except Exception:  # noqa: BLE001
        dst.unlink(missing_ok=True)
        return False


def preview_of(path: Path, dst: Path) -> bool:
    """A small picture of a file of your own (a frame a third of the way in, for a video)."""
    from PIL import Image, ImageOps

    try:
        if path.suffix.lower() in VIDEOS:
            dur = ac.probe(str(path)).duration
            subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{dur / 3:.2f}", "-i", str(path), "-frames:v", "1", "-vf", f"scale={TILE}:-2", str(dst)],
                           capture_output=True, timeout=60)
            return dst.is_file()
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        im.thumbnail((TILE, TILE))
        im.save(dst, quality=85)
        return True
    except Exception:  # noqa: BLE001 (an unreadable file is simply not a candidate)
        return False


def sheet(tiles: list[tuple[str, Path]], dst: Path) -> None:
    """Small pictures side by side, each under its letter, for Claude to look at in one go."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    cols = min(4, len(tiles))
    rows = -(-len(tiles) // cols)
    img = Image.new("RGB", (cols * TILE, rows * (TILE * 3 // 4 + 44)), (30, 30, 30))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    except OSError:
        font = ImageFont.load_default()
    for i, (letter, path) in enumerate(tiles):
        x, y = (i % cols) * TILE, (i // cols) * (TILE * 3 // 4 + 44)
        try:
            im = ImageOps.pad(ImageOps.exif_transpose(Image.open(path)).convert("RGB"), (TILE - 8, TILE * 3 // 4), color=(30, 30, 30))
            img.paste(im, (x + 4, y + 40))
        except Exception:  # noqa: BLE001
            pass
        d.text((x + 10, y + 4), letter, font=font, fill=(255, 220, 60))
    img.save(dst, quality=85)


def look_prompt(needs: list[dict], want: str) -> str:
    blocks = []
    for n in needs:
        rows = "\n".join(f"   {c['letter']}: {'YOUR OWN FILE ' if c.get('own') else ''}{c['kind']}, “{c['title']}”" + (f" — {c['about']}" if c.get("about") else "") for c in n["cands"])
        blocks.append(f"{n['id']}  WANTED: {n['what']}  (for: {n.get('why', '')})\n   sheet: {n['sheet']}\n{rows}")
    return f"""Choose the real pictures and video for an explainer video about: {want}
For each thing wanted below there is a sheet of small candidate pictures (a video is shown by one of its frames). Open every sheet with the Read tool and LOOK.
Take a candidate only when it truly shows the thing wanted: the right place, object, event or person, not something similar (another mission's
rocket, another city's skyline, a generic stock scene). It must be clear at a glance, without watermarks or big text burned in, and not a
chart, logo, screenshot or scanned page unless that is what is wanted. Prefer the creator's own file when it fits. Prefer a video where
movement helps. A WRONG PICTURE IS WORSE THAN NONE: when nothing fits, answer null, say in a few words what would be needed, and give in
"try" up to three other short searches, separated by commas, each 2-4 plain English words an archive might file it under (another name
for it, the agency or maker, the mission or model number).

{chr(10).join(blocks)}

Answer with only this JSON: {{"choices": [{{"id": "m1", "take": "B" or null, "why": "what it shows, or what is missing", "try": "other search words, when null"}}]}} with one entry per thing wanted."""


def best_part(need: dict, raw: Path, dur: float, work: Path, look) -> tuple[float, float] | None:
    """Where the part of a video worth keeping starts and how long it is. Claude looks at ten
    frames spread through it and names the run that shows what is wanted, without title slates,
    logos or end credits. None when it cannot be looked at (the caller then takes a part after the opening)."""
    n = 10
    times = [dur * (i + 0.5) / n for i in range(n)]
    tiles = []
    for i, t in enumerate(times):
        dst = work / f"{need['id']}-t{i}.jpg"
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(raw), "-frames:v", "1", "-vf", f"scale={TILE}:-2", str(dst)],
                       capture_output=True, timeout=60)
        if dst.is_file():
            tiles.append((f"{i + 1}", dst, t))
    if len(tiles) < 3:
        return None
    name = f"{need['id']}-parts.jpg"
    sheet([(label, path) for label, path, _ in tiles], work / name)
    prompt = f"""Open {name} with the Read tool. It shows {len(tiles)} numbered frames in order from one video ({dur:.0f} seconds long, the frames evenly spread).
A part of at most {CLIP_SECONDS:.0f} seconds will be shown in an explainer video, silent, to show: {need['what']}
Name the first and the last frame of the best continuous run for that. Leave out title slates, text cards, logos, black frames and end
credits: the run must begin on the real picture. When no frame shows it, answer null for both.
Answer with only this JSON: {{"first": 3, "last": 5}}"""
    try:
        ans = look(prompt, work)
        if not isinstance(ans, dict) or ans.get("first") is None:
            return (-1.0, 0.0) if isinstance(ans, dict) and "first" in ans else None  # (looked at, and nothing in it shows it)
        by = {label: t for label, _, t in tiles}
        a, b = by[str(int(ans["first"]))], by[str(int(ans.get("last") or ans["first"]))]
    except (AutocutError, KeyError, TypeError, ValueError):
        return None
    half = dur / n / 2
    # (a frame stands for its own stretch; right after a slate, start a little into that stretch to be clear of it)
    start = max(0.0, a - half * (0.5 if int(ans["first"]) > 1 else 1.0))
    return start, max(2.0, min(CLIP_SECONDS, max(a, b) + half - start, dur - start))


def keep(need: dict, cand: dict, media: Path, look=None) -> dict | None:
    """The chosen file in the video's folder: a photo as it is, a video as a short silent clip that
    plays everywhere, cut to the part that shows the thing (looked at, so it never opens on a title slate)."""
    src = Path(cand["path"]) if cand.get("own") else None
    if cand.get("nasa") and not cand.get("url"):
        cand["url"] = nasa_file(cand["nasa"], cand["kind"] == "video")
        if not cand["url"]:
            return None
    if cand["kind"] == "photo":
        dst = media / f"{need['id']}{(src.suffix.lower() if src else '.jpg')}"
        if src:
            dst.write_bytes(src.read_bytes())
        elif not fetch(cand["url"], dst, 40_000_000):
            return None
        return {"file": str(dst), "kind": "photo"}
    raw = src
    if raw is None:
        raw = media / f"{need['id']}.download"
        name = urllib.parse.unquote(cand["url"].rsplit("/", 1)[-1])
        small = cand["url"].replace("/commons/", "/commons/transcoded/", 1) + "/" + urllib.parse.quote(name) + ".720p.vp9.webm"
        if not (("/commons/" in cand["url"] and fetch(small, raw)) or (cand.get("bytes", 0) <= MAX_VIDEO_BYTES and fetch(cand["url"], raw))):
            return None
    dst = media / f"{need['id']}.mp4"
    try:
        dur = ac.probe(str(raw)).duration
        start, length = (0.0 if dur <= CLIP_SECONDS + 2 else min(dur * 0.15, dur - CLIP_SECONDS)), CLIP_SECONDS
        part = best_part(need, raw, dur, media / "candidates", look) if look and dur > 3 else None
        if part and part[0] < 0:
            cand["nothing"] = "the video was looked at frame by frame and no part of it shows this"
            return None
        if part:
            start, length = part
            log(f"  {need['id']}: the video's part from {start:.0f} s for {length:.0f} s is kept (chosen by looking at its frames)")
        ac.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-i", str(raw), "-t", f"{length:.2f}", "-an",
                "-vf", "scale='min(1920,iw)':-2,fps=30,format=yuv420p", *ac.video_encoder_args("h264", "8M"), "-movflags", "+faststart", str(dst)])
        seconds = ac.probe(str(dst)).duration
    except Exception:  # noqa: BLE001 (a file ffmpeg cannot read is not used)
        dst.unlink(missing_ok=True)
        return None
    finally:
        if src is None:
            raw.unlink(missing_ok=True)
    return {"file": str(dst), "kind": "video", "seconds": round(seconds, 2)} if seconds > 1 else None


def credit_of(c: dict) -> str:
    return "your own file" if c.get("own") else ", ".join(x for x in (c.get("title", "")[:60], c.get("author", ""), c.get("license", ""), c.get("provider", "")) if x)


def find(plan: dict, folder: Path, own: list[str] | None = None, model: str | None = None, *, search=None, look=None) -> tuple[int, int]:
    """Fills each of the plan's `media` (things to be seen) with a file that shows it, or the reason
    none was found. What was found before for the same search is kept. (found, wanted)"""
    import script_writer as sw
    import thumb_studio as ts

    search = search or candidates
    look = look or (lambda prompt, where: sw.extract_json(ts.ask_claude_seeing(prompt, where, model, timeout=900)))
    media, cdir = folder / "media", folder / "media" / "candidates"
    needs = [m for m in plan.get("media", []) if not (m.get("file") and Path(m["file"]).is_file() and m.get("for") == m["search"] + "|" + m.get("wanted", "photo"))]
    mine = []
    for p in own or []:
        p = Path(p).expanduser()
        mine += sorted(q for q in (p.iterdir() if p.is_dir() else [p]) if q.suffix.lower() in IMAGES + VIDEOS)
    mine += sorted(q for q in ((media / "mine").iterdir() if (media / "mine").is_dir() else []) if q.suffix.lower() in IMAGES + VIDEOS)
    if needs:
        cdir.mkdir(parents=True, exist_ok=True)
        log(f"looking for {len(needs)} real picture(s) or video(s)" + (f" ({len(mine)} of your own files are candidates)" if mine else ""))
    todo, seen_urls = needs, set()
    for rnd in (1, 2):
        asked = []
        for i, m in enumerate(todo):
            for key in ("file", "credit", "page", "seconds", "missing", "for", "own"):
                m.pop(key, None)
            m["kind"] = m.setdefault("wanted", m.get("kind", "photo"))
            web = []
            for words in ([w.strip() for w in re.split(r"[,;/|]| or ", m.pop("try", "")) if w.strip()][:3] if rnd == 2 else [m["search"]]):
                web += [c for c in (search({**m, "search": words}) or [])[: 8 if rnd == 1 else 4] if (c.get("page") or c.get("url")) not in seen_urls and c not in web]
            seen_urls.update(c.get("page") or c.get("url") for c in web)
            cands = ([{"own": True, "path": str(p), "kind": "video" if p.suffix.lower() in VIDEOS else "photo", "title": p.name[:80]} for p in mine[:6]] if rnd == 1 else []) + web
            tiles = []
            for c in cands:
                letter = chr(ord("A") + len(tiles))
                dst = cdir / f"{m['id']}{'' if rnd == 1 else 'r'}-{letter}.jpg"
                ok = preview_of(Path(c["path"]), dst) if c.get("own") else bool(c.get("preview")) and fetch(c["preview"], dst, 12_000_000)
                if ok:
                    c["letter"] = letter
                    tiles.append((letter, dst))
                if len(tiles) >= 8:
                    break
            m["cands"] = [c for c in cands if c.get("letter")]
            if not tiles:
                m["missing"] = m.get("was_missing") or "nothing free-licensed was found for these words"
                log(f"  {m['id']} {m['what'][:50]}: NOT FOUND ({m['missing'][:90]})")
                continue
            m["sheet"] = f"{m['id']}{'' if rnd == 1 else 'r'}.jpg"
            sheet(tiles, cdir / m["sheet"])
            asked.append(m)
            ac.stage_progress((rnd - 1) * 0.5 + 0.25 * (i + 1) / len(todo))
        todo = []
        if not asked:
            break
        try:
            answer = look(look_prompt(asked, plan.get("want", plan.get("title", ""))), cdir)
        except AutocutError as e:
            answer = {}
            log(f"the candidates could not be looked at ({e}): no picture is used unchecked")
        chosen = {str(c.get("id")): c for c in (answer.get("choices") if isinstance(answer, dict) else None) or [] if isinstance(c, dict)}
        for i, m in enumerate(asked):
            ch = chosen.get(m["id"], {})
            cand = next((c for c in m["cands"] if c["letter"] == str(ch.get("take") or "").strip().upper()[:1]), None)
            got = keep(m, cand, media, look) if cand else None
            if got:
                m.update(got, credit=credit_of(cand), page=cand.get("page", ""), own=bool(cand.get("own")), shows=str(ch.get("why", ""))[:200])
                m["for"] = m["search"] + "|" + m["wanted"]
                log(f"  {m['id']} {m['what'][:50]}: {got['kind']} · {m['credit'][:90]}")
            else:
                m["missing"] = str((cand or {}).get("nothing") or ("the chosen file could not be fetched" if cand else ch.get("why") or "none of the candidates shows it"))[:200]
                again = " ".join(str(ch.get("try") or "").split())[:80]
                if rnd == 1 and again and again.lower() != m["search"].lower():
                    m["try"], m["was_missing"] = again, m["missing"]
                    todo.append(m)
                else:
                    log(f"  {m['id']} {m['what'][:50]}: NOT FOUND ({m['missing'][:90]})")
            ac.stage_progress((rnd - 1) * 0.5 + 0.25 + 0.25 * (i + 1) / len(asked))
        if todo:
            log(f"  looking again with other words for {len(todo)}: " + "; ".join(f"{m['id']} “{m['try']}”" for m in todo)[:300])
    for m in todo:
        log(f"  {m['id']} {m['what'][:50]}: NOT FOUND ({m.get('missing', '')[:90]})")
    for m in plan.get("media", []):
        for key in ("cands", "sheet", "try", "was_missing"):
            m.pop(key, None)
        if not m.get("file"):
            m["kind"] = m.get("wanted", m.get("kind", "photo"))
    found = sum(1 for m in plan.get("media", []) if m.get("file"))
    return found, len(plan.get("media", []))


def clean_needs(raw, chapter: int, start: int) -> list[dict]:
    """A chapter's things to be seen, from the planner's answer, numbered on from `start`."""
    out = []
    for m in raw if isinstance(raw, list) else []:
        if not isinstance(m, dict) or not str(m.get("what", "")).strip() or not str(m.get("search", "")).strip():
            continue
        out.append({"id": f"m{start + len(out)}", "chapter": chapter, "what": " ".join(str(m["what"]).split())[:160], "search": re.sub(r"\s+", " ", str(m["search"])).strip()[:80],
                    "kind": "video" if str(m.get("kind", "")).lower().startswith("v") else "photo", "why": " ".join(str(m.get("why", "")).split())[:160]})
        if len(out) >= 5:
            break
    return out
