"""hooks: a local library of opening hooks (the first 1–3 seconds that keep a viewer watching).

The library lives at ~/Library/Application Support/EI/hooks/library.json and is used without the
internet: the script writer and Video from Topic get the best-fitting hooks for their topic, and
the voiceover edit gets an on-screen hook line. "Sync Hooks" (AI ▸ Hook Library) refreshes it:

  1. What is hot now: Google Trends' daily trending searches (India and the US by default; any
     country codes), and this week's most-viewed YouTube videos in your niches (titles + views).
  2. Claude (with web search) studies those and current hook formats, and writes new hooks in
     English, Hindi and Hinglish, each with its pattern, niche, emotion, why it works, the trend it
     rides and how hot it is. Without Claude, the most-viewed titles themselves become hooks.
  3. They are merged into the library (no duplicates), and old trends cool down over time.

Built in (offline from the first run): 24 proven hook patterns with examples.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import ssl
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import autocut as ac
import emotions as em
from autocut import AutocutError, log

DEFAULT_NICHES = ("gaming", "tech", "travel", "finance", "fitness", "food", "motivation", "movies", "news", "education")
DEFAULT_GEOS = ("IN", "US")
MAX_HOOKS = 3000
HALF_LIFE_DAYS = 21.0  # a trend's heat halves every three weeks

# pattern: (what it does, example in English, example in Hindi/Hinglish)
PATTERNS: dict[str, tuple[str, str, str]] = {
    "curiosity_gap": ("Opens a question the viewer needs closed", "Nobody is talking about this, but it changes everything.",
                      "इस बारे में कोई बात नहीं कर रहा, पर ये सब बदल देगा।"),
    "bold_claim": ("A strong statement that begs to be checked", "This is the most dangerous mission in the whole game.",
                   "ये अब तक का सबसे खतरनाक mission है।"),
    "question": ("Asks what the viewer already wonders", "What happens if you never stop running?", "अगर आप कभी रुको ही नहीं, तो क्या होगा?"),
    "stat_shock": ("A number that surprises", "Ninety million views. In one day.", "एक ही दिन में नौ करोड़ views।"),
    "in_media_res": ("Starts in the middle of the action", "I had three seconds left when the bridge collapsed.",
                     "मेरे पास सिर्फ तीन seconds बचे थे, और तभी पुल टूट गया।"),
    "contrarian": ("Goes against what everyone believes", "Everyone is wrong about this.", "सब इसके बारे में गलत सोचते हैं।"),
    "mistake": ("Warns about a costly error", "Stop doing this before you lose everything.", "ये गलती अभी बंद करो, वरना सब खो दोगे।"),
    "list": ("Promises a countable payoff", "Three things nobody tells you about Vice City.", "Vice City की तीन बातें जो कोई नहीं बताता।"),
    "pov": ("Puts the viewer inside the moment", "POV: you just found the rarest car in the city.", "POV: आपको शहर की सबसे rare car मिल गई।"),
    "challenge": ("A personal test with an unknown result", "I tried this for thirty days. Here's what happened.",
                  "मैंने ये तीस दिन किया, देखो क्या हुआ।"),
    "secret_reveal": ("Promises hidden knowledge", "Here's what they don't want you to know.", "ये वो बात है जो वो आपको नहीं बताना चाहते।"),
    "urgency": ("A deadline or stakes right now", "You have six days left.", "आपके पास सिर्फ छह दिन बचे हैं।"),
    "comparison": ("Two things side by side", "Old versus new. The difference is insane.", "पुराना vs नया, फर्क देखकर दंग रह जाओगे।"),
    "before_after": ("Shows a transformation", "This is what it looked like before. Now look.", "पहले ऐसा था। अब देखो।"),
    "myth_bust": ("Breaks a common belief", "That famous trick? It never worked.", "वो famous trick? वो कभी काम ही नहीं करती थी।"),
    "direct_address": ("Speaks to one kind of viewer", "If you play every night, this is for you.", "अगर आप हर रात खेलते हो, ये video आपके लिए है।"),
    "imagine": ("Asks the viewer to picture something", "Imagine waking up in a city that never sleeps.", "सोचो, आप एक ऐसे शहर में जागो जो कभी सोता नहीं।"),
    "story_tease": ("Promises a story with a twist", "This started as a normal day. It didn't end like one.",
                    "ये एक normal दिन की तरह शुरू हुआ, पर ऐसे खत्म नहीं हुआ।"),
    "reaction": ("Shows a strong reaction first", "I did not expect that ending.", "मुझे ये ending बिल्कुल expect नहीं थी।"),
    "authority": ("Borrows trust from an expert or record", "The developers just confirmed it.", "Developers ने अभी confirm कर दिया।"),
    "relatable_pain": ("Names a frustration the viewer has", "We've all waited way too long for this.", "हम सबने इसके लिए बहुत लंबा इंतज़ार किया है।"),
    "visual_hook": ("Something unusual on screen in the first second", "Look at this.", "ये देखो।"),
    "trend_jack": ("Rides what everyone is talking about today", "Everyone's searching for this today. Here's why.",
                   "आज हर कोई यही search कर रहा है, जानिए क्यों।"),
    "payoff_first": ("Shows the ending first, then how it happened", "This is how it ends. Now watch how we got here.",
                     "ऐसे खत्म होता है। अब देखो यहाँ तक कैसे पहुँचे।"),
}
PATTERN_EMOTION = {"curiosity_gap": "Curiosity", "bold_claim": "Excitement", "question": "Curiosity", "stat_shock": "Amazement",
                   "in_media_res": "Thrill", "contrarian": "Surprise", "mistake": "Anxiety", "list": "Curiosity", "pov": "Excitement",
                   "challenge": "Anticipation", "secret_reveal": "Mystery", "urgency": "Tension", "comparison": "Fascination",
                   "before_after": "Amazement", "myth_bust": "Surprise", "direct_address": "Curiosity", "imagine": "Wonder",
                   "story_tease": "Suspense", "reaction": "Surprise", "authority": "Anticipation", "relatable_pain": "Nostalgia",
                   "visual_hook": "Surprise", "trend_jack": "Excitement", "payoff_first": "Suspense"}


def hooks_dir() -> Path:
    base = Path(os.environ["EI_SETTINGS"]).parent if os.environ.get("EI_SETTINGS") else Path.home() / "Library" / "Application Support" / "EI"
    return base / "hooks"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def age_days(iso: str | None) -> float:
    try:
        t = datetime.strptime(str(iso), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - t).total_seconds() / 86400)
    except ValueError:
        return 365.0


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(t).lower())).strip()


def seed_hooks() -> list[dict]:
    out = []
    for p, (why, en, hi) in PATTERNS.items():
        for text, lang in ((en, "en"), (hi, "hi" if not re.search(r"[A-Za-z]", hi) else "hinglish")):
            out.append({"text": text, "pattern": p, "language": lang, "niche": "any", "emotion": PATTERN_EMOTION[p], "why": why,
                        "trend": "", "hot": 0.3, "source": "built in", "added": "2026-11-13T00:00:00Z"})
    return out


def load(lib: Path | None = None) -> dict:
    p = (lib or hooks_dir()) / "library.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data.setdefault("hooks", [])
    data.setdefault("trends", [])
    data.setdefault("syncs", [])
    if not any(h.get("source") == "built in" for h in data["hooks"]):
        data["hooks"] = seed_hooks() + data["hooks"]
    return data


def save(data: dict, lib: Path | None = None) -> Path:
    d = lib or hooks_dir()
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "library.json.part"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(d / "library.json")
    return d / "library.json"


def merge(data: dict, new: list[dict], limit: int = MAX_HOOKS) -> int:
    """Adds hooks that are not in the library yet (same words = same hook); keeps the hottest
    `limit`. Returns how many were added."""
    seen = {norm(h["text"]) for h in data["hooks"]}
    added = 0
    for h in new:
        k = norm(h.get("text", ""))
        if len(k) < 6 or k in seen:
            continue
        seen.add(k)
        data["hooks"].append(h)
        added += 1
    if len(data["hooks"]) > limit:
        builtin = [h for h in data["hooks"] if h.get("source") == "built in"]
        rest = sorted((h for h in data["hooks"] if h.get("source") != "built in"), key=heat, reverse=True)
        data["hooks"] = builtin + rest[: max(0, limit - len(builtin))]
    return added


def heat(h: dict) -> float:
    """How hot a hook is now: its heat when found, cooling with age (built-in patterns stay warm)."""
    if h.get("source") == "built in":
        return float(h.get("hot", 0.3))
    return float(h.get("hot", 0.5)) * 0.5 ** (age_days(h.get("added")) / HALF_LIFE_DAYS)


# ---------------------------------------------------------------------------------------------
# What is hot (internet)
# ---------------------------------------------------------------------------------------------


def _ctx() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _traffic(s: str) -> int:
    m = re.match(r"([\d,.]+)\s*([KM]?)", s.replace("+", "").strip(), re.I)
    if not m:
        return 0
    v = float(m.group(1).replace(",", ""))
    return int(v * {"K": 1e3, "M": 1e6}.get(m.group(2).upper(), 1))


def parse_trends_rss(xml: str, geo: str) -> list[dict]:
    out = []
    for item in re.findall(r"<item>(.*?)</item>", xml, re.S):
        title = re.search(r"<title>(.*?)</title>", item, re.S)
        if not title:
            continue
        traffic = re.search(r"<ht:approx_traffic>(.*?)</ht:approx_traffic>", item)
        news = re.findall(r"<ht:news_item_title>(.*?)</ht:news_item_title>", item, re.S)
        clean = lambda t: re.sub(r"<!\[CDATA\[|\]\]>|&amp;", lambda m: "&" if m.group(0) == "&amp;" else "", t).strip()  # noqa: E731
        out.append({"topic": clean(title.group(1)), "geo": geo, "traffic": _traffic(traffic.group(1)) if traffic else 0,
                    "news": [clean(n) for n in news[:3]], "seen": now_iso()})
    return out


def google_trends(geo: str) -> list[dict]:
    url = f"https://trends.google.com/trending/rss?geo={urllib.parse.quote(geo)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (EI video editor)"})
        xml = urllib.request.urlopen(req, context=_ctx(), timeout=30).read().decode("utf-8", "replace")
    except OSError as e:
        log(f"trends for {geo} unavailable ({type(e).__name__})")
        return []
    return parse_trends_rss(xml, geo)


def youtube_hot(niche: str, n: int = 15) -> list[dict]:
    """This week's most-viewed YouTube videos for a niche: titles (the hooks people clicked) and views."""
    import script_writer as sw

    url = "https://www.youtube.com/results?" + urllib.parse.urlencode({"search_query": niche, "sp": "CAMSAggD"})
    cmd = sw.yt_dlp_cmd() + ["--flat-playlist", "--playlist-end", str(n), "--print", "%(view_count)s\t%(duration)s\t%(title)s\t%(url)s", url]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except (subprocess.TimeoutExpired, OSError):
        return []
    vids = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        try:
            views = int(parts[0])
        except ValueError:
            views = 0
        vids.append({"title": parts[2].strip(), "views": views, "url": parts[3].strip(), "niche": niche})
    return vids


def clean_title(t: str) -> str:
    t = re.sub(r"#\S+", "", t)
    t = re.sub(r"[\U0001F000-\U0001FAFF☀-➿]", "", t)
    head = re.split(r"\s[|•]\s", t)[0]
    if len(head.strip()) >= 12:  # "Main title | Episode 1 ft. …": the title is the hook
        t = head
    t = re.sub(r"\s*[|•–-]\s*(ep(isode)?\.?\s*\d+|part\s*\d+|live|official.*|full.*)$", "", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip(" -|")


_PATTERN_RULES = [
    (r"\?$|^(what|why|how|who|can|is|are|would|did|does)\b", "question"), (r"^\d+|\b(top|best)\s+\d+", "list"),
    (r"\b(stop|never|don'?t|mistake)\b", "mistake"), (r"\bvs\.?\b|versus", "comparison"), (r"\bpov\b", "pov"),
    (r"\b(i tried|for \d+ days|challenge|24 hours)\b", "challenge"), (r"\b(secret|hidden|nobody|no one|don'?t want you)\b", "secret_reveal"),
    (r"\b(finally|confirmed|official|announced)\b", "authority"), (r"\b(before|after)\b", "before_after"),
    (r"\b(insane|crazy|unbelievable|shocking|impossible)\b", "bold_claim"), (r"\b(million|billion|crore|lakh|\d{2,})\b", "stat_shock"),
]


def guess_pattern(text: str) -> str:
    t = text.lower().strip()
    for pat, name in _PATTERN_RULES:
        if re.search(pat, t):
            return name
    return "bold_claim" if text.isupper() or "!" in text else "curiosity_gap"


def title_hooks(vids: list[dict]) -> list[dict]:
    """Most-viewed titles as hooks (when Claude is not available to write better ones)."""
    out = []
    top = max([v["views"] for v in vids] or [1])
    for v in vids:
        t = clean_title(v["title"])
        if len(t) < 12 or len(t) > 110:
            continue
        lang = "hi" if re.search(r"[ऀ-ॿ]", t) and not re.search(r"[A-Za-z]{3,}", t) else ("hinglish" if re.search(r"[ऀ-ॿ]", t) else "en")
        pat = guess_pattern(t)
        out.append({"text": t, "pattern": pat, "language": lang, "niche": v["niche"], "emotion": PATTERN_EMOTION.get(pat, "Curiosity"),
                    "why": f"{v['views']:,} views this week", "trend": "", "hot": round(0.3 + 0.6 * math.log1p(v["views"]) / math.log1p(top), 2),
                    "source": v["url"], "added": now_iso()})
    return out


def claude_prompt(niches: list[str], trends: list[dict], vids: list[dict], per_niche: int) -> str:
    tr = "\n".join(f"- {t['topic']} ({t['geo']}, {t['traffic']:,}+ searches){': ' + t['news'][0] if t['news'] else ''}" for t in trends[:40])
    vt = "\n".join(f"- [{v['niche']}] {v['title']} ({v['views']:,} views)" for v in sorted(vids, key=lambda v: -v["views"])[:80])
    pats = "\n".join(f"- {k}: {v[0]}" for k, v in PATTERNS.items())
    return f"""You are building a hook library for a video creator in India (YouTube Shorts, Instagram Reels, YouTube long form;
English, Hindi and Hinglish). A hook is the first spoken line (1–3 seconds) that makes a viewer stay.

Use web search to check what hook styles are working right now on Shorts/Reels/TikTok (this month), and use the
data below about what is hot today.

## Trending searches today
{tr or "(none)"}

## This week's most-viewed YouTube videos per niche
{vt or "(none)"}

## Hook patterns
{pats}

## Task
For each niche ({', '.join(niches)}), write {per_niche} new, original hooks: about half riding today's trends or hot
videos (say which), half evergreen for the niche. Mix English, Hindi (Devanagari) and Hinglish (Devanagari with English
words in English). Short and spoken (4–14 words), no hashtags or emojis, no clickbait lies, nothing defamatory. Also add up
to 10 "any"-niche hooks in formats that are trending this month.

Reply with only JSON:
{{"hooks": [{{"text": "...", "language": "en|hi|hinglish", "pattern": "one of the pattern names", "niche": "...",
  "emotion": "one of: {', '.join(n for n in em.EMOTIONS if n != 'Neutral')}", "why": "why it works, 6-15 words",
  "trend": "the trend or hot video it rides, or empty", "hot": 0.0-1.0}}],
 "formats": ["one line per hook format trending this month"]}}
"""


def sync(niches: list[str] | None = None, geos: list[str] | None = None, use_claude: bool = True, per_niche: int = 12,
         model: str | None = None, lib: Path | None = None) -> dict:
    """Fetch what is hot, turn it into hooks, merge into the local library."""
    niches = [n.strip() for n in (niches or DEFAULT_NICHES) if n.strip()][:20]
    geos = [g.strip().upper() for g in (geos or DEFAULT_GEOS) if g.strip()][:8]
    t0 = time.time()
    data = load(lib)
    trends = []
    for g in geos:
        tr = google_trends(g)
        log(f"trending searches ({g}): {len(tr)}" + (f", e.g. {', '.join(t['topic'] for t in tr[:4])}" if tr else ""))
        trends += tr
    vids = []
    for k, n in enumerate(niches, 1):
        v = youtube_hot(n)
        vids += v
        log(f"progress {k}/{len(niches)} niches: {n}: {len(v)} hot videos this week")
    new: list[dict] = []
    who = "titles"
    if use_claude:
        try:
            import topic_video as tv
            import script_writer as sw

            log("Claude is studying the trends and writing hooks (with web search)")
            res = sw.extract_json(tv.ask_claude_research(claude_prompt(niches, trends, vids, per_niche), model))
            for h in res.get("hooks", []):
                text = re.sub(r"\s+", " ", str(h.get("text", ""))).strip().strip('"')
                if not text:
                    continue
                pat = h.get("pattern") if h.get("pattern") in PATTERNS else guess_pattern(text)
                try:
                    hot = float(h.get("hot", 0.6))
                except (TypeError, ValueError):
                    hot = 0.6
                new.append({"text": text, "pattern": pat, "language": h.get("language") if h.get("language") in ("en", "hi", "hinglish") else "en",
                            "niche": str(h.get("niche") or "any").lower(), "emotion": em.resolve(h.get("emotion")), "why": str(h.get("why", ""))[:160],
                            "trend": str(h.get("trend", ""))[:120], "hot": round(min(1.0, max(0.0, hot)), 2), "source": "Claude + web", "added": now_iso()})
            if res.get("formats"):
                data["formats"] = [str(f)[:200] for f in res["formats"]][:20]
            who = "Claude"
        except AutocutError as e:
            log(f"Claude could not write hooks ({e}); using the most-viewed titles")
    new += title_hooks(vids)
    added = merge(data, new)
    data["trends"] = sorted(trends, key=lambda t: -t["traffic"])[:200]
    data["syncs"] = (data.get("syncs") or [])[-19:] + [{"at": now_iso(), "niches": niches, "geos": geos, "added": added, "by": who}]
    data["niches"] = niches
    path = save(data, lib)
    log(f"hook library: {added} new hook(s) ({who}), {len(data['hooks'])} in all, {len(trends)} trends, in {time.time() - t0:.0f}s")
    return {"outputs": [str(path)], "out_dir": str(path.parent), "added": added, "total": len(data["hooks"]), "trends": len(trends)}


def youtube_top(query: str, n: int = 15) -> list[dict]:
    """The most-viewed YouTube videos for a search (all time): how the biggest videos on a topic open."""
    import script_writer as sw

    url = "https://www.youtube.com/results?" + urllib.parse.urlencode({"search_query": query, "sp": "CAM="})
    cmd = sw.yt_dlp_cmd() + ["--flat-playlist", "--playlist-end", str(n), "--print", "%(view_count)s\t%(duration)s\t%(title)s\t%(url)s", url]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except (subprocess.TimeoutExpired, OSError):
        return []
    vids = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            try:
                views = int(parts[0])
            except ValueError:
                views = 0
            vids.append({"title": parts[2].strip(), "views": views, "url": parts[3].strip(), "niche": query})
    return vids


def topic_prompt(topic: str, niche: str, language: str, vids: list[dict], facts: str, n: int) -> str:
    vt = "\n".join(f"- {v['title']} ({v['views']:,} views)" for v in sorted(vids, key=lambda v: -v["views"])[:20])
    pats = "\n".join(f"- {k}: {v[0]}" for k, v in PATTERNS.items())
    lang = {"en": "English", "hi": "Hindi (Devanagari)", "hinglish": "Hinglish (Devanagari with English words in English)"}.get(language, "English")
    return f"""Find the best opening hooks for ONE short video. Topic: {topic}
Niche: {niche or "general"}. Language of the video: {lang}.

Use web search: how do the most successful videos, reels and posts about this topic (and very similar ones) open right
now? What angle, number, surprise or question makes people stop? What is new or hot about it this week?

The most-viewed YouTube videos on this topic:
{vt or "(none found)"}
{("Researched facts about the topic (stay true to them):" + chr(10) + facts) if facts else ""}

Hook patterns:
{pats}

Write {n} different hooks for this exact video, in {lang}: spoken, 4–14 words, true (no clickbait lie), each a
different pattern where possible, the strongest first.

Reply with only JSON:
{{"hooks": [{{"text": "...", "pattern": "one of the pattern names", "emotion": "one of: {', '.join(x for x in em.EMOTIONS if x != 'Neutral')}",
  "why": "why it works for this topic, 6-15 words", "source": "the video/post/article it learned from, or empty", "hot": 0.0-1.0}}]}}
"""


def find_for_topic(topic: str, niche: str = "", language: str = "en", facts: str = "", n: int = 8, model: str | None = None,
                   lib: Path | None = None) -> list[dict]:
    """Hooks found online for one topic (the topic's most-viewed videos + Claude with web search),
    best first. They are also kept in the library."""
    vids = youtube_top(topic)
    log(f"hook search: {len(vids)} most-viewed videos on “{topic[:60]}”")
    found: list[dict] = []
    try:
        import script_writer as sw
        import topic_video as tv

        log("Claude is looking online for hooks that work for this topic")
        res = sw.extract_json(tv.ask_claude_research(topic_prompt(topic, niche, language, vids, facts, n), model))
        for h in res.get("hooks", []):
            text = re.sub(r"\s+", " ", str(h.get("text", ""))).strip().strip('"')
            if not text:
                continue
            try:
                hot = float(h.get("hot", 0.8))
            except (TypeError, ValueError):
                hot = 0.8
            found.append({"text": text, "pattern": h.get("pattern") if h.get("pattern") in PATTERNS else guess_pattern(text), "language": language,
                          "niche": (niche or "any").lower(), "emotion": em.resolve(h.get("emotion")), "why": str(h.get("why", ""))[:160],
                          "trend": topic[:120], "hot": round(min(1.0, max(0.0, hot)), 2), "source": str(h.get("source") or "topic search")[:200],
                          "added": now_iso()})
    except AutocutError as e:
        log(f"Claude could not search for hooks ({e}); using the topic's most-viewed titles")
    if not found:
        found = [{**h, "trend": topic[:120]} for h in title_hooks(vids)][:n]
    data = load(lib)
    added = merge(data, found)
    save(data, lib)
    log(f"hook search: {len(found)} hook(s) for this topic ({added} new in the library)" + (f"; best: “{found[0]['text']}”" if found else ""))
    return found


def topic_block(found: list[dict]) -> str:
    """The prompt section for hooks found online for this exact topic."""
    if not found:
        return ""
    lines = "\n".join(f"- [{h['pattern']}] {h['text']}" + (f"  ({h['why']})" if h.get("why") else "") for h in found[:8])
    return ("\n## Hooks found online for this exact topic (strongest first): open with the best one, or a sharper version of it\n"
            + lines + "\n")


# ---------------------------------------------------------------------------------------------
# Choosing hooks for a video (offline)
# ---------------------------------------------------------------------------------------------

_STOP = set("the a an and or of to in on for with is are this that you your it its we our my i me be was were how what why who".split())


def words(t: str) -> set[str]:
    return {w for w in re.findall(r"[\wऀ-ॿ]+", str(t).lower()) if len(w) > 2 and w not in _STOP}


def suggest(topic: str = "", niche: str = "", language: str = "en", n: int = 8, lib: Path | None = None) -> dict:
    """The best hooks for a video: relevant to its topic and niche, hot, in its language, and in
    different patterns (so the writer sees a range of shapes). Plus the related trends."""
    data = load(lib)
    tw = words(topic) | words(niche)
    want_lang = {"en": ("en",), "hi": ("hi", "hinglish"), "hinglish": ("hinglish", "hi", "en")}.get(language, ("en",))
    scored = []
    for h in data["hooks"]:
        rel = len(tw & (words(h["text"]) | words(h.get("trend", "")) | words(h.get("niche", "")))) / max(1, min(len(tw), 6)) if tw else 0.0
        niche_ok = 1.0 if niche and niche.lower() in str(h.get("niche", "")).lower() else (0.4 if h.get("niche") == "any" else 0.0)
        lang = 1.0 if h.get("language") in want_lang[:1] else (0.5 if h.get("language") in want_lang else 0.0)
        scored.append((1.6 * rel + 0.8 * niche_ok + 1.2 * heat(h) + 0.6 * lang, h))
    scored.sort(key=lambda x: -x[0])
    out, used = [], set()
    for s, h in scored:  # one per pattern first, then the rest
        if h["pattern"] in used:
            continue
        used.add(h["pattern"])
        out.append(h)
        if len(out) >= n:
            break
    trends = [t for t in data["trends"] if tw & (words(t["topic"]) | {w for x in t.get("news", []) for w in words(x)})][:6]
    return {"hooks": out, "trends": trends, "formats": data.get("formats", [])[:6]}


def prompt_block(s: dict) -> str:
    """The hook section of a script-writing prompt."""
    if not s["hooks"]:
        return ""
    lines = [f"- [{h['pattern']}] {h['text']}" + (f"  (rides: {h['trend']})" if h.get("trend") else "") for h in s["hooks"]]
    out = "\n## Hooks that work now (from the creator's hook library: adapt the best-fitting one, never copy it word for word)\n" + "\n".join(lines) + "\n"
    if s["trends"]:
        out += "Related searches trending today: " + "; ".join(t["topic"] for t in s["trends"]) + "\n"
    if s["formats"]:
        out += "Hook formats trending this month: " + "; ".join(s["formats"]) + "\n"
    return out


def stats(lib: Path | None = None) -> dict:
    data = load(lib)
    by_lang: dict[str, int] = {}
    for h in data["hooks"]:
        by_lang[h.get("language", "en")] = by_lang.get(h.get("language", "en"), 0) + 1
    last = (data.get("syncs") or [{}])[-1]
    hot = sorted((h for h in data["hooks"] if h.get("source") != "built in"), key=heat, reverse=True)[:12]
    return {"total": len(data["hooks"]), "languages": by_lang, "last_sync": last.get("at"), "trends": len(data["trends"]),
            "hottest": [{k: h.get(k) for k in ("text", "pattern", "niche", "language", "trend")} for h in hot],
            "niches": data.get("niches") or list(DEFAULT_NICHES), "path": str((lib or hooks_dir()) / "library.json")}


def add_args(sub) -> None:
    p = sub.add_parser("hooks-sync", help="refresh the hook library from what is hot now (trends, top videos, Claude)")
    p.add_argument("--niches", nargs="*", help=f"default: {' '.join(DEFAULT_NICHES)}")
    p.add_argument("--geo", nargs="*", help="country codes for trending searches (default IN US)")
    p.add_argument("--no-claude", action="store_true", help="only collect trends and top titles")
    p.add_argument("--per-niche", type=int, default=12)
    p.add_argument("--model")
    p = sub.add_parser("hooks-list", help="the best hooks for a topic (offline), or the library's state")
    p.add_argument("--topic", default="")
    p.add_argument("--niche", default="")
    p.add_argument("--language", default="en", choices=["en", "hi", "hinglish"])
    p.add_argument("-n", type=int, default=8)


def run_args(a) -> dict:
    if a.cmd == "hooks-sync":
        return sync(a.niches, a.geo, use_claude=not a.no_claude, per_niche=max(3, min(30, a.per_niche)), model=a.model)
    if a.topic or a.niche:
        s = suggest(a.topic, a.niche, a.language, a.n)
        return {"hooks": s["hooks"], "trends": s["trends"]}
    return stats()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_args(sub)
    try:
        print(json.dumps(run_args(ap.parse_args(argv)), ensure_ascii=False))
    except AutocutError as e:
        log(f"error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
