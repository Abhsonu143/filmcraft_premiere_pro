#!/usr/bin/env python3
"""explainer: what you want to explain -> a researched plan you can read, change and approve.

    python ei_engine.py explain "how a Python generator works, with the bugs people hit" --minutes 8 --review
    python ei_engine.py explain "…the same words…" --note "2: show the error message first" --note "shorter intro"
    python ei_engine.py explain "…the same words…" --approve

For videos that teach or argue something: coding and technical topics (flows, code, a bug found and
fixed), and equally research, social, political, history, science or business topics.

1. Research. Claude (through the Claude Code app you are logged into) searches and reads the web:
   facts with their source link AND the exact sentence of the source that says so.
2. Plan. The video in chapters. For each chapter: what it is for, its own opening line, what is
   said, what is on screen (a flow, code, a bug and its fix, a passage of an article marked like in
   a book, an example, a chart…), which claims it makes and where they come from, and why the
   chapter works cut out on its own: every chapter is planned as a clip that can be posted alone.
3. Truth. Every quoted sentence is looked up in its source page. What is not found word for word
   is marked "not confirmed" in the plan, so you see it before anything is said in your voice.
4. Review (optional, `--review`): the plan waits for you. Give notes (for the whole video, or
   "3: …" for chapter 3) as often as you like; only what a note asks for changes. `--approve` when
   it is right. Without `--review` the plan is written, kept, and counts as approved.

Output folder: plan.md (to read), plan.json (the same, for the next steps), research.json.
The steps after the plan (script, pictures, voice, edit, clips) read plan.json.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import autocut as ac
from autocut import AutocutError, log

KINDS = ("auto", "coding", "technical", "research", "science", "history", "social", "political", "business", "general")
# what a chapter can put on screen (each is drawn by the picture step from the plan's words)
SHOWS = {
    "flow": "a flow or diagram built up step by step (boxes and arrows)",
    "code": "code typed or revealed line by line, the line being talked about lit up",
    "debug": "a bug: the failing run and its error, the cause pointed at, the fix, the passing run",
    "terminal": "commands and their output",
    "article": "a passage of a source page, shown as it stands with the sentence marked like in a book",
    "quote": "a person's or document's exact words with who said them and where",
    "example": "a small concrete case worked through",
    "compare": "two things side by side (before/after, A against B)",
    "chart": "numbers as a simple chart, with their source",
    "timeline": "dates in order",
    "footage": "a photo or video of the real thing (free licence, or your own)",
    "card": "a short line of text on screen",
}
MAX_CHAPTERS = 14
MAX_PAGES = 14  # source pages read to confirm quotes


def research_prompt(want: str, minutes: float, language: str, kind: str, hints: str) -> str:
    import topic_video as tv

    coding = ("\nFor code and tools: check behaviour against the official documentation or the source itself, name the version, and give real, "
              "runnable examples (never an API from memory). For a bug: the real error text as the tool prints it.\n") if kind in ("auto", "coding", "technical") else ""
    sides = ("\nWhere people disagree (social and political questions): the strongest form of each side, who holds it, and what the evidence does and "
             "does not show. Primary sources first (the law, the ruling, the dataset, the speech itself), then reporting.\n") if kind in ("auto", "social", "political", "research", "history") else ""
    return f"""Research this for a {minutes:.0f}-minute explainer video (language of the video: {language}; kind: {kind}). Today is {time.strftime('%d %B %Y')}.
What the creator wants to make: {want}{tv.hints_block(hints)}
Use web search and read the pages. Reliable, citable sources: official documentation and standards, papers, datasets, laws and rulings,
government and institution pages, major news. Prefer primary sources and the latest facts; note dates. Never invent a fact, a number or a quote.
{coding}{sides}
For every fact give the page it is on and "quote": ONE sentence copied from that page exactly as it stands there (same words, at most 45 words),
the sentence that says it. The quote is looked up in the page afterwards, so it must be word for word; leave it empty if you cannot copy one.

Reply with only a JSON object:
{{"title_idea": "...", "angle": "why this matters to a viewer now, one sentence",
 "summary": "4-6 sentences",
 "facts": [{{"fact": "one short verifiable fact", "url": "source link", "source": "publisher or site name", "date": "of the source, if known", "quote": "exact sentence from the page"}}],
 "examples": [{{"what": "a concrete example, case or snippet worth showing", "detail": "the code, numbers or story itself", "url": "where it is from"}}],
 "disputed": ["a point people get wrong or disagree on, and what the sources actually say"],
 "articles": [{{"title": "...", "url": "...", "publisher": "..."}}],
 "keywords": ["names and terms"]}}
Give {max(10, min(30, int(minutes * 3)))} or more facts, 3-8 examples, up to 8 articles."""


def plan_prompt(want: str, minutes: float, language: str, kind: str, data: dict, hints: str) -> str:
    import topic_video as tv

    facts = "\n".join(f"F{k}. {f['fact']} [{f.get('source') or ''} {f.get('date') or ''}] {f.get('url', '')}" + (f"\n    quote: “{f['quote']}”" if f.get("quote") else "")
                      for k, f in enumerate(data.get("facts", []), 1))
    examples = "\n".join(f"- {e.get('what')}: {str(e.get('detail', ''))[:600]} ({e.get('url', '')})" for e in data.get("examples", []) if isinstance(e, dict))
    disputed = "\n".join(f"- {d}" for d in data.get("disputed", []) if str(d).strip())
    shows = "\n".join(f"  {k}: {v}" for k, v in SHOWS.items())
    return f"""Plan a {minutes:.0f}-minute explainer video in {language} (kind: {kind}). The creator reads this plan and approves it before anything is written or spoken.
What the creator wants to make: {want}{tv.hints_block(hints)}
## Research (use ONLY this; a claim needs a fact number)
Angle: {data.get("angle", "")}
{facts}
## Examples found
{examples or "(none)"}
## What people get wrong or disagree on
{disputed or "(none)"}

## How to plan it
- {max(3, min(MAX_CHAPTERS, round(minutes * 1.3)))} chapters or so, 30 to 90 seconds each, in the order a viewer needs them.
- EVERY chapter must work cut out and posted on its own, to someone who saw nothing else: it opens with its own hook (a question, a
  surprising fact, a mistake everyone makes), makes ONE complete point, and ends on a line that lands. Never "as we saw", "earlier", "next".
- What is said must be true and checkable: each claim names the fact number it rests on. Leave out what the research does not support.
  Where sources disagree, say so plainly; do not pick a side the evidence does not pick.
- Show, do not only tell. For each chapter choose what is on screen from these (the picture step draws them from your words, so be exact:
  the boxes of the flow in order, the code itself, the error text, the sentence to mark):
{shows}
  Technical topics lean on flow, code, debug, terminal, example; research and social topics on article, quote, chart, timeline, compare.
  "article" and "quote" may only use a quote given in the research, word for word.
- REAL PICTURES AND VIDEO ("media"): a video of only drawn text feels like slides. Wherever the chapter is about something that exists and can
  be seen (a place, a machine, a building, an event, an animal, a product, a person in public life, a document, a landscape), name it in
  "media": free-licensed photos and videos are then searched for each, looked at, and used full screen or dimmed behind the text.
  "what": exactly what should be seen; "search": 2-4 plain English words a photo archive would file it under (the proper name, not a
  description); "kind": "video" where movement helps (a launch, a machine working, a crowd, weather), else "photo"; "why": what it does there.
  For a topic about the real world aim for something real on screen every 15-20 seconds (2-4 per chapter). Use judgment about when NOT to:
  none for code, logic, maths, an abstract idea or a number (a drawn scene explains those better), none that only decorates (no generic
  "person at laptop", "abstract technology"), none that could mislead (a different event passed off as this one). A coding chapter usually has none.
- how_said: how the voice should sound there (curious, dry, urgent, warm…) and how fast.
- catch: in one line, why someone would stop scrolling for this chapter, and what they would repeat to a friend afterwards.
- open_questions: what you could not settle from the research and the creator should decide or check.

Answer with only this JSON:
{{"title": "...", "promise": "what the viewer can do or understand afterwards, one sentence", "audience": "who it is for and what they already know",
 "chapters": [{{"title": "...", "goal": "what this chapter is for", "hook": "its first spoken line", "says": ["the points made, in order (3-6 short lines)"],
   "shows": [{{"type": "flow|code|debug|terminal|article|quote|example|compare|chart|timeline|footage|card", "what": "exactly what is on screen"}}],
   "media": [{{"what": "...", "search": "...", "kind": "photo|video", "why": "..."}}],
   "claims": [{{"claim": "...", "fact": 3}}], "ends_on": "its last line", "how_said": "...", "catch": "...", "clip_title": "the title it would have posted alone", "seconds": 45}}],
 "open_questions": ["..."]}}"""


def said_note(n: dict) -> str:
    """A note as the creator gave it, with its chapter."""
    return (f"chapter {n['chapter']}: " if n.get("chapter") else "") + str(n.get("text", ""))


def revise_prompt(plan: dict, notes: list[dict], data: dict, earlier: str = "") -> str:
    facts = "\n".join(f"F{k}. {f['fact']} {f.get('url', '')}" + (f" quote: “{f['quote']}”" if f.get("quote") else "") for k, f in enumerate(data.get("facts", []), 1))
    said = "\n".join(f"- {'chapter ' + str(n['chapter']) + ': ' if n.get('chapter') else 'the whole video: '}{n['text']}" for n in notes)
    keep = {k: plan.get(k) for k in ("title", "promise", "audience", "chapters", "open_questions")}
    return f"""The creator read the plan of their explainer video and wants changes. Make exactly those changes and nothing else:
a chapter no note is about stays word for word as it is. A note about how something should be made (what to show, in what order, how it
is said) is followed even where you would have done it differently. A note may add, remove, split, merge or reorder chapters.
Every chapter must still work cut out on its own, and every claim must still rest on a fact number from the research.

## The creator's notes
{said}

## The plan now
{json.dumps(keep, ensure_ascii=False, indent=1)}

## The research (the only facts there are)
{facts}

Each chapter's "media" lists the real things to be seen ({{"what", "search": 2-4 plain English words an archive files it under, "kind": "photo|video",
"why"}}): free-licensed photos and video are searched for each and looked at. A note may ask for more, fewer or other ones; one the creator
says to do without is removed.

{earlier}
Answer with only the whole plan again as JSON, in the same form, with one more field: "changed": one short sentence per change you made
(and per thing you did differently from what was asked, with the reason)."""


def clean_plan(raw: dict, data: dict) -> dict:
    """The plan as the next steps can rely on it: known picture types, claims tied to real facts
    (with their source and quote), sane lengths. Raises when there is no usable chapter."""
    import explain_media as xm

    facts = data.get("facts", [])
    chapters = []
    for c in (raw.get("chapters") if isinstance(raw, dict) else None) or []:
        if not isinstance(c, dict) or not str(c.get("title", "")).strip():
            continue
        shows = [{"type": str(s.get("type", "")).strip().lower(), "what": str(s.get("what", "")).strip()[:1500]}
                 for s in c.get("shows") or [] if isinstance(s, dict) and str(s.get("what", "")).strip()]
        for s in shows:
            if s["type"] not in SHOWS:
                s["type"] = "card"
        claims = []
        for cl in c.get("claims") or []:
            if not isinstance(cl, dict) or not str(cl.get("claim", "")).strip():
                continue
            try:
                k = int(str(cl.get("fact", 0)).lstrip("Ff") or 0)
            except ValueError:
                k = 0
            f = facts[k - 1] if 1 <= k <= len(facts) else None
            claims.append({"claim": str(cl["claim"]).strip()[:400], "fact": k if f else 0, "source": (f or {}).get("source", ""), "url": (f or {}).get("url", ""),
                           "quote": (f or {}).get("quote", ""), "confirmed": (f or {}).get("confirmed", "no source")})
        try:
            secs = float(c.get("seconds", 45))
        except (TypeError, ValueError):
            secs = 45.0
        chapters.append({"title": str(c["title"]).strip()[:120], "goal": str(c.get("goal", "")).strip()[:300], "hook": str(c.get("hook", "")).strip()[:300],
                         "says": [str(x).strip()[:300] for x in c.get("says") or [] if str(x).strip()][:8], "shows": shows[:6], "claims": claims[:8],
                         "ends_on": str(c.get("ends_on", "")).strip()[:300], "how_said": str(c.get("how_said", "")).strip()[:200],
                         "catch": str(c.get("catch", "")).strip()[:300], "clip_title": str(c.get("clip_title", "")).strip()[:120],
                         "seconds": round(min(180.0, max(15.0, secs))),
                         "media": [{k: m[k] for k in ("what", "search", "kind", "why")} for m in xm.clean_needs(c.get("media"), 0, 0)]})
    if not chapters:
        raise AutocutError("the plan came back without chapters")
    return {"title": str(raw.get("title", "")).strip()[:150] or "Explainer", "promise": str(raw.get("promise", "")).strip()[:300],
            "audience": str(raw.get("audience", "")).strip()[:300], "chapters": chapters[:MAX_CHAPTERS],
            "open_questions": [str(q).strip()[:300] for q in raw.get("open_questions") or [] if str(q).strip()][:10]}


def _plain(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", str(text).lower()).split())


def find_passage(text: str, quote: str) -> dict | None:
    """Where the quoted sentence stands in the page's text, with the rest of its paragraph around it."""
    words = re.findall(r"\w+", quote)
    m = re.search(r"[\W_]+".join(re.escape(w.strip("_")) for w in words if w.strip("_")), text, re.I) if len(words) >= 4 else None
    if not m:
        return None
    a, b = text.rfind("\n", 0, m.start()) + 1, text.find("\n", m.end())
    b = len(text) if b < 0 else b
    end = m.end()
    while end < b and text[end] in ".!?\"”’')":
        end += 1
    return {"before": text[max(a, m.start() - 500):m.start()].strip(), "exact": text[m.start():end].strip(), "after": text[end:min(b, end + 500)].strip()}


def confirm_quotes(data: dict, read=None) -> tuple[int, int]:
    """Every fact's quote looked up in its source page: `confirmed` becomes "yes" (the sentence stands
    there word for word; `context` keeps the passage around it, for showing it marked), "not found"
    (the page was read and does not have it), "page not read" (unreachable) or "no quote".
    (confirmed, with a quote)"""
    import topic_video as tv

    read = read or (lambda url: tv.page_text(url, limit=400_000))
    pages: dict[str, tuple[str, str] | None] = {}
    yes = total = 0
    for f in data.get("facts", []):
        quote, url = _plain(f.get("quote", "")), str(f.get("url", "")).strip()
        f.pop("context", None)
        if len(quote.split()) < 4 or not re.match(r"https?://", url):
            f["confirmed"] = "no quote"
            continue
        total += 1
        if url not in pages:
            if len(pages) >= MAX_PAGES:
                f["confirmed"] = "page not read"
                continue
            try:
                got = read(url)
                pages[url] = ("", got) if isinstance(got, str) else (str(got[0]), str(got[1]))
            except Exception:  # noqa: BLE001 (an unreachable page is a result, not a failure)
                pages[url] = None
        page = pages[url]
        f["confirmed"] = "page not read" if page is None else "yes" if quote in _plain(page[1]) else "not found"
        if f["confirmed"] == "yes":
            yes += 1
            f["context"] = {**(find_passage(page[1], f["quote"]) or {"before": "", "exact": f["quote"], "after": ""}), "title": page[0][:160]}
    return yes, total


def parse_note(text: str, chapters: int) -> dict | None:
    """"3: show the error first" is about chapter 3; anything else about the whole video."""
    t = " ".join(str(text).split())
    if not t:
        return None
    m = re.match(r"(?:ch(?:apter)?\.?\s*)?(\d{1,2})\s*[:.)\-]\s*(.+)", t, re.I)
    if m and 1 <= int(m.group(1)) <= chapters:
        return {"chapter": int(m.group(1)), "text": m.group(2).strip()[:600]}
    return {"chapter": 0, "text": t[:600]}


MARK = {"yes": "confirmed in the source", "not found": "NOT CONFIRMED: the sentence is not in the page", "page not read": "not confirmed: the page could not be read",
        "no quote": "not confirmed: no sentence was quoted", "no source": "NO SOURCE"}


def plan_md(plan: dict) -> str:
    total = sum(c["seconds"] for c in plan["chapters"])
    weak = sum(1 for c in plan["chapters"] for cl in c["claims"] if cl["confirmed"] != "yes")
    out = [f"# {plan['title']}", "", f"**Status:** {plan.get('status', 'draft')}  ·  about {total / 60:.1f} min in {len(plan['chapters'])} chapters  ·  "
           f"{sum(len(c['claims']) for c in plan['chapters'])} claims, {weak} not confirmed word for word", "",
           f"**What the viewer gets:** {plan['promise']}", f"**For:** {plan['audience']}", f"**You asked for:** {plan.get('want', '')}", ""]
    for n, c in enumerate(plan["chapters"], 1):
        out += [f"## {n}. {c['title']}  ({c['seconds']} s)", "", f"*What it is for:* {c['goal']}", f"*Opens with:* “{c['hook']}”", "", "Says:"]
        out += [f"- {s}" for s in c["says"]] + ["", "Shows:"] + [f"- **{s['type']}**: {s['what']}" for s in c["shows"]]
        if c["claims"]:
            out += ["", "Claims and where they come from:"]
            out += [f"- {cl['claim']}  \n  {cl['source'] or '(no source)'} · {MARK.get(cl['confirmed'], cl['confirmed'])}"
                    + (f"  \n  “{cl['quote']}”  \n  {cl['url']}" if cl["quote"] else (f"  \n  {cl['url']}" if cl["url"] else "")) for cl in c["claims"]]
        seen = [m for m in plan.get("media", []) if m["chapter"] == n]
        if seen:
            out += ["", "Real pictures and video:"]
            out += [f"- **{m['id']}** {m['kind']}: {m['what']}  \n  " + (f"found: {m['credit']}" + (f" ({m['seconds']:.0f} s)" if m.get("seconds") else "") + (f"  \n  {m['page']}" if m.get("page") else "")
                                                                         if m.get("file") else f"**NOT FOUND** ({m.get('missing', 'not looked for yet')}); searched: “{m['search']}”") for m in seen]
        out += ["", f"*Ends on:* “{c['ends_on']}”", f"*How it is said:* {c['how_said']}", f"*Why it catches:* {c['catch']}", f"*Posted alone as:* {c['clip_title']}", ""]
    lost = [m for m in plan.get("media", []) if not m.get("file") and m.get("missing")]
    if lost:
        out += ["## Pictures I could not find: can you help?", "", "Without them these moments are drawn instead. To give your own: add the photo or video files "
                "(in the app: “Your pictures and videos”), or put them in the folder `media/mine` here, and run it again. They are looked at and used where they fit.", ""]
        out += [f"- {m['id']} (chapter {m['chapter']}): {m['what']}" for m in lost] + [""]
    if plan.get("open_questions"):
        out += ["## For you to decide or check", ""] + [f"- {q}" for q in plan["open_questions"]] + [""]
    if plan.get("notes"):
        out += ["## Your notes so far", ""] + [f"- {n['when']}  {'chapter ' + str(n['chapter']) + ': ' if n.get('chapter') else ''}{n['text']}" for n in plan["notes"]] + [""]
    if plan.get("changed"):
        out += ["## Changed after your last notes", ""] + [f"- {c}" for c in plan["changed"]] + [""]
    out += ["---", "To change something: run it again with a note (\"3: show the error first\" for chapter 3, or a note for the whole video).",
            "To go on: approve the plan." if plan.get("status") != "approved" else "The plan is approved."]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------
# 2. Script: what is said line by line, and what is on screen while it is said
# ---------------------------------------------------------------------------------------------

WORDS_PER_SECOND = 2.4
SCENE_FORMS = """  {"type": "card", "text": "one short line, at most 12 words", "sub": "optional smaller line"}
  {"type": "flow", "title": "...", "boxes": ["step one", "step two", "..."]}   (2-8 boxes, at most 8 words each; step N = the first N boxes, box N lit)
  {"type": "code", "title": "file name", "lines": ["the code, one string per line"]}   (at most 16 lines of at most 60 characters; step N = the first N lines typed, 0 = all; "focus": [line numbers] lights lines)
  {"type": "terminal", "title": "...", "lines": [{"t": "in|out|err", "text": "..."}]}   (at most 12 lines; step N = the first N lines; "focus" lights lines)
  {"type": "debug", "title": "file name", "code": ["the buggy code"], "run": [{"t": "in|out|err", "text": "the failing run as printed"}], "cause": [line numbers of the cause],
   "fixed": ["the whole fixed code"], "changed": [line numbers changed in the fixed code], "passes": [{"t": "in|out", "text": "the passing run"}]}
     (steps: 1 the code and the failing run, 2 the cause pointed at, 3 the fix, 4 the passing run)
  {"type": "article", "fact": 7}   (the passage of fact 7's source page as it stands; step 1 the page, step 2 its sentence marked with a highlighter)
  {"type": "quote", "fact": 7}   (fact 7's exact sentence, large, with its source; step 2 marks it)
  {"type": "example", "title": "...", "lines": ["a worked case, one stage per line"]}   (at most 6; step N = the first N)
  {"type": "compare", "title": "...", "left": {"title": "...", "lines": ["..."], "code": false}, "right": {...}, "good_bad": true}   (step 1 the left side, 2 both; "code": true for code lines)
  {"type": "chart", "title": "...", "unit": "%", "bars": [{"label": "...", "value": 12.5}], "fact": 7}   (numbers ONLY from that fact; step N = the first N bars)
  {"type": "timeline", "title": "...", "points": [{"when": "1991", "what": "..."}]}   (at most 6; step N = the first N)
  {"type": "footage", "media": "m3", "caption": "what it is, one short line"}
     (the real photo or video m3 full screen, credited; ONLY a media id listed under "Real pictures and video found")
  Any card, quote, flow, example, compare, chart or timeline scene may also carry "behind": "m3": that picture or video fills the frame,
  dimmed, behind the scene. Never behind code, terminal, debug or article."""
SHOW_SOURCES = {"any": "any public page, briefly and with credit", "open": "only official and openly licensed sources", "none": "never"}
# sources whose words may be shown under "open": governments, courts, standards bodies, open encyclopedias, papers and documentation
OPEN_HOSTS = ("wikipedia.org", "wikimedia.org", "wikisource.org", "wikidata.org", "arxiv.org", "un.org", "who.int", "worldbank.org", "europa.eu", "ietf.org", "rfc-editor.org",
              "w3.org", "docs.python.org", "peps.python.org", "developer.mozilla.org", "kernel.org", "gnu.org", "nasa.gov", "isro.gov.in", "indiacode.nic.in")


def open_source(url: str) -> bool:
    """Whether a page is an official or openly licensed source (by its address: a rule of thumb, not a legal opinion)."""
    from urllib.parse import urlparse

    host = urlparse(str(url)).netloc.lower().removeprefix("www.")
    return bool(re.search(r"(^|\.)(gov|mil|int)(\.[a-z]{2})?$|\.nic\.in$|\.gouv\.fr$|\.bund\.de$", host)) or any(host == h or host.endswith("." + h) for h in OPEN_HOSTS)


def may_show(fact: dict | None, show: str) -> bool:
    """A source's own words go on screen only when the sentence was found in its page and the creator allows that kind of source."""
    return bool(fact) and fact.get("confirmed") == "yes" and (show == "any" or (show == "open" and open_source(fact.get("url", ""))))


def _facts_block(data: dict, show: str = "any") -> str:
    def mark(f: dict) -> str:
        if f.get("confirmed") != "yes":
            return "NOT confirmed: do not show or quote it"
        return "confirmed in the page" if may_show(f, show) else "confirmed, but this source's words are NOT to be shown on screen: say it in your own words"

    return "\n".join(f"F{k}. {f['fact']} [{f.get('source') or ''} {f.get('date') or ''}]" + (f"\n    exact sentence ({mark(f)}): “{f['quote']}”" if f.get("quote") else "")
                     for k, f in enumerate(data.get("facts", []), 1))


def script_prompt(plan: dict, data: dict) -> str:
    import emotions as em

    examples = "\n".join(f"- {e.get('what')}: {str(e.get('detail', ''))[:900]}" for e in data.get("examples", []) if isinstance(e, dict))
    chapters = json.dumps([{k: c.get(k) for k in ("title", "goal", "hook", "says", "shows", "claims", "ends_on", "how_said", "seconds")} | {"words": round(c["seconds"] * WORDS_PER_SECOND)}
                           for c in plan["chapters"]], ensure_ascii=False, indent=1)
    other = {"hi": "\nThe voice speaks Hindi, written in Devanagari, the everyday Hindi of a creator (common English words such as code, function, video stay as people say them). "
                   "Titles, cards, flow boxes and captions on screen are in Hindi too; code, commands, names of things and a source's own sentence stay as they are.\n",
             "hinglish": "\nThe voice speaks Hinglish (Hindi in Latin letters, the way people text), and so does the text on screen; code and a source's own sentence stay as they are.\n"
             }.get(plan.get("language"), "")
    show = plan.get("show_sources", "any")
    seen = "\n".join(f"{m['id']} (chapter {m['chapter']}, {m['kind']}" + (f", {m['seconds']:.0f} s" if m.get("seconds") else "") + f"): {m['what']}" + (f" — shows: {m['shows']}" if m.get("shows") else "")
                     for m in plan.get("media", []) if m.get("file"))
    return f"""Write the script of this explainer video from its approved plan: every spoken line, and what is on screen while it is said.
One voice narrates. The pictures are drawn by a program from the "scenes" you give, exactly as written, so they must be complete and correct.
Title: {plan['title']}  ·  For: {plan.get('audience', '')}  ·  Promise: {plan.get('promise', '')}{other}
## The plan (the creator approved it; keep its chapters, order, opening and closing lines, and what it shows)
{chapters}
## The research (the only facts there are)
{_facts_block(data, show)}
## Examples found
{examples or "(none)"}
## Real pictures and video found (each was looked at and shows what it says; nothing else exists)
{seen or "(none: everything is drawn)"}

## What is said
- TRUE: only what the research supports. A line that states something checkable carries "fact": the fact number. Numbers, names, dates and
  versions exactly as in the research. Where the plan marks a claim as not confirmed, say it carefully or leave it out. Never invent a quote.
- CONVINCING: the concrete case before the rule, the viewer's own mistake before the theory, one idea per line, plain words.
  No filler ("in this video", "let's dive in", "it is important to note"), no hype words, no rhetorical padding.
- CATCHY: each chapter opens with the plan's hook as its first line and closes on the plan's ending. Each chapter has one line worth repeating.
- EVERY chapter stands alone (it may be posted as a clip): nothing refers to another chapter.
- For the ear: sentences of at most 22 words. "text" is exactly what the voice says, so write symbols and code names the way they are said
  aloud ("dunder defaults", "f of one"); when that differs from how it is written, give "shown": the line as a caption should spell it.
- Each chapter about its "words" count ({WORDS_PER_SECOND} words a second).
- "emotion": one of [{em.prompt_list()}]; "intensity" 0.0-1.0; "pace" slow|normal|fast; "emphasis": the 0-2 words that carry the line;
  "pause_after" 0.2-1.2 seconds (longer after a reveal).

## What is on screen
Each chapter has "scenes" (numbered from 1 in the chapter). Each line names the "scene" on screen while it is said and its "step" (how far that
scene has built up). Steps only go forward inside a scene. Something new should appear every 4-10 seconds: a next step or a next scene.
Never talk about something that is not on screen. Scene forms:
{SCENE_FORMS}
- Code, commands, outputs and error texts must be real and runnable, taken from the research's examples; output exactly as the tool prints it.
- {'"article" and "quote" only for a fact whose sentence is marked confirmed in the page. Use them where a claim needs proof: the viewer sees the source say it.' if show != "none" else 'Do NOT use "article" or "quote" scenes: no source page is shown in this video. Name the source in the spoken line and on a card instead.'}
- Text on screen is short. The voice explains; the screen shows.
- Real pictures and video are what make this a video and not slides. USE EVERY ONE FOUND, and most of them more than once:
  * as a "footage" scene at the moment the voice first talks about exactly that thing (hold it 3-8 seconds); a chapter that has one
    opens on it, under its hook;
  * and "behind" the card, quote, flow, timeline, compare, example or chart scenes that are about that same thing (a flow of the
    landing steps over the photo of the lander; a card about the launch over the launch). A drawn scene whose subject has a found
    picture carries it behind; the same picture may be behind several scenes.
  Judgment: never a picture while the voice is on something else, never behind code, terminal, debug or a source page, and where a
  chapter has nothing real that fits (logic, maths, an abstract idea), drawn scenes alone are right. A picture from another chapter may
  be used when the voice comes back to that thing.

Answer with only this JSON:
{{"chapters": [{{"scenes": [...], "lines": [{{"text": "...", "shown": "", "scene": 1, "step": 1, "focus": [], "fact": 0, "emotion": "...", "intensity": 0.6,
   "pace": "normal", "emphasis": ["word"], "pause_after": 0.35}}]}}]}}
with one entry per chapter of the plan, in order."""


def script_revise_prompt(script: dict, notes: list[dict], plan: dict, data: dict, earlier: str = "") -> str:
    said = "\n".join(f"- {'chapter ' + str(n['chapter']) + ': ' if n.get('chapter') else 'the whole video: '}{n['text']}" for n in notes)
    which = sorted({n["chapter"] for n in notes if n.get("chapter")}) if all(n.get("chapter") for n in notes) else list(range(1, len(script["chapters"]) + 1))
    now = json.dumps([{"chapter": k, "title": script["chapters"][k - 1]["title"], "scenes": script["chapters"][k - 1]["scenes"], "lines": script["chapters"][k - 1]["lines"]} for k in which],
                     ensure_ascii=False, indent=1)
    return f"""The creator read the script of their explainer video and wants changes. Make exactly those changes and nothing else: a line or scene
no note is about stays as it is. A note about how something is made (what to show, in what order, how it is said) is followed even where you
would have done it differently. Everything said must still rest on the research; every chapter must still stand alone.

## The creator's notes
{said}

## The script now (the chapters concerned)
{now}

## The research (the only facts there are)
{_facts_block(data, plan.get("show_sources", "any"))}

## Scene forms
{SCENE_FORMS}

{earlier}
Answer with only this JSON: {{"chapters": [{{"chapter": 2, "scenes": [...], "lines": [...]}}], "changed": ["one short sentence per change"]}}
with every chapter given above, whole, in the same form."""


def check_prompt(script: dict, data: dict) -> str:
    lines = "\n".join(f"{c}.{k}  {ln['text']}" + (f"  [F{ln['fact']}]" if ln.get("fact") else "") for c, ch in enumerate(script["chapters"], 1) for k, ln in enumerate(ch["lines"], 1))
    screens = "\n".join(f"{c}.s{k}  {json.dumps({a: b for a, b in sc.items() if a != 'type'}, ensure_ascii=False)[:700]}"
                        for c, ch in enumerate(script["chapters"], 1) for k, sc in enumerate(ch["scenes"], 1) if sc["type"] not in ("article", "quote"))
    return f"""Check the script of an explainer video against its research before it is spoken in the creator's voice. Be strict and exact.
A problem is: a line or on-screen text that states something the research does not say, says more than its source sentence supports,
gets a number, name, date or version wrong, presents a disputed point as settled, or code/output that would not run or print as shown.
Style is not a problem. A line with no checkable claim is not a problem.

## Research
{_facts_block(data)}
## What people get wrong or disagree on
{chr(10).join('- ' + str(d) for d in data.get('disputed', [])) or '(none)'}

## Spoken lines (chapter.line)
{lines}
## On screen (chapter.scene)
{screens}

Answer with only this JSON: {{"problems": [{{"where": "2.3 or 2.s1", "problem": "what is wrong, one sentence", "fix": "for a spoken line: the corrected line, as short as the original; otherwise empty"}}]}}
An empty list when everything holds."""


def _rows(v) -> list[dict]:
    out = []
    for r in (v.splitlines() if isinstance(v, str) else v or []):
        if isinstance(r, dict):
            t = str(r.get("t", "out")).lower()
            out.append({"t": t if t in ("in", "out", "err", "ok") else "out", "text": str(r.get("text", ""))[:200]})
        else:
            r = str(r)
            out.append({"t": "in" if re.match(r"\s*(\$|>>>|\.\.\.|>|%) ", r) else "err" if re.search(r"Error|Traceback|Exception|StopIteration|Interrupt|FAILED|panic", r) else "out", "text": r[:200]})
    return out[:30]


def _strs(v, n: int, width: int = 400) -> list[str]:
    return [str(x)[:width] for x in (v.splitlines() if isinstance(v, str) else v or []) if isinstance(x, (str, int, float))][:n]


def _ints(v, n: int = 12) -> list[int]:
    out = []
    for x in v if isinstance(v, list) else [v]:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return [x for x in out if x > 0][:n]


BEHIND_OK = ("card", "quote", "flow", "example", "compare", "chart", "timeline")


def clean_scene(sc: dict, facts: list[dict], show: str = "any", media: dict | None = None) -> dict:
    out = _clean_scene(sc, facts, show, media or {})
    b = sc.get("behind")
    m = (media or {}).get(str(b.get("id", "") if isinstance(b, dict) else b or "").strip().lower())
    if m and out["type"] in BEHIND_OK:
        out["behind"] = {k: m[k] for k in ("id", "file", "kind", "credit", "seconds") if k in m}
    return out


def _clean_scene(sc: dict, facts: list[dict], show: str, media: dict) -> dict:
    """A scene the drawing step can rely on. A source passage whose sentence is not confirmed in
    its page, or whose kind of source the creator does not want shown, is never shown as one: it
    becomes a plain card."""
    kind = str(sc.get("type", "")).strip().lower()
    title = str(sc.get("title", "")).strip()[:90]
    try:
        k = int(str(sc.get("fact", 0)).lstrip("Ff") or 0)
    except ValueError:
        k = 0
    fact = facts[k - 1] if 1 <= k <= len(facts) else None
    if kind in ("article", "quote"):
        if may_show(fact, show):
            return {"type": kind, "fact": k}
        return {"type": "card", "text": str((fact or {}).get("fact") or sc.get("text") or sc.get("what") or "")[:160], "sub": str((fact or {}).get("source", ""))[:80], **({"fact": k} if fact else {})}
    if kind == "footage" and str(sc.get("media", "")).strip().lower() in media:
        m = media[str(sc["media"]).strip().lower()]
        return {"type": kind, "media": m["id"], "caption": str(sc.get("caption", "")).strip()[:120], **{a: m[a] for a in ("file", "kind", "credit", "page", "seconds") if a in m}}
    if kind == "footage" and str(sc.get("search", "")).strip():
        kept = {a: str(sc[a]) for a in ("file", "credit", "page") if sc.get(a) and Path(str(sc.get("file", ""))).is_file()}
        return {"type": kind, "search": " ".join(str(sc["search"]).split())[:80], "caption": str(sc.get("caption", "")).strip()[:120], **kept}
    if kind == "flow":
        return {"type": kind, "title": title, "boxes": _strs(sc.get("boxes"), 8, 90)}
    if kind == "code":
        return {"type": kind, "title": title, "lines": _strs(sc.get("lines") or sc.get("code"), 40, 200)}
    if kind == "terminal":
        return {"type": kind, "title": title, "lines": _rows(sc.get("lines"))}
    if kind == "debug":
        return {"type": kind, "title": title, "code": _strs(sc.get("code"), 30, 200), "run": _rows(sc.get("run")), "cause": _ints(sc.get("cause")),
                "fixed": _strs(sc.get("fixed"), 30, 200), "changed": _ints(sc.get("changed")), "passes": _rows(sc.get("passes"))}
    if kind == "example":
        return {"type": kind, "title": title, "lines": _strs(sc.get("lines"), 7, 200)}
    if kind == "compare":
        side = lambda v: {"title": str((v or {}).get("title", ""))[:70], "lines": _strs((v or {}).get("lines"), 8, 160), "code": bool((v or {}).get("code"))}  # noqa: E731
        return {"type": kind, "title": title, "left": side(sc.get("left") if isinstance(sc.get("left"), dict) else {}),
                "right": side(sc.get("right") if isinstance(sc.get("right"), dict) else {}), "good_bad": bool(sc.get("good_bad"))}
    if kind == "chart":
        bars = [{"label": str(b.get("label", ""))[:40], "value": b.get("value")} for b in sc.get("bars") or [] if isinstance(b, dict) and isinstance(b.get("value"), (int, float))][:8]
        if bars:
            return {"type": kind, "title": title, "unit": str(sc.get("unit", ""))[:6], "bars": bars, **({"fact": k} if fact else {})}
    if kind == "timeline":
        pts = [{"when": str(p.get("when", ""))[:24], "what": str(p.get("what", ""))[:140]} for p in sc.get("points") or [] if isinstance(p, dict)][:7]
        if pts:
            return {"type": kind, "title": title, "points": pts}
    return {"type": "card", "text": str(sc.get("text") or sc.get("what") or sc.get("caption") or title or "")[:160], "sub": str(sc.get("sub", ""))[:160], **({"fact": k} if fact else {})}


def clean_chapter(raw: dict, plan_ch: dict, facts: list[dict], show: str = "any", media: dict | None = None) -> dict:
    import emotions as em

    scenes = [clean_scene(sc, facts, show, media) for sc in raw.get("scenes") or [] if isinstance(sc, dict)][:12] or [{"type": "card", "text": plan_ch["title"], "sub": ""}]
    lines, at = [], (1, 0)
    for ln in raw.get("lines") or []:
        if not isinstance(ln, dict) or not str(ln.get("text", "")).strip():
            continue
        num = lambda key, d, lo, hi: min(hi, max(lo, float(ln.get(key, d)))) if isinstance(ln.get(key, d), (int, float)) else d  # noqa: E731
        scene = int(num("scene", at[0], 1, len(scenes)))
        step = int(num("step", 0, 0, 40))
        if scene == at[0] and 0 < step < at[1]:
            step = at[1]  # (a scene only builds up)
        at = (scene, step)
        k = _ints(ln.get("fact"), 1)
        shown = " ".join(str(ln.get("shown") or "").split())[:300]
        text = " ".join(str(ln["text"]).split())[:400]
        lines.append({"text": text, **({"shown": shown} if shown and shown != text else {}), "scene": scene, "step": step, "focus": _ints(ln.get("focus")),
                      "fact": k[0] if k and k[0] <= len(facts) else 0, "emotion": em.resolve(ln.get("emotion")), "intensity": round(num("intensity", 0.6, 0.0, 1.0), 2),
                      "pace": ln.get("pace") if ln.get("pace") in ("slow", "normal", "fast") else "normal",
                      "emphasis": [str(w)[:40] for w in ln.get("emphasis") or [] if isinstance(w, str)][:2], "pause_after": round(num("pause_after", 0.35, 0.15, 1.5), 2)})
    if not lines:
        raise AutocutError(f"the script came back without lines for “{plan_ch['title']}”")
    return {"title": plan_ch["title"], "clip_title": plan_ch.get("clip_title", ""), "scenes": scenes, "lines": lines[:80]}


def plan_sig(plan: dict) -> str:
    import hashlib

    found = [(m["id"], m.get("file", "")) for m in plan.get("media", [])]
    return hashlib.sha1(json.dumps([plan["chapters"], found] if found else plan["chapters"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def media_needs(plan: dict) -> list[dict]:
    """The plan's things to be seen, numbered through the video (m1, m2…); what was already found
    for the same words is kept."""
    old = {(m["chapter"], m["search"], m.get("wanted", m["kind"])): m for m in plan.get("media", [])}
    out = []
    for c, ch in enumerate(plan["chapters"], 1):
        for m in ch.get("media", []):
            had = old.get((c, m["search"], m["kind"]), {})
            out.append({**m, "wanted": m["kind"], **{k: v for k, v in had.items() if k in ("file", "kind", "credit", "page", "seconds", "missing", "for", "own", "shows")},
                        "id": f"m{len(out) + 1}", "chapter": c})
    return out


def apply_checks(script: dict, problems: list) -> list[dict]:
    """A spoken line found untrue is replaced by its corrected wording; everything found is kept
    in the script's `checks` for the creator to read."""
    out = []
    for p in problems or []:
        if not isinstance(p, dict) or not str(p.get("problem", "")).strip():
            continue
        where, fix = str(p.get("where", "")).strip(), " ".join(str(p.get("fix") or "").split())
        m = re.fullmatch(r"(\d+)\.(\d+)", where)
        item = {"where": where, "problem": str(p["problem"]).strip()[:300], "corrected": False}
        if m and fix:
            c, k = int(m.group(1)), int(m.group(2))
            if 1 <= c <= len(script["chapters"]) and 1 <= k <= len(script["chapters"][c - 1]["lines"]):
                ln = script["chapters"][c - 1]["lines"][k - 1]
                item.update(was=ln["text"], now=fix[:400], corrected=True)
                ln["text"] = fix[:400]
                ln.pop("shown", None)
        out.append(item)
    return out[:40]


def _scene_text(sc: dict, facts: list[dict]) -> str:
    t = sc["type"]
    f = facts[sc["fact"] - 1] if sc.get("fact") and sc["fact"] <= len(facts) else {}
    if t in ("article", "quote"):
        return f"{'the source page with this sentence marked' if t == 'article' else 'the exact words'}: “{f.get('quote', '')}” ({f.get('source', '')})"
    if t == "footage" and sc.get("media"):
        return f"{sc['media']} full screen ({sc.get('kind', 'photo')}: {sc.get('credit', '')}): {sc.get('caption', '')}"
    if t == "footage":
        return f"a photo of “{sc['search']}”: {sc.get('caption', '')}" + (f"  \n  found: {sc['credit']}  \n  {sc.get('page', '')}" if sc.get("credit") else "  (not looked for yet)")
    if t == "flow":
        return " → ".join(sc["boxes"])
    if t == "code":
        return (sc["title"] + ":\n" if sc["title"] else "\n") + "\n".join("      " + ln for ln in sc["lines"])
    if t == "terminal":
        return "\n" + "\n".join("      " + r["text"] for r in sc["lines"])
    if t == "debug":
        return "the bug, its cause (line " + ", ".join(map(str, sc["cause"])) + "), the fix, the passing run\n" + "\n".join("      " + ln for ln in sc["code"]) \
            + "\n      ---\n" + "\n".join("      " + r["text"] for r in sc["run"]) + "\n      --- fixed\n" + "\n".join("      " + ln for ln in sc["fixed"])
    if t == "example":
        return sc["title"] + ": " + " / ".join(sc["lines"])
    if t == "compare":
        return f"{sc['left']['title']} ({' / '.join(sc['left']['lines'])})  against  {sc['right']['title']} ({' / '.join(sc['right']['lines'])})"
    if t == "chart":
        return sc["title"] + ": " + ", ".join(f"{b['label']} {b['value']:g}{sc['unit']}" for b in sc["bars"])
    if t == "timeline":
        return " · ".join(f"{p['when']}: {p['what']}" for p in sc["points"])
    return sc.get("text", "") + (f" ({sc['sub']})" if sc.get("sub") else "")


def script_md(script: dict, facts: list[dict]) -> str:
    words = sum(len(ln["text"].split()) for ch in script["chapters"] for ln in ch["lines"])
    out = [f"# {script['title']}: the script", "", f"**Status:** {script.get('status', 'draft')}  ·  {words} words, about {words / WORDS_PER_SECOND / 60:.1f} min", ""]
    if script.get("checks"):
        out += ["## The check against the research found", ""]
        out += [f"- {c['where']}: {c['problem']}" + (f"  \n  was: “{c['was']}”  \n  now: “{c['now']}”" if c.get("corrected") else "  \n  **not corrected: for you to decide**") for c in script["checks"]] + [""]
    for n, ch in enumerate(script["chapters"], 1):
        out += [f"## {n}. {ch['title']}", "", "On screen:"]
        out += [f"- **{k}. {sc['type']}**: {_scene_text(sc, facts)}" + (f"  \n  behind it, dimmed: {sc['behind']['id']}" if sc.get("behind") else "") for k, sc in enumerate(ch["scenes"], 1)]
        out += ["", "Said (the picture on screen in brackets):"]
        out += [f"- [{ln['scene']}" + (f".{ln['step']}" if ln["step"] else "") + f"] {ln['text']}" + (f"  *(F{ln['fact']}: {facts[ln['fact'] - 1].get('source', '')})*" if ln.get("fact") else "")
                for ln in ch["lines"]]
        out += [""]
    if script.get("notes"):
        out += ["## Your notes so far", ""] + [f"- {n['when']}  {'chapter ' + str(n['chapter']) + ': ' if n.get('chapter') else ''}{n['text']}" for n in script["notes"]] + [""]
    if script.get("changed"):
        out += ["## Changed after your last notes", ""] + [f"- {c}" for c in script["changed"]] + [""]
    out += ["---", "To change something: run it again with a note (\"3: show the error first\" for chapter 3, or a note for the whole video).",
            "To go on: approve the script." if script.get("status") != "approved" else "The script is approved."]
    return "\n".join(out) + "\n"


def found_media(plan: dict) -> dict:
    return {m["id"]: m for m in plan.get("media", []) if m.get("file") and Path(m["file"]).is_file()}


def write_script(plan: dict, data: dict, model: str | None, ask=None) -> dict:
    """The script of an approved plan, checked once against the research."""
    import script_writer as sw

    ask = ask or (lambda prompt: sw.extract_json(sw.ask_claude(prompt, model, timeout=1200)))
    log("Claude is writing the script (what is said, and what is on screen while it is said)")
    raw = ask(script_prompt(plan, data))
    chs = [c for c in (raw.get("chapters") if isinstance(raw, dict) else None) or [] if isinstance(c, dict)]
    if len(chs) != len(plan["chapters"]):
        raise AutocutError(f"the script came back with {len(chs)} chapters for a plan of {len(plan['chapters'])}: try again")
    script = {"title": plan["title"], "promise": plan.get("promise", ""), "language": plan.get("language", "en"), "plan": plan_sig(plan), "status": "draft", "notes": [],
              "chapters": [clean_chapter(c, p, data["facts"], plan.get("show_sources", "any"), found_media(plan)) for c, p in zip(chs, plan["chapters"])], "made": time.strftime("%Y-%m-%d %H:%M")}
    check_script(script, data, ask)
    return script


def check_script(script: dict, data: dict, ask) -> None:
    log("checking every line against the research")
    try:
        found = ask(check_prompt(script, data))
        script["checks"] = apply_checks(script, found.get("problems") if isinstance(found, dict) else [])
    except AutocutError as e:
        script["checks"] = [{"where": "", "problem": f"the check could not be run ({e})", "corrected": False}]
    fixed = sum(1 for c in script["checks"] if c["corrected"])
    log(f"check: {len(script['checks'])} problem(s) found, {fixed} line(s) corrected" + (f", {len(script['checks']) - fixed} left for you (see script.md)" if len(script["checks"]) > fixed else ""))
    for c in script["checks"][:8]:
        log(f"  {c['where']}: {c['problem'][:140]}")


def revise_script(script: dict, notes: list[dict], plan: dict, data: dict, model: str | None, ask=None, folder=None) -> dict:
    import script_writer as sw

    ask = ask or (lambda prompt: sw.extract_json(sw.ask_claude(prompt, model, timeout=1200)))
    log(f"your notes on the script: {'; '.join(n['text'][:80] for n in notes)}")
    import fix_log

    answer = ask(script_revise_prompt(script, notes, plan, data, fix_log.block(folder) if folder else ""))
    for c in (answer.get("chapters") if isinstance(answer, dict) else None) or []:
        try:
            k = int(c.get("chapter", 0))
        except (TypeError, ValueError, AttributeError):
            continue
        if 1 <= k <= len(script["chapters"]):
            script["chapters"][k - 1] = clean_chapter(c, plan["chapters"][k - 1], data["facts"], plan.get("show_sources", "any"), found_media(plan))
    when = time.strftime("%Y-%m-%d %H:%M")
    ch = answer.get("changed") if isinstance(answer, dict) else None
    script["notes"] = (script.get("notes") or []) + [{**n, "when": when} for n in notes]
    script["changed"] = [str(c).strip()[:300] for c in (ch if isinstance(ch, list) else [ch]) if str(c or "").strip()][:20]
    script["status"] = "draft"
    for c in script["changed"]:
        log(f"changed: {c}")
    if folder:
        fix_log.add(folder, [said_note(n) for n in notes], script["changed"], tool="explain: script")
    check_script(script, data, ask)
    return script


def fetch_footage(script: dict, folder: Path, find=None) -> int:
    """A free-licensed photo for every footage scene (Wikimedia Commons, then Openverse), kept in the
    video's folder with its credit. A scene nothing is found for becomes a card. Returns how many were found."""
    import topic_video as tv

    find = find or (lambda q: tv.commons_search(q, 5) or tv.openverse_search(q, 5))
    media, got = folder / "media", 0
    for c, ch in enumerate(script["chapters"], 1):
        for k, sc in enumerate(ch["scenes"], 1):
            if sc["type"] != "footage" or (sc.get("file") and Path(sc["file"]).is_file()):
                continue
            hit = None
            for item in find(sc["search"]) or []:
                media.mkdir(parents=True, exist_ok=True)
                dst = media / f"c{c}-s{k}{Path(str(item.get('url', '')).split('?')[0]).suffix.lower() if Path(str(item.get('url', '')).split('?')[0]).suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp') else '.jpg'}"
                if item.get("url") and tv.download(item["url"], dst):
                    hit = (dst, item)
                    break
            if hit:
                dst, item = hit
                sc.update(file=str(dst), page=str(item.get("page", "")),
                          credit=", ".join(x for x in (str(item.get("title", ""))[:60], str(item.get("author", ""))[:40], str(item.get("license", "")), str(item.get("provider", ""))) if x))
                got += 1
                log(f"photo for “{sc['search']}”: {sc['credit']}")
            else:
                log(f"no free photo found for “{sc['search']}”: a card is shown instead")
                ch["scenes"][k - 1] = {"type": "card", "text": sc.get("caption") or sc["search"], "sub": ""}
    return got


def run_the_code(script: dict) -> None:
    """The script's Python is run (in the sandbox) and the real output put on screen; what was found joins the script's checks."""
    import explain_run as xr

    found = xr.check_script(script)
    script["checks"] = [c for c in script.get("checks", []) if not str(c.get("problem", "")).startswith(("run with Python", "the FIXED code"))] + found
    ran = script.get("ran", {})
    log(f"code: {ran.get('runs', 0)} run(s) with {ran.get('with', 'Python')}, {sum(1 for f in found if f['corrected'])} output(s) corrected to the real one"
        + (f", {sum(1 for f in found if not f['corrected'])} for you to look at" if any(not f["corrected"] for f in found) else ""))
    for f in found[:6]:
        log(f"  {f['where']}: {f['problem'][:150]}")


STEPS = ("plan", "script", "video")


def make(want: str, out_dir: str | None = None, *, minutes: float = 6, language: str = "en", kind: str = "auto", hints: str = "", review: bool = False,
         approve: bool = False, notes: list[str] | None = None, fresh: bool = False, model: str | None = None, to: str = "plan", note_for: str = "auto",
         voice: str = "mac", look: str = "clean", aspect: str = "horizontal", captions: str = "auto", music: str | None = None, clips: str = "same",
         takes: int = 2, speech_pace: str = "reference", show_sources: str = "any", run_code: bool = False, media: list[str] | None = None,
         find_media: bool = True) -> dict:
    """An explainer video for `want`, in its folder, as far as `to` (plan, script, video). `review`:
    the plan, then the script, wait for approval (`approve`); otherwise each counts as approved once
    written. `notes`: changes the creator asks for, applied to the step named by `note_for` (auto:
    the script when there is one, else the plan)."""
    import script_writer as sw
    import topic_video as tv

    want = " ".join((want or "").split())
    if not want:
        raise AutocutError("say what the video should explain")
    if kind not in KINDS:
        raise AutocutError(f"kind must be one of {', '.join(KINDS)}")
    if show_sources not in SHOW_SOURCES:
        raise AutocutError(f"sources are shown: {', '.join(SHOW_SOURCES)}")
    if to not in STEPS or note_for not in ("auto", "plan", "script"):
        raise AutocutError("go as far as plan, script or video; notes are for the plan or the script")
    minutes = min(60.0, max(1.0, float(minutes)))
    folder = Path(out_dir).expanduser().resolve() if out_dir else Path.home() / "Movies" / "EI" / "Explainers" / (sw.safe(want)[:50] or "explainer")
    folder.mkdir(parents=True, exist_ok=True)
    if out_dir and (approve or notes) and not fresh:
        # going on with the explainer in this folder: its own words count, not how they were typed again
        try:
            want = json.loads((folder / "plan.json").read_text(encoding="utf-8")).get("want") or want
        except (OSError, ValueError):
            pass
    pj, rj = folder / "plan.json", folder / "research.json"
    plan = None
    if pj.is_file() and not fresh:
        try:
            old = json.loads(pj.read_text(encoding="utf-8"))
            if old.get("want") == want and old.get("chapters"):
                plan = old
        except ValueError:
            plan = None
    asked = [n for n in (parse_note(t, len(plan["chapters"]) if plan else MAX_CHAPTERS) for t in notes or []) if n]
    sj = folder / "script.json"
    script = None
    if plan and sj.is_file() and not fresh:
        try:
            script = json.loads(sj.read_text(encoding="utf-8"))
            script = script if script.get("plan") == plan_sig(plan) and script.get("chapters") else None
        except ValueError:
            script = None
    for_script = asked if (note_for == "script" or (note_for == "auto" and script is not None)) else []
    asked = [] if for_script else asked
    if for_script and script is None:
        raise AutocutError("there is no script yet for these notes: they can change the plan (notes for: plan)")
    plan_was_approved = bool(plan and plan.get("status") == "approved")
    weights = {"research": 0.4, "plan": 0.3, "truth": 0.15, "media": 0.15}
    if to != "plan":
        weights = {"research": 0.15, "truth": 0.05, "plan": 0.1, "media": 0.1, "script": 0.2, **({"make": 0.4} if to == "video" else {})}
    ac.progress_plan(weights)
    data = None
    if rj.is_file() and not fresh:
        try:
            data = json.loads(rj.read_text(encoding="utf-8"))
            data = data if data.get("want") == want and data.get("facts") else None
        except ValueError:
            data = None
    with ac.stage("research"):
        if data is None:
            log("Claude is researching it on the web (facts with the sentence of the source that says so)")
            t0 = time.time()
            data = sw.extract_json(tv.ask_claude_research(research_prompt(want, minutes, language, kind, hints), model))
            data["facts"] = [f for f in data.get("facts", []) if isinstance(f, dict) and f.get("fact")][:60]
            if not data["facts"]:
                raise AutocutError("the research found no usable facts")
            data["want"] = want
            log(f"research: {len(data['facts'])} facts, {len(data.get('examples', []))} examples in {time.time() - t0:.0f}s")
            plan = None  # (new research: the plan is written from it)
        else:
            log(f"reusing the research from {time.strftime('%d %b %H:%M', time.localtime(rj.stat().st_mtime))}")
    with ac.stage("truth"):
        if any("confirmed" not in f or (f["confirmed"] == "yes" and "context" not in f) for f in data["facts"]):
            yes, total = confirm_quotes(data)
            log(f"truth: {yes} of {total} quoted sentences stand word for word in their source page"
                + (f"; {len(data['facts']) - total} fact(s) came without a sentence to check" if total < len(data["facts"]) else ""))
        rj.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    with ac.stage("plan"):
        if plan is None:
            log("Claude is planning the video in chapters")
            fresh_plan = clean_plan(sw.extract_json(sw.ask_claude(plan_prompt(want, minutes, language, kind, data, hints), model, timeout=900)), data)
            plan = {**fresh_plan, "want": want, "minutes": minutes, "language": language, "kind": kind, "hints": hints, "notes": [], "status": "draft",
                    "made": time.strftime("%Y-%m-%d %H:%M")}
        if asked:
            log(f"your notes: {'; '.join(n['text'][:80] for n in asked)}")
            import fix_log

            answer = sw.extract_json(sw.ask_claude(revise_prompt(plan, asked, data, fix_log.block(folder)), model, timeout=900))
            revised = clean_plan(answer, data)
            when = time.strftime("%Y-%m-%d %H:%M")
            plan = {**plan, **revised, "notes": (plan.get("notes") or []) + [{**n, "when": when} for n in asked],
                    "changed": [str(c).strip()[:300] for c in (answer.get("changed") if isinstance(answer.get("changed"), list) else [answer.get("changed")]) if str(c or "").strip()][:20],
                    "status": "draft"}
            for c in plan["changed"]:
                log(f"changed: {c}")
            fix_log.add(folder, [said_note(n) for n in asked], plan["changed"], tool="explain: plan")
        if asked:
            script = None  # (the plan changed: its script is written again)
    with ac.stage("media"):
        before = plan_sig(plan)
        plan["media"] = media_needs(plan) if find_media else []
        if plan["media"]:
            import explain_media as xm

            got, wanted = xm.find(plan, folder, media, model)
            log(f"real pictures and video: {got} of {wanted} found" + (f"; not found: {'; '.join(m['what'][:50] for m in plan['media'] if not m.get('file'))[:300]}" if got < wanted else ""))
        if script is not None and plan_sig(plan) != before and script.get("plan") == before:
            script = None  # (other pictures than the script was written for)
    with ac.stage("plan"):
        if approve or (not review and plan.get("status") != "approved"):
            plan["status"] = "approved"
            plan["approved"] = time.strftime("%Y-%m-%d %H:%M")
        elif plan.get("status") != "approved":
            plan["status"] = "waiting for your approval"
        tmp = pj.with_name(pj.name + ".part")
        tmp.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(pj)
        md = folder / "plan.md"
        md.write_text(plan_md(plan), encoding="utf-8")
    weak = [(n, cl["claim"]) for n, c in enumerate(plan["chapters"], 1) for cl in c["claims"] if cl["confirmed"] != "yes"]
    log(f"plan: {plan['title']} · {len(plan['chapters'])} chapters, about {sum(c['seconds'] for c in plan['chapters']) / 60:.1f} min · {plan['status']}")
    for n, c in enumerate(plan["chapters"], 1):
        log(f"  {n}. {c['title']} ({c['seconds']} s): {', '.join(sorted({s['type'] for s in c['shows']})) or 'no pictures planned'}")
    if weak:
        log(f"{len(weak)} claim(s) are not confirmed word for word in a source (marked in plan.md): " + "; ".join(f"ch {n}: {c[:60]}" for n, c in weak[:4]))
    res = {"outputs": [str(md), str(pj), str(rj)], "out_dir": str(folder), "status": plan["status"], "title": plan["title"], "chapters": len(plan["chapters"]),
           "seconds": sum(c["seconds"] for c in plan["chapters"]), "unconfirmed": len(weak), "facts": len(data["facts"]), "step": "plan",
           "changed": list(plan.get("changed") or []) if asked else []}
    if plan["status"] != "approved":
        log(f"the plan waits for you: read {md}, then approve it or give notes")
        return res
    if to == "plan":
        return res
    # 2. script
    with ac.stage("script"):
        plan["show_sources"] = show_sources
        if script is not None and script.get("show_sources", "any") != show_sources and not for_script:
            log(f"source pages are now shown: {SHOW_SOURCES[show_sources]}; the script is written again for that")
            script = None
        new = script is None
        if new:
            script = write_script(plan, data, model)
        if for_script:
            script = revise_script(script, for_script, plan, data, model, folder=folder)
        script["show_sources"] = show_sources
        if new or for_script or any(sc["type"] == "footage" and not sc.get("file") for ch in script["chapters"] for sc in ch["scenes"]):
            fetch_footage(script, folder)
        if run_code and (new or for_script or not script.get("ran")):
            run_the_code(script)
        if (approve and plan_was_approved and not new and not for_script) or (not review and script.get("status") != "approved"):
            script["status"] = "approved"
            script["approved"] = time.strftime("%Y-%m-%d %H:%M")
        elif script.get("status") != "approved":
            script["status"] = "waiting for your approval"
        tmp = sj.with_name(sj.name + ".part")
        tmp.write_text(json.dumps(script, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(sj)
        smd = folder / "script.md"
        smd.write_text(script_md(script, data["facts"]), encoding="utf-8")
    words = sum(len(ln["text"].split()) for ch in script["chapters"] for ln in ch["lines"])
    log(f"script: {words} words in {sum(len(ch['lines']) for ch in script['chapters'])} lines, {sum(len(ch['scenes']) for ch in script['chapters'])} scenes · {script['status']}")
    res = {**res, "outputs": [str(smd), str(sj)] + res["outputs"], "status": script["status"], "step": "script", "words": words, **({"changed": list(script.get("changed") or [])} if for_script else {}),
           "open_checks": sum(1 for c in script.get("checks", []) if not c["corrected"])}
    if script["status"] != "approved":
        log(f"the script waits for you: read {smd}, then approve it or give notes")
        return res
    if to == "script":
        return res
    # 3. voice, pictures, video, clips
    import explain_video as xv

    with ac.stage("make"):
        made = xv.make_video(folder, script, data["facts"], voice=voice, look=look, aspect=aspect, captions=None if captions == "auto" else captions, music=music,
                             clips=clips, takes=takes, speech_pace=speech_pace)
    return {**res, **made, "outputs": made["outputs"] + res["outputs"], "status": "made", "step": "video", "chapters": len(script["chapters"]), "chapter_times": made["chapters"]}


def add_args(p) -> None:
    p.add_argument("want", help="what the video should explain, in your own words")
    p.add_argument("--out-dir")
    p.add_argument("--minutes", type=float, default=6)
    p.add_argument("--language", default="en", choices=["en", "hi", "hinglish"])
    p.add_argument("--kind", default="auto", choices=list(KINDS))
    p.add_argument("--hints", default="", help="what to focus on, what to leave out, how it should come out (one per line)")
    p.add_argument("--review", action="store_true", help="the plan waits for your approval (otherwise it counts as approved once written)")
    p.add_argument("--approve", action="store_true", help="the plan is right: the next steps may use it")
    p.add_argument("--note", action="append", default=[], help="a change to the plan: \"3: show the error first\" (chapter 3) or a note for the whole video; may be given several times")
    p.add_argument("--note-for", default="auto", choices=["auto", "plan", "script"], help="what the notes change (auto: the script when there is one, else the plan)")
    p.add_argument("--to", default="plan", choices=list(STEPS), help="how far to go: the plan, the script, or the finished video")
    p.add_argument("--voice", default="mac", help="a cloned voice's name, or mac / mac:<name> for a macOS voice")
    p.add_argument("--look", default="clean", choices=["clean", "dark", "viral"], help="clean (paper, documentary), dark (code), viral (bold, word-by-word captions)")
    p.add_argument("--aspect", default="horizontal", choices=["horizontal", "vertical", "square"])
    p.add_argument("--captions", default="auto", choices=["auto", "sentence", "words", "none"])
    p.add_argument("--music", help="a music file to lay quietly under the voice")
    p.add_argument("--clips", default="same", choices=["none", "same", "vertical"], help="each chapter as a clip of its own: cut out of the video, or drawn again tall for Shorts")
    p.add_argument("--takes", type=int, default=2)
    p.add_argument("--speech-pace", default="reference")
    p.add_argument("--show-sources", default="any", choices=list(SHOW_SOURCES),
                   help="a source's own sentence on screen, marked: from any public page (briefly, with credit), only from official and openly licensed sources, or never")
    p.add_argument("--media", nargs="*", default=[], help="your own photos and videos (files or folders): looked at and used where they show what the plan wants seen")
    p.add_argument("--no-media", action="store_true", help="do not look for real photos and video: everything is drawn")
    p.add_argument("--run-code", action="store_true", help="run the script's Python (sandboxed: no network, no files outside a temporary folder) and show the real output")
    p.add_argument("--fresh", action="store_true", help="research and plan again")
    p.add_argument("--model")


def run_args(a) -> dict:
    return make(a.want, a.out_dir, minutes=a.minutes, language=a.language, kind=a.kind, hints=a.hints or "", review=a.review, approve=a.approve, notes=a.note,
                fresh=a.fresh, model=a.model, to=a.to, note_for=a.note_for, voice=a.voice, look=a.look, aspect=a.aspect, captions=a.captions, music=a.music,
                clips=a.clips, takes=a.takes, speech_pace=a.speech_pace, show_sources=a.show_sources, run_code=a.run_code, media=a.media, find_media=not a.no_media)
