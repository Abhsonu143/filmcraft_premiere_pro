#!/usr/bin/env python3
"""script_writer: voiceover scripts written for your voice clone, in your own talking style.

1. Style: learn how you (or a creator you like) talk from transcripts: .txt, .srt, .vtt files,
   or YouTube links (only the captions are downloaded, with yt-dlp). Measured: speaking rate,
   sentence length, how much Hindi/English is mixed, questions and exclamations, fillers,
   signature phrases, how videos open and close. A few real passages are kept as examples.
   Saved as a style profile you can reuse (`~/Library/Application Support/EI/styles/<name>.json`).

2. Script: Claude writes it (through the Claude Code app you are logged into, no API key), or
   the local Qwen model offline. The script is made for speech synthesis, segment by segment:
       text        what to say (numbers and symbols written as spoken words)
       emotion     calm, excited, curious, serious, warm, suspense, story, hook, cta, neutral
       pace        slow / normal / fast;  energy 0–1;  pitch low / mid / high
       pause_after seconds of silence after the segment
       emphasis    words to stress
       visual      what the picture should show (English), used by the voiceover editor to
                   pick footage
   Saved as `<name>.script.json` (for the voice) and `<name>.txt` (to read).

    python script_writer.py style --name mine talk1.srt talk2.txt https://youtu.be/...
    python script_writer.py write --topic "Spiti valley road trip" --language hinglish --seconds 90 --style mine --out spiti
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import autocut as ac
from autocut import AutocutError, log

EMOTIONS = ("neutral", "calm", "warm", "story", "excited", "curious", "serious", "suspense", "hook", "cta")
LANGUAGES = {"en": "English", "hi": "Hindi (Devanagari script)", "hinglish": "Hinglish: Hindi in Devanagari with the English words people really use written in English"}
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN_WORD = re.compile(r"\b[A-Za-z][A-Za-z']+\b")
FILLERS_EN = {"like", "basically", "actually", "literally", "so", "um", "uh", "you know", "i mean", "right", "okay"}
FILLERS_HI = {"तो", "मतलब", "यानी", "अच्छा", "हाँ", "ना", "बस", "वैसे", "देखो", "समझे"}


def styles_dir() -> Path:
    base = Path(os.environ["EI_SETTINGS"]).parent if os.environ.get("EI_SETTINGS") else Path.home() / "Library" / "Application Support" / "EI"
    return base / "styles"


# ---------------------------------------------------------------------------------------------
# Reading transcripts
# ---------------------------------------------------------------------------------------------


def _ts(t: str) -> float:
    t = t.strip().replace(",", ".")
    parts = t.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return 0.0
    while len(nums) < 3:
        nums.insert(0, 0.0)
    return nums[0] * 3600 + nums[1] * 60 + nums[2]


def parse_timed(text: str) -> list[dict]:
    """SRT or WebVTT -> cues [{start, end, text}]; YouTube's rolling auto-captions de-duplicated."""
    cues: list[dict] = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "")):
        lines = [l for l in block.strip().split("\n") if l.strip()]
        tl = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if tl is None:
            continue
        a, b = lines[tl].split("-->")[:2]
        body = " ".join(lines[tl + 1 :])
        body = re.sub(r"<[^>]+>", "", body)  # vtt word timing tags
        body = re.sub(r"\[[^\]]*\]|\([^)]*music[^)]*\)", " ", body, flags=re.I)
        body = re.sub(r"\s+", " ", body).strip()
        if body:
            cues.append({"start": _ts(a), "end": _ts(b.split()[0]), "text": body})
    out: list[dict] = []
    for c in cues:  # rolling captions repeat the previous line: keep only what is new
        if out:
            prev = out[-1]["text"]
            if c["text"] == prev or prev.endswith(c["text"]):
                continue
            if c["text"].startswith(prev):
                c = {**c, "text": c["text"][len(prev) :].strip()}
                if not c["text"]:
                    continue
            else:
                words = prev.split()
                for k in range(min(len(words), 12), 0, -1):  # overlap of the tail of the previous cue
                    if c["text"].startswith(" ".join(words[-k:])):
                        c = {**c, "text": c["text"][len(" ".join(words[-k:])) :].strip()}
                        break
                if not c["text"]:
                    continue
        out.append(c)
    return out


def yt_dlp_cmd() -> list[str]:
    """yt-dlp from EI's own Python (kept current with pip; YouTube changes often), else the system one."""
    import importlib.util

    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    raise AutocutError("yt-dlp is not installed (AI Engine Setup installs it)")


def pick_track(info: dict) -> tuple[str, bool] | None:
    """(language code, automatic?) of the best caption track: human captions in Hindi/English
    first, else the automatic captions of the video's own language (never a machine translation)."""
    subs = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    for code in ("hi", "en", "en-IN", "en-US", "en-GB"):
        if code in subs:
            return code, False
    lang = (info.get("language") or "").split("-")[0]
    for code in ([f"{lang}-orig", lang] if lang else []) + ["hi-orig", "en-orig"]:
        if code in auto and code.split("-")[0] in ("hi", "en"):
            return code, True
    return None


def fetch_youtube(url: str, folder: Path) -> Path | None:
    """Captions of one YouTube video (only the captions are downloaded)."""
    folder.mkdir(parents=True, exist_ok=True)
    base = yt_dlp_cmd() + ["--no-playlist", "--quiet", "--no-warnings"]
    proc = subprocess.run(base + ["-J", "--skip-download", url], capture_output=True, text=True, timeout=180)
    try:
        info = json.loads(proc.stdout)
    except ValueError:
        log(f"could not read {url}: {(proc.stderr or '').strip()[:160]}")
        return None
    tr = pick_track(info)
    if tr is None:
        log(f"no Hindi or English captions on {url}")
        return None
    code, is_auto = tr
    cmd = base + ["--skip-download", "--write-auto-subs" if is_auto else "--write-subs", "--sub-langs", code, "--sub-format", "vtt",
                  "--convert-subs", "srt", "-o", str(folder / "%(id)s.%(ext)s"), url]
    for attempt in range(3):
        subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        subs = sorted(folder.glob("*.srt"))
        if subs:
            log(f"captions ({code}{', automatic' if is_auto else ''}): {info.get('title', url)[:70]}")
            return subs[0]
        time.sleep(4 * (attempt + 1))  # YouTube rate limit
    return None


def load_sources(sources: list[str]) -> list[dict]:
    """Each source -> {name, text, cues (timed or [])}."""
    out = []
    with tempfile.TemporaryDirectory(prefix="ei-yt-") as tmp:
        for k, s in enumerate(sources):
            if re.match(r"https?://", s):
                p = fetch_youtube(s, Path(tmp) / f"v{k}")
                if p is None:
                    log(f"no captions found for {s}")
                    continue
                cues = parse_timed(p.read_text(encoding="utf-8", errors="replace"))
                out.append({"name": s, "cues": cues, "text": " ".join(c["text"] for c in cues)})
                continue
            path = Path(s)
            if not path.is_file():
                raise AutocutError(f"transcript not found: {s}")
            raw = path.read_text(encoding="utf-8", errors="replace")
            if path.suffix.lower() in (".srt", ".vtt") or "-->" in raw[:2000]:
                cues = parse_timed(raw)
                out.append({"name": path.name, "cues": cues, "text": " ".join(c["text"] for c in cues)})
            else:
                out.append({"name": path.name, "cues": [], "text": re.sub(r"\s+", " ", raw).strip()})
    return [o for o in out if o["text"]]


# ---------------------------------------------------------------------------------------------
# Style profile
# ---------------------------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?।])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 1]


def words_of(text: str) -> list[str]:
    return re.findall(r"[\wऀ-ॿ']+", text)


def analyse_style(sources: list[dict]) -> dict:
    texts = [s["text"] for s in sources]
    allt = " ".join(texts)
    words = words_of(allt)
    n = max(1, len(words))
    deva = sum(1 for w in words if DEVANAGARI.search(w))
    latin = sum(1 for w in words if re.fullmatch(r"[A-Za-z']+", w))
    deva_share = deva / n
    if deva_share > 0.2 and latin / n >= 0.08:
        lang = "hinglish"  # Hindi with real English words mixed in
    elif deva_share > 0.5:
        lang = "hi"
    else:
        lang = "en"
    sents = [s for t in texts for s in split_sentences(t)]
    sent_words = [len(words_of(s)) for s in sents] or [0]
    # speaking rate from timed captions: words per minute of speech
    timed = [(c["end"] - c["start"], len(words_of(c["text"]))) for s in sources for c in s["cues"] if 0 < c["end"] - c["start"] < 20]
    wpm = round(60 * sum(w for _, w in timed) / max(1.0, sum(d for d, _ in timed))) if timed else None
    low = [w.lower() for w in words]
    fillers = Counter(w for w in low if w in FILLERS_EN or w in FILLERS_HI)
    stop = {"the", "a", "to", "and", "of", "is", "in", "it", "that", "you", "i", "this", "for", "on", "we", "with", "है", "के", "में", "की", "और", "को", "से", "का", "भी", "तो", "हैं", "ये", "यह", "वो", "पर", "कि", "एक"}
    grams = Counter()
    for k in (2, 3):
        for i in range(len(low) - k + 1):
            g = low[i : i + k]
            if g[0] in stop or g[-1] in stop:
                continue
            grams[" ".join(g)] += 1
    signature = [g for g, c in grams.most_common(40) if c >= max(2, len(sources))][:15]
    openers = [" ".join(words_of(t)[:40]) for t in texts[:6]]
    closers = [" ".join(words_of(t)[-40:]) for t in texts[:6]]
    examples = []
    budget = 2400
    for t in texts:
        w = t.split()
        for frac in (0.0, 0.45):
            a = int(len(w) * frac)
            chunk = " ".join(w[a : a + 220])
            if chunk and budget > 0:
                examples.append(chunk)
                budget -= 220
    english_in_hindi = round(latin / max(1, deva + latin), 3) if deva else None
    return {
        "version": 1,
        "sources": [s["name"] for s in sources],
        "words": n,
        "language": lang,
        "devanagari_share": round(deva_share, 3),
        "english_word_share_in_hindi": english_in_hindi,
        "words_per_minute": wpm,
        "avg_sentence_words": round(sum(sent_words) / max(1, len(sent_words)), 1),
        "question_rate": round(sum(1 for s in sents if s.rstrip().endswith("?")) / max(1, len(sents)), 3),
        "exclamation_rate": round(sum(1 for s in sents if s.rstrip().endswith("!")) / max(1, len(sents)), 3),
        "fillers": [f for f, _ in fillers.most_common(8)],
        "signature_phrases": signature,
        "openers": openers,
        "closers": closers,
        "examples": examples,
    }


def save_style(name: str, profile: dict) -> Path:
    d = styles_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{safe(name)}.json"
    p.write_text(json.dumps({**profile, "name": name, "saved": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def load_style(name_or_path: str | None) -> dict | None:
    if not name_or_path:
        return None
    p = Path(name_or_path)
    if not p.is_file():
        p = styles_dir() / f"{safe(name_or_path)}.json"
    if not p.is_file():
        raise AutocutError(f"style profile not found: {name_or_path}")
    return json.loads(p.read_text(encoding="utf-8"))


def list_styles() -> list[str]:
    d = styles_dir()
    return sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []


def safe(name: str) -> str:
    return re.sub(r"[^\w\-]+", "-", name.strip()).strip("-")[:60] or "style"


# ---------------------------------------------------------------------------------------------
# Writing the script
# ---------------------------------------------------------------------------------------------


def voice_wpm(voice_name: str | None, language: str) -> float | None:
    """Speaking rate of a cloned voice (its references), so the script fits the asked length."""
    if not voice_name:
        return None
    try:
        import voice as vo

        prof, _ = vo.load_profile(voice_name)
    except AutocutError:
        return None
    st = prof.get("stats", {}).get("hi" if language in ("hi", "hinglish") else "en") or next(iter(prof.get("stats", {}).values()), {})
    wps = st.get("words_per_s")
    return float(wps) * 60 * 0.9 if wps else None  # the clone keeps a little of the read pace


def target_words(seconds: float, language: str, style: dict | None, wpm_override: float | None = None) -> int:
    wpm = wpm_override or (style or {}).get("words_per_minute") or {"en": 150, "hi": 135, "hinglish": 140}.get(language, 145)
    wpm = min(190, max(100, wpm))
    return max(15, int(seconds / 60 * wpm * 0.92))  # pauses take time too


def build_prompt(topic: str, *, niche: str, seconds: float, language: str, tone: str, audience: str, notes: str,
                 style: dict | None, footage: list[str] | None, wpm: float | None = None) -> str:
    nwords = target_words(seconds, language, style, wpm)
    lang_rule = {
        "en": "Write in natural spoken English.",
        "hi": "Write in natural spoken Hindi in Devanagari script (not shuddh/literary Hindi: the everyday Hindi a YouTuber speaks). Avoid English words unless there is no common Hindi one.",
        "hinglish": "Write in natural Hinglish: Hindi sentences in Devanagari, with the English words Indian creators actually use (video, camera, drone, budget, trip, subscribe …) written in English letters. Never write whole sentences in Roman-script Hindi.",
    }[language]
    style_txt = ""
    if style:
        style_txt = (
            "\n## The speaker's own style (match it closely)\n"
            f"- Speaking rate about {style.get('words_per_minute') or 'unknown'} words per minute; sentences about {style.get('avg_sentence_words')} words on average.\n"
            f"- Questions in {round(100 * style.get('question_rate', 0))}% of sentences, exclamations in {round(100 * style.get('exclamation_rate', 0))}%.\n"
            + (f"- English words make up {round(100 * style['english_word_share_in_hindi'])}% of their Hindi speech.\n" if style.get("english_word_share_in_hindi") else "")
            + (f"- Words and phrases they use: {', '.join(style.get('signature_phrases', [])[:12])}.\n" if style.get("signature_phrases") else "")
            + (f"- Typical fillers (use sparingly, only where natural): {', '.join(style.get('fillers', [])[:6])}.\n" if style.get("fillers") else "")
            + "- How their videos open:\n" + "\n".join(f"  > {o}" for o in style.get("openers", [])[:3])
            + "\n- Real passages of their speech (copy the voice, rhythm and attitude; never the content):\n"
            + "\n".join(f"  > {e}" for e in style.get("examples", [])[:6])
            + "\n"
        )
    foot = ""
    if footage:
        foot = "\n## Footage available for the edit (prefer visuals that exist)\n" + "\n".join(f"- {f}" for f in footage[:60]) + "\n"
    return f"""You are writing a voiceover script for a video. It will be spoken by a voice-clone text-to-speech model of the creator, then a video editor will cut footage to it, segment by segment.

## Brief
- Topic: {topic}
- Niche / channel type: {niche or "general creator"}
- Audience: {audience or "the creator's regular viewers"}
- Tone: {tone or "match the speaker's style"}
- Length: about {seconds:.0f} seconds of speech, about {nwords} words in total.
- Language: {lang_rule}
{('- Notes from the creator: ' + notes) if notes else ''}
{style_txt}{foot}
## How to write it
- Structure: a hook in the first 3 seconds (a surprising fact, a question, a bold claim, or the payoff teased), then build, the core, a peak moment, and a short ending (with a call to action only if it fits the style).
- Write for the ear: short spoken sentences, contractions, rhythm. Vary sentence length. Real emotion, not marketing.
- Depth: concrete details, numbers, sensory words, one small story or personal moment. No filler sentences.
- For the speech model: write numbers, dates, units and symbols as words in the script's language (e.g. "दो हज़ार", "fifteen percent"); no emojis, hashtags, brackets, stage directions or abbreviations inside "text".
- One segment = one breath / one idea, 4 to 25 words. Mark the delivery of every segment.
- "visual" is an English description of the ideal shot for that segment (what is in frame, shot type), used to choose footage.

## Output
Reply with only a JSON object, no commentary, in exactly this form:
{{"title": "...", "language": "{language}", "hook": "the hook line",
 "segments": [{{"section": "hook|intro|build|core|peak|outro|cta", "text": "...", "emotion": "one of {', '.join(EMOTIONS)}",
   "pace": "slow|normal|fast", "energy": 0.0-1.0, "pitch": "low|mid|high", "pause_after": 0.2-1.5, "emphasis": ["word"], "visual": "..."}}]}}
"""


def extract_json(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    raw = m.group(1) if m else text[text.find("{") : text.rfind("}") + 1]
    try:
        return json.loads(raw)
    except ValueError as e:
        raise AutocutError(f"the script came back in an unexpected form ({e})") from e


def ask_claude(prompt: str, model: str | None = None, timeout: int = 600) -> str:
    exe = shutil.which("claude") or next((p for p in ("/opt/homebrew/bin/claude", "/usr/local/bin/claude", str(Path.home() / ".local/bin/claude")) if Path(p).is_file()), None)
    if not exe:
        raise AutocutError("Claude Code is not installed (it is used to write scripts with Claude): https://claude.com/claude-code")
    cmd = [exe, "-p", "--output-format", "json", "--system-prompt",
           "You are an expert scriptwriter for YouTube and Instagram creators in India, fluent in English, Hindi and Hinglish. You only output the requested JSON."]
    if model:
        cmd += ["--model", model]
    with tempfile.TemporaryDirectory(prefix="ei-claude-") as tmp:  # no project context leaks in
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired as e:
            raise AutocutError("Claude took too long to answer") from e
    try:
        res = json.loads(proc.stdout)
    except ValueError as e:
        raise AutocutError(f"Claude Code did not answer ({(proc.stderr or proc.stdout).strip()[:200]}); is it logged in? Run `claude` once in Terminal.") from e
    if res.get("is_error"):
        raise AutocutError(f"Claude: {str(res.get('result'))[:200]}")
    return str(res.get("result", ""))


def ask_local(prompt: str, model: str | None) -> str:
    import llm

    outs = llm.ask([prompt], model, max_tokens=3000)
    if not outs:
        raise AutocutError("no local language model available")
    return outs[0]


def clean_segment(s: dict) -> dict | None:
    text = re.sub(r"[#*_~`\[\]{}<>]|\([^)]*\)", "", str(s.get("text", "")))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    emo = str(s.get("emotion", "neutral")).lower()
    try:
        energy = float(s.get("energy", 0.5))
    except (TypeError, ValueError):
        energy = 0.5
    try:
        pause = float(s.get("pause_after", 0.35))
    except (TypeError, ValueError):
        pause = 0.35
    return {
        "section": str(s.get("section", "core")),
        "text": text,
        "emotion": emo if emo in EMOTIONS else "neutral",
        "pace": s.get("pace") if s.get("pace") in ("slow", "normal", "fast") else "normal",
        "energy": min(1.0, max(0.0, energy)),
        "pitch": s.get("pitch") if s.get("pitch") in ("low", "mid", "high") else "mid",
        "pause_after": min(2.0, max(0.1, pause)),
        "emphasis": [str(w) for w in (s.get("emphasis") or [])][:4],
        "visual": str(s.get("visual", "")).strip(),
    }


def write_script(topic: str, out: str, *, niche: str = "", seconds: float = 60, language: str = "en", tone: str = "", audience: str = "",
                 notes: str = "", style: str | None = None, footage: list[str] | None = None, backend: str = "claude",
                 model: str | None = None, voice: str | None = None, speak_takes: int = 2) -> dict:
    if language not in LANGUAGES:
        raise AutocutError(f"language must be one of {', '.join(LANGUAGES)}")
    if not (5 <= seconds <= 1800):
        raise AutocutError("length must be 5–1800 seconds")
    prof = load_style(style)
    wpm = voice_wpm(voice, language)
    prompt = build_prompt(topic, niche=niche, seconds=seconds, language=language, tone=tone, audience=audience, notes=notes, style=prof, footage=footage, wpm=wpm)
    t0 = time.time()
    if backend == "claude":
        log("Claude is writing the script")
        raw = ask_claude(prompt, model)
    else:
        log("the local model is writing the script (Claude writes better Hindi; use it when online)")
        raw = ask_local(prompt, model)
    data = extract_json(raw)
    segs = [c for c in (clean_segment(s) for s in data.get("segments", []) if isinstance(s, dict)) if c]
    if not segs:
        raise AutocutError("the script has no segments")
    script = {"version": 1, "title": str(data.get("title", topic)).strip(), "topic": topic, "language": language, "niche": niche,
              "style": (prof or {}).get("name"), "target_seconds": seconds, "hook": str(data.get("hook", segs[0]["text"])),
              "segments": segs, "written_by": backend, "words": sum(len(words_of(s["text"])) for s in segs)}
    outp = Path(out)
    if outp.suffix.lower() in (".json", ".txt"):
        outp = outp.with_suffix("")
    outp.parent.mkdir(parents=True, exist_ok=True)
    jpath = outp.parent / f"{outp.name}.script.json"
    jpath.write_text(json.dumps(script, ensure_ascii=False, indent=1), encoding="utf-8")
    tpath = outp.parent / f"{outp.name}.txt"
    tpath.write_text(f"{script['title']}\n\n" + "\n".join(f"[{s['emotion']}] {s['text']}" for s in segs) + "\n", encoding="utf-8")
    log(f"script: {len(segs)} segments, {script['words']} words (aim {target_words(seconds, language, prof, wpm)}) in {time.time() - t0:.0f}s")
    res = {"outputs": [str(jpath), str(tpath)], "script": str(jpath), "out_dir": str(jpath.parent), "title": script["title"]}
    if voice:  # write and speak in one go
        import voice as vo

        sp = vo.speak(voice, str(jpath), str(outp.parent / f"{outp.name}.wav"), takes=speak_takes)
        res["outputs"] = sp["outputs"] + res["outputs"]
        res["narration"] = sp["narration"]
    return res


def add_args(sub, prefix: str = "") -> None:
    p = sub.add_parser(f"{prefix}style", help="learn a talking style from transcripts / YouTube links")
    p.add_argument("sources", nargs="+", help=".txt / .srt / .vtt files or YouTube URLs")
    p.add_argument("--name", required=True)
    p = sub.add_parser(f"{prefix}write", help="write a voiceover script")
    p.add_argument("--topic", required=True)
    p.add_argument("--out", required=True, help="output name (writes <out>.script.json and <out>.txt)")
    p.add_argument("--niche", default="")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--language", default="en", choices=list(LANGUAGES))
    p.add_argument("--tone", default="")
    p.add_argument("--audience", default="")
    p.add_argument("--notes", default="")
    p.add_argument("--style", help="style profile name or file")
    p.add_argument("--footage", nargs="*", help="clips/folders: their names tell the writer what visuals exist")
    p.add_argument("--backend", default="claude", choices=["claude", "local"])
    p.add_argument("--model", help="Claude model (e.g. sonnet, opus) or local model key")
    p.add_argument("--voice", help="also speak the script with this cloned voice (and size it to the voice's pace)")
    p.add_argument("--takes", type=int, default=2, help="takes per line when speaking")


def run_args(a) -> dict:
    if a.cmd.endswith("style"):
        srcs = load_sources(a.sources)
        if not srcs:
            raise AutocutError("no usable transcript text")
        prof = analyse_style(srcs)
        p = save_style(a.name, prof)
        log(f"style '{a.name}': {prof['words']} words from {len(srcs)} source(s), {prof['language']}, {prof['words_per_minute'] or '?'} wpm")
        return {"outputs": [str(p)], "style": str(p), "out_dir": str(p.parent), "profile": {k: prof[k] for k in ("language", "words", "words_per_minute", "avg_sentence_words")}}
    footage = None
    if a.footage:
        import voiceover_cut as vc

        footage = [Path(f).stem for f in vc.collect_clips(a.footage)]
    return write_script(a.topic, a.out, niche=a.niche, seconds=a.seconds, language=a.language, tone=a.tone, audience=a.audience, notes=a.notes,
                        style=a.style, footage=footage, backend=a.backend, model=a.model, voice=a.voice, speak_takes=a.takes)


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
    sys.exit(main())
