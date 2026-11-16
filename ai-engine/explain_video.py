#!/usr/bin/env python3
"""explain_video: the script of an explainer -> the voice, the drawn pictures in time with it, the
video, and every chapter as a clip of its own.

The voice is spoken once for the whole script (your cloned voice, or a macOS voice). Each line of
the script names the scene and step on screen while it is said, so the pictures change exactly
where the voice gets there. Frames are drawn here and piped to ffmpeg (hardware encoder).
Clips: each chapter cut out in the video's shape, and (optional) drawn again tall for Shorts,
which is a new layout of the same scenes, not a crop.
"""

from __future__ import annotations

import bisect
import json
import subprocess
from pathlib import Path

import autocut as ac
import explain_draw as xd
from autocut import AutocutError, log

FPS = 30
LEAD = 0.12  # a picture is there a moment before its line starts
SWEEP = 0.9  # the marker's stroke over a sentence (s)
ASPECTS = {"horizontal": (1920, 1080), "vertical": (1080, 1920), "square": (1080, 1080)}


def voice_script(script: dict) -> dict:
    """The script as the voice reads it (script_writer's form: one segment per line)."""
    segs = []
    for c, ch in enumerate(script["chapters"]):
        for k, ln in enumerate(ch["lines"]):
            last = k == len(ch["lines"]) - 1
            segs.append({"text": ln["text"], "emotion": ln.get("emotion", "Neutral"), "intensity": ln.get("intensity", 0.6), "pace": ln.get("pace", "normal"),
                         "emphasis": ln.get("emphasis") or [], "pause_after": max(0.9, float(ln.get("pause_after", 0.35))) if last else float(ln.get("pause_after", 0.35)),
                         "section": "hook" if k == 0 else "core", "visual": "", "chapter": c + 1})
    return {"title": script.get("title", ""), "language": script.get("language", "en"), "segments": segs}


def chunks(words: list[str], most: int, chars: int) -> list[list[str]]:
    out, cur = [], []
    for w in words:
        if cur and (len(cur) >= most or len(" ".join(cur)) + 1 + len(w) > chars):
            out.append(cur)
            cur = []
        cur.append(w)
        if w[-1:] in ".?!," and len(cur) >= max(2, most // 2):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def timeline(script: dict, segments: list[dict], *, mode: str = "sentence", tall: bool = False) -> tuple[list[dict], list[dict], list[dict]]:
    """(what is on screen from when, the captions, each chapter's start and end) from the script
    and the times its lines were spoken at."""
    lines = [(c, ln) for c, ch in enumerate(script["chapters"]) for ln in ch["lines"]]
    if len(lines) != len(segments):
        raise AutocutError(f"the narration has {len(segments)} lines and the script {len(lines)}: speak it again")
    states, cues, starts = [], [], {}
    most, chars = (3, 22) if mode == "words" else ((7, 34) if tall else (11, 58))
    for (c, ln), seg in zip(lines, segments):
        a, b = float(seg["start"]), float(seg["end"])
        t = max(0.0, a - LEAD, states[-1]["start"] if states else 0.0)
        starts.setdefault(c, 0.0 if c == 0 else max(0.0, a - 0.3))
        key = (c, int(ln.get("scene", 1)), int(ln.get("step", 0)), tuple(ln.get("focus") or ()))
        if not states or states[-1]["key"] != key:
            states.append({"start": t if states else 0.0, "key": key})
        text = str(ln.get("shown") or ln["text"]).replace("`", "")
        if mode != "none":
            groups = chunks(text.split(), most, chars)
            total = sum(len(w) + 1 for g in groups for w in g) or 1
            at = a
            for g in groups:
                ws, t0 = [], at
                for w in g:
                    ws.append(round(at, 3))
                    at += (b - a) * (len(w) + 1) / total
                cues.append({"start": round(t0, 3), "end": round(at, 3), "words": g, "at": ws})
    end = float(segments[-1]["end"]) + 0.6 if segments else 0.0
    order = sorted(starts)
    chapters = [{"start": starts[c], "end": starts[order[i + 1]] if i + 1 < len(order) else end, "chapter": c} for i, c in enumerate(order)]
    return states, cues, chapters


def _sweep(pre, img, marks: list, p: float):
    out = pre.copy()
    done = p * sum(max(0, m[2] - m[0]) for m in marks)
    for m in marks:
        if done <= 0:
            break
        take = int(min(m[2] - m[0], done))
        if take > 0:
            out.paste(img.crop((m[0], m[1], m[0] + take, m[3])), (m[0], m[1]))
        done -= m[2] - m[0]
    return out


def render(out: Path, script: dict, facts: list[dict], states: list[dict], cues: list[dict], chapters: list[dict], wav: str, *, W: int, H: int, look: str,
           captions: str | None = None, t0: float = 0.0, t1: float | None = None, music: str | None = None, codec: str = "h264", progress=None) -> None:
    """Frames t0..t1 of the video drawn and encoded with the voice (and music under it)."""
    from PIL import Image

    pen = xd.Pen(W, H, look, captions)
    t1 = chapters[-1]["end"] if t1 is None else t1
    n = max(1, int(round((t1 - t0) * FPS)))
    st_starts = [s["start"] for s in states]
    cue_starts = [c["start"] for c in cues]
    ch_starts = [c["start"] for c in chapters]
    drawn: dict[int, tuple] = {}

    def fact_of(sc: dict) -> dict | None:
        k = sc.get("fact")
        return facts[k - 1] if isinstance(k, int) and 1 <= k <= len(facts) else None

    def scene_of(k: int) -> tuple:
        c, s, step, focus = states[k]["key"]
        ch = script["chapters"][c]
        return ch, ch["scenes"][min(len(ch["scenes"]), max(1, s)) - 1], step, focus

    def scene_start(k: int) -> float:
        while k > 0 and states[k - 1]["key"][:2] == states[k]["key"][:2]:
            k -= 1
        return states[k]["start"]

    live = [xd.is_live(scene_of(k)[1]) for k in range(len(states))]

    def get(k: int) -> tuple:
        if k not in drawn:
            ch, sc, step, focus = scene_of(k)
            img, marks = xd.draw_scene(pen, sc, step, focus, title=ch["title"], fact=fact_of(sc))
            pre = xd.draw_scene(pen, sc, 1, focus, title=ch["title"], fact=fact_of(sc))[0] if marks else None
            for old in [o for o in drawn if o < k - 1]:
                del drawn[old]
            drawn[k] = (img, pre, marks)
        return drawn[k]

    fade, punch = pen.k["fade"], pen.k["punch"]
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-ss", f"{t0:.3f}", "-t", f"{n / FPS:.3f}", "-i", str(wav)]
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(music), "-filter_complex", ac.music_mix_filter("1:a", "2:a", "a", n / FPS, -24.0), "-map", "0:v", "-map", "[a]"]
    else:
        cmd += ["-map", "0:v", "-map", "1:a"]
    cmd += [*ac.video_encoder_args(codec, ac.bitrate_for(W, H, 8.0)), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-t", f"{n / FPS:.3f}", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    last_key, last_bytes, cap_cache = None, b"", {}
    try:
        for i in range(n):
            t = t0 + i / FPS
            k = max(0, bisect.bisect_right(st_starts, t) - 1)
            if live[k]:  # (a video plays, a photo is pushed in: drawn anew for this moment)
                ch, sc, step, focus = scene_of(k)
                img, pre, marks = xd.draw_scene(pen, sc, step, focus, title=ch["title"], fact=fact_of(sc), t=max(0.0, t - scene_start(k)))[0], None, []
            else:
                img, pre, marks = get(k)
            age = t - states[k]["start"]
            new_scene = k > 0 and states[k - 1]["key"][:2] != states[k]["key"][:2]
            moving = True
            if k > 0 and age < fade and (new_scene or not marks):
                base = Image.blend(get(k - 1)[0], pre or img, min(1.0, age / fade) ** 0.7)
            elif marks and age < (fade if new_scene else 0) + SWEEP:
                base = _sweep(pre, img, marks, max(0.0, age - (fade if new_scene else 0)) / SWEEP)
            else:
                base, moving = img, live[k]
            if punch and new_scene and age < 0.28:
                z = 1 + punch * (1 - age / 0.28) ** 2
                big = base.resize((int(W * z), int(H * z)), Image.BILINEAR)
                base = big.crop(((big.width - W) // 2, (big.height - H) // 2, (big.width - W) // 2 + W, (big.height - H) // 2 + H))
                moving = True
            ci = bisect.bisect_right(cue_starts, t) - 1
            cue = cues[ci] if ci >= 0 and t < cues[ci]["end"] else None
            wi = max(0, bisect.bisect_right(cue["at"], t) - 1) if cue else -1
            if pen.captions != "words":
                wi = 0 if cue else -1
            hi = max(0, bisect.bisect_right(ch_starts, t) - 1)
            span = max(0.1, chapters[hi]["end"] - chapters[hi]["start"])
            bar = int(W * min(1.0, max(0.0, (t - chapters[hi]["start"]) / span)) / 4) * 4
            key = (k, ci if cue else -1, wi, bar)
            if moving or key != last_key:
                frame = base.copy()
                if cue:
                    ck = (ci, wi)
                    if ck not in cap_cache:
                        if len(cap_cache) > 6:
                            cap_cache.clear()
                        cap_cache[ck] = xd.draw_caption(pen, cue["words"], wi)
                    cap, pos = cap_cache[ck]
                    if cap is not None:
                        frame.paste(cap, pos, cap)
                if bar:
                    frame.paste(pen.k["accent"], (0, 0, bar, pen.px(7)))
                last_bytes = frame.tobytes()
                last_key = None if moving else key
            proc.stdin.write(last_bytes)
            if progress and i % 90 == 0:
                progress(i / n)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode("utf-8", "replace")
    if proc.wait() != 0 or not out.is_file():
        raise AutocutError(f"the video could not be encoded: {err.strip()[-300:]}")


def clock(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def description(script: dict, facts: list[dict], chapters: list[dict]) -> str:
    out = [script.get("title", ""), "", script.get("promise", ""), "", "Chapters"]
    out += [f"{clock(c['start'])} {script['chapters'][c['chapter']]['title']}" for c in chapters]
    used = sorted({ln.get("fact") for ch in script["chapters"] for ln in ch["lines"] if ln.get("fact")} | {sc.get("fact") for ch in script["chapters"] for sc in ch["scenes"] if sc.get("fact")})
    shown = {}
    for ch in script["chapters"]:
        for sc in ch["scenes"]:
            for item in (sc if sc.get("type") == "footage" else None, sc.get("behind")):
                if isinstance(item, dict) and item.get("credit") and item.get("file"):
                    shown[item["file"]] = f"- {item['credit']}" + (f": {item['page']}" if item.get("page") else "")
    seen, src = set(), []
    for k in used:
        f = facts[k - 1] if isinstance(k, int) and 1 <= k <= len(facts) else None
        if f and f.get("url") and f["url"] not in seen:
            seen.add(f["url"])
            src.append(f"- {f.get('source', '')}: {f['url']}")
    return "\n".join(out + (["", "Sources"] + src if src else []) + (["", "Pictures and video"] + list(shown.values()) if shown else [])) + "\n"


def make_video(folder: Path, script: dict, facts: list[dict], *, voice: str = "mac", look: str = "clean", aspect: str = "horizontal", captions: str | None = None,
               music: str | None = None, clips: str = "same", takes: int = 2, speech_pace: str = "reference", only: list[int] | None = None) -> dict:
    """The voice, the video and the clips of a written script, in its folder. `clips`: none, same
    (each chapter cut out of the video) or vertical (each chapter drawn again tall, for Shorts).
    `only`: chapters whose clips to make (all when empty)."""
    import hashlib

    import script_writer as sw
    import voice as vo

    if aspect not in ASPECTS:
        raise AutocutError(f"the shape is one of {', '.join(ASPECTS)}")
    if look not in xd.LOOKS:
        raise AutocutError(f"the look is one of {', '.join(xd.LOOKS)}")
    if music and not Path(music).is_file():
        raise AutocutError(f"the music file is not there: {music}")
    W, H = ASPECTS[aspect]
    name = sw.safe(script.get("title", "explainer"))[:60] or "explainer"
    vs = voice_script(script)
    vpath = folder / "voice-script.json"
    wav = folder / f"{name}.wav"
    sig = hashlib.sha1(json.dumps([vs["segments"], voice, takes, speech_pace], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    sig_file = folder / ".voice-sig"
    ac.progress_plan({"voice": 0.5, "video": 0.35, "clips": 0.15})
    with ac.stage("voice"):
        if wav.is_file() and wav.with_suffix(".narration.json").is_file() and sig_file.is_file() and sig_file.read_text() == sig:
            log("reusing the narration (the script and the voice did not change)")
        else:
            vpath.write_text(json.dumps(vs, ensure_ascii=False, indent=1), encoding="utf-8")
            vo.speak(voice, str(vpath), str(wav), takes=takes, pace=speech_pace)
            sig_file.write_text(sig)
    nar = json.loads(wav.with_suffix(".narration.json").read_text(encoding="utf-8"))
    mode = xd.Pen(W, H, look, captions).captions
    states, cues, chapters = timeline(script, nar["segments"], mode=mode, tall=H > W)
    video = folder / f"{name}.mp4"
    outputs = [str(video)]
    with ac.stage("video"):
        log(f"drawing {chapters[-1]['end']:.0f}s of video ({W}x{H}, {look} look, {len(states)} pictures)")
        render(video, script, facts, states, cues, chapters, str(wav), W=W, H=H, look=look, captions=captions, music=music, progress=ac.stage_progress)
        if cues:
            srt = video.with_suffix(".srt")
            srt.write_text(ac.to_srt([{"start": c["start"], "end": c["end"], "text": " ".join(c["words"])} for c in cues]), encoding="utf-8")
            outputs.append(str(srt))
        desc = folder / "description.txt"
        desc.write_text(description(script, facts, chapters), encoding="utf-8")
        outputs.append(str(desc))
    made = []
    with ac.stage("clips"):
        if clips != "none":
            cdir = folder / "clips"
            cdir.mkdir(exist_ok=True)
            tall = clips == "vertical" and aspect != "vertical"
            todo = [c for c in chapters if not only or c["chapter"] + 1 in only]
            for i, c in enumerate(todo):
                ch = script["chapters"][c["chapter"]]
                dst = cdir / f"{c['chapter'] + 1:02d} {sw.safe(ch.get('clip_title') or ch['title'])[:70]}.mp4"
                if tall:
                    tmode = xd.Pen(1080, 1920, look, captions).captions
                    s2, q2, _ = timeline(script, nar["segments"], mode=tmode, tall=True)
                    render(dst, script, facts, s2, q2, chapters, str(wav), W=1080, H=1920, look=look, captions=captions, t0=c["start"], t1=c["end"], music=music)
                else:
                    ac.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-ss", f"{c['start']:.3f}", "-to", f"{c['end']:.3f}",
                            *ac.video_encoder_args("h264", ac.bitrate_for(W, H, 8.0)), "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(dst)])
                made.append(str(dst))
                long = c["end"] - c["start"] > 60 and tall
                log(f"clip {c['chapter'] + 1}: {dst.name} ({c['end'] - c['start']:.0f}s{', longer than a 60 s Short' if long else ''})")
                ac.stage_progress((i + 1) / len(todo))
    return {"outputs": outputs + made, "video": str(video), "clips": made, "seconds": round(chapters[-1]["end"], 1),
            "chapters": [{"title": script["chapters"][c["chapter"]]["title"], "start": round(c["start"], 2), "end": round(c["end"], 2)} for c in chapters]}
