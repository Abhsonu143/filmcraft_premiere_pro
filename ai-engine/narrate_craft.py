"""narrate_craft: what makes a voiceover edit feel edited (used by narrate.py).

  - The hook (the first 3 seconds decide if anyone watches):
      * cold open: the strongest line of the narration (its peak emotion, 1.5–4.5 s) plays first,
        then a white flash into the start, so the video opens on its best moment;
      * on-screen hook: a short line in big text over the first ~2.5 s, written for this video from
        the hook library's patterns (Claude, or the local model, or the first line itself);
      * a fast open: the first line is cut into quick shots of the best-looking, moving footage.
  - Pacing by emotion: tense and exciting lines get shorter shots, calm, sad and nostalgic lines
    longer ones (the 42 emotions of emotions.py).
  - Punch-ins on the words the script marks for emphasis and on peak lines.
  - Transitions by mood: a dissolve into a calm, sad, nostalgic or loving part; a white flash into
    an intense one; hard cuts everywhere else.
  - Slow motion on awe, wonder, triumph, sorrow and love lines when the footage has the frames
    (50/60 fps and up).
  - Framing for vertical and square: the crop follows the subject (people, animals, things that
    stand out) instead of the centre.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import autocut as ac
import emotions as em
from autocut import log

HOOK_MODES = ("auto", "teaser", "text", "both", "off")
SLOW_EMOTIONS = {"Awe", "Wonder", "Wonderment", "Triumph", "Sorrow", "Grief", "Nostalgia", "Love", "Romance", "Serenity", "Bliss", "Freedom", "Hope"}
CALM_FAMILIES = {"calm", "sorrow", "nostalgia", "love"}


def emotion_of(line: dict) -> tuple[str, float, dict]:
    name = em.resolve(line.get("emotion"))
    try:
        inten = float(line.get("intensity", 0.6))
    except (TypeError, ValueError):
        inten = 0.6
    return name, float(np.clip(inten, 0, 1)), em.EMOTIONS[name]


def pace_factor(line: dict) -> float:
    """How long this line's shots may be, relative to the pace setting: 0.6 (intense) … 1.35 (calm)."""
    _, inten, e = emotion_of(line)
    return float(np.clip(1.35 - 0.75 * e["arousal"] * (0.6 + 0.4 * inten), 0.6, 1.35))


# ---------------------------------------------------------------------------------------------
# The hook
# ---------------------------------------------------------------------------------------------


def teaser_pick(lines: list[dict], lo: float = 1.5, hi: float = 4.5) -> int | None:
    """The line to open with: the most intense, energetic one after the opening (not the first
    two lines, which play right after it anyway), short enough for a teaser."""
    best, best_s = None, 0.0
    for i, ln in enumerate(lines):
        d = float(ln["end"]) - float(ln["start"])
        if i < 2 or not (lo <= d <= hi):
            continue
        _, inten, e = emotion_of(ln)
        s = inten * (0.4 + e["arousal"]) + (0.3 if em.resolve(ln.get("emotion")) in ("Surprise", "Amazement", "Triumph", "Awe", "Thrill") else 0.0)
        if s > best_s:
            best, best_s = i, s
    return best if best_s >= 0.55 else None


def with_teaser(voice: str, lines: list[dict], i: int, dst: Path, gap: float = 0.35) -> tuple[list[dict], float]:
    """A new voice track: line i first (with tiny fades), a short gap, then the whole narration.
    Returns the lines on the new timeline (the teaser line marked) and the time added."""
    ln = lines[i]
    a, b = max(0.0, float(ln["start"]) - 0.06), float(ln["end"]) + 0.12
    head = b - a + gap
    graph = (f"[0:a]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS,afade=t=in:d=0.04,afade=t=out:st={max(0.0, b - a - 0.08):.3f}:d=0.08,"
             f"apad=pad_dur={gap:.3f}[t];[0:a]asetpts=PTS-STARTPTS[v];[t][v]concat=n=2:v=0:a=1,aresample=48000[o]")
    ac.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", voice, "-filter_complex", graph, "-map", "[o]", "-c:a", "pcm_s16le", str(dst)])

    def shift(x: dict, by: float) -> dict:
        y = {**x, "start": round(float(x["start"]) + by, 3), "end": round(float(x["end"]) + by, 3)}
        if x.get("words"):
            y["words"] = [{**w, "start": w["start"] + by, "end": w["end"] + by} for w in x["words"]]
        return y

    teaser = shift(ln, -a)  # the clip starts at a, so the line starts 0.06 s in
    teaser["teaser"] = True
    teaser["section"] = "teaser"
    return [teaser] + [shift(x, head) for x in lines], head


def screen_hook(lines: list[dict], niche: str, language: str, how: str = "claude", model: str | None = None) -> str:
    """A short on-screen hook (3–7 words) for this video, in its language."""
    import hooks as hk

    text = " ".join(str(ln.get("text", "")) for ln in lines)[:1500]
    s = hk.suggest(text[:300], niche, language, n=6)
    first = str(lines[0].get("text", "")).strip() if lines else ""
    if how in ("claude", "local"):
        pats = "\n".join(f"- [{h['pattern']}] {h['text']}" for h in s["hooks"])
        prompt = (f"A short video's narration:\n{text}\n\nWrite the on-screen hook text for its first 2 seconds: 3 to 7 words, in the "
                  f"narration's language ({language}), that makes a scrolling viewer stop, true to the video (no clickbait lie). "
                  f"Hook styles that work now:\n{pats}\n\nReply with only the hook text.")
        try:
            if how == "claude":
                import script_writer as sw

                out = sw.ask_claude(prompt, model, timeout=180)
            else:
                import llm

                out = (llm.ask([prompt], model, max_tokens=40) or [""])[0]
            out = re.sub(r"[\"“”#*]|^hook:\s*", "", str(out).strip().splitlines()[0] if str(out).strip() else "", flags=re.I).strip()
            if 2 <= len(out.split()) <= 10:
                return out
        except Exception as e:  # no Claude / model: the first line still makes a hook
            log(f"on-screen hook from the first line ({type(e).__name__})")
    w = first.split()
    return " ".join(w[:7]) + ("…" if len(w) > 7 else "")


def fast_open(slots: list[dict], hook_len: float = 3.0, cut: float = 1.4) -> list[dict]:
    """The first ~3 s cut quicker: slots there longer than 2 s become shots of about `cut` s."""
    out = []
    for s in slots:
        d = s["end"] - s["start"]
        if s["start"] < hook_len and d > 2.0:
            n = int(min(3, max(2, round(d / cut))))
            pts = [s["start"] + d * k / n for k in range(n + 1)]
            out += [{**s, "start": a, "end": b, "part": s["part"] * 10 + k, "hook": True} for k, (a, b) in enumerate(zip(pts, pts[1:]))]
        else:
            out.append({**s, "hook": s["start"] < hook_len})
    return out


# ---------------------------------------------------------------------------------------------
# Shot treatment
# ---------------------------------------------------------------------------------------------


def emphasis_times(line: dict) -> list[float]:
    """When the emphasised words are said (word timings, or estimated by their place in the text)."""
    marks = [str(w).lower().strip(".,!?") for w in (line.get("emphasis") or []) if str(w).strip()]
    _, inten, e = emotion_of(line)
    if not marks and inten >= 0.85 and e["arousal"] >= 0.7:
        return [float(line["start"]) + 0.15]  # a peak line: punch in on its first word
    if not marks:
        return []
    out = []
    if line.get("words"):
        for w in line["words"]:
            if re.sub(r"[^\w]", "", str(w["word"]).lower()) in {re.sub(r"[^\w]", "", m) for m in marks}:
                out.append(float(w["start"]))
        return out
    text = str(line.get("text", ""))
    span = float(line["end"]) - float(line["start"])
    for m in marks:
        k = text.lower().find(m)
        if k >= 0 and len(text):
            out.append(float(line["start"]) + span * k / len(text))
    return sorted(out)


def treat(shots: list[dict], slots: list[dict], lines: list[dict], clips_by_path: dict[str, dict], fps: float) -> dict:
    """Punch-ins, mood transitions and slow motion on the planned shots (in place). Returns counts."""
    t = 0.0
    starts = []
    for s in shots:
        starts.append(t)
        t += s["frames"] / fps
    n_punch = n_slow = n_trans = 0
    for k, (sh, sl) in enumerate(zip(shots, slots)):
        ln = lines[sl["line"]]
        name, inten, e = emotion_of(ln)
        c = clips_by_path.get(sh["clip"], {})
        still = bool(c.get("still") or c.get("card"))
        d = sh["frames"] / fps
        # punch-ins on emphasised words inside this shot (one per shot)
        if not still:
            for tw in emphasis_times(ln):
                rel = tw - starts[k]
                if 0.15 <= rel <= d - 0.3:
                    sh["punches"] = [round(rel, 3)]
                    sh.pop("zoom", None)
                    n_punch += 1
                    break
        # slow motion where the footage has the frames and the moment wants it
        if (not still and name in SLOW_EMOTIONS and inten >= 0.6 and float(c.get("fps", 30.0)) >= 47 and d >= 1.2 and not sh.get("punches")
                and not sl.get("hook")):
            sh["speed"] = 0.5
            n_slow += 1
        # transitions at a change of mood
        if k > 0 and sl.get("part", 0) == 0:
            prev = lines[slots[k - 1]["line"]]
            pf = emotion_of(prev)[2]["family"]
            if prev.get("teaser") and not ln.get("teaser"):
                sh["transition_in"] = {"type": "fadewhite", "frames": int(round(0.3 * fps))}  # out of the cold open
                n_trans += 1
            elif sh.get("transition_in"):
                pass  # (a pause in the narration already dissolves)
            elif e["family"] != pf and e["family"] in CALM_FAMILIES:
                sh["transition_in"] = {"type": "dissolve", "frames": int(round(0.5 * fps))}
                n_trans += 1
            elif e["family"] != pf and e["arousal"] >= 0.75 and inten >= 0.7:
                sh["transition_in"] = {"type": "fadewhite", "frames": int(round(0.15 * fps))}
                n_trans += 1
    return {"punches": n_punch, "slow": n_slow, "transitions": n_trans}


def frame_for_shape(shots: list[dict], W: int, H: int, fps: float, clips_by_path: dict[str, dict]) -> int:
    """Crop paths that follow the subject when the footage's shape differs from the output's
    (vertical/square from horizontal footage). Returns the number of shots framed."""
    import reframe

    by_clip: dict[str, list[int]] = {}
    for k, s in enumerate(shots):
        c = clips_by_path.get(s["clip"], {})
        if c.get("still") or c.get("card"):
            continue
        by_clip.setdefault(s["clip"], []).append(k)
    n = 0
    for path, ks in by_clip.items():
        info = ac.probe(path)
        dw, dh = info.display_size
        if abs(W / H - dw / max(dh, 1)) < 0.03:
            continue
        ranges = [(float(shots[k]["in"]), float(shots[k]["in"]) + shots[k]["frames"] / fps * float(shots[k].get("speed", 1.0))) for k in ks]
        try:
            frames = reframe.reframe_pieces(path, ranges, dw, dh, W, H, talk=False)
        except Exception as e:  # framing is a nicety: a failure keeps the centred crop
            log(f"framing skipped for {Path(path).name} ({type(e).__name__})")
            continue
        for k, fr in zip(ks, frames):
            track, active = reframe.track_for_shot(fr, float(shots[k]["in"]), float(shots[k].get("speed", 1.0)), shots[k]["frames"] / fps)
            if track:
                shots[k]["track"] = track
                n += 1
            if active:
                shots[k]["active"] = active
    return n


def hook_ass(ass: str, text: str, W: int, H: int, seconds: float = 2.8) -> str:
    """Adds the on-screen hook to an ASS file: big bold words on a dark box in the upper third,
    popping in, gone after ~2.8 s."""
    import captions as cap

    devanagari = bool(re.search(r"[ऀ-ॿ]", text))
    font = "Kohinoor Devanagari" if devanagari else "Avenir Next Heavy"
    fs = int(min(W, H) * (0.095 if H > W else 0.07))
    style = (f"Style: Hook,{font},{fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H99000000,1,0,0,0,100,100,0,0,3,{max(10, fs // 4)},0,8,"
             f"{W // 9},{W // 9},{int(H * (0.2 if H > W else 0.12))},1")
    ev = (f"Dialogue: 2,{cap.ass_time(0.05)},{cap.ass_time(seconds)},Hook,,0,0,0,,"
          f"{{\\fad(60,260)\\fscx70\\fscy70\\t(0,170,\\fscx104\\fscy104)\\t(170,260,\\fscx100\\fscy100)}}{cap.esc(text)}")
    ev = ev.replace("\\\\", "\\")
    out = re.sub(r"(\[V4\+ Styles\]\nFormat:[^\n]*\n)", lambda m: m.group(1) + style + "\n", ass, count=1)
    return out.rstrip("\n") + "\n" + ev + "\n"
