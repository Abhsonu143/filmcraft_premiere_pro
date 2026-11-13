"""sections: cut the best continuous parts (15 s – 3 min) out of a long video, uncut inside.

Unlike `clip_edit` (short clips edited from many pieces), a section plays one uninterrupted part of
the video. The intelligence is in where it starts and ends, the way an editor would choose:

  1. Every moment of the video gets an interest value (loudness, sound onsets, motion, speech;
     dead air, pause screens and black strongly negative), the same measurements the short clips
     use, so a video analysed once is analysed for both.
  2. Possible starts: a picture cut (a new shot begins), the first word of a sentence after a
     pause (a new topic is better), or the calm just before the action builds. Possible ends: a
     picture cut, a fade to black, the end of a sentence followed by a pause, or the moment the
     action has settled after a peak. Never inside a spoken sentence, never on black.
  3. Each start/end pair of the allowed length is valued: how much happens inside (no dead
     stretches), how good both edges are, and how close it is to the usual length. The best
     non-overlapping sections win.
  4. Edges are made exact: a start or end on a picture cut is moved onto the exact frame of the
     cut (the analysis runs at 5 frames per second), so no frame of the neighbouring shot shows.
     A section that starts or ends inside a shot gets short fades; one that starts on a cut does
     not.

Rendering keeps the section continuous: the picture is split into shots only where the video
itself cuts (and every ~20 s, so crop paths stay small), all back to back, and the sound is one
uninterrupted piece.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

import autocut as ac
import clip_edit as ce
from autocut import log

HZ = ce.HZ
SHOT_MAX = 20.0  # seconds per rendered shot (crop paths stay small); the cut is invisible
LEAD_IN = 0.25  # seconds before the first word of a section
TAIL = 0.45  # seconds after the last word


# ---------------------------------------------------------------------------------------------
# Edge candidates (pure logic, unit-tested)
# ---------------------------------------------------------------------------------------------


def inside_speech(t: float, sents: list[dict], margin: float = 0.05) -> bool:
    return any(s["start"] - margin < t < s["end"] + margin for s in sents)


def dedupe(cands: list[tuple[float, float, str]], gap: float = 0.6) -> list[tuple[float, float, str]]:
    """Of candidates closer than `gap`, keep the best one."""
    out: list[tuple[float, float, str]] = []
    for c in sorted(cands, key=lambda c: -c[1]):
        if all(abs(c[0] - o[0]) >= gap for o in out):
            out.append(c)
    return sorted(out)


def edge_candidates(score: np.ndarray, sig: dict[str, np.ndarray], sents: list[dict], topics: set[int], duration: float
                    ) -> tuple[list[tuple[float, float, str]], list[tuple[float, float, str]]]:
    """(starts, ends): lists of (time, quality 0–1.5, why)."""
    n = len(score)
    f = ce.smooth(score, 30)  # 3 s
    br = sig["bright"]
    starts: list[tuple[float, float, str]] = [(0.0, 0.5, "start of the video")]
    ends: list[tuple[float, float, str]] = [(duration, 0.5, "end of the video")]

    def dark(t: float) -> bool:
        i = int(t * HZ)
        return 0 <= i < n and br[i] < 18

    # picture cuts
    for i in np.nonzero(sig["cut"] > 0)[0]:
        t = float(i) / HZ
        if not inside_speech(t, sents):
            starts.append((t, 1.0, "cut"))
            ends.append((t, 1.0, "cut"))
        else:  # a cut during a line: fine to end on after the line, not to start on
            pass
    # fades to / from black
    dk = br < 18
    for i in range(1, n):
        if inside_speech(i / HZ, sents):
            continue
        if dk[i] and not dk[i - 1]:
            ends.append((i / HZ, 1.2, "fade to black"))
        elif dk[i - 1] and not dk[i]:
            starts.append((i / HZ, 1.0, "fade in from black"))
    # sentences: start before the first word, end after the last
    for k, s in enumerate(sents):
        before = s["start"] - (sents[k - 1]["end"] if k else 0.0)
        after = (sents[k + 1]["start"] if k + 1 < len(sents) else duration) - s["end"]
        if before >= 0.35:
            q = 0.45 + 0.5 * min(1.0, (before - 0.35) / 1.2) + (0.45 if k in topics else 0.0)
            starts.append((max(0.0, s["start"] - min(LEAD_IN, before / 2)), q, "topic" if k in topics else "sentence"))
        if after >= 0.35 and s["text"].rstrip()[-1:] in ".?!":
            q = 0.45 + 0.5 * min(1.0, (after - 0.35) / 1.2) + (0.45 if k + 1 in topics else 0.0)
            ends.append((min(duration, s["end"] + min(TAIL, after / 2)), q, "end of a topic" if k + 1 in topics else "end of a sentence"))
    # the calm before the action builds / after it settles (3 s smoothed interest)
    w = 8 * HZ
    for i in range(w, n - w, HZ // 2):
        here = f[i]
        if here > f[max(0, i - HZ) : i + HZ + 1].min() + 1e-9:
            continue  # not a local minimum
        rise = float(f[i : i + w].max() - here)
        fall = float(f[i - w : i].max() - here)
        t = i / HZ
        if inside_speech(t, sents):
            continue
        if rise > 0.6:
            starts.append((t, min(1.1, 0.4 + 0.5 * rise), "before the action"))
        if fall > 0.6:
            ends.append((t, min(1.1, 0.4 + 0.5 * fall), "action settled"))
    starts = [c for c in starts if not dark(c[0] + 0.05)]
    return dedupe(starts), dedupe(ends)


# ---------------------------------------------------------------------------------------------
# Choosing sections (pure logic, unit-tested)
# ---------------------------------------------------------------------------------------------


def dead_mask(sig: dict[str, np.ndarray], score: np.ndarray, words: list[dict]) -> np.ndarray:
    """Dead moments: black, or a still picture with nothing said (menus, pause screens, loading)."""
    mot, br = sig["motion"], sig["bright"]
    still = ce.smooth(mot, 20) < max(0.3, float(np.percentile(mot, 10)))
    quiet = ce.speech_mask(words, len(score)) < 0.5
    return (br < 18) | (still & quiet)


def plan_sections(score: np.ndarray, dead: np.ndarray, starts: list[tuple[float, float, str]], ends: list[tuple[float, float, str]],
                  count: int, mn: float, mx: float, target: float, gap: float = 3.0) -> list[dict]:
    """The best `count` non-overlapping sections [{start, end, value, why_in, why_out}], in time order."""
    n = len(score)
    if n == 0 or not starts or not ends:
        return []
    cs = np.concatenate([[0.0], np.cumsum(score)])
    cp = np.concatenate([[0.0], np.cumsum(np.maximum(score, 0.0))])
    cd = np.concatenate([[0.0], np.cumsum(dead.astype(np.float64))])
    w = 3 * HZ  # interest in the 3 s just after / before each moment
    idx = np.arange(n + 1)
    after = (cs[np.minimum(n, idx + w)] - cs[idx]) / np.maximum(1, np.minimum(n, idx + w) - idx)
    before = (cs[idx] - cs[np.maximum(0, idx - w)]) / np.maximum(1, idx - np.maximum(0, idx - w))
    usual = float(np.percentile(score, 60))  # ordinary moments of this video
    et = np.asarray([e[0] for e in ends])
    eq = np.asarray([e[1] for e in ends])
    rows = []
    for s_t, s_q, s_why in starts:
        lo, hi = np.searchsorted(et, s_t + mn - 1e-6), np.searchsorted(et, s_t + mx + 1e-6)
        if hi <= lo:
            continue
        b = et[lo:hi]
        i0 = min(n, int(round(s_t * HZ)))
        i1 = np.minimum(n, np.round(b * HZ).astype(int))
        cnt = np.maximum(1, i1 - i0)
        mean = (cs[i1] - cs[i0]) / cnt
        pos = (cp[i1] - cp[i0]) / cnt
        dead_f = (cd[i1] - cd[i0]) / cnt
        # the first and last 15% matter most: a section must not open or close on nothing
        L = b - s_t
        head = np.minimum(n, i0 + np.maximum(1, (0.15 * L * HZ).astype(int)))
        tail = np.maximum(i0, i1 - np.maximum(1, (0.15 * L * HZ).astype(int)))
        open_dead = (cd[head] - cd[i0]) / np.maximum(1, head - i0)
        close_dead = (cd[i1] - cd[tail]) / np.maximum(1, i1 - tail)
        # length: anything within ±35% of the usual length is fine; beyond that it costs a little.
        # A longer section is worth it while the action keeps going (substance), not when it fades.
        rel = (L - target) / max(target, 1.0)
        off = np.maximum(0.0, np.abs(rel) - 0.35)
        closeness = 1.0 - np.minimum(1.0, off * np.where(rel < 0, 2.0, 0.6))
        substance = np.log1p((cp[i1] - cp[i0]) / (max(target, 1.0) * HZ))
        # never stop in the middle of the action: what happens just outside the edges must be less
        through = np.maximum(0.0, after[np.minimum(i1, n)] - usual) + max(0.0, float(before[i0]) - usual)
        value = (mean + 0.6 * pos + 0.5 * substance - 2.2 * dead_f - 0.8 * (open_dead + close_dead) - 0.7 * through
                 + 0.45 * (s_q + eq[lo:hi]) + 0.7 * closeness)
        for j in range(len(b)):
            rows.append((float(value[j]), s_t, float(b[j]), s_why, ends[lo + j][2]))
    rows.sort(key=lambda r: -r[0])
    chosen: list[dict] = []
    for v, a, b, wi, wo in rows:
        if len(chosen) >= count:
            break
        if any(a < c["end"] + gap and b > c["start"] - gap for c in chosen):
            continue
        chosen.append({"start": round(a, 3), "end": round(b, 3), "value": round(v, 3), "why_in": wi, "why_out": wo})
    return sorted(chosen, key=lambda c: c["start"])


# ---------------------------------------------------------------------------------------------
# Exact edges and shots
# ---------------------------------------------------------------------------------------------


def trim_held(sec: dict, sig: dict[str, np.ndarray], mn: float, still: float = 1.0, longest: float = 4.0, moving: int = 6,
              speech: np.ndarray | None = None) -> dict:
    """Don't open or close on a held picture (a loading card, a freeze before a wipe, a pause
    screen): the edge moves to where the picture really moves (at least 0.6 s of motion; a card
    animating in for a moment doesn't count), keeping a 0.2 s beat of the hold. Someone talking
    is never a held picture (a webcam interview hardly moves)."""
    mot = np.asarray(sig["motion"], dtype=np.float64)
    n = len(mot)
    if speech is not None and len(speech):
        mot = np.where(np.asarray(speech[:n]) > 0.5, max(still, 1.0) * 10, mot) if len(speech) >= n else mot
    out = dict(sec)
    i, j = max(0, int(round(sec["start"] * HZ))), min(n, int(round(sec["end"] * HZ)))
    run, k = 0, None
    for x in range(j - 1, max(i, j - int(longest * HZ)) - 1, -1):
        run = run + 1 if mot[x] >= still else 0
        if run >= moving:
            k = x + moving  # the last moving sample + 1
            break
    if k is not None and j - k >= 4 and (k / HZ + 0.2) - sec["start"] >= mn:
        out["end"], out["why_out"] = round(k / HZ + 0.2, 3), "held picture"
    run, k = 0, None
    for x in range(i, min(j, i + int(longest * HZ), n)):
        run = run + 1 if mot[x] >= still else 0
        if run >= moving:
            k = x - moving + 1  # the first moving sample
            break
    if k is not None and k - i >= 4 and out["end"] - (k / HZ - 0.1) >= mn:
        out["start"], out["why_in"] = round(k / HZ - 0.1, 3), "after a held picture"
    return out


def frame_diffs(src: str, t: float, span: float = 0.6) -> tuple[float, np.ndarray]:
    """Mean absolute difference between consecutive frames around t (every source frame, tiny)."""
    a = max(0.0, t - span)
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", f"{a:.3f}", "-t", f"{2 * span:.3f}", "-i", src,
           "-an", "-vf", "scale=64:36,format=gray", "-f", "rawvideo", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=60).stdout
    except (subprocess.TimeoutExpired, OSError):
        return a, np.zeros(0)
    fr = np.frombuffer(raw, np.uint8)
    k = len(fr) // (64 * 36)
    if k < 3:
        return a, np.zeros(0)
    fr = fr[: k * 64 * 36].reshape(k, 36, 64).astype(np.float32)
    return a, np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))


def exact_cut(src: str, t: float, src_fps: float) -> float | None:
    """The exact time of the picture cut near t (the first frame of the new shot), or None."""
    a, d = frame_diffs(src, t)
    if len(d) < 2:
        return None
    j = int(np.argmax(d))
    rest = np.delete(d, j)
    if d[j] < 12.0 or d[j] < 3.0 * (float(np.median(rest)) + 1.0):
        return None
    return a + (j + 1) / max(1.0, src_fps)


def sharpen_edges(src: str, sec: dict, src_fps: float, fps: float) -> dict:
    """Start/end on the exact frame of a cut; fades when an edge is inside a shot."""
    half = 0.5 / fps
    out = dict(sec)
    out["fade_in"], out["fade_out"] = 0.25, 0.4
    if sec["why_in"] == "cut":
        c = exact_cut(src, sec["start"], src_fps)
        if c is not None:  # a real cut: start hard on its first frame
            out["start"], out["fade_in"] = round(c + half, 4), 0.0
    if sec["why_out"] == "cut":
        c = exact_cut(src, sec["end"], src_fps)
        if c is not None:
            out["end"], out["fade_out"] = round(c - half, 4), 0.0
    elif sec["why_out"] == "fade to black":
        out["fade_out"] = 0.0  # the video fades out itself
    return out


def split_shots(a: float, b: float, cuts: list[float], fps: float, shot_max: float = SHOT_MAX) -> list[dict]:
    """Back-to-back shots covering [a, b) exactly: split at the video's own cuts, and so that no
    shot is longer than `shot_max`. Frame counts add up to the section's length exactly."""
    total = max(1, int(round((b - a) * fps)))
    marks = sorted({int(round((c - a) * fps)) for c in cuts if a + 1.0 < c < b - 1.0})
    bounds = [0]
    for m in marks + [total]:
        while m - bounds[-1] > shot_max * fps:
            bounds.append(bounds[-1] + int(np.ceil((m - bounds[-1]) / np.ceil((m - bounds[-1]) / (shot_max * fps)))))
        if m - bounds[-1] >= 0.5 * fps or m == total:
            bounds.append(m)
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < 0.5 * fps:
        bounds.pop(-2)
    return [{"in": round(a + f0 / fps, 4), "frames": f1 - f0, "speed": 1.0, "role": "section"} for f0, f1 in zip(bounds, bounds[1:]) if f1 > f0]


def section_audio(src: str, a: float, b: float, dst: Path, fade_in: float, fade_out: float, has_audio: bool) -> None:
    """The section's sound, one uninterrupted piece (loudness for social platforms, -14 LUFS)."""
    L = b - a
    if not has_audio:
        ac.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                "-t", f"{L:.4f}", "-c:a", "pcm_s16le", str(dst)])
        return
    fi, fo = max(0.02, fade_in), max(0.03, fade_out)
    chain = (f"aresample=48000,asetpts=PTS-STARTPTS,atrim=0:{L:.4f},apad,atrim=0:{L:.4f},afade=t=in:d={fi:.3f},"
             f"afade=t=out:st={max(0.0, L - fo):.4f}:d={fo:.3f},loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000,atrim=0:{L:.4f}")
    ac.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{a:.4f}", "-t", f"{L + 0.3:.4f}", "-i", src,
            "-vn", "-af", chain, "-c:a", "pcm_s16le", str(dst)])


def frame_section(src: str, sec: dict, shots: list[dict], src_w: int, src_h: int, W: int, H: int, fps: float, framing: str, talk: bool) -> str:
    """Framing for a section's shots, from one crop path over the whole section (no jump where
    the section is split into shots for rendering)."""
    import highlights as hl
    import reframe

    same_shape = abs(W / H - src_w / max(src_h, 1)) < 0.03
    mode = "smart" if framing == "auto" else framing
    if mode in ("blur", "bars", "center") or same_shape:
        return hl.frame_shots(src, shots, src_w, src_h, W, H, fps, framing, talk)
    fr = reframe.reframe_pieces(src, [(sec["start"], sec["end"])], src_w, src_h, W, H, talk)
    if not fr:
        return hl.frame_shots(src, shots, src_w, src_h, W, H, fps, "center", talk)
    for s in shots:
        track, active = reframe.track_for_shot(fr[0], float(s["in"]), 1.0, s["frames"] / fps)
        if track:
            s["track"] = track
        if active:
            s["active"] = active
    return "smart"


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def find_sections(src: str, info, words: list[dict], count: int, mn: float, mx: float, target: float, content: str) -> tuple[list[dict], str, dict]:
    """Sections for `src` plus the content mode and the signals used."""
    import highlights as hl

    sig = ce.signals(src, info.duration)
    n = len(sig["loud"])
    speech = ce.speech_mask(words, n)
    score = ce.interest(sig, speech)
    sents = hl.sentences_from_words(words) if words else []
    talk_share = float(speech.mean()) if n else 0.0
    mode = content if content in ("talk", "gameplay") else ("talk" if talk_share >= 0.4 else "gameplay")
    topics: set[int] = set()
    if sents:
        loud = hl.loudness_per_sentence(src, sents) if mode == "talk" else [0.0] * len(sents)
        for s, l in zip(sents, loud):
            s["score"] = hl.sentence_score(s, l)
        if mode == "talk":
            topics = set(hl.topic_boundaries(sents))
            # in talk, what is said matters most: each sentence's value over its time
            sv = np.zeros(n)
            for s in sents:
                sv[int(s["start"] * HZ) : int(np.ceil(s["end"] * HZ))] = s["score"]
            score = 0.5 * score + 1.2 * ce.zs(ce.smooth(sv, 30))
    dead = dead_mask(sig, score, words)
    starts, ends = edge_candidates(score, sig, sents, topics, info.duration)
    log(f"{len(starts)} possible starts and {len(ends)} possible ends ({mode} mode)")
    secs = plan_sections(score, dead, starts, ends, count, mn, mx, target)
    return secs, mode, {"sig": sig, "score": score, "sents": sents}


def make_sections(src: str, out_dir: str, count: int = 3, mn: float = 15, mx: float = 180, target: float = 60, aspect: str = "same",
                  captions: str = "off", title: str = "off", model: str | None = None, language: str | None = None,
                  content: str = "auto", framing: str = "auto", whisper_model: str = ac.DEFAULT_WHISPER, plan_only: bool = False,
                  caption_style: str = "clean", workers: int = 2, sound: dict | None = None) -> dict:
    import json
    import time

    import captions as cap
    import highlights as hl

    ac.require_tools()
    src = str(Path(src).resolve())
    info = ac.probe(src)
    if not info.has_video:
        raise ac.AutocutError("the file has no video")
    if not (1 <= count <= 50):
        raise ac.AutocutError("number of sections must be 1–50")
    if not (5 <= mn <= mx <= 600):
        raise ac.AutocutError("section lengths must satisfy 5 ≤ min ≤ max ≤ 600 seconds")
    framing = {"crop": "smart", "fit": "blur"}.get(framing, framing)
    if framing not in hl.FRAMINGS:
        raise ac.AutocutError(f"unknown framing {framing!r} ({', '.join(hl.FRAMINGS)})")
    title = {"on": "always", "true": "always", "false": "off"}.get(str(title).lower(), str(title).lower())
    if title not in cap.TITLE_MODES:
        raise ac.AutocutError(f"unknown title mode {title!r} ({', '.join(cap.TITLE_MODES)})")
    target = float(np.clip(target, mn, mx))
    if info.duration < mn:
        raise ac.AutocutError(f"the video ({info.duration:.0f}s) is shorter than the shortest section")
    dw, dh = info.display_size
    W, H = hl.output_size(aspect, dw, dh)
    fps = 30.0
    t0 = time.time()
    words: list[dict] = ac.transcribe(src, model=whisper_model, language=language)["words"] if info.has_audio else []
    secs, mode, an = find_sections(src, info, words, count, mn, mx, target, content)
    src_fps = info.fps or 30.0
    cuts = [float(i) / HZ for i in np.nonzero(an["sig"]["cut"] > 0)[0]]
    clips = []
    for s in secs:
        s = sharpen_edges(src, trim_held(s, an["sig"], mn, speech=ce.speech_mask(words, len(an["score"]))), src_fps, fps)
        shots = split_shots(s["start"], s["end"], cuts, fps)
        if not shots:
            continue
        text = " ".join(w["word"].strip() for w in words if s["start"] <= w["start"] < s["end"])
        c = {**s, "shots": shots, "pieces": [(sh["in"], sh["in"] + sh["frames"] / fps) for sh in shots], "text": text, "title": None}
        if mode == "talk":
            inside = [i for i, x in enumerate(an["sents"]) if s["start"] <= x["start"] < s["end"]]
            c["title"] = hl.fallback_title(an["sents"], inside) if inside else None
        log(f"section {s['start']:.1f}–{s['end']:.1f}s ({s['end'] - s['start']:.0f}s): starts at {s['why_in']}, ends at {s['why_out']}")
        clips.append(c)
    if len(clips) < count:
        log(f"only {len(clips)} section(s) of {mn:g}–{mx:g}s fit without overlapping")
    hl.name_clips(src, clips, mode, model)
    used_framing = framing
    for c in clips:
        used_framing = frame_section(src, c, c["shots"], dw, dh, W, H, fps, framing, mode == "talk")
        c["framing"] = used_framing
        c["duration"] = round(sum(s["frames"] for s in c["shots"]) / fps, 2)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    plan = {"version": 1, "mode": "sections", "content": mode, "input": src, "width": W, "height": H, "fps": fps, "count": count,
            "min": mn, "max": mx, "target": target, "captions": captions, "caption_style": caption_style, "title": title,
            "framing": used_framing, "clips": clips}
    plan_path = out / f"{stem}.sections.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    log(f"planned {len(clips)} sections in {time.time() - t0:.0f}s ({used_framing} framing): " + ", ".join(f"{c['duration']:.0f}s" for c in clips))
    outputs = []
    if not plan_only:
        clean = cap.clean_words([w for w in words if hl.norm_word(w["word"]) not in hl.FILLERS]) if words else []
        for k, c in enumerate(clips, 1):
            name = f"{stem} - section {k:02d}" + (f" - {hl.safe_name(c['title'])}" if c.get("title") else "")
            dst = out / f"{name}.mp4"
            wout = cap.remap_to_shots(clean, c["shots"], fps) if clean else []
            caps_on = captions == "on" or (captions == "auto" and cap.coverage(wout, c["duration"]) >= 0.2)
            show_title = c.get("title") if title != "off" else None
            ass = cap.build_ass(wout if caps_on else [], show_title, W, H, c["duration"], caption_style, title) if ((caps_on and wout) or show_title) else None
            t1 = time.time()
            if sound:
                c["score_lines"] = section_score_lines(c, fps, wout, an, mode == "talk")
            render_section(src, c, W, H, fps, dst, ass, info.has_audio, workers, sound=sound)
            if wout:
                dst.with_suffix(".srt").write_text(ac.to_srt(ac.group_words(wout, max_words=7, max_chars=40)), encoding="utf-8")
            log(f"progress {k}/{len(clips)} sections: {dst.name} ({c['duration']:.0f}s, captions {'on' if caps_on and wout else 'off'}) "
                f"in {time.time() - t1:.0f}s")
            outputs.append(str(dst))
    return {"outputs": outputs, "out_dir": str(out), "plan": str(plan_path),
            "clips": [{"title": c.get("title"), "duration": c["duration"], "start": c["start"], "end": c["end"]} for c in clips]}


def section_score_lines(clip: dict, fps: float, words_out: list[dict], an: dict, talk: bool) -> list[dict]:
    """The emotional arc of a continuous section: what is said (talk), or how the action rises and
    falls (8 s steps: building = Tension, the most intense part = Thrill/Triumph, calm = Mystery)."""
    import highlights as hl

    if talk:
        return hl.clip_score_lines(clip, fps, words_out, True)
    sc = an["score"]
    a = float(clip["start"])
    total = sum(s["frames"] for s in clip["shots"]) / fps
    steps = max(1, int(round(total / 8.0)))
    vals = []
    for k in range(steps):
        i0, i1 = int((a + k * total / steps) * HZ), int((a + (k + 1) * total / steps) * HZ)
        seg = sc[max(0, i0) : max(i0 + 1, i1)]
        vals.append(float(seg.mean()) if len(seg) else 0.0)
    top = int(np.argmax(vals))
    hi, lo = max(vals), min(vals)
    lines = []
    for k, v in enumerate(vals):
        x = (v - lo) / (hi - lo + 1e-9)
        if k == top:
            emo, inten = ("Triumph" if k >= steps - 2 else "Thrill"), 0.9
        elif x > 0.6:
            emo, inten = "Excitement", 0.6 + 0.3 * x
        elif k < top and x > 0.25:
            emo, inten = "Tension", 0.5 + 0.3 * x
        else:
            emo, inten = "Mystery", 0.45
        lines.append({"start": round(k * total / steps, 3), "end": round((k + 1) * total / steps, 3), "text": "", "emotion": emo, "intensity": inten})
    return lines


def render_section(src: str, clip: dict, W: int, H: int, fps: float, dst: Path, ass_text: str | None, has_audio: bool, workers: int = 2,
                   sound: dict | None = None) -> None:
    import highlights as hl

    shots = [dict(s) for s in clip["shots"]]
    if clip.get("fade_in"):
        shots[0]["fade_in"] = max(1, int(round(clip["fade_in"] * fps)))
    if clip.get("fade_out"):
        shots[-1]["fade_out"] = min(shots[-1]["frames"], max(1, int(round(clip["fade_out"] * fps))))
    total = sum(s["frames"] for s in shots)

    def audio(dst_wav: Path) -> None:
        section_audio(src, float(shots[0]["in"]), float(shots[0]["in"]) + total / fps, dst_wav, clip.get("fade_in", 0.0), clip.get("fade_out", 0.0), has_audio)

    hl.render_edit(src, {**clip, "shots": shots}, W, H, fps, dst, ass_text, 0.0, None, workers, audio=audio, sound=sound)
