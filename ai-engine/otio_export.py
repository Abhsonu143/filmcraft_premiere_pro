"""OpenTimelineIO export of autocut / voiceover plans, for polishing in FilmCraft (or Resolve).

    python autocut.py export talk.plan.json          -> talk.otio  (+ talk.srt when captioned)
    python voiceover_cut.py export vo.vo.plan.json   -> vo.otio    (+ vo.srt)

Layout:
  V1  picture (autocut: the kept segments of the clip; voiceover: the matched shots)
  A1  autocut: the clip's own sound, linked to V1 / voiceover: the narration
  A2  music (when the plan has music), at its plan level; ducking is not exported (OTIO has no
      schema for it): use FilmCraft's Essential Sound ▸ Ducking, or render with our CLI.

FilmCraft-specific data rides in `metadata.filmcraft`, which FilmCraft's importer reads
(crates/interchange/src/otio.rs): full sequence settings (so a 9:16 short opens as 1080x1920 rather
than the 1920x1080 fallback), clip links (picture + sound move together) and clip gain.
Other tools ignore that metadata. Captions are not part of OTIO: import the .srt sidecar.
"""

from __future__ import annotations

import json
from pathlib import Path

import autocut as ac


def _otio():
    try:
        import opentimelineio as otio
    except ImportError as e:
        raise ac.AutocutError("opentimelineio is not installed: pip install -r requirements.txt") from e
    return otio


def sequence_settings(width: int, height: int, fps: float) -> dict:
    """FilmCraft `SequenceSettings` (crates/project/src/lib.rs) — every field, or the importer
    falls back to 1920x1080."""
    from fractions import Fraction

    r = Fraction(fps).limit_denominator(1001)
    label = f"{float(r):.3f}".rstrip("0").rstrip(".")
    return {
        "width": int(width),
        "height": int(height),
        "frame_rate": {"num": r.numerator, "den": r.denominator},
        "par": [1, 1],
        "sample_rate": 48000,
        "drop_frame": False,
        "preset": f"Custom {width}x{height} {label}",
        "audio_master": "Stereo",
        "preview_codec": "ProRes 422",
        "max_bit_depth": False,
        "max_render_quality": False,
        "working_space": "Rec. 709",
    }


class _Builder:
    def __init__(self, name: str, width: int, height: int, fps: float):
        self.otio = _otio()
        self.fps = fps
        self.refs: dict[str, object] = {}
        self.durations: dict[str, float] = {}
        self.link = 0
        self.tl = self.otio.schema.Timeline(name=name, global_start_time=self.rt(0))
        self.tl.metadata["filmcraft"] = {"settings": sequence_settings(width, height, fps)}

    def rt(self, frames: float):
        return self.otio.opentime.RationalTime(frames, self.fps)

    def tr(self, start_s: float, dur_s: float):
        # Whole frames at the timeline rate: what FilmCraft reads back exactly.
        return self.otio.opentime.TimeRange(self.rt(round(start_s * self.fps)), self.rt(round(dur_s * self.fps)))

    def track(self, name: str, kind: str):
        k = self.otio.schema.TrackKind.Video if kind == "video" else self.otio.schema.TrackKind.Audio
        t = self.otio.schema.Track(name=name, kind=k)
        self.tl.tracks.append(t)
        return t

    def ref(self, path: str):
        if path not in self.refs:
            dur = self.durations.get(path)
            if dur is None:
                try:
                    dur = ac.probe(path).duration
                except ac.AutocutError:
                    dur = 0.0
                self.durations[path] = dur
            avail = self.tr(0, dur) if dur > 0 else None
            self.refs[path] = (Path(path).resolve().as_uri(), avail)
        url, avail = self.refs[path]
        return self.otio.schema.ExternalReference(target_url=url, available_range=avail)

    def clip(self, path: str, start_s: float, dur_s: float, *, link: int | None = None, gain_db: float = 0.0):
        c = self.otio.schema.Clip(name=Path(path).name, media_reference=self.ref(path), source_range=self.tr(start_s, dur_s))
        meta = {}
        if link is not None:
            meta["link"] = link
        if gain_db:
            meta["gain_db"] = gain_db
        if meta:
            c.metadata["filmcraft"] = meta
        return c

    def music(self, plan: dict, total_s: float) -> None:
        m = plan.get("music") or {}
        path = m.get("path")
        if not path or not Path(path).is_file():
            return
        dur = ac.probe(path).duration
        a2 = self.track("Music", "audio")
        t = 0.0
        while t < total_s - 1.0 / self.fps and dur > 0:  # loop the bed to cover the timeline
            piece = min(dur, total_s - t)
            a2.append(self.clip(path, 0.0, piece, gain_db=float(m.get("db", -16.0))))
            t += piece


def build_autocut(plan: dict):
    b = _Builder(Path(plan["input"]).stem + " (autocut)", plan["width"], plan["height"], float(plan["fps"]))
    segs = ac.plan_segments(plan)
    v1, a1 = b.track("V1", "video"), b.track("A1", "audio")
    for s, e in segs:
        b.link += 1
        v1.append(b.clip(plan["input"], s, e - s, link=b.link))
        a1.append(b.clip(plan["input"], s, e - s, link=b.link))
    b.music(plan, ac.output_duration(segs))
    return b.tl


def build_voiceover(plan: dict):
    fps = float(plan["fps"])
    b = _Builder(Path(plan["voice"]).stem + " (voiceover)", plan["width"], plan["height"], fps)
    for c in plan.get("clips", []):
        b.durations[c["path"]] = float(c.get("duration", 0))
    v1, a1 = b.track("V1", "video"), b.track("Narration", "audio")
    total_frames = 0
    for s in plan.get("shots", []):
        frames = int(s["frames"])
        if frames <= 0:
            continue
        # Express the in point on the timeline grid; the shot length stays exactly `frames`.
        c = b.clip(s["clip"], float(s["in"]), frames / fps)
        c.source_range = b.otio.opentime.TimeRange(b.rt(round(float(s["in"]) * fps)), b.rt(frames))
        v1.append(c)
        total_frames += frames
    total = total_frames / fps
    a1.append(b.clip(plan["voice"], 0.0, total))
    b.music(plan, total)
    return b.tl


def captions_srt(plan: dict) -> str | None:
    cap = plan.get("captions")
    if not cap or not cap.get("words"):
        return None
    words = cap["words"]
    if plan.get("mode") == "autocut":
        words = ac.remap_words(words, ac.plan_segments(plan))
    big = cap.get("style") == "big"
    cues = ac.group_words(words, max_words=3 if big else 7, max_chars=18 if big else 38)
    return ac.to_srt(cues) if cues else None


def build(plan: dict):
    mode = plan.get("mode")
    if mode == "autocut":
        return build_autocut(plan)
    if mode == "voiceover":
        return build_voiceover(plan)
    raise ac.AutocutError(f"unknown plan mode {mode!r}")


def export_plan_file(plan_path: Path, out: Path | None = None) -> Path:
    plan = json.loads(plan_path.read_text())
    tl = build(plan)
    stem = plan_path.name.replace(".vo.plan.json", "").replace(".plan.json", "").replace(".json", "")
    out = out or plan_path.with_name(stem + ".otio")
    _otio().adapters.write_to_file(tl, str(out))
    ac.log(f"OTIO written: {out}")
    srt = captions_srt(plan)
    if srt:
        out.with_suffix(".srt").write_text(srt, encoding="utf-8")
        ac.log(f"captions sidecar: {out.with_suffix('.srt')} (import it as captions in FilmCraft)")
    return out
