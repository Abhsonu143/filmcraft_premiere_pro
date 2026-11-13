> **Location note (added when this file moved into the repo):** this is the owner's handoff spec,
> kept here as `ai-engine/CLAUDE.md` because the repository root already has FilmCraft's own
> `CLAUDE.md`. The AI engine lives in this folder (`ai-engine/`) of the owner's FilmCraft fork.
> Status: see `docs/progress.md`. Benchmarks: `docs/benchmarks.md`.

# Project: AI Auto-Editor for Mac (working name: Magic Movie Maker)

Handoff file for Claude Code. Read this fully before doing anything. Place it in the project root as `CLAUDE.md`.

## 1. Goal

An automated video-editing system for personal use on a MacBook (base M3, 8GB RAM). It turns raw clips, recorded audio and assets into edited videos, then lets the owner polish them in a Premiere-style editor.

Video types the owner makes:
- Shorts / Reels (9:16)
- YouTube long-form
- Travel videos (often silent footage + a voiceover recorded from a script)
- Talking videos
- Gaming videos (video audio is game sound, not the owner's voice)

Key requirement: a 5 to 10 minute video should finish in 30 minutes or less (a target, not yet measured). The app must not hang the Mac.

## 2. Hard constraints (do not violate)

- Hardware: base MacBook M3, **8GB RAM**. Run only one heavy model at a time. Unload models after use. Prefer small models.
- **No API keys or paid tokens.** Everything works locally with free models.
- Personal use only for now (not selling). Still respect licenses of any model or library, and tell the owner when one restricts commercial use.
- Never modify original media files. Work on copies, proxies and plans.
- Use hardware encode/decode (VideoToolbox) wherever possible.
- AI runs in a separate process from any UI, so a crash or heavy job never freezes the editor.
- Cache analysis results (transcripts, CLIP embeddings) so repeat runs are fast.
- Colab is an **optional** offload backend (off by default). The system must work fully offline without it. Do not wire the app to call Colab automatically (Google restricts non-interactive use); use a manual handoff (see Phase 5).
- The owner wants simple, clear explanations. Explain any Rust/Swift-side change plainly.

## 3. Decisions made so far

1. Not building a full editor from scratch. Use **FilmCraft** (open-source Rust NLE, https://github.com/storytold/filmcraft) as the editor/polish tool.
2. Build order: our Python AI engine first (companion mode), then optionally fork FilmCraft to add AI panels.
3. FilmCraft is alpha ("young and moving fast", self-estimated about 56% ready for real work). Keep **DaVinci Resolve (free)** installed as a fallback.
4. Colab/Kaggle = optional, manual, off by default.

## 4. Architecture

```
Raw clips + voiceover + music + (optional script)
        |
[ Python AI engine ]  --- analysis: VAD, Whisper, CLIP, beats, scenes, mood
        |
   Timeline plan (JSON)  --->  OTIO / FCP7 XML export
        |                              |
[ FFmpeg + VideoToolbox render ]   [ FilmCraft: polish + final export ]
```

Rule: the AI only produces plans (edit decisions). Plans are always human-editable before render.

## 5. What already exists (written in chat, NOT yet run on a real Mac)

Two files were written and unit-checked in a sandbox with no FFmpeg or models, so **expect bugs on first real run**. If they are not in the repo yet, ask the owner to place `autocut.py` and `voiceover_cut.py` in the project root (they were delivered in the chat), or rebuild them from the descriptions below.

### `autocut.py` (talking-head mode)
- Silence detection (`ffmpeg silencedetect`) -> speech segments with padding -> jump cuts.
- Optional word-level captions via `mlx_whisper`, remapped onto the cut timeline, burned in via the `subtitles` filter.
- Music mix with sidechain ducking.
- Presets: `short` (1080x1920 crop), `long` and `travel` (1920x1080).
- Commands: `plan`, `render`, `auto`. Plan is saved as `<name>.plan.json` (segments editable).
- Render: single ffmpeg call with `-filter_complex_script`, `hevc_videotoolbox` or `h264_videotoolbox`.

### `voiceover_cut.py` (narration-driven mode; imports from `autocut.py`)
- Transcribes narration (MLX Whisper, script text used as `initial_prompt`).
- Splits into beats; indexes clips with CLIP (`open_clip` ViT-B/32 on MPS, one frame per 2s, cached in `~/.cache/autocut`).
- Matches each beat's text to footage, with a reuse penalty; shots about 3s; exact frame counts (no drift).
- Renders each shot with `h264_videotoolbox` (3 in parallel), concatenates with `-c copy`, then muxes voice + ducked music and optionally burns captions.
- Commands: `plan`, `render`, `auto`. Plan: `<voice>.vo.plan.json`.

### Known risks to check first
- `-filter_complex_script` is deprecated in newer FFmpeg (7+); it may warn or need `-/filter_complex`.
- `subtitles` filter needs an FFmpeg build with libass. Verify with `ffmpeg -filters | grep subtitles`.
- On 8GB, 3 parallel hardware-encode jobs may be too many; make worker count configurable and default to 2.
- Whisper and CLIP models download on first run (needs internet once).
- `autocut.py` assumes the main clip has an audio track with speech. Gaming mode and silent clips need different handling (Phase 3).
- Not yet handled: variable-frame-rate phone footage, rotated-clip edge cases, clips with no audio in `autocut.py`.

## 6. Roadmap (do in this order)

**Phase 1: make what exists real**
1. Create the repo, a virtualenv, `requirements.txt`, and a `README`. Dependencies: `mlx-whisper`, `open_clip_torch`, `torch`, `pillow`, `numpy`; system: `ffmpeg` (Homebrew).
2. Smoke test both scripts on one short sample project. Fix bugs. Add a small automated test set for the pure-logic functions (segments, SRT mapping, beat splitting, shot assignment).
3. Benchmark on the owner's Mac: time per stage for a 10-minute video, peak memory. Record results in `docs/benchmarks.md`.

**Phase 2: FilmCraft integration (companion mode)**
4. Add OpenTimelineIO export (and FCP7 XML if needed) to both scripts (`pip install opentimelineio`).
5. Install FilmCraft (prebuilt release, or `brew install --cask filmcraft`; check the latest version). Test importing our OTIO/XML into it with real clips. Document what survives (cuts, audio, captions) and what does not.
6. Read FilmCraft docs: `docs/architecture.md`, `docs/contributing.md`, `docs/control-protocol.md`, `docs/agents.md`, `docs/project-files.md`, `ROADMAP.md`. Summarize how its command system, JSON control channel and MCP server could let our engine drive it.

**Phase 3: more modes**
7. Gaming highlights: no silence detection. Use audio energy spikes, motion bursts, optional OCR of kill-feed/score via Apple Vision, face-cam excitement. Pre-roll about 3s, post-roll about 2s. Duck game audio under a recorded voiceover. Use VAD (Silero) to separate commentary from game sound.
8. Shorts/Reels: pick the strongest hook first, crop to 9:16 following the face/subject (Apple Vision), big word-by-word captions, hit a target length (15/30/60s).
9. Mode auto-detection: check each clip for an audio stream, speech ratio, and whether a narration file was provided.

**Phase 4: voice restructuring (owner's own recorded voice)**
10. Sentence-level analysis: energy (RMS, pitch variation), pace, mood (small local text-emotion model or small local LLM, **must fit 8GB**), hook strength, filler/repeated takes.
11. Planner reorders/trims sentences to a style profile (calm storytelling, hype, documentary, hook-first). Join at clean points with 30 to 80 ms crossfades, loudness matching, room-tone fill. A/B preview and a text diff of what moved.
12. "Trend" = style presets plus optional reference videos the owner supplies (measure cut rate, caption style, music tempo). No live trend scraping.
13. Use only the owner's real recordings. No voice cloning unless the owner asks later (own voice only).

**Phase 5: optional backends**
14. Pluggable AI backend interface: `local` (default), `colab` (manual handoff), `cloud` (later, needs keys, off).
15. Colab handoff: the app exports a small job package (transcript JSON, short audio) -> owner uploads to Drive and runs a provided notebook -> the app imports the results file. Document the free-tier limits (unpublished weekly quota, session timeouts, GPU not guaranteed).

**Phase 6: UI**
16. CLI first (exists), then a simple local UI, then, if it proves worthwhile, fork FilmCraft and add an "AI Auto-Edit" panel plus a "Plan Review" panel (accept/reject each shot) via its UI crate and command registry.

## 7. FilmCraft facts (from its README, checked Nov 2026)

- Open-source pro NLE in pure Rust; macOS, Windows, Linux; MIT or Apache-2.0.
- Premiere-like panels, workspaces, tools and shortcuts; multitrack timeline; color, audio mixer, captions (SRT/WebVTT/SCC), about 90 effects.
- Interchange: FCP7 XML, FCPXML 1.9 to 1.11, OpenTimelineIO, CMX 3600 EDL, AAF, OMF. Cannot open `.prproj` directly.
- Every action is a command; there is a CLI (`filmcraft-cli`), a JSON control channel (TCP), and an MCP server. The UI is one crate (`ui-egui`) over the engine.
- Hardware decode via VideoToolbox on macOS. H.264 export can use the hardware encoder; H.265 export is hardware-only; no AV1 export. Color grading, keys and export rendering still run on the CPU, so heavy effects and 4K may be slow on 8GB.
- Alpha. Camera/phone footage is less tested than conformance streams. Speech-to-text there is optional and off by default.
- Trademark: the ArtCraft name and logos must be removed from any fork or modified version.
- Compiling the Rust workspace on 8GB RAM may be slow; prefer the prebuilt app until a fork is really needed.

## 8. Working agreement for Claude Code

- Make small changes, run them, show the output. Do not claim something works until it was run.
- Before adding a dependency, check it runs on Apple Silicon and fits 8GB.
- Keep every AI decision as an editable plan file, never a one-way render.
- After each phase, update `docs/progress.md` with what works, what was measured, and what is still broken.
- When a decision is the owner's (name, features, whether to fork), ask a short question instead of guessing.

## 9. First task for Claude Code

"Read CLAUDE.md. Set up the project with a virtualenv and requirements. Check that ffmpeg has `videotoolbox` and `subtitles` support. Then run `autocut.py plan` and `render` on one short test clip, fix whatever breaks, and report timings and memory use."
