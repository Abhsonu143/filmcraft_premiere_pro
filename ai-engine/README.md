# AI engine (working name: Magic Movie Maker)

A local, free AI auto-editor for a MacBook (built and measured on a base M3 with 8 GB). It turns raw
clips, a recorded voiceover and music into an edited video, then hands the edit to FilmCraft for polish.
It needs no API keys and makes no paid calls; after the first model download it works offline.

It lives in its own folder inside this FilmCraft fork and doesn't touch FilmCraft's Rust code.
FilmCraft is the editor for polishing; this folder holds the Python "companion" that writes the edits.

```
raw clips + voiceover + music (+ script)
        │
 [ Python AI engine ]  silence detection · Whisper (MLX) · CLIP footage search
        │
  plan.json (editable) ──► .otio (+ .srt) ──► FilmCraft: polish + final export
        │
 [ ffmpeg + VideoToolbox render ]  ──► finished .mp4
```

**Rule:** the AI only writes *plans*. A plan is a plain JSON file you can edit before anything is
rendered. Original media is never modified.

## Setup (once)

```sh
brew install ffmpeg            # needs VideoToolbox + libass; check below
cd ai-engine
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q  # 25 tests, ~6 s
```

Check that ffmpeg has what we use:

```sh
ffmpeg -hide_banner -encoders | grep videotoolbox   # h264_videotoolbox, hevc_videotoolbox
ffmpeg -hide_banner -filters  | grep -w subtitles    # burned-in captions (libass)
```

The first run that uses captions downloads Whisper small (~480 MB). The first footage match
downloads CLIP ViT-B/32 (~350 MB). Both land in `~/.cache/huggingface`. After that, everything works offline.

## In the EI app

Everything below is also on EI's **AI** menu (dialogs, progress window, results opened in the
timeline). Build and set-up steps are in [docs/building-ei.md](docs/building-ei.md). The app runs
`ei_engine.py`, the single entry point that wraps all the tools in this folder.

## Modes

### 0. Split a long video: `split_video.py`

```sh
.venv/bin/python split_video.py long.mp4 --out-dir clips/ --clip-len 60          # same quality, no re-encode
.venv/bin/python split_video.py long.mp4 --out-dir clips/ --quality exact        # exact lengths
```

`same` (default) copies the streams, so quality is identical and it's very fast (a 10-minute 1080p file
splits in under a second). Each cut lands on the next keyframe, so clips can run up to one keyframe
interval longer. `exact` re-encodes with the hardware encoder at the source bitrate for frame-exact lengths.
`high`/`medium`/`small` re-encode at 16/8/4 Mb/s. A tail shorter than 1 s joins the last clip.

### Montage: `montage.py`

```sh
# many short clips + a song -> 30 s vertical edit on the beat
.venv/bin/python montage.py music --clips clips/ --music song.mp3 --length 30 --aspect vertical --out edit.mp4
# same soundtrack and cut timing as a reference video, with your clips
.venv/bin/python montage.py reference ref.mp4 --clips clips/ --out like_ref.mp4
```

- **music:** finds the beats (accurate to about 15 ms on a test track) and uses the loudest stretch of
  the song. Shots change every 1/2/4/8 beats (`--pace fast|medium|slow`). Busy clip moments go on strong
  beats; dark and frozen moments are skipped; no moment is used twice. The music fades in and out and is
  loudness-normalised.
- **reference:** detects the reference's cuts (scene changes) and keeps its audio unchanged. For each
  reference shot it picks the clip moment that looks most like it (CLIP image similarity; `--no-clip` for
  motion only) and moves like it. The output keeps the reference's size by default.
- Plan: `<out>.montage.json` (edit `shots`, then `montage.py render`); `<out>.otio` for the timeline.

### 1. Talking head: `autocut.py`

Cuts out the pauses (jump cuts), and can add word captions and ducked music.

```sh
.venv/bin/python autocut.py plan   talk.mp4 --preset short --captions --music bed.mp3
.venv/bin/python autocut.py render talk.plan.json            # -> talk.cut.mp4 (+ talk.cut.srt)
.venv/bin/python autocut.py auto   talk.mp4 --preset long    # plan + render in one go
.venv/bin/python autocut.py export talk.plan.json            # -> talk.otio (+ talk.srt) for FilmCraft
```

Presets: `short` (1080×1920, centre crop, big captions), `long` and `travel` (1920×1080).
Tuning: `--silence-db -32` (raise to -28 in a noisy room, lower to -40 for quiet speech),
`--min-silence 0.5` (shortest pause that gets cut), `--pad 0.12` (breathing room kept around words).
Use `--codec h264` for the widest compatibility (default `hevc`, both hardware encoders).

### 2. Narration-driven (travel): `voiceover_cut.py`

You record a voiceover from your script, and it picks matching footage for every sentence.

```sh
.venv/bin/python voiceover_cut.py plan   vo.m4a --clips footage/ --script script.txt --music bed.mp3 --captions
.venv/bin/python voiceover_cut.py render vo.vo.plan.json --workers 2    # -> vo.vo.mp4
.venv/bin/python voiceover_cut.py export vo.vo.plan.json                # -> vo.otio for FilmCraft
```

How it chooses footage: it transcribes the voice (your script is given to Whisper as a spelling hint),
splits it into beats (sentences), and splits each beat into ~3 s shots. CLIP compares each beat's words
with a frame every 2 s of every clip. It avoids reusing the same moment and back-to-back shots from
the same clip. Shot lengths are whole frames and add up exactly to the voiceover.
`--no-clip` skips matching and rotates through clips. `--no-transcribe` skips Whisper and uses equal shots.

Re-rendering after a plan edit only re-encodes the shots you changed (they are kept in `<name>.work/`).

## Editing a plan

- `autocut` plan: `segments` is a list of `{start, end}` in seconds of the source clip. Delete a
  segment to drop it, widen one to keep more, and so on.
- `voiceover` plan: each entry in `shots` has `clip` (file), `in` (seconds into that clip), `frames`
  (length at the plan's `fps`), and the `text` it illustrates. Change `clip`/`in` to swap footage;
  keep `frames` unless you also adjust neighbours (the total must match the voice).
- `music.db` is the music level before ducking; `captions.words` holds the transcript words, which you
  can fix by hand (spelling) before rendering.

## Open the edit in FilmCraft

`export` writes an OpenTimelineIO timeline: V1 picture, A1 sound (linked to the picture in autocut
mode), and A2 music. In FilmCraft: **File ▸ Import…**, pick the `.otio`, then import the `.srt` as
captions (**File ▸ Import** or the `captions.import` command). The same `.otio` opens in DaVinci
Resolve (File ▸ Import ▸ Timeline). What survives and what doesn't: [docs/filmcraft-integration.md](docs/filmcraft-integration.md).

## Memory and speed on 8 GB

- One model at a time, each run in its own child process. When the process exits the memory is
  returned, so the editor and the rest of the Mac stay responsive.
- Transcripts and CLIP frame embeddings are cached in `~/.cache/autocut` (override with
  `AUTOCUT_CACHE`), keyed by file path, size and modified time, so a second run on the same media
  takes seconds.
- Decode and encode use VideoToolbox (hardware). Shot rendering defaults to 2 parallel encodes.
- Measured numbers: [docs/benchmarks.md](docs/benchmarks.md). Status and known gaps: [docs/progress.md](docs/progress.md).

## Test media

```sh
./scripts/make_fixtures.sh        # small synthetic clips in fixtures/ (macOS `say` voice)
./scripts/make_bench_media.sh     # 10-minute benchmark media in fixtures/bench/
.venv/bin/python scripts/bench.py -- .venv/bin/python autocut.py plan fixtures/bench/talk10.mp4 --captions
```

`fixtures/` is gitignored. Never commit media.

## Licences of what we use

| Component | Licence | Commercial use |
|---|---|---|
| ffmpeg (Homebrew build, run as an external program) | GPL (this build includes x264/x265) | fine to *use*; we don't ship or link it |
| mlx-whisper / Whisper weights | MIT | yes |
| open_clip / OpenAI CLIP ViT-B/32 weights | MIT | yes |
| PyTorch | BSD-3 | yes |
| OpenTimelineIO | Apache-2.0 | yes |
| numpy, Pillow, psutil, pytest | BSD / HPND / BSD / MIT | yes |

None of these limits personal use. If this is ever sold, keep ffmpeg an external tool (don't
bundle a GPL build).
