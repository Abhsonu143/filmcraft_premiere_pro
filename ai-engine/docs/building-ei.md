# Building and running EI on a Mac

EI is this repository's desktop app (FilmCraft rebranded, plus the **AI** menu). These steps were
run on a base M3 MacBook with 8 GB of RAM and macOS 26.

## 1. What you need (once)

| What | Why | Install |
|---|---|---|
| Xcode command line tools | compiler, `codesign`, `iconutil` | `xcode-select --install` |
| Rust ≥ 1.95 | builds the app | `brew install rustup && rustup default stable` (add `/opt/homebrew/opt/rustup/bin` to `PATH`) |
| ffmpeg (with VideoToolbox + libass) | the AI tools render with it | `brew install ffmpeg` |
| Python 3.10+ | runs the AI engine | python.org installer or `brew install python@3.12` |

## 2. Build the app

```sh
packaging/macos/ei/package.sh --install      # build + EI.app + copy to /Applications
# or: packaging/macos/ei/package.sh          # just dist/EI.app
#     packaging/macos/ei/package.sh --dmg    # also dist/EI-<version>-macos-arm64.dmg
```

A first build takes about 4 minutes on an M3; rebuilds after small changes take about 1 minute. The script
limits the build to 4 parallel jobs (`CARGO_BUILD_JOBS`) so 8 GB doesn't swap.

The app is signed ad hoc, so it runs on the Mac that built it. If macOS still blocks it, right-click
EI.app and choose **Open** once.

## 3. Set up the AI engine (once)

Open EI and choose **AI ▸ AI Engine Setup… ▸ Install**. This creates
`~/Library/Application Support/EI/venv` and installs the Python packages (about 1.5 GB: PyTorch,
Whisper on MLX, CLIP, OpenTimelineIO). The window then shows which AI features are ready.
The speech and image models download the first time a feature needs them.

Doing it by hand gives the same result:

```sh
python3 -m venv ~/Library/Application\ Support/EI/venv
~/Library/Application\ Support/EI/venv/bin/pip install -r ai-engine/requirements.txt
```

Overrides: `EI_PYTHON=/path/to/python` picks the Python, and `EI_ENGINE_DIR=/path/to/ai-engine` picks
the engine (useful when you're editing the engine). Without them, the app uses the engine copy inside
`EI.app/Contents/Resources/ai-engine`.

## 4. Use it

| AI ▸ … | You choose | You get |
|---|---|---|
| Split Long Video into Clips… | a video, clip length (default 60 s), quality (default *same as original*: no re-encode, cuts on the nearest keyframe; or *exact lengths*: re-encoded at the original bitrate; or High/Medium/Small), output folder | `<name>_part001.mp4`, `…002`, … in the folder; optionally imported into the project |
| Auto Edit Clips to Music… | clips (files or a whole folder), music (**any audio or video file**), length, shape (16:9 / 9:16 / 1:1), pace, music start, look, effects, story order | a director-made edit: cuts on accents, shot lengths following the music, slow motion, punch-ins, a flash on the drop, one grade; plus an `.otio` |
| Edit Like a Reference Video… | a reference video, your clips, optional other soundtrack (audio or video), shape, look (incl. *match reference*), effects | your footage edited in the reference's style (cut timing, transitions, flashes, punches, motion), plus an `.otio` |
| Make Short Clips from Long Video… | a long video, number of clips, lengths (shortest / mostly / longest, default 10 / 48 / 60 s), shape, content (auto / talk / gameplay), framing, captions, language model | N clips like Kapwing's clip maker: each edited from several pieces (talk: best sentences of a topic with fillers and pauses removed and an optional hook first; gameplay: the most intense moments of one scene cut at natural points), reframed, titled, with captions; plus `<name>.clips.json` |
| Extract Audio from Video… | videos or folders, format (MP3, M4A, WAV, AIFF, FLAC, Apple Lossless, Opus, Ogg), quality (or original: no re-encode), range, mono, loudness | audio files next to the videos (or in a folder you choose) |
| Talking Head Auto-Cut… | a video, format, captions, music | jump-cut `.mp4` (+ `.srt`) and an `.otio` |
| Voiceover Edit… | voiceover, clips, script, music | narration-matched `.mp4` and an `.otio` |

Jobs run in the background (**AI ▸ AI Jobs** shows progress, Cancel, Play, Show in Finder, Import into
Project). The engine is a separate process, so the editor stays responsive. If the engine fails or
you cancel, nothing else is affected, and your original files are never changed.

Everything is also scriptable through EI's control channel and MCP (`ai.split`, `ai.music`,
`ai.reference`, `ai.talk`, `ai.voiceover` with parameters, `ai.jobs`, `ai.cancel`; see the comment
at the top of `crates/ui-egui/src/panels/ai_tools.rs`).

Language models: AI ▸ AI Engine Setup ▸ Language model sets the default (saved in
`~/Library/Application Support/EI/settings.json`); each clip job can pick another. Default **Qwen3 4B**
(2.3 GB, Apache-2.0, the best that fits 8 GB). Also Qwen2.5 7B (4.3 GB, for 16 GB Macs), Qwen2.5 3B
(research licence, non-commercial), Qwen2.5 1.5B (fastest), or none (rules only).

## 5. Tests

```sh
cd ai-engine && .venv/bin/python -m pytest -q            # engine (35 tests)
cargo test -p filmcraft-ui-egui --test ai_ui             # AI menu end to end in the real UI
```
