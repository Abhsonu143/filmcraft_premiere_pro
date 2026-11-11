# Progress

## 2026-11-12 (later): EI app, AI menu, split / music / reference editing

- **Rebrand:** the desktop app is now **EI** (window, menus, About, macOS bundle `EI.app`,
  `io.github.abhsonu143.ei`, its own icon drawn by `packaging/macos/ei/make_icon.py`). ArtCraft marks,
  Discord/website links and `docs/brand/` were removed, as upstream's licence requires of forks. Internal
  crate names, the `.fcproj` format and preference folders keep their FilmCraft names, so projects and
  settings still load.
- **AI menu** (`crates/ui-egui/src/panels/ai_tools.rs`): dialogs for Split, Auto Edit to Music, Edit Like a
  Reference, Talking Head and Voiceover. A background jobs window offers progress, Cancel, Play, Show in Finder
  and Import. An engine set-up window installs and checks the Python environment. Results are imported
  into the project (split clips into the bin, edits as a sequence via `.otio`). Every control has an
  automation id and every tool is a command (`ai.split {…}` etc.) for the control channel and MCP.
- **New engine tools:** `split_video.py` (same-quality split with no re-encode, or exact/re-encoded),
  `montage.py` (beat-cut music edits in 16:9 / 9:16 / 1:1; reference-matched edits with the reference's
  soundtrack and cut timing), and `ei_engine.py` (the app's single entry point; `EI_RESULT` protocol).
- **Measured on the test media:** a 10-minute 1080p video split into 10 one-minute clips in 0.6 s (no
  re-encode). A 20 s vertical music edit from 9 clips took 8 s. A 9 s reference edit took 4 s. Beat timing was
  within 16 ms on a 120 BPM track.
- **Tests:** 35 Python tests; Rust tests for the AI module (argument building, hostile parameters, the job
  runner with cancel) and a UI test that drives the real app headlessly: every dialog, plus real split
  and music jobs whose results are imported (`crates/ui-egui/tests/ai_ui.rs`).
- **Known upstream failure (not caused by EI):** `crates/ui-egui/tests/chrome_ui.rs`
  `header_and_status_bar_are_shown_by_default` fails on upstream `main` too.

## 2026-11-12: Phase 1 done, Phase 2 mostly done

`autocut.py` and `voiceover_cut.py` were not in the repo, so both were rebuilt from the spec
(`CLAUDE.md` §5) and run for real on this Mac (M3, 8 GB, ffmpeg 7.1.1).

### Works (run and checked)

- **Setup:** `.venv` + `requirements.txt`. ffmpeg has `h264_videotoolbox`, `hevc_videotoolbox`, `subtitles`
  (libass), `sidechaincompress` and `loudnorm`.
- **autocut:** `plan`, `render`, `auto`, `export`. Presets short/long/travel. Captions (Whisper small on
  MLX) are remapped onto the cut timeline and burned in. Music is ducked by a sidechain on the voice, then
  loudness-normalised to -14 LUFS.
- **voiceover_cut:** `plan`, `render`, `auto`, `export`. The script is used as a Whisper prompt. Beats become
  ~3 s shots, CLIP matches footage (checked: "the mountains" → MOUNTAINS clip, "Down in the city" → CITY,
  "along the beach" → BEACH), with penalties for reuse and back-to-back same-clip shots. Shots are cached:
  re-rendering a plan only re-encodes changed shots.
- **OTIO export** (both modes): V1/A1 linked, A2 music, full FilmCraft sequence settings (9:16 kept), and an
  `.srt` sidecar. The files validate with the OpenTimelineIO library.
- **Tests:** 25 (`pytest -q`, ~6 s). The pure-logic tests cover silence parsing, segments, frame snapping, word
  remap, SRT, beats, shots and assignment. End-to-end tests with real ffmpeg check that A/V sync holds across
  cuts (flash and beep stay within 1 frame), that frame counts are exact, that voiceover length is exact, that
  shots are reused on re-render, and that captions burn correctly in a folder named `it's a: dir`.
- **Benchmarks:** `docs/benchmarks.md`. 10-minute talking head ≈ 1 min 45 s, 10-minute narration ≈ 9 min 15 s
  (cold). Our processes peak at ≈ 1.1 GB.

### Bugs found and fixed during the first real runs

1. **Lost frames:** ffmpeg's default output sync dropped 2–3 selected frames, so the video was shorter
   than the plan. Fixed with `-fps_mode cfr -r <fps>`. Test: frame count must equal the plan.
2. **Audio drift risk:** `aselect` keeps or drops whole audio packets (~21 ms), not single samples.
   Audio is now re-chunked into one-video-frame packets before cutting, so picture and sound cut on the
   same instant. Test: the flash/beep sync test.
3. **`-filter_complex_script` is deprecated in FFmpeg 7:** we use `-/filter_complex <file>` on 7+ and the
   old flag on older versions.
4. **Voiceover concat failed with a relative plan path** (the concat list resolved paths against the temp
   folder). Fixed by resolving the plan path.
5. **CLIP architecture mismatch:** the OpenAI ViT-B/32 weights need the `quickgelu` variant
   (`ViT-B-32-quickgelu`). Otherwise embeddings are subtly wrong.
6. **Footage reuse window too wide:** a used shot marked the whole short clip as used, so the same moment
   repeated. Fixed: only frames inside the shot count as used.

### Handled from the "known risks" list

- Variable frame rate: everything goes through `fps=<timeline rate>` first, so the output is constant frame rate.
- Rotated phone clips: ffmpeg auto-rotates on decode, and `probe()` reports the rotation (`fixtures/phone_rotated.mp4`).
- Clips with no audio in autocut: a clear error pointing to voiceover mode, not a crash.
- Workers: configurable, default 2 (measured: 3 is only 4 % faster).

### Not done / still broken

- **FilmCraft in-app import is not verified.** Running the downloaded FilmCraft binary wasn't allowed in this
  session. The mapping was checked against FilmCraft's importer source instead. The steps to verify are in
  `docs/filmcraft-integration.md` §2.
- In FilmCraft the 9:16 crop is not exported (clips come in at native size), and neither is ducking.
- Tested only on synthetic media (`say` voice, test patterns). Real camera/phone footage is the next test.
- FCP7 XML export: not added. FilmCraft and Resolve both read OTIO, so it doesn't seem needed. Add it if a tool needs it.
- Phases 3–6 not started.

### Next

1. Owner: run the FilmCraft import check (10 minutes) and one real project through both modes.
2. CLIP indexing speed-up (keyframe-only decode) and per-shot caption burn (see benchmarks).
3. Phase 3: mode auto-detection (cheap), then Shorts hook + face-follow crop, then gaming highlights.
