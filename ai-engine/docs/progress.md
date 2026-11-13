# Progress

## 2026-11-12 (night): Kapwing-style clip maker, audio extraction, model choice

- `highlights.py` (AI ▸ Make Short Clips from Long Video): transcript, then topic sections (lexical cohesion and
  pauses), then sentence scores (hooks, delivery energy, filler and channel-talk penalties), then a language
  model rating plus titles, then clips assembled from many pieces near the target length (10–60 s, mostly
  45–50 s), never sharing material. Gameplay/action mode: per-second loudness, attacks, motion, scene changes
  and dialogue, then hotspots, then 4–7 pieces each, cut at scene changes or quiet moments and never inside a
  line. Framing: face-following crop for talk, an enlarged picture over a blurred background for gameplay.
  Word-by-word captions and a title (ASS).
- Real test: a 92-min 1080p60 gameplay recording, 10 clips of 46–53 s (4–7 pieces each), spread across the
  whole video. First run: transcript 141 s plus a one-time action analysis (~20 min decode); after that,
  2 min 8 s for all 10 clips.
- `extract_audio.py` (AI ▸ Extract Audio from Video): 8 formats, original-quality copy, range, mono, loudness.
- `llm.py`: model choice, with Qwen3-4B-Instruct-2507 as the default and a saved default in settings.json.
- Known: titles of similar fights can sound alike; the first analysis of a very long video is slow.

## 2026-11-12 (evening): the director (edits that feel hand-made)

Feedback on the first version: reference edits copied the *content* of the reference (green field ->
green field) and held a near-frozen shot for 10 s; music edits used fixed shot lengths, any clip
anywhere, no grade, no effects. Rebuilt as four layers, tested on the owner's 149 4K/60 drone clips:

- `footage.py`: per clip (cached), at 5 fps: sharpness, exposure, shake, camera pan direction and speed
  (phase correlation), plus CLIP shot size, subject tags and a look score from keyframes (3x faster than
  a full decode: 149 clips index in about 2 min). It picks the usable moments in each clip.
- `music.py`: beats, bars, energy per beat, sections (calm/mid/high), the drop, and hits. It picks the best
  stretch of a long song. Any video works as the music source.
- `director.py`: cut points by dynamic programming (strong accents and bar lines, with shot length
  following the energy, an accelerating bar before the drop, and a held ending). A beam search over the
  whole sequence covers quality, look, motion versus energy, roles (opener, drop, closer), trip order,
  no repeats, no near-duplicate jump cuts, a subject not repeated within 5 shots, motion continuity and
  no brightness jumps. Effects are chosen for a reason: 60 fps slow motion when calm, speed-ups (never
  on people) and a ramp into the drop, push-ins on static shots, punch-ins on hits, a white flash on the
  drop, dissolves only at calm section changes, whips only between shots panning the same way, and fades
  at the ends. Reference mode borrows **style** (cut timing, transitions, flashes, punches, motion per
  shot, optional colour match), and can use another soundtrack.
- `render.py`: speed ramps (quadratic time mapping), zoom paths with zoompan at 1.5x resolution, a
  look preset plus per-shot exposure matching, flashes, and transitions centred on the cut with handles
  so the length stays frame-exact. A quality check looks for wrong length, black stretches and frozen
  picture.
- Results on the test media: Apsara (13.8 s, vertical) was 9 shots, calm slow-motion intro, flash and
  punch on the drop, a held ending; rendered in 11 s. The Newbies reference (25 s) kept its long-take rhythm
  with varied footage; rendered in 13 s.
- Tests: 40 Python tests (drop detection, beat-aligned cuts, a held ending, frame-exact transitions and
  ramps, flash-versus-cut in references, music from a video file) and the UI tests, which run the
  director from the app.

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

## Short clips v3: edited clips, true-fill framing, better captions (2026-11-12)

- **Edited, not cut out** (`clip_edit.py`): each clip takes one scene of 2–4x its length and keeps only the beats.
  - Pieces are cut on action: in 0.4–1.4 s before an impact, out after the reaction. Cuts never land inside a spoken phrase.
  - Structure: a 2 s teaser of the best moment with a flash cut, then story order with pieces shortening as it builds (6 s → 3 s caps). The climax plays in real-frame slow motion (0.5x from 60 fps), and the payoff is held at the end.
  - Effects: punch-ins on the biggest hits and slow push-ins on calm pieces.
  - Fades, black frames and frozen frames are skipped.
  - Optional music: cuts snap to its beats and it is ducked under the clip's sound.
  - Gameplay test (92 min): 15–22 shots per 50 s clip, drawn from 110–158 s of source. The old version gave 4–7 near-contiguous pieces.
- **True vertical, square and horizontal** (`reframe.py`, renderer `track` and `active`):
  - Apple Vision (faces, people, salient objects, attention) at 4 fps drives a crop that fills the shape. It holds still, pans smoothly at limited speed, and jumps only on picture cuts.
  - Cinematic letterbox bars are detected (even with subtitles inside them) and cropped away.
  - Other framings: `center`, `blur` (the old look) and `bars`.
- **Captions** (`captions.py`):
  - Styles: pop (heavy font, word-by-word highlight), clean, boxed. Placement sits clear of platform buttons.
  - Whisper junk is removed: loops, stutters, lone grunts, stock phrases.
  - Captions: auto/on/off; auto turns them on only when the clip has clear speech.
  - Title card: off by default, or intro/always. The title is always in the file name; duplicate titles are re-asked.
- Timing (M3 8 GB, cached analysis): planning 10 clips takes ~100 s; rendering takes 13–22 s per 50 s clip. `quality_check` is clean on all 10.

## Voice clone, script writer and voiceover editor v2 (2026-11-12)

Full description: `docs/voice-and-voiceover.md`.

- **Voice Clone** (`voice.py`, `voice_prompts.py`): record sentence by sentence in the app, or use existing
  recordings. 24 English + 24 Hindi + 8 Hinglish sentences cover every sound and the main deliveries.
  - Each reference phrase is analysed: language, delivery, pitch, energy, pace, SNR, and an ECAPA similarity check.
  - Speech uses Chatterbox Multilingual v3 (MLX, MIT): the reference is matched to each line's language and
    emotion, 1–4 takes are generated, and the best is kept by Whisper error rate plus voice similarity.
  - Bad lines are regenerated. Pauses are filled with room tone. Output: `.wav` + `.narration.json`.
  - Benchmarked on this Mac against OmniVoice: OmniVoice garbled Hindi and used 5.6 GB, so it is not offered.
- **Write Voiceover Script** (`script_writer.py`):
  - Learns a talking style from `.txt`/`.srt`/`.vtt` or YouTube links (captions only, via yt-dlp in the venv).
  - Tested on 3 Hindi travel vlogs: Hinglish, 175 wpm, signature phrases "हम लोग", "यहां पे".
  - Claude writes through `claude -p` (no API key); Qwen3-4B writes offline.
  - Each line carries emotion, pace, pause, emphasis and a shot idea. A 60 s Hinglish script took 34 s.
  - Can speak the script right away with a chosen voice.
- **Voiceover Edit** (`narrate.py`, replaces `voiceover_cut` in the app; that module and its tests remain):
  - Each line is matched to footage meaning with CLIP. Queries blend the shot idea, its subject alone, and the
    words; Hindi works through Claude or Qwen shot ideas, or the Whisper translation.
  - Scores are relative per line. The whole edit is chosen together by beam search: no reuse, varied clips and
    shot sizes, shooting order kept.
  - Cuts are timed 0.12 s before each line and long lines split at breaths.
  - Sound: natural clip sound under the voice, optional ducked music. Devanagari captions.
- **App:** AI ▸ Voice Clone… (recorder with microphone choice, playback, level/noise/clipping hints), AI ▸ Write
  Voiceover Script… (style learning, Claude/local, speak with a voice), and a reworked AI ▸ Voiceover Edit…
  (shape, pace, shot ideas, caption style, natural sound).
- **Not changed:** Make Short Clips (highlights, clip_edit, reframe, captions): same code, same tests.
