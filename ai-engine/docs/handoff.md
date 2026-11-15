# EI handoff: what was done, why, where everything is, and what's next

Written 2026-11-15 11:40 at the end of a long working session, so a new session can pick up where this one left off. Times are local (IST), from git and the files on disk.

Read this first, then `ai-engine/CLAUDE.md` (the owner's original spec), then `ai-engine/docs/progress.md` (detailed per-feature notes) and `ai-engine/docs/tasks.md` (the longer task list).

---

## 1. What EI is (in one minute)

EI is the owner's AI auto-editor for a **MacBook Air M3 with 8 GB of RAM**. It has two parts:

- **The app:** a fork of FilmCraft, an open-source pure-Rust non-linear editor. It's branded "EI", lives in `/Applications/EI.app`, and has an **AI** menu.
- **A Python AI engine:** `ai-engine/`, shipped inside the app at `EI.app/Contents/Resources/ai-engine`, running in its own virtualenv.

The app starts engine jobs as separate processes and reads their progress lines. When a job finishes, the app imports its result, usually an `.otio` edit, into the timeline.

- **Repo:** `/Users/abhsonu/Documents/Abhay/Projects/claude/ei`, branch `ai-auto-editor`. There are 51+ commits ahead of `main`, and nothing has been pushed yet.
- **Owner's rules:**
  - The Mac must not hang. Only one heavy model at a time.
  - Never modify original media.
  - Every AI result stays an editable plan.
  - Use clear, plain explanations.
  - **Ask before rebuilding the app.**
  - Don't disturb running jobs.
  - Confirm destructive actions.
  - Commit only green states, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Repo rules:** see the root `CLAUDE.md` and `AGENTS.md`. In short:
  - Never crash: no `unwrap`/`panic` outside tests.
  - Clean-room only.
  - Pure Rust in the product.
  - All time is `Tick`.
  - Everything is a command.
  - Every widget has an automation id.

## 2. How to build, test and drive the app

```sh
# build + install (closes nothing by itself; quit EI first or restart it after)
export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"
packaging/macos/ei/package.sh --install
"$HOME/Library/Application Support/EI/venv/bin/pip" install -q -r ai-engine/requirements.txt

# run with the control channel (JSON lines on TCP 9876)
open -a /Applications/EI.app --args --control 9876
```

Gates, all of which must pass before a commit:

- `cargo fmt --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test -p filmcraft-ui-egui` (and `--workspace` when engine crates change)
- `cargo xtask layers`
- `cargo xtask assets`
- `cargo xtask wasm`. The `wasm32-unknown-unknown` target was installed on 2026-11-15; before that this gate silently could not run.
- `ai-engine/.venv/bin/python -m pytest -q ai-engine/tests` (189 tests on 2026-11-15)

Known failures that come from this machine, not the code: 4 APV tests (`crates/apv/tests/oracle.rs`, `export_tests::every_builtin_preset_exports_and_ffprobe_confirms`). Homebrew ffmpeg can't read APV.

Driving the app, using the helper scripts in the session scratchpad (recreate them if they're gone; each is about 10 lines):

- `ctl.py PORT COMMAND JSON` sends `{"method":"engine.execute","params":{"command":…,"params":…}}`.
- `ctlm.py PORT METHOD JSON` sends any method: `ui.click {"id":…}`, `ui.elements {"prefix":…}`, `ui.screenshot`.
- Useful commands:
  - `ai.<tool>` opens a dialog: `ai.talk`, `ai.voice`, `ai.tv`, …
  - `ai.jobs`
  - `ai.queue {"paused":bool}`
  - `ai.queueImport {"path":…}`
  - `ai.queueExport`
  - `ai.queueCancel {"current":bool}`. This removes **every** waiting task, and with `current:true` it also stops the running job. The removed tasks are kept in `~/Movies/EI/Queue/cancelled-<time>.queue.json` (after the next rebuild).
  - `ai.queueRemove {"index":n}` removes one waiting task, 0 = next to run (after the next rebuild).
  - `ai.editJob`
- Headless UI tests render to PNG with `FILMCRAFT_UI_SNAPSHOT_DIR=<dir> cargo test -p filmcraft-ui-egui --test ai_ui`. Use this to look at dialogs when the display is asleep.

## 3. Where things live

| What | Path |
|---|---|
| Engine source | `ai-engine/*.py` (entry `ei_engine.py <subcommand>`) |
| AI UI | `crates/ui-egui/src/panels/ai_tools.rs` (+ `ai_tools/social.rs`), tests `crates/ui-egui/tests/ai_ui.rs` |
| Installed app / engine | `/Applications/EI.app`, `…/Contents/Resources/ai-engine` |
| Engine virtualenv (1.8 GB) | `~/Library/Application Support/EI/venv` |
| Settings | `~/Library/Application Support/EI/settings.json` |
| Job queue | `~/Library/Application Support/EI/queue.json` (`{"items":[…],"paused":bool}`) |
| Queue backups | `~/Movies/EI/Queue/*.json` |
| Voices | `~/Library/Application Support/EI/voices/<name>/` (`profile.json`, `refs/*.wav`, `takes/`) |
| Talking styles | `~/Library/Application Support/EI/styles/<name>.json` |
| Hook library | `~/Library/Application Support/EI/hooks/` |
| Music library (623 MB) | `~/Library/Application Support/EI/music-library/` |
| Analysis cache | `~/.cache/autocut/` (transcripts, footage, footage-clip, action, motion, music, mood, speech, tvlight, games, package…), about 75 MB, keyed by path + size + mtime |
| Models | `~/.cache/huggingface` (17 GB: Whisper, Chatterbox, CLIP, LCM, Qwen…), `~/.cache/torch` (Demucs 320 MB) |
| Outputs | `~/Movies/EI/` (Downloads, Plans, Queue, Scripts, Sections, Sound, Thumbnails, Topics, TV, Voice-tests, Voiceover) |
| Game recordings | `/Volumes/Extreme SSD/game_rec/ghost-of-yotei/goy-*.mov`: PS5, 1080p60 H.264, a keyframe every ~2 s, 0.1–25 GB each |
| Talk-cut outputs | `…/ghost-of-yotei/auto-cuts/` (e.g. `goy-2026-09-10_21-55-42-10 - EI cut.mp4`, 24 GB, 4.4 h, plus `.otio`, `.srt`, `.plan.json`) |
| Gameplay analysis | next to each recording: `<rec>.ei-analysis.json`, `<rec>.ei.srt`, `<rec>.ei-clip.npy` |
| Gameplay plans | `…/ghost-of-yotei/Ghost of Yotei - plans/plans.md` + `plans.json` |
| Reference thumbnails | `~/Movies/EI/Thumbnails/ghost-of-yotei-references/` (atsu.png, hana.png, ina.png) |
| Voice reference audio | `/Users/abhsonu/Documents/Abhay/Projects/claude/ref/clone-audio/roddy-game/` (9 GTA 6 Shorts, ~8 min) |

**Never delete** the samples generated during development (owner's rule).

## 4. Timeline

### 2026-11-12: foundation
- 01:14 `516a7543`: Python auto-editor companion: talking-head cuts, voiceover cut, OTIO export.
- 02:15 `5f46fe14`: Rebrand to EI, AI menu in the app, split, music edit and reference edit.
- 08:30 `1e4c831e`: Director engine for human-like music and reference edits.
- 09:36 `b7cc4f32`: Short clips from long videos, audio extraction, model choice.
- 15:46 `6edeec6b`: Short clips edited from scene beats, reframing, captions.
- 23:00 `2d84cd34`: Voice clone (Hindi + English, Chatterbox), script writer, smarter voiceover edit.
- 23:12–23:33: Docs on models and local data; the Voice Clone recorder follows the chosen languages and shows live status.

### 2026-11-13: voice, topic videos, TV Light Sync, sound
- 00:00 `d4c4cf1c`: Recording that never finished is fixed.
- 01:29 `72c23c86`: **Video from Topic**: research, script, voice, credited media, edit.
- 07:53 `e6fbb862`: The voice clone sounds like one person and less robotic, using one voice sample per narration.
- 09:26–10:43: **TV Light Sync**: a clip played onto a TV in a filmed shot, with the backlight matched; corner re-check, preview, batch, bezel.
- 13:13 `ad3bb67b`: Continuous sections (15 s – 3 min) from a long video.
- 15:21 `e3bbdcb5`: Cmd+C/X/V/A/Z work in text fields on macOS.
- 16:45–17:35: **Sound design**: 42 emotions, mood-matched music checked by listening, effects, music felt rather than noticed, 4K everywhere.
- 17:53–19:59: **Hook library**, plus voiceover craft: cold open, on-screen hook, pacing, punch-ins. You can give your own hook line or search for one.
- 22:57 `5f6224c7`: Jobs get progress bars, pause/resume, retry that resumes, copyable logs. Topic videos are faster and resumable.
- 23:05 `138f5d33`: Voice gets one extra take for weak passages.

### 2026-11-14: queue, health, downloads, thumbnails, social, pacing, TV quality
- 00:02 `3c52e079`: **Queue**: Add to Queue in every tool, edit and reorder waiting tasks, import and export queue files.
- 00:34 `6e83a91c` / 00:44 `bb6e5bdf`: `ei_check.py` health tool. One large model at a time across jobs; workers exit with their job; guards against runaway or frozen takes.
- 06:10–06:40: **Download from YouTube**: playlists, quality or audio only, resume, subtitles only. yt-dlp runs with a JS runtime. A talking style can be learned from subtitles.
- 06:51 `6f2327ab`: **AI Health** window.
- 08:11 `c4f16ef1`: **Parallel YouTube downloads** (1–8 at a time), skip existing or redo all. **Idle memory went from 3.3 GB to 0.17 GB**: the Chinese-font search had been keeping every font in memory.
- 08:22 `104a969b`: Title font fallback for Hindi, Chinese, Arabic and emoji, without loading every font.
- 09:32 `6070c518`: Music edits on external drives no longer fail at sound design (cross-device rename). Long edits no longer repeat one clip.
- 10:11 `78448e18`: **Thumbnails, Titles & Hashtags** (`package`): best frames, huge words, platforms, English/Hindi, a hashtags `.txt`.
- 10:46 `7a51aca6`: **Social Posting** (`social`): YouTube, Instagram Reels, Facebook, Threads, now or scheduled.
- 11:53 `909acadf`: **Studio thumbnails** (the default). Claude acts as art director, a local image model (LCM Dreamshaper v7) draws backgrounds, and Apple Vision does the cut-out.
- 18:54 `5835babd`: Topic videos no longer fail on Openverse null titles.
- 20:57 `17de86e4`: **Topic hints**, **Edit & Retry** for finished jobs (jobs keep their settings), numbers spoken in English inside Hindi voiceovers.
- 21:28 `df51769a`: **Dynamic pace** for music edits: cuts follow how busy the music is. A different edit every run, and feedback on retry.
- 21:54 `f1f13d06`: TV Light Sync is **steady**: per-frame tracking, a stabilised path, no 50→30 fps judder. Framed on the TV, with a bright, clean screen.
- 22:18 `e20bf1ae`: The TV screen looks **filmed, not pasted**. **Horizontal clips can sit in a vertical frame** over a zoomed, adjustable-opacity background.

### Night of 2026-11-14 → 15: the memory crisis and the gameplay planner
- Around 23:00: The owner asked for a rebuild and for the queued Talking Head jobs to be re-queued (5 Ghost of Yōtei recordings, each 1–5 hours).
- **01:56:** The first job (goy-2026-09-12_19-20-15-16) finished. **EI then closed by itself and `queue.json` disappeared**, with no crash report. Found later: the cause was most likely the `.otio` import bug (see 10:02).
- The owner asked for a **gameplay video planner**: give a list of recordings, analyse each one once (`.srt` plus video analysis, kept and reused), research the game online, and plan YouTube, Shorts and Reels videos. Long files must be handled in chunks, because the owner had run out of memory before.
- **I re-queued the remaining talk jobs before fixing memory.** The owner objected: *"why did you run it again before handling large file fix, kill those and only run after you fix it."* I killed them. **Lesson: never run large-file jobs before a memory fix is in place.**
- **02:45 `45baade8`**: Long recordings are read in pieces everywhere:
  - `autocut.transcribe`: files over 15 min go in 10-minute pieces, Whisper loaded once, word times offset, same cache key.
  - `footage.analyse`: frames streamed one at a time (`iter_frames`), only the previous grey frame kept.
  - CLIP frames: `iter_keyframes_rgb`, `voiceover_cut.iter_frames_rgb`, `footage.embed_stream`, batches of 16.
  - Sound in 10-minute blocks: `footage.iter_audio`, `montage.rms_blocks`, `clip_edit.signals`, `highlights` loudness, `montage` motion and loudest window.
  - `music.analyse` reads at most the first 30 min (`MAX_SECONDS`).
  - **Gameplay planner** (`gameplay_plan.py`, subcommand `gameplay-plan`; UI: Talking Head Auto-Cut ▸ "Plan videos (several recordings)").
  - Tests: `tests/test_large_files.py`.
- **02:50–03:00**: Rebuilt; re-queued the 4 talk tasks. The owner reported high usage. Found: the Whisper worker held **5.4 GB of GPU memory** (`IOAccelerator`). MLX keeps every freed GPU buffer in its cache and never returns it. **My earlier "1.2 GB" measurement was wrong:** it used RSS, which doesn't count GPU memory. **Measure with `footprint <pid>` from now on.** I stopped the queue. This also cancelled a "Plan Videos" task the owner had added; its settings were not saved at the time.
- **03:10 `6040dcc4`**: Every model worker runs through `autocut._isolated_call`, which caps MLX's buffer cache at 256 MB before each call and empties it afterwards. 86 minutes of transcription now stays at 1.1–1.6 GB footprint.
- 03:12: Rebuilt; talk jobs running with flat memory (1.2–1.9 GB).

### 2026-11-15 morning: the EI app's own memory leak
- **09:43**: When the next talk job started, the **EI app itself** grew to 34 GB (swap 27 GB). A stack sample showed `timeline::request_peaks` reading through a `MemReader`. The cause:
  - When a talk job finishes, EI imports its `.otio`.
  - `interchange.rs` read **each linked media file whole** (`read_file` + `open_bytes`), which here was the 17.6 GB PS5 recording.
  - The conversion briefly made a second copy, so it peaked at 34 GB and kept 17 GB.
  - That's the likely cause of the 01:56 crash.

  I stopped the job.
- **10:02 `75daa5c8`**:
  - Interchange imports now open linked media in place (`MediaPool::open_file`, `FileReader`).
  - Project Manager reads a consolidated clip in place too, with `cfg(any(unix, windows))` and a whole-file fallback for the web build.
  - Regression test: `an_interchange_import_opens_linked_media_in_place`, which failed before the fix.
- **10:07 `22ea224b`**: Opt-in test `a_big_real_edit_opens_and_plays` (`EI_BIG_EDIT=/path/edit.otio`). Opening the real 4.4-hour edit now takes 0.14 s and peaks at 0.75 GB, compared with 34 GB before.
- Checked and confirmed safe: **export** writes straight to the file in batches of 2–16 frames, and faststart moves the data in 1 MB pieces.
- 10:10: The owner chose to **run only the Plan Videos task and drop the talk tasks**. The Plan task (5 recordings) had survived in `queue.json`. I quit EI, kept only that task (backup in `before-plan-only-2026-11-15.queue.json`), rebuilt, started EI and resumed it.
- 10:12–10:44: **Plan Videos finished.** All 5 recordings were analysed, and each `.ei-analysis.json` / `.ei.srt` is next to it. EI peaked at 2.9 GB. The result was 5 YouTube story episodes in playing order.
- **10:45 `80ca8f27`**: A recording given several ways (video, `.json`, `.srt`) is counted once. Previously a recording given as both `.json` and `.srt` was counted twice.
- **11:16 `0b58b4b5`**: **Voice clone and talking style for "roddy-game"**:
  - The owner confirmed they have permission to use this voice.
  - The source is 9 GTA 6 Shorts with music under the voice. Only 12 phrases (1.4 min) passed the quality check, because the voice was only about 7–11 dB above the music.
  - Added **music removal**: Hybrid Demucs (MIT, through torchaudio, model about 320 MB), 10 s at a time, in an isolated worker. It runs automatically when the voice sits less than 20 dB above the rest (`voice.SEPARATE_BELOW`), via `voice-enroll --separate auto|always|never`.
  - Result: 68 phrases (6.8 min), peak 1.5 GB.
  - Talking style learning (`script-style`) now accepts **audio and video** recordings. They're transcribed, and the pace comes from their timing. Style: 193 wpm, about 13 words per sentence, straight-in hooks, fillers "so/right/like/basically".
  - Test clip: `~/Movies/EI/Voice-tests/roddy-game-test.wav` (31 s, voice match 0.73).
  - `certifi` supplies SSL certificates, because python.org Python has none.
- **11:25 `eab2e168`**: Voice Clone dialog **Make** choice: clone the voice, learn the talking style only, or both (`voice-enroll --make`). The style is saved under the same name, and only free talking counts toward it, not read prompts.
- 11:30: Rebuilt and restarted EI (verified the Make choice in the running app). No jobs queued; the memory log is running.

## 5. Decisions and why

| Decision | Why |
|---|---|
| Fork FilmCraft instead of writing an editor | The owner's spec: a pro NLE in pure Rust with OTIO, a command system and a control channel |
| AI in separate Python processes; heavy models in a spawned worker (`ac.Isolated`) with a global one-model lock | 8 GB Mac: a crash or a heavy model must never freeze the editor, and memory is freed when the worker exits |
| Every AI result is a plan or an `.otio` edit, imported into the timeline | The owner polishes in the editor; nothing is a one-way render |
| Long files processed in 10-minute pieces, frames streamed | A 4-hour, 15–25 GB recording loaded whole needs several GB; pieces keep each job around 1–2 GB |
| Same cache keys after chunking | Earlier transcripts and analyses stay valid, so nothing is recomputed |
| Measure memory with `footprint`, not RSS | MLX and torch on MPS hold GPU memory that RSS doesn't show (the 1.2 vs 5.4 GB mistake) |
| Cap and clear MLX's cache in the shared worker wrapper | One fix covers Whisper, the local LLM and voice cloning |
| Open interchange media in place | The import path was the only one that still read media whole; normal imports already streamed |
| Gameplay analysis stored next to each recording (not in the cache) | The owner wants to reuse or submit the `.json` and `.srt` later, even from another place |
| Planner analyses keyframes only (`-skip_frame nokey`, 320×180) | PS5 recordings have a keyframe every ~2 s: about 30× less decoding |
| Game research cached for a week (`~/.cache/autocut/games/<slug>.json`); the game's names used as a Whisper prompt | Better plans, and correct spelling of characters and places |
| Demucs voice separation only when music is detected | It costs about 1 s per second of audio and adds nothing on clean recordings |
| Style and voice share one name; style only from free talking | Simple to pick in the script tool; read prompts aren't the speaker's own phrasing |
| Clone only your own voice or one you have permission for | Spec rule, and channel strike risk. The owner confirmed permission for roddy-game |
| Ask before every rebuild; back up the queue before changing it | The owner's standing instruction; a rebuild restarts EI and can disturb jobs |

## 6. Problems and their solutions

| Problem | Cause | Fix |
|---|---|---|
| EI closed after the first talk job at 01:56; `queue.json` gone | Most likely: the `.otio` import read the 17.6 GB source whole | `75daa5c8` (in-place import). `queue.json` vanishing isn't fully explained; it may be written on quit and lost when the process is killed |
| EI at 34 GB, 27 GB swap | Same as above | Same |
| Whisper worker at 5.4 GB | MLX buffer cache grows without limit | `6040dcc4` |
| Talk jobs could use several GB | Whole-file audio, frames and CLIP loads | `45baade8` |
| Voice clone kept only 1.4 of 8 minutes | Music under the voice | `0b58b4b5` (Demucs) |
| Torch model download failed (`CERTIFICATE_VERIFY_FAILED`) | python.org Python has no CA bundle | `SSL_CERT_FILE` from `certifi` in the worker |
| Planner counted a recording twice | Dedup key was each file's own path | `80ca8f27` (key by recording stem) |
| UI test clicks missed in a new dialog | The centred dialog moves for a few frames while it sizes | Wait 8 frames before clicking (`ai_ui.rs`) |
| Earlier (14 Nov): idle EI at 3.3 GB | Font search kept every font | `c4f16ef1` |
| Earlier (14 Nov): music edit failed on the SSD | `rename` across devices | `6070c518` |
| Earlier: a commit with a failing test | Gates not all run | Always run every gate; the wasm target is now installed |

## 7. Status right now (2026-11-15 11:40)

- `/Applications/EI.app` is built from `eab2e168` and running with `--control 9876`. The queue is empty.
- **The memory watcher may still be running**, writing `mem.log` in the old session's scratchpad (`memwatch.sh`, every 20 s). Stop it with `pkill -f memwatch.sh` if it isn't needed.
- **Two Talking Head tasks were dropped** at the owner's request: goy-2026-09-08_22-47-17-8 and goy-2026-09-08_19-22-28-7. goy-2026-09-09_22-40-14-9 was stopped earlier. They can be re-imported from `~/Movies/EI/Queue/before-plan-only-2026-11-15.queue.json`.
- Finished talk cuts: 2026-09-12_19-20-15-16 (night of 14th), 2026-09-10_21-55-42-10 (09:42, 24 GB).
- Gameplay analyses exist for:
  - goy-2026-09-07_00-27-04-1
  - goy-2026-09-07_02-01-08-2
  - goy-2026-09-07_23-50-12-3
  - goy-2026-09-08_02-03-37-4
  - goy-2026-09-08_16-07-35-5
  - goy-2026-09-08_19-18-47-6
  - goy-2026-09-11_16-35-48-11
- Voices: `abhay`, `nxstory-a1`, `test1`, `roddy-game`. Styles: `test-hindi-explainer`, `roddy-game`.

## 8. Pending improvements and plan

Next, in order of value:

1. **Shorts and Reels plans** from the analysed Ghost of Yōtei recordings: give the 5 `.json` files, tick Shorts and Reels, set the Focus.
2. **A one-click "Make this plan"** button that sends a plan's parts to the right tool (talk, sections or clips) with their times. Plans already name the tool.
3. **Explain why `queue.json` vanished.** Write it atomically, and keep a copy that survives a killed process.
4. **Per-task removal** from the queue (`ai.queueCancel` only removes everything).
5. **Waveform for 4-hour clips:** it decodes the whole soundtrack in the background. Memory is fine, but it uses CPU for minutes. Consider a cached peak file.
6. **A full-length check:** run one 4-hour talk job end to end with `footprint` logging, now that all the fixes are in.
7. **Voice:** listen to `roddy-game-test.wav`, and consider more takes for weak lines (lowest match 0.52). See `docs/tasks.md` for the voice-training plan (training scripts, "Add to this voice", RVC filter).

Older offers not yet taken up:
- Thumbnail studio: before/after and text-impact layouts, 3-level text, palettes, critic checks, A/B pairs.
- Re-render old TV Light Sync videos with the new look.
- An optional Thumbcraft backend.
- Topic sound design sometimes picks comedic music.

### Update, 2026-11-15 afternoon (second session)

- **Item 1 done.** Shorts and Reels plans made from the 7 analysed recordings (engine run directly, 0 new analysis). `plans.md` / `plans.json` now hold 17 plans: the 5 story episodes, plus 7 Shorts, 2 Reels and 3 YouTube plans that slipped in although only Shorts and Reels were asked for. The earlier plans are in `plans.before-shorts-2026-11-15.*` in the same folder.
- **Planner keeps to the ticked platforms** (prompt and `clean_plans`), so that can't happen again.
- **Items 3 and 4 done in code, not yet in the installed app (needs a rebuild; ask first):**
  - The running task is saved in `queue.json` as `"running"` and comes back first, paused, if EI is killed. A queue that hasn't read the file can no longer delete it. Cancelling writes a `cancelled-<time>.queue.json`.
  - `ai.queueRemove {index}`.
  - What is still not proven is why the file vanished at 01:56. The known hole was: a started task was only in the file after a clean quit, and the file was deleted when nothing else waited.
- Still open: items 2, 5, 6, 7.

Keep in mind:
- The owner's display often sleeps; use headless UI snapshots.
- A heavy job and a rebuild must never overlap.
- Back up the queue before touching it.

## Animated character (2026-11-15)

Engine and the Topic dialog are done and committed (see `progress.md`, 2026-11-15 step 2). The installed EI.app
does not have the dialog block yet: ask the owner before rebuilding. Test videos for the owner to watch:
`~/Movies/EI/Character-tests/` (`gta-studio.mp4` circle, `gta-cutout.mp4` cut-out, `*.character*.mp4` the
character alone). Next: the owner's verdict on the motion, then a Characters window with tints, then the
other tools (`character.add_character_args` / `options` / `add_to_video` are the three calls to reuse).

### Shelved by the owner (2026-11-15)

The owner did not like the 3D sample characters and asked for a real-face presenter from a reference photo.
A trial of Ditto (Ant Group, Apache-2.0; its face finder uses InsightFace models, non-commercial) ran on this
Mac from the scratch folder only: real-looking lips, blinks and head movement, but about 1.1 s per frame
(27x the voiceover's length) and 3.8 GB peak footprint with models freed early (5.1 GB as shipped). No hand
gestures are possible this way. During the 72 s full run the owner said to stop and shelve the feature.
Nothing of the trial is in the repo. The 3D character code (`character.py`, `vrm.py`, the Topic dialog block)
is committed and left as it is, off by default. Do not continue either without the owner asking.

