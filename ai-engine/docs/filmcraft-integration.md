# FilmCraft integration (Phase 2)

## 1. Getting an edit into FilmCraft

`autocut.py export` / `voiceover_cut.py export` write `<name>.otio` plus `<name>.srt` when the plan
has captions. Import both in one go:

- **App:** File ▸ Import…, select the `.otio` and the `.srt`.
- **CLI (headless check):** `filmcraft-cli import talk.otio talk.srt`, then `filmcraft-cli inspect sequence`.
  Note that each `filmcraft-cli` call is its own headless session, so use `--project` or a `run` script to
  keep the result.
- **Live app over the control channel:** start `filmcraft --control 9876`, then send
  `{"id":1,"method":"engine.execute","params":{"command":"file.import","params":{"paths":["/abs/talk.otio","/abs/talk.srt"]}}}`.

FilmCraft's `file.import` sniffs each file: caption formats (SRT/WebVTT/SCC) become a caption track,
and interchange formats (`.otio`, FCP7 XML, FCPXML, EDL, AAF, OMF) become a sequence
(`crates/engine/src/commands.rs`, `crates/interchange/src/otio.rs`).

## 2. What survives the trip

These results come from reading FilmCraft's OTIO importer (`crates/interchange/src/otio.rs`, v0.20.0) and
round-tripping our files through the OpenTimelineIO library. **They have not yet been checked in the
running app** (see "To verify" below).

| Our plan | In FilmCraft | Notes |
|---|---|---|
| Cuts (segments / shots) | ✅ clips on V1 at the exact frames | times are whole frames at the plan fps |
| Clip sound (autocut) | ✅ A1, **linked** to its V1 clip | `metadata.filmcraft.link`: picture and sound move together |
| Narration (voiceover) | ✅ one clip on A1 | |
| Music | ✅ A2, looped to cover the timeline, at the plan level (`gain_db`) | |
| Music ducking | ❌ | OTIO has no schema for it. In FilmCraft: Essential Sound ▸ tag A1 Dialogue, A2 Music ▸ Ducking ▸ Generate (`essentialSound.generateDucking`) |
| Loudness normalisation | ❌ | set it in FilmCraft: Essential Sound ▸ Auto-Match / Volume level (`essentialSound.autoMatch`, `volume.levelDb`) |
| Frame size / rate (9:16 shorts) | ✅ | we write full `SequenceSettings` in `metadata.filmcraft.settings`; without them FilmCraft falls back to 1920×1080 |
| Scale-to-fill / centre crop | ⚠️ | not exported. Clips come in at their own size, so a 16:9 clip in a 9:16 sequence needs Motion ▸ Scale (or "Scale to Frame Size" and then crop). Next step: write `metadata.filmcraft.effects` with Motion scale |
| Captions | ✅ via the `.srt` sidecar | timings already moved onto the cut timeline; styling is FilmCraft's caption style, not our burn-in style |
| Rotated phone clips | ⚠️ untested | depends on FilmCraft honouring the clip's rotation tag; check with `fixtures/phone_rotated.mp4` |

### To verify (needs the app; not run in this session)

1. Install FilmCraft 0.20.0 (`filmcraft-0.20.0-macos-universal.dmg` from GitHub releases; the Homebrew
   cask was still on 0.19.0 when this was written).
2. `./scripts/make_fixtures.sh && .venv/bin/python autocut.py plan fixtures/talk.mp4 --preset short --captions --music fixtures/music.m4a && .venv/bin/python autocut.py export fixtures/talk.plan.json`
3. File ▸ Import `fixtures/talk.otio` and `fixtures/talk.srt`. Check: the sequence is 1080×1920 at 30 fps,
   with 7 clips on V1 and A1 (linked: dragging one moves the other), music on A2, and captions in sync.
4. Record what you see in the table above.

## 3. How the engine could drive FilmCraft (summary of its docs)

From `docs/architecture.md`, `docs/control-protocol.md`, `docs/agents.md`, `docs/project-files.md`:

- **Everything is a command.** Every action (UI button, menu, shortcut) is a registered engine command with an
  id, params and an `enabled()` check, for example `file.import`, `sequence.addEdit`, `timeline.move`,
  `clip.audioGain`, `essentialSound.generateDucking`, `captions.import`, `file.exportMedia`.
  The UI, CLI, control channel and MCP all dispatch by the same id, so anything a person can do, our engine can do.
- **Three ways in:**
  1. `filmcraft-cli`: headless engine from the shell (`exec <id> key=value`, `run script.jsonl`,
     `import`, `inspect`, `export`). Good for batch jobs with no window.
  2. **JSON control channel**: `filmcraft --control 9876` listens on 127.0.0.1 only. Send one JSON
     request per line (`engine.execute`, `engine.commands`, `ui.inspect`, `ui.screenshot`, …) and get one JSON reply.
     It drives the *running* app, so the owner sees the edit appear live. There's no authentication,
     so only enable it while using it.
  3. **MCP server**: `filmcraft-cli mcp` (headless) or `--bridge 127.0.0.1:9876` (live app). Tools include
     `command_run`, `command_batch`, `sequence_inspect`, `media_import`, `render_frame`.
- **Time** is in ticks (254,016,000,000 per second), but commands also accept `seconds`, `frame` or `timecode`.
- **Projects** are `.fcproj` (JSON). Auto-save and crash recovery exist, and every edit is undoable.

**Suggested path (no FilmCraft fork needed):**
1. Now: OTIO + SRT import (done on our side; needs the in-app check above).
2. Next: a small `filmcraft_bridge.py` that talks to `--control`. It imports the `.otio`, runs
   `essentialSound.setType` + `essentialSound.generateDucking` to rebuild the ducking natively, applies
   Motion scale for 9:16, and takes a `ui.screenshot` to confirm. This keeps the AI in our own process: a
   crash in Python never takes the editor down.
3. Later (Phase 6, only if worthwhile): fork FilmCraft's `ui-egui` crate to add "AI Auto-Edit" and
   "Plan Review" panels that read our plan JSON and call the same commands.

## 4. Fork housekeeping (owner decisions)

- FilmCraft's licence requires forks and modified versions to remove the **ArtCraft name, wordmark and
  logos** (`docs/brand/`, README). This fork currently only *adds* the `ai-engine/` folder. Before
  shipping a modified FilmCraft build (Phase 6), remove or replace that branding.
- Compiling the Rust workspace on 8 GB is slow. Keep using the prebuilt app until a fork is really needed.
