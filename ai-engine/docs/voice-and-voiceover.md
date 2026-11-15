# Voice clone, scripts and the voiceover editor

Three tools that work together (AI menu in EI, or `ei_engine.py` from Terminal):

```
transcripts / YouTube links ──► talking-style profile ──┐
                                                        ▼
topic, niche, length, language ─────────────► Write Voiceover Script (Claude or local)
                                                        │  .script.json: line, emotion, pace, pause, shot idea
your voice (mic or files) ──► Voice Clone ──► speak ◄───┘
                                                │  narration .wav + .narration.json
footage (many short clips / few long ones) ──► Voiceover Edit ──► finished video
```

## 1. Voice Clone (`voice.py`)

* **Record** (recommended): EI shows sentences to read (`voice_prompts.py`): 24 English and 24 Hindi lines plus 8
  Hinglish lines, and a minute of free talking per language. Together they cover every sound of each language
  (Hindi aspirated, retroflex, nasal and nukta sounds, conjuncts) and the deliveries a video needs: calm,
  excited, curious, serious, warm, suspense, hook, call to action. Each take stops when you stop talking. It is
  checked for level, clipping and noise, and against the sentence (Whisper) when the voice is built.
* **Files**: any recordings or videos of you talking. They are cleaned (rumble filter, light denoise, level), cut
  into 3–12 s phrases at pauses, transcribed, and scored for noise and clipping. A phrase that does not sound like
  the rest is scored down: speaker verification (ECAPA) catches someone else talking, or music.
* The voice is saved in `~/Library/Application Support/EI/voices/<name>/` with a `profile.json` and a `refs/`
  folder. Each reference phrase stores its language, delivery, pitch, energy, pace and quality. The folder also
  holds a speaker fingerprint (`centroid.npy`) and your room tone. **Record both Hindi and English.** Cloning a
  Hindi line from an English sample sounds less like you (similarity ~0.45 instead of ~0.8).

**Speaking a script.** Each line goes through these steps:

1. **Pick a sample.** Choose the reference that fits best: same language, same delivery, good quality, spread
   across references.
2. **Generate takes.** Make 1–4 takes. Emotion intensity follows the line's energy; pace follows its tempo.
3. **Keep the best take.** It must say the words (Whisper; large-v3-turbo for Hindi), sound like you (similarity
   to your fingerprint), and have a natural speed. A bad line is generated again.
4. **Join the lines.** Use the script's pauses, filled with your own room tone, then level and compress the
   result so it sounds like one recording.

Output: `<name>.wav` + `<name>.narration.json` (line times, text, emotion, shot idea, quality checks).

**Model:** Chatterbox Multilingual v3, from Resemble AI. It is MIT-licensed and runs on MLX.

| | |
|---|---|
| Download | 2.7 GB, once |
| Memory while speaking | about 4.8 GB, so it runs alone |
| Speed | about 4–5× slower than real time, so 60 s of narration takes about 5 min per take |

Measured on FLEURS Hindi and English speakers:

| | Chatterbox Multilingual v3 | OmniVoice (not used) |
|---|---|---|
| Hindi error rate (Whisper) | 14–29% | about 80%, garbled |
| English error rate | ~0% | |
| Voice similarity | 0.70–0.83 | |
| Memory | about 4.8 GB | 5.6 GB |

Only clone your own voice, or a voice you have permission to use.

## 2. Write Voiceover Script (`script_writer.py`)

* **Talking style.** Learned from `.txt`, `.srt` or `.vtt` files and YouTube links. For YouTube only the captions
  are downloaded: human-made in Hindi or English first, else the automatic captions of the video's own language,
  never a machine translation.
* **What the style measures:**
  - words per minute and sentence length
  - Hindi/English mix
  - question and exclamation rate
  - fillers and signature phrases
  - how the videos open and close
  - a few real passages as examples
* **Where styles are saved:** `~/Library/Application Support/EI/styles/<name>.json`.
* **Claude writes the script** through the Claude Code CLI you are logged into (`claude -p`; no API key, needs
  internet). It runs in an empty folder so no project files are sent. **Local** (Qwen3-4B) works offline but
  writes weaker Hindi.
* **The script is written for speech:**
  - a hook in the first 3 s
  - numbers written as words
  - no symbols or stage directions
  - one breath per line
* **What each line carries:** emotion, pace, energy, pitch, pause and words to stress, plus an English shot idea
  that the voiceover editor uses to pick footage.
* **Length.** With a voice chosen, the script is sized to that voice's measured speaking rate. "Write and Speak"
  produces the narration in the same job.

## 3. Voiceover Edit (`narrate.py`)

### How it picks the picture for each line

1. **Lines.** The narration is split into phrases: from the `.narration.json` (found automatically next to a
   cloned narration), or from a transcript aligned to the script.
2. **Shot ideas.** Each line gets an English shot idea: from the script, else written by the local model or
   Claude (from Hindi too), else Whisper's English translation. The query is built from:
   - the shot idea
   - the same idea without camera words ("drone", "aerial", "shot" match every clip of a drone library equally)
   - the English words themselves
3. **Footage.** Each clip is analysed and cached:
   - every second described with CLIP: content, plus shot size (aerial, wide, medium or close)
   - quality: sharpness, exposure, shake
   - motion
   - the clip's own internal cuts, so long recordings become separate shots
4. **Candidates.** Windows of exactly the slot's length are scored on:
   - meaning, relative to all footage for that line, which dominates
   - quality and look
   - camera energy against the delivery: excited lines get motion, calm lines get stillness
   - a wide or aerial shot to establish the place first
5. **The whole edit is chosen together** (beam search):
   - no moment is used twice
   - no clip back to back
   - shot sizes vary
   - footage stays in shooting order where possible
6. **Timing.**
   - cuts land 0.12 s before each line (picture leads sound)
   - long lines split at breaths
   - pauses are covered, never black
   - dissolves at section changes
   - fade in and out
7. **Sound.**
   - the voice
   - each shot's own natural sound at −24 dB
   - optional music, ducked under the voice
   - −14 LUFS overall

**Captions** use the script's exact words (Devanagari for Hindi) and also go to an `.srt`.

**Plan:** `<out>.narrate.json` records, for every shot, its clip, in-point, line, shot idea and match score.

### Test results

**Footage:** the 149 drone clips in `ref/clips` (analysis already cached).

**English:**
- Claude wrote a 30 s script (13 s).
- The clone spoke it: 2 takes per line, Whisper error 0–10%, voice match 0.65–0.79.
- The edit: 10 shots, 64 s.
- Waterfall, river and sunset lines got waterfall, river and sunset shots.
- The "valley" lines got fields, because the library has no river-valley footage.

**Hindi:**
- The test voice was a single 10 s FLEURS speaker.
- The clone spoke a 61 s Hinglish script written by Claude: 13 lines, voice match 0.82.
- Every line was said correctly. The check counts a Hindi line's own words as said even when Whisper spells
  English words in Devanagari or numbers as digits; the first version of the check did not, so 6 correct lines
  were regenerated for nothing.
- Speed on this 8 GB Mac with other apps open: about 30 s per take of 4–5 s. "Takes per line: 1" halves the time.

## Voiceover edit: hook and craft (2026-11-13)

`narrate_craft.py`, option **Hook** in AI ▸ Voiceover Edit (`--hook auto|teaser|text|both|off`, `--niche`):

- **Cold open:** the narration's strongest line (peak emotion, 1.5–4.5 s) plays first with quick cuts, then a white
  flash into the start. The voice track, the lines, the captions and the score all move with it.
- **On-screen hook:** 3–7 big words on a dark box for the first ~2.8 s, written for this video from the hook
  library (Claude; the first line when Claude is not available).
- **Fast open:** the first ~3 s are cut quicker, on the best-looking, moving footage.
- **Pacing by emotion:** shot length × 0.6 (rage, euphoria) … × 1.35 (calm).
- **Punch-ins:** on the script's emphasis words and on peak lines.
- **Transitions:** a dissolve into calm, sad, nostalgic or loving parts and at real pauses; a white flash into
  intense parts; cuts otherwise.
- **Slow motion:** 0.5× on awe, wonder, triumph, sorrow and love lines when the footage is 50/60 fps.
- **No repeats:** a shot that looks like the same scene as the one before it is avoided (CLIP picture similarity).
- **Framing:** vertical and square crops follow the subject when the footage has another shape.
- **Samples:** `~/Movies/EI/Voiceover/`, where `v1 before` is the previous editor and `v2`/`v3` are the new one.

## Recordings with music under the voice

YouTube videos and Shorts usually have music under the voice. When a recording's voice sits less than 20 dB above the rest (`SEPARATE_BELOW`), Voice Clone first takes the music out. It uses Hybrid Demucs (MIT, through torchaudio; the model downloads once, about 320 MB), 10 seconds at a time, in its own process. Only then does it cut the recording into phrases.

`voice-enroll --separate auto|always|never` (default `auto`).

Measured on 9 GTA Shorts (8 min): without separation, 12 phrases (1.4 min) passed the quality check. With it, 68 phrases (6.8 min) passed, and memory peaked at 1.5 GB.

## Talking style from recordings

Learn Style (Script ▸ Learn a style, or `script-style --name N <files>`) also takes audio and video recordings. They're transcribed on this Mac, and the pace is measured from their timing.

## Voice, talking style, or both (one place)

The Voice Clone dialog has a **Make** choice:
- **Clone the voice:** as before.
- **Learn the talking style only:** pace, sentence length, hooks, fillers and endings, saved as a style for the script writer.
- **Both.**

The style is saved under the same name as the voice. It's learned only from recordings of free talking, not from sentences read aloud from EI's prompts.

Engine: `voice-enroll --make voice|style|both`.
