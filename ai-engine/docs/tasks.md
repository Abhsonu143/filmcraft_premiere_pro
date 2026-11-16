# EI task list (for later)

## Voice training (make the clone sound exactly like you)

Context: the current clone (Chatterbox, zero-shot) listens to only about 10 s of your voice per narration.
Consistency is already fixed (2026-11-13: one voice sample per narration, passages, light processing).
"Exactly you" needs training on more of your speech. Plan agreed: record first, then Option A, then
optionally Option B.

- [ ] **Training reading scripts in 3 languages** (English, Hindi, Hinglish), about 10–15 minutes of reading
  each.
  - [ ] Cover all sounds of each language (as `voice_prompts.py` does) plus long natural passages:
    storytelling, explaining, excited reactions, calm narration, questions, numbers, names, English words
    inside Hindi.
  - [ ] Split into ~2-minute sessions, so recording can happen over several days.
  - [ ] Show them in the Voice Clone recorder as a "Training session" mode, with progress per language.
- [ ] **"Add to this voice"** in AI ▸ Voice Clone: new recordings or files are added to an existing voice instead
  of replacing it (engine already supports `voice-enroll --add`; the dialog needs the option).
- [ ] **Training pack export:**
  - [ ] Collect all clean speech of a voice (recorded takes + your videos).
  - [ ] Cut it into 3–12 s phrases, transcribe them, and drop music, other voices and noise.
  - [ ] Write the dataset (wav + transcript list) as one zip, with a report of minutes per language.
- [ ] **Option A, "your-voice filter"** (voice conversion, RVC method, MIT):
  - [ ] Check that the recipe works for Hindi + English on an 8 GB Mac for inference.
  - [ ] Write a Colab notebook: upload pack, train (about 1 h on a free GPU), download the model.
  - [ ] **Voice Clone ▸ Import trained voice**: load the model.
  - [ ] Apply it after Chatterbox in `voice.speak` (Chatterbox gives the words and emotion, the filter gives
    your exact timbre).
  - [ ] Measure before/after (voice match, line consistency) the same way as the A/B test of 2026-11-13.
- [ ] **Option B (optional), fine-tune the speech model** on 30–60+ min:
  - [ ] Check which model has a working fine-tuning recipe for Hindi + English (Chatterbox community recipes,
    XTTS-v2, F5-TTS).
  - [ ] Colab notebook, import, and an A/B test against Option A.
- [ ] Rebuild EI.app after the voice changes (`gta6-release-improved.wav` is the reference for the current
  quality).

## Other open items

- [ ] Push the `ai-auto-editor` branch to GitHub (needs `gh auth login`).
- [ ] Video from Topic: skip YouTube sections that start on a fade to black.
- [ ] Optional: "Speak Script…" and "Test this voice" in the app (speak an existing script, play the result in
  EI).
