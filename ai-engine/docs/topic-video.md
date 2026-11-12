# Video from Topic

AI ▸ **Video from Topic…** (engine: `ei_engine.py topic`, code: `topic_video.py`).

Give a topic ("GTA 6 release") or a link to a post or article. EI researches it, writes the script, and
(optionally) makes the video.

## Steps

1. **Research.** Claude, through the Claude Code app you are logged into, with web search, collects:
   - 6–12 facts, each with its source link
   - the angle (why it matters now)
   - a summary
   - articles
   - primary YouTube videos (official channels first)
   - short image searches for the real subjects

   With a link, the page is read first. The research is saved as `research.json` and reused when the same topic
   runs again in the same folder ("Research again" / `--fresh` to refresh). Offline ("Local model") needs a
   link: it reads that page only.
2. **Script.** `script_writer` writes it from the researched facts only. It says where a claim comes from and
   invents no personal anecdotes. Length is in seconds, sized to the voice's pace. Language: English, Hindi or
   Hinglish, optionally in a learned talking style. Every line has an emotion and a shot idea.
3. **Voice.** One of:
   - a cloned voice (Voice Clone)
   - a macOS voice (`mac:Rishi`, `mac:Tara`, `mac:Aman`, Indian English; `mac:Lekha`, Hindi). These are
     instant, for previews.
   - **Script only:** no voice, no video, just the script and sources.
4. **Pictures,** from the sources you tick:
   - *Free-licensed:* Wikimedia Commons and Openverse photos and videos. Licences that forbid edits are skipped;
     licence and author are recorded. The researcher's searches run first and each is tied to the line that
     mentions it; lines with no match get a short search of their own.
   - *Fact cards:* headline cards made from the research facts, with the source named.
   - *YouTube clips:* 3 short sections (8 s each) of up to 3 videos the research found. Opt-in: only where you
     have the right to use them (commentary, news). Every source is listed.
   - *Your footage:* any clips or folders.

   Photos and cards become slow push-in shots.
5. **Edit.** The voiceover editor (`narrate.py`) chooses a shot for every line:
   - media found for that line gets a strong bonus
   - a photo or card is shown once (never split across two slots)
   - captions use the exact script words; music is optional

## Output folder

`~/Movies/EI/Topics/<topic>/`, or the one you choose.

| File | Contents |
|---|---|
| `<topic>.mp4` | the video |
| `<topic>.wav` + `.narration.json` | the voiceover and its line times |
| `<topic>.script.json` / `.txt` | the script |
| `sources.md` | summary; every fact with its link; articles; videos found; every media item with licence, author and page |
| `description.txt` | ready to paste: summary, sources, credits for every photo and video |
| `research.json` | the raw research |
| `media/` | clips made for the edit; `media/originals/` holds the downloaded originals |

## Measured (M3 8 GB, 13 Nov 2026)

| Run | Time |
|---|---|
| GTA 6 release, script only (45 s) | research 55 s + script 19 s = **74 s**. 11 current facts from Rockstar Newswire, Forbes, GameSpot, Reuters/Yahoo Finance |
| Chandrayaan-3 landing, 30 s vertical, macOS voice, free media + cards | research 47 s, script 20 s, voice 1 s, media 3 min (16 credited ISRO/NASA/Commons photos), edit 1.5 min: **about 5 min** |

With a cloned voice, add about 4–8 minutes per minute of speech, depending on takes per line.
