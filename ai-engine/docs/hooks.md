# Hook library

The first 1–3 seconds decide whether anyone watches. EI keeps a library of opening hooks on this Mac and uses it
in the script writer, Video from Topic and the voiceover edit, offline.

- **Where:** `~/Library/Application Support/EI/hooks/library.json`.
- **Built in:** 24 proven hook patterns (curiosity gap, bold claim, stat shock, in medias res, contrarian,
  mistake, POV, challenge, secret reveal, urgency, before/after, payoff first …), each with an English and a
  Hindi/Hinglish example.
- **Sync (AI ▸ Hook Library… ▸ Sync Hooks):**
  1. Google Trends' daily trending searches for your countries (default India and the US).
  2. This week's most-viewed YouTube videos in your niches (titles and views).
  3. Claude (with web search) studies those and this month's hook formats, and writes new hooks in English,
     Hindi and Hinglish. Each hook comes with its pattern, niche, emotion, why it works, the trend it rides
     and how hot it is. Without Claude, the most-viewed titles become hooks.
  4. Results are merged with no duplicates. A trend's heat halves every three weeks, so old trends sink.
  A 3-niche sync took 77 s and added 68 hooks.
- **Use:**
  - **Script writer:** gets the best hooks for the topic (relevance, niche, heat, language, a range of
    patterns) and is told to adapt one, never copy it.
  - **Voiceover edit:** writes a 3–7-word on-screen hook from them.
- **Command line:** `ei_engine.py hooks-sync --niches gaming travel --geo IN US`;
  `ei_engine.py hooks-list --topic "…" --niche travel`.

## Video from Topic: your own hook, or a hook searched for the topic

- **Hook (optional):** type your own first line. It is spoken word for word as the first segment, and the script
  continues from it and pays it off. It is also the on-screen hook (when 10 words or fewer), and no cold open plays
  before it. `topic … --hook-line "…"`.
- **Search online for a hook for this topic** (tick box): before the script is written, EI looks at the topic's
  most-viewed YouTube videos, and Claude (with web search) finds how successful videos and posts on this topic open
  now. It writes up to 8 topic-specific hooks, strongest first, true to the researched facts. They lead the hook
  suggestions for the script and are kept in the library. `topic … --find-hook`. Your own line wins when both are
  given.

## Learning a creator's style from YouTube subtitles

AI ▸ Download from YouTube ▸ **Subtitles only** (English and/or Hindi) downloads what is said in a video, a
playlist or a whole channel (`youtube.com/@name/videos`) as clean text:
- **Real speech only:** the creator's own subtitles, else YouTube's automatic captions of the language
  actually spoken. A machine translation is never used, and a "Hindi" track that is really English is
  recognised and replaced by the captions of the speech. `subtitles.json` records which text is spoken
  and which is a translation.
- **Cleaned:** rolling duplicates, [Music] tags, >> markers and stray symbols are removed. Automatic captions are
  polished by Claude (punctuation, sentences, obviously misheard words), keeping the speaker's own words,
  slang and Hinglish. Without Claude, pauses become sentence ends.
- **Files:** per video, `<title>.<lang>.txt` and `.srt`; per playlist, `all.<lang>.txt`; plus
  `subtitles-missing.txt`.
- **Learn a style (a name):** the talking style is learned from the spoken text of this download only, and
  appears in Write Voiceover Script and Video from Topic ▸ Style. The profile holds:
  - the words they reach for again and again (new: `vocabulary`), signature phrases and sentence length;
  - questions and exclamations, fillers, how they open, and real passages.

  Several videos teach it better than one; with one video, topic words dominate the vocabulary.
