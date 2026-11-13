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
