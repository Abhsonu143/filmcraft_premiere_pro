# Sound design (optional): emotions you can hear

Turn on **Sound design** in Speak Script, Voiceover Edit, Video from Topic, Make Short Clips (both clip
types) or Auto Edit Clips to Music. Off, everything sounds as before.

## The emotions

Every line carries one of 42 emotions plus an intensity (0–1):

| Mood (music family) | Emotions |
|---|---|
| joy | Joy, Happiness, Cheerful, Delight |
| euphoria | Euphoria, Excitement, Thrill, Energy |
| love | Love, Passion, Affection, Romance |
| nostalgia | Nostalgia, Melancholy |
| sorrow | Sorrow, Grief |
| fear | Fear, Dread |
| tension | Anxiety, Suspense, Tension, Anticipation |
| mystery | Mystery, Curiosity, Fascination |
| wonder | Wonder, Awe, Amazement, Wonderment, Surprise |
| calm | Serenity, Tranquility, Peace, Calmness, Bliss, Neutral |
| heroic | Courage, Triumph, Pride, Hope, Freedom |
| anger | Anger, Rage |

- **Who sets them:** Claude, when it writes the script. It is asked for an arc: a hook emotion, at least three
  moods a minute, a reveal and one peak. For your own recorded voiceover or talk clips, Claude reads the lines
  (word cues when Claude is not available). For gameplay clips, the edit sets them: teaser = Thrill,
  build = Tension, slow-motion climax = Triumph, payoff = Pride. For a music edit, the song's sections set them.
- **Changing them:** Speak Script and Voiceover Edit show **Emotions**, a list of every line with its emotion and
  intensity. Changes are saved into the script file at once. In a `.txt` script, write `[Awe 0.8] line`.

## What each part does

- **Voice:** each passage is spoken with its emotion's delivery: how expressive (Chatterbox exaggeration
  0.3–1.15), how freely paced, and a real pause before reveals, awe and grief.
- **Music** (one cue per act): the script is split into the acts that best follow its arc (dynamic programming:
  every line's mood counts by length and intensity; each music change has a cost; no act shorter than a fifth
  of the video). Each act gets its own track:
  - **Free music from the internet:** searched by mood on Openverse (Jamendo, ccMixter, Freesound). Then **listened
    to** with CLAP, because titles lie: "Nostalgic walk" is euphoric dance music. Every track is filed under the mood it
    really has. Licence "Safe for monetized videos" takes only CC0, CC BY and CC BY-SA. Credits go into
    `<video>.credits.txt` and Video from Topic's description.
  - **My music folder:** your tracks, sorted by mood by listening.
  - **AI-composed score:** MusicGen small composes a cue per mood on this Mac (slow; personal use only).
- **Transitions:** a surprise, triumph, rage or fear after a calmer act is a hard cut: the music stops for a beat,
  an impact lands on the first word, and the new cue starts on it (with a riser before it when the act was
  building). Other acts crossfade in the pause between lines.
- **Effects** (made in code; free, offline):
  - an impact or braam on the hook;
  - a riser and impact into peaks;
  - a dip to silence before a reveal;
  - a shimmer for wonder, a chime for joy, a heartbeat under dread, a sub drop under grief or rage, a swell into love
    and hope;
  - whooshes on cuts while the line is energetic.
- **Mix:** each track levelled, ducked about 10 dB under the voice and swelling in the pauses, louder in intense
  acts; -14 LUFS.

## Files

- `<name> - scored.wav` (Speak Script): voice + score. The voice alone stays in `<name>.wav`.
- `<name>.score.json`: the whole plan (acts, tracks, offsets, transitions, effects with times). Edit it and mix
  again with `ei_engine.py score VOICE.wav --narration X.score.json --out new.wav`.
- Library: `~/Library/Application Support/EI/music-library/` (`index.json` lists every track, its moods and credit).

## Samples

`~/Movies/EI/Sound/`: effects (`v1-effects`), and the GTA 6 test in versions (`v1` … `v4`).
