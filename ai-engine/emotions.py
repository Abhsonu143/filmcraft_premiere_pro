"""emotions: the emotional palette of EI's sound design (voice delivery, music mood, effects).

Every script line carries one emotion from EMOTIONS and an intensity 0–1. The emotion decides:
  - the voice delivery: how expressive (Chatterbox exaggeration), how free the pacing (cfg weight),
    pauses before and after the line;
  - the music: a mood family (12 families: joy, euphoria, love, nostalgia, sorrow, fear, tension,
    mystery, wonder, calm, heroic, anger) with its search words, tempo and an AI-music prompt;
  - the sound effects that suit it (impacts for triumph and surprise, risers into a peak, a
    shimmer for wonder, a heartbeat under dread, silence before a reveal).

Valence (-1 sad/dark … +1 happy/bright) and arousal (0 still … 1 intense) place each emotion on
the usual circumplex map, so the nearest music can be found when a family has no track.
"""

from __future__ import annotations

# family: valence, arousal, music search words, AI-music prompt, tempo range
FAMILIES: dict[str, dict] = {
    "joy": {"listen": "happy cheerful upbeat feel-good music", "valence": 0.8, "arousal": 0.6, "search": ["happy upbeat instrumental", "cheerful acoustic", "feel good ukulele"],
            "prompt": "happy upbeat acoustic pop, ukulele, claps, bright piano, cheerful", "tempo": (100, 130)},
    "euphoria": {"listen": "euphoric energetic electronic dance music with a big drop", "valence": 0.8, "arousal": 0.95, "search": ["energetic electronic", "edm instrumental", "upbeat energetic"],
                 "prompt": "euphoric energetic electronic dance music, big synth chords, driving drums, festival drop", "tempo": (120, 140)},
    "love": {"listen": "romantic tender soft love song music", "valence": 0.7, "arousal": 0.35, "search": ["romantic piano", "romantic strings", "love theme instrumental"],
             "prompt": "romantic tender piano and strings, warm, intimate, love theme", "tempo": (60, 90)},
    "nostalgia": {"listen": "nostalgic bittersweet wistful gentle music", "valence": 0.1, "arousal": 0.3, "search": ["nostalgic piano", "lofi chill", "melancholic piano"],
                  "prompt": "nostalgic bittersweet piano, soft lo-fi warmth, memories, gentle", "tempo": (65, 90)},
    "sorrow": {"listen": "sad slow mournful emotional music", "valence": -0.8, "arousal": 0.2, "search": ["sad piano", "sad cello", "emotional sad instrumental"],
               "prompt": "sad slow cello and piano, grief, emotional, minor key, sparse", "tempo": (50, 75)},
    "fear": {"listen": "scary dark horror music with dissonant drones", "valence": -0.8, "arousal": 0.75, "search": ["horror soundtrack", "dark ambient", "scary music"],
             "prompt": "dark horror atmosphere, dissonant strings, low drones, scary, unsettling", "tempo": (60, 100)},
    "tension": {"listen": "tense suspenseful thriller music with a pulsing beat", "valence": -0.3, "arousal": 0.7, "search": ["suspense", "thriller soundtrack", "dark cinematic tension"],
                "prompt": "suspenseful thriller underscore, pulsing low synth, ticking percussion, tension building", "tempo": (90, 120)},
    "mystery": {"listen": "mysterious curious intriguing quirky music", "valence": -0.1, "arousal": 0.4, "search": ["mystery soundtrack", "mysterious", "investigation music"],
                "prompt": "mysterious curious underscore, pizzicato, celesta, soft mallets, intrigue", "tempo": (80, 110)},
    "wonder": {"listen": "magical wondrous awe-inspiring cinematic music", "valence": 0.6, "arousal": 0.55, "search": ["ethereal ambient", "cinematic atmospheric", "orchestral ambient"],
               "prompt": "awe inspiring cinematic ambient, shimmering pads, choir, sense of wonder, majestic", "tempo": (60, 95)},
    "calm": {"listen": "calm peaceful relaxing ambient music", "valence": 0.5, "arousal": 0.1, "search": ["calm piano", "relaxing ambient", "meditation music"],
             "prompt": "calm peaceful ambient, soft piano, gentle pads, relaxing, tranquil", "tempo": (55, 80)},
    "heroic": {"listen": "epic heroic triumphant orchestral trailer music", "valence": 0.7, "arousal": 0.85, "search": ["epic cinematic", "epic orchestral", "trailer music"],
               "prompt": "epic heroic orchestral trailer music, big drums, brass, triumphant, inspiring", "tempo": (90, 130)},
    "anger": {"listen": "aggressive angry heavy intense rock music", "valence": -0.7, "arousal": 0.95, "search": ["aggressive rock instrumental", "intense action music", "heavy metal instrumental"],
              "prompt": "aggressive intense action music, distorted guitars, heavy drums, rage", "tempo": (130, 170)},
}

# emotion: family, how expressive the voice is (exaggeration low…high over intensity), cfg weight,
# pauses (s) before/after, effects that suit it
_E = {
    # joy
    "Joy": ("joy", (0.55, 0.85), 0.4, 0.0, 0.3, ["chime"]),
    "Happiness": ("joy", (0.5, 0.8), 0.4, 0.0, 0.3, ["chime"]),
    "Cheerful": ("joy", (0.55, 0.85), 0.35, 0.0, 0.25, ["chime"]),
    "Bliss": ("calm", (0.4, 0.6), 0.5, 0.1, 0.5, ["shimmer"]),
    "Delight": ("joy", (0.55, 0.85), 0.4, 0.0, 0.3, ["chime"]),
    "Euphoria": ("euphoria", (0.75, 1.05), 0.3, 0.0, 0.2, ["riser", "impact"]),
    "Excitement": ("euphoria", (0.7, 1.0), 0.3, 0.0, 0.2, ["whoosh", "impact"]),
    "Thrill": ("euphoria", (0.7, 1.0), 0.3, 0.0, 0.2, ["riser", "whoosh", "impact"]),
    "Energy": ("euphoria", (0.7, 1.0), 0.3, 0.0, 0.15, ["whoosh"]),
    # love
    "Love": ("love", (0.45, 0.7), 0.5, 0.1, 0.5, ["shimmer"]),
    "Passion": ("love", (0.6, 0.9), 0.4, 0.0, 0.4, ["swell"]),
    "Affection": ("love", (0.4, 0.6), 0.5, 0.0, 0.4, []),
    "Romance": ("love", (0.4, 0.65), 0.5, 0.1, 0.5, ["shimmer"]),
    # nostalgia / sadness
    "Nostalgia": ("nostalgia", (0.4, 0.6), 0.5, 0.2, 0.6, ["swell"]),
    "Melancholy": ("nostalgia", (0.35, 0.55), 0.55, 0.2, 0.7, []),
    "Sorrow": ("sorrow", (0.35, 0.6), 0.55, 0.3, 0.8, ["sub"]),
    "Grief": ("sorrow", (0.4, 0.65), 0.55, 0.4, 0.9, ["sub"]),
    # fear / tension
    "Fear": ("fear", (0.6, 0.9), 0.4, 0.2, 0.4, ["heartbeat", "sub"]),
    "Dread": ("fear", (0.45, 0.7), 0.55, 0.3, 0.6, ["heartbeat", "braam"]),
    "Anxiety": ("tension", (0.55, 0.8), 0.35, 0.0, 0.25, ["heartbeat"]),
    "Suspense": ("tension", (0.45, 0.7), 0.5, 0.3, 0.6, ["riser", "heartbeat"]),
    "Tension": ("tension", (0.5, 0.75), 0.45, 0.2, 0.4, ["riser"]),
    "Anticipation": ("tension", (0.5, 0.75), 0.4, 0.1, 0.5, ["riser"]),
    # mystery / curiosity
    "Mystery": ("mystery", (0.4, 0.6), 0.5, 0.2, 0.5, ["braam"]),
    "Curiosity": ("mystery", (0.45, 0.7), 0.45, 0.0, 0.35, []),
    "Fascination": ("mystery", (0.5, 0.75), 0.45, 0.0, 0.4, ["shimmer"]),
    # wonder
    "Wonder": ("wonder", (0.5, 0.75), 0.5, 0.2, 0.5, ["shimmer"]),
    "Awe": ("wonder", (0.5, 0.8), 0.55, 0.4, 0.7, ["braam", "shimmer"]),
    "Amazement": ("wonder", (0.65, 0.95), 0.4, 0.3, 0.5, ["impact", "shimmer"]),
    "Wonderment": ("wonder", (0.5, 0.75), 0.5, 0.2, 0.5, ["shimmer"]),
    # calm
    "Serenity": ("calm", (0.3, 0.5), 0.55, 0.1, 0.6, []),
    "Tranquility": ("calm", (0.3, 0.5), 0.55, 0.1, 0.6, []),
    "Peace": ("calm", (0.3, 0.5), 0.55, 0.1, 0.6, []),
    "Calmness": ("calm", (0.3, 0.5), 0.55, 0.0, 0.5, []),
    # heroic
    "Courage": ("heroic", (0.6, 0.9), 0.4, 0.1, 0.4, ["swell"]),
    "Triumph": ("heroic", (0.7, 1.05), 0.35, 0.2, 0.5, ["impact", "riser"]),
    "Pride": ("heroic", (0.6, 0.85), 0.45, 0.1, 0.4, ["impact"]),
    "Hope": ("heroic", (0.5, 0.75), 0.45, 0.1, 0.5, ["swell", "shimmer"]),
    "Freedom": ("heroic", (0.6, 0.9), 0.4, 0.1, 0.5, ["whoosh", "swell"]),
    # anger
    "Anger": ("anger", (0.7, 1.0), 0.35, 0.0, 0.3, ["impact"]),
    "Rage": ("anger", (0.85, 1.15), 0.3, 0.0, 0.3, ["impact", "sub"]),
    # surprise
    "Surprise": ("wonder", (0.7, 1.0), 0.35, 0.45, 0.4, ["impact"]),
    # plain narration
    "Neutral": ("calm", (0.4, 0.55), 0.5, 0.0, 0.35, []),
}

EMOTIONS: dict[str, dict] = {
    name: {"family": fam, "exag": ex, "cfg": cfg, "pause_before": pb, "pause_after": pa, "sfx": sfx,
           "valence": FAMILIES[fam]["valence"], "arousal": FAMILIES[fam]["arousal"]}
    for name, (fam, ex, cfg, pb, pa, sfx) in _E.items()
}
# arousal differs inside a family (Excitement > Joy; Dread < Fear): fine-tune where it matters
for _name, _ar in {"Bliss": 0.25, "Cheerful": 0.65, "Delight": 0.6, "Energy": 0.95, "Passion": 0.7, "Affection": 0.25, "Melancholy": 0.2,
                   "Dread": 0.55, "Anxiety": 0.75, "Suspense": 0.6, "Anticipation": 0.65, "Curiosity": 0.45, "Fascination": 0.5,
                   "Awe": 0.6, "Amazement": 0.8, "Surprise": 0.85, "Triumph": 0.9, "Hope": 0.55, "Pride": 0.7, "Rage": 1.0,
                   "Neutral": 0.35}.items():
    EMOTIONS[_name]["arousal"] = _ar

# the old 10 delivery tags of script_writer, and a few common words
ALIASES = {"neutral": "Neutral", "calm": "Calmness", "warm": "Affection", "story": "Nostalgia", "excited": "Excitement",
           "curious": "Curiosity", "serious": "Tension", "suspense": "Suspense", "hook": "Surprise", "cta": "Hope",
           "happy": "Happiness", "sad": "Sorrow", "scared": "Fear", "angry": "Anger", "epic": "Triumph", "romantic": "Romance",
           "shock": "Surprise", "shocked": "Surprise", "inspiring": "Hope", "inspired": "Hope", "funny": "Cheerful", "energetic": "Energy"}


def resolve(name: str | None) -> str:
    """The canonical emotion name for any tag ("hook", "awe", "TRIUMPH" …); Neutral when unknown."""
    if not name:
        return "Neutral"
    n = str(name).strip()
    for k in EMOTIONS:
        if k.lower() == n.lower():
            return k
    return ALIASES.get(n.lower(), "Neutral")


def info(name: str | None) -> dict:
    return EMOTIONS[resolve(name)]


def family(name: str | None) -> str:
    return info(name)["family"]


def voice_params(name: str | None, intensity: float = 0.6) -> dict:
    """Chatterbox delivery for an emotion at an intensity (0–1)."""
    e = info(name)
    x = min(1.0, max(0.0, float(intensity)))
    lo, hi = e["exag"]
    exag = lo + (hi - lo) * x
    cfg = e["cfg"] - 0.1 * max(0.0, x - 0.6)  # very intense lines run freer (faster, less measured)
    return {"exaggeration": round(min(1.15, exag), 2), "cfg_weight": round(max(0.2, cfg), 2), "temperature": 0.8 if e["arousal"] < 0.7 else 0.85}


def nearest_family(valence: float, arousal: float) -> str:
    return min(FAMILIES, key=lambda f: (FAMILIES[f]["valence"] - valence) ** 2 + 1.3 * (FAMILIES[f]["arousal"] - arousal) ** 2)


def family_distance(a: str, b: str) -> float:
    fa, fb = FAMILIES[a], FAMILIES[b]
    return float(((fa["valence"] - fb["valence"]) ** 2 + 1.3 * (fa["arousal"] - fb["arousal"]) ** 2) ** 0.5)


def prompt_list() -> str:
    """The emotion names for a script-writing prompt, grouped by family."""
    by: dict[str, list[str]] = {}
    for k, v in EMOTIONS.items():
        by.setdefault(v["family"], []).append(k)
    return "; ".join(f"{f}: {', '.join(n)}" for f, n in by.items())
