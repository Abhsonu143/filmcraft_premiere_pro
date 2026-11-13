"""Voice cloning, script writing and the voiceover editor: pure logic (no models)."""

import json

import numpy as np
import pytest

import narrate as nr
import script_writer as sw
import voice as vo
import voice_prompts as vp


def W(word, s, e):
    return {"word": word, "start": s, "end": e}


# ---------------------------------------------------------------- prompts


def test_prompts_cover_both_languages_and_deliveries():
    ps = vp.prompts(["en", "hi"])
    langs = {p["lang"] for p in ps}
    assert langs == {"en", "hi"}
    emos = {p["emotion"] for p in ps if not p.get("free")}
    assert {"calm", "excited", "curious", "serious", "warm", "suspense", "hook", "cta"} <= emos
    hindi = " ".join(p["text"] for p in ps if p["lang"] == "hi")
    for sound in ["ख", "घ", "छ", "झ", "ठ", "ढ", "थ", "ध", "फ", "भ", "ड़", "ढ़", "ँ", "ज़", "फ़", "क्ष", "त्र", "ज्ञ"]:
        assert sound in hindi, sound
    assert len({p["id"] for p in ps}) == len(ps)


# ---------------------------------------------------------------- transcripts & style


def test_parse_srt_and_rolling_vtt():
    srt = "1\n00:00:01,000 --> 00:00:02,500\nHello there\n\n2\n00:00:02,500 --> 00:00:04,000\nfriends of mine\n"
    cues = sw.parse_timed(srt)
    assert [c["text"] for c in cues] == ["Hello there", "friends of mine"] and cues[0]["start"] == pytest.approx(1.0)
    vtt = ("WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nतो दोस्तों आज\n\n00:00:02.000 --> 00:00:04.000\nतो दोस्तों आज हम चलेंगे\n\n"
           "00:00:04.000 --> 00:00:06.000\nहम चलेंगे <c>spiti</c> की ओर\n")
    assert [c["text"] for c in sw.parse_timed(vtt)] == ["तो दोस्तों आज", "हम चलेंगे", "spiti की ओर"]


def test_style_profile_detects_hinglish_and_rate():
    cues = [{"start": i * 4.0, "end": i * 4.0 + 3.5, "text": "तो अभी हम लोग drone उड़ाने वाले हैं, बहुत ही amazing view है!"} for i in range(20)]
    prof = sw.analyse_style([{"name": "a", "cues": cues, "text": " ".join(c["text"] for c in cues)}])
    assert prof["language"] == "hinglish"
    assert 150 <= prof["words_per_minute"] <= 250
    assert prof["exclamation_rate"] > 0.5 and "हम लोग" in prof["signature_phrases"]


def test_pick_caption_track_never_a_translation():
    assert sw.pick_track({"subtitles": {"en": []}, "automatic_captions": {"hi-orig": []}}) == ("en", False)
    assert sw.pick_track({"language": "hi", "automatic_captions": {"hi-orig": [], "en": []}}) == ("hi-orig", True)
    assert sw.pick_track({"language": "fr", "automatic_captions": {"fr-orig": [], "en": []}}) is None


def test_script_prompt_and_cleaning():
    p = sw.build_prompt("Spiti trip", niche="travel", seconds=60, language="hinglish", tone="", audience="", notes="", style=None, footage=["river"])
    assert "Hinglish" in p and "JSON" in p and "river" in p
    assert sw.target_words(60, "en", None, wpm_override=120) < sw.target_words(60, "en", None, wpm_override=180)
    seg = sw.clean_segment({"text": "  Hello (laughs) **world** #travel ", "emotion": "ANGRY", "energy": "x", "pause_after": 9})
    assert seg["text"] == "Hello world travel" and seg["emotion"] == "neutral" and seg["energy"] == 0.5 and seg["pause_after"] == 2.0
    data = sw.extract_json('Sure!\n```json\n{"title": "T", "segments": []}\n```')
    assert data["title"] == "T"


# ---------------------------------------------------------------- voice


def test_audio_features_and_pitch():
    t = np.arange(vo.SR * 2) / vo.SR
    y = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    f = vo.features(y, words=5)
    assert f["pitch_hz"] == pytest.approx(150, rel=0.05) and f["clipped"] == 0
    padded = np.concatenate([np.zeros(vo.SR), y, np.zeros(vo.SR)])
    assert len(vo.trim_silence(padded)) < len(padded) * 0.6


def test_cer_and_language():
    assert vo.cer("Hello, world!", "hello world") == 0
    assert vo.cer("नमस्ते दोस्तों", "नमस्ते दोस्तो") < 0.15
    assert vo.detect_lang("आज का video") == "hi" and vo.detect_lang("today") == "en"


def test_phrase_cuts_for_references():
    ws = [W(" w", i * 0.4, i * 0.4 + 0.3) for i in range(10)] + [W(" x", 6 + i * 0.4, 6 + i * 0.4 + 0.3) for i in range(12)]
    cuts = vo.phrase_cuts(ws)
    assert len(cuts) == 2 and all(3 <= b - a <= 12 for a, b, _ in cuts)


def test_reference_choice_prefers_language_and_delivery():
    def ref(f, lang, emo, q=0.9, lvl=-20):
        return {"file": f, "lang": lang, "emotion": emo, "quality": q, "similarity": 0.8, "text": "x", "features": {"level_db": lvl, "dur": 7}}

    prof = {"refs": [ref("a", "en", "calm"), ref("b", "hi", "calm"), ref("c", "hi", "excited", lvl=-14), ref("d", "hi", "excited", q=0.2)]}
    assert vo.choose_ref(prof, "hi", "excited", 0.85, {})["file"] == "c"
    assert vo.choose_ref(prof, "en", "excited", 0.85, {})["file"] in ("a", "c")
    assert vo.choose_ref(prof, "hi", "calm", 0.2, {})["file"] == "b"
    p1, p2 = vo.engine_params({"emotion": "calm", "energy": 0.2}, "chatterbox"), vo.engine_params({"energy": 0.9, "pace": "fast"}, "chatterbox")
    assert p1["exaggeration"] < p2["exaggeration"] and p2["cfg_weight"] < p1["cfg_weight"]


def test_script_segments_from_json_and_text(tmp_path):
    j = tmp_path / "s.script.json"
    j.write_text(json.dumps({"language": "hinglish", "segments": [{"text": "नमस्ते", "emotion": "warm"}, {"text": " "}]}), encoding="utf-8")
    segs, lang = vo.script_segments(str(j))
    assert lang == "hi" and len(segs) == 1
    segs, lang = vo.script_segments("[excited] What a view! [calm] Then silence.")
    assert lang == "en" and [s["emotion"] for s in segs] == ["excited", "calm"]


# ---------------------------------------------------------------- voiceover editor


def test_subject_only_strips_camera_words():
    assert nr.subject_only("Low drone tracking shot following a winding river up a green valley") == "winding river green valley"
    assert nr.subject_only("aerial shot") == "aerial shot"  # nothing left: keep the brief


def test_slots_cover_the_timeline_and_split_long_lines():
    lines = [{"start": 0.3, "end": 3.0, "text": "a", "words": []}, {"start": 3.4, "end": 14.0, "text": "b",
             "words": [W(" w", 3.4 + i, 3.4 + i + 0.7) for i in range(10)]}]
    slots = nr.shots_for_lines(lines, 15.0, "medium")
    assert slots[0]["start"] == 0 and slots[-1]["end"] == 15.0
    assert all(abs(a["end"] - b["start"]) < 1e-9 for a, b in zip(slots, slots[1:]))  # no gaps, never black
    assert slots[1]["start"] == pytest.approx(3.4 - 0.12)  # picture leads the words
    assert sum(1 for s in slots if s["line"] == 1) >= 3 and max(s["end"] - s["start"] for s in slots) <= nr.PACE_MAX["medium"] + 1.3


def test_choice_never_repeats_a_moment_and_varies_clips():
    clips = [{"created": 0.0}, {"created": 0.0}]
    slots = [{"start": i * 3.0, "end": i * 3.0 + 3, "line": i, "part": 0} for i in range(4)]
    best = [{"clip": 0, "in": 0.0, "score": 3.0, "sem": 0.3, "size": "wide"}, {"clip": 0, "in": 10.0, "score": 2.9, "sem": 0.3, "size": "wide"},
            {"clip": 1, "in": 0.0, "score": 2.8, "sem": 0.3, "size": "close"}, {"clip": 1, "in": 8.0, "score": 2.7, "sem": 0.3, "size": "medium"}]
    picks = nr.choose(slots, [best] * 4, clips, story_order=False)
    spans = {(p["clip"], p["in"]) for p in picks}
    assert len(spans) == 4  # every moment used once
    assert all(a["clip"] != b["clip"] for a, b in zip(picks, picks[1:]))  # alternate clips


def test_caption_words_follow_the_script_text():
    ws = nr.caption_words([{"start": 1.0, "end": 3.0, "text": "आज हम पहाड़ों पर हैं", "words": []}])
    assert [w["word"].strip() for w in ws] == ["आज", "हम", "पहाड़ों", "पर", "हैं"]
    assert ws[0]["start"] == 1.0 and ws[-1]["end"] <= 3.0


def test_hindi_check_ignores_spelling_of_english_words_and_numbers():
    text = "बस तीस percent battery बची थी"
    assert vo.speech_error("बस 30% बैटरी बची थी", text) < 0.2  # said right, written differently
    assert vo.speech_error("मैं कुछ और ही बोल रहा हूँ", "पूरा पहाड़ golden हो गया") > 0.5
    assert vo.speech_error("Today I am here", "Today I'm here") < 0.3
