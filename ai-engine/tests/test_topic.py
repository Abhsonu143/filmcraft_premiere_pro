"""Video from Topic: research handling, media matching, cards, credits (no network, no models)."""

import json
import subprocess

import pytest
from conftest import needs_ffmpeg

import narrate as nr
import topic_video as tv

DATA = {
    "topic": "Chandrayaan-3 moon landing",
    "summary": "India landed near the lunar south pole.",
    "angle": "three years on",
    "facts": [
        {"fact": "Vikram touched down on 23 August 2023.", "url": "https://isro.gov.in/a", "source": "ISRO"},
        {"fact": "Pragyan rover drove about 101 m.", "url": "https://isro.gov.in/b", "source": "ISRO"},
    ],
    "articles": [{"title": "Landing", "url": "https://isro.gov.in/a", "publisher": "ISRO"}],
    "youtube": [{"url": "https://www.youtube.com/watch?v=abc", "title": "Landing live", "channel": "ISRO", "why": "official"}],
    "image_queries": ["Vikram lander", "Pragyan rover Moon"],
    "keywords": ["Chandrayaan-3"],
}
SCRIPT = {"title": "Moon", "segments": [
    {"text": "Vikram touched down near the south pole.", "visual": "Aerial view of the Vikram lander on the Moon"},
    {"text": "Then the Pragyan rover rolled out.", "visual": "Pragyan rover on the lunar surface"},
    {"text": "Where were you that night?", "visual": "Crowd watching a big screen"},
]}


def test_facts_notes_forbid_invention():
    notes = tv.facts_notes(DATA)
    assert "ONLY these researched facts" in notes and "(source: ISRO)" in notes
    assert "do not invent personal experiences" in notes


def test_research_prompt_has_the_link_and_json_shape():
    p = tv.research_prompt("https://x.com/post", 45, "hinglish", ("A post", "Body text here"))
    assert "A post" in p and "Body text here" in p and '"youtube"' in p and "45-second" in p


def test_media_queries_tied_to_lines(monkeypatch, tmp_path):
    calls = []

    def fake_search(q, limit=6, video=False):
        calls.append(q)
        return [{"kind": "image", "url": f"https://img/{q.replace(' ', '_')}.jpg", "page": "p", "title": q, "license": "CC BY 4.0",
                 "author": "A", "provider": "Commons", "width": 2000}]

    monkeypatch.setattr(tv, "commons_search", fake_search)
    monkeypatch.setattr(tv, "openverse_search", lambda q, limit=6: [])
    monkeypatch.setattr(tv, "download", lambda url, dst: dst.write_bytes(b"x" * 6000) or True)
    monkeypatch.setattr(tv, "image_to_clip", lambda img, dst, W, H, seconds=6.0, push_in=True: dst.write_bytes(b"v") or True)
    items, hints = tv.collect_media(SCRIPT, DATA, tmp_path / "media", ["free"], 1080, 1920)
    plain = [q for q in calls if not q.startswith("Chandrayaan-3")]
    assert plain[:2] == ["Vikram lander", "Pragyan rover Moon"]  # the researcher's searches first
    lines = {i["query"]: i["line"] for i in items}
    assert lines["Vikram lander"] == 0 and lines["Pragyan rover Moon"] == 1
    assert any(q.startswith("Chandrayaan-3") for q in calls)  # a line with no match gets its own search, with the topic
    assert all(str(p).endswith(".mp4") for p in hints) and sorted({li for v in hints.values() for li in v}) == [0, 1, 2]


def test_sources_and_description_credit_everything():
    items = [{"kind": "image", "file": "/m/m000.mp4", "license": "CC BY 2.0", "author": "Jane", "page": "https://c/1", "title": "Lander", "provider": "Commons"},
             {"kind": "card", "file": "/m/card0.mp4", "license": "made by EI", "author": "", "page": "https://isro.gov.in/a", "title": "fact", "provider": "ISRO"}]
    md = tv.sources_md(DATA, items, SCRIPT)
    assert "[ISRO](https://isro.gov.in/a)" in md and "| m000.mp4 | image | CC BY 2.0 | Jane |" in md and "Landing live" in md
    desc = tv.description_txt(DATA, items, SCRIPT)
    assert "Credits:" in desc and "Jane, CC BY 2.0" in desc and "card" not in desc.split("Credits:")[1]


def test_card_png(tmp_path):
    png = tmp_path / "c.png"
    assert tv.make_card("Vikram touched down near the lunar south pole on 23 August 2023", "ISRO", png, 1080, 1920)
    from PIL import Image

    assert Image.open(png).size == (1080, 1920)
    assert tv.make_card("विक्रम ने चाँद के दक्षिणी ध्रुव पर लैंड किया", "इसरो", tmp_path / "h.png", 1920, 1080)


@needs_ffmpeg
def test_image_becomes_a_push_in_clip(tmp_path):
    png = tmp_path / "c.png"
    tv.make_card("A headline", "", png, 640, 360)
    clip = tmp_path / "c.mp4"
    assert tv.image_to_clip(png, clip, 640, 360, seconds=2.0)
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-count_frames", "-show_entries", "stream=width,height,nb_read_frames",
                          "-of", "json", str(clip)], capture_output=True, text=True).stdout
    st = json.loads(out)["streams"][0]
    assert (st["width"], st["height"], int(st["nb_read_frames"])) == (640, 360, 60)


def test_stills_are_shown_once_and_hints_win():
    clips = [{"created": 0.0, "still": True}, {"created": 0.0, "still": False}]
    slots = [{"start": 0, "end": 2.5, "line": 0, "part": 0}, {"start": 2.5, "end": 5, "line": 0, "part": 1}]
    card = [{"clip": 0, "in": 0.0, "score": 5.0, "sem": 0.3, "size": "wide"}, {"clip": 0, "in": 2.5, "score": 5.0, "sem": 0.3, "size": "wide"},
            {"clip": 1, "in": 0.0, "score": 1.0, "sem": 0.2, "size": "wide"}]
    picks = nr.choose(slots, [card, card], clips, story_order=False)
    assert [p["clip"] for p in picks].count(0) == 1  # the card once, not split over both halves


def test_offline_research_needs_a_link():
    with pytest.raises(tv.AutocutError):
        tv.research("some topic", 30, "en", writer="local")


def test_cards_never_back_to_back_and_capped():
    clips = [{"created": 0.0, "still": True, "card": True}, {"created": 0.0, "still": True, "card": True}, {"created": 0.0, "still": False}]
    slots = [{"start": i * 3.0, "end": i * 3.0 + 3, "line": i, "part": 0} for i in range(4)]
    c = [{"clip": 0, "in": 0.0, "score": 9.0, "sem": 0.3, "size": "wide"}, {"clip": 1, "in": 0.0, "score": 9.0, "sem": 0.3, "size": "wide"}]
    c += [{"clip": 2, "in": float(k * 4), "score": 1.0, "sem": 0.2, "size": "wide"} for k in range(4)]
    picks = nr.choose(slots, [c] * 4, clips, story_order=False)
    kinds = [clips[p["clip"]].get("card", False) for p in picks]
    assert sum(kinds) == 1 and not any(a and b for a, b in zip(kinds, kinds[1:]))
