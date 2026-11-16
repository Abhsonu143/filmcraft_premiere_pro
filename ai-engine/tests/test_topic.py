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


def test_text_screens_differ_and_follow_the_fact(tmp_path):
    import topic_cards as tc
    from PIL import Image

    facts = ["The mission cost about ₹615 crore, less than many films", "Luna-25 crashed four days earlier, on 19 August 2023",
             "Water ice may lie in permanently shadowed craters near the pole", "The lander chose its own landing spot in the last 150 metres of the descent",
             "India became the fourth country to land on the Moon", "No one steered it from Earth"]
    assert tc.big_number(facts[0])[0] == "₹615 crore" and tc.big_number(facts[1]) is None and tc.big_number("In 2023 it landed") is None
    used: list[str] = []
    for k, f in enumerate(facts):
        used.append(tc.design_for(f, k, "moon", used))
    assert used[0] == "number" and len(set(used)) >= 5 and all(a != b for a, b in zip(used, used[1:]))
    seen = set()
    for k, design in enumerate(tc.DESIGNS):
        for W, H in ((960, 540), (540, 960)):
            png = tmp_path / f"{design}{W}.png"
            assert tc.make_card(facts[0] if design == "number" else facts[2], "Reuters", png, W, H, url="https://www.reuters.com/a", design=design) == design
            im = Image.open(png)
            assert im.size == (W, H) and len(im.convert("L").resize((32, 32)).getcolors(1024)) > 6  # (something is drawn)
            seen.add(im.resize((16, 16)).tobytes())
    assert len(seen) == 2 * len(tc.DESIGNS)
    # the same publisher keeps its colours; another publisher gets others
    tc.make_card(facts[4], "Reuters", tmp_path / "a.png", 640, 360, design="headline")
    tc.make_card(facts[4], "The Hindu", tmp_path / "b.png", 640, 360, design="headline")
    assert Image.open(tmp_path / "a.png").tobytes() != Image.open(tmp_path / "b.png").tobytes()
    assert tc.make_card("विक्रम ने ख़ुद जगह चुनी", "बीबीसी", tmp_path / "h.png", 640, 360, design="number") == "headline"  # (no number: never an empty number card)


def test_text_screens_can_be_limited_or_left_out():
    clips = [{"created": 0.0, "still": True, "card": True}, {"created": 0.0, "still": True, "card": True}, {"created": 0.0, "still": False}]
    slots = [{"start": i * 3.0, "end": i * 3.0 + 3, "line": i, "part": 0} for i in range(12)]
    c = [{"clip": 0, "in": 0.0, "score": 9.0, "sem": 0.3, "size": "wide"}, {"clip": 1, "in": 0.0, "score": 9.0, "sem": 0.3, "size": "wide"}]
    c += [{"clip": 2, "in": float(k * 4), "score": 1.0, "sem": 0.2, "size": "wide"} for k in range(12)]
    count = lambda picks: sum(1 for p in picks if p["clip"] < 2)  # noqa: E731
    assert count(nr.choose(slots, [c] * 12, clips, story_order=False)) == 2
    assert count(nr.choose(slots, [c] * 12, clips, story_order=False, max_cards=1)) == 1
    assert count(nr.choose(slots, [c] * 12, clips, story_order=False, max_cards=0)) == 0


def test_own_footage_covers_the_share_asked_for():
    clips = [{"created": 0.0, "still": False}, {"created": 0.0, "still": False}]  # 0: found online, 1: the creator's own
    slots = [{"start": i * 3.0, "end": i * 3.0 + 3, "line": i, "part": 0} for i in range(10)]
    c = [{"clip": 0, "in": float(k * 4), "score": 2.0 + 0.3 * k, "sem": 0.3, "size": "wide"} for k in range(10)]
    c += [{"clip": 1, "in": float(k * 4), "score": 1.0 + 0.55 * k, "sem": 0.2, "size": "wide"} for k in range(10)]
    got = {}
    for share in (0.0, 0.5, 1.0):
        picks, reached = nr.choose_with_share(slots, [c] * 10, clips, False, {1}, share)
        got[share] = reached
        assert abs(nr.own_share_of(slots, picks, {1}) - reached) < 1e-9
    assert got[0.0] <= 0.1 and got[1.0] >= 0.9 and got[0.0] < got[0.5] < got[1.0]


def test_own_pictures_and_videos_are_collected(tmp_path, monkeypatch):
    from pathlib import Path

    from PIL import Image

    Image.new("RGB", (800, 600), (200, 30, 30)).save(tmp_path / "a.jpg")
    Image.new("RGB", (800, 600), (30, 200, 30)).save(tmp_path / "b.webp")
    (tmp_path / "c.mp4").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    files = tv.own_footage([str(tmp_path)])
    assert [Path(f).name for f in files] == ["a.jpg", "b.webp", "c.mp4"]
    with pytest.raises(tv.AutocutError):
        tv.own_footage([str(tmp_path / "missing")])
    made = []
    monkeypatch.setattr(tv, "image_to_clip", lambda img, dst, W, H, seconds=6.0, push_in=True: (made.append(Path(img).suffix), Path(dst).write_bytes(b"v"))[1] > 0)
    clips, stills = tv.own_clips(files, tmp_path / "own", 1920, 1080)
    assert len(clips) == 3 and len(stills) == 2 and str((tmp_path / "c.mp4").resolve()) in clips and made == [".jpg", ".jpg"]  # (webp went through a jpg copy)
    assert (tmp_path / "b.webp").is_file() and Image.open(tmp_path / "b.webp").size == (800, 600)  # (the original is untouched)


@needs_ffmpeg
def test_a_wide_video_becomes_tall_with_itself_dimmed_behind(tmp_path):
    src, out = tmp_path / "w.mp4", tmp_path / "t.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=white:s=640x360:d=1:r=30", "-pix_fmt", "yuv420p", str(src)], check=True)
    g = nr.tall_frame_filter(640, 360, 360, 640, 0.4).replace("[framed]", "[vout]")
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-filter_complex", g, "-map", "[vout]", "-frames:v", "1", str(tmp_path / "f.png")], check=True)
    from PIL import Image

    im = Image.open(tmp_path / "f.png").convert("L")
    assert im.size == (360, 640)
    middle, above = im.getpixel((180, 320)), im.getpixel((180, 60))
    assert middle > 235 and 80 < above < 125  # (the picture itself, and the same picture at 40% behind)
    _ = out
