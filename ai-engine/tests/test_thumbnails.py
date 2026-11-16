"""Thumbnails, titles and hashtags: the parts that need no network or language model."""

import numpy as np
import pytest
from PIL import Image

import thumbnails as th


def test_lines_hashtags_and_copy_checks():
    assert th.lines_of("IT'S OVER?") == ["IT'S", "OVER?"]
    assert len(th.lines_of("GTA 6 IS ONLY 30 FPS", 2)) == 2
    assert th.hashtags(["GTA 6", "#gta6", "#PS5 Pro!", "", "#GTA6"]) == ["#GTA6", "#PS5Pro"]
    assert th.strip_unsupported("IT'S OVER 😱🔥") == "IT'S OVER"
    c = th.clean_copy({"titles": ["A" * 150], "thumbnail": [{"text": "खेल खत्म?", "highlight": "खत्म"}, "SECOND"],
                       "youtube_hashtags": ["a", "b", "c", "d", "e", "f", "g"], "instagram_hashtags": ["x"] * 9}, "hi")
    assert len(c["titles"][0]) == 100 and c["thumbnail"][1] == {"text": "SECOND", "highlight": ""}
    assert len(c["youtube_hashtags"]) == 5 and c["instagram_hashtags"] == ["#x"]
    r = th.rules_copy("GTA 6 will only run at 30 FPS", "", "en")
    assert r["titles"] and r["thumbnail"][0]["text"] and all(t.startswith("#") for t in r["youtube_hashtags"])
    assert "TITLE" in th.txt_for("youtube", "en", th.clean_copy(r, "en")) and "HASHTAGS" in th.txt_for("reels", "en", th.clean_copy(r, "en"))


@pytest.fixture
def frame(tmp_path):
    # a landscape frame with a "face" on the right
    rng = np.random.default_rng(1)
    arr = (rng.random((1080, 1920, 3)) * 80 + 60).astype(np.uint8)
    arr[300:700, 1300:1600] = (200, 150, 120)
    p = tmp_path / "f.png"
    Image.fromarray(arr).save(p)
    return {"src": str(p), "t": 1.0, "path": p, "det": {"faces": [(0.68, 0.28, 0.83, 0.65, 0.9)], "humans": [], "objects": [], "attention": []}}


@pytest.mark.parametrize("platform,lang", [("youtube", "en"), ("youtube", "hi"), ("shorts", "hi"), ("instagram", "en"), ("reels", "en")])
def test_compose_sizes_and_placement(frame, tmp_path, platform, lang):
    out = tmp_path / f"{platform}-{lang}.jpg"
    words = "पैसे बर्बाद?" if lang == "hi" else "IT'S OVER?"
    notes = th.compose(frame, platform, words, words.split()[-1], lang, "pop", out, tmp_path)
    im = Image.open(out)
    assert im.size == th.PLATFORMS[platform]["size"]
    assert notes["words"] == words and not notes.get("under_badge")
    assert notes.get("covers_subject", 0) <= 0.15, notes
    assert notes["text_height"] >= (0.25 if platform == "youtube" else 0.08), notes  # big enough to read on a phone
    assert notes["layout"] == ("crop" if platform == "youtube" else "framed")


def test_words_leave_the_youtube_badge(frame, tmp_path):
    # the subject on the left puts the words on the right, where the duration badge is
    frame["det"]["faces"] = [(0.15, 0.3, 0.3, 0.65, 0.9)]
    notes = th.compose(frame, "youtube", "SIX WORDS THAT FILL THE SIDE", "SIDE", "en", "box", tmp_path / "b.jpg", tmp_path)
    assert not notes.get("under_badge"), notes
