"""The thumbnail studio without the network, Claude or the image model."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

import thumb_studio as ts


def test_props_must_match_and_sizes_fit_the_model():
    assert not ts.matches("PlayStation 5 console", "Sony-PlayStation-3-2001A-wController-L.jpg")
    assert ts.matches("PlayStation 5 console", "PlayStation 5 and DualSense.jpg")
    assert not ts.matches("Mumbai skyline", "Delhi at night.jpg")
    for W, H in [(1280, 720), (1080, 1920), (1080, 1350)]:
        w, h = ts.gen_size(W, H)
        assert w % 8 == 0 and h % 8 == 0 and abs(w / h - W / H) < 0.03 and 280_000 < w * h < 380_000
    assert ts.bright((20, 20, 60)) == (255, 212, 0) and ts.bright((255, 255, 0)) == (255, 255, 0)


def test_the_directors_plan_is_checked():
    fb = ts.rule_concepts([{}] * 4, {"en": [{"text": "IT'S OVER", "highlight": "OVER"}]}, ["en", "hi"], 2, "GTA 6")
    raw = [{"layout": "spiral", "subject": 99, "background": {"type": "generate", "prompt": "neon city"}, "palette": ["#zzz", "#FF0000"],
            "props": [{"query": "PlayStation 5"}, {"query": "second"}], "words": {"en": {"main": "30 FPS?!", "highlight": "30"}}}, "not a concept"]
    got = ts.clean_concepts(raw, 4, ["en", "hi"], fb)
    assert len(got) == 1
    c = got[0]
    assert c["layout"] == fb[0]["layout"] and 1 <= c["subject"] <= 4 and c["palette"] == ["#FF0000"] and len(c["props"]) == 1
    assert c["words"]["en"]["main"] == "30 FPS?!" and c["words"]["hi"]["main"]  # (a missing language falls back)


@pytest.mark.parametrize("lang,text", [("en", "IT'S OVER?"), ("hi", "सिर्फ 30 FPS?")])
def test_pro_words_fill_their_box(tmp_path, lang, text):
    layer = ts.pro_words({"kicker": "GTA 6", "main": text, "highlight": "30 FPS" if lang == "hi" else "OVER"}, lang, ["#FFD400", "#E01E37"], 600, 300, tmp_path, "t", False)
    assert layer is not None and layer.width <= 600 + 1 and layer.height <= 300 * 1.35 + 1
    assert layer.width > 300 or layer.height > 150  # (big, not a corner note)
    a = np.asarray(layer.getchannel("A"))
    assert (a > 0).mean() > 0.25


def subject(tmp_path):
    im = Image.new("RGBA", (500, 900), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((150, 40, 350, 260), fill=(220, 170, 140, 255))
    d.rectangle((90, 250, 410, 900), fill=(240, 110, 30, 255))
    return im


@pytest.mark.parametrize("layout", ts.LAYOUTS)
@pytest.mark.parametrize("platform,size", [("youtube", (1280, 720)), ("shorts", (1080, 1920))])
def test_every_layout_composes(tmp_path, layout, platform, size):
    c = {"layout": layout, "palette": ["#FFD400", "#E01E37"], "props": [], "arrow": True, "ring": True, "badge": "PS5 PRO",
         "words": {"en": {"kicker": "GTA 6", "main": "30 FPS?!", "highlight": "30"}}}
    prop = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    ImageDraw.Draw(prop).rectangle((40, 40, 360, 260), fill=(240, 240, 240, 255))
    assets = {"subject": subject(tmp_path), "second": subject(tmp_path), "prop": prop, "frame_bg": None}
    out = tmp_path / f"{layout}-{platform}.jpg"
    notes = ts.compose(c, assets, platform, "en", size, out, tmp_path, None)
    assert Image.open(out).size == size
    assert notes["words_height"] >= (0.18 if platform == "youtube" else 0.08), notes
    if platform == "youtube" and layout != "versus":
        x0, y0, x1, y1 = notes["words_box"]
        assert not (x1 > 1280 * 0.8 and y1 > 720 * 0.8), "the words stay out of the duration badge"


def test_light_colours_stay_readable(tmp_path):
    # a white palette colour made white words on a white pill
    b = ts.badge("30 FPS", 100, (255, 255, 255), tmp_path, "w")
    px = np.asarray(b.convert("RGBA"))
    solid = px[px[..., 3] > 250][..., :3]
    assert (solid.mean(axis=1) < 200).mean() > 0.3, "the pill is coloured, not white"
