"""Pure logic of the clip editor: assembly, framing paths, captions."""

import numpy as np
import pytest

import captions as cap
import clip_edit as ce
import reframe as rf
import render


def W(word, s, e):
    return {"word": word, "start": s, "end": e}


def cands(n=14, every=10.0):
    out = []
    for k in range(n):
        t = 5 + k * every
        out.append({"in": t - 1.0, "out": t + 2.5, "impact": t, "score": 0.3 + 0.05 * k if k != 9 else 1.0, "kind": "action"})
    return out


def test_edit_clip_structure():
    clip = ce.edit_clip(cands(), 0, 150, 30, 10, 60, hook=True, src_fps=60, words=[])
    shots = clip["shots"]
    total = sum(s["len"] for s in shots)
    assert 24 <= total <= 34
    assert shots[0]["role"] == "hook" and shots[0]["in"] <= 95 < shots[0]["in"] + shots[0]["len"]  # teaser of the best (impact at 95)
    assert shots[1]["transition_in"]["type"] == "fadewhite"
    body = [s for s in shots[1:]]
    assert all(a["in"] <= b["in"] for a, b in zip(body, body[1:]))  # story order
    slow = [s for s in shots if s.get("speed", 1) < 1]
    assert len(slow) == 1 and slow[0]["speed"] == pytest.approx(0.5)  # 60 fps -> 0.5x, real frames
    assert shots[-1].get("fade_out_s")
    # pieces never play as one continuous stretch
    pieces = clip["pieces"]
    assert all(b2[0] - b1[1] >= 0.8 for b1, b2 in zip(pieces, pieces[1:]))


def test_no_slow_motion_without_frames_or_over_speech():
    c = ce.edit_clip(cands(), 0, 150, 30, 10, 60, hook=False, src_fps=30, words=[])
    assert all(s.get("speed", 1) == 1 for s in c["shots"]) and c["shots"][0]["role"] != "hook"
    talk = [W(" no", 94.9, 95.3)]
    c = ce.edit_clip(cands(), 0, 150, 30, 10, 60, hook=False, src_fps=60, words=talk)
    assert all(s.get("speed", 1) == 1 for s in c["shots"])


def test_pieces_shorten_as_it_builds():
    assert ce.piece_cap(0.0, 1.0) > ce.piece_cap(0.9, 1.0) >= 3.0
    long = [{"in": 0, "out": 9, "impact": 2, "score": 0.5, "kind": "action"}, {"in": 20, "out": 29, "impact": 25, "score": 0.5, "kind": "action"},
            {"in": 40, "out": 49, "impact": 44, "score": 0.5, "kind": "action"}]
    out = ce.normalise(long, 1.0)
    mid = out[1]
    assert mid["out"] - mid["in"] <= ce.piece_cap(0.3, 1.0) + 1e-6 and mid["in"] <= 25 <= mid["out"]


def test_sentence_safe_cuts():
    sents = [{"start": 10.0, "end": 13.0}, {"start": 20.0, "end": 30.0}]
    assert ce.sentence_safe(11.0, 12.5, sents) == pytest.approx((9.9, 13.2))
    a, b = ce.sentence_safe(25.0, 26.0, sents)  # deep inside a long sentence: leave it out
    assert a == pytest.approx(30.1) and b == pytest.approx(19.9)


def test_snap_to_beats():
    shots = [{"len": 2.1, "speed": 1.0}, {"len": 1.8, "speed": 1.0}, {"len": 3.0, "speed": 1.0}]
    ce.snap_to_beats(shots, [0.0, 2.0, 4.0, 6.0])
    assert shots[0]["len"] == pytest.approx(2.0) and shots[0]["len"] + shots[1]["len"] == pytest.approx(4.0)


def test_best_centre_and_smoothing():
    prof = np.zeros(rf.BINS)
    prof[70:80] = 5
    assert rf.best_centre(prof, 0.32, None) == pytest.approx(0.75, abs=0.06)
    times = [i * 0.25 for i in range(12)]
    still = rf.smooth_path(times, [0.5 + 0.01 * (i % 2) for i in range(12)], set(), 0.32)
    assert len({round(v, 4) for _, v in still}) == 1  # holds instead of jittering
    pan = rf.smooth_path(times, [0.2] * 6 + [0.8] * 6, set(), 0.32)
    steps = [abs(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(pan, pan[1:])]
    assert max(steps) <= 0.35 + 1e-6  # speed-limited pan
    cut = rf.smooth_path(times, [0.2] * 6 + [0.8] * 6, {6}, 0.32)
    assert cut[-1][1] == pytest.approx(0.8, abs=0.01)  # jumps with a picture cut
    assert rf.to_fraction(0.5, 0.32) == pytest.approx(0.5) and rf.to_fraction(0.16, 0.32) == 0.0


def test_letterbox_found_even_with_subtitles_in_the_bar():
    g = np.full((90, 160), 60, np.uint8)
    g[:12] = 0
    g[-12:] = 0
    g[-6, 60:100] = 255  # subtitle text inside the bottom bar
    box = rf.letterbox([g, g])
    assert box is not None and box[1] == pytest.approx(13 / 90) and box[3] == pytest.approx(77 / 90)
    assert rf.letterbox([np.full((90, 160), 60, np.uint8)] * 2) is None


def test_track_expression():
    e = render.track_expr([[0.0, 0.2, 0.5], [1.0, 0.6, 0.5]], 1, 0.1)
    assert e.startswith("if(lt((t-0.1),0),0.2") and "0.6" in e
    assert render.track_expr([], 1, 0) == "0.5"


def test_caption_cleanup():
    ws = [W(" we", 0, .2), W(" are", .3, .5)] * 4 + [W(" done", 3, 3.3), W(" the", 3.4, 3.5), W(" the", 3.55, 3.7), W(" end", 3.8, 4)]
    ws += [W(" Ah", 10, 10.3), W(" Thank you.", 20, 21)]
    out = [w["word"].strip() for w in cap.clean_words(ws)]
    assert out == ["we", "are", "done", "the", "end"]


def test_caption_remap_with_slow_motion_and_title_modes():
    shots = [{"in": 10.0, "frames": 60, "speed": 1.0}, {"in": 50.0, "frames": 60, "speed": 0.5}]
    out = cap.remap_to_shots([W(" hi", 10.5, 10.8), W(" boom", 50.2, 50.4), W(" gone", 70, 71)], shots, 30)
    assert [w["word"].strip() for w in out] == ["hi", "boom"]
    assert out[1]["start"] == pytest.approx(2.0 + 0.4)
    assert "Title" not in cap.build_ass(out, "T", 1080, 1920, 4, "pop", "off").split("[Events]")[1]
    intro = cap.build_ass(out, "My clip", 1080, 1920, 4, "clean", "intro")
    assert "0:00:03.00,Title" in intro and "\\fad" in intro
    assert "BOOM" in cap.build_ass(out, None, 1080, 1920, 4, "pop", "off")
