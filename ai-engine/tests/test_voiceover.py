import pytest

import voiceover_cut as vc
from autocut import AutocutError


def w(t, s, e):
    return {"word": t, "start": s, "end": e}


def test_split_beats_tiles_whole_voiceover():
    words = [w(" We", 0.4, 0.6), w(" arrived.", 0.6, 1.0), w(" The", 1.2, 1.4), w(" sea", 1.4, 1.8), w(" again", 3.0, 3.5)]
    beats = vc.split_beats(words, 4.0)
    assert [b["text"] for b in beats] == ["We arrived.", "The sea", "again"]
    assert beats[0]["start"] == 0.0 and beats[-1]["end"] == 4.0
    assert all(a["end"] == b["start"] for a, b in zip(beats, beats[1:]))


def test_split_beats_caps_long_sentences_and_empty_input():
    words = [w(f" w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(40)]  # 20 s, no punctuation, no pauses
    beats = vc.split_beats(words, 20.0, max_beat=9.0)
    assert len(beats) >= 3 and all(b["end"] - b["start"] <= 9.6 for b in beats)
    assert vc.split_beats([], 5.0) == [{"start": 0.0, "end": 5.0, "text": ""}]
    assert vc.split_beats([], 0.0) == []


def test_make_shots_exact_frame_total():
    beats = [{"start": 0.0, "end": 7.31, "text": "a"}, {"start": 7.31, "end": 8.0, "text": "b"}, {"start": 8.0, "end": 20.017, "text": "c"}]
    shots = vc.make_shots(beats, 30, 3.0)
    assert sum(s["frames"] for s in shots) == round(20.017 * 30)
    assert all(a["start_frame"] + a["frames"] == b["start_frame"] for a, b in zip(shots, shots[1:]))
    assert all(1.5 <= s["frames"] / 30 <= 4.5 for s in shots if s["beat"] != 1)


def test_fixed_beats():
    beats = vc.fixed_beats(10.0, 3.0)
    assert len(beats) == 3 and beats[-1]["end"] == 10.0


def test_assign_prefers_similar_footage():
    samples = [(0, 0.0), (0, 2.0), (1, 0.0), (1, 2.0)]
    sims = [[0.1, 0.1, 0.3, 0.2], [0.35, 0.1, 0.1, 0.1]]
    shots = [{"beat": 0, "frames": 60}, {"beat": 1, "frames": 60}]
    picks = vc.assign_shots(shots, samples, sims, [10.0, 10.0], 30)
    assert [p["clip"] for p in picks] == [1, 0]


def test_assign_rotates_without_similarity_and_clamps_in_point():
    samples = [(0, 0.0), (0, 2.0), (0, 4.0), (1, 0.0), (1, 2.0)]
    shots = [{"beat": 0, "frames": 90} for _ in range(4)]
    picks = vc.assign_shots(shots, samples, [[0.0] * 5], [6.0, 3.0], 30)
    assert [p["clip"] for p in picks] == [0, 1, 0, 1]
    for p in picks:
        dur = 3.0
        assert 0.0 <= p["in"] <= max(0.0, [6.0, 3.0][p["clip"]] - dur) + 1e-9
    assert picks[0]["in"] != picks[2]["in"]  # does not reuse the same moment


def test_assign_without_footage_errors():
    with pytest.raises(AutocutError):
        vc.assign_shots([{"beat": 0, "frames": 10}], [], [[]], [], 30)
