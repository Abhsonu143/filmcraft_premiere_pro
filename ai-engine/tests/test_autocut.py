import autocut as ac

SILENCE_LOG = """
[silencedetect @ 0x1] silence_start: 1.5
[silencedetect @ 0x1] silence_end: 3.25 | silence_duration: 1.75
[silencedetect @ 0x1] silence_start: -0.01
[silencedetect @ 0x1] silence_end: 7 | silence_duration: 1
[silencedetect @ 0x1] silence_start: 9.5
"""


def test_parse_silencedetect_handles_negative_and_open_end():
    assert ac.parse_silencedetect(SILENCE_LOG, 10.0) == [(1.5, 3.25), (0.0, 7.0), (9.5, 10.0)]


def test_parse_silencedetect_empty():
    assert ac.parse_silencedetect("", 5.0) == []


def test_speech_segments_inverts_pads_and_merges():
    segs = ac.speech_segments([(2.0, 4.0), (4.1, 6.0)], 10.0, pad=0.1, min_segment=0.3, merge_gap=0.15)
    assert segs == [(0.0, 2.1), (5.9, 10.0)]  # the 0.1 s blip at 4.0-4.1 (padded to 0.3) merges away


def test_speech_segments_drops_blips_and_handles_all_silence():
    assert ac.speech_segments([(0.0, 5.0), (5.1, 10.0)], 10.0, pad=0.0, min_segment=0.3) == []
    assert ac.speech_segments([], 0.0) == []
    assert ac.speech_segments([], 4.0, pad=0.1) == [(0.0, 4.0)]


def test_snap_segments_on_frame_grid_without_overlap():
    segs = ac.snap_segments([(0.01, 1.02), (1.03, 2.0), (5.0, 5.01)], 30)
    for s, e in segs:
        assert abs(s * 30 - round(s * 30)) < 1e-9 and abs(e * 30 - round(e * 30)) < 1e-9
    assert all(a[1] <= b[0] for a, b in zip(segs, segs[1:]))
    assert segs[-1] == (5.0, 5.0 + 0) or len(segs) == 2  # 10 ms rounds to zero frames and is dropped


def test_map_to_output_and_remap_words():
    segs = [(1.0, 2.0), (5.0, 7.0)]
    assert ac.map_to_output(1.5, segs) == 0.5
    assert ac.map_to_output(3.0, segs) is None
    assert ac.map_to_output(6.0, segs) == 2.0
    words = [
        {"word": "a", "start": 1.2, "end": 1.4},
        {"word": "gone", "start": 3.0, "end": 3.5},
        {"word": "edge", "start": 4.8, "end": 5.3},
    ]
    out = ac.remap_words(words, segs)
    assert [w["word"] for w in out] == ["a", "edge"]
    assert abs(out[0]["start"] - 0.2) < 1e-9
    assert abs(out[1]["start"] - 1.0) < 1e-9 and abs(out[1]["end"] - 1.3) < 1e-9


def test_srt_time_format():
    assert ac.srt_time(0) == "00:00:00,000"
    assert ac.srt_time(3661.5) == "01:01:01,500"
    assert ac.srt_time(-1) == "00:00:00,000"


def test_group_words_breaks_on_punctuation_size_and_pause():
    w = lambda t, s, e: {"word": t, "start": s, "end": e}
    words = [w(" Hello", 0, 0.3), w(" world.", 0.3, 0.6), w(" one", 0.7, 0.8), w(" two", 0.8, 0.9), w(" three", 0.9, 1.0), w(" later", 3.0, 3.2)]
    cues = ac.group_words(words, max_words=2)
    assert [c["text"] for c in cues] == ["Hello world.", "one two", "three", "later"]
    assert all(a["end"] <= b["start"] for a, b in zip(cues, cues[1:]))


def test_to_srt_numbering():
    srt = ac.to_srt([{"start": 0, "end": 1, "text": "hi"}, {"start": 1, "end": 2, "text": "yo"}])
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
    assert "\n2\n00:00:01,000 --> 00:00:02,000\nyo\n" in srt


def test_autocut_graph_selects_same_ranges_for_audio_and_video():
    plan = {"fps": 30, "width": 1080, "height": 1920, "captions": None}
    g = ac.build_autocut_graph(plan, [(0.0, 1.0), (2.0, 3.0)], None, has_music=False)
    v, a = g.split(";\n")[:2]
    sel = "between(t,-0.016667,0.983333)+between(t,1.983333,2.983333)"
    assert sel in v and sel in a
    assert "asetnsamples=n=1600" in a  # one audio packet per video frame at 48 kHz / 30 fps
    assert "crop=1080:1920" in v


def test_rate_arg():
    assert ac.rate_arg(30) == "30"
    assert ac.rate_arg(29.97) == "2997/100" or ac.rate_arg(30000 / 1001) == "30000/1001"
    assert ac.rate_arg(30000 / 1001) == "30000/1001"
