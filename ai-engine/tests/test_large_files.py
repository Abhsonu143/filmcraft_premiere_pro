"""Long recordings are read in pieces: frames one at a time, sound in blocks, speech in 10-minute parts."""

import json
import subprocess

import numpy as np
import pytest

import autocut as ac
import footage as fo
import gameplay_plan as gp
import montage as mt


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("big")
    p = d / "c.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=12",
                    "-f", "lavfi", "-i", "sine=f=440:d=12", "-g", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(p)], check=True)
    return str(p)


def test_audio_in_blocks_matches_the_whole(clip):
    blocks = list(fo.iter_audio(clip, 8000, block_s=2.0))
    assert len(blocks) >= 6 and all(len(b) <= 16000 for b in blocks)
    whole = np.concatenate(blocks)
    rms = mt.rms_blocks(clip, 8000, 4000)
    n = len(whole) // 4000
    assert len(rms) == n and np.allclose(rms, np.sqrt((whole[: n * 4000].reshape(n, 4000) ** 2).mean(axis=1)), atol=1e-6)


def test_frames_stream_and_keyframes_sample(clip):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", clip, "-an", "-vf", "fps=5,scale=64:36", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    n = sum(1 for _ in fo.iter_frames(cmd, 64, 36))
    assert 58 <= n <= 61
    k = list(fo.iter_keyframes_rgb(clip, 2.0))
    assert 5 <= len(k) <= 7 and k[0].shape == (224, 224, 3)  # one every 2 s over 12 s


def test_footage_analysis_streams(clip):
    a = fo.analyse(clip)
    assert 58 <= len(a["motion"]) <= 61 and len(a["dx"]) == len(a["motion"]) == len(a["bright"])


def test_long_speech_goes_in_pieces(monkeypatch, tmp_path):
    calls = []

    class FakeIso:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def run(self, fn, wav, model, prompt, language):
            calls.append(language)
            return {"language": "en", "segments": [{"start": 1.0, "end": 2.0, "text": "hi"}], "words": [{"word": "hi", "start": 1.0, "end": 2.0}]}

    class Info:
        duration = 2500.0

    monkeypatch.setattr(ac, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ac, "probe", lambda p: Info())
    monkeypatch.setattr(ac, "extract_audio", lambda *a, **k: None)
    monkeypatch.setattr(ac, "Isolated", FakeIso)
    monkeypatch.setattr(ac, "file_key", lambda *a: "k")
    r = ac.transcribe("x.mp4")
    assert len(calls) == 5 and calls[0] is None and calls[1:] == ["en"] * 4  # (the first piece decides the language)
    assert [w["start"] for w in r["words"]] == [1.0, 601.0, 1201.0, 1801.0, 2401.0]


def test_planner_inputs_moments_and_plans(tmp_path):
    a = {"source": str(tmp_path / "r1.mov"), "name": "r1", "duration": 300, "complete": True,
         "loud": [0.01] * 100 + [0.5] * 30 + [0.01] * 170, "keyframes": [{"t": t, "motion": (40 if 100 <= t < 130 else 1), "cut": False} for t in range(0, 300, 2)],
         "dialogue": [{"start": 200, "end": 205, "text": "Hana: welcome"}, {"start": 207, "end": 230, "text": "Atsu: thank you"}],
         "tags": [{"t": t, "top": ["combat" if 100 <= t < 130 else "scenic"], "sim": [0.3]} for t in range(0, 300, 4)]}
    (tmp_path / "r1.ei-analysis.json").write_text(json.dumps(a))
    (tmp_path / "r1.ei.srt").write_text("1\n00:00:01,000 --> 00:00:02,500\nHello there\n\n2\n00:01:00,000 --> 00:01:03,000\nBye\n")
    ents = gp.resolve_inputs([str(tmp_path / "r1.ei-analysis.json"), str(tmp_path / "r1.ei.srt")])
    assert len(ents) == 1 and ents[0]["analysis"].name == "r1.ei-analysis.json"  # (its .json and .srt: one recording)
    (tmp_path / "r2.ei.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n")
    two = gp.resolve_inputs([str(tmp_path / "r1.ei.srt"), str(tmp_path / "r2.ei.srt"), str(tmp_path / "r1.ei-analysis.json")])
    assert [e["srt"].name for e in two] == ["r1.ei.srt", "r2.ei.srt"]
    assert [s["text"] for s in gp.srt_segments(tmp_path / "r1.ei.srt")] == ["Hello there", "Bye"]
    m = gp.moments(a)
    assert m["action"] and 85 <= m["action"][0]["start"] <= 130
    assert m["story"] and m["story"][0]["start"] == 200
    d = gp.digest(1, a, 60)
    assert "R1" in d and "Hana: welcome" in d and "combat" in d
    plans = gp.clean_plans([{"title": "Boss fight", "platform": "shorts", "edit_tool": "clips", "parts": [{"recording": "R1", "start": "0:01:40", "end": "0:02:10"}]},
                            {"title": "Bad", "parts": [{"recording": "R9", "start": "0:00:00", "end": "0:00:10"}]},
                            {"title": "Past the end", "parts": [{"recording": "R1", "start": "0:09:00", "end": "0:10:00"}]}], [a])
    assert [p["title"] for p in plans] == ["Boss fight"] and plans[0]["parts"][0]["start"] == 100.0 and plans[0]["total_s"] == 30.0
    # a plan for a platform that was not asked for is dropped
    long = {"title": "Story part 1", "platform": "youtube", "parts": [{"recording": "R1", "start": "0:00:00", "end": "0:05:00"}]}
    assert [p["title"] for p in gp.clean_plans([long], [a])] == ["Story part 1"]
    assert gp.clean_plans([long], [a], ["shorts", "reels"]) == []
    assert "shorts|reels" in gp.plan_prompt("G", {}, [], ["shorts", "reels"], 3, "", []) and "youtube|" not in gp.plan_prompt("G", {}, [], ["shorts", "reels"], 3, "", [])


def test_model_workers_keep_mlx_cache_small(monkeypatch):
    calls = []
    monkeypatch.setattr(ac, "_mlx_tidy", lambda start: calls.append(start))
    assert ac._isolated_call(lambda a, b: a + b, 2, 3) == 5
    with pytest.raises(ValueError):
        ac._isolated_call(lambda: (_ for _ in ()).throw(ValueError("x")))
    assert calls == [True, False, True, False]  # capped before, emptied after, even when the model fails
