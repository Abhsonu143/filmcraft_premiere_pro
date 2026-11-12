import json
import subprocess

import numpy as np
import pytest
from conftest import needs_ffmpeg

import autocut as ac
import ei_engine
import extract_audio as ea
import highlights as hl
import llm


def W(word, s, e):
    return {"word": word, "start": s, "end": e}


def words_from(text, start=0.0, gap=0.05, dur=0.3):
    out, t = [], start
    for w in text.split():
        out.append(W(" " + w, t, t + dur))
        t += dur + gap
    return out


def test_sentences_fillers_and_pauses():
    ws = words_from("So um this is the first point. And here, uh, is the second?")
    sents = hl.sentences_from_words(ws)
    assert [s["text"].strip()[-1] for s in sents] == [".", "?"]
    spans = hl.keep_spans(sents[0])
    kept = [w for w in sents[0]["words"] if any(a <= w["start"] < b for a, b in spans)]
    assert all(hl.norm_word(w["word"]) not in hl.FILLERS for w in kept)
    assert len(spans) >= 2  # the filler leaves a cut
    pause = sents[0]["words"][:2] + [W(" later", 10.0, 10.3)]
    assert len(hl.keep_spans({"words": pause})) == 2  # a long pause is cut out


def test_topic_boundaries_find_the_change():
    a = ["The drone battery lasts twenty minutes.", "Wind drains the drone battery faster.", "Land the drone with battery left.",
         "Drone rules say never fly over crowds.", "Check the drone propellers before flight."]
    b = ["The waterfall was hidden in the forest.", "We walked to the waterfall for hours.", "Kids jumped into the waterfall pool.",
         "The forest path to the waterfall was muddy.", "That waterfall was the most beautiful."]
    t, sents = 0.0, []
    for k, s in enumerate(a + b):
        ws = words_from(s, t)
        t = ws[-1]["end"] + (1.5 if k == len(a) - 1 else 0.3)
        sents += hl.sentences_from_words(ws)
    bounds = hl.topic_boundaries(sents)
    assert 5 in bounds, bounds


def make_sents(n, dur=5.0, score=0.5):
    out = []
    for i in range(n):
        s = {"start": i * dur, "end": (i + 1) * dur - 0.2, "text": f"Sentence {i}.", "words": [W(" x", i * dur, (i + 1) * dur - 0.2)], "score": score}
        s["spans"] = [(s["start"], s["end"])]
        s["dur"] = s["end"] - s["start"]
        out.append(s)
    return out


def test_plan_clips_lengths_and_no_overlap():
    sents = make_sents(60)
    for i in (5, 22, 41):
        sents[i]["score"] = 1.2
    clips = hl.plan_clips(sents, [0, 15, 30, 45], 4, 48, 10, 60)
    assert len(clips) == 4
    used = [set(range(min(c["idx"]), max(c["idx"]) + 1)) for c in clips]
    for i in range(len(used)):
        for j in range(i + 1, len(used)):
            assert not used[i] & used[j]
    assert all(10 <= c["dur"] <= 60 for c in clips)
    assert sum(1 for c in clips if 40 <= c["dur"] <= 56) >= 3  # mostly near the target


def test_pieces_teaser_and_caption_remap():
    sents = make_sents(8, 4.0)
    sents[6]["score"] = 1.3
    pieces, teaser = hl.pieces_for(sents, list(range(8)), hook_first=True)
    assert teaser == 6 and pieces[0][0] == pytest.approx(sents[6]["start"], abs=0.1)
    ws = [W(" hello", 1.0, 1.4), W(" um", 1.5, 1.7), W(" world", 9.0, 9.5)]
    out = hl.remap_to_pieces(ws, [(0.8, 2.0), (8.5, 10.0)])
    assert [w["word"].strip() for w in out] == ["hello", "world"]
    assert out[1]["start"] == pytest.approx(1.2 + 0.5)
    ass = hl.build_ass(out, "My Title", 1080, 1920, 2.7)
    assert "Title,,0,0,0,,My Title" in ass and "HELLO" in ass


def test_model_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("EI_SETTINGS", str(tmp_path / "s.json"))
    assert llm.resolve(None) == llm.DEFAULT
    llm.save_default("qwen2.5-1.5b")
    assert llm.resolve("default") == "qwen2.5-1.5b"
    assert llm.resolve("qwen2.5-7b") == "qwen2.5-7b"
    llm.save_default("none")
    assert llm.resolve(None) is None
    with pytest.raises(ac.AutocutError):
        llm.save_default("gpt-9")
    assert llm.clean_title('Title: "The Hidden Waterfall"\nmore') == "The Hidden Waterfall"


def ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@needs_ffmpeg
def test_action_mode_builds_multi_piece_clips(tmp_path):
    """3 minutes with three loud, busy stretches: each clip collects pieces from one of them."""
    src = tmp_path / "game.mp4"
    busy = "+".join(f"between(t,{a},{a + 20})" for a in (20, 80, 140))
    ff("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30,format=yuv420p", "-f", "lavfi",
       "-i", f"aevalsrc='(0.03+0.6*({busy}))*sin(2*PI*90*t)*exp(-12*mod(t,0.33))':s=44100:d=180",
       "-t", "180", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(src))
    clips = hl.action_clips(str(src), 180.0, [], 3, 18, 10, 25)
    assert len(clips) == 3
    spans = sorted(p for c in clips for p in c["pieces"])
    assert all(b <= c for (_, b), (c, _) in zip(spans, spans[1:]))  # never reuse material
    for c in clips:
        d = sum(b - a for a, b in c["pieces"])
        assert 10 <= d <= 25 and len(c["pieces"]) >= 2
        mid = np.mean([(a + b) / 2 for a, b in c["pieces"]])
        assert any(a - 15 <= mid <= a + 35 for a in (20, 80, 140)), c["pieces"]


def run_engine(capsys, argv):
    assert ei_engine.main(argv) == 0
    line = [l for l in capsys.readouterr().out.splitlines() if l.startswith("EI_RESULT ")][-1]
    res = json.loads(line[len("EI_RESULT "):])
    assert res["ok"], res
    return res


@needs_ffmpeg
def test_extract_audio_formats(tmp_path, capsys):
    v = tmp_path / "v.mp4"
    ff("-f", "lavfi", "-i", "color=c=black:s=160x120:r=10", "-f", "lavfi", "-i", "sine=f=440:d=4", "-t", "4", "-c:v", "libx264", "-c:a", "aac", str(v))
    for fmt, codec in [("mp3", "mp3"), ("wav", "pcm_s16le"), ("flac", "flac"), ("alac", "alac"), ("opus", "opus")]:
        res = run_engine(capsys, ["audio", str(v), "--format", fmt, "--out-dir", str(tmp_path / "out")])
        out = res["outputs"][0]
        info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name:format=duration", "-of", "json", out], capture_output=True, text=True).stdout)
        assert info["streams"][0]["codec_name"] == codec
        assert abs(float(info["format"]["duration"]) - 4) < 0.1
    res = run_engine(capsys, ["audio", str(tmp_path), "--format", "m4a", "--quality", "original", "--start", "1", "--end", "3", "--out-dir", str(tmp_path / "out")])
    assert len(res["outputs"]) == 1 and res["outputs"][0].endswith(".m4a")  # folder input, copy, range
    assert v.is_file()
    with pytest.raises(ac.AutocutError):
        ea.extract([str(v)], str(tmp_path), "wav", start=3, end=2)


@needs_ffmpeg
def test_clips_from_long_talk_end_to_end(tmp_path, capsys):
    """A spoken video (macOS `say`) becomes clips assembled from several pieces, with captions."""
    import shutil

    if shutil.which("say") is None:
        pytest.skip("needs macOS say")
    text = ("Let's talk about drones. The battery lasts twenty minutes. Wind drains it faster, so land early. Never fly over crowds. "
            "Always check the propellers. The number one tip is to watch the wind. Here is the secret of good footage. Fly slowly and smoothly. ") * 3
    aiff = tmp_path / "t.aiff"
    subprocess.run(["say", "-o", str(aiff), text], check=True)
    dur = ac.probe(str(aiff)).duration
    src = tmp_path / "talk.mp4"
    ff("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30", "-i", str(aiff), "-t", f"{dur:.2f}", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(src))
    res = run_engine(capsys, ["clips", str(src), "--out-dir", str(tmp_path / "clips"), "--count", "2", "--min", "10", "--target", "20", "--max", "30", "--llm", "none", "--content", "talk"])
    assert len(res["outputs"]) == 2
    for c, out in zip(res["clips"], res["outputs"]):
        info = ac.probe(out)
        assert (info.width, info.height) == (1080, 1920) and info.has_audio
        assert 10 <= info.duration <= 30.5 and c["pieces"] >= 2
        assert out.endswith(".mp4") and (tmp_path / "clips" / (out.rsplit("/", 1)[1][:-4] + ".srt")).is_file()
