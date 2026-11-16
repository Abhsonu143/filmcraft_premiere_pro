"""Your camera as the presenter, step 1: the recording session and its page server."""

import json
import threading
import urllib.error
import urllib.request
import wave

import pytest

import presenter as pr
from autocut import AutocutError


def topic(tmp_path, moved=False):
    """A generated video's files: the video (a stand-in), its voice and its lines."""
    video = tmp_path / "news.mp4"
    video.write_bytes(b"video")
    with wave.open(str(tmp_path / "news.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"".join(int(k % 100).to_bytes(2, "little") for k in range(8000 * 6)))  # 6 s
    segs = [{"start": 0.0, "end": 2.0, "text": "First line."}, {"start": 2.5, "end": 5.5, "text": "Second line!", "emotion": "Joy"}]
    (tmp_path / "news.narration.json").write_text(json.dumps({"segments": segs}))
    if moved:  # the edit put the second line first (a cold open), then both
        lines = [{"start": 0.0, "end": 3.0, "text": "Second line!"}, {"start": 3.5, "end": 5.5, "text": "First line."}, {"start": 6.0, "end": 9.0, "text": "Second line!"}]
        (tmp_path / "news.narrate.json").write_text(json.dumps({"voice": str(tmp_path / "news.wav"), "lines": lines}))
    return video


def test_lines_come_from_the_video_s_own_files(tmp_path):
    voice, lines = pr.lines_of(topic(tmp_path))
    assert voice.name == "news.wav" and [(ln["start"], ln["src"], ln["text"]) for ln in lines] == [(0.0, 0.0, "First line."), (2.5, 2.5, "Second line!")]
    _, moved = pr.lines_of(topic(tmp_path, moved=True))
    assert [(ln["index"], ln["start"], ln["src"]) for ln in moved] == [(0, 0.0, 2.5), (1, 3.5, 0.0), (2, 6.0, 2.5)]
    bare = tmp_path / "bare.mp4"
    bare.write_bytes(b"x")
    with pytest.raises(AutocutError):
        pr.lines_of(bare)
    with pytest.raises(AutocutError):
        pr.Session(tmp_path / "missing.mp4")


def test_takes_are_numbered_kept_and_remembered(tmp_path):
    s = pr.Session(topic(tmp_path))
    assert s.recorded_until() == 0.0
    t1 = s.add_take("1", b"take one", "webm", 3.2)
    assert t1["file"] == "line-002-t1.webm" and s.recorded_until() == 0.0  # line 1 is still missing
    s.add_take("0", b"first", "mp4")
    assert s.recorded_until() == 5.5
    t2 = s.add_take("1", b"take two", "webm")
    assert t2["file"] == "line-002-t2.webm" and (s.dir / "takes" / "line-002-t1.webm").read_bytes() == b"take one"  # the old take is not deleted
    s.add_take("idle", b"look", "webm", 10)
    assert s.take_path("1").name == "line-002-t2.webm" and s.take_path("idle").name == "idle-t1.webm"
    for bad in (("9", b"x", "webm"), ("x", b"x", "webm"), ("0", b"", "webm"), ("0", b"x", "exe")):
        with pytest.raises(AutocutError):
            s.add_take(*bad)
    again = pr.Session(tmp_path / "news.mp4")  # opened another day
    st = again.state()
    assert [ln["take"]["file"] for ln in st["lines"]] == ["line-001-t1.mp4", "line-002-t2.webm"] and st["idle"]["file"] == "idle-t1.webm"
    again.forget("1")
    assert pr.Session(tmp_path / "news.mp4").state()["lines"][1]["take"] is None and (again.dir / "takes" / "line-002-t2.webm").is_file()


def test_a_take_follows_its_sentence_when_the_video_is_made_again(tmp_path):
    s = pr.Session(topic(tmp_path))
    s.add_take("1", b"second", "webm")
    topic(tmp_path, moved=True)  # the same sentences, now in another order
    st = pr.Session(tmp_path / "news.mp4").state()
    assert [bool(ln["take"]) for ln in st["lines"]] == [False, False, True]  # "Second line!" (its last place keeps the take)


def test_the_voice_of_one_line(tmp_path):
    s = pr.Session(topic(tmp_path, moved=True))
    import io

    with wave.open(io.BytesIO(s.line_audio("1")), "rb") as w:  # "First line.": 2 s from the start of the voice
        assert w.getframerate() == 8000 and w.getnframes() == 16000 and w.readframes(2) == (0).to_bytes(2, "little") + (1).to_bytes(2, "little")
    with wave.open(io.BytesIO(s.line_audio("0")), "rb") as w:  # "Second line!": from 2.5 s
        assert w.getnframes() == 24000 and w.readframes(1) == (20000 % 100).to_bytes(2, "little")


def test_the_page_server(tmp_path):
    video = topic(tmp_path)
    got, stop = {}, threading.Event()
    th = threading.Thread(target=lambda: got.update(pr.serve(str(video), open_browser=False, ready=lambda u: got.update(url=u), stop=stop)), daemon=True)
    th.start()
    for _ in range(100):
        if "url" in got:
            break
        stop.wait(0.05)
    base, key = got["url"].split("/?k=")

    def call(path, data=None, extra=""):
        req = urllib.request.Request(f"{base}{path}?k={key}{extra}", data=data, method="POST" if data is not None else "GET")
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.headers.get("Content-Type"), r.read()

    assert b"Record your lines" in call("/")[2]
    assert [ln["text"] for ln in json.loads(call("/api/session")[2])["lines"]] == ["First line.", "Second line!"]
    assert call("/api/audio/0")[1] == "audio/wav"
    st = json.loads(call("/api/take/0", b"webm bytes", "&ext=webm&seconds=2.1")[2])
    assert st["take"]["file"] == "line-001-t1.webm" and st["session"]["recordedUntil"] == 2.0
    assert call("/api/take/0")[2] == b"webm bytes"
    for path, data, code in (("/api/take/7", b"x", 400), ("/api/take/1", None, 404), ("/nope", None, 404)):
        with pytest.raises(urllib.error.HTTPError) as e:
            call(path, data)
        assert e.value.code == code
    with pytest.raises(urllib.error.HTTPError) as e:  # without the page's key nothing is served
        urllib.request.urlopen(f"{base}/api/session?k=wrong", timeout=10)
    assert e.value.code == 403
    call("/api/done", b"")
    th.join(10)
    assert not th.is_alive() and got["recorded"] == 1 and got["lines"] == 2


def test_a_sentence_is_shown_ten_words_at_a_time():
    assert pr.split_words("Short one.") == ["Short one."] and pr.split_words("   ") == []
    long = "Story one. Guys, GTA Six is finally days away. According to Rockstar, it drops this Thursday, nineteenth November."
    parts = pr.split_words(long)
    assert parts == ["Story one. Guys, GTA Six is finally days away.", "According to Rockstar, it drops this Thursday, nineteenth November."]
    assert all(len(p.split()) <= 10 for p in parts) and " ".join(parts) == long
    run_on = " ".join(f"w{k}" for k in range(23))  # no punctuation at all: equal pieces
    assert [len(p.split()) for p in pr.split_words(run_on)] == [8, 8, 7]
    for limit in (4, 5, 7):
        got = pr.split_words(long, limit)
        assert max(len(p.split()) for p in got) <= limit and " ".join(got) == long


def test_parts_are_cut_where_the_voice_pauses():
    line = {"src": 10.0, "start": 20.0, "end": 26.0}
    parts = ["aaaa bbbb", "cccc dddd"]  # by letters the cut would be at 13.0
    assert pr.part_times(line, parts, []) == [(10.0, 13.0), (13.0, 16.0)]
    heard = [{"start": 10.0, "end": 11.0}, {"start": 11.1, "end": 12.4}, {"start": 12.8, "end": 14.0}, {"start": 14.1, "end": 16.0}]
    (a, cut), (cut2, b) = pr.part_times(line, parts, heard)
    assert (a, b) == (10.0, 16.0) and cut == cut2 == pytest.approx(12.6)  # the breath between two words


def test_long_lines_become_parts_in_the_session(tmp_path):
    video = topic(tmp_path)
    segs = [{"start": 0.0, "end": 6.0, "text": "One two three four five six, seven eight nine ten eleven twelve."}]
    (tmp_path / "news.narration.json").write_text(json.dumps({"segments": segs}))
    s = pr.Session(video, listen=False)
    assert [(ln["index"], ln["line"], ln["part"], ln["parts"], ln["text"]) for ln in s.lines] == [
        (0, 0, 1, 2, "One two three four five six,"), (1, 0, 2, 2, "seven eight nine ten eleven twelve.")]
    assert s.lines[0]["start"] == 0.0 and s.lines[0]["end"] == s.lines[1]["start"] and s.lines[1]["end"] == 6.0
    s.add_take("0", b"x", "webm")
    assert s.recorded_until() == s.lines[0]["end"]


def test_framing_advice():
    good = (0.4, 0.25, 0.6, 0.6)
    assert pr.framing_advice([good, good, None]) is None and pr.framing_advice([]) is None
    assert "not in the picture" in pr.framing_advice([None, None, good])
    assert "chin" in pr.framing_advice([(0.4, 0.5, 0.6, 0.9)] * 3)
    assert "top of your head" in pr.framing_advice([(0.4, 0.01, 0.6, 0.4)] * 3)
    assert "closer" in pr.framing_advice([(0.45, 0.4, 0.55, 0.5)] * 3)

