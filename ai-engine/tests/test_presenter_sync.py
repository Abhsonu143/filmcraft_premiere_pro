"""Your camera as the presenter, step 2: takes lined up with the voice and laid over the video."""

import json
import shutil
import subprocess

import numpy as np
import pytest

import presenter as pr
import presenter_sync as ps

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def words(pattern, seconds_each=0.25, lead=0.0):
    """A stand-in for speech: one 'word' per number, each its own pitch, with a short gap after it."""
    out = [np.zeros(int(ps.SR * lead))]
    for k, f in enumerate(pattern):
        n = int(ps.SR * seconds_each[k] if isinstance(seconds_each, list) else ps.SR * seconds_each)
        t = np.arange(n) / ps.SR
        out += [0.3 * np.sin(2 * np.pi * f * t) * np.hanning(n), np.zeros(int(ps.SR * 0.05))]
    return np.concatenate(out) + 1e-4 * np.random.default_rng(1).standard_normal(sum(len(o) for o in out))


def test_where_someone_speaks():
    x = words([300, 900, 500], lead=0.8)
    a, b = ps.speech_span(np.concatenate([x, np.zeros(ps.SR)]) + 1e-4 * np.random.default_rng(2).standard_normal(len(x) + ps.SR))
    assert a == pytest.approx(0.8, abs=0.1) and b == pytest.approx(0.8 + 0.9, abs=0.12)
    assert ps.speech_span(np.zeros(100)) == (0.0, 100 / ps.SR)


def test_a_take_follows_the_voice_word_by_word():
    pitches = [300, 900, 500, 1500, 700, 2000]
    voice = words(pitches, 0.2)
    take = words(pitches, [0.2, 0.2, 0.6, 0.6, 0.2, 0.2], lead=0.5)  # said slower in the middle, after half a second
    anchors, how = ps.anchors_of(take, voice)
    a = np.array(anchors)
    assert (np.diff(a[:, 1]) >= 0).all() and how["fit"] > how["even"]
    # the fourth word starts at 0.75 s in the voice and at 0.5 + 0.25 + 0.25 + 0.65 = 1.65 s in the take
    assert float(np.interp(0.80, a[:, 0], a[:, 1])) == pytest.approx(1.75, abs=0.2)
    assert a[0, 1] == pytest.approx(0.5, abs=0.1)


def test_every_frame_knows_what_to_show():
    pieces = [{"start": 1.0, "end": 2.0, "anchors": [[0.0, 0.5], [1.0, 2.5]]}, {"start": 2.2, "end": 3.0, "anchors": [[0.0, 0.2], [0.8, 1.0]]},
              {"start": 6.0, "end": 7.0, "anchors": [[0.0, 0.0], [1.0, 1.0]]}]
    plan = ps.schedule(pieces, 80, 10.0, True)
    assert plan[0][0] == -1 and plan[5][0] == -1  # the idle look before the first words
    assert plan[15] == (0, pytest.approx(1.5)) and plan[10] == (0, pytest.approx(0.5))  # twice as fast as the voice
    assert plan[20][0] == 0 and plan[21][0] == 1  # the short pause is shared
    assert plan[45][0] == -1 and plan[45][1] < plan[50][1]  # the long pause: the idle look, going on
    assert plan[58] == (2, pytest.approx(-0.2)) and plan[65] == (2, pytest.approx(0.5))
    bare = ps.schedule(pieces, 80, 10.0, False)
    assert all(p is not None for p in bare[10:]) and bare[45][0] in (1, 2)  # no idle look: the takes cover the pause
    assert ps.schedule([], 5, 10.0, False) == [None] * 5


def test_the_face_stays_in_place():
    x, y, side = ps.crop_of([0.5, 0.4, 0.3], 0.3, 1280, 720)
    assert side == 560 and x == 360 and 0 <= y <= 720 - side
    assert ps.crop_of(None, 0.3, 1280, 720)[2] == 560 and ps.crop_of([0.02, 0.02, 0.9], 0.9, 1280, 720) == (0, 0, 720)


@needs_ffmpeg
def test_takes_are_laid_over_the_video(tmp_path):
    video = tmp_path / "news.mp4"
    run = lambda *a: subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *a], check=True)  # noqa: E731
    run("-f", "lavfi", "-i", "color=c=blue:s=320x568:r=30:d=4", "-f", "lavfi", "-i", "sine=f=300:d=4", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(video))
    run("-f", "lavfi", "-i", "sine=f=500:d=4", "-ar", "16000", "-ac", "1", str(tmp_path / "news.wav"))
    segs = [{"start": 0.2, "end": 1.6, "text": "First line."}, {"start": 2.0, "end": 3.6, "text": "Second line!"}]
    (tmp_path / "news.narration.json").write_text(json.dumps({"segments": segs}))
    take = tmp_path / "take.webm"
    run("-f", "lavfi", "-i", "color=c=red:s=320x240:r=30:d=2", "-f", "lavfi", "-i", "sine=f=700:d=2", "-c:v", "libvpx", "-b:v", "200k", "-c:a", "libopus", "-shortest", str(take))
    s = pr.Session(video, listen=False)
    with pytest.raises(pr.AutocutError):
        ps.render(str(video))
    s.add_take("0", take.read_bytes(), "webm", 2)
    r = ps.render(str(video))  # only the first line is recorded: a preview up to its end
    assert r["outputs"][0].endswith("news.presenter/preview.mp4") and r["seconds"] == pytest.approx(1.6, abs=0.05) and not r["whole"]
    s.add_take("1", take.read_bytes(), "webm", 2)
    plan = json.loads((s.dir / "plan.json").read_text())
    plan["pieces"][0]["anchors"] = [[0.0, 0.3], [1.4, 1.7]]  # an edit of the plan is kept
    (s.dir / "plan.json").write_text(json.dumps(plan))
    r = ps.render(str(video), look={"shape": "rounded", "position": "top-left", "size": None})
    out = tmp_path / "news - presenter.mp4"
    assert r["whole"] and r["outputs"] == [str(out)] and video.stat().st_size > 0
    plan = json.loads((s.dir / "plan.json").read_text())
    assert plan["pieces"][0]["anchors"] == [[0.0, 0.3], [1.4, 1.7]] and plan["look"]["shape"] == "rounded" and plan["look"]["size"] == 0.4 and len(plan["pieces"]) == 2
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration", "-of", "json", str(out)], capture_output=True, text=True)
    d = json.loads(probe.stdout)
    assert {st["codec_type"] for st in d["streams"]} == {"video", "audio"} and float(d["format"]["duration"]) == pytest.approx(4.0, abs=0.15)
    # the take (red) is where it was asked for, the video (blue) elsewhere
    px = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", "1", "-i", str(out), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    img = np.frombuffer(px, np.uint8).reshape(568, 320, 3)
    assert img[104, 104, 0] > 150 and img[104, 104, 2] < 100 and img[400, 200, 2] > 150 and img[400, 200, 0] < 100
    with pytest.raises(pr.AutocutError):
        ps.render(str(video), out=str(video))
