"""Continuous sections: where they start and end, which are chosen, exact shots, rendering."""

import json
import subprocess
from pathlib import Path

import numpy as np
from conftest import needs_ffmpeg

import sections as se

HZ = se.HZ


def sig_of(n, cuts=(), dark=()):
    s = {"loud": np.full(n, -25.0, np.float32), "flux": np.zeros(n, np.float32), "motion": np.full(n, 8.0, np.float32),
         "bright": np.full(n, 80.0, np.float32), "cut": np.zeros(n, np.float32)}
    for c in cuts:
        s["cut"][int(c * HZ)] = 1.0
    for a, b in dark:
        s["bright"][int(a * HZ) : int(b * HZ)] = 5.0
    return s


def sent(a, b, text="Something happened here."):
    return {"start": a, "end": b, "text": text, "words": []}


def test_dedupe_keeps_the_best_nearby():
    out = se.dedupe([(10.0, 0.5, "a"), (10.3, 0.9, "b"), (20.0, 0.4, "c")])
    assert out == [(10.3, 0.9, "b"), (20.0, 0.4, "c")]


def test_edges_never_inside_a_sentence_or_on_black():
    n = 120 * HZ
    score = np.zeros(n)
    score[60 * HZ : 90 * HZ] = 2.0  # the action builds at 60 s and settles at 90 s
    sents = [sent(20.0, 26.0), sent(28.0, 33.0, "And then?")]
    sig = sig_of(n, cuts=(15.0, 24.0, 100.0), dark=((110.0, 120.0),))
    starts, ends = se.edge_candidates(score, sig, sents, {0}, 120.0)
    for t, _, _ in starts + ends:
        assert not any(s["start"] < t < s["end"] for s in sents), t
    why_s = {round(t): w for t, _, w in starts}
    why_e = {round(t): w for t, _, w in ends}
    assert why_s.get(15) == "cut" and 24 not in why_s  # a cut during a line is not a start
    assert any(w == "topic" and 19 <= t <= 20 for t, w in why_s.items())  # just before the first word
    assert any(w == "end of a sentence" for w in why_e.values())  # "And then?" ends with a pause after it
    assert any(w == "before the action" and 50 <= t <= 60 for t, w in why_s.items())
    assert any(w == "action settled" and 88 <= t <= 100 for t, w in why_e.values() and why_e.items())
    assert why_e.get(110) == "fade to black"
    assert all(not (110 <= t < 120) for t, _, _ in starts)


def test_plan_picks_the_action_and_lets_its_length_follow_it():
    n = 600 * HZ
    score = np.full(n, -0.5)
    score[100 * HZ : 230 * HZ] = 1.5  # a 130 s fight
    score[400 * HZ : 430 * HZ] = 1.2  # a 30 s burst
    dead = np.zeros(n, bool)
    dead[300 * HZ : 360 * HZ] = True
    starts = [(float(t), 0.8, "cut") for t in range(0, 600, 5)]
    ends = [(float(t), 0.8, "cut") for t in range(0, 600, 5)]
    secs = se.plan_sections(score, dead, starts, ends, 2, 15, 180, 60)
    assert len(secs) == 2 and secs[0]["start"] < secs[1]["start"]
    a, b = secs[0]["start"], secs[0]["end"]
    assert 95 <= a <= 105 and 225 <= b <= 235, secs[0]  # the whole fight, longer than the usual 60 s
    assert 395 <= secs[1]["start"] and secs[1]["end"] <= 440 and secs[1]["end"] - secs[1]["start"] >= 15
    assert not any(s["start"] < 360 and s["end"] > 300 for s in secs)  # never the dead stretch


def test_plan_respects_lengths_and_never_overlaps():
    rng = np.random.default_rng(3)
    n = 900 * HZ
    score = rng.normal(0, 1, n)
    starts = [(float(t), rng.uniform(0.3, 1.2), "cut") for t in range(0, 900, 3)]
    ends = [(float(t), rng.uniform(0.3, 1.2), "cut") for t in range(0, 900, 3)]
    secs = se.plan_sections(score, np.zeros(n, bool), starts, ends, 6, 20, 90, 45)
    assert len(secs) == 6
    for s in secs:
        assert 20 <= s["end"] - s["start"] <= 90
    for x, y in zip(secs, secs[1:]):
        assert x["end"] <= y["start"]


def test_split_shots_back_to_back_and_exact():
    shots = se.split_shots(100.0, 165.5, [110.0, 110.2, 140.0, 165.2], 30.0, shot_max=20.0)
    assert sum(s["frames"] for s in shots) == round(65.5 * 30)
    t = 100.0
    for s in shots:
        assert abs(s["in"] - t) < 1e-3  # each shot continues exactly where the last one stopped
        assert s["frames"] <= 20 * 30 and s["frames"] >= 15
        t += s["frames"] / 30.0
    joins = [round(s["in"], 1) for s in shots[1:]]
    assert 110.0 in joins and 140.0 in joins  # split where the video cuts


def test_sharpen_edges_moves_onto_the_exact_cut(monkeypatch):
    monkeypatch.setattr(se, "exact_cut", lambda src, t, fps: t + 0.13)
    s = se.sharpen_edges("x", {"start": 10.0, "end": 70.0, "why_in": "cut", "why_out": "end of a sentence"}, 60.0, 30.0)
    assert abs(s["start"] - (10.13 + 0.5 / 30)) < 1e-3 and s["fade_in"] == 0.0 and s["fade_out"] > 0
    monkeypatch.setattr(se, "exact_cut", lambda src, t, fps: None)  # not a real cut after all: fade
    s = se.sharpen_edges("x", {"start": 10.0, "end": 70.0, "why_in": "cut", "why_out": "fade to black"}, 60.0, 30.0)
    assert s["start"] == 10.0 and s["fade_in"] > 0 and s["fade_out"] == 0.0


@needs_ffmpeg
def test_exact_cut_and_render_are_continuous(tmp_path):
    src = tmp_path / "src.mp4"
    # red for 2 s, then blue: one cut at exactly 2.0 s; a tone throughout
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x180:r=30:d=2", "-f", "lavfi", "-i",
                    "color=c=blue:s=320x180:r=30:d=4", "-f", "lavfi", "-i", "sine=f=330:d=6", "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]",
                    "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(src)], check=True)
    c = se.exact_cut(str(src), 2.1, 30.0)
    assert c is not None and abs(c - 2.0) < 0.04
    clip = {"start": 0.5, "end": 5.5, "fade_in": 0.25, "fade_out": 0.3, "shots": se.split_shots(0.5, 5.5, [2.0], 30.0, shot_max=1.5)}
    assert len(clip["shots"]) >= 4
    for s in clip["shots"]:
        s["focus"] = [0.5, 0.5]
    dst = tmp_path / "out.mp4"
    se.render_section(str(src), clip, 320, 180, 30.0, dst, None, True)
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=codec_type,nb_read_frames,duration", "-of", "json", str(dst)],
                         capture_output=True, text=True).stdout
    st = {s["codec_type"]: s for s in json.loads(out)["streams"]}
    assert int(st["video"]["nb_read_frames"]) == 150 and abs(float(st["audio"]["duration"]) - 5.0) < 0.1
    # the picture turns blue exactly once, 1.5 s in (2.0 s in the source)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(dst), "-vf", "scale=8:8", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    px = np.frombuffer(raw, np.uint8).reshape(-1, 8, 8, 3).mean(axis=(1, 2))
    blue = px[:, 2] > px[:, 0]
    first_blue = int(np.argmax(blue))
    assert abs(first_blue - 45) <= 1 and blue[first_blue:].all() and not blue[:first_blue].any()
    assert Path(dst).stat().st_size > 0


def test_held_pictures_are_trimmed_off_the_edges():
    n = 100 * HZ
    sig = sig_of(n)
    sig["motion"][:] = 10.0
    sig["motion"][int(57.0 * HZ) : int(60.0 * HZ)] = 0.2  # a loading card at the end …
    sig["motion"][int(58.5 * HZ) : int(58.9 * HZ)] = 12.0  # … that animates in for a moment
    sig["motion"][int(10.0 * HZ) : int(11.5 * HZ)] = 0.1  # a frozen first second and a half
    s = se.trim_held({"start": 10.0, "end": 60.0, "why_in": "cut", "why_out": "fade to black"}, sig, 15)
    assert abs(s["end"] - 57.2) < 0.05 and s["why_out"] == "held picture"
    assert abs(s["start"] - 11.4) < 0.05 and s["why_in"] == "after a held picture"
    calm = se.trim_held({"start": 20.0, "end": 50.0, "why_in": "cut", "why_out": "cut"}, sig, 15)
    assert (calm["start"], calm["end"]) == (20.0, 50.0)
    # never shorter than the shortest length
    short = se.trim_held({"start": 41.0, "end": 60.0, "why_in": "cut", "why_out": "fade to black"}, sig, 18)
    assert short["end"] == 60.0


def test_talking_is_never_a_held_picture():
    n = 100 * HZ
    sig = sig_of(n)
    sig["motion"][:] = 0.3  # a webcam interview hardly moves
    speech = np.zeros(n)
    speech[int(10.2 * HZ) : int(59.8 * HZ)] = 1.0
    s = se.trim_held({"start": 10.0, "end": 60.0, "why_in": "sentence", "why_out": "end of a sentence"}, sig, 15, speech=speech)
    assert (s["start"], s["end"]) == (10.0, 60.0)
