import json
import subprocess

import numpy as np
import pytest
from conftest import needs_ffmpeg

import autocut as ac
import ei_engine
import montage as m
import split_video as sv


def test_split_points_fold_short_tail():
    assert sv.split_points(600, 60) == [60 * i for i in range(1, 10)]
    assert sv.split_points(600.5, 60) == [60 * i for i in range(1, 10)]  # 0.5 s tail joins the last clip
    assert sv.split_points(605, 60) == [60 * i for i in range(1, 11)]
    assert sv.split_points(30, 60) == []
    assert sv.segment_count(0, 60) == 0 and sv.segment_count(10, 0) == 0


def test_beat_tracker_on_synthetic_clicks():
    sr, bpm = m.SR, 120
    t = np.arange(int(sr * 12)) / sr
    y = (np.sin(2 * np.pi * 60 * t) * np.exp(-30 * np.mod(t, 60 / bpm))).astype(np.float32)
    env = m.onset_envelope(y)
    period = m.estimate_period(env)
    assert abs(60 * (sr / m.HOP) / period - bpm) < 2
    beats = np.array(m.track_beats(env, period)) / (sr / m.HOP) + (1024 + m.HOP) / sr
    assert len(beats) >= 20
    assert np.abs(beats - np.round(beats * 2) / 2).max() < 0.03


def test_cuts_on_beats_with_exact_total_and_no_slivers():
    beats = [i * 0.5 for i in range(1, 60)]
    bounds, energy = m.cut_frames_from_beats(beats, [1.0] * len(beats), 20.0, 30, 2.0)
    assert bounds[0] == 0 and bounds[-1] == 600
    assert all(b - a >= 20 for a, b in zip(bounds, bounds[1:]))
    assert all((b % 15) == 0 for b in bounds)  # every cut on a beat (0.5 s = 15 frames)
    assert len(energy) == len(bounds) - 1
    no_beats, _ = m.cut_frames_from_beats([], [], 10.0, 30, 2.0)
    assert no_beats == [0, 60, 120, 180, 240, 300]


def clip(dur, motion, bright):
    n = int(dur * m.ANALYSIS_FPS)
    return {"duration": dur, "motion": [motion] * n, "brightness": [bright] * n}


def test_choose_footage_avoids_black_and_reuse_and_matches_energy():
    clips = [clip(10, 5.0, 120), clip(10, 0.5, 120), clip(10, 3.0, 5)]  # busy, calm, black
    shots = [{"frames": 60, "energy": 1.0}, {"frames": 60, "energy": 0.0}, {"frames": 60, "energy": 1.0}, {"frames": 60, "energy": 0.0}]
    picks = m.choose_footage(shots, clips, 30)
    assert [p["clip"] for p in picks] == [0, 1, 0, 1]
    assert picks[0]["in"] != picks[2]["in"] and picks[1]["in"] != picks[3]["in"]
    used = {(p["clip"], p["in"]) for p in picks}
    assert len(used) == 4


def test_choose_footage_uses_visual_similarity():
    clips = [clip(10, 1.0, 120), clip(10, 1.0, 120)]
    samples = [(0, 0.0), (0, 2.0), (1, 0.0), (1, 2.0)]
    sims = [[0.1, 0.1, 0.9, 0.2]]
    picks = m.choose_footage([{"frames": 30, "energy": 0.5}], clips, 30, sims=sims, samples=samples)
    assert picks[0]["clip"] == 1


def test_choose_footage_requires_clips():
    with pytest.raises(ac.AutocutError):
        m.choose_footage([{"frames": 30}], [], 30)


def ff(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture
def material(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    for i, src in enumerate(["testsrc2", "mandelbrot", "rgbtestsrc", "smptehdbars"]):
        ff("-f", "lavfi", "-i", f"{src}=size=640x360:rate=30", "-t", str(3 + i), "-an", "-c:v", "h264_videotoolbox", "-b:v", "1M", str(clips / f"c{i}.mp4"))
    music = tmp_path / "beat.wav"
    ff("-f", "lavfi", "-i", "aevalsrc='sin(2*PI*60*t)*exp(-30*mod(t,0.5))':s=44100:d=20", str(music))
    return clips, music


def run_engine(capsys, argv):
    assert ei_engine.main(argv) == 0
    line = [l for l in capsys.readouterr().out.splitlines() if l.startswith("EI_RESULT ")][-1]
    res = json.loads(line[len("EI_RESULT "):])
    assert res["ok"], res
    return res


@needs_ffmpeg
def test_engine_music_square_edit(material, tmp_path, capsys):
    clips, music = material
    out = tmp_path / "out" / "edit.mp4"
    res = run_engine(capsys, ["music", "--clips", str(clips), "--music", str(music), "--out", str(out), "--length", "8", "--aspect", "square"])
    info = ac.probe(res["outputs"][0])
    assert (info.width, info.height) == (1080, 1080) and abs(info.duration - 8.0) < 0.05 and info.has_audio
    plan = json.loads((tmp_path / "out" / "edit.montage.json").read_text())
    assert sum(s["frames"] for s in plan["shots"]) == 240
    assert res["otio"].endswith(".otio")


@needs_ffmpeg
def test_engine_reference_keeps_reference_cuts_and_sound(material, tmp_path, capsys):
    clips, music = material
    ref = tmp_path / "ref.mp4"
    ff("-f", "lavfi", "-i", "color=c=red:size=480x854:rate=30:d=2", "-f", "lavfi", "-i", "testsrc2=size=480x854:rate=30:d=3", "-i", str(music),
       "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a", "-t", "5", "-c:v", "h264_videotoolbox", "-b:v", "1M", str(ref))
    out = tmp_path / "like.mp4"
    res = run_engine(capsys, ["reference", str(ref), "--clips", str(clips), "--out", str(out), "--no-clip"])
    plan = json.loads((tmp_path / "like.montage.json").read_text())
    assert [s["frames"] for s in plan["shots"]] == [60, 90]
    info = ac.probe(res["outputs"][0])
    assert info.display_size == (480, 854) and info.has_audio and abs(info.duration - 5.0) < 0.05


@needs_ffmpeg
def test_engine_split_same_quality(material, tmp_path, capsys):
    clips, _ = material
    src = tmp_path / "long.mp4"
    ff("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30", "-f", "lavfi", "-i", "sine=d=25", "-t", "25", "-g", "30",
       "-c:v", "h264_videotoolbox", "-b:v", "1M", "-c:a", "aac", str(src))
    res = run_engine(capsys, ["split", str(src), "--out-dir", str(tmp_path / "parts"), "--clip-len", "10"])
    durs = [ac.probe(f).duration for f in res["outputs"]]
    assert len(durs) == 3 and all(abs(d - 10) < 1.1 for d in durs[:2]) and abs(sum(durs) - 25) < 0.2
    assert res["outputs"][0].endswith("long_part001.mp4")


def test_engine_reports_errors_as_result(capsys):
    assert ei_engine.main(["split", "/does/not/exist.mp4", "--out-dir", "/tmp/x"]) == 1
    line = [l for l in capsys.readouterr().out.splitlines() if l.startswith("EI_RESULT ")][-1]
    assert json.loads(line[len("EI_RESULT "):])["ok"] is False
