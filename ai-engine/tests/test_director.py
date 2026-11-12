import json
import subprocess

import numpy as np
import pytest
from conftest import needs_ffmpeg

import autocut as ac
import director as dr
import ei_engine
import music as mu
import render as rd


def ff(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def song(tmp_path_factory):
    """16 s at 120 BPM: a quiet intro, then from 8 s a loud section (the drop)."""
    d = tmp_path_factory.mktemp("song")
    p = d / "song.wav"
    expr = "(0.15+0.75*gte(t,8))*sin(2*PI*55*t)*exp(-25*mod(t,0.5))+0.05*sin(2*PI*330*t)"
    ff("-f", "lavfi", "-i", f"aevalsrc='{expr}':s=44100:d=16", str(p))
    return p


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    for i, src in enumerate(["testsrc2", "mandelbrot", "smptehdbars", "rgbtestsrc", "testsrc", "cellauto=rule=30"]):
        ff("-f", "lavfi", "-i", f"{src}:size=640x360:rate=60" if "=" in src else f"{src}=size=640x360:rate=60", "-t", "4", "-an", "-c:v", "libx264", "-preset", "ultrafast", str(d / f"c{i}.mp4"))
    return d


@needs_ffmpeg
def test_music_finds_the_drop_and_tempo(song):
    m = mu.analyse(str(song))
    assert abs(m["tempo"] - 120) < 3
    drops = [s for s in m["sections"] if s["drop"]]
    assert drops and abs(drops[0]["start"] - 8.0) < 0.6, m["sections"]
    assert drops[0]["level"] == "high"


@needs_ffmpeg
def test_cuts_follow_energy_and_land_on_beats(song):
    m = mu.analyse(str(song))
    w = mu.window(m, 0.0, 16.0)
    slots = dr.cut_points(w, 30, 1.0)
    assert sum(s["frames"] for s in slots) == 480
    calm = [s["frames"] for s in slots if s["start_f"] < 8 * 30 - 5]
    loud = [s["frames"] for s in slots if s["start_f"] >= 8 * 30 and not s.get("last")]
    assert np.mean(calm) > np.mean(loud), (calm, loud)  # quick cuts in the loud part
    assert any(s["drop"] for s in slots)
    beats = [round(b * 30) for b in w["beats"]]
    for s in slots[1:]:
        assert min(abs(s["start_f"] - b) for b in beats) <= 1, s["start_f"]
    assert slots[-1]["frames"] >= 30  # the ending holds


@needs_ffmpeg
def test_render_transitions_and_ramps_keep_exact_length(clips, tmp_path):
    c = sorted(str(p) for p in clips.glob("*.mp4"))
    plan = {"width": 360, "height": 640, "fps": 30, "look": "cinematic", "audio": None, "shots": [
        {"clip": c[0], "in": 0.2, "frames": 40, "speed": 0.5, "zoom": {"from": 1.0, "to": 1.1}, "fade_in": 6},
        {"clip": c[1], "in": 0.2, "frames": 30, "speed": 1.0, "punches": [0.0], "flash": [0.0], "transition_in": {"type": "dissolve", "frames": 10}},
        {"clip": c[2], "in": 0.2, "frames": 30, "speed": 1.0, "ramp_to": 2.0, "transition_in": {"type": "fadewhite", "frames": 6}},
        {"clip": c[3], "in": 0.2, "frames": 25, "speed": 1.5, "transition_in": {"type": "whipleft", "frames": 6}},
        {"clip": c[4], "in": 0.2, "frames": 35, "speed": 1.0, "fade_out": 10},
    ]}
    p = tmp_path / "t.montage.json"
    p.write_text(json.dumps(plan))
    out = rd.render(p)
    assert rd.quality_check(str(out), 160, 30) == []
    assert rd.source_time({"speed": 1.0, "ramp_to": 2.0, "frames": 30, "_fps": 30}, 1.0) == pytest.approx(1.5)


@needs_ffmpeg
def test_reference_style_reads_cuts_and_flash(tmp_path):
    ref = tmp_path / "ref.mp4"
    ff("-f", "lavfi", "-i", "testsrc2=size=360x640:rate=30:d=2", "-f", "lavfi", "-t", "2", "-i", "mandelbrot=size=360x640:rate=30",
       "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,eq=brightness='0.7*between(t,3.0,3.1)':eval=frame[v]", "-map", "[v]", "-c:v", "libx264", "-preset", "ultrafast", str(ref))
    st = dr.reference_style(str(ref), 30)
    assert len(st["shots"]) == 2 and abs(st["shots"][1]["start"] - 2.0) < 0.1
    assert st["shots"][1]["flash"] and abs(st["shots"][1]["flash"][0] - 1.0) < 0.15


def run_engine(capsys, argv):
    assert ei_engine.main(argv) == 0
    line = [l for l in capsys.readouterr().out.splitlines() if l.startswith("EI_RESULT ")][-1]
    res = json.loads(line[len("EI_RESULT "):])
    assert res["ok"], res
    return res


@needs_ffmpeg
def test_music_edit_from_a_video_soundtrack(song, clips, tmp_path, capsys):
    vid = tmp_path / "music_video.mp4"  # the music lives in a video file: no extraction step needed
    ff("-f", "lavfi", "-i", "color=c=black:s=320x240:r=30", "-i", str(song), "-t", "16", "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(vid))
    out = tmp_path / "edit.mp4"
    res = run_engine(capsys, ["music", "--clips", str(clips), "--music", str(vid), "--out", str(out), "--length", "12", "--aspect", "square", "--no-clip"])
    info = ac.probe(res["outputs"][0])
    assert (info.width, info.height) == (1080, 1080) and abs(info.duration - 12) < 0.05 and info.has_audio
    assert res["issues"] == []
    plan = json.loads((tmp_path / "edit.montage.json").read_text())
    used = [s["clip"] for s in plan["shots"]]
    assert len(set(used)) == len(used) or len(used) > 6  # no clip twice while unused ones remain
    assert plan["shots"][-1].get("fade_out")
