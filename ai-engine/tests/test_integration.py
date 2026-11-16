"""End-to-end runs with real ffmpeg on synthetic media (no models: no captions, no CLIP)."""

import json
import re
import subprocess

from conftest import needs_ffmpeg

import autocut as ac
import voiceover_cut as vc

BEEPS = [1.0, 4.0, 4.9, 9.0, 15.5]  # flash + beep together; everything else black and silent
BEEP_LEN = 0.3


def ff(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def make_sync_clip(path, dur=18):
    on = "+".join(f"between(t,{t},{t + BEEP_LEN - 0.001})" for t in BEEPS)
    ff("-f", "lavfi", "-i", f"color=black:s=640x360:r=30,drawbox=enable='{on}':color=white:t=fill",
       "-f", "lavfi", "-i", f"aevalsrc='0.5*sin(2*PI*1000*t)*({on})':s=48000:d={dur}",
       "-t", str(dur), "-c:v", "h264_videotoolbox", "-b:v", "2M", "-c:a", "aac", str(path))


def flash_starts(path):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-an", "-vf", "signalstats,metadata=print:key=lavfi.signalstats.YAVG", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    times = [float(x) for x in re.findall(r"pts_time:([\d.]+)", out)]
    yavg = [float(x) for x in re.findall(r"YAVG=([\d.]+)", out)]
    return [t for i, (t, y) in enumerate(zip(times, yavg)) if y > 128 and (i == 0 or yavg[i - 1] <= 128)]


def beep_starts(path):
    dur = ac.probe(path).duration
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-vn", "-af", "silencedetect=noise=-30dB:d=0.05", "-f", "null", "-"], capture_output=True, text=True).stderr
    sil = ac.parse_silencedetect(out, dur)
    return [e for s, e in sil if e < dur - 0.05]


@needs_ffmpeg
def test_autocut_keeps_audio_and_video_in_sync(tmp_path):
    src = tmp_path / "sync.mp4"
    make_sync_clip(src)
    plan_path = tmp_path / "sync.plan.json"
    assert ac.main(["plan", str(src), "--preset", "long", "--silence-db", "-40", "-o", str(plan_path)]) == 0
    plan = json.loads(plan_path.read_text())
    plan["width"], plan["height"] = 640, 360  # small and fast
    plan_path.write_text(json.dumps(plan))
    assert len(plan["segments"]) == 5
    out = tmp_path / "sync.cut.mp4"
    assert ac.main(["render", str(plan_path), "-o", str(out), "--codec", "h264"]) == 0
    expect_frames = sum(round(s["end"] * 30) - round(s["start"] * 30) for s in plan["segments"])
    frames = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
                            capture_output=True, text=True).stdout.strip()
    assert int(frames) == expect_frames
    flashes, beeps = flash_starts(out), beep_starts(out)
    assert len(flashes) == len(BEEPS) and len(beeps) == len(BEEPS), (flashes, beeps)
    for f, b in zip(flashes, beeps):
        assert abs(f - b) <= 1.5 / 30, (flashes, beeps)  # within about one frame
    assert src.is_file()  # original untouched


@needs_ffmpeg
def test_voiceover_render_matches_voice_length(tmp_path):
    for name, color in (("a", "red"), ("b", "blue")):
        ff("-f", "lavfi", "-i", f"color={color}:s=640x360:r=25", "-t", "4", "-c:v", "h264_videotoolbox", "-b:v", "1M", str(tmp_path / f"{name}.mp4"))
    voice = tmp_path / "vo.wav"
    ff("-f", "lavfi", "-i", "sine=f=300:d=10.37", str(voice))
    plan_path = tmp_path / "vo.vo.plan.json"
    assert vc.main(["plan", str(voice), "--clips", str(tmp_path), "--no-transcribe", "--no-clip", "-o", str(plan_path)]) == 0
    plan = json.loads(plan_path.read_text())
    plan["width"], plan["height"] = 640, 360
    plan_path.write_text(json.dumps(plan))
    total = sum(s["frames"] for s in plan["shots"])
    assert total == round(10.37 * 30)
    assert {s["clip"].rsplit("/", 1)[-1] for s in plan["shots"]} == {"a.mp4", "b.mp4"}
    out = tmp_path / "vo.mp4"
    assert vc.main(["render", str(plan_path), "--out", str(out), "--workers", "2"]) == 0
    info = ac.probe(out)
    assert abs(info.duration - total / 30) < 0.05 and info.has_audio
    # A second render reuses every shot (only changed shots are re-encoded).
    work = tmp_path / "vo.work"
    before = {p.name: p.stat().st_mtime for p in work.glob("shot_*.mp4")}
    assert vc.main(["render", str(plan_path), "--out", str(out)]) == 0
    assert {p.name: p.stat().st_mtime for p in work.glob("shot_*.mp4")} == before


@needs_ffmpeg
def test_burned_captions_with_awkward_paths(tmp_path):
    d = tmp_path / "it's a: dir"
    d.mkdir()
    src = d / "clip.mp4"
    make_sync_clip(src, dur=6)
    plan_path = d / "clip.plan.json"
    assert ac.main(["plan", str(src), "--silence-db", "-40", "-o", str(plan_path)]) == 0
    plan = json.loads(plan_path.read_text())
    plan["width"], plan["height"] = 640, 360
    plan["captions"] = {"style": "big", "words": [{"word": " Hello", "start": 1.0, "end": 1.2}, {"word": " there.", "start": 4.0, "end": 4.2}]}
    plan_path.write_text(json.dumps(plan))
    out = d / "clip.cut.mp4"
    assert ac.main(["render", str(plan_path), "-o", str(out), "--codec", "h264"]) == 0
    assert "Hello" in out.with_suffix(".srt").read_text()
    # Captions are burned in: frames differ from a render without them.
    plain = json.loads(plan_path.read_text())
    plain["captions"] = None
    plan_path.write_text(json.dumps(plain))
    out2 = d / "plain.mp4"
    assert ac.main(["render", str(plan_path), "-o", str(out2), "--codec", "h264"]) == 0
    sig = lambda p: subprocess.run(["ffmpeg", "-v", "error", "-ss", "0.2", "-i", str(p), "-frames:v", "1", "-f", "md5", "-"], capture_output=True, text=True).stdout
    assert sig(out) != sig(out2)
