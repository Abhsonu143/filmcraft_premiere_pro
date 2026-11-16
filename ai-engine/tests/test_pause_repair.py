"""A sound in the pauses of a finished video's voice is taken out, in a copy of its folder."""

import json
import subprocess
import wave

import numpy as np
import pytest

import pause_repair as pr
from autocut import AutocutError

SR = 48000


def tone(freq, seconds, level=0.3):
    return level * np.sin(2 * np.pi * freq * np.arange(int(seconds * SR)) / SR)


def folder_with_bursts(tmp_path, video=True, shift=0.0):
    rng = np.random.default_rng(1)
    burst = 0.25 * rng.standard_normal(int(0.4 * SR)) * np.hanning(int(0.4 * SR))  # the "piece of a word" in every pause
    parts, segs, t = [], [], 0.0
    for k in range(4):
        parts += [tone(200 + 40 * k, 1.5), burst]
        segs.append({"start": t, "end": t + 1.5, "text": f"Line {k}.", "check": {"passage": k}})
        t += 1.9
    voice = np.concatenate(parts)
    d = tmp_path / "topic"
    d.mkdir()
    with wave.open(str(d / "v.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((voice * 32767).astype(np.int16).tobytes())
    (d / "v.narration.json").write_text(json.dumps({"segments": segs}))
    (d / "notes.txt").write_text("kept")
    if video:
        lead = np.zeros(int(shift * SR))
        music = 0.02 * np.sin(2 * np.pi * 90 * np.arange(len(voice) + len(lead)) / SR)
        mix = np.stack([0.7 * np.concatenate([lead, voice]) + music] * 2, axis=1).astype(np.float32)
        (tmp_path / "mix.f32").write_bytes(mix.tobytes())
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c=blue:s=64x64:r=10:d={len(mix) / SR:.3f}", "-f", "f32le", "-ar", str(SR), "-ac", "2",
                        "-i", str(tmp_path / "mix.f32"), "-c:v", "libx264", "-c:a", "aac", "-b:a", "192k", "-shortest", str(d / "v.mp4")], check=True)
        if shift:
            (d / "v.narrate.json").write_text(json.dumps({"lines": [{"start": s["start"] + shift, "end": s["end"] + shift, "text": s["text"]} for s in segs]}))
    return d, segs


def pause_levels(path, segs, shift=0.0):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    return [20 * np.log10(max(1e-9, float(np.sqrt((x[int((s["end"] + shift + 0.05) * SR) : int((s["end"] + shift + 0.35) * SR)] ** 2).mean())))) for s in segs[:-1]]


@pytest.mark.parametrize("shift", [0.0, 2.0])
def test_the_copy_has_quiet_pauses_and_the_original_is_untouched(tmp_path, shift):
    d, segs = folder_with_bursts(tmp_path, shift=shift)
    original = {p.name: p.read_bytes() for p in d.iterdir()}
    before = pause_levels(d / "v.mp4", segs, shift)
    res = pr.repair(str(d))
    fixed = d.with_name("topic - fixed")
    assert res["folder"] == str(fixed) and res["with_sound"] == 4 and res["repaired"] == 4 and res["left"] == 0
    assert {p.name: p.read_bytes() for p in d.iterdir()} == original  # nothing in the original changed
    assert (fixed / "notes.txt").read_text() == "kept" and json.loads((fixed / "pause-repair.json").read_text())["repaired"] == 4
    after = pause_levels(fixed / "v.mp4", segs, shift)
    assert max(before) > -25 and max(after) < min(before) - 15, (before, after)  # the bursts are gone, the music under them stays
    assert max(after) > -60
    with wave.open(str(fixed / "v.wav")) as w:
        v = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    assert not v[int((segs[0]["end"] + 0.01) * SR) : int((segs[1]["start"] - 0.01) * SR)].any() and v[: SR].any()
    # the copy is clean now: a second run finds nothing and copies nothing; a third copy would get its own name
    again = pr.repair(str(fixed))
    assert again["repaired"] == 0 and not again["outputs"] and not fixed.with_name("topic - fixed - fixed").exists()
    assert pr.repair(str(d))["folder"].endswith("topic - fixed 2")


def test_a_voice_without_a_video_and_folders_that_cannot_be_repaired(tmp_path):
    d, segs = folder_with_bursts(tmp_path, video=False)
    res = pr.repair(str(d / "v.wav"))  # a file in the folder is fine too
    assert res["repaired"] == 4 and [p.rsplit("/", 1)[-1] for p in res["outputs"]] == ["v.wav"]
    empty = tmp_path / "empty"
    empty.mkdir()
    for bad in (empty, tmp_path / "missing"):
        with pytest.raises(AutocutError):
            pr.repair(str(bad))


def test_pauses_follow_passages_and_a_cold_open():
    nar = {"segments": [{"start": 0, "end": 2, "text": "A.", "check": {"passage": 0}}, {"start": 2.0, "end": 4, "text": "B.", "check": {"passage": 0}},
                        {"start": 4.4, "end": 6, "text": "C.", "check": {"passage": 1}}]}
    assert pr.pauses(nar, None, 6.5) == [(4.0, 4.4, 0.0), (6.0, 6.5, 0.0)]
    tl = {"lines": [{"start": 0.0, "text": "C."}, {"start": 3.0, "text": "A."}, {"start": 5.0, "text": "B."}, {"start": 7.4, "text": "C."}]}
    assert pr.pauses(nar, tl, 6.0) == [(4.0, 4.4, 3.0)]
