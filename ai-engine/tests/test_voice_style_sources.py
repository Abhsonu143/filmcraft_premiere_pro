"""Styles learned from recordings, and finding recordings with music under the voice."""

import subprocess

import pytest

import autocut as ac
import script_writer as sw
import voice


def test_a_recording_is_a_style_source(monkeypatch, tmp_path):
    rec = tmp_path / "clip.mp3"
    rec.write_bytes(b"x")
    monkeypatch.setattr(ac, "transcribe", lambda p, **k: {"segments": [{"start": 0.0, "end": 2.0, "text": " So, GTA 6 is close. "},
                                                                          {"start": 2.0, "end": 3.0, "text": "   "}]})
    src = sw.load_sources([str(rec)])
    assert src == [{"name": "clip.mp3", "cues": [{"start": 0.0, "end": 2.0, "text": "So, GTA 6 is close."}], "text": "So, GTA 6 is close."}]


@pytest.mark.parametrize("under, low", [("", False), ("anoisesrc=color=pink:amplitude=0.25:d=8", True)])
def test_music_under_the_voice_is_noticed(tmp_path, under, low):
    # bursts of "talking" (a tone switched on and off), alone or with constant sound under them
    talk = "sine=f=220:d=8,volume='if(lt(mod(t,1),0.6),1,0)':eval=frame"
    out = tmp_path / "v.wav"
    if under:
        cmd = ["-f", "lavfi", "-i", talk, "-f", "lavfi", "-i", under, "-filter_complex", "amix=inputs=2:normalize=0"]
    else:
        cmd = ["-f", "lavfi", "-i", talk]
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *cmd, "-ar", "24000", str(out)], check=True)
    assert (voice.music_under(str(out), tmp_path) < voice.SEPARATE_BELOW) is low
