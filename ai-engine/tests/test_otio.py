import json

import pytest

otio = pytest.importorskip("opentimelineio")
import otio_export


def test_sequence_settings_has_every_filmcraft_field():
    s = otio_export.sequence_settings(1080, 1920, 30000 / 1001)
    assert s["frame_rate"] == {"num": 30000, "den": 1001}
    assert set(s) >= {"width", "height", "frame_rate", "par", "sample_rate", "drop_frame", "preset", "audio_master", "preview_codec", "max_bit_depth", "max_render_quality", "working_space"}


def test_autocut_timeline_links_picture_and_sound(tmp_path):
    plan = {"mode": "autocut", "input": str(tmp_path / "missing.mp4"), "width": 1080, "height": 1920, "fps": 30, "segments": [{"start": 0, "end": 1}, {"start": 2, "end": 2.5}], "music": None}
    tl = otio_export.build(plan)
    path = tmp_path / "t.otio"
    otio.adapters.write_to_file(tl, str(path))
    back = otio.adapters.read_from_file(str(path))
    v1, a1 = back.tracks[0], back.tracks[1]
    assert a1.kind == "Audio" and len(v1) == len(a1) == 2
    assert v1.duration().value == 45 and v1.duration().rate == 30
    assert [c.metadata["filmcraft"]["link"] for c in v1] == [c.metadata["filmcraft"]["link"] for c in a1]
    assert v1[1].source_range.start_time.value == 60
    assert back.metadata["filmcraft"]["settings"]["height"] == 1920
    assert v1[0].media_reference.target_url.startswith("file:///")


def test_voiceover_timeline_matches_shot_frames(tmp_path):
    plan = {"mode": "voiceover", "voice": str(tmp_path / "vo.wav"), "width": 1920, "height": 1080, "fps": 30, "clips": [{"path": str(tmp_path / "a.mp4"), "duration": 10}],
            "shots": [{"clip": str(tmp_path / "a.mp4"), "in": 1.01, "frames": 91}, {"clip": str(tmp_path / "a.mp4"), "in": 5.0, "frames": 89}], "music": None}
    tl = otio_export.build(plan)
    assert tl.tracks[0].duration().value == 180
    assert tl.tracks[1].duration().value == 180  # narration covers the picture exactly
    assert tl.tracks[0][0].source_range.start_time.value == 30
    json.loads(otio.adapters.write_to_string(tl))


def test_srt_sidecar_is_remapped_for_autocut():
    plan = {"mode": "autocut", "segments": [{"start": 5.0, "end": 6.0}], "captions": {"style": "bottom", "words": [{"word": "hi", "start": 5.2, "end": 5.5}]}}
    assert "00:00:00,200 --> 00:00:00,500" in otio_export.captions_srt(plan)
