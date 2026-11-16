"""The animated character: lips, feelings, gestures, timing, placement, and a small end-to-end clip."""

import json
import subprocess

import numpy as np
import pytest

import character as ch
import vrm
from autocut import AutocutError
from test_vrm import tiny_vrm


def W(word, s, e):
    return {"word": word, "start": s, "end": e}


def test_mouth_shapes_english_and_hindi():
    assert [u[0] for u in ch.mouth_units("mama")] == ["closed", "aa", "closed", "aa"]
    assert [u[0] for u in ch.mouth_units("moon")] == ["closed", "ou", "rest"]
    assert [u[0] for u in ch.mouth_units("game")] == ["rest", "aa", "closed"]  # the silent e is not mouthed
    assert ch.mouth_units("...") == [] and ch.mouth_units("2026")[0][0] == "aa"
    hi = [u[0] for u in ch.mouth_units("नमस्ते")]
    assert hi[0] == "rest" and "closed" in hi and hi[-1] == "ee"  # न(a) म(a) स् ते
    assert [u[0] for u in ch.mouth_units("की")] == ["rest", "ih"]
    keys = ch.mouth_keys([W("mama", 1.0, 1.4), W("", 2.0, 2.1)])
    assert [k[1] for k in keys] == ["closed", "aa", "closed", "aa"] and keys[0][0] == pytest.approx(1.05) and keys[-1][0] < 1.4


def test_feelings_follow_the_lines_emotion():
    sad, body = ch.feeling("Grief", 1.0)
    assert sad["sad"] > 0.7 and body["pitch"] > 5 and body["amp"] < 0.7  # head sinks, small gestures
    joy, jb = ch.feeling("Excitement", 1.0)
    assert joy["happy"] > 0.7 and jb["amp"] > 1.1
    mild, _ = ch.feeling("Excitement", 0.0)
    assert 0 < mild["happy"] < joy["happy"]
    assert ch.feeling("Neutral", 0.5)[0] == ch.feeling(None, 0.5)[0] == ch.feeling("nonsense", 0.5)[0]


def spans(*texts, emotion="Neutral"):
    out, t = [], 0.0
    for x in texts:
        out.append({"start": t, "end": t + 3.0, "src": t, "text": x, "emotion": emotion, "intensity": 0.7})
        t += 3.4
    return out


def words_for(sp):
    out = []
    for s in sp:
        ws = s["text"].split()
        for k, w in enumerate(ws):
            out.append(W(w, s["start"] + 3.0 * k / len(ws), s["start"] + 3.0 * (k + 0.8) / len(ws)))
    return out


def test_gestures_come_from_what_is_said():
    sp = spans("Hello friends, welcome back.", "There is only one rule, remember it.", "Why would anyone do that?", "It is absolutely huge, the whole world saw it.",
               "Thank you for watching.")
    ws = words_for(sp)
    rng = np.random.default_rng(1)
    gs = ch.plan_gestures(sp, ws, [], 1.0, "r", rng)
    kinds = [g["kind"] for g in gs]
    assert kinds[0] == "wave" and gs[0]["arm"] == "r" and gs[0]["t"] < 1.0  # on "Hello", with the arm towards the picture
    assert "point_up" in kinds and "shrug" in kinds and "open_both" in kinds and kinds[-1] == "namaste"
    assert all(a != b for a, b in zip(kinds, kinds[1:]))
    assert all(b["t"] - a["t"] >= 1.3 for a, b in zip(gs, gs[1:]))
    assert next(g for g in gs if g["kind"] == "open_both")["arm"] == "both"
    # a greeting word in the middle of the video is not a greeting
    mid = ch.plan_gestures(spans("One.", "Say hello to the new map.", "Two."), words_for(spans("One.", "Say hello to the new map.", "Two.")), [], 1.0, "l", rng)
    assert "wave" not in [g["kind"] for g in mid]


def test_lines_are_found_where_the_edit_put_them():
    nar = {"segments": [{"start": 0.0, "end": 3.0, "text": "First line.", "emotion": "Joy", "intensity": 0.8},
                        {"start": 3.3, "end": 6.0, "text": "Second line!", "emotion": "Fear", "intensity": 0.5}]}
    assert [s["start"] for s in ch.spans_from(nar, None)] == [0.0, 3.3]
    # a cold open: the second line first, then both
    tl = {"lines": [{"start": "0.1", "end": "2.8", "text": "Second line!", "emotion": "Fear"}, {"start": "3.2", "end": "6.2", "text": "First line."},
                    {"start": "6.5", "end": "9.2", "text": "Second line!"}, {"start": "9.5", "end": "10", "text": "Not in the narration"}]}
    got = ch.spans_from(nar, tl)
    assert [(s["start"], s["src"], s["emotion"]) for s in got] == [(0.1, 3.3, "Fear"), (3.2, 0.0, "Joy"), (6.5, 3.3, "Fear")]
    with pytest.raises(AutocutError):
        ch.spans_from({"segments": []}, None)


def test_plan_moves_the_voice_to_video_time_and_the_mouth_follows_it(tmp_path):
    fps = 30.0
    sp = [{"start": 2.0, "end": 4.0, "src": 0.0, "text": "mama mia", "emotion": "Joy", "intensity": 0.9}]
    loud = np.zeros(90, np.float32)
    loud[3:55] = 0.9  # the voice, in its own time
    plan = ch.make_plan(sp, [W("mama", 0.1, 0.8), W("mia", 0.9, 1.8)], loud, fps, 6.0, seed=3)
    assert plan["mouth"][0][0] > 2.0 and plan["loud"][int(2.5 * fps)] > 0.5 and plan["loud"][10] == 0
    assert plan["feelings"][0]["face"]["happy"] > 0.5 and plan["blinks"] and json.dumps(plan)
    m = vrm.Model(tiny_vrm(tmp_path))
    an = ch.Animator(plan, m)
    quiet, talking = [], []
    for k in range(int(6 * fps)):
        pose, shapes, lift = an.frame(k / fps)
        assert all(np.isfinite(v).all() for v in pose.values()) and all(0.0 <= v <= 1.0 for v in shapes.values())
        (talking if 2.2 < k / fps < 3.6 else quiet if k / fps < 1.5 or k / fps > 5 else []).append(sum(shapes.get(v, 0.0) for v in ch.VOWELS))
    assert max(quiet) < 0.02 and max(talking) > 0.3  # shut in silence, open in speech
    assert min(talking) < 0.1  # and it closes on the m's
    # the same plan performs the same way again (a plan can be edited and re-run)
    a1, a2 = ch.Animator(plan, m), ch.Animator(plan, m)
    assert all(np.allclose(a1.frame(k / fps)[0]["head"], a2.frame(k / fps)[0]["head"]) for k in range(40))


def test_arms_mirror_each_other():
    a = dict(ch.GESTURES["explain"]["a"])
    left, right = ch.arm_pose("l", a, 3.0), ch.arm_pose("r", a, 3.0)
    mirror = np.diag([-1.0, 1.0, 1.0])
    for name, m in left.items():
        assert np.allclose(mirror @ m @ mirror, right[name.replace("left", "right")], atol=1e-9), name


def test_placement_and_stage():
    assert ch.place(1920, 1080, 400, 20, "bottom-right", None, None, False) == (1920 - 400 + 20 - 32, 1080 - 400 + 20 - 32)
    assert ch.place(1920, 1080, 400, 0, "top-left", None, None, False) == (32, 32)
    assert ch.place(1920, 1080, 400, 0, "bottom-left", None, None, True)[1] == 680  # a cut-out stands on the bottom edge
    assert ch.place(1000, 1000, 200, 0, "bottom-right", 0.5, 0.25, False) == (400, 150)
    frame = np.zeros((64, 64, 4), np.uint8)
    frame[16:48, 16:48] = 255  # an opaque white square in the middle (rows bottom first)
    st = ch.Stage("studio", 64, "color", "#102030", "#ff0000")
    over, clip = st.frame(frame.tobytes())
    o = np.frombuffer(over, np.uint8).reshape(st.canvas, st.canvas, 4)
    c = np.frombuffer(clip, np.uint8).reshape(64, 64, 3)
    assert o[0, 0, 3] == 0 and o[st.canvas // 2, st.canvas // 2].tolist() == [255, 255, 255, 255]  # clear outside the circle
    assert c[0, 0].tolist() == [0x10, 0x20, 0x30] and c[32, 32].tolist() == [255, 255, 255]
    ring = o[st.canvas // 2, st.pad + 1]
    assert ring[0] > 200 and ring[1] < 60  # the red border
    g = ch.Stage("green", 64)
    over, clip = g.frame(frame.tobytes())
    assert np.frombuffer(clip, np.uint8).reshape(64, 64, 3)[0, 0].tolist() == list(ch.GREEN)
    assert np.frombuffer(over, np.uint8).reshape(64, 64, 4)[0, 0, 3] == 0
    for kind in ch.BACKGROUNDS:
        assert ch.backdrop(kind, 48).size == (48, 48)
    with pytest.raises(AutocutError):
        ch.add_to_video(None, "/nonexistent.wav", {}, "/tmp/x.mp4", {"look": "hologram"})


def test_characters_are_kept_and_found(tmp_path, monkeypatch):
    monkeypatch.setenv("EI_SETTINGS", str(tmp_path / "settings.json"))
    assert ch.list_characters() == []
    with pytest.raises(AutocutError):
        ch.resolve_character(None)
    src = tiny_vrm(tmp_path)
    got = ch.add_character(src, "My Host!")
    assert got["name"] == "My Host" and [c["name"] for c in ch.list_characters()] == ["My Host"]
    path, prof = ch.resolve_character("my host")
    assert path.endswith("model.vrm") and prof["title"] == "New"
    assert ch.resolve_character(None)[0] == path and ch.resolve_character(src)[0] == src
    with pytest.raises(AutocutError):
        ch.resolve_character("nobody")
    bad = tmp_path / "bad.vrm"
    bad.write_bytes(b"nope")
    with pytest.raises(AutocutError):
        ch.add_character(str(bad))


def test_a_spoken_line_becomes_a_character_clip(tmp_path, monkeypatch):
    try:
        vrm.Renderer(vrm.Model(tiny_vrm(tmp_path)), 64).close()
    except AutocutError:
        pytest.skip("no graphics context here")
    monkeypatch.setenv("EI_SETTINGS", str(tmp_path / "settings.json"))
    wav = tmp_path / "voice.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=220:duration=1.5", "-ar", "16000", str(wav)], check=True)
    nar = {"script": "x", "segments": [{"start": 0.0, "end": 1.5, "text": "Hello, this is huge!", "emotion": "Excitement", "intensity": 0.9}]}
    out = tmp_path / "out.mp4"
    res = ch.add_to_video(None, str(wav), nar, str(out), {"character": tiny_vrm(tmp_path), "look": "green", "detail": 240}, listen=False)
    assert res["video"] is None and res["clip"].endswith(".character.green.mp4")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames", "-of", "csv=p=0", res["clip"]],
                           capture_output=True, text=True, check=True).stdout.strip().split(",")
    assert probe[:2] == ["240", "240"] and int(probe[2]) >= 45
    plan = json.loads(open(res["plan"]).read())
    assert plan["gestures"] and plan["gestures"][0]["kind"] == "wave"
    # the saved plan, edited, is performed as it is
    plan["gestures"] = []
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(plan))
    again = ch.add_to_video(None, str(wav), nar, str(tmp_path / "again.mp4"), {"character": tiny_vrm(tmp_path), "look": "studio", "detail": 240}, plan_path=str(edited))
    assert again["gestures"] == 0 and again["clip"].endswith("again.character.mp4")
