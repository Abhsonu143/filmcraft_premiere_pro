"""Dynamic pace, variety between runs and feedback for Auto Edit Clips to Music."""

import json
import subprocess

import dynamic_cut as dc


def test_phrase_pauses_in_a_voice_track(tmp_path):
    # three "phrases" of tone with pauses after 2 s and 4.5 s
    wav = tmp_path / "v.wav"
    expr = "0.6*sin(2*PI*400*t)*(lt(t,2)+gte(t,2.5)*lt(t,4.5)+gte(t,5))"
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"aevalsrc='{expr}':s=44100:d=7", str(wav)], check=True)
    gaps = dc.phrase_gaps(str(wav), 0.0, 7.0)
    assert len(gaps) == 2 and abs(gaps[0] - 2.25) < 0.2 and abs(gaps[1] - 4.75) < 0.2, gaps


def test_pace_follows_how_busy_the_music_is():
    beats = [i * 0.5 for i in range(40)]
    onset = [0.1] * 20 + [1.0, 0.9] * 10  # sparse first half, busy second half
    at = dc.pace_profile({"beats": beats, "onset": onset}, 1.0)
    assert at(2.0) > 1.1 and at(16.0) < 0.95
    assert dc.pace_profile({"beats": [], "onset": []}, 1.3)(5.0) == 1.3


def test_scene_changes_and_sentences_shape_the_windows():
    n = int(3 * dc.fo.FPS)
    motion = [2.0] * n + [60.0] + [2.0] * n  # a hard cut at 3.0 s
    cuts = dc.scene_cuts({"motion": motion})
    assert cuts == [3.0]
    c = {"path": "a.mp4", "cuts": cuts, "speech": [[4.0, 5.5]], "has_audio": True, "a": {"shake": []}}
    slot = {"frames": 60, "energy": 0.5}
    opts = {"dynamic": True, "fps": 30.0}
    assert dc.window_extra(slot, 0, c, 2.5, 1.0, 1.0, opts) == -9.0  # across the scene change
    whole = dc.window_extra(slot, 0, c, 3.8, 2.0, 1.0, opts)
    cut = dc.window_extra(slot, 0, c, 4.6, 1.2, 1.0, opts)
    fast = dc.window_extra(slot, 0, c, 3.8, 2.0, 2.0, opts)
    assert whole > 0 > cut and fast < whole  # a sentence whole, not cut, never sped up


def test_variety_and_avoiding_the_last_edit(tmp_path):
    c = {"path": "a.mp4", "a": {}}
    slot = {"frames": 30, "energy": 0.5}
    vals = {round(dc.window_extra(slot, 0, c, 1.0, 1.0, 1.0, {"seed": s}), 4) for s in range(8)}
    assert len(vals) > 4  # a different nudge each run
    assert dc.window_extra(slot, 0, c, 1.0, 1.0, 1.0, {"avoid": [("a.mp4", 0.5, 1.5)]}) == -0.8
    plan = tmp_path / "x.montage.json"
    plan.write_text(json.dumps({"fps": 30, "shots": [{"clip": "a.mp4", "in": 2.0, "frames": 30, "speed": 2.0}]}))
    assert dc.previous_windows(plan) == [("a.mp4", 2.0, 4.0)]


def test_joining_only_where_it_is_calm():
    calm = [{"start_f": i * 30, "frames": 30, "energy": 0.3, "section": 0, "hits": [0.5]} for i in range(3)]
    assert dc.can_join(calm, 30.0)
    m = dc.merged(calm, 30.0)
    assert m["frames"] == 90 and m["joined"] == 3 and len(m["hits"]) == 3 and abs(m["hits"][2] - 2.5) < 1e-6
    assert not dc.can_join([calm[0], {**calm[1], "drop": True}], 30.0)
    assert not dc.can_join([{**s, "energy": 0.9} for s in calm], 30.0)
    assert not dc.can_join([calm[0], {**calm[1], "section": 1}], 30.0)


def test_dynamic_sequence_joins_when_the_footage_carries_it():
    class Dr:
        CANDIDATES_PER_SHOT = 28

        @staticmethod
        def candidates(slot, clips, fps, progress, opts, k=None):
            joined = slot.get("joined", 1)
            # clip 0 holds a long sentence: much better as one joined shot than cut up
            out = [{"clip": ci, "in": 0.5, "speed": 1.0, "src": slot["frames"] / fps, "score": (3.0 if ci == 0 and joined == 3 else 1.0) - ci / 100, "f": {}}
                   for ci in range(len(clips))]
            return sorted(out, key=lambda x: -x["score"])

        pair_cost = staticmethod(lambda *a: 0.0)
        recent_subject_penalty = staticmethod(lambda *a: 0.0)
        emb_at = staticmethod(lambda *a: None)

    slots = [{"start_f": i * 30, "frames": 30, "energy": 0.3, "section": 0, "hits": []} for i in range(6)]
    clips = [{"path": f"{i}.mp4"} for i in range(10)]
    seq, used = dc.sequence_dynamic(Dr, slots, clips, 30.0, {})
    assert sum(s["frames"] for s in used) == 180
    j = next(k for k, s in enumerate(used) if s.get("joined") == 3)
    assert seq[j]["clip"] == 0  # (the long shot is the clip that carries it)


def test_feedback_is_kept_and_read(tmp_path):
    out = tmp_path / "edit.mp4"
    dc.add_feedback(out, "too many car shots")
    dc.add_feedback(out, "too many car shots")
    items = dc.add_feedback(out, "it is too hectic, less zooming")
    assert [i["text"] for i in items] == ["too many car shots", "it is too hectic, less zooming"]
    r = dc.rules_feedback("it is too hectic, less zooming, more faces, too many car shots")
    assert r["pace"] == "slower" and r["effects"] == "less" and "faces" in r["more"] and any("car" in x for x in r["less"])
    assert dc.interpret([], "rules")["pace"] == "same"
