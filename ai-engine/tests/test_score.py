"""Emotional sound design: emotions, effects, acts, events, music choice and the mix (no network)."""

import json

import numpy as np
from conftest import needs_ffmpeg

import emotions as em
import music_library as ml
import score as sc
import sfx
import voice as vo

USER_LIST = ("Joy Happiness Cheerful Bliss Delight Euphoria Excitement Thrill Love Passion Affection Romance Nostalgia Melancholy Sorrow "
             "Grief Fear Dread Anxiety Suspense Tension Mystery Curiosity Wonder Awe Amazement Serenity Tranquility Peace Calmness Courage "
             "Triumph Pride Hope Freedom Anger Rage Surprise Anticipation Fascination Wonderment Energy").split()


def test_every_emotion_is_known_and_has_a_delivery():
    for name in USER_LIST:
        assert em.resolve(name.lower()) == name
        e = em.info(name)
        assert e["family"] in em.FAMILIES and 0 <= e["arousal"] <= 1
        p = em.voice_params(name, 0.9)
        assert 0.25 <= p["exaggeration"] <= 1.15 and 0.2 <= p["cfg_weight"] <= 0.6
    assert em.resolve("hook") == "Surprise" and em.resolve("whatever") == "Neutral"
    # intensity matters: a peak is more expressive than a gentle line
    assert em.voice_params("Triumph", 1.0)["exaggeration"] > em.voice_params("Triumph", 0.2)["exaggeration"]
    assert em.voice_params("Rage", 0.8)["exaggeration"] > em.voice_params("Serenity", 0.8)["exaggeration"]
    assert em.nearest_family(-0.8, 0.2) == "sorrow" and em.nearest_family(0.75, 0.9) in ("heroic", "euphoria")


def test_effects_are_clean():
    for k in sfx.KINDS:
        y = sfx.make(k)
        assert y.ndim == 2 and y.shape[1] == 2 and len(y) > 1000 and np.isfinite(y).all()
        assert 0.5 < float(np.abs(y).max()) <= 0.9
        if k not in ("riser", "swell"):  # (those stop dead on the moment they lead into)
            assert float(np.abs(y[-200:]).max()) < 0.05  # every effect dies away (no click at the end)
    r = sfx.make("riser", 2.0)
    assert abs(len(r) / sfx.SR - 2.0) < 0.01 and float(np.abs(r[-4000:-300]).max()) > 0.3  # loudest right before its moment


def L(s, e, emo, inten=0.6, text="line"):
    return {"start": s, "end": e, "emotion": emo, "intensity": inten, "text": text}


def test_acts_follow_the_emotional_arc():
    lines = sc.normalise_lines([L(0, 4, "Curiosity"), L(4.3, 9, "Mystery"), L(9.4, 16, "Nostalgia"), L(16.5, 22, "Melancholy"),
                                L(23, 27, "Surprise", 0.9), L(27.3, 34, "Triumph", 0.9), L(34.3, 40, "Hope")])
    acts = sc.make_acts(lines, 42.0, min_len=6.0)
    fams = [a["family"] for a in acts]
    assert fams[0] == "mystery" and "nostalgia" in fams and fams[-1] == "heroic"
    assert acts[0]["start"] == 0.0 and acts[-1]["end"] == 42.0
    assert all(a["end"] == b["start"] for a, b in zip(acts, acts[1:]))
    # a 3 s act in the middle is too short for its own music: it joins the closer mood
    short = sc.make_acts(sc.normalise_lines([L(0, 10, "Calmness"), L(10, 13, "Joy"), L(13, 25, "Serenity")]), 25.0, min_len=6.0)
    assert len(short) == 1 and short[0]["family"] == "calm"


def test_events_hook_reveal_peak_and_cuts():
    lines = sc.normalise_lines([L(0, 4, "Mystery", 0.7), L(4.3, 10, "Suspense", 0.6), L(10.2, 16, "Tension", 0.7),
                                L(16.8, 20, "Surprise", 0.9), L(20.2, 26, "Awe", 0.7), L(26.5, 32, "Triumph", 0.95)])
    acts = sc.make_acts(lines, 34.0, min_len=5.0)
    ev, trans = sc.plan_events(lines, acts, [5.0, 6.0, 12.0, 27.0, 28.0, 29.0, 30.5], 34.0)
    kinds = [(e["kind"], round(e["t"], 1)) for e in ev]
    assert kinds[0] == ("braam", 0.0)  # a mysterious hook
    assert ("impact", 16.8) in kinds and ("riser", 16.8) in kinds  # the reveal lands after a build
    hit = [t for t in trans if t["type"] == "hit"]
    assert hit and abs(hit[0]["at"] - 16.8) < 1e-6 and 16.0 <= hit[0]["silence_from"] < 16.8  # a beat of silence first
    whooshes = [e["t"] for e in ev if e["kind"] == "whoosh"]
    assert whooshes and all(b - a >= 2.5 for a, b in zip(whooshes, whooshes[1:]))
    assert not any(e["kind"] == "whoosh" and e["t"] < 10 for e in ev)  # calm cuts get no whoosh
    impacts = sorted(e["t"] for e in ev if e["kind"] in ("impact", "braam"))
    assert all(b - a >= 1.5 for a, b in zip(impacts, impacts[1:]))


def test_mode_and_mood_classification():
    major = np.zeros(12)
    major[[0, 4, 7]] = 1.0  # C E G
    minor = np.zeros(12)
    minor[[9, 0, 4]] = 1.0  # A C E
    minor[9] += 0.6
    assert ml.mode_score(major) > 0 > ml.mode_score(minor)
    fam, v, a = ml.classify({"tempo": 128, "loud_db": -9, "centroid": 2600, "density": 0.25, "mode": 0.8})
    assert v > 0.3 and a > 0.7 and fam in ("euphoria", "heroic", "joy")
    fam, v, a = ml.classify({"tempo": 62, "loud_db": -26, "centroid": 900, "density": 0.03, "mode": -0.7})
    assert fam in ("sorrow", "nostalgia") and v < 0 and a < 0.35


def test_choose_music_prefers_the_family_and_no_repeats(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    tracks = []
    for fam in ("mystery", "heroic", "heroic", "calm"):
        f = tmp_path / f"{fam}{len(tracks)}.wav"
        f.write_bytes(b"x")
        tracks.append({"file": str(f), "family": fam, "source": "internet", "safe": True, "title": f.stem, "author": "A", "license": "CC BY 4.0",
                       "page": "p", "duration": 120})
    ml.save_index({"tracks": tracks}, lib)
    monkeypatch.setattr(ml, "fetch_family", lambda fam, licence="monetized", want=4, lib=None: [t for t in ml.load_index(lib)["tracks"] if t["family"] == fam])
    import music as mu

    monkeypatch.setattr(mu, "analyse", lambda p: {"duration": 120.0})
    monkeypatch.setattr(mu, "choose_window", lambda m, length, start="auto": 10.0)
    acts = [{"family": "heroic", "start": 0, "end": 20, "lines": []}, {"family": "heroic", "start": 20, "end": 40, "lines": []},
            {"family": "sorrow", "start": 40, "end": 60, "lines": []}]
    sc.choose_music(acts, "internet", lib=lib)
    assert acts[0]["track"] != acts[1]["track"] and {acts[0]["title"], acts[1]["title"]} == {"heroic1", "heroic2"}
    assert acts[2]["track"]  # no sorrow music: the nearest mood instead of none
    assert acts[0]["offset"] == 10.0 and acts[0]["credit"]["license"] == "CC BY 4.0"


def test_voice_follows_the_emotion():
    p = vo.engine_params({"emotion": "Rage", "intensity": 0.9}, "chatterbox")
    q = vo.engine_params({"emotion": "Serenity", "intensity": 0.9}, "chatterbox")
    assert p["exaggeration"] > q["exaggeration"] + 0.3 and p["cfg_weight"] < q["cfg_weight"]
    assert vo.engine_params({"emotion": "excited"}, "chatterbox")["exaggeration"] == 0.69  # old tags unchanged
    assert vo.pause_before({"emotion": "Surprise", "intensity": 1.0}) > vo.pause_before({"emotion": "Joy", "intensity": 1.0})
    segs = [{"text": "Calm here.", "emotion": "Calmness"}, {"text": "And more calm.", "emotion": "Serenity"},
            {"text": "Then this!", "emotion": "Surprise", "intensity": 0.9}]
    chunks = vo.group_lines(segs, [{"lang": "en", "energy": vo.line_energy(s)} for s in segs])
    assert [c["lines"] for c in chunks] == [[0, 1], [2]] and chunks[1]["emotion"] == "Surprise"


@needs_ffmpeg
def test_mix_ducks_music_under_the_voice(tmp_path, monkeypatch):
    sr = sc.SR
    t = np.arange(10 * sr) / sr
    voice = np.zeros((len(t), 2), np.float32)
    voice[2 * sr : 6 * sr] = (0.2 * np.sin(2 * np.pi * 220 * t[2 * sr : 6 * sr]))[:, None]
    vpath = tmp_path / "v.f32"
    voice.tofile(vpath)
    import subprocess

    wav = tmp_path / "v.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", str(vpath), str(wav)], check=True)
    music = tmp_path / "m.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anoisesrc=d=12:c=pink:a=0.3", "-ac", "2", "-ar", str(sr), str(music)], check=True)
    plan = {"total": 10.0, "music": True, "effects": True, "music_db": -2.0,
            "acts": [{"start": 0.0, "end": 10.0, "track": str(music), "offset": 0.0, "intensity": 0.6}], "transitions": [],
            "events": [{"kind": "impact", "t": 2.0}]}
    music_only = sc.music_bed(plan["acts"], [], 10.0, -2.0)
    pres = sc.presence(np.pad(voice, ((0, 0), (0, 0))))
    assert pres[int(4 * sr)] > 0.9 and pres[int(8 * sr)] < 0.05
    out = tmp_path / "mix.wav"
    sc.mix(str(wav), plan, out)
    y = sc.decode(str(out))

    def db(x):
        return 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)

    under = db(y[int(4.5 * sr) : int(5.5 * sr)] - 0)  # voice + ducked music
    gap = db(y[int(7.5 * sr) : int(9.0 * sr)])  # music alone, swelled back up
    assert gap > under - 12 and gap < under  # music is clearly there in the pause, under the voice level
    assert len(music_only) == 10 * sr
    j = json.dumps(plan)
    assert "impact" in j
