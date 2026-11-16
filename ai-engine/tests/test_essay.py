"""essay: a narrated video essay from analysed recordings (and the units it is built from)."""

import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

import essay as es
import gameplay_plan as gp
import pieces as pc
from autocut import AutocutError
from recordings import Library

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def library(tmp_path, real_video: bool = False, games=("Ghost Of Yotei",)) -> Library:
    """Two recordings per game with their analysis next to them, and a plans file per game."""
    files = []
    for g, game in enumerate(games):
        recs = []
        for k in range(2):
            rec = tmp_path / f"g{g}-rec{k}.mp4"
            if real_video:
                subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s=320x180:d=40:r=30", "-f", "lavfi", "-i",
                                f"sine=f={440 + 220 * k}:d=40", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(rec)], check=True)
            else:
                rec.write_bytes(b"x")
            a = {"version": "gp1", "source": str(rec), "name": rec.stem, "duration": 40.0, "loud": [0.5] * 40, "keyframes": [], "tags": [],
                 "dialogue": [{"start": 10.0, "end": 12.5, "text": "You chose vengeance."}, {"start": 13.0, "end": 15.0, "text": "I chose to live."},
                              {"start": 30.0, "end": 33.0, "text": "Then walk away."}]}
            gp.sidecars(rec)[0].write_text(json.dumps(a), encoding="utf-8")
            recs.append(str(rec))
        folder = tmp_path / f"plans{g}"
        folder.mkdir()
        (folder / "plans.json").write_text(json.dumps({"game": game, "recordings": recs + [str(tmp_path / "gone.mp4")], "plans": []}), encoding="utf-8")
        files.append(str(folder / "plans.json"))
    return Library(files)


def test_the_library_reads_what_was_analysed_and_never_cuts_a_line(tmp_path):
    lib = library(tmp_path, games=("Ghost Of Yotei", "Ghost Of Tsushima"))
    assert len(lib.items) == 4 and lib.games == ["Ghost Of Yotei", "Ghost Of Tsushima"] and lib.seconds == 160.0  # (the missing recording is left out)
    assert lib.get("R3")["game"] == "Ghost Of Tsushima" and lib.get(2)["n"] == 2
    with pytest.raises(AutocutError):
        lib.get(9)
    d = lib.digests()
    assert len(d) == 4 and d[0].startswith("## [Ghost Of Yotei] R1") and "You chose vengeance." in d[0]
    assert "“I chose to live.”" in lib.detail(1, 5, 20) and [x["text"] for x in lib.lines(1, 11, 14)] == ["You chose vengeance.", "I chose to live."]
    assert lib.whole(1, 11.0, 14.0) == (9.6, 15.5)  # (moved out to whole lines)
    # a shot does not start on a black screen
    lib.get(1)["analysis"]["keyframes"] = [{"t": float(t), "bright": 3.0 if 20 <= t < 26 else 60.0} for t in range(0, 40, 2)]
    assert lib.lit(1, 10, 4) == 10 and lib.lit(1, 20, 4) == 26.0 and lib.lit(2, 20, 4) == 20
    with pytest.raises(AutocutError):
        Library([str(tmp_path / "nothing.json")])


RAW = {"title": "Revenge or Honor", "chapters": [
    {"title": "The choice", "beats": [
        {"say": "Atsu is not saving an island.", "emotion": "tension", "show": [{"r": 1, "start": "0:00:20", "end": "0:00:26", "what": "Atsu rides"}, {"r": 99, "start": "0:00:01"}]},
        {"play": {"r": 1, "start": "0:00:11", "end": "0:00:14", "line": "made up", "who": "Jubei"}},
        {"say": "She answers to no one.", "show": []},
        {"play": {"r": 2, "start": "0:00:20", "end": "0:00:22", "line": "nothing is said here"}},
        {"play": {"r": 77, "start": "0:00:11", "end": "0:00:14"}}, {"say": "  "}, "junk"]},
    {"title": "Empty", "beats": []}], "doubts": ["Jin is not in these recordings"], "publish": {"title": "T", "description": "D", "tags": ["a", "b"]}}


def test_the_essay_is_made_safe(tmp_path):
    lib = library(tmp_path)
    out = es.clean_outline({"title": "T", "thesis": "x", "chapters": [{"title": "A", "point": "p", "stretches": [
        {"r": 1, "start": "0:00:05", "end": "0:30:00", "why": "the duel"}, {"r": 5, "start": "0:00:00", "end": "0:01:00"}, "junk"]}, {"no": "title"}]}, lib)
    assert out["chapters"][0]["stretches"] == [{"r": 1, "start": 5, "end": 40, "why": "the duel"}]
    with pytest.raises(AutocutError):
        es.clean_outline({"chapters": [{"title": "A", "stretches": []}]}, lib)
    e = es.clean_essay(RAW, lib)
    beats = e["chapters"][0]["beats"]
    assert len(e["chapters"]) == 1 and [("say" in b) for b in beats] == [True, False, True, False]
    assert beats[0]["show"] == [{"r": 1, "start": 20.0, "what": "Atsu rides"}] and beats[0]["emotion"] == "Tension"
    # the game's line: from the first line's start to the last line's end, with what was really said
    assert beats[1]["play"] == {"r": 1, "start": 9.75, "end": 15.35, "line": "You chose vengeance. I chose to live.", "who": "Jubei", "heard": True}
    assert beats[2]["show"][0]["r"] == 1 and beats[2]["show"][0]["start"] == 26.0  # (no shot of its own: the scene goes on)
    assert beats[3]["play"]["heard"] is False
    assert es.clean_essay(e, lib)["chapters"] == e["chapters"]  # (reading it back from essay.json changes nothing)
    md = es.essay_md({**e, "idea": "i", "thesis": "t"}, lib)
    assert "**You:** Atsu is not saving an island." in md and "THE GAME" in md and "no line was heard" in md and "Jin is not in these recordings" in md
    with pytest.raises(AutocutError):
        es.clean_essay({"chapters": [{"title": "x", "beats": [{"play": {"r": 1, "start": "0:00:11", "end": "0:00:14"}}]}]}, lib)


def test_lines_get_their_shots_for_as_long_as_they_are_said(tmp_path):
    lib = library(tmp_path)
    e = es.clean_essay(RAW, lib)
    assert [s["text"] for s in es.voice_script(e, "en")["segments"]] == ["Atsu is not saving an island.", "She answers to no one."]
    segs = [{"start": 0.0, "end": 2.4}, {"start": 3.0, "end": 5.0}]
    b = es.build(e, segs, 6.0, lib, "voice.wav", game_db=-20)
    kinds = [(p["kind"], round(p["seconds"], 2), round(p["gain"], 2)) for p in b["pieces"]]
    assert kinds == [("say", 3.0, 0.1), ("play", 5.6, 1.0), ("say", 3.0, 0.1), ("play", 2.0, 1.0)]
    assert b["voice"] == [("voice.wav", 0.0, 3.0, 0.0), ("voice.wav", 3.0, 3.0, 8.6)] and abs(b["seconds"] - 13.6) < 1e-6
    assert [(c["text"], c["game"], c["start"]) for c in b["cues"]][:3] == [("Atsu is not saving an island.", False, 0.0), ("You chose vengeance.", True, 3.25), ("I chose to live.", True, 6.25)]
    # no second of a recording is shown twice
    spans = sorted((p["r"], p["start"], p["start"] + p["seconds"]) for p in b["pieces"])
    assert all(not (a[0] == c[0] and a[2] > c[1] + 1e-6) for a, c in zip(spans, spans[1:]))
    assert pc.frames_of(b["pieces"], 30) == [90, 168, 90, 60] and b["chapters"] == [{"title": "The choice", "start": 0.0}]
    with pytest.raises(AutocutError):
        es.build(e, segs[:1], 6.0, lib, "voice.wav")
    words = es.cue_words([c for c in b["cues"] if not c["game"]])
    assert words[0]["word"] == " Atsu" and words[-1]["end"] <= 8.6 + 2.0 + 1e-6
    assert "0:00 The choice" in es.publish_text(e, b["chapters"]) and "Tags: a, b" in es.publish_text(e, b["chapters"])


@needs_ffmpeg
def test_pieces_become_a_video_with_the_voice_over_them(tmp_path):
    lib = library(tmp_path, real_video=True)
    voice = tmp_path / "voice.wav"
    with wave.open(str(voice), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes((np.sin(np.arange(24000 * 6) * 2 * np.pi * 300 / 24000) * 12000).astype(np.int16).tobytes())
    e = es.clean_essay(RAW, lib)
    b = es.build(e, [{"start": 0.0, "end": 2.4}, {"start": 3.0, "end": 5.0}], 6.0, lib, str(voice))
    out = tmp_path / "out" / "essay.mp4"
    r = pc.render(b["pieces"], b["voice"], str(out), W=640, H=360)
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,nb_frames:format=duration", "-of", "json", str(out)],
                                     capture_output=True, text=True).stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"], int(v["nb_frames"])) == (640, 360, 408) and abs(float(info["format"]["duration"]) - 13.6) < 0.15 and r["pieces"] == 4
    assert any(s["codec_type"] == "audio" for s in info["streams"]) and not list(out.parent.glob("ei-pieces-*"))
    # tall: the wide picture across the middle of a tall frame
    tall = tmp_path / "out" / "tall.mp4"
    pc.render(b["pieces"][:1], b["voice"][:1], str(tall), W=640, H=360, tall_frame=0.35, tall_size=(360, 640))
    size = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(tall)], capture_output=True, text=True).stdout.strip()
    assert size == "360,640"
    with pytest.raises(AutocutError):
        pc.render([{"src": str(tmp_path / "missing.mp4"), "start": 0, "seconds": 1, "gain": 1}], [], str(tmp_path / "x.mp4"))


def test_a_plan_is_written_and_kept_and_changed_with_notes(tmp_path, monkeypatch):
    import script_writer as sw

    lib = library(tmp_path)
    plans = str(tmp_path / "plans0" / "plans.json")
    asked = []

    def claude(prompt, model=None, timeout=600):
        asked.append(prompt)
        if "Lay out the argument" in prompt:
            return json.dumps({"title": "Revenge or Honor", "thesis": "Atsu is not Jin.", "hook": "h", "doubts": ["no Tsushima footage"],
                               "chapters": [{"title": "The choice", "point": "p", "stretches": [{"r": 1, "start": "0:00:00", "end": "0:00:40", "why": "Jubei"}]}]})
        if "wants changes" in prompt:
            return json.dumps({**RAW, "title": "Revenge or Honor (2)", "changed": ["opens on the duel"]})
        return json.dumps(RAW)

    monkeypatch.setattr(sw, "ask_claude", claude)
    monkeypatch.setattr(gp, "research_game", lambda g, *a, **k: {"summary": "A revenge story.", "characters": ["Atsu", "Jubei"]})
    out = tmp_path / "essay"
    r = es.make("Is Atsu the next Ghost?", [plans], str(out), minutes=3, language="hi", to="plan")
    assert r["video"] is None and r["lines"] == 2 and r["plays"] == 2 and len(asked) == 2
    assert "ONLY of: Ghost Of Yotei" in asked[0] and "“You chose vengeance.”" in asked[1] and "Hindi" in asked[1]
    saved = json.loads((out / "essay.json").read_text())
    assert saved["doubts"] == ["no Tsushima footage", "Jin is not in these recordings"] and "THE GAME" in (out / "essay.md").read_text()
    # again: nothing is asked again; with notes: only the revision
    es.make("", [], str(out), to="plan")
    assert len(asked) == 2
    es.make("", [], str(out), to="plan", notes="open on the duel")
    assert len(asked) == 3 and json.loads((out / "essay.json").read_text())["title"] == "Revenge or Honor (2)"
    with pytest.raises(AutocutError):
        es.make("", [], str(out), to="video")  # (no voice chosen)
    with pytest.raises(AutocutError):
        es.make("x", [], str(tmp_path / "new"), to="plan")  # (no plans file)
    _ = lib
