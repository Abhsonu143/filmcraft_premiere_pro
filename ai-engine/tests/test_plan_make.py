"""Making the video of a plan from the gameplay planner: the cuts and the two kinds of video."""

import json
import shutil
import subprocess

import pytest

import plan_make as pm
import script_writer as sw
from autocut import AutocutError

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

ANALYSIS = {"duration": 600.0, "dialogue": [{"start": 100.0, "end": 104.0, "text": "Where did you get that sword?"}, {"start": 110.0, "end": 113.0, "text": "It was my father's."}],
            "keyframes": [{"t": 2.0 * k, "motion": 1.0, "bright": 40, "cut": k == 50} for k in range(300)], "tags": [{"t": 100.0, "top": ["cutscene"]}], "loud": [0.1] * 600}


def plan(platform="youtube"):
    return {"title": "Game – The Sword: Part 1!", "platform": platform, "format": "story", "length": "2 min", "hook": 'On-screen text: "bhai zinda hai" over the first line',
            "why": "w", "edit_tool": "sections", "tags": ["#game"], "thumbnail": "t",
            "parts": [{"recording": "/r/a.mov", "r": 1, "start": 90.0, "end": 130.0, "note": "the sword"}, {"recording": "/r/a.mov", "r": 1, "start": 300.0, "end": 340.0, "note": "the fight"}]}


def test_a_plan_is_picked_by_number_or_title():
    plans = [{"title": "Story Part 1"}, {"title": "The Ring Duel"}]
    assert pm.pick_plan(plans, "2") == 1 and pm.pick_plan(plans, "ring") == 1 and pm.pick_plan(plans, "1") == 0
    for bad in ("", "3", "0", "nothing like it"):
        with pytest.raises(AutocutError):
            pm.pick_plan(plans, bad)


def test_cuts_never_split_a_line_and_stay_near_their_stretch():
    p = plan()
    raw = [{"part": 1, "start": "0:01:42", "end": "0:01:51", "note": "sword"},  # starts and ends inside lines
           {"part": 1, "start": "0:01:50", "end": "0:02:05", "note": "overlaps the one before"},
           {"part": 2, "start": "0:00:10", "end": "0:05:10", "note": "far before its stretch"},
           {"part": 9, "start": "0:00:00", "end": "0:01:00"}, {"part": 1, "start": "x", "end": "y"}, "junk"]
    cuts = pm.clean_cuts(raw, p, [ANALYSIS, ANALYSIS])
    assert [(c["part"], c["start"], c["end"]) for c in cuts] == [(1, 99.6, 113.5), (1, 113.5, 125.0), (2, 240.0, 310.0)]
    assert pm.clean_cuts(None, p, [None, None]) == [] and pm.keep_lines_whole(50.0, 60.0, None) == (50.0, 60.0)
    whole = pm.whole_parts(p, [ANALYSIS, None])
    assert [(c["start"], c["end"]) for c in whole] == [(90.0, 130.0), (300.0, 340.0)]
    assert sum(c["end"] - c["start"] for c in pm.fit_short(whole, 60.0)) == 60.0 and len(pm.fit_short(whole, 41.0)) == 1  # (a 1 s rest is not a piece)
    keyed = pm.on_keyframes([{"recording": "/r/a.mov", "start": 99.6, "end": 113.5}, {"recording": "/r/a.mov", "start": 113.5, "end": 125.0}], {"/r/a.mov": ANALYSIS})
    assert [c["start"] for c in keyed] == [98.0, 113.5]  # back to the keyframe, but never into the piece before


def test_words_chapters_and_the_hook():
    cuts = [{"recording": "/r/a.mov", "start": 99.0, "end": 105.0, "note": "the sword"}, {"recording": "/r/a.mov", "start": 109.0, "end": 124.0, "note": "the answer"}]
    words = pm.said_words(cuts, {"/r/a.mov": ANALYSIS})
    assert [w["word"] for w in words[:2]] == ["Where", "did"] and words[0]["start"] == 1.0 and words[6]["word"] == "It" and words[6]["start"] == pytest.approx(7.0)
    assert pm.hook_text(plan()) == "bhai zinda hai" and pm.hook_text({"hook": 'He says "run" and leaves'}) is None
    text = pm.publish_text(plan(), cuts, True)
    assert "0:00 the sword" in text and "0:06" not in text and "#game" in text  # (chapters are at least 10 s apart)
    assert "Chapters" not in pm.publish_text(plan(), cuts, False) and pm.safe_name('A: "B" / C?') == "A B C"
    assert "0:01:40–0:01:44 “Where did you get that sword?”" in pm.detail(ANALYSIS, 95, 115) and "scene-cut" in pm.detail(ANALYSIS, 95, 115)


@needs_ffmpeg
def test_a_plan_becomes_a_video(tmp_path, monkeypatch):
    rec = tmp_path / "a.mov"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=40", "-f", "lavfi", "-i", "sine=f=300:d=40",
                    "-c:v", "libx264", "-preset", "ultrafast", "-g", "30", "-c:a", "aac", "-shortest", str(rec)], check=True)
    an = {"duration": 40.0, "dialogue": [{"start": 6.0, "end": 9.0, "text": "Hello there friend."}], "keyframes": [{"t": float(k), "motion": 1.0, "bright": 40, "cut": False} for k in range(40)],
          "tags": [], "loud": [0.1] * 40}
    (tmp_path / "a.ei-analysis.json").write_text(json.dumps(an))
    p = plan()
    p["parts"] = [{"recording": str(rec), "r": 1, "start": 5.5, "end": 12.0, "note": "hello"}, {"recording": str(rec), "r": 1, "start": 20.3, "end": 31.0, "note": "later"}]
    short = {**plan("shorts"), "parts": p["parts"]}
    pj = tmp_path / "plans" / "plans.json"
    pj.parent.mkdir()
    pj.write_text(json.dumps({"game": "Game", "recordings": [str(rec)], "plans": [p, short]}))
    r = pm.make(str(pj), pick="1", writer="rules")  # a long video: joined as it is, from keyframes
    video = pj.parent / "videos" / "01 Game The Sword Part 1" / "01 Game The Sword Part 1.mp4"
    assert r["outputs"][0] == str(video) and r["pieces"] == 2 and r["otio"].endswith(".otio") and r["srt"].endswith(".srt")
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)], capture_output=True, text=True).stdout)
    assert dur == pytest.approx(7.0 + 11.0, abs=0.4)  # (5.0–12.0 and 20.0–31.0)
    saved = json.loads(pj.read_text())["plans"][0]
    assert [c["start"] for c in saved["cuts"]] == [5.0, 20.0] and "0:00 hello" in (video.parent / "publish.txt").read_text() and (pj.parent / "plans.md").is_file()
    saved["cuts"][1]["end"] = 25.0  # the cuts are edited by hand, and the video made again
    d = json.loads(pj.read_text())
    d["plans"][0] = saved
    pj.write_text(json.dumps(d))
    assert pm.make(str(pj), pick="sword", writer="rules")["seconds"] == pytest.approx(12.0)
    r = pm.make(str(pj), pick="2", writer="rules")  # a Short: vertical, with the hook and captions
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", r["outputs"][0]], capture_output=True, text=True).stdout
    assert out.strip() == "1080,1920" and r["seconds"] == pytest.approx(17.2, abs=0.2)
    # asking for a video in words: a new plan is written from the recordings, added and made
    monkeypatch.setattr(sw, "ask_claude", lambda prompt, model=None, timeout=600: json.dumps(
        {"plans": [{"title": "Game – Hello", "platform": "shorts", "format": "moment", "length": "20 s", "hook": "h", "why": "w", "edit_tool": "clips",
                    "parts": [{"recording": "R1", "start": "0:00:05", "end": "0:00:25", "note": "hello"}]}]} if "Plan exactly ONE video" in prompt
        else {"cuts": [{"part": 1, "start": "0:00:07", "end": "0:00:20", "note": "hello"}]}))
    monkeypatch.setattr(pm.gp, "research_game", lambda game, writer="claude", fresh=False: {"summary": "s"})
    r = pm.make(str(pj), request="a short of the hello", cuts_only=True)
    d = json.loads(pj.read_text())
    assert r["plan"] == 3 and d["plans"][2]["requested"] == "a short of the hello" and [(c["start"], c["end"]) for c in d["plans"][2]["cuts"]] == [(5.6, 20.0)]
    with pytest.raises(AutocutError):
        pm.make(str(tmp_path / "none.json"), pick="1")
    rec.unlink()
    with pytest.raises(AutocutError, match="not found"):
        pm.make(str(pj), pick="1", writer="rules")
