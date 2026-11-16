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
    monkeypatch.setattr(pm, "hear", lambda rec, pieces, lang, hint='': None)  # (no listening model in tests: the planner's times are used)
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
    # a caption correction is kept in the plan and made in every later make
    monkeypatch.setattr(pm, "check_captions", lambda video, cues, lang, hint="": {"lines": len(cues), "found": len(cues), "worst": 0.1, "late": 0, "which": [], "moves": []})
    with pytest.raises(AutocutError, match="=>"):
        pm.make(str(pj), pick="2", writer="rules", caption_fix="no arrow here")
    r = pm.make(str(pj), pick="2", writer="rules", caption_fix="hello there => Hi there, old")
    assert "Hi there, old friend." in " ".join(c["text"] for c in pm.read_srt(pm.Path(r["srt"])))
    r = pm.make(str(pj), pick="2", writer="rules")  # (still there without being said again)
    assert "Hi there, old friend." in " ".join(c["text"] for c in pm.read_srt(pm.Path(r["srt"]))) and json.loads(pj.read_text())["plans"][1]["caption_fixes"] == [{"wrong": "hello there", "right": "Hi there, old"}]
    # the finished video is checked without making anything
    c = pm.check_video(str(pj), pick="2")
    report = pm.Path(c["outputs"][0]).read_text()
    assert c["problems"] == [] and c["captions"] >= 1 and "No problems found" in report and "Hi there, old friend." in report
    monkeypatch.setattr(pm, "check_captions", lambda video, cues, lang, hint="": {"lines": 2, "found": 2, "worst": 1.4, "late": 1, "which": ["3.0s “Hi there” +1.4s"], "moves": []})
    c = pm.check_video(str(pj), pick="2")
    assert len(c["problems"]) == 1 and "1.4" in c["problems"][0] and "PROBLEMS" in pm.Path(c["outputs"][0]).read_text()
    # saying what is wrong: Claude corrects the plan (pieces, teaser, caption words), the note is kept
    monkeypatch.setattr(pm.gp, "research_game", lambda game, writer="claude", fresh=False: {"summary": "s"})
    asked = []
    monkeypatch.setattr(sw, "ask_claude", lambda prompt, model=None, timeout=600: (asked.append(prompt), json.dumps(
        {"cuts": [{"part": 1, "start": "0:00:05", "end": "0:00:11", "note": "only the hello"}], "hook": "none", "captions": [{"wrong": "friend", "right": "Friend"}], "did": "Kept only the hello."}
        if "found something wrong" in prompt else {"lines": []}))[1])
    r = pm.make(str(pj), pick="2", fix="the second part is boring, drop it; no teaser; Friend is a name", cuts_only=True)
    fixed = json.loads(pj.read_text())["plans"][1]
    assert [c["part"] for c in fixed["cuts"]] == [1] and fixed["no_hook"] is True and not r["hook"] and fixed["fixes"][0]["did"] == "Kept only the hello."
    assert {"wrong": "friend", "right": "Friend"} in fixed["caption_fixes"] and "the second part is boring" in asked[0] and "Hi there, old friend." in asked[0]
    assert "The creator found these wrong" in pm.cuts_prompt(fixed, "Game", [])
    with pytest.raises(AutocutError, match="needs Claude"):
        pm.make(str(pj), pick="2", fix="x", writer="rules")
    # asking for a video in words: a new plan is written from the recordings, added and made
    monkeypatch.setattr(sw, "ask_claude", lambda prompt, model=None, timeout=600: json.dumps(
        {"plans": [{"title": "Game – Hello", "platform": "shorts", "format": "moment", "length": "20 s", "hook": "h", "why": "w", "edit_tool": "clips",
                    "parts": [{"recording": "R1", "start": "0:00:05", "end": "0:00:25", "note": "hello"}]}]} if "Plan exactly ONE video" in prompt
        else {"cuts": [{"part": 1, "start": "0:00:07", "end": "0:00:20", "note": "hello"}]}))
    r = pm.make(str(pj), request="a short of the hello", cuts_only=True)
    d = json.loads(pj.read_text())
    assert r["plan"] == 3 and d["plans"][2]["requested"] == "a short of the hello" and [(c["start"], c["end"]) for c in d["plans"][2]["cuts"]] == [(5.6, 20.0)]
    with pytest.raises(AutocutError, match="no finished video"):
        pm.check_video(str(pj), pick="3")
    with pytest.raises(AutocutError):
        pm.make(str(tmp_path / "none.json"), pick="1")
    rec.unlink()
    with pytest.raises(AutocutError, match="not found"):
        pm.make(str(pj), pick="1", writer="rules")


def test_the_hook_comes_from_inside_the_pieces():
    a = {**ANALYSIS, "dialogue": [{"start": 100.0, "end": 104.0, "text": "Where did you get that sword?"}, {"start": 118.0, "end": 120.0, "text": "It's me!"},
                                  {"start": 121.0, "end": 123.5, "text": "You don't recognize your own brother?"}]}
    cuts = [{"recording": "/r/a.mov", "part": 1, "start": 99.6, "end": 125.0, "note": "the sword"}]
    recs = {"/r/a.mov": a}
    h = pm.hook_cut(plan(), cuts, recs, False)  # no idea given: the line with the most force, not the opening one
    assert (h["start"], h["end"]) == (120.85, 123.75) and h["note"] == ""
    asked = {**plan(), "hook_idea": {"part": 1, "start": "0:01:58", "end": "0:01:59"}}
    h = pm.hook_cut(asked, cuts, recs, False)  # Claude's choice, with its line kept whole
    assert (h["start"], h["end"]) == (117.85, 120.25) and "hook_idea" not in asked
    opening = {**plan(), "hook_idea": {"part": 1, "start": "0:01:40", "end": "0:01:43"}}
    assert pm.hook_cut(opening, cuts, recs, False)["start"] == 120.85  # the first seconds of the video are not a hook
    keyed = pm.hook_cut(plan(), cuts, recs, True)
    assert keyed["start"] == 120.0 and pm.HOOK[0] <= keyed["end"] - keyed["start"] <= pm.HOOK[1]  # joined without re-encoding: from a keyframe
    assert pm.hook_cut(plan(), [{"recording": "/r/a.mov", "start": 99.6, "end": 110.0}], recs, False) is None  # too short a video for one
    quiet = {**ANALYSIS, "dialogue": [], "loud": [0.1] * 300 + [0.9] * 4 + [0.1] * 296}
    h = pm.hook_cut(plan(), [{"recording": "/r/a.mov", "part": 2, "start": 280.0, "end": 330.0, "note": "fight"}], {"/r/a.mov": quiet}, False)
    assert 296 <= h["start"] <= 304 and h["end"] - h["start"] == 2.5  # nothing said: the peak of the action
    text = pm.publish_text(plan(), [{"recording": "/r/a.mov", "start": 120.0, "end": 123.0, "note": ""}, *[{**c, "note": "the sword"} for c in cuts]], True)
    assert "0:00 the sword" in text  # the first chapter starts at 0:00, teaser included


def test_lines_are_heard_again_at_their_real_time(monkeypatch):
    a = {"duration": 600.0, "dialogue": [{"start": 50.0, "end": 52.0, "text": "Far away."}, {"start": 112.0, "end": 114.0, "text": "Where did you get that sword?"},
                                         {"start": 119.0, "end": 126.0, "text": "Don't lie to me."}, {"start": 144.0, "end": 148.0, "text": "You don't recognize your own brother."},
                                         {"start": 150.0, "end": 152.0, "text": "Never said at all."}]}
    line = lambda s, text: {"start": s, "end": s + 0.4 * len(text.split()), "text": text, "words": [[w, s + 0.4 * k, s + 0.4 * (k + 1)] for k, w in enumerate(text.split())]}  # noqa: E731
    asked = []

    def hear(rec, pieces, lang, hint=''):
        asked.append((pieces, lang))
        if len(asked) == 1:  # the stretch: two lines heard (one 18 s later than the planner had it), two not
            return [[line(126.5, "Don't lie to me."), line(162.6, "You don't recognise your own brother?")]]
        return [[line(111.5, "Where did you get that sword?")] if p[0] == 100.0 else [] for p in pieces]  # around each lost line

    monkeypatch.setattr(pm, "hear", hear)
    got = pm.listen_again("/r/a.mov", a, [(100.0, 170.0)])
    assert asked[0] == ([(96.0, 174.0)], "en") and asked[1][0] == [(100.0, 126.0), (138.0, 164.0)]
    assert [(d["start"], d["text"]) for d in got["dialogue"]] == [(50.0, "Far away."), (111.5, "Where did you get that sword?"), (126.5, "Don't lie to me."),
                                                                  (162.6, "You don't recognize your own brother.")]  # (the line never heard is left out; the planner's spelling is kept)
    cuts = [{"recording": "/r/a.mov", "start": 162.0, "end": 166.0}, {"recording": "/r/a.mov", "start": 110.0, "end": 130.0}]
    words = pm.heard_words(cuts, {"/r/a.mov": got})
    assert (words[0]["word"], words[0]["start"]) == ("You", pytest.approx(0.6)) and (words[6]["word"], words[6]["start"]) == ("Where", pytest.approx(5.5))
    assert pm.heard_words(cuts[1:], {"/r/a.mov": a})[0] == {"word": "Where", "start": 2.0, "end": pytest.approx(2.333, abs=0.001)}  # not heard again: spread over the line
    monkeypatch.setattr(pm, "hear", lambda rec, pieces, lang, hint='': None)
    assert pm.listen_again("/r/a.mov", a, [(100.0, 170.0)]) is a and pm.listen_again("/r/a.mov", None, [(0.0, 9.0)]) is None
    assert pm.merge_spans([(5.0, 9.0), (0.0, 4.5), (20.0, 30.0)]) == [(0.0, 9.0), (20.0, 30.0)]
    assert pm.same_line("For dead man.", "Not for dead man") and not pm.same_line("It's me.", "Jubei.")


def test_a_hook_never_stops_in_the_middle_of_a_line():
    words = lambda s, text, per=0.5: [[w, s + per * k, s + per * (k + 1)] for k, w in enumerate(text.split())]  # noqa: E731
    long = "So why are you picking corpses, when the whole town is watching you do it right now?"  # 8.5 s
    a = {**ANALYSIS, "dialogue": [{"start": 100.0, "end": 102.0, "text": "Opening line here."},
                                  {"start": 120.0, "end": 122.0, "text": "So why?", "words": words(120.0, "So why?", 1.0)},
                                  {"start": 122.4, "end": 124.4, "text": "It's a business.", "words": words(122.4, "It's a business.", 0.66)},
                                  {"start": 140.0, "end": 148.5, "text": long, "words": words(140.0, long)}]}
    cuts = [{"recording": "/r/a.mov", "part": 1, "start": 99.0, "end": 150.0, "note": "n"}]
    two = pm.hook_cut({**plan(), "hook_idea": {"part": 1, "start": "0:02:00", "end": "0:02:03"}}, cuts, {"/r/a.mov": a}, False)
    assert (two["start"], two["end"]) == (119.85, 124.65)  # the question and its answer, both whole (longer than a 4 s action hook)
    one = pm.hook_cut({**plan(), "hook_idea": {"part": 1, "start": "0:02:20", "end": "0:02:23"}}, cuts, {"/r/a.mov": a}, False)
    assert (one["start"], one["end"]) == (139.85, 143.2)  # a line too long for a hook stops after "corpses,"


def test_missed_lines_names_and_the_check_of_the_finished_video(monkeypatch, tmp_path):
    a = {"duration": 600.0, "dialogue": [{"start": 100.0, "end": 102.0, "text": "Hello there.", "words": [["Hello", 100.0, 101.0], ["there.", 101.0, 102.0]]}]}
    heard = [[{"start": 100.2, "end": 101.8, "text": "Hello there", "words": []}, {"start": 105.0, "end": 106.0, "text": "Push through!", "words": [["Push", 105.0, 105.5], ["through!", 105.5, 106.0]]}]]
    monkeypatch.setattr(pm, "hear", lambda rec, pieces, lang, hint="": heard if pieces == [(98.5, 110.5)] else None)
    got = pm.listen_closer("/r/a.mov", a, [(99.0, 110.0)])
    assert [d["text"] for d in got["dialogue"]] == ["Hello there.", "Push through!"]  # the missed line is added, the known one stays as it was
    assert pm.listen_closer("/r/a.mov", a, [(300.0, 310.0)]) is a and pm.listen_closer("/r/a.mov", None, [(0.0, 5.0)]) is None
    line = {"start": 1.0, "end": 2.0, "text": "Lord Scytho rides", "words": [["Lord", 1.0, 1.3], ["Scytho", 1.3, 1.7], ["rides", 1.7, 2.0]]}
    assert pm.spelled_as(line, "Lord Saito rides.")["words"][1] == ["Saito", 1.3, 1.7] and pm.spelled_as(line, "Lord Saito") is line
    monkeypatch.setattr(pm.ac, "extract_audio", lambda *a, **k: None)
    monkeypatch.setattr(pm.ac, "run_isolated", lambda fn, wavs, *a: [[{"words": [["Where", 4.6, 4.8], ["did", 4.8, 5.0], ["you", 5.0, 5.2], ["Don't", 21.0, 21.3], ["lie", 21.3, 21.5]]}]])
    cues = [{"start": 4.5, "end": 6.0, "text": "Where did you"}, {"start": 19.7, "end": 20.7, "text": "Don't lie"}, {"start": 30.0, "end": 31.0, "text": "Never heard"}]
    assert pm.check_captions(tmp_path / "v.mp4", cues, "en") == {"lines": 3, "found": 2, "worst": 1.3, "late": 1, "which": ["19.7s “Don't lie” +1.3s"],
                                                                  "moves": [(4.5, 6.0, pytest.approx(0.1)), (19.7, 20.7, pytest.approx(1.3))]}
    ws = [{"word": "Where", "start": 4.5, "end": 5.0}, {"word": "Don't", "start": 19.7, "end": 20.2, "est": True}, {"word": "lie", "start": 20.2, "end": 20.7, "est": True},
          {"word": "Heard", "start": 30.0, "end": 30.5}]
    moved, n = pm.moved_to_the_sound(ws, [(4.5, 6.0, 0.9), (19.7, 20.7, 1.3), (30.0, 31.0, 2.0)])
    assert n == 1 and [w["start"] for w in moved] == [4.5, 21.0, 21.5, 30.0]  # only the line timed by the subtitle alone moves


def test_the_game_s_own_subtitles_are_read():
    sub = lambda text, y=0.92, x=0.5: (text, x - 0.2, x + 0.2, y - 0.05, y)  # noqa: E731
    frames = [[], [sub("Where did you get that sword?")], [sub("Where did you get that sword?")], [sub("Where did you get that sword?"), sub("Atsu: Off a dead man.", 0.86)],
              [sub("Where did you get that sw0rd?"), sub("Atsu: Off a dead man.", 0.86)], [sub("Atsu: Off a dead man.", 0.86)], [sub("(Chuckles)")], [sub("(Chuckles)")], [sub("(Chuckles)")],
              [sub("HOLD X TO LOOT")], [sub("HOLD X TO LOOT")], [sub("HOLD X TO LOOT")], [sub("Matsumae", x=0.2)], [sub("Matsumae", x=0.2)], [sub("Matsumae", x=0.2)],
              [sub("I thought we killed", 0.88), sub("your whole family.", 0.93)], [sub("I thought we killed", 0.88), sub("your whole family.", 0.93)], [sub("I thought we killed", 0.88), sub("your whole family.", 0.93)],
              [sub("Flicker.")], [], [sub("I will av")], [sub("I will av")], [sub("I will av")], [sub("I will avenge my family...")], [sub("I will avenge my family...")],
              [sub("I will avenge my family...")], [sub("l am not a coward, l'm a killer")], [sub("l am not a coward, l'm a killer")], [sub("l am not a coward, l'm a killer")], [],
              [("I will aven", 0.3, 0.446, 0.87, 0.92), ("ge my family...", 0.45, 0.7, 0.87, 0.92)], [("I will aven", 0.3, 0.446, 0.87, 0.92), ("ge my family...", 0.45, 0.7, 0.87, 0.92)],
              [("I will aven", 0.3, 0.446, 0.87, 0.92), ("ge my family...", 0.45, 0.7, 0.87, 0.92)], []]
    ev = pm.subtitle_events(frames, 100.0, fps=2)
    assert [(e["start"], e["end"], e["text"]) for e in ev] == [(100.5, 102.5, "Where did you get that sword?"), (101.5, 103.0, "Off a dead man."), (107.5, 109.0, "I thought we killed your whole family."),
                                                                   (110.0, 113.0, "I will avenge my family..."), (113.0, 114.5, "I am not a coward, I'm a killer"), (115.0, 116.5, "I will avenge my family...")]


def test_subtitles_give_the_wording_and_heard_words_the_time():
    w = lambda s, text, per=0.4: [[x, round(s + per * k, 2), round(s + per * (k + 1), 2)] for k, x in enumerate(text.split())]  # noqa: E731
    line = lambda s, text: {"start": s, "end": round(s + 0.4 * len(text.split()), 2), "text": text, "words": w(s, text)}  # noqa: E731
    a = {"dialogue": [line(50.0, "Outside the stretch."), line(100.0, "My name is Arzu."), line(110.0, "You don't recognize your own brother."), line(120.0, "It was my father's."),
                      line(130.2, "Well, four dead men."), line(140.0, "Laundry")]}
    subs = [{"start": 100.5, "end": 102.5, "text": "My name is Atsu.", "y": 0.92}, {"start": 110.5, "end": 113.0, "text": "You don't recognise your own brother?", "y": 0.92},
            {"start": 120.4, "end": 122.0, "text": "It was my father's.", "y": 0.91}, {"start": 130.5, "end": 132.0, "text": "Off a dead man.", "y": 0.92},
            {"start": 150.5, "end": 152.0, "text": "Jubei?", "y": 0.93}, {"start": 160.0, "end": 162.0, "text": "Press to continue", "y": 0.6}]
    got = pm.use_subtitles(a, subs, [(90.0, 170.0)])["dialogue"]
    assert [d["text"] for d in got] == ["Outside the stretch.", "My name is Atsu.", "You don't recognise your own brother?", "It was my father's.", "Off a dead man.", "Jubei?"]
    assert got[1]["words"] == [["My", 100.0, 100.4], ["name", 100.4, 100.8], ["is", 100.8, 1.2 + 100.0], ["Atsu.", 101.2, 101.6]]  # the name takes the misheard word's time
    assert got[2]["words"][2] == ["recognise", 110.8, 111.2] and got[4]["start"] == 130.2 and got[5]["start"] == pytest.approx(150.05, abs=0.06)  # (a subtitle nothing was heard for: it comes up about 0.45 s after the voice here)
    assert pm.use_subtitles(a, [], [(0.0, 9.0)]) is a and pm.use_subtitles(None, subs, [(0.0, 9.0)]) is None
    quiet = {"dialogue": []}  # nothing heard at all (guards talking far away): subtitles that name their speaker, or several full sentences, are still taken
    named = [{"start": 10.0, "end": 12.0, "text": "They say she's here...", "y": 0.77, "named": True}, {"start": 20.0, "end": 22.0, "text": "REACH THE CELLS", "y": 0.2}]
    assert [d["text"] for d in pm.use_subtitles(quiet, named, [(0.0, 30.0)])["dialogue"]] == ["They say she's here..."]
    plain = [{"start": 10.0 + 5 * k, "end": 12.0 + 5 * k, "text": t, "y": 0.9} for k, t in enumerate(["Have you heard the rumours?", "She rose from the dead.", "I heard that too.", "Loot"])]
    assert [d["text"] for d in pm.use_subtitles(quiet, plain, [(0.0, 40.0)])["dialogue"]] == ["Have you heard the rumours?", "She rose from the dead.", "I heard that too."]
    assert [x[0] for x in pm.words_for("a b c d", [], 1.0, 3.0)] == ["a", "b", "c", "d"] and pm.words_for("a b c d", [], 1.0, 3.0)[3] == ["d", 2.5, 3.0]
    over = {"dialogue": [line(10.0, "Get him now please"), line(10.8, "Get back here")]}  # two people at once: no word is dropped
    words = pm.heard_words([{"recording": "/r", "start": 9.0, "end": 20.0}], {"/r": over})
    assert [x["word"] for x in words] == ["Get", "him", "now", "please", "Get", "back", "here"] and all(b["start"] >= a2["end"] - 1e-9 for a2, b in zip(words, words[1:]))


def test_heard_lines_are_proofread(monkeypatch):
    w = lambda s, text: [[x, s + 0.3 * k, s + 0.3 * (k + 1)] for k, x in enumerate(text.split())]  # noqa: E731
    a = {"dialogue": [{"start": 1.0, "end": 2.2, "text": "My name is Arzu.", "words": w(1.0, "My name is Arzu.")}, {"start": 5.0, "end": 5.9, "text": "Come on, boy!", "words": w(5.0, "Come on, boy!")},
                      {"start": 8.0, "end": 9.0, "text": "They call him 'mad?", "read": True}, {"start": 11.0, "end": 12.0, "text": "Read from the game.", "read": True}, {"start": 30.0, "end": 31.0, "text": "Outside."}]}
    asked = []
    monkeypatch.setattr(sw, "ask_claude", lambda prompt, model=None, timeout=600: asked.append(prompt) or json.dumps({"lines": ["My name is Atsu.", "Something else entirely, rewritten.", "They call him 'mad'?", "Read by the game, more or less."]}))
    got = pm.proofread(a, [(0.0, 20.0)], "Game", ["Atsu"])["dialogue"]
    assert [d["text"] for d in got] == ["My name is Atsu.", "Come on, boy!", "They call him 'mad'?", "Read from the game.", "Outside."]  # (a rewrite is not taken; a subtitle only lightly)
    assert got[0]["words"][3] == ["Atsu.", 1.9, 2.2] and "1. My name is Arzu." in asked[0] and "Outside" not in asked[0]
    pieces = {"dialogue": [{"start": 1.0, "end": 3.0, "text": "ge my family...", "read": True, "variants": ["Lavenge my family...", "will avenge my familly...", "ge my family...", "I will av"]},
                           {"start": 5.0, "end": 6.0, "text": "e my fam", "read": True, "variants": ["e my fam", "my fa"]}]}
    monkeypatch.setattr(sw, "ask_claude", lambda prompt, model=None, timeout=600: asked.append(prompt) or json.dumps({"lines": ["I will avenge my family...", "He is my famous friend"]}))
    got = pm.proofread(pieces, [(0.0, 20.0)], "Game", [])["dialogue"]
    assert [d["text"] for d in got] == ["I will avenge my family...", "e my fam"] and "READINGS: Lavenge my family... | will avenge" in asked[-1]  # put together from its readings; a guess is not taken
    assert pm.same_reading("ge my family...", "avenge my family...") and pm.same_reading("I will av", "I will avenge my family...") and not pm.same_reading("Get back here!", "You want me?")
    monkeypatch.setattr(sw, "ask_claude", lambda prompt, model=None, timeout=600: json.dumps({"lines": ["only one"]}))
    assert pm.proofread(a, [(0.0, 20.0)], "Game", [])is a


def test_no_piece_ends_in_the_middle_of_a_line():
    a = {"dialogue": [{"start": 98.0, "end": 101.0, "text": "began before the piece."}, {"start": 118.0, "end": 121.0, "text": "These are the sort of men who could use a lesson."},
                      {"start": 139.0, "end": 141.0, "text": "runs into the next piece"}, {"start": 205.0, "end": 215.0, "text": "too long to take whole"}]}
    cuts = [{"recording": "/r", "start": 100.0, "end": 119.5}, {"recording": "/r", "start": 130.0, "end": 140.0}, {"recording": "/r", "start": 140.5, "end": 150.0}, {"recording": "/r", "start": 200.0, "end": 206.0}]
    got = pm.whole_at_edges(cuts, {"/r": a}, 50.0)
    assert [(c["start"], c["end"]) for c in got] == [(97.8, 121.3), (130.0, 138.9), (141.1, 150.0), (200.0, 204.9)]  # grown to the whole line, or stopped before it
    assert pm.whole_at_edges(cuts[:1], {"/r": None}, 50.0) == cuts[:1]


def test_a_sentence_over_two_subtitles_is_one():
    lines = [{"start": 10.0, "end": 12.0, "text": "Before every duel,"}, {"start": 12.2, "end": 18.0, "text": "Takezo wrote the name on a strip of paper."}, {"start": 30.0, "end": 31.5, "text": "He disappeared."},
             {"start": 40.0, "end": 43.0, "text": "Only when he loses his title will he pass"}, {"start": 43.5, "end": 46.0, "text": "on the secret."}]
    got = pm.sentences(lines)
    assert [(d["start"], d["end"]) for d in got] == [(10.0, 18.0), (30.0, 31.5), (40.0, 46.0)] and got[2]["text"].endswith("will he pass on the secret.")
    cut = pm.whole_at_edges([{"recording": "/r", "start": 9.0, "end": 43.2}], {"/r": {"dialogue": lines}}, 36.0)
    assert [(c["start"], c["end"]) for c in cut] == [(9.0, 39.9)]  # no room for the whole sentence: the piece stops before it


def test_a_map_at_the_edge_of_a_piece_is_left_out(monkeypatch, tmp_path):
    import thumbnails as th

    def fake_run(cmd, **k):  # (a picture file for each asked second)
        from pathlib import Path

        Path(cmd[-1]).write_bytes(b"x")

    monkeypatch.setattr(pm.subprocess, "run", fake_run)
    # 20 s piece: 6 looks from the start, then 6 from the end backwards; the first two seconds are a map, and one second in the middle of the start has a line said over much text
    looks = [9, 7, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    monkeypatch.setattr(th, "text_and_turn", lambda files: [([(0, 0, 1, 1)] * n, []) for n in looks])
    cuts = [{"recording": "/r", "start": 100.0, "end": 120.0, "note": "n"}]
    assert [(c["start"], c["end"]) for c in pm.trim_menus(cuts, {"/r": {"dialogue": []}})] == [(102.0, 120.0)]
    assert [(c["start"], c["end"]) for c in pm.trim_menus(cuts, {"/r": {"dialogue": [{"start": 100.0, "end": 101.0, "text": "a line over a list"}]}})] == [(100.0, 120.0)]


def test_caption_corrections():
    words = [{"word": w, "start": float(k), "end": k + 1.0} for k, w in enumerate("My name is Arzu. Maybe could- stay, Arzu".split())]
    out, unused = pm.fix_captions(words, [{"wrong": "arzu", "right": "Atsu"}, {"wrong": "Maybe could", "right": "Maybe I could-"}, {"wrong": "not there", "right": "x"}, {"wrong": "stay", "right": ""}])
    assert " ".join(w["word"] for w in out) == "My name is Atsu. Maybe I could- Atsu" and unused == ["not there"]  # (every place; the full stop stays; empty takes words out)
    assert out[3]["start"] == 3.0 and (out[4]["start"], out[6]["end"]) == (4.0, 6.0)  # word for word keeps its time; more words share the old time
    assert pm.fix_captions(words, [])[0] == words and pm.fix_captions([], [{"wrong": "a", "right": "b"}]) == ([], ["a"])
    assert pm.parse_caption_fixes("Arzu => Atsu; a b -> c\n=> nothing\nno arrow") == [("Arzu", "Atsu"), ("a b", "c")]
    p = {}
    assert pm.add_caption_fixes(p, [("Arzu", "Atsu"), ("", "x"), ("same", "same")]) == 1 and pm.add_caption_fixes(p, [("arzu!", "Atsu-san")]) == 1
    assert p["caption_fixes"] == [{"wrong": "arzu!", "right": "Atsu-san"}]  # (the same wrong words: the later correction stands)



def test_several_plans_are_picked_at_once():
    assert pm.picks("3, 7-9,12;3") == ["3", "7", "8", "9", "12"] and pm.picks("6") == ["6"] and pm.picks("") == []
    assert pm.picks("the wolf") == ["the wolf"] and pm.picks("Part 2-3") == ["Part 2-3"]  # (a part of a title stays one pick)
    with pytest.raises(pm.AutocutError):
        pm.picks("9-7")
