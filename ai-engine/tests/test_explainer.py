"""explainer: the plan of an explainer video (research, chapters, truth, review)."""

import json
from pathlib import Path

import pytest

import explainer as ex
import script_writer as sw
import topic_video as tv
from autocut import AutocutError

RESEARCH = {"title_idea": "Generators", "angle": "a", "summary": "s", "facts": [
    {"fact": "A generator function returns a generator iterator.", "url": "https://docs.example/gen", "source": "Docs", "quote": "It returns a generator iterator, when called."},
    {"fact": "Generators were added in 2.2.", "url": "https://docs.example/history", "source": "Docs", "quote": "Generators first appeared in version 9.9 of the language."},
    {"fact": "Calling next() on a finished generator raises StopIteration.", "url": "https://gone.example/x", "source": "Blog", "quote": "A finished generator raises StopIteration when asked again."},
    {"fact": "People confuse them with lists.", "url": "https://docs.example/gen", "source": "Docs"}], "examples": [{"what": "countdown", "detail": "def c(n): ...", "url": "u"}]}
PLAN = {"title": "Generators, Finally", "promise": "p", "audience": "a", "chapters": [
    {"title": "What yield does", "goal": "g", "hook": "This function never returns.", "says": ["one", "two"],
     "shows": [{"type": "code", "what": "def c(n):\n  yield n"}, {"type": "hologram", "what": "something odd"}, {"type": "flow", "what": ""}],
     "claims": [{"claim": "It returns an iterator.", "fact": 1}, {"claim": "Since 2.2.", "fact": "F2"}, {"claim": "Made up.", "fact": 99}], "ends_on": "e", "how_said": "dry", "catch": "c",
     "clip_title": "t", "seconds": 4000},
    {"title": "The bug", "goal": "g", "hook": "h", "says": ["x"], "shows": [{"type": "debug", "what": "StopIteration"}], "claims": [{"claim": "It raises.", "fact": 3}], "seconds": "soon"},
    {"no": "title"}, "junk"], "open_questions": ["which version?"]}
PAGES = {"https://docs.example/gen": "Intro. It returns a generator iterator, when called. More.", "https://docs.example/history": "Generators first appeared in version 2.2 of the language."}


def read(url):
    if url not in PAGES:
        raise OSError("gone")
    return PAGES[url]


def test_quotes_are_looked_up_in_their_source_and_the_plan_shows_it():
    data = json.loads(json.dumps(RESEARCH))
    assert ex.confirm_quotes(data, read) == (1, 3)
    assert [f["confirmed"] for f in data["facts"]] == ["yes", "not found", "page not read", "no quote"]
    plan = ex.clean_plan(PLAN, data)
    one, two = plan["chapters"]
    assert len(plan["chapters"]) == 2 and one["seconds"] == 180 and two["seconds"] == 45
    assert [s["type"] for s in one["shows"]] == ["code", "card"]  # (an unknown picture becomes a card; one without words is dropped)
    assert [(c["fact"], c["confirmed"]) for c in one["claims"]] == [(1, "yes"), (2, "not found"), (0, "no source")] and two["claims"][0]["confirmed"] == "page not read"
    md = ex.plan_md({**plan, "status": "waiting for your approval", "want": "w", "notes": [{"when": "now", "chapter": 2, "text": "error first"}]})
    assert "3 not confirmed word for word" in md and "NOT CONFIRMED: the sentence is not in the page" in md and "NO SOURCE" in md and "chapter 2: error first" in md
    assert "## 1. What yield does  (180 s)" in md and "- **code**: def c(n)" in md and "approve the plan" in md
    with pytest.raises(AutocutError):
        ex.clean_plan({"chapters": []}, data)


def test_notes_are_about_a_chapter_or_the_whole_video():
    assert ex.parse_note("3: show the error first", 5) == {"chapter": 3, "text": "show the error first"}
    assert ex.parse_note("Chapter 2 - shorter", 5) == {"chapter": 2, "text": "shorter"} and ex.parse_note("9: x", 5) == {"chapter": 0, "text": "9: x"}
    assert ex.parse_note("make it funnier", 5)["chapter"] == 0 and ex.parse_note("  ", 5) is None


def test_a_plan_is_written_reviewed_changed_and_approved(tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(tv, "ask_claude_research", lambda prompt, model: (asked.append("research"), json.dumps(RESEARCH))[1])
    monkeypatch.setattr(tv, "page_text", lambda url, limit=0: ("t", read(url)))

    def claude(prompt, model=None, timeout=600):
        if "wants changes" in prompt:
            asked.append("revise")
            assert "chapter 2: show the traceback first" in prompt and "the whole video: no jokes" in prompt
            return json.dumps({**PLAN, "title": "Generators, Plainly", "changed": ["Chapter 2 opens on the traceback."]})
        asked.append("plan")
        assert "F1. A generator function" in prompt and "EVERY chapter must work cut out" in prompt and ("focus on bugs" in prompt) == ("how generators work" in prompt)
        return json.dumps(PLAN)

    monkeypatch.setattr(sw, "ask_claude", claude)
    out = str(tmp_path / "gen")
    r = ex.make("how generators work", out, minutes=5, kind="coding", hints="focus on bugs", review=True)
    assert r["status"] == "waiting for your approval" and r["chapters"] == 2 and r["unconfirmed"] == 3 and asked == ["research", "plan"]
    assert "waiting for your approval" in (tmp_path / "gen" / "plan.md").read_text()
    r = ex.make("how generators work", out, review=True, notes=["2: show the traceback first", "no jokes"])  # (research and plan are kept; only the notes are worked in)
    plan = json.loads((tmp_path / "gen" / "plan.json").read_text())
    assert asked == ["research", "plan", "revise"] and plan["title"] == "Generators, Plainly" and [n["chapter"] for n in plan["notes"]] == [2, 0]
    assert plan["changed"] == ["Chapter 2 opens on the traceback."] and r["status"] == "waiting for your approval" and plan["hints"] == "focus on bugs"
    assert ex.make("how generators work", out, approve=True)["status"] == "approved" and asked == ["research", "plan", "revise"]
    assert ex.make("another thing", str(tmp_path / "other"))["status"] == "approved"  # (no review asked for: written, kept, approved)
    with pytest.raises(AutocutError):
        ex.make("  ")


SCRIPT = {"chapters": [
    {"scenes": [{"type": "code", "title": "c.py", "lines": "def c(n):\n    yield n"}, {"type": "article", "fact": 1}, {"type": "quote", "fact": 2}, {"type": "hologram", "what": "odd"}],
     "lines": [{"text": "This function never returns.", "scene": 1, "step": 2, "focus": [2, "x"], "emotion": "curious", "emphasis": ["never"], "pause_after": 9},
               {"text": "It returns a generator iterator.", "shown": "It returns a generator iterator.", "scene": 1, "step": 1, "fact": 1},
               {"text": "The docs say so.", "scene": 2, "step": 2, "fact": "1"}, {"text": "Since two point two.", "shown": "Since 2.2.", "scene": 3, "fact": 2}, {"text": " "}]},
    {"scenes": [{"type": "debug", "title": "c.py", "code": ["g = c(1)", "next(g)", "next(g)"], "run": ["$ python c.py", "StopIteration"], "cause": [3], "fixed": ["for v in c(1):", "    print(v)"],
                 "changed": [1, 2], "passes": [{"t": "ok", "text": "1"}]},
                {"type": "flow", "boxes": ["call", "yield", "resume", "done"]}, {"type": "chart", "bars": [{"label": "a", "value": "many"}]}],
     "lines": [{"text": "Ask a finished generator again and it raises.", "scene": 1, "step": 1, "fact": 3}, {"text": "Here is the cause.", "scene": 1, "step": 2},
               {"text": "Loop instead.", "scene": 1, "step": 4}, {"text": "Call, yield, resume, done.", "scene": 2, "step": 4}, {"text": "A chart of nothing.", "scene": 3}]}]}


def planned(tmp_path, monkeypatch, checks=None):
    """An approved plan in a folder, and Claude answering for the script step."""
    asked = []
    monkeypatch.setattr(tv, "ask_claude_research", lambda prompt, model: json.dumps(RESEARCH))
    monkeypatch.setattr(tv, "page_text", lambda url, limit=0: ("Generators — Docs", read(url)))

    def claude(prompt, model=None, timeout=600):
        if prompt.startswith("Write the script"):
            asked.append("script")
            assert "exact sentence (confirmed in the page)" in prompt and "NOT confirmed: do not show or quote it" in prompt and '"words": 432' in prompt
            return json.dumps(SCRIPT)
        if prompt.startswith("Check the script"):
            asked.append("check")
            assert "1.1  This function never returns." in prompt and "2.s2" in prompt
            return json.dumps({"problems": checks if checks is not None else []})
        if "read the script" in prompt:
            asked.append("revise")
            assert "chapter 2: end on the loop" in prompt and '"chapter": 2' in prompt and '"chapter": 1' not in prompt
            two = json.loads(json.dumps(SCRIPT["chapters"][1]))
            two["lines"] = two["lines"][:3]
            return json.dumps({"chapters": [{"chapter": 2, **two}], "changed": ["Chapter 2 ends on the loop."]})
        asked.append("plan")
        return json.dumps(PLAN)

    monkeypatch.setattr(sw, "ask_claude", claude)
    out = str(tmp_path / "gen")
    assert ex.make("how generators work", out, review=True, approve=True)["status"] == "approved"
    return out, asked


def test_a_script_is_written_checked_reviewed_and_changed(tmp_path, monkeypatch):
    out, asked = planned(tmp_path, monkeypatch, checks=[
        {"where": "1.4", "problem": "The page says 9.9, not 2.2.", "fix": "Generators have been in the language for a long time."},
        {"where": "2.s1", "problem": "The output is not what Python prints.", "fix": ""}, {"where": "9.9", "problem": "nowhere", "fix": "x"}, "junk"])
    r = ex.make("how generators work", out, review=True, to="script")
    assert r["step"] == "script" and r["status"] == "waiting for your approval" and r["open_checks"] == 2 and asked == ["plan", "script", "check"]
    sc = json.loads((tmp_path / "gen" / "script.json").read_text())
    one, two = sc["chapters"]
    # a source page is only shown for a sentence that stands in it: fact 2's does not, so it is a plain card
    assert [s["type"] for s in one["scenes"]] == ["code", "article", "card", "card"] and one["scenes"][2]["text"] == "Generators were added in 2.2."
    assert one["scenes"][0]["lines"] == ["def c(n):", "    yield n"] and two["scenes"][0]["run"] == [{"t": "in", "text": "$ python c.py"}, {"t": "err", "text": "StopIteration"}]
    assert two["scenes"][2]["type"] == "card"  # (a chart without numbers)
    first, second = one["lines"][0], one["lines"][1]
    assert (first["emotion"], first["focus"], first["pause_after"], first["step"]) == ("Curiosity", [2], 1.5, 2) and second["step"] == 2 and "shown" not in second  # (a scene only builds up)
    assert len(one["lines"]) == 4 and one["lines"][2]["fact"] == 1
    assert one["lines"][3]["text"] == "Generators have been in the language for a long time." and "shown" not in one["lines"][3]  # (the check's correction)
    md = (tmp_path / "gen" / "script.md").read_text()
    assert "was: “Since two point two.”" in md and "**not corrected: for you to decide**" in md and "- [1.2] This function never returns." in md and "the source page with this sentence marked" in md
    # the same call again changes nothing and asks nobody; a note changes only its chapter; then it is approved
    assert ex.make("how generators work", out, review=True, to="script")["status"] == "waiting for your approval" and asked == ["plan", "script", "check"]
    r = ex.make("typed differently this time", out, review=True, to="script", notes=["2: end on the loop"])
    sc = json.loads((tmp_path / "gen" / "script.json").read_text())
    assert asked[3:] == ["revise", "check"] and len(sc["chapters"][1]["lines"]) == 3 and sc["chapters"][0] == one and sc["changed"] == ["Chapter 2 ends on the loop."]
    assert r["status"] == "waiting for your approval" and sc["notes"][0]["chapter"] == 2
    assert ex.make("how generators work", out, review=True, approve=True, to="script")["status"] == "approved" and len(asked) == 5
    # a note for the plan after that: the plan is worked over and approved, so its script is written again and waits
    r = ex.make("how generators work", out, review=True, approve=True, to="script", notes=["2: show the traceback first"], note_for="plan")
    assert asked[5:] == ["plan", "script", "check"] and r["step"] == "script" and r["status"] == "waiting for your approval"
    with pytest.raises(AutocutError):
        ex.make("how generators work", str(tmp_path / "none"), notes=["2: x"], note_for="script")  # (no script there to change)


def test_without_review_the_script_counts_as_approved(tmp_path, monkeypatch):
    out, asked = planned(tmp_path, monkeypatch)
    r = ex.make("how generators work", out, to="script")
    assert r["status"] == "approved" and r["open_checks"] == 0 and r["words"] > 20
    with pytest.raises(AutocutError):
        ex.make("how generators work", out, to="film")


def test_the_passage_around_a_confirmed_sentence_is_kept():
    data = json.loads(json.dumps(RESEARCH))
    ex.confirm_quotes(data, lambda url: ("Generators — Docs", "First paragraph here.\nIntro words. It returns a generator iterator, when called. More after it.\nLast."))
    ctx = data["facts"][0]["context"]
    assert ctx == {"before": "Intro words.", "exact": "It returns a generator iterator, when called.", "after": "More after it.", "title": "Generators — Docs"}
    assert "context" not in data["facts"][1] and ex.find_passage("nothing like it", "It returns a generator iterator") is None


def test_pictures_follow_the_voice_and_every_scene_draws(tmp_path, monkeypatch):
    import explain_draw as xd
    import explain_video as xv

    out, _ = planned(tmp_path, monkeypatch)
    ex.make("how generators work", out, to="script")
    sc = json.loads((tmp_path / "gen" / "script.json").read_text())
    facts = json.loads((tmp_path / "gen" / "research.json").read_text())["facts"]
    vs = xv.voice_script(sc)
    assert len(vs["segments"]) == 9 and vs["segments"][3]["pause_after"] >= 0.9 and vs["segments"][0]["emphasis"] == ["never"]  # (a breath between chapters)
    segs, t = [], 0.5
    for s in vs["segments"]:
        segs.append({"start": t, "end": t + 2.0, "text": s["text"]})
        t += 2.5
    states, cues, chapters = xv.timeline(sc, segs)
    # (the first two lines are the same scene and step with another line lit: two pictures)
    assert [s["key"][:3] for s in states] == [(0, 1, 2), (0, 1, 2), (0, 2, 2), (0, 3, 0), (1, 1, 1), (1, 1, 2), (1, 1, 4), (1, 2, 4), (1, 3, 0)]
    assert states[0]["start"] == 0.0 and abs(states[2]["start"] - (5.5 - xv.LEAD)) < 1e-6  # (a picture is there just before its line)
    assert [c["chapter"] for c in chapters] == [0, 1] and chapters[0]["end"] == chapters[1]["start"] == 10.5 - 0.3 and chapters[1]["end"] == segs[-1]["end"] + 0.6
    assert cues[0]["words"][0] == "This" and all(a["end"] <= b["start"] + 1e-6 for a, b in zip(cues, cues[1:])) and len(cues[0]["at"]) == len(cues[0]["words"])
    words = xv.timeline(sc, segs, mode="words", tall=True)[1]
    assert max(len(c["words"]) for c in words) <= 3 and xv.timeline(sc, segs, mode="none")[1] == []
    with pytest.raises(AutocutError):
        xv.timeline(sc, segs[:-1])
    desc = xv.description(sc, facts, chapters)
    assert "0:00 What yield does" in desc and "0:10 The bug" in desc and "- Docs: https://docs.example/gen" in desc
    for look, (w, h) in (("clean", (640, 360)), ("viral", (360, 640))):
        pen = xd.Pen(w, h, look)
        for ch in sc["chapters"]:
            for scene in ch["scenes"]:
                for step in (0, 1, 2, 4):
                    fact = facts[scene["fact"] - 1] if scene.get("fact") else None
                    img, marks = xd.draw_scene(pen, scene, step, (1,), title=ch["title"], fact=fact)
                    assert img.size == (w, h) and (bool(marks) == (scene["type"] == "article" and step != 1))
        cap, pos = xd.draw_caption(pen, ["It", "returns", "an", "iterator."], 1)
        assert cap.width <= w and 0 <= pos[0] and pos[1] + cap.height <= h
    marked = xd.draw_scene(xd.Pen(640, 360), sc["chapters"][0]["scenes"][1], 2, fact=facts[0])
    plain = xd.draw_scene(xd.Pen(640, 360), sc["chapters"][0]["scenes"][1], 1, fact=facts[0])[0]
    x0, y0, x1, y1 = marked[1][0]
    assert marked[0].crop((x0, y0, x1, y1)).tobytes() != plain.crop((x0, y0, x1, y1)).tobytes()  # (the marker is on the sentence)
    half = xv._sweep(plain, marked[0], marked[1], 0.5)
    mid = (x0 + x1) // 2
    assert half.crop((x0, y0, mid - 4, y1)).tobytes() == marked[0].crop((x0, y0, mid - 4, y1)).tobytes() and half.crop((mid + 4, y0, x1, y1)).tobytes() == plain.crop((mid + 4, y0, x1, y1)).tobytes()


def test_a_sources_own_words_are_shown_only_where_the_creator_allows():
    data = json.loads(json.dumps(RESEARCH))
    ex.confirm_quotes(data, read)
    gov = {"fact": "The act has 44 sections.", "url": "https://www.meity.gov.in/act", "source": "MeitY", "quote": "q", "confirmed": "yes"}
    news = {**gov, "url": "https://www.example-news.com/a"}
    assert ex.open_source("https://en.wikipedia.org/wiki/X") and ex.open_source("https://indiacode.nic.in/x") and ex.open_source("https://www.gov.uk/x")
    assert not ex.open_source("https://docs.example/gen") and not ex.open_source("https://notgov.com/x.gov")
    assert ex.may_show(gov, "open") and ex.may_show(news, "any") and not ex.may_show(news, "open") and not ex.may_show(gov, "none")
    assert not ex.may_show({**gov, "confirmed": "not found"}, "any")
    facts = [gov, news]
    assert ex.clean_scene({"type": "article", "fact": 1}, facts, "open") == {"type": "article", "fact": 1}
    assert ex.clean_scene({"type": "quote", "fact": 2}, facts, "open")["type"] == "card" and ex.clean_scene({"type": "article", "fact": 1}, facts, "none")["type"] == "card"
    plan = {**ex.clean_plan(PLAN, data), "show_sources": "none", "language": "hi"}
    prompt = ex.script_prompt(plan, data)
    assert 'Do NOT use "article" or "quote" scenes' in prompt and "NOT to be shown on screen" in prompt and "Devanagari" in prompt
    assert "confirmed in the page" in ex.script_prompt({**plan, "show_sources": "any"}, data)


def test_a_photo_is_found_and_credited_or_the_scene_becomes_a_card(tmp_path, monkeypatch):
    from PIL import Image

    import explain_draw as xd

    def download(url, dst):
        Image.new("RGB", (1200, 800), (40, 90, 160)).save(dst)
        return "good" in url

    monkeypatch.setattr(tv, "download", download)
    find = lambda q: [{"url": "https://x/bad.png"}, {"url": "https://x/good.jpg?w=1", "title": "Hampi", "author": "A. Rao", "license": "CC BY-SA 4.0", "provider": "Wikimedia Commons",  # noqa: E731
                                                     "page": "https://commons/hampi"}] if "hampi" in q else []
    script = {"chapters": [{"title": "t", "scenes": [ex.clean_scene({"type": "footage", "search": "hampi  temple", "caption": "Hampi, 14th century"}, []),
                                                     ex.clean_scene({"type": "footage", "search": "nothing like it", "caption": "An idea"}, []),
                                                     ex.clean_scene({"type": "footage"}, [])], "lines": []}]}
    assert ex.fetch_footage(script, tmp_path, find) == 1
    one, two, three = script["chapters"][0]["scenes"]
    assert one["credit"] == "Hampi, A. Rao, CC BY-SA 4.0, Wikimedia Commons" and one["file"].endswith("media/c1-s1.jpg") and one["page"] == "https://commons/hampi"
    assert two == {"type": "card", "text": "An idea", "sub": ""} and three["type"] == "card"
    assert "found: Hampi, A. Rao" in ex._scene_text(one, [])
    assert ex.clean_scene({**one}, [])["file"] == one["file"] and ex.fetch_footage(script, tmp_path, lambda q: 1 / 0) == 0  # (kept: not looked for again)
    img, _ = xd.draw_scene(xd.Pen(640, 360), one, title="t")
    assert all(abs(a - b) <= 3 for a, b in zip(img.getpixel((320, 150)), (40, 90, 160)))
    assert xd.draw_scene(xd.Pen(640, 360), {**one, "file": "/nowhere.jpg"}, title="t")[0].size == (640, 360)  # (a lost file: a card, not a crash)


def test_code_is_run_and_the_real_output_shown():
    import shutil

    import explain_run as xr

    if not shutil.which("sandbox-exec"):
        pytest.skip("no macOS sandbox here")
    script = {"chapters": [{"scenes": [
        {"type": "code", "title": "f.py", "lines": ["def f(a, L=[]):", "    L.append(a)", "    return L", "print(f(1))", "print(f(2))"]},
        {"type": "terminal", "lines": [{"t": "out", "text": "python3 f.py"}, {"t": "out", "text": "[1]"}, {"t": "out", "text": "[2]"}]},
        {"type": "terminal", "lines": [{"t": "in", "text": ">>> x = [1, 2]"}, {"t": "in", "text": ">>> for v in x:"}, {"t": "in", "text": "...     print(v * 2)"}, {"t": "in", "text": "..."},
                                       {"t": "out", "text": "2"}, {"t": "out", "text": "4"}, {"t": "in", "text": ">>> x[5]"}, {"t": "err", "text": "IndexError: list index out of range"}]},
        {"type": "debug", "title": "g.py", "code": ["import socket", "socket.create_connection(('example.com', 80), 2)"], "run": [{"t": "in", "text": "$ python g.py"}, {"t": "out", "text": "connected"}],
         "fixed": ["open('/Users/Shared/ei-sandbox-test.txt', 'w')"], "passes": [{"t": "ok", "text": "saved"}]},
        {"type": "terminal", "lines": [{"t": "in", "text": "$ cargo run"}, {"t": "out", "text": "hello"}]}]}]}
    found = xr.check_script(script)
    scenes = script["chapters"][0]["scenes"]
    assert script["ran"]["runs"] == 4
    assert [r["text"] for r in scenes[1]["lines"]] == ["python3 f.py", "[1]", "[1, 2]"]  # (the script was wrong: the real output is shown)
    assert scenes[2]["lines"][-1]["text"] == "IndexError: list index out of range" and not any(f["where"] == "1.s3" for f in found)  # (a session that was right is left alone)
    # the sandbox: no network, and nothing is written outside the temporary folder
    assert scenes[3]["run"][-1]["t"] == "err" and "connected" not in str(scenes[3]["run"]) and "PermissionError" in scenes[3]["passes"][-1]["text"]
    assert not Path("/Users/Shared/ei-sandbox-test.txt").exists()
    assert [f["where"] for f in found] == ["1.s2", "1.s4", "1.s4", "1.s4"] and found[0]["was"] == "[1] / [2]" and found[0]["now"] == "[1] / [1, 2]" and found[0]["corrected"]
    assert "FIXED code still fails" in found[-1]["problem"] and not found[-1]["corrected"]
    assert scenes[4]["lines"][1]["text"] == "hello"  # (not Python: not run, not touched)
    assert xr.short_error('Traceback (most recent call last):\n  File "/tmp/x/f.py", line 3, in <module>\n    f()\nValueError: no') == [
        "Traceback (most recent call last):", '  File "f.py", line 3', "ValueError: no"]


def test_hindi_is_drawn_with_its_letters_joined():
    import explain_draw as xd

    if xd.shaped("हिंदी", 40, True, (0, 0, 0)) is None:
        pytest.skip("no macOS text drawing here")
    from PIL import Image, ImageFont

    img = Image.new("RGB", (600, 120), (255, 255, 255))
    d = xd.Draw(img)
    font = xd.Pen(1920, 1080).font("bold", 48, "हिंदी")
    d.text((10, 10), "डिफ़ॉल्ट सिर्फ़ एक बार", font=font, fill=(0, 0, 0))
    plain = Image.new("RGB", (600, 120), (255, 255, 255))
    from PIL import ImageDraw

    ImageDraw.Draw(plain).text((10, 10), "डिफ़ॉल्ट सिर्फ़ एक बार", font=font, fill=(0, 0, 0))
    assert img.getbbox() and img.tobytes() != plain.tobytes()  # (drawn, and not the letter-by-letter way)
    assert 0 < d.textlength("डिफ़ॉल्ट सिर्फ़ एक बार", font=font) < ImageDraw.Draw(plain).textlength("डिफ़ॉल्ट सिर्फ़ एक बार", font=font) * 1.2
    assert d.textlength("plain", font=font) == ImageDraw.Draw(plain).textlength("plain", font=font)
    pen = xd.Pen(360, 640, "viral")
    cap, _ = xd.draw_caption(pen, ["यह", "बग", "है"], 1)
    assert cap.getbbox() is not None
    assert isinstance(font, ImageFont.FreeTypeFont)


def test_real_pictures_are_planned_looked_at_and_used_only_when_they_fit(tmp_path):
    import explain_draw as xd
    import explain_media as xm
    from PIL import Image

    plan = ex.clean_plan({"title": "t", "chapters": [
        {"title": "The lander", "seconds": 30, "media": [{"what": "the lander on the ground", "search": "Vikram lander", "kind": "video", "why": "w"},
                                                         {"what": "the control room", "search": "ISRO control room", "kind": "photo"}, {"what": "no search"}, "junk"]},
        {"title": "The maths", "seconds": 30}]}, RESEARCH)
    assert [m["kind"] for m in plan["chapters"][0]["media"]] == ["video", "photo"] and plan["chapters"][1]["media"] == []
    plan["media"] = ex.media_needs(plan)
    assert [(m["id"], m["chapter"]) for m in plan["media"]] == [("m1", 1), ("m2", 1)]

    mine = tmp_path / "mine.png"
    Image.new("RGB", (1200, 800), (200, 40, 40)).save(mine)
    seen = {}

    def look(prompt, where):
        seen["prompt"] = prompt
        assert (where / "m1.jpg").is_file() and "YOUR OWN FILE" in prompt
        return {"choices": [{"id": "m1", "take": "A", "why": "the lander"}, {"id": "m2", "take": None, "why": "only rockets, no control room"}]}

    sig = ex.plan_sig(plan)
    assert xm.find(plan, tmp_path, [str(mine)], search=lambda need: [], look=look) == (1, 2)
    m1, m2 = plan["media"]
    assert Path(m1["file"]).is_file() and m1["credit"] == "your own file" and m1["kind"] == "photo" and "file" not in m2 and "control room" in m2["missing"]
    assert ex.plan_sig(plan) != sig  # (the script is written again for other pictures)
    md = ex.plan_md({**plan, "promise": "", "audience": "", "status": "draft"} | {"chapters": [dict(c, goal="", hook="", says=[], shows=[], claims=[], ends_on="", how_said="", catch="", clip_title="") for c in plan["chapters"]]})
    assert "could not find" in md and "found: your own file" in md and "NOT FOUND" in md
    # found again for the same words: not looked for twice
    assert xm.find(plan, tmp_path, [], search=lambda need: [], look=lambda *a: {}) == (1, 2)
    assert plan["media"][0]["file"] == m1["file"]

    # the script may use only what was found: full screen, or dimmed behind a card; never behind code
    media = ex.found_media(plan)
    full = ex.clean_scene({"type": "footage", "media": "M1", "caption": "The lander"}, [], "any", media)
    assert full["file"] == m1["file"] and full["credit"] == "your own file"
    assert ex.clean_scene({"type": "footage", "media": "m2", "caption": "Control room"}, [], "any", media)["type"] == "card"
    card = ex.clean_scene({"type": "card", "text": "19 minutes", "behind": "m1"}, [], "any", media)
    assert card["behind"]["file"] == m1["file"] and ex.clean_scene(card, [], "any", media)["behind"]["id"] == "m1"
    assert "behind" not in ex.clean_scene({"type": "code", "title": "a.py", "lines": ["x = 1"], "behind": "m1"}, [], "any", media)
    assert "behind" not in ex.clean_scene({"type": "card", "text": "x", "behind": "m2"}, [], "any", media)

    pen = xd.Pen(1280, 720, "dark")
    plain = xd.draw_scene(pen, {"type": "card", "text": "19 minutes"}, title="T")[0]
    backed = xd.draw_scene(pen, card, title="T")[0]
    assert backed.getpixel((1200, 400))[0] > plain.getpixel((1200, 400))[0] + 15  # (the red photo shows through, dimmed)
    a, b = xd.draw_scene(pen, full, title="T", t=0.0)[0], xd.draw_scene(pen, full, title="T", t=6.0)[0]
    assert a.getpixel((640, 300))[0] > 150 and xd.is_live(full) and not xd.is_live(card)


def test_a_found_video_is_cut_to_the_part_that_shows_it(tmp_path):
    import shutil
    import subprocess

    import explain_media as xm

    if not shutil.which("ffmpeg"):
        pytest.skip("needs ffmpeg")
    raw = tmp_path / "v.mp4"  # 10 s of a black "title slate", then 30 s of picture
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:d=10:r=30", "-f", "lavfi", "-i", "testsrc2=s=320x180:d=30:r=30",
                    "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]", "-pix_fmt", "yuv420p", str(raw)], check=True)
    need = {"id": "m1", "what": "the test picture"}
    (tmp_path / "candidates").mkdir()
    asked = {}

    def look(prompt, where):
        asked["prompt"] = prompt
        assert (where / "m1-parts.jpg").is_file()
        return {"first": 4, "last": 6}  # (frames 1-3 are the slate: 2, 6 and 10 s)

    start, length = xm.best_part(need, raw, 40.0, tmp_path / "candidates", look)
    assert 10.0 < start < 14.0 and 8.0 < length <= xm.CLIP_SECONDS and "title slates" in asked["prompt"]
    got = xm.keep(need, {"own": True, "path": str(raw), "kind": "video"}, tmp_path, look)
    assert got["kind"] == "video" and 8.0 < got["seconds"] <= xm.CLIP_SECONDS + 0.2
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", got["file"], "-frames:v", "1", str(tmp_path / "first.png")], check=True)
    from PIL import Image

    assert max(Image.open(tmp_path / "first.png").convert("L").tobytes()) > 120  # (it opens on the picture, not on the slate)
    # nothing in it shows the thing: not used at all; and when it cannot be looked at, a part after the opening is kept
    cand = {"own": True, "path": str(raw), "kind": "video"}
    assert xm.keep(need, cand, tmp_path, lambda p, w: {"first": None, "last": None}) is None and "no part" in cand["nothing"]
    assert xm.keep(need, {"own": True, "path": str(raw), "kind": "video"}, tmp_path, lambda p, w: {})["seconds"] > 10
