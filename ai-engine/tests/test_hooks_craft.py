"""Hook library (offline parts) and the voiceover edit's craft: cold open, pacing, punch-ins, transitions."""

import subprocess

import numpy as np
from conftest import needs_ffmpeg

import hooks as hk
import narrate_craft as nc

RSS = """<rss><channel><item><title>rajasthan royals</title><ht:approx_traffic>2,000+</ht:approx_traffic>
<ht:news_item_title><![CDATA[Royals win &amp; top the table]]></ht:news_item_title></item>
<item><title>gta 6 trailer</title><ht:approx_traffic>50K+</ht:approx_traffic></item></channel></rss>"""


def test_trends_rss_and_titles():
    tr = hk.parse_trends_rss(RSS, "IN")
    assert [(t["topic"], t["traffic"]) for t in tr] == [("rajasthan royals", 2000), ("gta 6 trailer", 50000)]
    assert tr[0]["news"] == ["Royals win & top the table"]
    hooks = hk.title_hooks([{"title": "I Tried GTA 6 For 24 Hours #gaming 🔥", "views": 5_000_000, "url": "u", "niche": "gaming"},
                            {"title": "Why Nobody Plays This Anymore?", "views": 50_000, "url": "v", "niche": "gaming"},
                            {"title": "ok", "views": 1, "url": "w", "niche": "gaming"}])
    assert [h["text"] for h in hooks] == ["I Tried GTA 6 For 24 Hours", "Why Nobody Plays This Anymore?"]
    assert hooks[0]["pattern"] == "challenge" and hooks[1]["pattern"] == "question" and hooks[0]["hot"] > hooks[1]["hot"]


def test_library_merges_ranks_and_cools(tmp_path):
    data = hk.load(tmp_path)
    assert len(data["hooks"]) == 2 * len(hk.PATTERNS)  # the built-in patterns, offline from the first run
    new = [{"text": "GTA 6 just broke every record.", "pattern": "stat_shock", "language": "en", "niche": "gaming", "emotion": "Amazement",
            "why": "", "trend": "gta 6 trailer", "hot": 0.9, "source": "x", "added": hk.now_iso()},
           {"text": "gta 6 just broke every record!", "pattern": "stat_shock", "language": "en", "niche": "gaming", "hot": 0.9, "added": hk.now_iso()},
           {"text": "Old news from last year about rice.", "pattern": "bold_claim", "language": "en", "niche": "food", "hot": 0.9,
            "added": "2025-11-01T00:00:00Z"}]
    assert hk.merge(data, new) == 2  # the same words are the same hook
    data["trends"] = hk.parse_trends_rss(RSS, "IN")
    hk.save(data, tmp_path)
    s = hk.suggest("GTA 6 release", "gaming", "en", n=5, lib=tmp_path)
    assert s["hooks"][0]["text"] == "GTA 6 just broke every record."
    assert len({h["pattern"] for h in s["hooks"]}) == len(s["hooks"])  # a range of hook shapes
    assert [t["topic"] for t in s["trends"]] == ["gta 6 trailer"]
    old = next(h for h in hk.load(tmp_path)["hooks"] if "rice" in h["text"])
    assert hk.heat(old) < 0.01  # a year-old trend has cooled down
    block = hk.prompt_block(s)
    assert "GTA 6 just broke every record." in block and "never copy it" in block
    st = hk.stats(tmp_path)
    assert st["total"] == len(data["hooks"]) and st["hottest"][0]["text"] == "GTA 6 just broke every record."


def L(s, e, emo, inten=0.6, text="a line of narration here", **kw):
    return {"start": s, "end": e, "emotion": emo, "intensity": inten, "text": text, **kw}


def test_pacing_follows_emotion():
    assert nc.pace_factor(L(0, 1, "Rage", 0.9)) < nc.pace_factor(L(0, 1, "Curiosity")) < nc.pace_factor(L(0, 1, "Serenity", 0.4))
    assert 0.6 <= nc.pace_factor(L(0, 1, "Euphoria", 1.0)) and nc.pace_factor(L(0, 1, "Calmness", 0.1)) <= 1.35


def test_teaser_pick_and_fast_open():
    lines = [L(0, 3, "Curiosity", 0.9), L(3, 6, "Nostalgia"), L(6, 9, "Calmness", 0.4), L(9, 12, "Triumph", 0.9), L(12, 20, "Euphoria", 1.0)]
    assert nc.teaser_pick(lines) == 3  # the strongest line short enough for a teaser (not the first two, not an 8 s one)
    assert nc.teaser_pick([L(0, 3, "Calmness"), L(3, 6, "Calmness"), L(6, 9, "Serenity", 0.3)]) is None
    slots = [{"start": 0.0, "end": 3.0, "line": 0, "part": 0, "of": 1}, {"start": 3.0, "end": 7.0, "line": 1, "part": 0, "of": 1}]
    out = nc.fast_open(slots)
    assert [round(s["end"] - s["start"], 2) for s in out[:2]] == [1.5, 1.5] and out[0]["hook"] and not out[-1]["hook"]
    assert out[-1]["start"] == 3.0 and len(out) == 3


def test_emphasis_and_treatment():
    ln = L(0, 4, "Excitement", 0.7, text="This is the biggest launch ever", emphasis=["biggest"])
    t = nc.emphasis_times(ln)
    assert len(t) == 1 and 1.0 < t[0] < 2.0
    assert nc.emphasis_times(L(5, 8, "Triumph", 0.95)) == [5.15]  # a peak line punches in on its first word
    lines = [L(0, 4, "Excitement", 0.7, text="This is the biggest launch ever", emphasis=["biggest"]), L(4, 8, "Nostalgia", 0.7),
             L(8, 12, "Awe", 0.8), L(12, 15, "Thrill", 0.9)]
    slots = [{"start": i * 4.0, "end": i * 4.0 + 4, "line": i, "part": 0} for i in range(4)]
    fps = 30.0
    shots = [{"clip": "a.mp4", "in": 0.0, "frames": 120, "speed": 1.0} for _ in range(4)]
    clips = {"a.mp4": {"fps": 59.94}}
    r = nc.treat(shots, slots, lines, clips, fps)
    assert shots[0]["punches"] and 1.0 < shots[0]["punches"][0] < 2.0
    assert shots[1]["transition_in"]["type"] == "dissolve"  # into a nostalgic part
    assert shots[2]["speed"] == 0.5  # awe on 60 fps footage
    assert shots[3]["transition_in"]["type"] == "fadewhite"  # a flash into an intense part
    assert r == {"punches": 2, "slow": 1, "transitions": 2} or r["punches"] >= 1
    still = [{"clip": "p.mp4", "in": 0.0, "frames": 120, "speed": 1.0}]
    nc.treat(still, slots[2:3], lines, {"p.mp4": {"fps": 60, "still": True}}, fps)
    assert still[0]["speed"] == 1.0  # photos never go slow motion


@needs_ffmpeg
def test_cold_open_voice(tmp_path):
    v = tmp_path / "v.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=300:d=10", "-ar", "48000", str(v)], check=True)
    lines = [L(0, 2, "Curiosity"), L(2.5, 5, "Calmness"), L(5.5, 8.5, "Triumph", 0.9, words=[{"word": "win", "start": 6.0, "end": 6.4}])]
    out, head = nc.with_teaser(str(v), lines, 2, tmp_path / "t.wav")
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(tmp_path / "t.wav")],
                               capture_output=True, text=True).stdout)
    assert abs(dur - (10 + head)) < 0.05 and abs(head - (3.18 + 0.35)) < 0.01
    assert out[0]["teaser"] and out[0]["start"] == 0.06 and out[0]["words"][0]["start"] == np.float64(6.0 - 5.44)
    assert out[1]["start"] == round(head, 3) and out[3]["start"] == round(5.5 + head, 3)


def test_same_scene_is_not_shown_twice_in_a_row():
    import narrate as nr

    e1 = np.zeros((10, 4), np.float32)
    e1[:, 0] = 1.0
    e2 = e1.copy()  # another clip of the same place
    e3 = np.zeros((10, 4), np.float32)
    e3[:, 1] = 1.0  # somewhere else
    clips = [{"emb": e1, "created": 0}, {"emb": e2, "created": 0}, {"emb": e3, "created": 0}]
    assert nr.same_scene(clips, {"clip": 0, "in": 1}, {"clip": 1, "in": 2}) == 1.0
    assert nr.same_scene(clips, {"clip": 0, "in": 1}, {"clip": 2, "in": 2}) == 0.0
    slots = [{"start": 0, "end": 3, "line": 0, "part": 0}, {"start": 3, "end": 6, "line": 1, "part": 0}]
    cands = [[{"clip": 0, "in": 0.0, "score": 2.0, "size": "wide"}],
             [{"clip": 1, "in": 0.0, "score": 2.0, "size": "aerial"}, {"clip": 2, "in": 0.0, "score": 1.5, "size": "aerial"}]]
    assert [p["clip"] for p in nr.choose(slots, cands, clips, story_order=False)] == [0, 2]


def test_your_hook_line_opens_the_script():
    import script_writer as sw

    segs = [{"section": "hook", "text": "Twelve years of waiting.", "emotion": "Awe", "intensity": 0.8, "emphasis": ["Twelve"]},
            {"section": "build", "text": "Then it got delayed.", "emotion": "Tension", "intensity": 0.6}]
    out = sw.with_hook_line(segs, "GTA 6 is finally here!")
    assert [s["text"] for s in out] == ["GTA 6 is finally here!", "Then it got delayed."]  # the model's hook is replaced
    assert out[0]["emotion"] == "Awe" and out[0]["emphasis"] == []
    body = [{"section": "intro", "text": "Let me tell you a story.", "emotion": "Nostalgia", "intensity": 0.5}]
    out = sw.with_hook_line(body, "Nobody saw this coming.")
    assert [s["text"] for s in out] == ["Nobody saw this coming.", "Let me tell you a story."] and out[0]["emotion"] == "Surprise"
    same = sw.with_hook_line([{"section": "hook", "text": "nobody saw this coming", "emotion": "Surprise", "intensity": 0.9}], "Nobody saw this coming!")
    assert [s["text"] for s in same] == ["Nobody saw this coming!"]


def test_topic_hook_search_is_kept(monkeypatch, tmp_path):
    import script_writer as sw
    import topic_video as tv

    monkeypatch.setattr(hk, "youtube_top", lambda q, n=15: [{"title": "I Waited 12 Years For This", "views": 9_000_000, "url": "u", "niche": q}])
    reply = '{"hooks": [{"text": "Twelve years. Six days left.", "pattern": "urgency", "emotion": "Anticipation", "why": "countdown", "hot": 0.9}]}'
    monkeypatch.setattr(tv, "ask_claude_research", lambda prompt, model: reply)
    found = hk.find_for_topic("GTA 6 release", "gaming", "en", lib=tmp_path)
    assert found[0]["text"] == "Twelve years. Six days left." and found[0]["trend"] == "GTA 6 release"
    assert any(h["text"] == "Twelve years. Six days left." for h in hk.load(tmp_path)["hooks"])
    assert "Hooks found online for this exact topic" in hk.topic_block(found)
    monkeypatch.setattr(tv, "ask_claude_research", lambda prompt, model: (_ for _ in ()).throw(sw.AutocutError("offline")))
    found = hk.find_for_topic("GTA 6 release", "gaming", "en", lib=tmp_path)
    assert found[0]["text"] == "I Waited 12 Years For This"  # without Claude: the topic's most-viewed titles


def test_download_commands():
    from pathlib import Path

    import yt_download as yd

    pl = "https://youtube.com/playlist?list=PLabc&si=x"
    cmd = yd.build_cmd(pl, Path("/tmp/bgm"), audio_only=True, items="1-3")
    s = " ".join(cmd)
    assert "--yes-playlist" in cmd and "/tmp/bgm/%(playlist_title)s/%(playlist_index)03d - %(title)s.%(ext)s" in s
    assert "-x" in cmd and "--audio-format" not in cmd and "--playlist-items" in cmd and "--download-archive" in cmd
    assert cmd[-1] == pl
    v = yd.build_cmd("https://youtu.be/abc", Path("/tmp/bgm"), audio_only=False, container="mkv")
    assert "--no-playlist" in v and "/tmp/bgm/%(title)s.%(ext)s" in " ".join(v) and "bv*+ba/b" in v and "mkv" in v
    mp3 = yd.build_cmd(pl, Path("/tmp/bgm"), audio_only=True, audio_format="mp3")
    assert "--audio-format" in mp3 and "mp3" in mp3
    for bad in ({"audio_format": "aac"}, {"container": "avi"}, {"items": "1; rm -rf"}):
        try:
            yd.build_cmd(pl, Path("/tmp/x"), audio_only=True, **bad)
            raise AssertionError(bad)
        except yd.AutocutError:
            pass
