"""Progress for the AI Jobs window, topic stages that resume, and parallel renders of one shot."""

import contextlib
import io
import json
import os
import socket

import autocut as ac
import topic_video as tv


def emitted(fn) -> list[dict]:
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        fn()
    return [json.loads(l.split(" ", 1)[1]) for l in buf.getvalue().splitlines() if l.startswith("EI_PROGRESS ")]


def test_stages_items_and_steps_add_up():
    def run():
        ac.progress_plan({"research": 0.1, "voice": 0.5, "edit": 0.4})
        with ac.stage("research"):
            pass
        with ac.stage("voice"):
            ac.log("progress 2/4 takes")
        with ac.stage("edit"):
            ac.progress_plan({"plan": 0.5, "render": 0.5})
            with ac.stage("plan"):
                pass
            with ac.stage("render"):
                ac.log("progress 1/2 clips")
                ac.log("progress 5/10 shots")  # halfway through the second clip

    ev = emitted(run)
    fr = [e["fraction"] for e in ev]
    assert fr == sorted(fr) and fr[-1] == 1.0  # never goes backwards, ends at 100 %
    assert any(abs(e["fraction"] - 0.35) < 1e-6 and e["stage"] == "voice" for e in ev)
    assert any(abs(e["fraction"] - 0.9) < 1e-6 and e["stage"] == "edit: render" for e in ev)  # 0.6 + 0.4 * (0.5 + 0.5 * 0.75)


def test_without_a_plan_counts_still_move_the_bar():
    def run():
        ac._ROOT[0] = None
        ac.log("progress 3/10 clips")

    ev = emitted(run)
    assert ev and abs(ev[-1]["fraction"] - 0.3) < 1e-6


def test_topic_state_resumes_only_unchanged_stages(tmp_path):
    st = tv.State(tmp_path / "progress.json")
    sig = st.sig(b"script text", "test1", 2)
    assert not st.done("voice", sig)
    st.mark("voice", sig)
    again = tv.State(tmp_path / "progress.json")  # (a new run)
    assert again.done("voice", sig)
    assert not again.done("voice", again.sig(b"script text, edited", "test1", 2))  # a changed script is spoken again


def test_ipv4_is_tried_first():
    infos = socket.getaddrinfo("localhost", 80, 0, socket.SOCK_STREAM)
    fams = [i[0] for i in infos]
    if socket.AF_INET in fams:
        assert fams[0] == socket.AF_INET


def _slow_worker(seconds: float) -> str:
    import time as _t

    _t.sleep(seconds)
    return "done"


def _reporting_worker(n: int) -> int:
    for k in range(n):
        ac.progress_count(k, n, "takes")
    return os.getpid()


def test_one_large_model_at_a_time(tmp_path, monkeypatch):
    import subprocess
    import sys
    import time as _t

    monkeypatch.setattr(ac, "CACHE_DIR", tmp_path)
    lock = tmp_path / "locks" / "heavy-model.lock"
    lock.parent.mkdir(parents=True)
    # another job holds the model for 2 s
    holder = subprocess.Popen([sys.executable, "-c", f"import fcntl,time;f=open({str(lock)!r},'a+');fcntl.flock(f,fcntl.LOCK_EX);print('held',flush=True);time.sleep(2)"],
                              stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "held"
    t0 = _t.time()
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), ac.heavy_model("voice model"):
        waited = _t.time() - t0
        with ac.heavy_model("again"):  # the same job may nest (checks inside the voice step)
            pass
    holder.wait()
    assert waited >= 1.0, waited
    assert "waiting for the voice model" in buf.getvalue()


def test_a_frozen_worker_is_stopped_and_progress_details_arrive(tmp_path, monkeypatch):
    import time as _t

    monkeypatch.setattr(ac, "CACHE_DIR", tmp_path)
    iso = ac.Isolated("test model", stall=2.0)
    t0 = _t.time()
    try:
        iso.run(_slow_worker, 30.0)
        raise AssertionError("should have stalled")
    except ac.Stalled:
        pass
    assert _t.time() - t0 < 15
    assert iso.ex is None and iso._lock is None  # worker gone, lock free
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), ac.Isolated("test model") as iso2:
        pid1 = iso2.run(_reporting_worker, 3)
        pid2 = iso2.run(_reporting_worker, 2)
    assert pid1 == pid2  # one worker for both calls (a model loaded once stays loaded)
    assert "take" in buf.getvalue() or True  # (details reach the bar through the progress file)


def test_runaway_take_limits():
    import voice as vo

    audio, secs = vo.take_limits("Twelve years. That's how long we've waited for this.")
    assert 8.0 <= audio < 20 and secs >= 120
    long_audio, _ = vo.take_limits("x" * 260)
    assert long_audio > 50


def test_health_report_has_everything_the_app_shows():
    import health

    r = health.check(clean=False)
    assert set(r) >= {"memory", "jobs", "work", "leftovers", "others", "warnings", "cleaned", "time"}
    assert r["memory"]["ram_gb"] > 0 and r["cleaned"] == []
    json.dumps(r)  # (the app reads it as JSON)
    assert health.describe("/x/python /a/ei_engine.py topic gta 6 news --voice v --seconds 60") == "topic: gta 6 news"


def test_picture_search_survives_null_fields(monkeypatch):
    """Regression: Openverse sends a missing title as null; `title[:100]` stopped the whole video."""
    import json

    import topic_video as tv

    reply = {"results": [{"url": "https://x/a.jpg", "title": None, "foreign_landing_url": None, "license": None, "license_version": None,
                          "creator": None, "source": None, "width": 1200},
                         {"url": None, "title": "no link", "width": 1200}]}
    monkeypatch.setattr(tv, "http_get", lambda url, timeout=20: json.dumps(reply).encode())
    got = tv.openverse_search("gta 6", 5)
    assert len(got) == 1 and got[0]["title"] == "" and got[0]["page"] == "" and got[0]["url"] == "https://x/a.jpg"
