"""Progress for the AI Jobs window, topic stages that resume, and parallel renders of one shot."""

import contextlib
import io
import json
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
