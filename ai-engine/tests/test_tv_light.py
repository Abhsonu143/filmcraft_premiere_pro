"""TV Light Sync: geometry, light model, path choice, relight maps, calibration (no recordings needed)."""

import numpy as np
import pytest

import tv_light as tl


def test_order_quad():
    q = tl.order_quad([[0.9, 0.8], [0.1, 0.1], [0.1, 0.9], [0.8, 0.2]])
    assert np.allclose(q, [[0.1, 0.1], [0.8, 0.2], [0.9, 0.8], [0.1, 0.9]])


def test_light_model_recovers_room_response():
    rng = np.random.default_rng(0)
    n = 300
    a = {"edge": {}, "glow": {}}
    for z in tl.ZONES:
        E = rng.uniform(0, 1, (n, 3)).astype(np.float32)
        A = np.float32([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.05, 0.0, 0.6]])
        G = np.vstack([E[:1], E[:-1]]) @ A.T + 0.02  # one step of lag, like the HDMI sync
        a["edge"][z], a["glow"][z] = E.tolist(), G.tolist()
    m = tl.fit_light(a)
    assert m["lag"] == 1 and all(m[z]["r2"] > 0.99 for z in tl.ZONES)


def test_path_prefers_continuous_playback_and_min_runs():
    n, T = 200, 60
    D = np.ones((T, n), np.float32)
    for t in range(T):
        D[t, 50 + t] = 0.0  # a continuous stretch that matches perfectly
        D[t, (7 * t) % n] = -0.2  # isolated slightly better frames everywhere: a jump fest if allowed
    quads = np.tile(np.float32([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]), (n, 1, 1))
    path, cost = tl.choose_path(D, quads, np.zeros(T, np.float32))
    assert int((np.diff(path) != 1).sum()) <= 2
    assert tl.min_runs(np.array([0, 1, 2, 90, 91, 5, 6, 7, 8, 9, 10, 11, 12]), 100, 5).tolist()[3:5] == [3, 4]


def test_relight_weights_zero_on_screen_and_selflit():
    shape = (tl.GH, tl.GW)
    selflit = np.zeros(shape, np.uint8)
    r, c = tl.zone_slices()["right"]
    selflit[r.start + 5 : r.start + 25, c.start + 5 : c.start + 25] = 1  # a monitor on the right
    a = {"selflit": np.packbits(selflit).tolist(), "selflit_shape": list(shape)}
    w = tl.relight_weights(a)
    u0, v0 = int(-tl.U0 * tl.TS), int(-tl.V0 * tl.TS)
    assert w[v0 + 50, u0 + 50].sum() == 0  # inside the TV: replaced by the clip, not relit
    assert w[v0 + 80, u0 - 10, 0] > 0.8  # just left of the screen: the left zone
    assert w[r.start + 15, c.start + 15].sum() < 0.05  # the monitor keeps its own light


def test_screen_match_maps_levels():
    a = {"screen": {"p05": 0.05, "p50": 0.3, "p95": 0.7}}
    img = np.linspace(0, 1, 300, dtype=np.float32).reshape(10, 10, 3)
    out = tl.screen_match(img, a)
    lum = out @ np.float32([0.2126, 0.7152, 0.0722])
    assert abs(float(np.median(lum)) - 0.3) < 0.05 and np.all(np.diff(lum.ravel()) >= -1e-4)


def test_calibration_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(tl.ac, "CACHE_DIR", tmp_path)
    rec = tmp_path / "rec.mp4"
    rec.write_bytes(b"x")
    assert tl.calibration(str(rec)) is None
    tl.save_calibration(str(rec), 12.5, [[0.1, 0.1], [0.9, 0.12], [0.88, 0.8], [0.12, 0.82]])
    c = tl.calibration(str(rec))
    assert c["confirmed"] and c["ref_t"] == 12.5 and c["quad"][1] == [0.9, 0.12]


def test_make_validates(tmp_path):
    with pytest.raises(tl.AutocutError):
        tl.make(str(tmp_path / "missing.mp4"), str(tmp_path), str(tmp_path / "o.mp4"))
    with pytest.raises(tl.AutocutError):
        tl.make(str(tmp_path / "x.mp4"), str(tmp_path), str(tmp_path / "o.mp4"), orientation="square")


def test_good_quad_rejects_triangles():
    assert tl.good_quad(np.float32([[0.1, 0.2], [0.8, 0.25], [0.78, 0.6], [0.12, 0.65]]))
    assert not tl.good_quad(np.float32([[0.02, 0.53], [0.74, 0.28], [0.72, 0.51], [0.02, 0.53]]))  # GX011573's first guess
    assert not tl.good_quad(np.float32([[0.1, 0.1], [0.12, 0.1], [0.12, 0.12], [0.1, 0.12]]))


def test_batch_keeps_going_and_names_outputs(tmp_path, monkeypatch):
    made = []

    def fake_make(clip, folder, out, orientation="horizontal", relight=1.0, fps=30.0, recording=None, bezel=0.0):
        if "bad" in clip:
            raise tl.AutocutError("cannot read")
        made.append(out)

    monkeypatch.setattr(tl, "make", fake_make)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "a - on TV vertical.mp4").write_bytes(b"old")  # never overwritten
    res = tl.make_batch([str(tmp_path / "a.mp4"), str(tmp_path / "bad.mp4"), str(tmp_path / "c.mp4")], str(tmp_path), str(tmp_path / "out"), "vertical")
    assert [p.split("/")[-1] for p in res["outputs"]] == ["a - on TV vertical 2.mp4", "c - on TV vertical.mp4"]
    assert res["failed"][0]["clip"].endswith("bad.mp4")
    with pytest.raises(tl.AutocutError):
        tl.make_batch([str(tmp_path / "bad.mp4")], str(tmp_path), str(tmp_path / "out"))


def test_bezel_thin_sides_wider_bottom():
    img = np.full((720, 1280, 3), 200, np.uint8)
    out = tl.add_bezel(img, 0.012)
    assert out.shape == img.shape
    b = round(0.012 * 1280)
    assert out[360, 2].max() < 40 and out[360, 1280 - 3].max() < 40 and out[3, 640].max() < 40  # dark sides and top
    assert out[720 - b - 3, 640].max() < 40  # the bottom bezel is wider than the sides
    assert out[720 - 2 * b - 4, 640].min() > 150 and out[b + 3, b + 3].min() > 150  # the picture inside
