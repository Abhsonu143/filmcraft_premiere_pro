"""TV Light Sync: framing on the TV, stretches of the recording, and the screen's look."""

import numpy as np

import tv_light as tv


def quads(x0, y0, x1, y1, n=20):
    q = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return np.repeat(q[None], n, axis=0)


def test_vertical_framing_makes_the_tv_big_and_keeps_it_whole():
    # a vertical recording (2160x3840) with the TV in the upper middle
    Q = quads(0.10, 0.30, 0.90, 0.52)
    x0, y0, cw, ch = tv.framing_rect(Q, 2160, 3840, 1080, 1920, "vertical")
    assert 0 <= x0 and x0 + cw <= 1 and 0 <= y0 and y0 + ch <= 1
    assert x0 <= 0.10 and x0 + cw >= 0.90 and y0 <= 0.30 and y0 + ch >= 0.52  # the whole TV
    assert 0.8 / cw > 0.9  # it fills more than 90% of the width (was about 75%)
    centre = ((0.30 + 0.52) / 2 - y0) / ch
    assert 0.4 < centre < 0.6  # low enough that little of the dark floor shows


def test_a_tv_at_the_edge_of_the_recording_is_never_cut():
    Q = quads(0.0, 0.30, 0.85, 0.52)
    x0, y0, cw, ch = tv.framing_rect(Q, 2160, 3840, 1080, 1920, "vertical")
    assert x0 == 0.0 and x0 + cw >= 0.85


def test_stretches_of_the_recording():
    path = np.array([0, 1, 2, 10, 11, 5])
    segs = tv.segments_of(path, 30.0)
    assert [round(t, 2) for t, _ in segs] == [0.0, 1.0, 0.5] and [n for _, n in segs] == [9, 6, 3]


def test_the_screen_is_bright_and_keeps_its_blacks():
    a = {"screen": {"p05": 0.05, "p50": 0.12, "p95": 0.3}}  # the camera saw a dim, flat screen
    clip = np.zeros((90, 160, 3), np.float32)
    clip[:, :80] = 0.01
    clip[:, 80:] = 0.6
    out = tv.screen_look(clip, a)
    lum = out @ np.float32([0.2126, 0.7152, 0.0722])
    assert np.percentile(lum, 95) > 0.7  # bright, like a screen in a dark room
    assert np.percentile(lum, 5) < 0.04  # dark parts stay dark (no grey haze)
