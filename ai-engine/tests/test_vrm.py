"""VRM characters: reading the file (both versions), posing, and drawing (when a GPU context exists)."""

import json
import struct

import numpy as np
import pytest

import vrm
from autocut import AutocutError


def tiny_vrm(tmp_path, old=False):
    """A triangle "face" on a head bone above the hips, with one mouth shape."""
    pos = np.float32([[-0.1, 1.5, 0], [0.1, 1.5, 0], [0, 1.7, 0]])
    delta = np.float32([[0, 0, 0], [0, 0, 0], [0, 0.1, 0]])
    idx = np.uint16([0, 1, 2])
    blob = pos.tobytes() + delta.tobytes() + idx.tobytes() + b"\0\0"
    j = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}, {"buffer": 0, "byteOffset": 36, "byteLength": 36},
                        {"buffer": 0, "byteOffset": 72, "byteLength": 6}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [0.1, 1.7, 0], "min": [-0.1, 1.5, 0]},
                      {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
                      {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 2, "targets": [{"POSITION": 1}]}]}],
        "nodes": [{"name": "hips", "translation": [0, 1, 0], "children": [1]}, {"name": "head", "translation": [0, 0.5, 0], "children": [2]},
                  {"name": "tip", "translation": [0, 0, 0.2]}, {"name": "face", "mesh": 0}],
        "scenes": [{"nodes": [0, 3]}],
    }
    if old:
        j["extensions"] = {"VRM": {"meta": {"title": "Old"}, "humanoid": {"humanBones": [{"bone": "hips", "node": 0}, {"bone": "head", "node": 1}]},
                                   "blendShapeMaster": {"blendShapeGroups": [{"name": "A", "presetName": "a", "binds": [{"mesh": 0, "index": 0, "weight": 100}]},
                                                                             {"name": "Joy", "presetName": "joy", "binds": [{"mesh": 0, "index": 0, "weight": 50}]}]}}}
    else:
        j["extensions"] = {"VRMC_vrm": {"meta": {"name": "New"}, "humanoid": {"humanBones": {"hips": {"node": 0}, "head": {"node": 1}}},
                                        "expressions": {"preset": {"aa": {"morphTargetBinds": [{"node": 3, "index": 0, "weight": 1.0}]}}}}}
    js = json.dumps(j).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942) + blob
    p = tmp_path / ("old.vrm" if old else "new.vrm")
    p.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)
    return str(p)


def test_reads_both_versions(tmp_path):
    m = vrm.Model(tiny_vrm(tmp_path))
    assert m.name == "New" and m.v1 and m.bones == {"hips": 0, "head": 1} and m.shapes == {"aa": [(0, 0, 1.0)]}
    rows, delta = m.target(0, 0, 0)
    assert rows.tolist() == [2] and delta[0].tolist() == pytest.approx([0, 0.1, 0])
    assert m.target(0, 0, 9)[0].size == 0  # a shape that is not there moves nothing
    top, low, _ = m.frame_box()
    assert top == pytest.approx(1.7) and low < top
    old = vrm.Model(tiny_vrm(tmp_path, old=True))
    assert old.name == "Old" and not old.v1 and old.shapes == {"aa": [(0, 0, 1.0)], "happy": [(0, 0, 0.5)]}


def test_pose_turns_a_bone_in_the_characters_own_space(tmp_path):
    for old in (False, True):
        m = vrm.Model(tiny_vrm(tmp_path, old=old))
        rest = m.world({})
        fwd = -1.0 if old else 1.0  # VRM 0.x is stored facing -Z, and turned round when read
        assert rest[2][:3, 3] == pytest.approx([0, 1.5, 0.2 * fwd], abs=1e-6)
        assert m.root[2, 2] == pytest.approx(fwd)
        # a turn to the character's left swings a point on its Z axis towards X; the hips stay
        w = m.world({"head": vrm.euler(yaw=90)})
        assert w[2][:3, 3] == pytest.approx([0.2 * fwd, 1.5, 0], abs=1e-6) and w[0][:3, 3] == pytest.approx([0, 1, 0], abs=1e-6)
        assert m.world({"nose": vrm.euler(yaw=90)}) == pytest.approx(rest)  # an unknown bone is ignored
        assert m.world({}, {"hips": [0, 0.1, 0]})[1][:3, 3] == pytest.approx([0, 1.6, 0])


def test_damaged_files_are_refused_not_crashed_on(tmp_path):
    bad = tmp_path / "bad.vrm"
    bad.write_bytes(b"hello")
    with pytest.raises(AutocutError):
        vrm.Model(str(bad))
    with pytest.raises(AutocutError):
        vrm.Model(str(tmp_path / "missing.vrm"))
    good = open(tiny_vrm(tmp_path), "rb").read()
    cut = tmp_path / "cut.vrm"
    cut.write_bytes(good[:-40])  # the vertex data runs past the end of the file
    m = vrm.Model(str(cut))
    with pytest.raises(AutocutError):
        m.accessor(2)


def test_material_parts():
    assert [vrm.material_part(n) for n in ("Hair_00_HAIR", "EyeIris_00_EYE", "Face_00_SKIN", "Tops_01_CLOTH", "glass")] == ["hair", "eyes", "skin", "clothes", "other"]


def test_draws_a_frame_and_the_mouth_shape_changes_it(tmp_path):
    m = vrm.Model(tiny_vrm(tmp_path))
    try:
        r = vrm.Renderer(m, 128)
    except AutocutError:
        pytest.skip("no graphics context here")
    a = np.frombuffer(r.render(), np.uint8).reshape(128, 128, 4)
    b = np.frombuffer(r.render(shapes={"aa": 1.0}), np.uint8).reshape(128, 128, 4)
    c = np.frombuffer(r.render(), np.uint8).reshape(128, 128, 4)
    assert a[..., 3].max() == 255 and a[0, 0, 3] == 0  # the character over a clear background
    assert (a != b).any() and (a == c).all()  # the shape moves it, and it goes back
    r.close()
