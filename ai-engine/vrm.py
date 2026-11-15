"""vrm: load a VRM character (the open 3D avatar format, a glTF binary) and draw it on the Mac's GPU.

A VRM file carries what a talking character needs: a skinned body with named bones (head, neck,
chest, arms…), and named face shapes: the mouth for each vowel (aa, ih, ou, ee, oh), blinking and
expressions (happy, angry, sad, relaxed, surprised). Both versions are read: VRM 1.0 and VRM 0.x
(what VRoid Studio exports by default); 0.x names are mapped to the 1.0 ones and the model is
turned to face the same way.

`Model` holds the data (numpy only, no GPU). `Renderer` draws one frame for a pose (a rotation per
bone) and face-shape weights, off screen, with toon shading, and returns RGBA bytes
(premultiplied, bottom row first). Nothing here loads an AI model; a frame takes a few
milliseconds.

Not drawn (yet): hair/cloth physics (spring bones), outlines, MToon rim light and UV animation.
"""

from __future__ import annotations

import io
import json
import math
import struct
from pathlib import Path

import numpy as np

from autocut import AutocutError

CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
# VRM 0.x face-shape names -> VRM 1.0
OLD_SHAPES = {"a": "aa", "i": "ih", "u": "ou", "e": "ee", "o": "oh", "joy": "happy", "sorrow": "sad", "fun": "relaxed",
              "blink_l": "blinkLeft", "blink_r": "blinkRight", "lookup": "lookUp", "lookdown": "lookDown", "lookleft": "lookLeft",
              "lookright": "lookRight"}
MAX_FILE = 300 * 1024 * 1024


def rot(axis: str, deg: float) -> np.ndarray:
    """3x3 rotation about x, y or z (degrees)."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def euler(pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Rotation of a body part in the character's own space: pitch = nod (+ down), yaw = turn
    (+ to its left), roll = tilt (+ towards its left shoulder). Degrees."""
    return rot("y", yaw) @ rot("x", pitch) @ rot("z", roll)


def quat_to_mat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float64)


class Model:
    """A VRM file's meshes, skeleton and face shapes."""

    def __init__(self, path: str):
        p = Path(path)
        if not p.is_file():
            raise AutocutError(f"character file not found: {path}")
        if p.stat().st_size > MAX_FILE:
            raise AutocutError("the character file is larger than 300 MB")
        raw = p.read_bytes()
        if len(raw) < 20 or raw[:4] != b"glTF":
            raise AutocutError("not a VRM file (a VRM is a binary glTF, .vrm or .glb)")
        self.j, self.bin = self._chunks(raw)
        self.path = str(p)
        ext = self.j.get("extensions") or {}
        self.v1 = "VRMC_vrm" in ext
        vrm = ext.get("VRMC_vrm") or ext.get("VRM")
        if not vrm:
            raise AutocutError("this file is a plain 3D model, not a VRM character (no face shapes or named bones)")
        self.meta = vrm.get("meta") or {}
        self.name = str(self.meta.get("name") or self.meta.get("title") or p.stem)
        self._nodes()
        self._bones(vrm)
        self._shapes(vrm)
        self._targets: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}

    # ---- file ---------------------------------------------------------------------------------

    @staticmethod
    def _chunks(raw: bytes) -> tuple[dict, bytes]:
        pos, js, blob = 12, None, b""
        while pos + 8 <= len(raw):
            n, kind = struct.unpack_from("<II", raw, pos)
            body = raw[pos + 8 : pos + 8 + n]
            if kind == 0x4E4F534A:
                js = body
            elif kind == 0x004E4942 and not blob:
                blob = body
            pos += 8 + n + (-n % 4)
        if js is None:
            raise AutocutError("the VRM file is damaged (no scene description)")
        try:
            return json.loads(js), blob
        except ValueError as e:
            raise AutocutError(f"the VRM file is damaged ({e})") from None

    def accessor(self, i: int) -> np.ndarray:
        """One glTF accessor as a (count, components) array (float32 when normalised)."""
        try:
            a = self.j["accessors"][i]
            n, dt = NC[a["type"]], np.dtype(CT[a["componentType"]])
            count = int(a["count"])
            if "bufferView" in a:
                arr = self._view(a["bufferView"], int(a.get("byteOffset", 0)), dt, count, n)
            else:
                arr = np.zeros((count, n), dt)
            sp = a.get("sparse")
            if sp:
                arr = arr.copy()
                k = int(sp["count"])
                idx = self._view(sp["indices"]["bufferView"], int(sp["indices"].get("byteOffset", 0)), np.dtype(CT[sp["indices"]["componentType"]]), k, 1)[:, 0]
                arr[idx.astype(np.int64)] = self._view(sp["values"]["bufferView"], int(sp["values"].get("byteOffset", 0)), dt, k, n)
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise AutocutError(f"the VRM file is damaged (accessor {i}: {e})") from None
        if a.get("normalized") and dt.kind in "iu":
            return (arr.astype(np.float32) / float(np.iinfo(dt).max)).clip(-1.0, 1.0)
        return arr

    def _view(self, view: int, offset: int, dt: np.dtype, count: int, n: int) -> np.ndarray:
        bv = self.j["bufferViews"][view]
        off = int(bv.get("byteOffset", 0)) + offset
        size = dt.itemsize * n
        stride = int(bv.get("byteStride") or size)
        if count <= 0:
            return np.zeros((0, n), dt)
        if off < 0 or stride < size or off + stride * (count - 1) + size > len(self.bin):
            raise ValueError("data outside the file")
        if stride == size:
            return np.frombuffer(self.bin, dt, count * n, off).reshape(count, n)
        rows = np.frombuffer(self.bin, np.uint8, stride * (count - 1) + size, off)
        pick = (np.arange(count)[:, None] * stride + np.arange(size)[None, :]).ravel()
        return rows[pick].view(dt).reshape(count, n)

    def image(self, i: int):
        """A texture image (PIL, RGBA), or None when it cannot be read."""
        from PIL import Image

        try:
            im = self.j["images"][i]
            bv = self.j["bufferViews"][im["bufferView"]]
            off = int(bv.get("byteOffset", 0))
            return Image.open(io.BytesIO(self.bin[off : off + int(bv["byteLength"])])).convert("RGBA")
        except Exception:  # noqa: BLE001 (a broken texture is drawn without one)
            return None

    # ---- skeleton -----------------------------------------------------------------------------

    def _nodes(self) -> None:
        nodes = self.j.get("nodes") or []
        n = len(nodes)
        self.parent = [-1] * n
        for i, nd in enumerate(nodes):
            for c in nd.get("children") or []:
                if 0 <= int(c) < n:
                    self.parent[int(c)] = i
        self.t = np.zeros((n, 3))
        self.s = np.ones((n, 3))
        self.r = np.tile(np.eye(3), (n, 1, 1))
        for i, nd in enumerate(nodes):
            if "matrix" in nd:
                m = np.asarray(nd["matrix"], np.float64).reshape(4, 4).T
                self.t[i] = m[:3, 3]
                sc = np.linalg.norm(m[:3, :3], axis=0)
                self.s[i] = np.where(sc > 1e-9, sc, 1.0)
                self.r[i] = m[:3, :3] / self.s[i]
            else:
                self.t[i] = nd.get("translation", [0, 0, 0])
                self.s[i] = nd.get("scale", [1, 1, 1])
                self.r[i] = quat_to_mat(nd.get("rotation", [0, 0, 0, 1]))
        # parents before children (a damaged file with a loop is cut where it repeats)
        self.order, seen = [], set()
        for i in range(n):
            chain = []
            k = i
            while k >= 0 and k not in seen and len(chain) <= n:
                chain.append(k)
                seen.add(k)
                k = self.parent[k]
            self.order.extend(reversed(chain))
        # VRM 0.x characters face -Z: turn them to face +Z like VRM 1.0
        self.root = np.eye(4)
        if not self.v1:
            self.root[:3, :3] = rot("y", 180)
        rest = self.world({})
        self.rest_world = rest
        self.parent_rot = np.tile(np.eye(3), (n, 1, 1))  # rest rotation of each node's parent, in the world
        for i in range(n):
            m = rest[self.parent[i]][:3, :3] if self.parent[i] >= 0 else self.root[:3, :3]
            sc = np.linalg.norm(m, axis=0)
            self.parent_rot[i] = m / np.where(sc > 1e-9, sc, 1.0)

    def _bones(self, vrm: dict) -> None:
        hb = (vrm.get("humanoid") or {}).get("humanBones") or {}
        n = len(self.parent)
        if isinstance(hb, dict):
            pairs = [(k, (v or {}).get("node")) for k, v in hb.items()]
        else:
            pairs = [(b.get("bone"), b.get("node")) for b in hb if isinstance(b, dict)]
        self.bones = {str(k): int(v) for k, v in pairs if k and isinstance(v, int) and 0 <= v < n}
        if "head" not in self.bones:
            raise AutocutError("the character has no head bone (not a humanoid VRM)")

    def _shapes(self, vrm: dict) -> None:
        """name -> [(mesh, target, weight)]"""
        self.shapes: dict[str, list[tuple[int, int, float]]] = {}
        nodes = self.j.get("nodes") or []
        if self.v1:
            ex = vrm.get("expressions") or {}
            for name, e in {**(ex.get("custom") or {}), **(ex.get("preset") or {})}.items():
                binds = []
                for b in (e or {}).get("morphTargetBinds") or []:
                    nd = b.get("node")
                    if isinstance(nd, int) and 0 <= nd < len(nodes) and "mesh" in nodes[nd]:
                        binds.append((int(nodes[nd]["mesh"]), int(b.get("index", 0)), float(b.get("weight", 1.0))))
                if binds:
                    self.shapes[name] = binds
        else:
            for g in (vrm.get("blendShapeMaster") or {}).get("blendShapeGroups") or []:
                preset = str(g.get("presetName") or "").lower()
                name = OLD_SHAPES.get(preset, preset) if preset and preset != "unknown" else str(g.get("name") or "")
                binds = [(int(b["mesh"]), int(b.get("index", 0)), float(b.get("weight", 100.0)) / 100.0) for b in g.get("binds") or [] if "mesh" in b]
                if name and binds:
                    self.shapes.setdefault(name, binds)

    def world(self, pose: dict[str, np.ndarray], lift: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """World matrix of every node for a pose (bone name -> 3x3 rotation in the character's
        space, on top of the rest pose). `lift`: bone name -> movement (metres)."""
        n = len(self.parent)
        r = self.r
        t = self.t
        if pose or lift:
            r, t = r.copy(), t.copy()
            for name, m in pose.items():
                i = self.bones.get(name)
                if i is not None:
                    p = self.parent_rot[i]
                    r[i] = p.T @ m @ p @ self.r[i]
            for name, d in (lift or {}).items():
                i = self.bones.get(name)
                if i is not None:
                    t[i] = t[i] + self.parent_rot[i].T @ np.asarray(d, np.float64)
        local = np.zeros((n, 4, 4))
        local[:, :3, :3] = r * self.s[:, None, :]
        local[:, :3, 3] = t
        local[:, 3, 3] = 1.0
        out = np.zeros((n, 4, 4))
        for i in self.order:
            p = self.parent[i]
            out[i] = (out[p] if p >= 0 else self.root) @ local[i]
        return out

    def bone_pos(self, name: str) -> np.ndarray | None:
        i = self.bones.get(name)
        return None if i is None else self.rest_world[i][:3, 3]

    # ---- face shapes --------------------------------------------------------------------------

    def target(self, mesh: int, prim: int, index: int) -> tuple[np.ndarray, np.ndarray]:
        """A morph target of one primitive as (rows that move, how far)."""
        key = (mesh, prim, index)
        if key not in self._targets:
            empty = (np.zeros(0, np.int64), np.zeros((0, 3), np.float32))
            try:
                tg = self.j["meshes"][mesh]["primitives"][prim].get("targets") or []
                d = self.accessor(tg[index]["POSITION"]).astype(np.float32) if 0 <= index < len(tg) and "POSITION" in tg[index] else None
            except (KeyError, IndexError, TypeError):
                d = None
            if d is None:
                self._targets[key] = empty
            else:
                rows = np.nonzero(np.abs(d).max(axis=1) > 1e-7)[0]
                self._targets[key] = (rows, d[rows])
        return self._targets[key]

    def frame_box(self) -> tuple[float, float, float]:
        """(top of the head, stomach height, centre x) in the rest pose, for a head-to-stomach view."""
        head = self.bone_pos("head")
        top = float(head[1]) + 0.2
        ys = []
        for m in self.j.get("meshes") or []:
            for pr in m.get("primitives") or []:
                a = self.j["accessors"][pr["attributes"]["POSITION"]]
                if "max" in a:
                    ys.append(float(a["max"][1]))
        if ys:
            top = max(float(head[1]) + 0.05, min(max(ys), float(head[1]) + 0.45))
        hips, spine = self.bone_pos("hips"), self.bone_pos("spine")
        low = float(spine[1]) if spine is not None else (float(hips[1]) + 0.1 if hips is not None else top - 0.6)
        return top, low, float(head[0])


VERT = """
#version 410
uniform mat4 u_vp;
uniform mat4 u_model;
uniform int u_skinned;
uniform sampler2D u_joints;
in vec3 in_pos;
in vec3 in_nrm;
in vec2 in_uv;
in vec4 in_j;
in vec4 in_w;
out vec3 v_n;
out vec2 v_uv;
mat4 jm(float f) {
    int i = int(f + 0.5);
    return mat4(texelFetch(u_joints, ivec2(0, i), 0), texelFetch(u_joints, ivec2(1, i), 0),
                texelFetch(u_joints, ivec2(2, i), 0), texelFetch(u_joints, ivec2(3, i), 0));
}
void main() {
    mat4 m = u_model;
    if (u_skinned == 1) {
        m = in_w.x * jm(in_j.x) + in_w.y * jm(in_j.y) + in_w.z * jm(in_j.z) + in_w.w * jm(in_j.w);
    }
    v_n = mat3(m) * in_nrm;
    v_uv = in_uv;
    gl_Position = u_vp * (m * vec4(in_pos, 1.0));
}
"""

FRAG = """
#version 410
uniform sampler2D u_tex;
uniform int u_has_tex;
uniform vec4 u_color;
uniform vec3 u_shade;
uniform vec3 u_tint;
uniform float u_cutoff;
uniform vec3 u_light;
in vec3 v_n;
in vec2 v_uv;
out vec4 f;
void main() {
    vec4 c = u_color;
    if (u_has_tex == 1) c *= texture(u_tex, v_uv);
    if (c.a < u_cutoff) discard;
    vec3 n = normalize(v_n);
    if (!gl_FrontFacing) n = -n;
    float lit = smoothstep(-0.06, 0.06, dot(n, u_light) + 0.2);   // two tones with a soft edge
    vec3 rgb = c.rgb * mix(u_shade, vec3(1.0), lit) * u_tint;
    f = vec4(rgb * c.a, c.a);                                       // premultiplied
}
"""


def perspective(fov_deg: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    return np.array([[f, 0, 0, 0], [0, f, 0, 0], [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]], np.float64)


def look_at(eye, target) -> np.ndarray:
    eye, target = np.asarray(eye, np.float64), np.asarray(target, np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd) or 1.0
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right) or 1.0
    up = np.cross(right, fwd)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = right, up, -fwd
    m[:3, 3] = -m[:3, :3] @ eye
    return m


class Renderer:
    """Draws a Model off screen. One per process (it owns a GPU context)."""

    def __init__(self, model: Model, size: int = 720, samples: int = 4, tints: dict[str, list[float]] | None = None):
        import moderngl

        self.gl = moderngl
        self.m = model
        self.size = int(max(64, min(2160, size)))
        try:
            self.ctx = moderngl.create_standalone_context()
        except Exception as e:  # noqa: BLE001
            raise AutocutError(f"could not start the graphics card for the character ({e})") from None
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        sz = (self.size, self.size)
        self.msaa = self.ctx.framebuffer([self.ctx.renderbuffer(sz, 4, samples=samples)], self.ctx.depth_renderbuffer(sz, samples=samples))
        self.out = self.ctx.framebuffer([self.ctx.renderbuffer(sz, 4)])
        self.tints = tints or {}
        self.textures: dict[int, object] = {}
        self.skins: list[dict] = []
        self.draws: list[dict] = []
        self._build()
        self.set_camera()

    def _texture(self, tex_index):
        if tex_index is None:
            return None
        try:
            src = self.m.j["textures"][tex_index].get("source")
        except (KeyError, IndexError, TypeError):
            return None
        if src is None:
            return None
        if src not in self.textures:
            im = self.m.image(src)
            if im is None:
                self.textures[src] = None
            else:
                if max(im.size) > 2048:  # plenty for a small picture-in-picture, and a quarter of the memory
                    im.thumbnail((2048, 2048))
                t = self.ctx.texture(im.size, 4, im.tobytes())
                t.build_mipmaps()
                t.filter = (self.gl.LINEAR_MIPMAP_LINEAR, self.gl.LINEAR)
                self.textures[src] = t
        return self.textures[src]

    def _material(self, i) -> dict:
        mats = self.m.j.get("materials") or []
        mat = mats[i] if isinstance(i, int) and 0 <= i < len(mats) else {}
        pbr = mat.get("pbrMetallicRoughness") or {}
        mtoon = (mat.get("extensions") or {}).get("VRMC_materials_mtoon") or {}
        shade = np.asarray(mtoon.get("shadeColorFactor", [0.78, 0.74, 0.8]), np.float64)[:3]
        base = np.asarray(pbr.get("baseColorFactor", [1, 1, 1, 1]), np.float64)
        # the shade tone as a share of the lit tone, never darker than 55% (a small face must stay readable)
        ratio = np.clip(shade / np.maximum(base[:3], 1e-3), 0.55, 1.0)
        mode = mat.get("alphaMode", "OPAQUE")
        name = str(mat.get("name", ""))
        return {"color": base, "shade": ratio, "tex": self._texture((pbr.get("baseColorTexture") or {}).get("index")), "blend": mode == "BLEND",
                "cutoff": float(mat.get("alphaCutoff", 0.5)) if mode == "MASK" else (0.004 if mode == "BLEND" else 0.0),
                "double": bool(mat.get("doubleSided")), "name": name, "part": material_part(name)}

    def _build(self) -> None:
        j = self.m.j
        for sk in j.get("skins") or []:
            joints = [int(x) for x in sk.get("joints") or []]
            ibm = self.m.accessor(sk["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1).astype(np.float64) if "inverseBindMatrices" in sk else np.tile(np.eye(4), (len(joints), 1, 1))
            if len(ibm) < len(joints):
                raise AutocutError("the VRM file is damaged (a skin has fewer bind matrices than joints)")
            tex = self.ctx.texture((4, max(1, len(joints))), 4, dtype="f4")
            tex.filter = (self.gl.NEAREST, self.gl.NEAREST)
            self.skins.append({"joints": np.asarray(joints, np.int64), "ibm": ibm[: len(joints)], "tex": tex})
        for ni, nd in enumerate(j.get("nodes") or []):
            if "mesh" not in nd:
                continue
            mi = int(nd["mesh"])
            skin = nd.get("skin")
            for pi, pr in enumerate(j["meshes"][mi].get("primitives") or []):
                at = pr.get("attributes") or {}
                if "POSITION" not in at or pr.get("mode", 4) != 4:
                    continue
                pos = self.m.accessor(at["POSITION"]).astype(np.float32)
                n = len(pos)
                nrm = self.m.accessor(at["NORMAL"]).astype(np.float32) if "NORMAL" in at else np.tile(np.float32([0, 0, 1]), (n, 1))
                uv = self.m.accessor(at["TEXCOORD_0"]).astype(np.float32) if "TEXCOORD_0" in at else np.zeros((n, 2), np.float32)
                skinned = isinstance(skin, int) and 0 <= skin < len(self.skins) and "JOINTS_0" in at and "WEIGHTS_0" in at
                if skinned:
                    jn = self.m.accessor(at["JOINTS_0"]).astype(np.float32)
                    jn = np.minimum(jn, max(0, len(self.skins[skin]["joints"]) - 1))
                    w = self.m.accessor(at["WEIGHTS_0"]).astype(np.float32)
                    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-6)
                else:
                    jn, w = np.zeros((n, 4), np.float32), np.zeros((n, 4), np.float32)
                if not (len(nrm) == len(uv) == len(jn) == len(w) == n):
                    continue
                idx = self.m.accessor(pr["indices"]).astype(np.uint32).ravel() if "indices" in pr else np.arange(n, dtype=np.uint32)
                idx = idx[: len(idx) // 3 * 3]
                if len(idx) == 0 or int(idx.max()) >= n:
                    continue
                vpos = self.ctx.buffer(pos.tobytes(), dynamic=True)
                rest = self.ctx.buffer(np.hstack([nrm, uv, jn, w]).astype(np.float32).tobytes())
                vao = self.ctx.vertex_array(self.prog, [(vpos, "3f", "in_pos"), (rest, "3f 2f 4f 4f", "in_nrm", "in_uv", "in_j", "in_w")],
                                            self.ctx.buffer(idx.tobytes()))
                self.draws.append({"vao": vao, "vpos": vpos, "pos": pos, "node": ni, "mesh": mi, "prim": pi, "skin": skin if skinned else None,
                                   "mat": self._material(pr.get("material")), "morphed": False, "has_targets": bool(pr.get("targets"))})
        if not self.draws:
            raise AutocutError("the character has nothing to draw")
        self.draws.sort(key=lambda d: d["mat"]["blend"])  # see-through parts (lashes, blush) after the solid ones

    def set_camera(self, zoom: float = 1.0, raise_by: float = 0.0, fov: float = 18.0) -> None:
        """Head-to-stomach view from the front. zoom > 1 = closer; raise_by moves the view up (metres)."""
        top, low, cx = self.m.frame_box()
        half = max(0.12, (top - low) / 2 * 1.06) / max(0.2, zoom)
        cy = (top + low) / 2 + raise_by + (top - low) / 2 * (1 - 1 / max(0.2, zoom))  # zooming keeps the head in view
        dist = half / math.tan(math.radians(fov) / 2)
        vp = perspective(fov, 0.05, dist + 10.0) @ look_at([cx, cy, dist], [cx, cy, 0.0])
        self.prog["u_vp"].write(vp.T.astype("f4").tobytes())
        light = np.array([0.35, 0.45, 0.82])
        self.prog["u_light"].value = tuple(light / np.linalg.norm(light))

    def render(self, pose: dict[str, np.ndarray] | None = None, shapes: dict[str, float] | None = None,
               lift: dict[str, np.ndarray] | None = None, background: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> bytes:
        """One frame as RGBA bytes (size x size, premultiplied alpha, bottom row first)."""
        gl, ctx = self.gl, self.ctx
        world = self.m.world(pose or {}, lift)
        for sk in self.skins:
            mats = world[sk["joints"]] @ sk["ibm"]
            sk["tex"].write(mats.transpose(0, 2, 1).astype("f4").tobytes())
        # face shapes: moved vertices of each primitive that has any
        moves: dict[tuple[int, int], np.ndarray] = {}
        for name, w in (shapes or {}).items():
            w = float(min(1.0, max(0.0, w)))
            if w < 1e-3:
                continue
            for mesh, index, bw in self.m.shapes.get(name, []):
                for d in self.draws:
                    if d["mesh"] == mesh and d["has_targets"]:
                        rows, delta = self.m.target(mesh, d["prim"], index)
                        if len(rows):
                            key = (mesh, d["prim"])
                            if key not in moves:
                                moves[key] = d["pos"].copy()
                            moves[key][rows] += delta * (w * bw)
        self.msaa.use()
        self.msaa.clear(*background)
        ctx.enable(gl.DEPTH_TEST)
        ctx.blend_func = (gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA)
        for d in self.draws:
            key = (d["mesh"], d["prim"])
            if key in moves:
                d["vpos"].write(moves[key].tobytes())
                d["morphed"] = True
            elif d["morphed"]:
                d["vpos"].write(d["pos"].tobytes())
                d["morphed"] = False
            mat = d["mat"]
            self.prog["u_skinned"].value = 1 if d["skin"] is not None else 0
            if d["skin"] is not None:
                self.skins[d["skin"]]["tex"].use(1)
                self.prog["u_joints"].value = 1
            else:
                self.prog["u_model"].write(world[d["node"]].T.astype("f4").tobytes())
            if mat["tex"] is not None:
                mat["tex"].use(0)
                self.prog["u_tex"].value = 0
            self.prog["u_has_tex"].value = 1 if mat["tex"] is not None else 0
            self.prog["u_color"].value = tuple(float(x) for x in mat["color"])
            self.prog["u_shade"].value = tuple(float(x) for x in mat["shade"])
            self.prog["u_tint"].value = tuple(float(x) for x in self.tints.get(mat["part"], (1.0, 1.0, 1.0)))[:3]
            self.prog["u_cutoff"].value = mat["cutoff"]
            if mat["double"]:
                ctx.disable(gl.CULL_FACE)
            else:
                ctx.enable(gl.CULL_FACE)
            if mat["blend"]:
                ctx.enable(gl.BLEND)
                self.msaa.depth_mask = False
            else:
                ctx.disable(gl.BLEND)
                self.msaa.depth_mask = True
            d["vao"].render()
        self.msaa.depth_mask = True
        ctx.copy_framebuffer(self.out, self.msaa)
        return self.out.read(components=4)

    def close(self) -> None:
        try:
            self.ctx.release()
        except Exception:  # noqa: BLE001
            pass


def material_part(name: str) -> str:
    """Which part of the character a material paints (for colour changes): hair, skin, eyes,
    clothes or other. From the material's name (VRoid names them this way)."""
    n = name.lower()
    for part, keys in (("hair", ("hair",)), ("eyes", ("iris", "eye", "pupil", "highlight")), ("skin", ("skin", "face", "body", "mouth")),
                       ("clothes", ("cloth", "tops", "bottoms", "onepiece", "shoes", "accessory", "outfit", "shirt", "pants", "skirt", "jacket"))):
        if any(k in n for k in keys):
            return part
    return "other"
