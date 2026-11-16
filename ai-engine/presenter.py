"""Your own camera as the presenter of a generated video.

A finished video (Video from Topic, a voiceover edit) already has its voice and its lines. Here the owner
records themself saying those lines, one at a time, in a page in the browser; only the picture of each
take is used later, synced to the generated voice and laid over the video.

This file is step 1: the recording session.
- `Session`: what was recorded for which line, kept next to the video in `<name>.presenter/session.json`
  (takes are numbered and never deleted; the session says which take of a line is the chosen one).
- `serve`: a small page on this Mac only (127.0.0.1) that shows a line, plays how the voice says it,
  records a take with the camera and saves it.
"""

from __future__ import annotations

import io
import json
import os
import re
import secrets
import threading
import time
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from autocut import AutocutError, log

MAX_TAKE_BYTES = 400 * 1024 * 1024  # one line is seconds long: anything bigger is not a take
TAKE_TYPES = {"webm": "video/webm", "mp4": "video/mp4", "mov": "video/quicktime"}
IDLE = "idle"  # the take of just looking at the camera, shown where nothing is said


def _atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


MAX_WORDS = 10  # what is shown to be said in one go


def split_words(text: str, limit: int = MAX_WORDS) -> list[str]:
    """A sentence in parts of at most `limit` words, cut at its punctuation where it can be."""
    words = text.split()
    if len(words) <= limit:
        return [text.strip()] if words else []
    phrases, cur = [], []
    for w in words:  # phrases end at , ; : . ! ? and dashes
        cur.append(w)
        if re.search(r"[,;:.!?…—–]$", w.rstrip("\"')”’")):
            phrases.append(cur)
            cur = []
    if cur:
        phrases.append(cur)
    parts: list[list[str]] = []
    for ph in phrases:
        if len(ph) > limit:  # one long phrase: equal pieces
            n = -(-len(ph) // limit)
            size = -(-len(ph) // n)
            parts += [ph[i : i + size] for i in range(0, len(ph), size)]
        elif parts and len(parts[-1]) + len(ph) <= limit and (len(parts[-1]) <= 3 or not re.search(r"[.!?…]$", parts[-1][-1].rstrip("\"')”’"))):
            parts[-1] = parts[-1] + ph  # phrases of one sentence stay together, and a very short sentence joins the next
        else:
            parts.append(ph)
    return [" ".join(p) for p in parts]


def part_times(line: dict, parts: list[str], words: list[dict]) -> list[tuple[float, float]]:
    """When each part of a line is said, in the voice file's time: by its share of the letters, moved
    to the nearest gap between two heard words (`words`, from Whisper) when there is one close by."""
    a, dur = line["src"], line["end"] - line["start"]
    total = sum(len(p) for p in parts) or 1
    gaps = [((float(u["end"]) + float(v["start"])) / 2, float(v["start"]) - float(u["end"])) for u, v in zip(words, words[1:])
            if a + 0.15 < float(u["end"]) and float(v["start"]) < a + dur - 0.15]
    cuts, acc = [a], 0
    for p in parts[:-1]:
        acc += len(p)
        t = a + dur * acc / total
        near = [(abs(g - t) - 0.5 * min(w, 0.4), g) for g, w in gaps if abs(g - t) < 0.7 and g > cuts[-1] + 0.3]
        cuts.append(min(near)[1] if near else max(t, cuts[-1] + 0.05))
    cuts.append(a + dur)
    return list(zip(cuts, cuts[1:]))


def heard_words(voice: Path) -> list[dict]:
    """Word timings of the voice (Whisper, kept in the cache); empty when it cannot listen."""
    try:
        import autocut as ac

        return ac.transcribe(str(voice), language=None).get("words") or []
    except Exception as e:  # noqa: BLE001 (parts are then timed by their share of the letters)
        log(f"presenter: could not listen for word timings ({e})")
        return []


def lines_of(video: Path, *, max_words: int = MAX_WORDS, listen: bool = True) -> tuple[Path, list[dict]]:
    """The voice file and what is to be said, piece by piece (at most `max_words` words each), in the
    video's time. `src` is where a piece starts in the voice file (it differs when the edit moved lines);
    `line` is the sentence it belongs to, `part`/`parts` its place in it."""
    base = video.with_suffix("")
    timeline_path = Path(f"{base}.narrate.json")
    narration_path = Path(f"{base}.narration.json")
    if not timeline_path.is_file() and not narration_path.is_file():
        raise AutocutError(f"{video.name} has no lines next to it (.narrate.json or .narration.json): choose a video made by Video from Topic or a voiceover edit")
    timeline = json.loads(timeline_path.read_text(encoding="utf-8")) if timeline_path.is_file() else {}
    narration = json.loads(narration_path.read_text(encoding="utf-8")) if narration_path.is_file() else {}
    voice = Path(str(timeline.get("voice") or f"{base}.wav"))
    if not voice.is_file():
        voice = Path(f"{base}.wav")
    if not voice.is_file():
        raise AutocutError(f"the voice of {video.name} was not found ({voice.name})")
    src_of = {re.sub(r"\W+", " ", str(s.get("text", ""))).strip().lower(): float(s.get("start", 0)) for s in narration.get("segments") or []}
    raw = [ln for ln in timeline.get("lines") or [] if str(ln.get("text", "")).strip()] or [s for s in narration.get("segments") or [] if str(s.get("text", "")).strip()]
    lines = []
    for ln in raw:
        a, b = float(ln.get("start", 0)), float(ln.get("end", 0))
        if b <= a:
            continue
        text = str(ln["text"]).strip()
        lines.append({"line": len(lines), "start": a, "end": b, "src": src_of.get(re.sub(r"\W+", " ", text).strip().lower(), a), "text": text,
                      "emotion": ln.get("emotion")})
    if not lines:
        raise AutocutError(f"{video.name} has no spoken lines")
    split = [split_words(ln["text"], max_words) for ln in lines]
    words = heard_words(voice) if listen and any(len(p) > 1 for p in split) else []
    out = []
    for ln, parts in zip(lines, split):
        for k, (text, (a, b)) in enumerate(zip(parts, part_times(ln, parts, words))):
            out.append({"index": len(out), "line": ln["line"], "part": k + 1, "parts": len(parts), "text": text, "emotion": ln["emotion"],
                        "src": round(a, 3), "start": round(ln["start"] + a - ln["src"], 3), "end": round(ln["start"] + b - ln["src"], 3)})
    return voice, out


def face_boxes(images: list[Path]) -> list[tuple[float, float, float, float] | None]:
    """The biggest face in each picture as (left, top, right, bottom) in fractions of the picture, or None.
    (Apple's Vision, on this Mac; None for all when it is not there.)"""
    try:
        import objc
        import Vision
        from Foundation import NSURL
    except ImportError:
        return [None] * len(images)
    out = []
    for p in images:
        box = None
        with objc.autorelease_pool():
            h = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(str(p)), None)
            rq = Vision.VNDetectFaceRectanglesRequest.alloc().init()
            ok = h.performRequests_error_([rq], None)
            if ok[0] if isinstance(ok, tuple) else ok:
                best = 0.0
                for r in rq.results() or []:
                    b = r.boundingBox()
                    if float(b.size.width * b.size.height) > best:
                        best = float(b.size.width * b.size.height)
                        box = (float(b.origin.x), float(1 - b.origin.y - b.size.height), float(b.origin.x + b.size.width), float(1 - b.origin.y))
        out.append(box)
    return out


def framing_advice(boxes: list[tuple[float, float, float, float] | None]) -> str | None:
    """What to change about how the camera sees you, from the face found in a few frames of a take
    (None: it is fine). The presenter is shown from the shoulders up, and the lips must be seen."""
    found = [b for b in boxes if b]
    if not boxes:
        return None
    if len(found) * 2 < len(boxes):
        return "Your whole face is not in the picture (the mouth or chin may be cut off). Tilt the screen or move back until your face and shoulders are inside the outline."
    top = sorted(b[1] for b in found)[len(found) // 2]
    bottom = sorted(b[3] for b in found)[len(found) // 2]
    if bottom > 0.82:
        return "Your chin is at the bottom edge. Tilt the screen down or sit higher, so your shoulders are in the picture too."
    if top < 0.04:
        return "The top of your head is cut off. Tilt the screen up or move back a little."
    if bottom - top < 0.16:
        return "You are far from the camera. Come closer, so your face fills the outline."
    return None


def check_take(path: Path) -> str | None:
    """Look at a few frames of a take and say what is wrong with the framing, if anything."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="ei-take-") as tmp:
        r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(path), "-vf", "fps=1,scale=640:-2", "-frames:v", "4", str(Path(tmp) / "f%d.jpg")],
                           capture_output=True)
        frames = sorted(Path(tmp).glob("f*.jpg"))
        if r.returncode != 0 or not frames:
            return None
        return framing_advice(face_boxes(frames))


class Session:
    """The takes recorded for one video. Saved after every change."""

    def __init__(self, video: str | Path, *, max_words: int = MAX_WORDS, listen: bool = True):
        self.video = Path(video).expanduser().resolve()
        if not self.video.is_file():
            raise AutocutError(f"no such video: {video}")
        self.voice, self.lines = lines_of(self.video, max_words=max_words, listen=listen)
        self.dir = self.video.with_name(self.video.stem + ".presenter")
        self.path = self.dir / "session.json"
        self.takes: dict[str, dict] = {}  # "3" or "idle" -> {"file", "chosen", "n", "seconds", "at"}
        self._lock = threading.Lock()
        if self.path.is_file():
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                saved = {}
            by_text = {ln["text"]: ln["index"] for ln in self.lines}
            for key, t in (saved.get("takes") or {}).items():
                # a take belongs to its sentence: if the video was made again, it follows the text
                k = IDLE if key == IDLE else by_text.get(t.get("text", ""))
                if k is not None and (self.dir / "takes" / str(t.get("file", ""))).is_file():
                    self.takes[str(k)] = t

    def key(self, which: str) -> str:
        if which == IDLE:
            return IDLE
        if not which.isdigit() or int(which) >= len(self.lines):
            raise AutocutError(f"no such line: {which}")
        return str(int(which))

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        _atomic(self.path, json.dumps({"version": 1, "video": str(self.video), "voice": str(self.voice), "takes": self.takes}, indent=1, ensure_ascii=False))

    def add_take(self, which: str, data: bytes, ext: str, seconds: float | None = None) -> dict:
        """Keep a new take of a line (or of the idle look) and make it the chosen one. Older takes stay on disk."""
        key = self.key(which)
        ext = ext.lower().lstrip(".")
        if ext not in TAKE_TYPES:
            raise AutocutError(f"a take must be one of {', '.join(TAKE_TYPES)}")
        if not data or len(data) > MAX_TAKE_BYTES:
            raise AutocutError("the take is empty or too large")
        with self._lock:
            folder = self.dir / "takes"
            folder.mkdir(parents=True, exist_ok=True)
            stem = IDLE if key == IDLE else f"line-{int(key) + 1:03d}"
            n = 1 + max([int(m.group(1)) for p in folder.glob(f"{stem}-t*.*") if (m := re.search(r"-t(\d+)\.", p.name))], default=0)
            name = f"{stem}-t{n}.{ext}"
            tmp = folder / (name + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, folder / name)
            take = {"file": name, "n": n, "seconds": round(float(seconds), 2) if seconds and seconds > 0 else None, "at": int(time.time()),
                    "text": "" if key == IDLE else self.lines[int(key)]["text"]}
            try:
                take["advice"] = check_take(folder / name)
            except Exception as e:  # noqa: BLE001 (a take is never lost to the check)
                log(f"presenter: the framing of {name} could not be checked ({e})")
                take["advice"] = None
            self.takes[key] = take
            self.save()
        return take

    def forget(self, which: str) -> None:
        """Un-choose the take of a line (the file stays)."""
        with self._lock:
            if self.takes.pop(self.key(which), None) is not None:
                self.save()

    def take_path(self, which: str) -> Path | None:
        t = self.takes.get(self.key(which))
        return self.dir / "takes" / t["file"] if t else None

    def recorded_until(self) -> float:
        """The video time up to which every line has a take (what a preview can show)."""
        end = 0.0
        for ln in self.lines:
            if str(ln["index"]) not in self.takes:
                break
            end = ln["end"]
        return end

    def state(self) -> dict:
        return {"video": self.video.name, "folder": str(self.dir), "recordedUntil": self.recorded_until(), "idle": self.takes.get(IDLE),
                "lines": [{**ln, "seconds": round(ln["end"] - ln["start"], 2), "take": self.takes.get(str(ln["index"]))} for ln in self.lines]}

    def line_audio(self, which: str) -> bytes:
        """How the generated voice says one line, as a small WAV (cut from the voice file, not re-encoded)."""
        ln = self.lines[int(self.key(which))]
        try:
            with wave.open(str(self.voice), "rb") as w:
                rate = w.getframerate()
                a = max(0, int(ln["src"] * rate))
                n = max(1, int((ln["end"] - ln["start"]) * rate))
                w.setpos(min(a, w.getnframes()))
                frames = w.readframes(min(n, max(0, w.getnframes() - a)))
                buf = io.BytesIO()
                with wave.open(buf, "wb") as o:
                    o.setnchannels(w.getnchannels())
                    o.setsampwidth(w.getsampwidth())
                    o.setframerate(rate)
                    o.writeframes(frames)
                return buf.getvalue()
        except (wave.Error, EOFError, OSError) as e:
            raise AutocutError(f"the voice file could not be read: {e}") from e


def make_handler(session: Session, token: str, stop: threading.Event):
    from presenter_page import PAGE

    class Handler(BaseHTTPRequestHandler):
        server_version = "EI"

        def log_message(self, *_):  # quiet: the job log is for the owner
            pass

        def _send(self, code: int, body: bytes, kind: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

        def _allowed(self, q: dict) -> bool:
            # only the page EI opened may talk to this server (another site in the browser cannot guess the token)
            if secrets.compare_digest((q.get("k") or [""])[0], token):
                return True
            self._json({"error": "not allowed"}, 403)
            return False

        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if not self._allowed(q):
                return
            try:
                if u.path == "/":
                    return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                if u.path == "/api/session":
                    return self._json(session.state())
                m = re.fullmatch(r"/api/audio/(\d+)", u.path)
                if m:
                    return self._send(200, session.line_audio(m.group(1)), "audio/wav")
                m = re.fullmatch(r"/api/take/(\d+|idle)", u.path)
                if m:
                    p = session.take_path(m.group(1))
                    if p is None or not p.is_file():
                        return self._json({"error": "no take"}, 404)
                    return self._send(200, p.read_bytes(), TAKE_TYPES.get(p.suffix.lstrip("."), "application/octet-stream"))
                self._json({"error": "not found"}, 404)
            except AutocutError as e:
                self._json({"error": str(e)}, 400)

        def do_POST(self):  # noqa: N802
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if not self._allowed(q):
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0 or length > MAX_TAKE_BYTES:
                return self._json({"error": "too large"}, 413)
            body = self.rfile.read(length) if length else b""
            try:
                m = re.fullmatch(r"/api/take/(\d+|idle)", u.path)
                if m:
                    try:
                        seconds = float((q.get("seconds") or ["0"])[0])
                    except ValueError:
                        seconds = 0.0
                    take = session.add_take(m.group(1), body, (q.get("ext") or ["webm"])[0], seconds)
                    log(f"presenter: {'the idle look' if m.group(1) == IDLE else 'line ' + str(int(m.group(1)) + 1)} recorded ({take['file']})")
                    return self._json({"take": take, "session": session.state()})
                m = re.fullmatch(r"/api/forget/(\d+|idle)", u.path)
                if m:
                    session.forget(m.group(1))
                    return self._json({"session": session.state()})
                if u.path == "/api/framing":  # one picture from the live camera: is the face framed well?
                    import tempfile

                    if not body or len(body) > 4 * 1024 * 1024:
                        return self._json({"advice": None})
                    with tempfile.NamedTemporaryFile(suffix=".jpg") as tf:
                        tf.write(body)
                        tf.flush()
                        return self._json({"advice": framing_advice(face_boxes([Path(tf.name)]))})
                if u.path == "/api/done":
                    self._json({"ok": True})
                    stop.set()
                    return
                self._json({"error": "not found"}, 404)
            except AutocutError as e:
                self._json({"error": str(e)}, 400)

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):  # (the stock one looks up this Mac's network name first, which can take seconds)
        import socketserver

        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


def serve(video: str, *, port: int = 0, open_browser: bool = True, ready=None, stop: threading.Event | None = None,
          max_words: int = MAX_WORDS, listen: bool = True) -> dict:
    """Run the recording page until the owner presses Done (or the job is cancelled)."""
    session = Session(video, max_words=max_words, listen=listen)
    session.save()
    token = secrets.token_urlsafe(16)
    stop = stop or threading.Event()
    httpd = _Server(("127.0.0.1", port), make_handler(session, token, stop))
    url = f"http://127.0.0.1:{httpd.server_address[1]}/?k={token}"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    log(f"presenter: record your lines at {url}")
    if ready:
        ready(url)
    if open_browser:
        webbrowser.open(url)
    try:
        while not stop.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        httpd.server_close()
    st = session.state()
    done = sum(1 for ln in st["lines"] if ln["take"])
    log(f"presenter: {done} of {len(st['lines'])} lines recorded, video covered until {st['recordedUntil']:.1f}s")
    return {"outputs": [str(session.path)], "session": str(session.path), "recorded": done, "lines": len(st["lines"]), "recordedUntil": st["recordedUntil"]}


def add_args(p) -> None:
    sub = p.add_subparsers(dest="presenter_cmd", required=True)
    r = sub.add_parser("record", help="record yourself saying the lines of a generated video, one by one, in the browser")
    r.add_argument("--video", required=True, help="a video made by Video from Topic or a voiceover edit")
    r.add_argument("--port", type=int, default=0, help="the page's port on this Mac (default: any free one)")
    r.add_argument("--no-open", action="store_true", help="do not open the browser (the address is printed)")
    r.add_argument("--max-words", type=int, default=MAX_WORDS, help="most words shown to be said in one go (default 10)")
    v = sub.add_parser("render", help="the video with your takes over it, synced to the voice (a preview up to what is recorded, or the whole video)")
    v.add_argument("--video", required=True)
    v.add_argument("--until", type=float, default=None, help="stop at this many seconds (default: as far as is recorded)")
    v.add_argument("--out", default=None, help="where to write it (default: '<video> - presenter.mp4' when whole, else preview.mp4 in the presenter folder)")
    v.add_argument("--shape", default=None, choices=["circle", "rounded", "square"])
    v.add_argument("--size", type=float, default=None, help="width of your picture as a share of the video's shorter side (default 0.4)")
    v.add_argument("--position", default=None, choices=["bottom-right", "bottom-left", "bottom-center", "top-right", "top-left", "top-center"])
    v.add_argument("--margin", type=int, default=None, help="distance from the video's edge, in pixels")
    v.add_argument("--fresh", action="store_true", help="work the plan out again (drops edits made to plan.json)")
    s = sub.add_parser("status", help="what is recorded so far")
    s.add_argument("--video", required=True)


def run_args(a) -> dict:
    if a.presenter_cmd == "record":
        return serve(a.video, port=a.port, open_browser=not a.no_open, max_words=max(3, a.max_words))
    if a.presenter_cmd == "render":
        import presenter_sync

        return presenter_sync.render(a.video, until=a.until, out=a.out, fresh=a.fresh, look={"shape": a.shape, "size": a.size, "position": a.position, "margin": a.margin})
    st = Session(a.video).state()
    print(json.dumps(st, indent=1, ensure_ascii=False))
    return {"outputs": []}
