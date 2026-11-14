"""social: post a video (with its title, caption, hashtags and thumbnail) to YouTube, Instagram,
Facebook and Threads from EI, now or at a set time.

    python social.py accounts                                   what is connected
    python social.py connect youtube --client-secret client_secret.json   (opens the browser once)
    python social.py connect meta --app-id ID --app-secret S --token T     (Instagram + Facebook Page)
    python social.py connect threads --token T [--app-secret S]
    python social.py disconnect youtube|meta|threads
    python social.py post --video V [--package P --language en] --targets youtube reels threads [--at "2026-11-20 18:00"] [--dry-run]
    python social.py run-due                                    post what is scheduled and due
    python social.py list                                       the posts and how each went

Targets: youtube (a video), shorts (a YouTube Short: vertical, up to 3 minutes), reels (an Instagram
Reel, also shown in the feed), facebook (a video on your Facebook Page), fbreels (a Facebook Reel),
threads (a text post with the link to the first published video: Threads only takes media from
public web links).

The words come from a Thumbnails, Titles & Hashtags package (package.json) in the chosen language
(or --title/--text), and can be edited per target before posting (in the app). Every post and its
result per platform is kept in posts.json (Application Support/EI/social); tokens are kept in the
macOS Keychain, never in files.

What each platform allows (Nov 2026):
  - YouTube Data API: uploads from a personal Google Cloud project made after July 2020 are private
    until the project passes YouTube's API compliance audit; ~6 uploads a day on the default quota
    (10,000 units, an upload costs 1,600); a custom thumbnail needs a verified channel.
  - Instagram (Instagram API with Facebook Login for Business): a professional account linked to a
    Facebook Page; Reels upload straight from this Mac (resumable upload); no scheduling in the API,
    so EI posts at the set time (EI must be open); the cover is a frame of the video.
  - Facebook Pages: videos and Reels upload from this Mac; not personal profiles.
  - Threads: text (500 characters) with a link; images and videos only from public web links.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import http.server
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from pathlib import Path

import autocut as ac
from autocut import AutocutError, log

GRAPH = "https://graph.facebook.com"
GRAPH_VIDEO = "https://graph-video.facebook.com"
RUPLOAD = "https://rupload.facebook.com"
THREADS = "https://graph.threads.net"
GRAPH_VERSION = os.environ.get("EI_GRAPH_VERSION", "v24.0")
YT_UPLOAD = "https://www.googleapis.com/upload/youtube/v3"
YT_API = "https://www.googleapis.com/youtube/v3"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
YT_SCOPES = "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube"
KEYCHAIN_SERVICE = "EI Social"
TARGETS = {
    "youtube": "YouTube video",
    "shorts": "YouTube Short",
    "reels": "Instagram Reel",
    "facebook": "Facebook Page video",
    "fbreels": "Facebook Reel",
    "threads": "Threads post",
}
ACCOUNT_OF = {"youtube": "youtube", "shorts": "youtube", "reels": "meta", "facebook": "meta", "fbreels": "meta", "threads": "threads"}
CHUNK = 8 * 1024 * 1024
# platforms that publish at a set time themselves (sent ahead); the others are posted by EI at the time
SELF_SCHEDULING = ("youtube", "shorts", "facebook", "fbreels")


def home() -> Path:
    d = Path(os.environ.get("EI_SOCIAL_HOME") or Path.home() / "Library" / "Application Support" / "EI" / "social")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------------------------
# Secrets (macOS Keychain) and accounts
# ---------------------------------------------------------------------------------------------


def secret_get(name: str) -> dict | None:
    if os.environ.get("EI_SOCIAL_HOME"):  # (tests: a file next to the posts)
        f = home() / f".secret-{name}.json"
        return json.loads(f.read_text()) if f.is_file() else None
    r = subprocess.run(["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", name, "-w"], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        return json.loads(r.stdout.strip())
    except ValueError:
        return None


def secret_set(name: str, value: dict | None) -> None:
    if os.environ.get("EI_SOCIAL_HOME"):
        f = home() / f".secret-{name}.json"
        if value is None:
            f.unlink(missing_ok=True)
        else:
            f.write_text(json.dumps(value))
            f.chmod(0o600)
        return
    if value is None:
        subprocess.run(["security", "delete-generic-password", "-s", KEYCHAIN_SERVICE, "-a", name], capture_output=True)
        return
    r = subprocess.run(["security", "add-generic-password", "-U", "-s", KEYCHAIN_SERVICE, "-a", name, "-w", json.dumps(value)], capture_output=True, text=True)
    if r.returncode != 0:
        raise AutocutError(f"could not save the login in the Keychain: {r.stderr.strip()[:200]}")


def load_accounts() -> dict:
    f = home() / "accounts.json"
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_accounts(a: dict) -> None:
    f = home() / "accounts.json"
    tmp = f.with_suffix(".part")
    tmp.write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(f)


# ---------------------------------------------------------------------------------------------
# HTTP (one place, so tests can stand in for the platforms)
# ---------------------------------------------------------------------------------------------


class ApiError(AutocutError):
    def __init__(self, msg: str, status: int = 0, body: str = ""):
        super().__init__(msg)
        self.status = status
        self.body = body


def http(method: str, url: str, *, params: dict | None = None, data: bytes | None = None, json_body: dict | None = None,
         headers: dict | None = None, timeout: int = 300) -> tuple[int, dict, dict]:
    """(status, response headers, JSON body or {"_text": …}); errors raise ApiError with the platform's message."""
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    h = dict(headers or {})
    if json_body is not None:
        data = json.dumps(json_body).encode()
        h.setdefault("Content-Type", "application/json; charset=UTF-8")
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            status, hdrs = r.status, {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        raw = e.read() or b""
        status, hdrs = e.code, {k.lower(): v for k, v in (e.headers or {}).items()}
        if status not in (308,):  # (YouTube's "resume incomplete" is not an error)
            raise ApiError(api_message(raw, status), status, raw.decode("utf-8", "replace")[:2000]) from None
    except urllib.error.URLError as e:
        raise ApiError(f"no connection ({e.reason})") from None
    try:
        body = json.loads(raw) if raw else {}
    except ValueError:
        body = {"_text": raw.decode("utf-8", "replace")[:2000]}
    return status, hdrs, body if isinstance(body, dict) else {"_list": body}


def api_message(raw: bytes, status: int) -> str:
    try:
        j = json.loads(raw)
        e = j.get("error", j)
        if isinstance(e, dict):
            msg = e.get("error_user_msg") or e.get("message") or e.get("error_description") or json.dumps(e)[:300]
        else:
            msg = str(j.get("error_description") or e)
    except ValueError:
        msg = raw.decode("utf-8", "replace")[:300]
    return f"HTTP {status}: {msg}"


def form(d: dict) -> bytes:
    return urllib.parse.urlencode({k: v for k, v in d.items() if v is not None}).encode()


def multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    b = f"----ei{uuid.uuid4().hex}"
    out = bytearray()
    for k, v in fields.items():
        if v is None:
            continue
        out += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    for k, (name, data, ctype) in files.items():
        out += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{name}\"\r\nContent-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n"
    out += f"--{b}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={b}"


def retrying(fn, what: str, tries: int = 4):
    """Network hiccups and the platforms' "try again" answers get a few more chances."""
    for k in range(tries):
        try:
            return fn()
        except ApiError as e:
            last = k == tries - 1
            if last or not (e.status in (0, 429, 500, 502, 503, 504)):
                raise
            wait = 5 * 2**k
            log(f"{what}: {e} — trying again in {wait}s")
            time.sleep(wait)
    raise AutocutError(f"{what} failed")


# ---------------------------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------------------------


def connect_youtube(client_secret: str, *, open_browser=webbrowser.open, timeout: float = 300) -> dict:
    """The one-time Google login (in the browser, back to this Mac on a local port, PKCE)."""
    cs = json.loads(Path(client_secret).expanduser().read_text())
    c = cs.get("installed") or cs.get("web") or {}
    if not c.get("client_id"):
        raise AutocutError("that is not a Google OAuth client file (Google Cloud ▸ Credentials ▸ OAuth client ID ▸ Desktop app ▸ Download JSON)")
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(16)
    got: dict = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if q.get("state", [""])[0] == state:
                got.update({k: v[0] for k, v in q.items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write("<h2>EI is connected to YouTube. You can close this tab.</h2>".encode() if "code" in got else b"<h2>Not connected.</h2>")

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    redirect = f"http://127.0.0.1:{srv.server_port}"
    url = GOOGLE_AUTH + "?" + urllib.parse.urlencode({
        "client_id": c["client_id"], "redirect_uri": redirect, "response_type": "code", "scope": YT_SCOPES, "access_type": "offline",
        "prompt": "consent", "code_challenge": challenge, "code_challenge_method": "S256", "state": state})
    log("opening the browser: sign in with the Google account of your channel and allow EI to upload")
    threading.Thread(target=lambda: open_browser(url), daemon=True).start()
    srv.timeout = 2
    t0 = time.time()
    while "code" not in got and "error" not in got and time.time() - t0 < timeout:
        srv.handle_request()
    srv.server_close()
    if "code" not in got:
        raise AutocutError(f"the Google login did not finish ({got.get('error', 'timed out')})")
    _, _, tok = http("POST", GOOGLE_TOKEN, data=form({
        "code": got["code"], "client_id": c["client_id"], "client_secret": c.get("client_secret"), "redirect_uri": redirect,
        "grant_type": "authorization_code", "code_verifier": verifier}), headers={"Content-Type": "application/x-www-form-urlencoded"})
    if not tok.get("refresh_token"):
        raise AutocutError("Google gave no refresh token; remove EI's access at myaccount.google.com/permissions and connect again")
    secret_set("youtube", {"client_id": c["client_id"], "client_secret": c.get("client_secret"), "refresh_token": tok["refresh_token"],
                           "access_token": tok.get("access_token"), "expires": time.time() + float(tok.get("expires_in", 3600)) - 60})
    _, _, me = http("GET", f"{YT_API}/channels", params={"part": "snippet", "mine": "true"}, headers=yt_headers())
    ch = (me.get("items") or [{}])[0]
    acc = load_accounts()
    acc["youtube"] = {"name": ch.get("snippet", {}).get("title", "your channel"), "id": ch.get("id", ""), "connected": now_iso()}
    save_accounts(acc)
    log(f"YouTube connected: {acc['youtube']['name']}")
    return acc["youtube"]


def yt_headers() -> dict:
    s = secret_get("youtube")
    if not s:
        raise AutocutError("YouTube is not connected (Social Posting ▸ Accounts ▸ Connect YouTube)")
    if not s.get("access_token") or time.time() > float(s.get("expires", 0)):
        _, _, tok = http("POST", GOOGLE_TOKEN, data=form({"client_id": s["client_id"], "client_secret": s.get("client_secret"),
                                                          "refresh_token": s["refresh_token"], "grant_type": "refresh_token"}),
                         headers={"Content-Type": "application/x-www-form-urlencoded"})
        s["access_token"] = tok["access_token"]
        s["expires"] = time.time() + float(tok.get("expires_in", 3600)) - 60
        secret_set("youtube", s)
    return {"Authorization": f"Bearer {s['access_token']}"}


def upload_youtube(video: Path, t: dict) -> dict:
    """Resumable upload, then the thumbnail. t: title, text (description), tags, privacy, publish_at, thumbnail, category."""
    size = video.stat().st_size
    status = {"privacyStatus": t.get("privacy") or "public", "selfDeclaredMadeForKids": False}
    if t.get("publish_at"):  # YouTube publishes it itself at that time (it must be private until then)
        status.update(privacyStatus="private", publishAt=t["publish_at"])
    meta = {"snippet": {"title": t["title"][:100], "description": t.get("text", "")[:5000], "tags": t.get("tags", [])[:30],
                        "categoryId": str(t.get("category") or "20"), "defaultLanguage": t.get("language", "en")}, "status": status}
    _, hdrs, _ = retrying(lambda: http("POST", f"{YT_UPLOAD}/videos", params={"uploadType": "resumable", "part": "snippet,status"},
                                       json_body=meta, headers={**yt_headers(), "X-Upload-Content-Length": str(size), "X-Upload-Content-Type": "video/*"}),
                          "YouTube upload start")
    session = hdrs.get("location")
    if not session:
        raise AutocutError("YouTube did not open an upload")
    sent, body = 0, {}
    with open(video, "rb") as f:
        while sent < size:
            f.seek(sent)
            chunk = f.read(CHUNK)
            end = sent + len(chunk) - 1

            def put(chunk=chunk, end=end, start=sent):
                return http("PUT", session, data=chunk, headers={"Content-Length": str(len(chunk)), "Content-Range": f"bytes {start}-{end}/{size}"}, timeout=600)

            st, h2, body = retrying(put, "YouTube upload")
            if st == 308:
                rng = h2.get("range", "")
                sent = int(rng.split("-")[-1]) + 1 if rng else 0
            else:
                sent = size
            ac.stage_progress(sent / size, f"YouTube: {sent / 1e6:.0f} of {size / 1e6:.0f} MB")
    vid = body.get("id")
    if not vid:
        raise AutocutError(f"YouTube did not confirm the upload: {json.dumps(body)[:300]}")
    out = {"id": vid, "url": f"https://youtu.be/{vid}", "privacy": (body.get("status") or {}).get("privacyStatus", status["privacyStatus"])}
    thumb = t.get("thumbnail")
    if thumb and Path(thumb).is_file():
        try:
            retrying(lambda: http("POST", f"{YT_UPLOAD}/thumbnails/set", params={"videoId": vid}, data=Path(thumb).read_bytes(),
                                  headers={**yt_headers(), "Content-Type": "image/jpeg"}), "YouTube thumbnail")
            out["thumbnail"] = "set"
        except ApiError as e:
            out["thumbnail"] = f"not set: {e}"
            log(f"YouTube: the thumbnail was not set ({e}); a custom thumbnail needs a verified channel (youtube.com/verify)")
    if out["privacy"] == "private" and not t.get("publish_at") and (t.get("privacy") or "public") != "private":
        out["note"] = "YouTube kept it private: uploads from an unaudited API project stay private (see the YouTube API compliance audit)"
        log(f"YouTube: {out['note']}")
    return out


# ---------------------------------------------------------------------------------------------
# Meta: Instagram and Facebook Pages
# ---------------------------------------------------------------------------------------------


def g(path: str) -> str:
    return f"{GRAPH}/{GRAPH_VERSION}/{path.lstrip('/')}"


def connect_meta(app_id: str, app_secret: str, token: str) -> dict:
    """A user token from your Meta app (Graph API Explorer with pages_show_list, pages_read_engagement,
    pages_manage_posts, instagram_basic, instagram_content_publish, business_management): made
    long-lived, then the Page tokens (they do not expire) and the Instagram account of each Page."""
    _, _, ll = http("GET", g("oauth/access_token"), params={"grant_type": "fb_exchange_token", "client_id": app_id, "client_secret": app_secret,
                                                            "fb_exchange_token": token})
    user_token = ll.get("access_token") or token
    _, _, pages = http("GET", g("me/accounts"), params={"fields": "id,name,access_token,instagram_business_account{id,username}", "access_token": user_token})
    found = []
    for p in pages.get("data", []):
        ig = p.get("instagram_business_account") or {}
        found.append({"page_id": p["id"], "page": p.get("name", ""), "page_token": p.get("access_token", ""), "ig_id": ig.get("id", ""),
                      "ig": ig.get("username", "")})
    if not found:
        raise AutocutError("no Facebook Page found for this login (Instagram posting needs a professional account linked to a Page)")
    use = next((f for f in found if f["ig_id"]), found[0])
    secret_set("meta", {"app_id": app_id, "app_secret": app_secret, "user_token": user_token, "page_token": use["page_token"],
                        "expires": time.time() + float(ll.get("expires_in", 60 * 86400))})
    acc = load_accounts()
    acc["meta"] = {"page": use["page"], "page_id": use["page_id"], "instagram": use["ig"], "ig_id": use["ig_id"], "connected": now_iso(),
                   "pages": [{k: f[k] for k in ("page", "page_id", "ig", "ig_id")} for f in found]}
    save_accounts(acc)
    log(f"Meta connected: Facebook Page “{use['page']}”" + (f", Instagram @{use['ig']}" if use["ig"] else " (no Instagram account linked to it)"))
    return acc["meta"]


def meta_secret() -> tuple[dict, dict]:
    s, a = secret_get("meta"), load_accounts().get("meta")
    if not s or not a:
        raise AutocutError("Instagram and Facebook are not connected (Social Posting ▸ Accounts ▸ Connect Instagram & Facebook)")
    left = float(s.get("expires", 0)) - time.time()
    if left < 7 * 86400:  # the 60-day login renews itself while it is still valid
        if left <= 0:
            raise AutocutError("the Instagram/Facebook login has expired: connect again (Social Posting ▸ Accounts)")
        try:
            _, _, ll = http("GET", g("oauth/access_token"), params={"grant_type": "fb_exchange_token", "client_id": s["app_id"],
                                                                    "client_secret": s["app_secret"], "fb_exchange_token": s["user_token"]})
            if ll.get("access_token"):
                s.update(user_token=ll["access_token"], expires=time.time() + float(ll.get("expires_in", 60 * 86400)))
                secret_set("meta", s)
        except ApiError as e:
            log(f"Meta: the login could not be renewed yet ({e})")
    return s, a


def wait_container(cid: str, token: str, what: str, base: str = GRAPH, field: str = "status_code", limit: float = 1800) -> None:
    t0 = time.time()
    while time.time() - t0 < limit:
        _, _, st = retrying(lambda: http("GET", f"{base}/{GRAPH_VERSION}/{cid}", params={"fields": f"{field},status", "access_token": token}), what)
        code = str(st.get(field, "")).upper()
        if code in ("FINISHED", "PUBLISHED", "READY"):
            return
        if code in ("ERROR", "EXPIRED"):
            raise AutocutError(f"{what}: Meta could not process the video ({st.get('status') or code})")
        ac.stage_progress(min(0.95, 0.5 + (time.time() - t0) / limit), f"{what}: Meta is processing the video")
        time.sleep(5)
    raise AutocutError(f"{what}: Meta took too long to process the video")


def post_reel(video: Path, t: dict) -> dict:
    s, a = meta_secret()
    if not a.get("ig_id"):
        raise AutocutError("no Instagram professional account is linked to the connected Facebook Page")
    tok = s["user_token"]
    size = video.stat().st_size
    if size > 1_000_000_000:
        raise AutocutError("Instagram Reels are limited to 1 GB here; export a smaller file")
    params = {"media_type": "REELS", "upload_type": "resumable", "caption": t.get("text", "")[:2200], "share_to_feed": "true", "access_token": tok}
    if t.get("cover_ms") is not None:
        params["thumb_offset"] = str(int(t["cover_ms"]))
    _, _, c = retrying(lambda: http("POST", g(f"{a['ig_id']}/media"), data=form(params), headers={"Content-Type": "application/x-www-form-urlencoded"}),
                       "Instagram upload start")
    cid = c.get("id")
    if not cid:
        raise AutocutError(f"Instagram did not open an upload: {json.dumps(c)[:300]}")
    up = c.get("uri") or f"{RUPLOAD}/ig-api-upload/{GRAPH_VERSION}/{cid}"
    retrying(lambda: http("POST", up, data=video.read_bytes(),
                          headers={"Authorization": f"OAuth {tok}", "offset": "0", "file_size": str(size)}, timeout=1800), "Instagram upload")
    ac.stage_progress(0.5, "Instagram: uploaded, waiting for processing")
    wait_container(cid, tok, "Instagram")
    _, _, pub = retrying(lambda: http("POST", g(f"{a['ig_id']}/media_publish"), data=form({"creation_id": cid, "access_token": tok}),
                                      headers={"Content-Type": "application/x-www-form-urlencoded"}), "Instagram publish")
    mid = pub.get("id", "")
    link = ""
    try:
        _, _, m = http("GET", g(mid), params={"fields": "permalink", "access_token": tok})
        link = m.get("permalink", "")
    except ApiError:
        pass
    return {"id": mid, "url": link}


def post_facebook(video: Path, t: dict) -> dict:
    """A video on the Page (chunked upload); a set time is scheduled by Facebook itself."""
    s, a = meta_secret()
    tok, page = s["page_token"], a["page_id"]
    url = f"{GRAPH_VIDEO}/{GRAPH_VERSION}/{page}/videos"
    size = video.stat().st_size
    _, _, st = retrying(lambda: http("POST", url, data=form({"upload_phase": "start", "file_size": size, "access_token": tok}),
                                     headers={"Content-Type": "application/x-www-form-urlencoded"}), "Facebook upload start")
    sess, start, end = st.get("upload_session_id"), int(st.get("start_offset", 0)), int(st.get("end_offset", 0))
    with open(video, "rb") as f:
        while start < end:
            f.seek(start)
            chunk = f.read(end - start)
            body, ctype = multipart({"upload_phase": "transfer", "upload_session_id": sess, "start_offset": start, "access_token": tok},
                                    {"video_file_chunk": ("chunk", chunk, "application/octet-stream")})
            _, _, r = retrying(lambda body=body, ctype=ctype: http("POST", url, data=body, headers={"Content-Type": ctype}, timeout=900), "Facebook upload")
            start, end = int(r.get("start_offset", end)), int(r.get("end_offset", end))
            ac.stage_progress(min(1.0, start / max(1, size)), f"Facebook: {start / 1e6:.0f} of {size / 1e6:.0f} MB")
    fields = {"upload_phase": "finish", "upload_session_id": sess, "access_token": tok, "title": t.get("title", "")[:255],
              "description": t.get("text", "")[:5000]}
    if t.get("publish_ts"):
        fields.update(published="false", scheduled_publish_time=str(int(t["publish_ts"])))
    files = {}
    if t.get("thumbnail") and Path(t["thumbnail"]).is_file():
        files["thumb"] = ("thumb.jpg", Path(t["thumbnail"]).read_bytes(), "image/jpeg")
    body, ctype = multipart(fields, files)
    retrying(lambda: http("POST", url, data=body, headers={"Content-Type": ctype}, timeout=600), "Facebook publish")
    return {"id": sess or "", "url": f"https://www.facebook.com/{page}/videos", "scheduled": bool(t.get("publish_ts"))}


def post_fb_reel(video: Path, t: dict) -> dict:
    s, a = meta_secret()
    tok, page = s["page_token"], a["page_id"]
    url = g(f"{page}/video_reels")
    _, _, st = retrying(lambda: http("POST", url, data=form({"upload_phase": "start", "access_token": tok}),
                                     headers={"Content-Type": "application/x-www-form-urlencoded"}), "Facebook Reel start")
    vid = st.get("video_id")
    if not vid:
        raise AutocutError(f"Facebook did not open a Reel upload: {json.dumps(st)[:300]}")
    size = video.stat().st_size
    retrying(lambda: http("POST", f"{RUPLOAD}/video-upload/{GRAPH_VERSION}/{vid}", data=video.read_bytes(),
                          headers={"Authorization": f"OAuth {tok}", "offset": "0", "file_size": str(size)}, timeout=1800), "Facebook Reel upload")
    fields = {"upload_phase": "finish", "video_id": vid, "video_state": "SCHEDULED" if t.get("publish_ts") else "PUBLISHED",
              "description": t.get("text", "")[:2200], "access_token": tok}
    if t.get("publish_ts"):
        fields["scheduled_publish_time"] = str(int(t["publish_ts"]))
    retrying(lambda: http("POST", url, data=form(fields), headers={"Content-Type": "application/x-www-form-urlencoded"}), "Facebook Reel publish")
    return {"id": vid, "url": f"https://www.facebook.com/reel/{vid}", "scheduled": bool(t.get("publish_ts"))}


# ---------------------------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------------------------


def connect_threads(token: str, app_secret: str = "") -> dict:
    tok = token
    expires = time.time() + 3600
    if app_secret:  # a short-lived token becomes a 60-day one
        try:
            _, _, ll = http("GET", f"{THREADS}/access_token", params={"grant_type": "th_exchange_token", "client_secret": app_secret, "access_token": token})
            tok, expires = ll.get("access_token", token), time.time() + float(ll.get("expires_in", 60 * 86400))
        except ApiError as e:
            log(f"Threads: the token was kept as it is ({e})")
    _, _, me = http("GET", f"{THREADS}/{GRAPH_VERSION}/me", params={"fields": "id,username", "access_token": tok})
    secret_set("threads", {"token": tok, "expires": expires, "app_secret": app_secret})
    acc = load_accounts()
    acc["threads"] = {"name": me.get("username", ""), "id": me.get("id", ""), "connected": now_iso()}
    save_accounts(acc)
    log(f"Threads connected: @{acc['threads']['name']}")
    return acc["threads"]


def threads_token() -> str:
    s = secret_get("threads")
    if not s:
        raise AutocutError("Threads is not connected (Social Posting ▸ Accounts ▸ Connect Threads)")
    if time.time() > float(s.get("expires", 0)) - 7 * 86400:  # refresh in the last week (a long-lived token keeps going)
        try:
            _, _, r = http("GET", f"{THREADS}/refresh_access_token", params={"grant_type": "th_refresh_token", "access_token": s["token"]})
            if r.get("access_token"):
                s.update(token=r["access_token"], expires=time.time() + float(r.get("expires_in", 60 * 86400)))
                secret_set("threads", s)
        except ApiError:
            pass
    return s["token"]


def post_threads(t: dict, link: str = "") -> dict:
    tok = threads_token()
    text = t.get("text", "").strip()
    if link and link not in text:
        text = f"{text}\n\n{link}".strip()
    if len(text) > 500:
        text = text[: 500 - len(link) - 3].rstrip() + "…\n\n" + link if link else text[:499] + "…"
    base = f"{THREADS}/{GRAPH_VERSION}/me"
    _, _, c = retrying(lambda: http("POST", f"{base}/threads", data=form({"media_type": "TEXT", "text": text, "access_token": tok}),
                                    headers={"Content-Type": "application/x-www-form-urlencoded"}), "Threads")
    cid = c.get("id")
    if not cid:
        raise AutocutError(f"Threads did not accept the post: {json.dumps(c)[:300]}")
    time.sleep(float(os.environ.get("EI_THREADS_WAIT", "3")))  # (Meta asks for a short wait before publishing)
    _, _, p = retrying(lambda: http("POST", f"{base}/threads_publish", data=form({"creation_id": cid, "access_token": tok}),
                                    headers={"Content-Type": "application/x-www-form-urlencoded"}), "Threads publish")
    pid = p.get("id", "")
    url = ""
    try:
        _, _, m = http("GET", f"{THREADS}/{GRAPH_VERSION}/{pid}", params={"fields": "permalink", "access_token": tok})
        url = m.get("permalink", "")
    except ApiError:
        pass
    return {"id": pid, "url": url}


# ---------------------------------------------------------------------------------------------
# Posts: what to post where, when; and how it went
# ---------------------------------------------------------------------------------------------


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def parse_when(s: str | None) -> dt.datetime | None:
    """“2026-11-20 18:00” (this Mac's time zone), an ISO time, or empty for now."""
    if not s or not str(s).strip() or str(s).strip().lower() == "now":
        return None
    t = str(s).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M"):
        try:
            return dt.datetime.strptime(t, fmt).astimezone()
        except ValueError:
            pass
    try:
        d = dt.datetime.fromisoformat(str(s).strip())
        return d if d.tzinfo else d.astimezone()
    except ValueError as e:
        raise AutocutError(f"the time “{s}” is not like 2026-11-20 18:00") from e


def load_posts() -> list[dict]:
    try:
        return json.loads((home() / "posts.json").read_text(encoding="utf-8")).get("posts", [])
    except (OSError, ValueError):
        return []


def save_posts(posts: list[dict]) -> None:
    f = home() / "posts.json"
    tmp = f.with_suffix(f".{os.getpid()}.part")
    tmp.write_text(json.dumps({"version": 1, "posts": posts}, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(f)


def words_for(target: str, pkg: dict, lang: str) -> dict:
    """Title, text, tags and thumbnail for one target from a Thumbnails, Titles & Hashtags package."""
    c = (pkg.get("copy") or {}).get(lang) or next(iter((pkg.get("copy") or {}).values()), {})
    thumbs = [t for t in pkg.get("thumbnails", []) if t.get("language") == lang]

    def thumb(platform):
        return next((t["path"] for t in thumbs if t.get("platform") == platform and t.get("variant") == 1), "") or next(
            (t["path"] for t in thumbs if t.get("platform") == platform), "")

    yt_tags = " ".join(c.get("youtube_hashtags", []))
    if target == "youtube":
        desc = c.get("description", "")
        if yt_tags and yt_tags.split()[0] not in desc:
            desc = f"{desc}\n\n{yt_tags}".strip()
        return {"title": (c.get("titles") or [pkg.get("topic", "")])[0], "text": desc, "tags": c.get("youtube_tags", []), "thumbnail": thumb("youtube")}
    if target == "shorts":
        title = (c.get("shorts_titles") or c.get("titles") or [pkg.get("topic", "")])[0]
        tags = " ".join(c.get("shorts_hashtags", []))
        return {"title": title, "text": f"{(c.get('hooks') or [''])[0]}\n\n{tags}".strip(), "tags": c.get("youtube_tags", []), "thumbnail": thumb("shorts")}
    if target in ("reels", "fbreels"):
        cap = (c.get("instagram_captions") or [(c.get("hooks") or [""])[0]])[0]
        tags = c.get("instagram_hashtags", [])[:5]
        body = re.sub(r"(?:\s*#\w+)+\s*$", "", cap, flags=re.UNICODE).strip()  # (one set of hashtags, at most 5)
        return {"title": "", "text": f"{body}\n\n{' '.join(tags)}".strip(), "tags": [], "thumbnail": thumb("reels")}
    if target == "facebook":
        return {"title": (c.get("titles") or [pkg.get("topic", "")])[0], "text": (c.get("instagram_captions") or [c.get("description", "")])[0],
                "tags": [], "thumbnail": thumb("youtube")}
    if target == "threads":
        hook = (c.get("hooks") or [(c.get("titles") or [""])[0]])[0]
        return {"title": "", "text": f"{hook}\n\n{' '.join(c.get('instagram_hashtags', [])[:1])}".strip(), "tags": [], "thumbnail": ""}
    return {"title": "", "text": "", "tags": [], "thumbnail": ""}


def new_post(video: str, targets: list[str], *, package: str | None = None, language: str = "en", at: str | None = None,
             title: str = "", text: str = "", privacy: str = "public", overrides: dict | None = None) -> dict:
    v = Path(video).expanduser().resolve()
    if not v.is_file():
        raise AutocutError(f"video not found: {video}")
    bad = [t for t in targets if t not in TARGETS]
    if bad or not targets:
        raise AutocutError(f"choose where to post ({', '.join(TARGETS)})")
    pkg = {}
    if package:
        pp = Path(package).expanduser()
        pp = pp / "package.json" if pp.is_dir() else pp
        try:
            pkg = json.loads(pp.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise AutocutError(f"could not read the package {pp}: {e}") from e
    when = parse_when(at)
    if when and when < dt.datetime.now().astimezone() - dt.timedelta(minutes=1):
        raise AutocutError("that time has passed")
    info = ac.probe(v)
    items = []
    for t in targets:
        w = words_for(t, pkg, language) if pkg else {"title": title or v.stem, "text": text, "tags": [], "thumbnail": ""}
        if title and t in ("youtube", "shorts", "facebook"):
            w["title"] = title
        if text:
            w["text"] = text
        w.update((overrides or {}).get(t, {}))
        if t in ("shorts", "reels", "fbreels") and info.has_video and info.width > info.height:
            log(f"{TARGETS[t]}: the video is horizontal; it is meant to be vertical (9:16)")
        if t == "shorts" and info.duration > 180:
            raise AutocutError("a YouTube Short can be at most 3 minutes")
        if t in ("youtube", "shorts", "facebook") and not w.get("title"):
            raise AutocutError(f"{TARGETS[t]} needs a title")
        items.append({"target": t, "status": "scheduled" if when else "waiting", "privacy": privacy, "language": language, **w, "result": {}})
    post = {"id": f"p{int(time.time())}{secrets.token_hex(2)}", "video": str(v), "package": str(package or ""), "created": now_iso(),
            "at": when.isoformat(timespec="minutes") if when else "", "at_ts": int(when.timestamp()) if when else 0, "items": items}
    posts = load_posts()
    posts.append(post)
    save_posts(posts)
    return post


def run_post(post_id: str, *, dry_run: bool = False, only_failed: bool = False, only: tuple[str, ...] | None = None) -> dict:
    """Post one post to each of its targets (videos first, then Threads with the first video's link)."""
    posts = load_posts()
    post = next((p for p in posts if p["id"] == post_id), None)
    if not post:
        raise AutocutError(f"no post {post_id}")
    video = Path(post["video"])
    if not video.is_file():
        raise AutocutError(f"the video is gone: {video}")
    when = parse_when(post.get("at"))
    order = sorted(post["items"], key=lambda i: i["target"] == "threads")
    todo = [i for i in order if i["status"] in (("failed",) if only_failed else ("waiting", "scheduled", "failed")) and (only is None or i["target"] in only)]
    ac.progress_plan({i["target"]: 1.0 for i in todo} or {"none": 1.0})
    link = next((i["result"].get("url") for i in post["items"] if i["status"] == "posted" and i["result"].get("url")), "")
    for it in todo:
        with ac.stage(it["target"]):
            t = dict(it)
            if when and it["target"] in ("youtube", "shorts") and when > dt.datetime.now().astimezone():
                t["publish_at"] = when.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if when and it["target"] in ("facebook", "fbreels") and when > dt.datetime.now().astimezone() + dt.timedelta(minutes=10):
                t["publish_ts"] = when.timestamp()
            if dry_run:
                log(f"[dry run] {TARGETS[it['target']]}: “{(t.get('title') or t.get('text', ''))[:70]}” ({video.name})")
                continue
            it["status"] = "posting"
            save_post(post)
            try:
                if it["target"] in ("youtube", "shorts"):
                    res = upload_youtube(video, t)
                elif it["target"] == "reels":
                    res = post_reel(video, t)
                elif it["target"] == "facebook":
                    res = post_facebook(video, t)
                elif it["target"] == "fbreels":
                    res = post_fb_reel(video, t)
                else:
                    res = post_threads(t, link)
                it.update(status="posted", result=res, posted=now_iso())
                link = link or res.get("url", "")
                log(f"posted: {TARGETS[it['target']]} {res.get('url', '')}")
            except Exception as e:  # (any failure is recorded; a post is never left "posting")
                it.update(status="failed", result={"error": str(e)[:500] or e.__class__.__name__})
                log(f"failed: {TARGETS[it['target']]}: {e}")
            save_post(post)
    done = {i["target"]: i["status"] for i in post["items"]}
    return {"post": post["id"], "status": done, "links": [i["result"].get("url") for i in post["items"] if i["result"].get("url")],
            "outputs": [], "out_dir": str(home())}


def save_post(post: dict) -> None:
    posts = load_posts()
    for k, p in enumerate(posts):
        if p["id"] == post["id"]:
            posts[k] = post
            break
    else:
        posts.append(post)
    save_posts(posts)


def due() -> list[str]:
    """Posts whose time has come (YouTube and Facebook schedule themselves; the rest are posted then)."""
    now = dt.datetime.now().astimezone()
    out = []
    for p in load_posts():
        w = parse_when(p.get("at"))
        if any(i["status"] == "scheduled" for i in p["items"]) and (w is None or w <= now + dt.timedelta(minutes=1)):
            out.append(p["id"])
    return out


def send_ahead() -> list[str]:
    """Scheduled posts whose YouTube/Facebook parts can go now, scheduled by the platform itself."""
    now = dt.datetime.now().astimezone()
    out = []
    for p in load_posts():
        w = parse_when(p.get("at"))
        if w and w > now + dt.timedelta(minutes=15) and any(i["status"] == "scheduled" and i["target"] in ("youtube", "shorts", "facebook", "fbreels")
                                                            and i.get("privacy", "public") == "public" for i in p["items"]):
            out.append(p["id"])
    return out


def run_due(dry_run: bool = False) -> dict:
    ids = due()
    res = [run_post(i, dry_run=dry_run) for i in ids]
    if not ids:
        log("nothing is due")
    return {"posted": res, "outputs": [], "out_dir": str(home())}


def accounts_state() -> dict:
    a = load_accounts()
    return {"youtube": a.get("youtube"), "meta": a.get("meta"), "threads": a.get("threads"), "posts": load_posts()[-50:], "due": due()}


def handle(argv: list[str]) -> dict:
    """One command (the same as the command line); raises AutocutError."""
    ap = argparse.ArgumentParser(prog="social", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("accounts")
    c = sub.add_parser("connect")
    c.add_argument("platform", choices=["youtube", "meta", "threads"])
    c.add_argument("--client-secret")
    c.add_argument("--app-id")
    c.add_argument("--app-secret", default="")
    c.add_argument("--token")
    d = sub.add_parser("disconnect")
    d.add_argument("platform", choices=["youtube", "meta", "threads"])
    p = sub.add_parser("post")
    p.add_argument("--video", required=True)
    p.add_argument("--targets", nargs="+", required=True, choices=list(TARGETS))
    p.add_argument("--package")
    p.add_argument("--language", default="en", choices=["en", "hi"])
    p.add_argument("--at", help='when, e.g. "2026-11-20 18:00" (this Mac\'s time); empty: now')
    p.add_argument("--title", default="")
    p.add_argument("--text", default="")
    p.add_argument("--privacy", default="public", choices=["public", "unlisted", "private"])
    p.add_argument("--edits", help="a JSON file of per-target edits {target: {title, text, thumbnail}}")
    p.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("id")
    r.add_argument("--failed", action="store_true", help="only the targets that failed")
    r.add_argument("--dry-run", action="store_true")
    rd = sub.add_parser("run-due")
    rd.add_argument("--dry-run", action="store_true")
    x = sub.add_parser("cancel")
    x.add_argument("id")
    sub.add_parser("list")
    w = sub.add_parser("words", help="the title, text and thumbnail per target from a package (nothing is posted)")
    w.add_argument("--package", required=True)
    w.add_argument("--language", default="en", choices=["en", "hi"])
    w.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    a = ap.parse_args(argv)
    if a.cmd == "accounts" or a.cmd == "list":
        res = accounts_state()
    elif a.cmd == "connect":
        if a.platform == "youtube":
            if not a.client_secret:
                raise AutocutError("give --client-secret (the OAuth client JSON from Google Cloud)")
            res = connect_youtube(a.client_secret)
        elif a.platform == "meta":
            if not (a.app_id and a.app_secret and a.token):
                raise AutocutError("give --app-id, --app-secret and --token (from your Meta app)")
            res = connect_meta(a.app_id, a.app_secret, a.token)
        else:
            if not a.token:
                raise AutocutError("give --token (from your Meta app's Threads use case)")
            res = connect_threads(a.token, a.app_secret)
    elif a.cmd == "disconnect":
        secret_set(a.platform, None)
        acc = load_accounts()
        acc.pop(a.platform, None)
        save_accounts(acc)
        res = {"disconnected": a.platform}
        log(f"{a.platform} disconnected (EI's saved login removed)")
    elif a.cmd == "post":
        edits = json.loads(Path(a.edits).read_text(encoding="utf-8")) if a.edits else None
        post = new_post(a.video, a.targets, package=a.package, language=a.language, at=a.at, title=a.title, text=a.text,
                        privacy=a.privacy, overrides=edits)
        if post["at"]:
            log(f"scheduled for {post['at']}: " + ", ".join(TARGETS[i["target"]] for i in post["items"]) + " (EI posts it then; keep EI open)")
            res = {"post": post["id"], "scheduled": post["at"], "outputs": [], "out_dir": str(home())}
            if not a.dry_run and post["id"] in send_ahead():
                # YouTube and Facebook take the time themselves: send those now; the rest wait for the time
                res = run_post(post["id"], only=SELF_SCHEDULING)
        else:
            res = run_post(post["id"], dry_run=a.dry_run)
    elif a.cmd == "run":
        res = run_post(a.id, dry_run=a.dry_run, only_failed=a.failed)
    elif a.cmd == "run-due":
        res = run_due(a.dry_run)
    elif a.cmd == "cancel":
        posts = load_posts()
        n = 0
        for p in posts:
            if p["id"] == a.id:
                for i in p["items"]:
                    if i["status"] in ("scheduled", "waiting", "failed"):
                        i["status"] = "cancelled"
                        n += 1
        save_posts(posts)
        res = {"cancelled": n}
    elif a.cmd == "words":
        pkg = json.loads(Path(a.package).expanduser().joinpath("package.json" if Path(a.package).expanduser().is_dir() else "").read_text(encoding="utf-8"))
        res = {"words": {t: words_for(t, pkg, a.language) for t in a.targets}}
    else:
        res = {}
    return res


def main(argv: list[str] | None = None) -> int:
    try:
        print(json.dumps(handle(sys.argv[1:] if argv is None else argv), ensure_ascii=False))
    except AutocutError as e:
        log(f"error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
