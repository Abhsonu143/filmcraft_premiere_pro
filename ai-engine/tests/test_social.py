"""Social posting with a stand-in for the platforms: no network, no real accounts."""

import datetime as dt
import json
import subprocess

import pytest

import social as so


class FakePlatforms:
    """Answers like YouTube, Instagram, Facebook and Threads do; records every call."""

    def __init__(self, fail: set[str] | None = None):
        self.calls: list[tuple[str, str, dict]] = []
        self.fail = fail or set()
        self.yt_parts = 0

    def __call__(self, method, url, *, params=None, data=None, json_body=None, headers=None, timeout=300):
        self.calls.append((method, url, {"params": params or {}, "json": json_body, "headers": headers or {}, "data": data}))
        for f in self.fail:
            if f in url:
                raise so.ApiError("HTTP 400: (#9004) The media could not be fetched", 400)
        if "oauth2.googleapis.com/token" in url:
            return 200, {}, {"access_token": "ya29.new", "expires_in": 3600}
        if "upload/youtube/v3/videos" in url:
            return 200, {"location": "https://upload.example/session-1"}, {}
        if url == "https://upload.example/session-1":
            self.yt_parts += 1
            return 200, {}, {"id": "VID123", "status": {"privacyStatus": (json.loads(json.dumps(self.calls[-2][2]["json"] or {})).get("status") or {}).get("privacyStatus", "public")}}
        if "thumbnails/set" in url:
            return 200, {}, {"items": []}
        if url.endswith("/media") and "ig" in url:
            return 200, {}, {"id": "CONT1", "uri": "https://rupload.facebook.com/ig-api-upload/v24.0/CONT1"}
        if "rupload.facebook.com" in url:
            return 200, {}, {"success": True}
        if url.endswith("/CONT1"):
            return 200, {}, {"status_code": "FINISHED"}
        if url.endswith("/media_publish"):
            return 200, {}, {"id": "IGMEDIA9"}
        if url.endswith("/IGMEDIA9"):
            return 200, {}, {"permalink": "https://www.instagram.com/reel/abc/"}
        if url.endswith("/page1/videos"):
            phase = (so.urllib.parse.parse_qs(data.decode()) if data and data[:2] != b"--" else {}).get("upload_phase", [""])[0]
            if phase == "start":
                return 200, {}, {"upload_session_id": "S1", "start_offset": "0", "end_offset": str(self.size)}
            if data and b'name="upload_phase"\r\n\r\ntransfer' in data:
                return 200, {}, {"start_offset": str(self.size), "end_offset": str(self.size)}
            return 200, {}, {"success": True}
        if url.endswith("/page1/video_reels"):
            return 200, {}, {"video_id": "FBR1"} if b"start" in (data or b"") else {"success": True}
        if url.endswith("/me/threads"):
            return 200, {}, {"id": "TC1"}
        if url.endswith("/me/threads_publish"):
            return 200, {}, {"id": "TP1"}
        if url.endswith("/TP1"):
            return 200, {}, {"permalink": "https://www.threads.net/@me/post/TP1"}
        raise AssertionError(f"unexpected call {method} {url}")


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("EI_SOCIAL_HOME", str(tmp_path / "social"))
    monkeypatch.setenv("EI_THREADS_WAIT", "0")
    video = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=360x640:d=2", "-pix_fmt", "yuv420p", str(video)],
                   check=True)
    thumb = tmp_path / "t.jpg"
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=red:s=1280x720", "-frames:v", "1", str(thumb)], check=True)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    copy = {"titles": ["GTA 6 Will Only Run at 30 FPS"], "description": "Why GTA 6 is 30 FPS.", "youtube_tags": ["gta 6 fps"], "youtube_hashtags": ["#GTA6", "#PS5"],
            "shorts_titles": ["30 FPS?!"], "shorts_hashtags": ["#GTA6"], "hooks": ["Your PS5 Pro won't save you."],
            "instagram_captions": ["Your PS5 Pro won't save you.\n\nComment FPS #a #b #c #d #e #f #g"], "instagram_hashtags": ["#gta6", "#ps5", "#gaming", "#gta", "#vicecity"]}
    (pkg / "package.json").write_text(json.dumps({"topic": "GTA 6 at 30 FPS", "copy": {"en": copy},
                                                  "thumbnails": [{"platform": "youtube", "language": "en", "variant": 1, "path": str(thumb)}]}))
    so.secret_set("youtube", {"client_id": "c", "client_secret": "s", "refresh_token": "r", "access_token": "", "expires": 0})
    so.secret_set("meta", {"app_id": "a", "app_secret": "s", "user_token": "U", "page_token": "P", "expires": so.time.time() + 50 * 86400})
    so.secret_set("threads", {"token": "T", "expires": so.time.time() + 50 * 86400})
    so.save_accounts({"youtube": {"name": "ch"}, "meta": {"page_id": "page1", "ig_id": "ig1", "page": "My Page", "instagram": "me"}, "threads": {"name": "me"}})
    fake = FakePlatforms()
    fake.size = video.stat().st_size
    monkeypatch.setattr(so, "http", fake)
    monkeypatch.setattr(so.time, "sleep", lambda s: None)
    return {"video": video, "pkg": pkg, "fake": fake, "thumb": thumb}


def test_words_come_from_the_package(world):
    p = so.new_post(str(world["video"]), ["youtube", "reels", "threads"], package=str(world["pkg"]))
    yt, reel, th = p["items"]
    assert yt["title"] == "GTA 6 Will Only Run at 30 FPS" and "#GTA6" in yt["text"] and yt["thumbnail"] == str(world["thumb"])
    assert reel["text"].count("#") == 5 and "#f" not in reel["text"]  # Instagram allows five
    assert th["text"].startswith("Your PS5 Pro")


def test_post_everywhere_in_order(world):
    p = so.new_post(str(world["video"]), ["threads", "youtube", "reels", "facebook", "fbreels"], package=str(world["pkg"]))
    res = so.run_post(p["id"])
    assert set(res["status"].values()) == {"posted"}, res
    urls = [c[1] for c in world["fake"].calls]
    # videos first; Threads last, with the first video's link
    assert urls.index(next(u for u in urls if "threads" in u)) > urls.index(next(u for u in urls if "media_publish" in u))
    text = next(so.urllib.parse.parse_qs(c[2]["data"].decode())["text"][0] for c in world["fake"].calls if c[1].endswith("/me/threads"))
    assert "https://youtu.be/VID123" in text
    # the YouTube upload: resumable, then the thumbnail; Instagram: resumable to rupload with the file size
    assert any("thumbnails/set" in u for u in urls)
    up = next(c for c in world["fake"].calls if "rupload.facebook.com/ig-api-upload" in c[1])
    assert up[2]["headers"]["file_size"] == str(world["video"].stat().st_size) and up[2]["headers"]["Authorization"] == "OAuth U"
    saved = so.load_posts()[0]
    assert all(i["status"] == "posted" for i in saved["items"]) and saved["items"][1]["result"]["url"] == "https://youtu.be/VID123"


def test_a_failure_is_kept_and_retried_alone(world):
    world["fake"].fail = {"/ig1/media"}
    p = so.new_post(str(world["video"]), ["youtube", "reels"], package=str(world["pkg"]))
    res = so.run_post(p["id"])
    assert res["status"] == {"youtube": "posted", "reels": "failed"}
    assert "could not be fetched" in so.load_posts()[0]["items"][1]["result"]["error"]
    world["fake"].fail = set()
    n = len([c for c in world["fake"].calls if "upload/youtube" in c[1]])
    res = so.run_post(p["id"], only_failed=True)
    assert res["status"] == {"youtube": "posted", "reels": "posted"}
    assert len([c for c in world["fake"].calls if "upload/youtube" in c[1]]) == n  # YouTube was not posted twice


def test_scheduling(world, monkeypatch):
    when = (dt.datetime.now() + dt.timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")
    p = so.new_post(str(world["video"]), ["youtube", "facebook", "reels", "threads"], package=str(world["pkg"]), at=when)
    assert all(i["status"] == "scheduled" for i in p["items"]) and so.due() == []
    assert p["id"] in so.send_ahead()
    so.run_post(p["id"], only=so.SELF_SCHEDULING)
    meta = next(c for c in world["fake"].calls if "upload/youtube/v3/videos" in c[1])[2]["json"]
    assert meta["status"]["privacyStatus"] == "private" and meta["status"]["publishAt"].endswith("Z")  # YouTube publishes it at the time
    finish = next(c for c in world["fake"].calls if c[1].endswith("/page1/videos") and b"finish" in c[2]["data"])
    assert b"scheduled_publish_time" in finish[2]["data"]
    st = {i["target"]: i["status"] for i in so.load_posts()[0]["items"]}
    assert st == {"youtube": "posted", "facebook": "posted", "reels": "scheduled", "threads": "scheduled"}
    assert not any("ig1" in c[1] for c in world["fake"].calls)  # Instagram waits for the time
    # the time comes
    posts = so.load_posts()
    posts[0]["at"] = (dt.datetime.now().astimezone() - dt.timedelta(minutes=1)).isoformat(timespec="minutes")
    so.save_posts(posts)
    assert so.due() == [p["id"]]
    so.run_due()
    assert {i["status"] for i in so.load_posts()[0]["items"]} == {"posted"}
    assert so.due() == []


def test_bad_requests_are_refused(world):
    with pytest.raises(so.AutocutError):
        so.new_post(str(world["video"]), ["tiktok"])
    with pytest.raises(so.AutocutError):
        so.new_post(str(world["video"]), ["youtube"], at="2001-01-01 10:00", package=str(world["pkg"]))
    with pytest.raises(so.AutocutError):
        so.parse_when("next tuesday")
    assert so.parse_when("2026-11-20 18:00").hour == 18 and so.parse_when("") is None
