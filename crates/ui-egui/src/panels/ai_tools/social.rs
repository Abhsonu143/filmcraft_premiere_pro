//! AI ▸ Social Posting: connect YouTube, Instagram & Facebook and Threads, post a video with the
//! words of a Thumbnails, Titles & Hashtags package to one or more of them, now or at a set time,
//! and see how each post went. The engine does the posting (`ai-engine/social.py`, as AI jobs); the
//! app posts scheduled items when their time comes (EI must be open then).

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use super::{Status, child_path, engine_dir, engine_step, jobs, path_row, python, start_job, support_dir};
use crate::FilmcraftApp;

/// Where a post can go: (engine id, label, uses a title, uses a thumbnail).
pub(super) const TARGETS: [(&str, &str, bool, bool); 6] = [
    ("youtube", "YouTube video", true, true),
    ("shorts", "YouTube Short", true, true),
    ("reels", "Instagram Reel (also in the feed)", false, false),
    ("facebook", "Facebook Page video", true, true),
    ("fbreels", "Facebook Reel", false, false),
    ("threads", "Threads (text with the video's link)", false, false),
];
const LANGS: [(&str, &str); 2] = [("en", "English"), ("hi", "Hindi")];
const PRIVACY: [(&str, &str); 3] = [("public", "Public"), ("unlisted", "Unlisted"), ("private", "Private")];

pub(super) fn open_id() -> egui::Id {
    egui::Id::new("ei-ai-social-open")
}

#[derive(Clone, Default)]
struct Words {
    title: String,
    text: String,
    thumbnail: String,
}

#[derive(Default)]
struct State {
    // accounts and posts, as the engine reports them
    refreshing: bool,
    report: Option<Value>,
    error: String,
    last: Option<Instant>,
    // a new post
    video: String,
    package: String,
    language: &'static str,
    targets: [bool; 6],
    when: String,
    privacy: &'static str,
    words: BTreeMap<String, Words>,
    loading_words: bool,
    message: String,
    // connecting
    meta_app_id: String,
    meta_app_secret: String,
    meta_token: String,
    threads_token: String,
    threads_secret: String,
    // the scheduler
    last_due_check: Option<Instant>,
    /// "post", "posts" or "accounts"
    tab: &'static str,
}

fn state() -> &'static Mutex<State> {
    static S: OnceLock<Mutex<State>> = OnceLock::new();
    S.get_or_init(|| {
        Mutex::new(State { language: "en", privacy: "public", tab: "post", targets: [true, false, true, false, false, false], ..State::default() })
    })
}

/// Run an engine command and read its EI_RESULT (for quick reads: accounts, words).
fn engine_json(args: &[&str]) -> Result<Value, String> {
    let engine = engine_dir().ok_or("The AI engine folder was not found.")?;
    let out = Command::new(python(Some(&engine)))
        .arg(engine.join("ei_engine.py"))
        .args(args)
        .current_dir(&engine)
        .env("PATH", child_path())
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .map_err(|e| format!("could not run the engine: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().rev().find_map(|l| l.strip_prefix("EI_RESULT ")).ok_or("the engine gave no answer")?;
    let v: Value = serde_json::from_str(line).map_err(|_| "the engine's answer was not readable".to_string())?;
    if v.get("ok").and_then(Value::as_bool) == Some(false) {
        return Err(v.get("error").and_then(Value::as_str).unwrap_or("the engine failed").to_string());
    }
    Ok(v)
}

/// Read the accounts and posts in the background.
fn refresh() {
    {
        let mut s = state().lock().unwrap_or_else(PoisonError::into_inner);
        if s.refreshing {
            return;
        }
        s.refreshing = true;
    }
    let spawned = std::thread::Builder::new().name("ei-social-refresh".into()).spawn(|| {
        let r = catch_unwind(AssertUnwindSafe(|| engine_json(&["social", "accounts"])));
        let mut s = state().lock().unwrap_or_else(PoisonError::into_inner);
        s.refreshing = false;
        s.last = Some(Instant::now());
        match r {
            Ok(Ok(v)) => {
                s.report = Some(v);
                s.error.clear();
            }
            Ok(Err(e)) => s.error = e,
            Err(_) => s.error = "reading the accounts stopped unexpectedly".into(),
        }
    });
    if spawned.is_err() {
        state().lock().unwrap_or_else(PoisonError::into_inner).refreshing = false;
    }
}

/// Fill the words of every target from the package (in the background).
fn load_words(package: String, language: &'static str) {
    {
        let mut s = state().lock().unwrap_or_else(PoisonError::into_inner);
        if s.loading_words {
            return;
        }
        s.loading_words = true;
    }
    let spawned = std::thread::Builder::new().name("ei-social-words".into()).spawn(move || {
        let r = catch_unwind(AssertUnwindSafe(|| engine_json(&["social", "words", "--package", &package, "--language", language])));
        let mut s = state().lock().unwrap_or_else(PoisonError::into_inner);
        s.loading_words = false;
        match r {
            Ok(Ok(v)) => {
                if let Some(m) = v.get("words").and_then(Value::as_object) {
                    for (t, w) in m {
                        let g = |k: &str| w.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
                        s.words.insert(t.clone(), Words { title: g("title"), text: g("text"), thumbnail: g("thumbnail") });
                    }
                    s.message = "Words filled in from the package: check them below.".into();
                }
            }
            Ok(Err(e)) => s.message = format!("Could not read the package: {e}"),
            Err(_) => s.message = "Reading the package stopped unexpectedly.".into(),
        }
    });
    if spawned.is_err() {
        state().lock().unwrap_or_else(PoisonError::into_inner).loading_words = false;
    }
}

/// Where the engine keeps the posts (as `social.py` does).
fn social_dir() -> std::path::PathBuf {
    std::env::var_os("EI_SOCIAL_HOME").map(std::path::PathBuf::from).unwrap_or_else(|| support_dir().join("social"))
}

fn now_ts() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

fn social_job_running() -> bool {
    jobs().lock().unwrap_or_else(PoisonError::into_inner).iter().any(|j| j.status == Status::Running && j.title.starts_with("Social"))
}

/// Post what is scheduled and due (every minute, whether the window is open or not).
pub(super) fn tick() {
    {
        let mut s = state().lock().unwrap_or_else(PoisonError::into_inner);
        if s.last_due_check.is_some_and(|t| t.elapsed() < Duration::from_secs(60)) {
            return;
        }
        s.last_due_check = Some(Instant::now());
    }
    let path = social_dir().join("posts.json");
    let Some(posts) = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) else {
        return;
    };
    let now = now_ts();
    let due = posts.get("posts").and_then(Value::as_array).into_iter().flatten().any(|p| {
        let at = p.get("at_ts").and_then(Value::as_i64).unwrap_or(0);
        at <= now + 60 && p.get("items").and_then(Value::as_array).into_iter().flatten().any(|i| i.get("status").and_then(Value::as_str) == Some("scheduled"))
    });
    if due && !social_job_running() {
        start_social("Social: post what is due", vec!["social".into(), "run-due".into()]);
    }
}

fn start_social(title: &str, args: Vec<String>) {
    match engine_step(args) {
        Ok((step, engine)) => {
            start_job(title.to_string(), None, vec![step], false, Some(engine));
        }
        Err(e) => state().lock().unwrap_or_else(PoisonError::into_inner).message = e,
    }
}

fn post_args(s: &State, dir: &std::path::Path) -> Result<Vec<String>, String> {
    let video = s.video.trim();
    if video.is_empty() || !std::path::Path::new(video).is_file() {
        return Err("Choose the video to post.".into());
    }
    let targets: Vec<&str> = TARGETS.iter().zip(s.targets).filter(|(_, on)| *on).map(|(t, _)| t.0).collect();
    if targets.is_empty() {
        return Err("Choose where to post it.".into());
    }
    for (id, label, titled, _) in TARGETS.iter().zip(s.targets).filter(|(_, on)| *on).map(|(t, _)| *t) {
        let w = s.words.get(id).cloned().unwrap_or_default();
        if titled && w.title.trim().is_empty() && s.package.trim().is_empty() {
            return Err(format!("{label} needs a title (or choose a package to take it from)."));
        }
        if id == "threads" && w.text.chars().count() > 500 {
            return Err("Threads takes at most 500 characters.".into());
        }
    }
    let mut a: Vec<String> = vec!["social".into(), "post".into(), "--video".into(), video.into(), "--targets".into()];
    a.extend(targets.iter().map(|t| t.to_string()));
    a.extend(["--language".into(), s.language.into(), "--privacy".into(), s.privacy.into()]);
    if !s.package.trim().is_empty() {
        a.extend(["--package".into(), s.package.trim().into()]);
    }
    let when = s.when.trim();
    if !when.is_empty() && !when.eq_ignore_ascii_case("now") {
        a.extend(["--at".into(), when.into()]);
    }
    // the words as edited here
    let edits: serde_json::Map<String, Value> = s
        .words
        .iter()
        .filter(|(t, _)| targets.contains(&t.as_str()))
        .map(|(t, w)| (t.clone(), json!({"title": w.title, "text": w.text, "thumbnail": w.thumbnail})))
        .collect();
    if !edits.is_empty() {
        std::fs::create_dir_all(dir).map_err(|e| format!("could not save the words: {e}"))?;
        let f = dir.join(format!("edits-{}.json", now_ts()));
        std::fs::write(&f, Value::Object(edits).to_string()).map_err(|e| format!("could not save the words: {e}"))?;
        a.extend(["--edits".into(), f.to_string_lossy().to_string()]);
    }
    Ok(a)
}

pub(super) fn route(id: &str, ctx: &egui::Context, params: &Value) -> Option<Result<Value, String>> {
    match id {
        "ai.social" => {
            ctx.data_mut(|m| m.insert_temp(open_id(), true));
            let mut s = state().lock().unwrap_or_else(PoisonError::into_inner);
            let st = |k: &str| params.get(k).and_then(Value::as_str).map(str::to_string);
            if let Some(v) = st("video") {
                s.video = v;
            }
            if let Some(v) = st("package") {
                s.package = v;
            }
            if let Some(v) = st("when") {
                s.when = v;
            }
            s.tab = match st("tab").as_deref() {
                Some("posts") => "posts",
                Some("accounts") => "accounts",
                Some("post") => "post",
                _ => s.tab,
            };
            if let Some(a) = params.get("targets").and_then(Value::as_array) {
                for (k, t) in TARGETS.iter().enumerate() {
                    if let Some(on) = s.targets.get_mut(k) {
                        *on = a.iter().any(|v| v.as_str() == Some(t.0));
                    }
                }
            }
            Some(Ok(json!({"report": s.report, "error": s.error, "message": s.message, "loadingWords": s.loading_words, "words": s.words.len()})))
        }
        _ => None,
    }
}

pub(super) fn show(app: &mut FilmcraftApp, ctx: &egui::Context) {
    tick();
    let mut open = ctx.data(|m| m.get_temp::<bool>(open_id())).unwrap_or(false);
    if !open {
        return;
    }
    let stale = {
        let s = state().lock().unwrap_or_else(PoisonError::into_inner);
        !s.refreshing && s.last.is_none_or(|t| t.elapsed() > Duration::from_secs(if social_job_running() { 4 } else { 20 }))
    };
    if stale {
        refresh();
    }
    ctx.request_repaint_after(Duration::from_millis(700));
    let mut elems: Vec<(String, egui::Rect, String)> = Vec::new();
    let mut pick_video = false;
    let mut pick_package = false;
    let mut pick_secret = false;
    let mut pick_thumb: Option<String> = None;
    let mut open_link: Option<String> = None;
    let mut close = false;
    let warn_c = egui::Color32::from_rgb(0xE5, 0x9B, 0x4B);
    let ok_c = egui::Color32::from_rgb(0x5C, 0xC0, 0x7A);
    let bad_c = egui::Color32::from_rgb(0xE5, 0x6B, 0x6B);
    {
        let mut guard = state().lock().unwrap_or_else(PoisonError::into_inner);
        let s = &mut *guard;
        let report = s.report.clone().unwrap_or(Value::Null);
        let acc = |k: &str| report.get(k).filter(|v| !v.is_null()).cloned();
        let n_posts = report.get("posts").and_then(Value::as_array).map(Vec::len).unwrap_or(0);
        egui::Window::new("Social Posting")
            .id(egui::Id::new("ei-ai-social"))
            .open(&mut open)
            .default_width(680.0)
            .default_height((ctx.content_rect().height() * 0.85).min(820.0))
            .resizable(true)
            .show(ctx, |ui| {
            ui.horizontal(|ui| {
                for (id, label) in [("post", "New Post".to_string()), ("posts", format!("Posts ({n_posts})")), ("accounts", "Accounts".to_string())] {
                    let r = ui.selectable_label(s.tab == id, label);
                    elems.push((format!("ai.social.tab.{id}"), r.rect, id.into()));
                    if r.clicked() && s.tab != id {
                        s.tab = id;
                        s.message.clear();
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let c = ui.button("Close");
                    elems.push(("ai.social.close".into(), c.rect, "Close".into()));
                    close = c.clicked();
                });
            });
            ui.separator();
            if !s.message.is_empty() {
                ui.colored_label(if s.message.starts_with("Could") || s.message.starts_with("Choose") || s.message.contains("needs") { bad_c } else { ok_c }, &s.message);
            }
            if s.tab == "post" {
            ui.horizontal(|ui| {
                let scheduled = !s.when.trim().is_empty() && !s.when.trim().eq_ignore_ascii_case("now");
                let b = ui.add(egui::Button::new(egui::RichText::new(if scheduled { "Schedule" } else { "Post Now" }).color(egui::Color32::WHITE)).fill(app.tokens.accent));
                elems.push(("ai.social.post".into(), b.rect, "Post".into()));
                let d = ui.button("Dry Run").on_hover_text("Goes through everything without posting");
                elems.push(("ai.social.dryRun".into(), d.rect, "Dry Run".into()));
                for (clicked, dry) in [(b.clicked(), false), (d.clicked(), true)] {
                    if !clicked {
                        continue;
                    }
                    match post_args(s, &social_dir()) {
                        Ok(mut a) => {
                            if dry {
                                a.push("--dry-run".into());
                            }
                            let name = std::path::Path::new(s.video.trim()).file_stem().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                            start_social(&format!("Social: {}{name}", if dry { "dry run of " } else if scheduled { "schedule " } else { "post " }), a);
                            s.message = if dry {
                                "Dry run started (AI ▸ AI Jobs shows what would be posted).".into()
                            } else if scheduled {
                                "Scheduled.".into()
                            } else {
                                "Posting (AI ▸ AI Jobs shows the progress).".into()
                            };
                            s.last = None;
                            s.tab = "posts";
                        }
                        Err(e) => s.message = e,
                    }
                }
            });
            }
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                // ---------------- accounts
                let any = acc("youtube").is_some() || acc("meta").is_some() || acc("threads").is_some();
                if !any && s.report.is_some() && s.tab != "accounts" {
                    ui.colored_label(warn_c, "No account is connected yet: see Accounts.");
                }
                if s.tab == "accounts" {
                    ui.weak("Each platform needs a one-time setup on its developer site: the steps are in docs/social.md (in the AI engine folder). Logins are kept in the macOS Keychain.");
                    egui::Grid::new("ei-social-acc").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                        ui.strong("YouTube");
                        ui.horizontal(|ui| match acc("youtube") {
                            Some(a) => {
                                ui.colored_label(ok_c, format!("✔ {}", a.get("name").and_then(Value::as_str).unwrap_or("connected")));
                                let b = ui.button("Disconnect");
                                elems.push(("ai.social.youtube.disconnect".into(), b.rect, "Disconnect".into()));
                                if b.clicked() {
                                    start_social("Social: disconnect YouTube", vec!["social".into(), "disconnect".into(), "youtube".into()]);
                                }
                            }
                            None => {
                                let b = ui.button("Connect…").on_hover_text("Choose your Google OAuth client file (Desktop app); the browser opens to sign in");
                                elems.push(("ai.social.youtube.connect".into(), b.rect, "Connect…".into()));
                                pick_secret = b.clicked();
                                ui.weak("uploads stay private until your API project passes YouTube's audit");
                            }
                        });
                        ui.end_row();
                        ui.strong("Instagram & Facebook");
                        ui.vertical(|ui| match acc("meta") {
                            Some(a) => {
                                ui.horizontal(|ui| {
                                    let ig = a.get("instagram").and_then(Value::as_str).unwrap_or_default();
                                    let page = a.get("page").and_then(Value::as_str).unwrap_or_default();
                                    ui.colored_label(ok_c, if ig.is_empty() { format!("✔ Page “{page}” (no Instagram linked)") } else { format!("✔ @{ig} · Page “{page}”") });
                                    let b = ui.button("Disconnect");
                                    elems.push(("ai.social.meta.disconnect".into(), b.rect, "Disconnect".into()));
                                    if b.clicked() {
                                        start_social("Social: disconnect Instagram & Facebook", vec!["social".into(), "disconnect".into(), "meta".into()]);
                                    }
                                });
                            }
                            None => {
                                for (k, label, field, secret) in [
                                    ("appId", "App ID", &mut s.meta_app_id, false),
                                    ("appSecret", "App secret", &mut s.meta_app_secret, true),
                                    ("token", "User token", &mut s.meta_token, true),
                                ] {
                                    ui.horizontal(|ui| {
                                        ui.label(label);
                                        let r = ui.add(egui::TextEdit::singleline(field).password(secret).desired_width(300.0));
                                        elems.push((format!("ai.social.meta.{k}"), r.rect, String::new()));
                                    });
                                }
                                let ready = !s.meta_app_id.trim().is_empty() && !s.meta_app_secret.trim().is_empty() && !s.meta_token.trim().is_empty();
                                let b = ui.add_enabled(ready, egui::Button::new("Connect"));
                                elems.push(("ai.social.meta.connect".into(), b.rect, "Connect".into()));
                                if b.clicked() {
                                    start_social(
                                        "Social: connect Instagram & Facebook",
                                        vec![
                                            "social".into(),
                                            "connect".into(),
                                            "meta".into(),
                                            "--app-id".into(),
                                            s.meta_app_id.trim().into(),
                                            "--app-secret".into(),
                                            s.meta_app_secret.trim().into(),
                                            "--token".into(),
                                            s.meta_token.trim().into(),
                                        ],
                                    );
                                    s.meta_token.clear();
                                    s.meta_app_secret.clear();
                                }
                            }
                        });
                        ui.end_row();
                        ui.strong("Threads");
                        ui.vertical(|ui| match acc("threads") {
                            Some(a) => {
                                ui.horizontal(|ui| {
                                    ui.colored_label(ok_c, format!("✔ @{}", a.get("name").and_then(Value::as_str).unwrap_or_default()));
                                    let b = ui.button("Disconnect");
                                    elems.push(("ai.social.threads.disconnect".into(), b.rect, "Disconnect".into()));
                                    if b.clicked() {
                                        start_social("Social: disconnect Threads", vec!["social".into(), "disconnect".into(), "threads".into()]);
                                    }
                                });
                            }
                            None => {
                                for (k, label, field) in [("token", "Token", &mut s.threads_token), ("appSecret", "App secret (optional)", &mut s.threads_secret)] {
                                    ui.horizontal(|ui| {
                                        ui.label(label);
                                        let r = ui.add(egui::TextEdit::singleline(field).password(true).desired_width(300.0));
                                        elems.push((format!("ai.social.threads.{k}"), r.rect, String::new()));
                                    });
                                }
                                let b = ui.add_enabled(!s.threads_token.trim().is_empty(), egui::Button::new("Connect"));
                                elems.push(("ai.social.threads.connect".into(), b.rect, "Connect".into()));
                                if b.clicked() {
                                    let mut a: Vec<String> = vec!["social".into(), "connect".into(), "threads".into(), "--token".into(), s.threads_token.trim().into()];
                                    if !s.threads_secret.trim().is_empty() {
                                        a.extend(["--app-secret".into(), s.threads_secret.trim().into()]);
                                    }
                                    start_social("Social: connect Threads", a);
                                    s.threads_token.clear();
                                    s.threads_secret.clear();
                                }
                            }
                        });
                        ui.end_row();
                    });
                }
                if s.tab == "post" {
                // ---------------- a new post

                egui::Grid::new("ei-social-new").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                    ui.label("Video");
                    pick_video = path_row(ui, &mut s.video, "ai.social.video", "the finished video", &mut elems);
                    ui.end_row();
                    ui.label("Words from");
                    ui.vertical(|ui| {
                        pick_package = path_row(ui, &mut s.package, "ai.social.package", "a “… - package” folder (Thumbnails, Titles & Hashtags)", &mut elems);
                        ui.horizontal(|ui| {
                            radios(ui, &mut s.language, &LANGS, "ai.social.language", &mut elems);
                            let b = ui.add_enabled(!s.package.trim().is_empty() && !s.loading_words, egui::Button::new(if s.loading_words { "Reading…" } else { "Fill In the Words" }));
                            elems.push(("ai.social.fill".into(), b.rect, "Fill In the Words".into()));
                            if b.clicked() {
                                let (p, l) = (s.package.trim().to_string(), s.language);
                                drop_words_later(p, l);
                            }
                        });
                    });
                    ui.end_row();
                    ui.label("Post to");
                    ui.vertical(|ui| {
                        for (k, (id, label, _, _)) in TARGETS.iter().enumerate() {
                            let need = match *id {
                                "youtube" | "shorts" => "youtube",
                                "threads" => "threads",
                                _ => "meta",
                            };
                            if let Some(on) = s.targets.get_mut(k) {
                                ui.horizontal(|ui| {
                                    let r = ui.checkbox(on, *label);
                                    elems.push((format!("ai.social.target.{id}"), r.rect, on.to_string()));
                                    if *on && acc(need).is_none() {
                                        ui.colored_label(warn_c, "not connected");
                                    }
                                });
                            }
                        }
                    });
                    ui.end_row();
                    ui.label("When");
                    ui.vertical(|ui| {
                        let r = ui.add(egui::TextEdit::singleline(&mut s.when).hint_text("now, or e.g. 2026-11-20 18:00 (this Mac's time)").desired_width(300.0));
                        elems.push(("ai.social.when".into(), r.rect, s.when.clone()));
                        ui.weak("YouTube and Facebook publish at that time themselves; Instagram and Threads are posted by EI then (keep EI open).");
                    });
                    ui.end_row();
                    if s.targets.first().copied().unwrap_or(false) || s.targets.get(1).copied().unwrap_or(false) {
                        ui.label("YouTube");
                        ui.horizontal(|ui| radios(ui, &mut s.privacy, &PRIVACY, "ai.social.privacy", &mut elems));
                        ui.end_row();
                    }
                });
                // the words per target
                for (k, (id, label, titled, thumbed)) in TARGETS.iter().enumerate() {
                    if !s.targets.get(k).copied().unwrap_or(false) {
                        continue;
                    }
                    let w = s.words.entry(id.to_string()).or_default();
                    egui::CollapsingHeader::new(format!("{label}: the words")).id_salt(("ei-social-words", *id)).default_open(true).show(ui, |ui| {
                        if *titled {
                            ui.horizontal(|ui| {
                                ui.label("Title");
                                let r = ui.add(egui::TextEdit::singleline(&mut w.title).desired_width(480.0));
                                elems.push((format!("ai.social.{id}.title"), r.rect, w.title.clone()));
                                let n = w.title.chars().count();
                                ui.colored_label(if n > 100 { bad_c } else { egui::Color32::GRAY }, format!("{n}/100"));
                            });
                        }
                        let r = ui.add(egui::TextEdit::multiline(&mut w.text).desired_rows(3).desired_width(560.0).hint_text(match *id {
                            "youtube" | "facebook" => "description",
                            "threads" => "the post (the video's link is added)",
                            _ => "caption with hashtags (Instagram allows 5)",
                        }));
                        elems.push((format!("ai.social.{id}.text"), r.rect, w.text.clone()));
                        let tags = w.text.split_whitespace().filter(|t| t.starts_with('#')).count();
                        if matches!(*id, "reels" | "fbreels") && tags > 5 {
                            ui.colored_label(warn_c, format!("{tags} hashtags: Instagram keeps at most 5"));
                        }
                        if *id == "threads" && w.text.chars().count() > 450 {
                            ui.colored_label(warn_c, "Threads takes 500 characters, the link included");
                        }
                        if *thumbed {
                            ui.horizontal(|ui| {
                                ui.label("Thumbnail");
                                let r = ui.add(egui::TextEdit::singleline(&mut w.thumbnail).hint_text("optional .jpg").desired_width(400.0));
                                elems.push((format!("ai.social.{id}.thumbnail"), r.rect, w.thumbnail.clone()));
                                let b = ui.button("Choose…");
                                if b.clicked() {
                                    pick_thumb = Some(id.to_string());
                                }
                            });
                        }
                    });
                }
                }
                if s.tab == "posts" {
                let posts = report.get("posts").and_then(Value::as_array).cloned().unwrap_or_default();
                if posts.is_empty() {
                    ui.weak("Nothing posted yet.");
                }
                if !s.error.is_empty() {
                    ui.colored_label(bad_c, &s.error);
                }
                for p in posts.iter().rev() {
                    let pid = p.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
                    let video = p.get("video").and_then(Value::as_str).map(|v| std::path::Path::new(v).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()).unwrap_or_default();
                    let at = p.get("at").and_then(Value::as_str).unwrap_or_default();
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(video).strong());
                        ui.weak(if at.is_empty() { p.get("created").and_then(Value::as_str).unwrap_or_default().chars().take(16).collect::<String>().replace('T', " ") } else { format!("at {}", at.replace('T', " ")) });
                    });
                    let items = p.get("items").and_then(Value::as_array).cloned().unwrap_or_default();
                    let mut failed = false;
                    let mut pending = false;
                    for it in &items {
                        let t = it.get("target").and_then(Value::as_str).unwrap_or_default();
                        let st = it.get("status").and_then(Value::as_str).unwrap_or_default();
                        let res = it.get("result").cloned().unwrap_or(Value::Null);
                        failed |= st == "failed";
                        pending |= matches!(st, "scheduled" | "waiting");
                        ui.horizontal(|ui| {
                            ui.add_space(12.0);
                            ui.label(TARGETS.iter().find(|x| x.0 == t).map(|x| x.1).unwrap_or(t));
                            let (c, label) = match st {
                                "posted" => (ok_c, "posted"),
                                "failed" => (bad_c, "failed"),
                                "posting" => (warn_c, "posting…"),
                                "scheduled" => (egui::Color32::GRAY, "scheduled"),
                                "cancelled" => (egui::Color32::GRAY, "cancelled"),
                                _ => (egui::Color32::GRAY, st),
                            };
                            ui.colored_label(c, label);
                            if let Some(u) = res.get("url").and_then(Value::as_str).filter(|u| !u.is_empty()) {
                                let b = ui.small_button("Open");
                                elems.push((format!("ai.social.post.{pid}.{t}.open"), b.rect, "Open".into()));
                                if b.clicked() {
                                    open_link = Some(u.to_string());
                                }
                            }
                            for k in ["error", "note"] {
                                if let Some(e) = res.get(k).and_then(Value::as_str) {
                                    ui.weak(e.chars().take(140).collect::<String>());
                                }
                            }
                        });
                    }
                    ui.horizontal(|ui| {
                        ui.add_space(12.0);
                        if failed {
                            let b = ui.small_button("Retry Failed");
                            elems.push((format!("ai.social.post.{pid}.retry"), b.rect, "Retry Failed".into()));
                            if b.clicked() {
                                start_social("Social: retry", vec!["social".into(), "run".into(), pid.clone(), "--failed".into()]);
                            }
                        }
                        if pending {
                            let b = ui.small_button("Cancel");
                            elems.push((format!("ai.social.post.{pid}.cancel"), b.rect, "Cancel".into()));
                            if b.clicked() {
                                start_social("Social: cancel", vec!["social".into(), "cancel".into(), pid.clone()]);
                            }
                        }
                    });
                }
                }
            });
        });
    }
    for (id, r, l) in elems {
        app.auto.add(&id, r, &l);
    }
    // pickers (the state lock is not held while a dialog is open)
    if pick_video && let Some(f) = app.hooks.pick_open_file.as_mut().and_then(|f| f("Video", &["mp4", "mov", "m4v"])) {
        let mut s = state().lock().unwrap_or_else(PoisonError::into_inner);
        s.video = f.clone();
        // a package made next to the video is a good guess
        if s.package.trim().is_empty()
            && let Some(dir) = std::path::Path::new(&f).parent()
            && let Ok(rd) = std::fs::read_dir(dir)
            && let Some(p) = rd.flatten().map(|e| e.path()).find(|p| p.is_dir() && p.file_name().is_some_and(|n| n.to_string_lossy().ends_with(" - package")))
        {
            s.package = p.to_string_lossy().to_string();
        }
    }
    if pick_package && let Some(d) = app.hooks.pick_folder.as_mut().and_then(|f| f()) {
        let lang = {
            let mut s = state().lock().unwrap_or_else(PoisonError::into_inner);
            s.package = d.clone();
            s.language
        };
        load_words(d, lang);
    }
    if pick_secret && let Some(f) = app.hooks.pick_open_file.as_mut().and_then(|f| f("Google OAuth client", &["json"])) {
        start_social("Social: connect YouTube (finish in the browser)", vec!["social".into(), "connect".into(), "youtube".into(), "--client-secret".into(), f]);
    }
    if let Some(t) = pick_thumb
        && let Some(f) = app.hooks.pick_open_file.as_mut().and_then(|f| f("Image", &["jpg", "jpeg", "png"]))
    {
        state().lock().unwrap_or_else(PoisonError::into_inner).words.entry(t).or_default().thumbnail = f;
    }
    if let Some(u) = open_link
        && let Some(f) = app.hooks.open_path.as_mut()
        && let Err(e) = f(&u, false)
    {
        app.ui.status = e;
    }
    if let Some((p, l)) = take_words_later() {
        load_words(p, l);
    }
    if close {
        open = false;
    }
    ctx.data_mut(|m| m.insert_temp(open_id(), open));
}

/// Choices side by side (they take one line, the dialogs' radio rows one line each).
fn radios(ui: &mut egui::Ui, v: &mut &'static str, opts: &[(&'static str, &'static str)], id: &str, elems: &mut Vec<(String, egui::Rect, String)>) {
    for (key, label) in opts {
        let r = ui.radio_value(v, *key, *label);
        elems.push((format!("{id}.{key}"), r.rect, label.to_string()));
    }
}

// "Fill In the Words" is clicked while the state is locked: remembered, then started after.
fn later() -> &'static Mutex<Option<(String, &'static str)>> {
    static L: OnceLock<Mutex<Option<(String, &'static str)>>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(None))
}

fn drop_words_later(p: String, l: &'static str) {
    *later().lock().unwrap_or_else(PoisonError::into_inner) = Some((p, l));
}

fn take_words_later() -> Option<(String, &'static str)> {
    later().lock().unwrap_or_else(PoisonError::into_inner).take()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_arguments_carry_targets_time_and_edits() {
        let dir = std::env::temp_dir().join(format!("ei-social-args-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let video = dir.join("v.mp4");
        let _ = std::fs::write(&video, b"x");
        let mut s = State { language: "hi", privacy: "unlisted", video: video.to_string_lossy().to_string(), ..State::default() };
        assert!(post_args(&s, &dir).is_err(), "no target");
        s.targets = [true, false, true, false, false, true];
        assert!(post_args(&s, &dir).unwrap_err().contains("title"), "YouTube needs a title without a package");
        s.words.insert("youtube".into(), Words { title: "T".into(), text: "D".into(), thumbnail: String::new() });
        s.when = "2026-12-01 18:00".into();
        let a = post_args(&s, &dir).unwrap();
        assert!(a.windows(4).any(|w| w == ["--targets", "youtube", "reels", "threads"]), "{a:?}");
        assert!(a.windows(2).any(|w| w == ["--at", "2026-12-01 18:00"]) && a.windows(2).any(|w| w == ["--language", "hi"]));
        assert!(a.windows(2).any(|w| w == ["--privacy", "unlisted"]) && a.contains(&"--edits".to_string()));
        s.words.insert("threads".into(), Words { title: String::new(), text: "x".repeat(501), thumbnail: String::new() });
        assert!(post_args(&s, &dir).unwrap_err().contains("500"));
        s.video = "/no/such/video.mp4".into();
        assert!(post_args(&s, &dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
