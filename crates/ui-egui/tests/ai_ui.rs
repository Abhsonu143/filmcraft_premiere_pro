//! Headless UI test of EI's AI menu: every dialog opens with its controls, and real jobs run the
//! Python engine end to end (split a video, auto edit clips to music) and import their results.
//!
//! The job tests need ffmpeg and the engine's Python (`ai-engine/.venv` or EI's own environment);
//! without them they are skipped. Set `FILMCRAFT_UI_SNAPSHOT_DIR=<dir>` to also write `ai-*.png`.

use std::sync::mpsc::{Sender, channel};
use std::time::{Duration, Instant};

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use filmcraft_engine::Session;
use filmcraft_ui_egui::FilmcraftApp;
use filmcraft_ui_egui::control::ControlRequest;
use serde_json::{Value, json};

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
    snapshots: Option<std::path::PathBuf>,
}

impl Driver {
    fn demo() -> Self {
        let mut session = Session::default();
        session.execute("file.openDemoProject", json!({})).expect("demo project");
        let (tx, rx) = channel();
        let app = FilmcraftApp::new(session).with_control(rx);
        let snapshots = std::env::var_os("FILMCRAFT_UI_SNAPSHOT_DIR").map(std::path::PathBuf::from);
        let mut b = Harness::builder().with_size(egui::vec2(1600.0, 980.0)).with_max_steps(10_000);
        if snapshots.is_some() {
            b = b.wgpu();
        }
        let harness = b.build_eframe(move |_cc| app);
        let mut d = Driver { harness, tx, snapshots };
        d.frames(4);
        d
    }

    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            let ctx = self.harness.ctx.clone();
            let mut raw = std::mem::take(self.harness.input_mut());
            eframe::App::raw_input_hook(self.harness.state_mut(), &ctx, &mut raw);
            *self.harness.input_mut() = raw;
            self.harness.step();
        }
    }

    fn call(&mut self, method: &str, params: Value) -> Value {
        let (req, reply) = ControlRequest::new(method, params.clone());
        self.tx.send(req).unwrap();
        for _ in 0..600 {
            self.frames(1);
            if let Ok(v) = reply.try_recv() {
                return v;
            }
        }
        panic!("no reply to {method} {params}");
    }

    fn ok(&mut self, method: &str, params: Value) -> Value {
        let v = self.call(method, params.clone());
        assert_eq!(v["ok"], json!(true), "{method} {params} failed: {v}");
        v["result"].clone()
    }

    fn run(&mut self, command: &str, params: Value) -> Value {
        self.ok("engine.execute", json!({"command": command, "params": params}))
    }

    fn ids(&mut self, prefix: &str) -> Vec<String> {
        let v = self.ok("ui.elements", json!({"prefix": prefix}));
        v.as_array().unwrap().iter().filter_map(|e| e["id"].as_str().map(str::to_string)).collect()
    }

    fn snapshot(&mut self, name: &str) {
        let Some(dir) = self.snapshots.clone() else { return };
        self.frames(2);
        match self.harness.render() {
            Ok(img) => {
                std::fs::create_dir_all(&dir).unwrap();
                img.save(dir.join(format!("ai-{name}.png"))).unwrap();
            }
            Err(e) => eprintln!("snapshot {name} skipped: {e}"),
        }
    }

    /// Wait for a job to finish; returns its summary.
    fn wait_job(&mut self, job: u64, limit: Duration) -> Value {
        let end = Instant::now() + limit;
        loop {
            let list = self.run("ai.jobs", json!({}));
            let j = list["jobs"].as_array().unwrap().iter().find(|j| j["job"] == json!(job)).cloned().expect("job listed");
            if j["status"] != json!("running") {
                return j;
            }
            assert!(Instant::now() < end, "job {job} still running: {j}");
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

#[test]
fn ai_menu_and_dialogs() {
    let mut d = Driver::demo();
    let menu = d.ok("ui.menu.list", json!({})).to_string();
    for id in ["ai.split", "ai.music", "ai.reference", "ai.talk", "ai.voiceover", "ai.clips", "ai.audio", "ai.jobs", "ai.setup"] {
        assert!(menu.contains(id), "{id} in the menus");
    }
    let expect: [(&str, &[&str]); 11] = [
        ("split", &["input", "clipLength", "quality.same", "quality.exact", "out", "import", "run"]),
        (
            "music",
            &[
                "clips.add",
                "clips.addFolder",
                "music",
                "length",
                "aspect.horizontal",
                "aspect.vertical",
                "aspect.square",
                "pace.fast",
                "musicStart",
                "look",
                "effects",
                "order",
                "run",
            ],
        ),
        ("reference", &["input", "clips.add", "music", "aspect.same", "aspect.square", "look", "effects", "order", "run"]),
        ("talk", &["input", "preset.short", "captions", "music", "run"]),
        ("voiceover", &["input", "clips.add", "script", "captions", "captionStyle", "aspect.vertical", "pace.fast", "describe", "ambience", "run"]),
        (
            "script",
            &[
                "topic",
                "niche",
                "tone",
                "notes",
                "seconds",
                "language.hinglish",
                "style",
                "styleLinks",
                "learnStyle",
                "writer",
                "voice",
                "clips.add",
                "out",
                "run",
            ],
        ),
        ("tv", &["input", "recordings", "aspect.horizontal", "aspect.vertical", "relight", "out", "run"]),
        (
            "topic",
            &[
                "topic",
                "seconds",
                "seconds.60",
                "language.hinglish",
                "style",
                "writer",
                "video",
                "fresh",
                "voice",
                "free",
                "cards",
                "youtube",
                "captions",
                "out",
                "run",
            ],
        ),
        ("voice", &["name", "lang.hi", "lang.mix", "lang.en", "source.record", "source.files", "run"]),
        (
            "clips",
            &[
                "input",
                "count",
                "min",
                "target",
                "max",
                "aspect.vertical",
                "aspect.same",
                "content",
                "framing",
                "captions",
                "captionStyle",
                "titleMode",
                "pace.fast",
                "music",
                "llm",
                "hook",
                "out",
                "run",
            ],
        ),
        ("audio", &["clips.add", "clips.addFolder", "format", "quality", "start", "end", "mono", "normalize", "out", "run"]),
    ];
    for (kind, fields) in expect {
        let r = d.run(&format!("ai.{kind}"), json!({}));
        assert_eq!(r["dialog"], json!(format!("ai.{kind}")));
        d.frames(3);
        let ids = d.ids(&format!("ai.{kind}."));
        for f in fields {
            let id = format!("ai.{kind}.{f}");
            assert!(ids.contains(&id), "{id} in {ids:?}");
        }
        d.snapshot(kind);
        // Run with nothing filled in: the dialog stays open with an error, nothing starts.
        d.ok("ui.click", json!({"id": format!("ai.{kind}.run")}));
        d.frames(3);
        assert!(!d.ids(&format!("ai.{kind}.run")).is_empty(), "{kind}: dialog stays open on a missing input");
        d.ok("ui.key", json!({"key": "Escape"}));
        d.frames(3);
        assert!(d.ids(&format!("ai.{kind}.")).is_empty(), "{kind}: Escape closes the dialog");
    }
    // Hostile parameters are errors, not crashes.
    let bad = d.call("engine.execute", json!({"command": "ai.split", "params": {"input": "/no/such.mp4", "clipLength": -1}}));
    assert_eq!(bad["ok"], json!(false));
    let bad = d.call("engine.execute", json!({"command": "ai.cancel", "params": {"job": "x"}}));
    assert_eq!(bad["ok"], json!(false));
    // Set-up and jobs windows.
    let setup = d.run("ai.setup", json!({}));
    assert!(setup["engine"].as_str().is_some_and(|e| e.ends_with("ai-engine")), "{setup}");
    d.frames(3);
    assert!(!d.ids("ai.setup.").is_empty());
    d.snapshot("setup");
    d.ok("ui.click", json!({"id": "ai.setup.close"}));
    // Home screen: the AI tools open their dialogs.
    d.run("mode.import", json!({}));
    d.frames(3);
    d.ok("ui.click", json!({"id": "import.ai.music"}));
    d.frames(3);
    assert!(!d.ids("ai.music.run").is_empty());
    d.snapshot("home");
}

fn have_tools() -> bool {
    let ff = std::process::Command::new("ffmpeg").arg("-version").output().is_ok_and(|o| o.status.success());
    let engine = filmcraft_ui_egui::panels::ai_tools::engine_dir();
    let py = filmcraft_ui_egui::panels::ai_tools::python(engine.as_deref());
    let numpy = std::process::Command::new(&py).args(["-c", "import numpy"]).output().is_ok_and(|o| o.status.success());
    if !(ff && numpy && engine.is_some()) {
        eprintln!("skipped: needs ffmpeg and the AI engine's Python (ffmpeg {ff}, numpy {numpy}, engine {engine:?})");
    }
    ff && numpy && engine.is_some()
}

fn ffmpeg(args: &[&str]) {
    let ok = std::process::Command::new("ffmpeg").args(["-hide_banner", "-loglevel", "error", "-y"]).args(args).status().is_ok_and(|s| s.success());
    assert!(ok, "ffmpeg {args:?}");
}

#[test]
fn ai_jobs_run_the_engine_and_import_results() {
    if !have_tools() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("ei-ai-ui-{}", std::process::id()));
    let clips = dir.join("clips");
    std::fs::create_dir_all(&clips).unwrap();
    let long = dir.join("long.mp4");
    let l = long.to_string_lossy().to_string();
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=640x360:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=d=12",
        "-t",
        "12",
        "-g",
        "30",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-c:a",
        "aac",
        &l,
    ]);
    for (i, src) in ["testsrc2", "mandelbrot", "smptehdbars"].iter().enumerate() {
        let p = clips.join(format!("c{i}.mp4")).to_string_lossy().to_string();
        ffmpeg(&["-f", "lavfi", "-i", &format!("{src}=size=640x360:rate=30"), "-t", "4", "-an", "-c:v", "libx264", "-preset", "ultrafast", &p]);
    }
    let music = dir.join("beat.wav").to_string_lossy().to_string();
    ffmpeg(&["-f", "lavfi", "-i", "aevalsrc='sin(2*PI*60*t)*exp(-30*mod(t,0.5))':s=44100:d=20", &music]);

    let mut d = Driver::demo();
    let items_before = d.harness.state().session.project.items.len();
    // Split: 12 s into 5 s clips, imported into the project.
    let out_dir = dir.join("parts").to_string_lossy().to_string();
    let r = d.run("ai.split", json!({"input": l, "outDir": out_dir, "clipLength": 5, "import": true}));
    let job = r["job"].as_u64().expect("job id");
    let j = d.wait_job(job, Duration::from_secs(120));
    assert_eq!(j["status"], json!("done"), "{j}");
    let outputs = j["result"]["outputs"].as_array().cloned().unwrap_or_default();
    assert_eq!(outputs.len(), 3, "{j}");
    d.frames(3);
    let items_split = d.harness.state().session.project.items.len();
    assert!(items_split >= items_before + 3, "split clips imported: {items_before} -> {items_split}");
    // Music edit: square, 6 s, opened as a sequence.
    let out = dir.join("edit.mp4").to_string_lossy().to_string();
    let r = d.run("ai.music", json!({"clips": [clips.to_string_lossy()], "music": music, "out": out, "length": 6, "aspect": "square", "import": true}));
    let job = r["job"].as_u64().expect("job id");
    d.frames(2);
    assert!(!d.ids("ai.jobs.clearFinished").is_empty(), "the jobs window opened with the job");
    d.snapshot("jobs");
    let j = d.wait_job(job, Duration::from_secs(300));
    assert_eq!(j["status"], json!("done"), "{j}");
    assert!(std::path::Path::new(&out).is_file());
    d.frames(3);
    let session = &d.harness.state().session;
    let square = session.project.items.values().filter_map(|it| session.project.sequence(it.id)).any(|s| s.settings.width == 1080 && s.settings.height == 1080);
    assert!(square, "the 1080x1080 edit was imported as a sequence");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The Voice Clone recorder: the sentences follow the chosen languages (Hindi, Hinglish, English),
/// each language is a section with its own progress, and changing the languages reloads them.
#[test]
fn voice_recorder_follows_the_languages() {
    if !have_tools() {
        return;
    }
    let mut d = Driver::demo();
    d.run("ai.voice", json!({}));
    let wait_for = |d: &mut Driver, id: &str, present: bool| {
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(60) {
            d.frames(2);
            if d.ids(id).iter().any(|x| x == id) == present {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        panic!("{id} {}", if present { "never appeared" } else { "never went away" });
    };
    for set in ["hi", "mix", "en"] {
        wait_for(&mut d, &format!("ai.voice.section.{set}"), true);
    }
    d.snapshot("voice-recorder");
    // English only: the Hindi and Hinglish sections go, English stays
    d.ok("ui.click", json!({"id": "ai.voice.lang.hi"}));
    d.frames(3);
    d.ok("ui.click", json!({"id": "ai.voice.lang.mix"}));
    d.frames(3);
    wait_for(&mut d, "ai.voice.section.hi", false);
    wait_for(&mut d, "ai.voice.section.mix", false);
    wait_for(&mut d, "ai.voice.section.en", true);
    let prompt = d.ok("ui.elements", json!({"prefix": "ai.voice.prompt"}));
    let text = prompt.as_array().and_then(|a| a.first()).and_then(|e| e["label"].as_str()).unwrap_or_default().to_string();
    assert!(text.is_ascii() && !text.is_empty(), "an English sentence: {text}");
    d.snapshot("voice-recorder-en");
}

/// Records one sentence through the real microphone while a test sentence plays on the speakers:
/// the take is accepted and can be played. Opt-in (needs speakers + microphone):
/// `EI_TEST_MIC=<wav> EI_SETTINGS=<tmp>/settings.json` (EI_SETTINGS keeps the take out of your real voices).
#[test]
fn voice_recorder_records_a_take() {
    let Some(wav) = std::env::var_os("EI_TEST_MIC") else { return };
    let Some(settings) = std::env::var_os("EI_SETTINGS").map(std::path::PathBuf::from) else {
        eprintln!("skipped: set EI_SETTINGS to a temporary settings.json too");
        return;
    };
    if !have_tools() {
        return;
    }
    let mut d = Driver::demo();
    d.run("ai.voice", json!({}));
    let t0 = Instant::now();
    while !d.ids("ai.voice.record").iter().any(|x| x == "ai.voice.record") {
        assert!(t0.elapsed() < Duration::from_secs(60), "sentences never loaded");
        d.frames(2);
        std::thread::sleep(Duration::from_millis(100));
    }
    d.ok("ui.click", json!({"id": "ai.voice.section.en"}));
    d.frames(3);
    d.ok("ui.click", json!({"id": "ai.voice.record"}));
    let player = std::process::Command::new("sh").arg("-c").arg(format!("sleep 1.5; afplay '{}'", wav.to_string_lossy())).spawn().unwrap();
    let t0 = Instant::now();
    let mut saw_live = false;
    loop {
        d.frames(2);
        let e = d.ok("ui.elements", json!({"prefix": "ai.voice.play"}));
        let enabled = e.as_array().and_then(|a| a.first()).is_some_and(|x| x["enabled"].as_bool().unwrap_or(true));
        saw_live |= d.harness.query_by_label_contains("hearing you").is_some();
        if t0.elapsed() > Duration::from_secs(5) && enabled && d.harness.query_by_label_contains("Saved").is_some() {
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(45), "the take was never accepted");
        std::thread::sleep(Duration::from_millis(100));
    }
    let mut player = player;
    let _ = player.wait();
    assert!(saw_live, "live status shown while recording");
    // the recorder stays on the sentence just recorded: its take can be played right away
    let sections = d.ok("ui.elements", json!({"prefix": "ai.voice.section.en"}));
    assert!(sections.to_string().contains("English 1/"), "{sections}");
    d.snapshot("voice-recorded");
    let dir = settings.parent().unwrap().join("voices/My-voice/takes");
    assert!(std::fs::read_dir(&dir).unwrap().filter_map(Result::ok).any(|f| f.path().extension().is_some_and(|x| x == "wav")));
}

/// TV Light Sync with real recordings: the folder is listed with orientations, the corner check
/// opens with EI's suggestion, a corner can be dragged and the corners are saved. Opt-in:
/// `EI_TEST_TV=<folder with recordings> EI_SETTINGS=<tmp>/settings.json` (and a matching
/// `AUTOCUT_CACHE=<tmp>` keeps the calibration out of your real cache).
#[test]
fn tv_light_corner_check() {
    let Some(folder) = std::env::var_os("EI_TEST_TV") else { return };
    let Some(settings) = std::env::var_os("EI_SETTINGS").map(std::path::PathBuf::from) else { return };
    if !have_tools() {
        return;
    }
    std::fs::create_dir_all(settings.parent().unwrap()).unwrap();
    std::fs::write(&settings, serde_json::to_string(&json!({"tv_recordings": folder.to_string_lossy()})).unwrap()).unwrap();
    let mut d = Driver::demo();
    d.run("ai.tv", json!({}));
    let wait = |d: &mut Driver, id: &str, secs: u64| {
        let t0 = Instant::now();
        while !d.ids(id).iter().any(|x| x.starts_with(id)) {
            assert!(t0.elapsed() < Duration::from_secs(secs), "{id} never appeared");
            d.frames(2);
            std::thread::sleep(Duration::from_millis(150));
        }
    };
    wait(&mut d, "ai.tv.calibrate.", 60);
    d.snapshot("tv-list");
    let first = d.ids("ai.tv.calibrate.").into_iter().next().unwrap();
    d.frames(12); // the dialog grows with the list and re-centres: let it settle before clicking
    d.ok("ui.click", json!({"id": first}));
    wait(&mut d, "ai.tv.corner.0", 180);
    d.frames(3);
    d.snapshot("tv-corners");
    d.frames(12);
    d.ok("ui.drag", json!({"from": {"id": "ai.tv.corner.0"}, "to": {"id": "ai.tv.corner.0", "dx": 6, "dy": 4}}));
    d.frames(4);
    d.ok("ui.click", json!({"id": "ai.tv.corners.save"}));
    let t0 = Instant::now();
    while d.ids("ai.tv.corner.0").iter().any(|x| x == "ai.tv.corner.0") {
        assert!(t0.elapsed() < Duration::from_secs(60), "corners never saved");
        d.frames(2);
        std::thread::sleep(Duration::from_millis(150));
    }
    d.frames(3);
    d.snapshot("tv-saved");
    assert!(d.harness.query_by_label_contains("corners checked").is_some());
}
