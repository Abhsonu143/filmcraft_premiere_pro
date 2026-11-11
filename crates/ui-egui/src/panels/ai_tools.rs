//! EI's AI menu: dialogs that run the Python AI engine (`ai-engine/ei_engine.py`) as a separate
//! process, a jobs window that shows their progress, and an engine set-up window.
//!
//! The engine is a separate process on purpose: a heavy model or a crash there never freezes or
//! takes down the editor. Each job's stderr is its progress log; its last stdout line
//! `EI_RESULT {json}` carries the outputs. When a job finishes and "Open in EI" was ticked, the
//! result (`.otio` + `.srt`, or the split clips) is imported into the project on the UI thread.
//!
//! Where things are found (first match wins):
//! - engine: `$EI_ENGINE_DIR`; `EI.app/Contents/Resources/ai-engine`; an `ai-engine/` folder next
//!   to the executable or up to six folders above it (a development build in `target/`).
//! - Python: `$EI_PYTHON`; `~/Library/Application Support/EI/venv` (made by AI ▸ AI Engine
//!   Setup); `ai-engine/.venv`; `python3`.
//! - ffmpeg: the engine looks it up on `PATH`, which gets Homebrew's folders prepended (apps started
//!   from the Finder don't inherit the shell's `PATH`).
//!
//! Commands (all agent-drivable; without parameters they open the dialog):
//! - `ai.split {input, outDir?, clipLength?=60, quality?=same|exact|high|medium|small, import?}`
//! - `ai.music {clips: [..], music, out?, length?=30, aspect?=vertical|horizontal|square, pace?, import?}`
//! - `ai.reference {input, clips: [..], out?, aspect?=same|…, import?}`
//! - `ai.talk {input, out?, preset?=long|short|travel, captions?, music?, import?}`
//! - `ai.voiceover {input, clips: [..], out?, script?, music?, captions?, preset?, import?}`
//! - each returns `{job}`; `ai.jobs` returns every job (and shows the window); `ai.cancel {job}`;
//!   `ai.setup` shows the set-up window (`{install: true}` installs the Python packages).
//!
//! Automation ids: `ai.<kind>.<field>` for every field (`input`, `input.choose`, `clips.add`,
//! `clips.addFolder`, `clips.clear`, `music`, `music.choose`, `script`, `script.choose`, `out`,
//! `out.choose`, `clipLength`, `quality.<q>`, `length`, `aspect.<a>`, `pace.<p>`, `preset.<p>`,
//! `captions`, `import`), `ai.<kind>.run`, `ai.<kind>.cancel`; jobs window `ai.jobs.<n>.cancel`,
//! `ai.jobs.<n>.reveal`, `ai.jobs.<n>.open`, `ai.jobs.<n>.import`, `ai.jobs.clearFinished`; set-up
//! window `ai.setup.install`, `ai.setup.check`, `ai.setup.close`.

use std::io::{BufRead, BufReader};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::FilmcraftApp;

/// Most log lines kept per job (older ones are dropped).
const MAX_LOG: usize = 400;
/// Most jobs kept in the list (finished ones are dropped first).
const MAX_JOBS: usize = 50;

// ---------------------------------------------------------------------------------------------
// Dialog drafts
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Kind {
    #[default]
    Split,
    Music,
    Reference,
    Talk,
    Voiceover,
}

impl Kind {
    fn id(self) -> &'static str {
        match self {
            Kind::Split => "split",
            Kind::Music => "music",
            Kind::Reference => "reference",
            Kind::Talk => "talk",
            Kind::Voiceover => "voiceover",
        }
    }
    fn from_command(id: &str) -> Option<Kind> {
        Some(match id {
            "ai.split" => Kind::Split,
            "ai.music" => Kind::Music,
            "ai.reference" => Kind::Reference,
            "ai.talk" => Kind::Talk,
            "ai.voiceover" => Kind::Voiceover,
            _ => return None,
        })
    }
    fn title(self) -> &'static str {
        match self {
            Kind::Split => "Split Long Video into Clips",
            Kind::Music => "Auto Edit Clips to Music",
            Kind::Reference => "Edit Like a Reference Video",
            Kind::Talk => "Talking Head Auto-Cut",
            Kind::Voiceover => "Voiceover Edit",
        }
    }
    fn input_label(self) -> &'static str {
        match self {
            Kind::Reference => "Reference video",
            Kind::Voiceover => "Voiceover audio",
            _ => "Video",
        }
    }
    fn needs_input(self) -> bool {
        self != Kind::Music
    }
    fn needs_clips(self) -> bool {
        matches!(self, Kind::Music | Kind::Reference | Kind::Voiceover)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Draft {
    kind: Kind,
    input: String,
    clips: Vec<String>,
    music: String,
    script: String,
    out: String,
    clip_len: f64,
    quality: &'static str,
    length: f64,
    aspect: &'static str,
    pace: &'static str,
    preset: &'static str,
    captions: bool,
    import: bool,
    error: String,
}

impl Draft {
    fn new(kind: Kind) -> Draft {
        Draft {
            kind,
            input: String::new(),
            clips: Vec::new(),
            music: String::new(),
            script: String::new(),
            out: String::new(),
            clip_len: 60.0,
            quality: "same",
            length: 30.0,
            aspect: if kind == Kind::Reference { "same" } else { "vertical" },
            pace: "medium",
            preset: if kind == Kind::Voiceover { "travel" } else { "long" },
            captions: false,
            import: kind != Kind::Split,
            error: String::new(),
        }
    }

    /// A draft from control-channel parameters (unknown choices fall back to the defaults).
    fn from_params(kind: Kind, p: &Value) -> Draft {
        let mut d = Draft::new(kind);
        let s = |k: &str| p.get(k).and_then(Value::as_str).map(str::to_string);
        d.input = s("input").unwrap_or_default();
        d.music = s("music").unwrap_or_default();
        d.script = s("script").unwrap_or_default();
        d.out = s("out").or_else(|| s("outDir")).unwrap_or_default();
        d.clips = p.get("clips").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
        if let Some(v) = p.get("clipLength").and_then(Value::as_f64).filter(|v| v.is_finite()) {
            d.clip_len = v;
        }
        if let Some(v) = p.get("length").and_then(Value::as_f64).filter(|v| v.is_finite()) {
            d.length = v;
        }
        let pick = |k: &str, options: &[&'static str], default: &'static str| -> &'static str {
            p.get(k).and_then(Value::as_str).and_then(|v| options.iter().copied().find(|o| *o == v)).unwrap_or(default)
        };
        d.quality = pick("quality", &QUALITIES.map(|q| q.0), d.quality);
        d.aspect = pick("aspect", &ASPECTS.map(|a| a.0), d.aspect);
        d.pace = pick("pace", &PACES.map(|a| a.0), d.pace);
        d.preset = pick("preset", &PRESETS.map(|a| a.0), d.preset);
        d.captions = p.get("captions").and_then(Value::as_bool).unwrap_or(false);
        d.import = p.get("import").and_then(Value::as_bool).unwrap_or(d.import);
        d
    }

    /// Fill defaults (output paths) and check that the job can run; returns the engine arguments.
    fn engine_args(&mut self) -> Result<Vec<String>, String> {
        let exists = |p: &str| !p.trim().is_empty() && Path::new(p.trim()).exists();
        if self.kind.needs_input() && !exists(&self.input) {
            return Err(format!("Choose the {} (the file was not found).", self.kind.input_label().to_lowercase()));
        }
        if self.kind.needs_clips() && self.clips.is_empty() {
            return Err("Add some clips (files or a folder).".into());
        }
        if let Some(missing) = self.clips.iter().find(|c| !Path::new(c.as_str()).exists()) {
            return Err(format!("Clip not found: {missing}"));
        }
        if self.kind == Kind::Music && !exists(&self.music) {
            return Err("Choose the music file.".into());
        }
        for (label, p) in [("music", &self.music), ("script", &self.script)] {
            if !p.trim().is_empty() && !exists(p) {
                return Err(format!("The {label} file was not found."));
            }
        }
        if self.out.trim().is_empty() {
            self.out = default_output(self);
        }
        let mut a: Vec<String> = vec![self.kind.id().into()];
        let push = |a: &mut Vec<String>, k: &str, v: &str| {
            a.push(k.into());
            a.push(v.into());
        };
        match self.kind {
            Kind::Split => {
                if !(1.0..=86_400.0).contains(&self.clip_len) {
                    return Err("Clip length must be between 1 second and 24 hours.".into());
                }
                a.push(self.input.trim().into());
                push(&mut a, "--out-dir", self.out.trim());
                push(&mut a, "--clip-len", &format!("{}", self.clip_len));
                push(&mut a, "--quality", self.quality);
            }
            Kind::Music => {
                if !(1.0..=3_600.0).contains(&self.length) {
                    return Err("Length must be between 1 second and 1 hour.".into());
                }
                a.push("--clips".into());
                a.extend(self.clips.iter().cloned());
                push(&mut a, "--music", self.music.trim());
                push(&mut a, "--out", self.out.trim());
                push(&mut a, "--length", &format!("{}", self.length));
                push(&mut a, "--aspect", self.aspect);
                push(&mut a, "--pace", self.pace);
            }
            Kind::Reference => {
                a.push(self.input.trim().into());
                a.push("--clips".into());
                a.extend(self.clips.iter().cloned());
                push(&mut a, "--out", self.out.trim());
                push(&mut a, "--aspect", self.aspect);
            }
            Kind::Talk => {
                a.push(self.input.trim().into());
                push(&mut a, "--out", self.out.trim());
                push(&mut a, "--preset", self.preset);
                if self.captions {
                    a.push("--captions".into());
                }
                if !self.music.trim().is_empty() {
                    push(&mut a, "--music", self.music.trim());
                }
            }
            Kind::Voiceover => {
                a.push(self.input.trim().into());
                a.push("--clips".into());
                a.extend(self.clips.iter().cloned());
                push(&mut a, "--out", self.out.trim());
                push(&mut a, "--preset", self.preset);
                if self.captions {
                    a.push("--captions".into());
                }
                if !self.music.trim().is_empty() {
                    push(&mut a, "--music", self.music.trim());
                }
                if !self.script.trim().is_empty() {
                    push(&mut a, "--script", self.script.trim());
                }
            }
        }
        Ok(a)
    }
}

const QUALITIES: [(&str, &str); 5] = [
    ("same", "Same as original (fast, no re-encode; cuts on the nearest keyframe)"),
    ("exact", "Same as original, exact lengths (re-encode at the original bitrate)"),
    ("high", "High (re-encode, 16 Mb/s)"),
    ("medium", "Medium (re-encode, 8 Mb/s)"),
    ("small", "Small (re-encode, 4 Mb/s)"),
];
const ASPECTS: [(&str, &str); 4] = [("same", "Same as reference"), ("horizontal", "Horizontal 16:9"), ("vertical", "Vertical 9:16"), ("square", "Square 1:1")];
const PACES: [(&str, &str); 3] = [("fast", "Fast (~1 s shots)"), ("medium", "Medium (~2 s)"), ("slow", "Slow (~3 s)")];
const PRESETS: [(&str, &str); 3] = [("long", "YouTube 16:9"), ("short", "Short / Reel 9:16"), ("travel", "Travel 16:9")];

const VIDEO_EXT: [&str; 9] = ["mp4", "mov", "m4v", "mkv", "avi", "mts", "m2ts", "webm", "mxf"];
const AUDIO_EXT: [&str; 9] = ["mp3", "m4a", "aac", "wav", "aif", "aiff", "flac", "ogg", "opus"];

fn file_stem(p: &str) -> String {
    Path::new(p.trim()).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "edit".into())
}

fn parent_dir(p: &str) -> PathBuf {
    Path::new(p.trim()).parent().map(Path::to_path_buf).unwrap_or_else(movies_dir)
}

fn movies_dir() -> PathBuf {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Movies").join("EI")).unwrap_or_else(std::env::temp_dir)
}

/// A free output path: `<dir>/<stem>.mp4`, then `<stem> 2.mp4`, … (never an existing file).
fn free_path(dir: &Path, stem: &str, ext: &str) -> String {
    for i in 1..1000 {
        let name = if i == 1 { format!("{stem}{ext}") } else { format!("{stem} {i}{ext}") };
        let p = dir.join(name);
        if !p.exists() {
            return p.to_string_lossy().to_string();
        }
    }
    dir.join(format!("{stem}-{}{ext}", std::process::id())).to_string_lossy().to_string()
}

fn default_output(d: &Draft) -> String {
    match d.kind {
        Kind::Split => free_path(&parent_dir(&d.input), &format!("{}_clips", file_stem(&d.input)), ""),
        Kind::Music => free_path(&movies_dir(), &format!("EI {} {}", file_stem(&d.music), d.aspect), ".mp4"),
        Kind::Reference => free_path(&parent_dir(&d.input), &format!("{} - EI edit", file_stem(&d.input)), ".mp4"),
        Kind::Talk => free_path(&parent_dir(&d.input), &format!("{} - EI cut", file_stem(&d.input)), ".mp4"),
        Kind::Voiceover => free_path(&parent_dir(&d.input), &format!("{} - EI edit", file_stem(&d.input)), ".mp4"),
    }
}

fn draft_id() -> egui::Id {
    egui::Id::new("ei-ai-draft")
}

fn jobs_open_id() -> egui::Id {
    egui::Id::new("ei-ai-jobs-open")
}

fn setup_open_id() -> egui::Id {
    egui::Id::new("ei-ai-setup-open")
}

// ---------------------------------------------------------------------------------------------
// Engine discovery
// ---------------------------------------------------------------------------------------------

/// The folder that holds `ei_engine.py`.
pub fn engine_dir() -> Option<PathBuf> {
    let ok = |p: &Path| p.join("ei_engine.py").is_file();
    if let Some(d) = std::env::var_os("EI_ENGINE_DIR").map(PathBuf::from).filter(|d| ok(d)) {
        return Some(d);
    }
    let exe = std::env::current_exe().ok()?;
    let mut dir = exe.parent().map(Path::to_path_buf);
    if let Some(bundle) = dir.as_ref().map(|d| d.join("..").join("Resources").join("ai-engine")).filter(|d| ok(d)) {
        return Some(bundle);
    }
    for _ in 0..7 {
        let Some(d) = dir else { break };
        let cand = d.join("ai-engine");
        if ok(&cand) {
            return Some(cand);
        }
        dir = d.parent().map(Path::to_path_buf);
    }
    None
}

/// EI's own data folder (holds the Python virtual environment).
pub fn support_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    if cfg!(target_os = "macos") { home.join("Library").join("Application Support").join("EI") } else { home.join(".local").join("share").join("ei") }
}

fn venv_python(venv: &Path) -> PathBuf {
    if cfg!(windows) { venv.join("Scripts").join("python.exe") } else { venv.join("bin").join("python") }
}

/// The Python that runs the engine.
pub fn python(engine: Option<&Path>) -> PathBuf {
    if let Some(p) = std::env::var_os("EI_PYTHON").map(PathBuf::from).filter(|p| p.is_file()) {
        return p;
    }
    let app_venv = venv_python(&support_dir().join("venv"));
    if app_venv.is_file() {
        return app_venv;
    }
    if let Some(dev) = engine.map(|e| venv_python(&e.join(".venv"))).filter(|p| p.is_file()) {
        return dev;
    }
    PathBuf::from("python3")
}

/// A system Python to create the virtual environment with (newest first).
fn base_python() -> PathBuf {
    let cands = [
        "/opt/homebrew/bin/python3.12",
        "/opt/homebrew/bin/python3.11",
        "/Library/Frameworks/Python.framework/Versions/Current/bin/python3",
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
        "/usr/bin/python3",
    ];
    cands.iter().map(PathBuf::from).find(|p| p.is_file()).unwrap_or_else(|| PathBuf::from("python3"))
}

/// `PATH` for engine processes: Homebrew first (ffmpeg), then the inherited value.
fn child_path() -> String {
    let base = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin";
    match std::env::var("PATH") {
        Ok(p) if !p.is_empty() => format!("{base}:{p}"),
        _ => base.to_string(),
    }
}

// ---------------------------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Status {
    Running,
    Done,
    Failed,
    Cancelled,
}

struct Job {
    id: u64,
    title: String,
    kind: Option<Kind>,
    status: Status,
    log: Vec<String>,
    result: Value,
    error: String,
    child: Option<Arc<Mutex<Child>>>,
    cancel: bool,
    import: bool,
    imported: bool,
    started: Instant,
    elapsed: Option<Duration>,
}

impl Job {
    fn summary(&self) -> Value {
        json!({
            "job": self.id,
            "title": self.title,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "seconds": self.elapsed.unwrap_or_else(|| self.started.elapsed()).as_secs_f64(),
            "lastLog": self.log.last(),
        })
    }
}

fn jobs() -> &'static Mutex<Vec<Job>> {
    static JOBS: OnceLock<Mutex<Vec<Job>>> = OnceLock::new();
    JOBS.get_or_init(|| Mutex::new(Vec::new()))
}

fn with_job<R>(id: u64, f: impl FnOnce(&mut Job) -> R) -> Option<R> {
    let mut g = jobs().lock().unwrap_or_else(PoisonError::into_inner);
    g.iter_mut().find(|j| j.id == id).map(f)
}

fn push_log(id: u64, line: String) {
    with_job(id, |j| {
        j.log.push(line);
        if j.log.len() > MAX_LOG {
            let extra = j.log.len() - MAX_LOG;
            j.log.drain(..extra);
        }
    });
}

/// Start a job that runs `steps` (programs with arguments) one after another in a background
/// thread. The last `EI_RESULT` line any step prints becomes the job's result.
fn start_job(title: String, kind: Option<Kind>, steps: Vec<Vec<String>>, import: bool, cwd: Option<PathBuf>) -> u64 {
    static NEXT: OnceLock<Mutex<u64>> = OnceLock::new();
    let id = {
        let mut n = NEXT.get_or_init(|| Mutex::new(0)).lock().unwrap_or_else(PoisonError::into_inner);
        *n += 1;
        *n
    };
    {
        let mut g = jobs().lock().unwrap_or_else(PoisonError::into_inner);
        while g.len() >= MAX_JOBS {
            match g.iter().position(|j| j.status != Status::Running) {
                Some(i) => {
                    g.remove(i);
                }
                None => break,
            }
        }
        g.push(Job {
            id,
            title,
            kind,
            status: Status::Running,
            log: Vec::new(),
            result: Value::Null,
            error: String::new(),
            child: None,
            cancel: false,
            import,
            imported: false,
            started: Instant::now(),
            elapsed: None,
        });
    }
    let spawned = std::thread::Builder::new().name(format!("ei-ai-job-{id}")).spawn(move || {
        let outcome = catch_unwind(AssertUnwindSafe(|| run_steps(id, &steps, cwd.as_deref())));
        let (status, error) = match outcome {
            Ok(Ok(())) => (Status::Done, String::new()),
            Ok(Err(e)) => (if with_job(id, |j| j.cancel).unwrap_or(false) { Status::Cancelled } else { Status::Failed }, e),
            Err(_) => (Status::Failed, "the job runner stopped unexpectedly".to_string()),
        };
        with_job(id, |j| {
            j.status = status;
            j.error = error;
            j.child = None;
            j.elapsed = Some(j.started.elapsed());
        });
    });
    if let Err(e) = spawned {
        with_job(id, |j| {
            j.status = Status::Failed;
            j.error = format!("could not start a thread: {e}");
        });
    }
    id
}

fn run_steps(id: u64, steps: &[Vec<String>], cwd: Option<&Path>) -> Result<(), String> {
    for step in steps {
        if with_job(id, |j| j.cancel).unwrap_or(true) {
            return Err("cancelled".into());
        }
        let Some((prog, args)) = step.split_first() else { continue };
        push_log(id, format!("$ {}", step.join(" ")));
        let mut cmd = Command::new(prog);
        cmd.args(args).env("PATH", child_path()).env("PYTHONUNBUFFERED", "1").stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        if let Some(d) = cwd {
            cmd.current_dir(d);
        }
        let mut child = cmd.spawn().map_err(|e| format!("could not run {prog}: {e}"))?;
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let child = Arc::new(Mutex::new(child));
        with_job(id, |j| j.child = Some(child.clone()));
        let err_thread = stderr.map(|s| {
            std::thread::spawn(move || {
                for line in BufReader::new(s).lines().map_while(Result::ok) {
                    let line = line.strip_prefix("[autocut] ").map(str::to_string).unwrap_or(line);
                    if !line.trim().is_empty() {
                        push_log(id, line);
                    }
                }
            })
        });
        let mut result: Option<Value> = None;
        if let Some(s) = stdout {
            for line in BufReader::new(s).lines().map_while(Result::ok) {
                match line.strip_prefix("EI_RESULT ") {
                    Some(r) => result = serde_json::from_str(r).ok(),
                    None if !line.trim().is_empty() => push_log(id, line),
                    None => {}
                }
            }
        }
        if let Some(t) = err_thread {
            let _ = t.join();
        }
        let status = child.lock().unwrap_or_else(PoisonError::into_inner).wait().map_err(|e| e.to_string())?;
        if let Some(r) = result {
            let ok = r.get("ok").and_then(Value::as_bool).unwrap_or(false);
            let err = r.get("error").and_then(Value::as_str).unwrap_or_default().to_string();
            with_job(id, |j| j.result = r);
            if !ok {
                return Err(if err.is_empty() { "the AI engine reported an error".into() } else { err });
            }
        }
        if !status.success() {
            return Err(if with_job(id, |j| j.cancel).unwrap_or(false) {
                "cancelled".into()
            } else {
                format!("{} exited with {status} (see the log)", Path::new(prog).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default())
            });
        }
    }
    Ok(())
}

fn cancel_job(id: u64) -> bool {
    with_job(id, |j| {
        j.cancel = true;
        if let Some(c) = &j.child {
            let _ = c.lock().unwrap_or_else(PoisonError::into_inner).kill();
        }
        j.status == Status::Running
    })
    .unwrap_or(false)
}

/// Run a draft: returns the job id.
fn run_draft(d: &mut Draft) -> Result<u64, String> {
    let args = d.engine_args()?;
    let engine = engine_dir().ok_or("The AI engine was not found. Reinstall EI, or set EI_ENGINE_DIR to the ai-engine folder.")?;
    let py = python(Some(&engine));
    let mut step = vec![py.to_string_lossy().to_string(), engine.join("ei_engine.py").to_string_lossy().to_string()];
    step.extend(args);
    let title = match d.kind {
        Kind::Music => format!("{}: {}", d.kind.title(), file_stem(&d.music)),
        Kind::Split | Kind::Reference | Kind::Talk | Kind::Voiceover => format!("{}: {}", d.kind.title(), file_stem(&d.input)),
    };
    Ok(start_job(title, Some(d.kind), vec![step], d.import, Some(engine)))
}

fn install_steps() -> Result<Vec<Vec<String>>, String> {
    let engine = engine_dir().ok_or("The AI engine folder was not found.")?;
    let venv = support_dir().join("venv");
    let py = venv_python(&venv);
    let s = |p: &Path| p.to_string_lossy().to_string();
    let mut steps = Vec::new();
    if !py.is_file() {
        steps.push(vec![s(&base_python()), "-m".into(), "venv".into(), s(&venv)]);
    }
    steps.push(vec![s(&py), "-m".into(), "pip".into(), "install".into(), "--upgrade".into(), "pip".into()]);
    steps.push(vec![s(&py), "-m".into(), "pip".into(), "install".into(), "-r".into(), s(&engine.join("requirements.txt"))]);
    steps.push(vec![s(&py), s(&engine.join("ei_engine.py")), "doctor".into()]);
    Ok(steps)
}

fn doctor_job() -> Result<u64, String> {
    let engine = engine_dir().ok_or("The AI engine folder was not found.")?;
    let py = python(Some(&engine));
    let step = vec![py.to_string_lossy().to_string(), engine.join("ei_engine.py").to_string_lossy().to_string(), "doctor".into()];
    Ok(start_job("AI engine check".into(), None, vec![step], false, Some(engine)))
}

// ---------------------------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------------------------

/// Route `ai.*` commands. Returns None for other ids.
pub fn route(app: &mut FilmcraftApp, ctx: &egui::Context, id: &str, params: &Value) -> Option<Result<Value, String>> {
    if !id.starts_with("ai.") {
        return None;
    }
    let has_params = params.as_object().is_some_and(|m| !m.is_empty());
    let r = match id {
        "ai.jobs" => {
            ctx.data_mut(|m| m.insert_temp(jobs_open_id(), true));
            let list: Vec<Value> = jobs().lock().unwrap_or_else(PoisonError::into_inner).iter().map(Job::summary).collect();
            Ok(json!({"jobs": list}))
        }
        "ai.cancel" => match params.get("job").and_then(Value::as_u64) {
            Some(j) => Ok(json!({"cancelled": cancel_job(j)})),
            None => Err("ai.cancel needs {job}".into()),
        },
        "ai.setup" => {
            ctx.data_mut(|m| m.insert_temp(setup_open_id(), true));
            if params.get("install").and_then(Value::as_bool).unwrap_or(false) {
                install_steps().map(|steps| json!({"job": start_job("Install AI engine".into(), None, steps, false, None)}))
            } else {
                let engine = engine_dir();
                Ok(json!({
                    "engine": engine.as_ref().map(|e| e.to_string_lossy().to_string()),
                    "python": python(engine.as_deref()).to_string_lossy().to_string(),
                    "venv": support_dir().join("venv").to_string_lossy().to_string(),
                }))
            }
        }
        _ => {
            let Some(kind) = Kind::from_command(id) else { return Some(Err(format!("unknown AI command `{id}`"))) };
            if has_params {
                let mut d = Draft::from_params(kind, params);
                run_draft(&mut d).map(|job| {
                    ctx.data_mut(|m| m.insert_temp(jobs_open_id(), true));
                    json!({"job": job, "out": d.out})
                })
            } else {
                ctx.data_mut(|m| m.insert_temp(draft_id(), Some(Draft::new(kind))));
                Ok(json!({"dialog": format!("ai.{}", kind.id())}))
            }
        }
    };
    if let Err(e) = &r {
        app.ui.status = e.clone();
    }
    Some(r)
}

// ---------------------------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------------------------

enum Pick {
    Input,
    ClipsAdd,
    ClipsFolder,
    Music,
    Script,
    Out,
}

pub fn show(app: &mut FilmcraftApp, ctx: &egui::Context) {
    show_draft(app, ctx);
    show_jobs(app, ctx);
    show_setup(app, ctx);
    import_finished(app);
    if jobs().lock().unwrap_or_else(PoisonError::into_inner).iter().any(|j| j.status == Status::Running) {
        ctx.request_repaint_after(Duration::from_millis(300));
    }
}

fn radio_row(
    ui: &mut egui::Ui,
    value: &mut &'static str,
    options: &[(&'static str, &'static str)],
    prefix: &str,
    elems: &mut Vec<(String, egui::Rect, String)>,
) {
    ui.vertical(|ui| {
        for &(k, label) in options {
            let r = ui.radio(*value == k, label);
            if r.clicked() {
                *value = k;
            }
            elems.push((format!("{prefix}.{k}"), r.rect, label.to_string()));
        }
    });
}

fn path_row(ui: &mut egui::Ui, value: &mut String, id: &str, hint: &str, elems: &mut Vec<(String, egui::Rect, String)>) -> bool {
    let mut clicked = false;
    ui.horizontal(|ui| {
        let r = ui.add(egui::TextEdit::singleline(value).hint_text(hint).desired_width(330.0));
        elems.push((id.to_string(), r.rect, value.clone()));
        let b = ui.button("Choose…");
        elems.push((format!("{id}.choose"), b.rect, "Choose…".into()));
        clicked = b.clicked();
    });
    clicked
}

fn show_draft(app: &mut FilmcraftApp, ctx: &egui::Context) {
    let Some(Some(mut d)) = ctx.data(|m| m.get_temp::<Option<Draft>>(draft_id())) else { return };
    let p = format!("ai.{}", d.kind.id());
    let mut close = false;
    let mut run = false;
    let mut pick: Option<Pick> = None;
    let mut elems: Vec<(String, egui::Rect, String)> = Vec::new();
    egui::Window::new(d.kind.title())
        .id(egui::Id::new(("ei-ai-dialog", d.kind.id())))
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.set_min_width(520.0);
            ui.label(egui::RichText::new(description(d.kind)).weak());
            ui.add_space(6.0);
            egui::Grid::new("ei-ai-grid").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                if d.kind.needs_input() {
                    ui.label(d.kind.input_label());
                    if path_row(ui, &mut d.input, &format!("{p}.input"), "a file", &mut elems) {
                        pick = Some(Pick::Input);
                    }
                    ui.end_row();
                }
                if d.kind.needs_clips() {
                    ui.label("Clips");
                    ui.vertical(|ui| {
                        let summary = match d.clips.len() {
                            0 => "No clips yet".to_string(),
                            1 => d.clips.first().map(|c| file_stem(c)).unwrap_or_default(),
                            n => format!("{n} items: {}…", d.clips.iter().take(3).map(|c| file_stem(c)).collect::<Vec<_>>().join(", ")),
                        };
                        ui.label(summary);
                        ui.horizontal(|ui| {
                            let a = ui.button("Add Files…");
                            elems.push((format!("{p}.clips.add"), a.rect, "Add Files…".into()));
                            if a.clicked() {
                                pick = Some(Pick::ClipsAdd);
                            }
                            let f = ui.button("Add Folder…");
                            elems.push((format!("{p}.clips.addFolder"), f.rect, "Add Folder…".into()));
                            if f.clicked() {
                                pick = Some(Pick::ClipsFolder);
                            }
                            let c = ui.add_enabled(!d.clips.is_empty(), egui::Button::new("Clear"));
                            elems.push((format!("{p}.clips.clear"), c.rect, "Clear".into()));
                            if c.clicked() {
                                d.clips.clear();
                            }
                        });
                    });
                    ui.end_row();
                }
                if matches!(d.kind, Kind::Music | Kind::Talk | Kind::Voiceover) {
                    ui.label(if d.kind == Kind::Music { "Music" } else { "Music (optional)" });
                    if path_row(ui, &mut d.music, &format!("{p}.music"), "song or audio file", &mut elems) {
                        pick = Some(Pick::Music);
                    }
                    ui.end_row();
                }
                if d.kind == Kind::Voiceover {
                    ui.label("Script (optional)");
                    if path_row(ui, &mut d.script, &format!("{p}.script"), "a .txt file: helps spelling", &mut elems) {
                        pick = Some(Pick::Script);
                    }
                    ui.end_row();
                }
                match d.kind {
                    Kind::Split => {
                        ui.label("Clip length (seconds)");
                        let r = ui.add(egui::DragValue::new(&mut d.clip_len).range(1.0..=86_400.0).speed(1.0));
                        elems.push((format!("{p}.clipLength"), r.rect, d.clip_len.to_string()));
                        ui.end_row();
                        ui.label("Quality");
                        radio_row(ui, &mut d.quality, &QUALITIES, &format!("{p}.quality"), &mut elems);
                        ui.end_row();
                    }
                    Kind::Music => {
                        ui.label("Length (seconds)");
                        let r = ui.add(egui::DragValue::new(&mut d.length).range(1.0..=3_600.0).speed(1.0));
                        elems.push((format!("{p}.length"), r.rect, d.length.to_string()));
                        ui.end_row();
                        ui.label("Aspect ratio");
                        radio_row(ui, &mut d.aspect, &ASPECTS[1..], &format!("{p}.aspect"), &mut elems);
                        ui.end_row();
                        ui.label("Pace");
                        radio_row(ui, &mut d.pace, &PACES, &format!("{p}.pace"), &mut elems);
                        ui.end_row();
                    }
                    Kind::Reference => {
                        ui.label("Aspect ratio");
                        radio_row(ui, &mut d.aspect, &ASPECTS, &format!("{p}.aspect"), &mut elems);
                        ui.end_row();
                    }
                    Kind::Talk | Kind::Voiceover => {
                        ui.label("Format");
                        radio_row(ui, &mut d.preset, &PRESETS, &format!("{p}.preset"), &mut elems);
                        ui.end_row();
                        ui.label("");
                        let r = ui.checkbox(&mut d.captions, "Burn in captions (speech to text, on this Mac)");
                        elems.push((format!("{p}.captions"), r.rect, d.captions.to_string()));
                        ui.end_row();
                    }
                }
                ui.label(if d.kind == Kind::Split { "Output folder" } else { "Save as" });
                if path_row(ui, &mut d.out, &format!("{p}.out"), "automatic (next to the source)", &mut elems) {
                    pick = Some(Pick::Out);
                }
                ui.end_row();
                ui.label("");
                let label = if d.kind == Kind::Split { "Import the clips into the project" } else { "Open the edit in the timeline when done" };
                let r = ui.checkbox(&mut d.import, label);
                elems.push((format!("{p}.import"), r.rect, d.import.to_string()));
                ui.end_row();
            });
            if !d.error.is_empty() {
                ui.add_space(4.0);
                ui.colored_label(egui::Color32::from_rgb(0xE5, 0x6B, 0x6B), &d.error);
            }
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let c = ui.button("Cancel");
                elems.push((format!("{p}.cancel"), c.rect, "Cancel".into()));
                if c.clicked() {
                    close = true;
                }
                let o = ui.add(egui::Button::new(egui::RichText::new("Run").color(egui::Color32::WHITE)).fill(app.tokens.accent));
                elems.push((format!("{p}.run"), o.rect, "Run".into()));
                if o.clicked() {
                    run = true;
                }
            });
        });
    for (id, r, l) in elems {
        app.auto.add(&id, r, &l);
    }
    if let Some(pk) = pick {
        let video: Vec<&str> = VIDEO_EXT.to_vec();
        match pk {
            Pick::Input => {
                let exts: Vec<&str> = if d.kind == Kind::Voiceover { AUDIO_EXT.iter().chain(VIDEO_EXT.iter()).copied().collect() } else { video.clone() };
                let filter = if d.kind == Kind::Voiceover { "Audio" } else { "Video" };
                if let Some(f) = app.hooks.pick_open_file.as_mut().and_then(|f| f(filter, &exts)) {
                    d.input = f;
                }
            }
            Pick::ClipsAdd => {
                if let Some(f) = app.hooks.pick_files.as_mut() {
                    for c in f(&video) {
                        if !d.clips.contains(&c) {
                            d.clips.push(c);
                        }
                    }
                }
            }
            Pick::ClipsFolder => {
                if let Some(dir) = app.hooks.pick_folder.as_mut().and_then(|f| f())
                    && !d.clips.contains(&dir)
                {
                    d.clips.push(dir);
                }
            }
            Pick::Music => {
                let exts: Vec<&str> = AUDIO_EXT.iter().chain(VIDEO_EXT.iter()).copied().collect();
                if let Some(f) = app.hooks.pick_open_file.as_mut().and_then(|f| f("Audio", &exts)) {
                    d.music = f;
                }
            }
            Pick::Script => {
                if let Some(f) = app.hooks.pick_open_file.as_mut().and_then(|f| f("Text", &["txt", "md"])) {
                    d.script = f;
                }
            }
            Pick::Out => {
                if d.kind == Kind::Split {
                    if let Some(dir) = app.hooks.pick_folder.as_mut().and_then(|f| f()) {
                        d.out = dir;
                    }
                } else {
                    let suggested = Path::new(&default_output(&d)).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "EI edit.mp4".into());
                    if let Some(f) = app.hooks.pick_save_as.as_mut().and_then(|f| f("MP4 video", &["mp4"], &suggested)) {
                        d.out = f;
                    }
                }
            }
        }
    }
    if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
        close = true;
    }
    if run {
        match run_draft(&mut d) {
            Ok(job) => {
                app.ui.status = format!("Started: {} (AI ▸ AI Jobs shows progress)", d.kind.title());
                ctx.data_mut(|m| m.insert_temp(jobs_open_id(), true));
                let _ = job;
                close = true;
            }
            Err(e) => d.error = e,
        }
    }
    ctx.data_mut(|m| m.insert_temp(draft_id(), if close { None } else { Some(d) }));
}

fn description(kind: Kind) -> &'static str {
    match kind {
        Kind::Split => "Cuts a long video into clips of equal length, saved in a folder. The original is not changed.",
        Kind::Music => {
            "Edits your clips to the beat of the music, at the length and shape you choose. Busy shots go on energetic beats; dark or frozen moments are skipped."
        }
        Kind::Reference => "Uses the reference video's soundtrack and cut rhythm, and fills each shot with the clip moment that looks and moves most like it.",
        Kind::Talk => "Removes the pauses from a talking video (jump cuts), with optional captions and music that ducks under your voice.",
        Kind::Voiceover => "Matches your footage to what the voiceover says, shot by shot, so the edit follows the narration.",
    }
}

fn show_jobs(app: &mut FilmcraftApp, ctx: &egui::Context) {
    let mut open = ctx.data(|m| m.get_temp::<bool>(jobs_open_id())).unwrap_or(false);
    if !open {
        return;
    }
    #[derive(Clone)]
    struct Row {
        id: u64,
        title: String,
        status: Status,
        last: String,
        error: String,
        secs: f64,
        outputs: Vec<String>,
        out_dir: Option<String>,
        importable: bool,
        log: Vec<String>,
    }
    let rows: Vec<Row> = jobs()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .rev()
        .map(|j| Row {
            id: j.id,
            title: j.title.clone(),
            status: j.status,
            // the newest progress line (a "$ command" line means the step just started)
            last: match j.log.last() {
                Some(l) if l.starts_with("$ ") => "Starting…".to_string(),
                Some(l) => l.clone(),
                None => "Starting…".to_string(),
            },
            error: j.error.clone(),
            secs: j.elapsed.unwrap_or_else(|| j.started.elapsed()).as_secs_f64(),
            outputs: j
                .result
                .get("outputs")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
                .unwrap_or_default(),
            out_dir: j.result.get("out_dir").and_then(Value::as_str).map(str::to_string),
            importable: j.kind.is_some() && j.status == Status::Done,
            log: j.log.iter().rev().take(12).rev().cloned().collect(),
        })
        .collect();
    let mut elems: Vec<(String, egui::Rect, String)> = Vec::new();
    let mut cancel: Option<u64> = None;
    let mut reveal: Option<String> = None;
    let mut open_file: Option<String> = None;
    let mut import: Option<u64> = None;
    let mut clear = false;
    egui::Window::new("AI Jobs").id(egui::Id::new("ei-ai-jobs")).open(&mut open).default_width(560.0).resizable(true).show(ctx, |ui| {
        if rows.is_empty() {
            ui.label("No AI jobs yet. Start one from the AI menu.");
        }
        egui::ScrollArea::vertical().max_height(480.0).show(ui, |ui| {
            for r in &rows {
                let (icon, color) = match r.status {
                    Status::Running => ("⏳", egui::Color32::from_rgb(0xE0, 0xB0, 0x40)),
                    Status::Done => ("✔", egui::Color32::from_rgb(0x5C, 0xC0, 0x7A)),
                    Status::Failed => ("✖", egui::Color32::from_rgb(0xE5, 0x6B, 0x6B)),
                    Status::Cancelled => ("■", egui::Color32::GRAY),
                };
                ui.horizontal(|ui| {
                    ui.colored_label(color, icon);
                    ui.strong(&r.title);
                    ui.weak(format!("{:.0}s", r.secs));
                });
                match r.status {
                    Status::Running => {
                        ui.label(&r.last);
                    }
                    Status::Failed | Status::Cancelled => {
                        ui.colored_label(color, if r.error.is_empty() { "Stopped" } else { &r.error });
                    }
                    Status::Done => {
                        let n = r.outputs.len();
                        ui.label(match n {
                            0 => "Finished".to_string(),
                            1 => format!("Saved {}", r.outputs.first().map(String::as_str).unwrap_or_default()),
                            n => format!("Saved {n} files in {}", r.out_dir.as_deref().unwrap_or_default()),
                        });
                    }
                }
                ui.horizontal(|ui| {
                    let pid = format!("ai.jobs.{}", r.id);
                    if r.status == Status::Running {
                        let b = ui.button("Cancel");
                        elems.push((format!("{pid}.cancel"), b.rect, "Cancel".into()));
                        if b.clicked() {
                            cancel = Some(r.id);
                        }
                    }
                    if let Some(first) = r.outputs.first() {
                        let o = ui.button(if r.outputs.len() == 1 { "Play" } else { "Open First" });
                        elems.push((format!("{pid}.open"), o.rect, "Open".into()));
                        if o.clicked() {
                            open_file = Some(first.clone());
                        }
                        let v = ui.button("Show in Finder");
                        elems.push((format!("{pid}.reveal"), v.rect, "Show in Finder".into()));
                        if v.clicked() {
                            reveal = Some(first.clone());
                        }
                    }
                    if r.importable {
                        let i = ui.button("Import into Project");
                        elems.push((format!("{pid}.import"), i.rect, "Import into Project".into()));
                        if i.clicked() {
                            import = Some(r.id);
                        }
                    }
                });
                egui::CollapsingHeader::new("Log").id_salt(("ei-ai-log", r.id)).show(ui, |ui| {
                    for l in &r.log {
                        ui.monospace(l);
                    }
                });
                ui.separator();
            }
        });
        let c = ui.button("Clear Finished");
        elems.push(("ai.jobs.clearFinished".into(), c.rect, "Clear Finished".into()));
        clear = c.clicked();
    });
    for (id, r, l) in elems {
        app.auto.add(&id, r, &l);
    }
    if let Some(j) = cancel {
        cancel_job(j);
    }
    for (path, reveal_it) in [(reveal, true), (open_file, false)] {
        if let Some(p) = path
            && let Some(f) = app.hooks.open_path.as_mut()
            && let Err(e) = f(&p, reveal_it)
        {
            app.ui.status = e;
        }
    }
    if let Some(j) = import {
        with_job(j, |job| {
            job.import = true;
            job.imported = false;
        });
    }
    if clear {
        jobs().lock().unwrap_or_else(PoisonError::into_inner).retain(|j| j.status == Status::Running);
    }
    ctx.data_mut(|m| m.insert_temp(jobs_open_id(), open));
}

/// Import finished jobs' results into the project (on the UI thread: the session lives here).
fn import_finished(app: &mut FilmcraftApp) {
    let todo: Vec<(u64, Kind, Value)> = jobs()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter_mut()
        .filter(|j| j.status == Status::Done && j.import && !j.imported)
        .filter_map(|j| {
            j.imported = true;
            j.kind.map(|k| (j.id, k, j.result.clone()))
        })
        .collect();
    for (id, kind, result) in todo {
        let strs = |k: &str| -> Vec<String> {
            result.get(k).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default()
        };
        let one = |k: &str| result.get(k).and_then(Value::as_str).map(str::to_string);
        let paths: Vec<String> = match kind {
            Kind::Split => strs("outputs"),
            _ => match one("otio") {
                Some(o) => std::iter::once(o).chain(one("srt")).collect(),
                None => strs("outputs"),
            },
        };
        let paths: Vec<String> = paths.into_iter().filter(|p| Path::new(p).is_file()).collect();
        if paths.is_empty() {
            continue;
        }
        match app.session.execute("file.import", json!({"paths": paths})) {
            Ok(_) => {
                app.ui.status = format!("Imported the result of AI job {id} ({} file(s))", paths.len());
            }
            Err(e) => {
                let msg = format!("Import failed: {e}");
                push_log(id, msg.clone());
                app.ui.status = msg;
            }
        }
    }
}

fn show_setup(app: &mut FilmcraftApp, ctx: &egui::Context) {
    let mut open = ctx.data(|m| m.get_temp::<bool>(setup_open_id())).unwrap_or(false);
    if !open {
        return;
    }
    let engine = engine_dir();
    let py = python(engine.as_deref());
    let venv = support_dir().join("venv");
    // The newest finished engine check, if any.
    // The newest engine check or install: its result, whether it is running, and whether one
    // ever ran (the automatic check on opening runs once, even if it failed).
    let (check, running, checked): (Option<Value>, bool, bool) = {
        let g = jobs().lock().unwrap_or_else(PoisonError::into_inner);
        let last = g.iter().rev().find(|j| j.kind.is_none() && (j.title == "AI engine check" || j.title == "Install AI engine"));
        (last.filter(|j| j.status == Status::Done).map(|j| j.result.clone()), last.is_some_and(|j| j.status == Status::Running), last.is_some())
    };
    let mut elems: Vec<(String, egui::Rect, String)> = Vec::new();
    let mut do_install = false;
    let mut do_check = false;
    let mut close = false;
    egui::Window::new("AI Engine Setup").id(egui::Id::new("ei-ai-setup")).open(&mut open).collapsible(false).resizable(false).default_width(520.0).show(ctx, |ui| {
        ui.label("EI's AI features run locally on this Mac with free models (no accounts or keys). They need ffmpeg (brew install ffmpeg) and a few Python packages, which Install sets up once (about 1.5 GB; models download on first use).");
        ui.add_space(6.0);
        egui::Grid::new("ei-ai-setup-grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
            ui.label("Engine");
            ui.monospace(engine.as_ref().map(|e| e.to_string_lossy().to_string()).unwrap_or_else(|| "not found".into()));
            ui.end_row();
            ui.label("Python");
            ui.monospace(py.to_string_lossy());
            ui.end_row();
            ui.label("Environment");
            ui.monospace(format!("{}{}", venv.to_string_lossy(), if venv.exists() { "" } else { " (not created yet)" }));
            ui.end_row();
            if let Some(c) = &check {
                let yes = |b: Option<bool>| if b.unwrap_or(false) { "✔" } else { "✖" };
                ui.label("ffmpeg");
                ui.label(format!("{} {}", yes(c.get("ffmpeg").map(|v| !v.is_null())), c.get("ffmpeg").and_then(Value::as_str).unwrap_or("not found: brew install ffmpeg")));
                ui.end_row();
                if let Some(f) = c.get("features").and_then(Value::as_object) {
                    for (k, label) in [
                        ("split", "Split long video"),
                        ("music", "Auto edit to music"),
                        ("reference", "Edit like a reference"),
                        ("reference_visual_match", "Visual matching (CLIP)"),
                        ("talk", "Talking head auto-cut"),
                        ("captions", "Captions (Whisper)"),
                        ("voiceover", "Voiceover edit"),
                    ] {
                        ui.label(label);
                        ui.label(yes(f.get(k).and_then(Value::as_bool)));
                        ui.end_row();
                    }
                }
            }
        });
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            let i = ui.add_enabled(!running && engine.is_some(), egui::Button::new(if venv.exists() { "Update Packages" } else { "Install" }));
            elems.push(("ai.setup.install".into(), i.rect, "Install".into()));
            do_install = i.clicked();
            let c = ui.add_enabled(!running && engine.is_some(), egui::Button::new("Check Again"));
            elems.push(("ai.setup.check".into(), c.rect, "Check Again".into()));
            do_check = c.clicked();
            if running {
                ui.spinner();
                ui.label("Working… (progress in AI ▸ AI Jobs)");
            }
            let x = ui.button("Close");
            elems.push(("ai.setup.close".into(), x.rect, "Close".into()));
            close = x.clicked();
        });
    });
    for (id, r, l) in elems {
        app.auto.add(&id, r, &l);
    }
    if do_install {
        match install_steps() {
            Ok(steps) => {
                start_job("Install AI engine".into(), None, steps, false, None);
                ctx.data_mut(|m| m.insert_temp(jobs_open_id(), true));
            }
            Err(e) => app.ui.status = e,
        }
    }
    if (do_check || (!checked && engine.is_some()))
        && !running
        && let Err(e) = doctor_job()
    {
        app.ui.status = e;
    }
    ctx.data_mut(|m| m.insert_temp(setup_open_id(), open && !close));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_build_engine_arguments() {
        let dir = std::env::temp_dir().join(format!("ei-ai-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let video = dir.join("long.mp4");
        std::fs::write(&video, b"x").unwrap();
        let mut d = Draft::from_params(Kind::Split, &json!({"input": video.to_string_lossy(), "clipLength": 30, "quality": "exact"}));
        let a = d.engine_args().unwrap();
        assert_eq!(a.first().map(String::as_str), Some("split"));
        assert!(a.windows(2).any(|w| w == ["--clip-len", "30"]));
        assert!(a.windows(2).any(|w| w == ["--quality", "exact"]));
        assert!(d.out.ends_with("long_clips"), "{}", d.out);
        // hostile values fall back or are rejected, never panic
        let mut bad = Draft::from_params(Kind::Split, &json!({"input": video.to_string_lossy(), "clipLength": -5, "quality": "ultra"}));
        assert_eq!(bad.quality, "same");
        assert!(bad.engine_args().is_err());
        let mut no_clips = Draft::from_params(Kind::Music, &json!({"music": video.to_string_lossy()}));
        assert!(no_clips.engine_args().is_err());
        let mut music =
            Draft::from_params(Kind::Music, &json!({"music": video.to_string_lossy(), "clips": [dir.to_string_lossy()], "aspect": "square", "length": 15}));
        let a = music.engine_args().unwrap();
        assert!(a.windows(2).any(|w| w == ["--aspect", "square"]));
        assert!(music.out.ends_with(".mp4"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_input_is_an_error() {
        let mut d = Draft::from_params(Kind::Reference, &json!({"input": "/no/such/file.mp4", "clips": ["/tmp"]}));
        assert!(d.engine_args().is_err());
        assert!(Draft::from_params(Kind::Talk, &json!(null)).engine_args().is_err());
    }

    #[test]
    fn job_runner_reports_results_and_failures() {
        let ok = start_job(
            "t".into(),
            None,
            vec![vec!["/bin/sh".into(), "-c".into(), "echo progress >&2; echo 'EI_RESULT {\"ok\": true, \"outputs\": [\"/x\"]}'".into()]],
            false,
            None,
        );
        let bad = start_job(
            "t".into(),
            None,
            vec![vec!["/bin/sh".into(), "-c".into(), "echo 'EI_RESULT {\"ok\": false, \"error\": \"boom\"}'; exit 1".into()]],
            false,
            None,
        );
        let missing = start_job("t".into(), None, vec![vec!["/no/such/program".into()]], false, None);
        let slow = start_job("t".into(), None, vec![vec!["/bin/sleep".into(), "30".into()]], false, None);
        std::thread::sleep(Duration::from_millis(300));
        assert!(cancel_job(slow));
        let deadline = Instant::now() + Duration::from_secs(10);
        let done = |id| with_job(id, |j| j.status != Status::Running).unwrap_or(false);
        while !(done(ok) && done(bad) && done(missing) && done(slow)) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(
            with_job(ok, |j| (j.status, j.result["outputs"][0].as_str().map(str::to_string), j.log.clone())).map(|(s, o, l)| (
                s,
                o,
                l.iter().any(|x| x == "progress")
            )),
            Some((Status::Done, Some("/x".into()), true))
        );
        assert_eq!(with_job(bad, |j| (j.status, j.error.clone())), Some((Status::Failed, "boom".into())));
        assert_eq!(with_job(missing, |j| j.status), Some(Status::Failed));
        assert_eq!(with_job(slow, |j| j.status), Some(Status::Cancelled));
    }

    #[test]
    fn engine_is_found_from_a_dev_build() {
        // tests run from target/…/deps: the repo's ai-engine/ is a few folders up
        assert!(engine_dir().is_some_and(|d| d.join("ei_engine.py").is_file()));
    }
}
