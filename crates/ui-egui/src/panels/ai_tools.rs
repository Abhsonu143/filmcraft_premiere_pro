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
//! - `ai.music {clips: [..], music (audio or video file), out?, length?=30, aspect?=vertical|horizontal|square,
//!   pace?, musicStart?, look?=cinematic|natural|vivid|warm|mono, effects?=full|subtle|none, order?=time|free, import?}`
//! - `ai.reference {input, clips: [..], audio? (another soundtrack), out?, aspect?=same|…, look? (also `match`),
//!   effects?, order?, import?}`
//! - `ai.talk {input, out?, preset?=long|short|travel, captions?, music?, import?}`
//! - `ai.voiceover {input, clips: [..], out?, script?, music?, captions?, preset?, import?}`
//! - `ai.clips {input, outDir?, count?=5, min?=10, target?=48, max?=60, aspect?=vertical|square|horizontal|same,
//!   content?=auto|talk|gameplay, framing?=auto|center|blur|bars, captionMode?=auto|on|off, captionStyle?=pop|clean|boxed,
//!   titleMode?=off|intro|always, pace?=fast|medium|slow, music?, llm?=default|<model>|none, hook?, import?}`
//! - `ai.audio {clips: [files or folders], outDir?, format?=mp3|m4a|wav|aiff|flac|alac|opus|ogg,
//!   audioQuality?=original|best|high|medium|small, start?, end?, mono?, normalize?, import?}`
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
    Clips,
    Audio,
    Script,
    Voice,
}

impl Kind {
    fn id(self) -> &'static str {
        match self {
            Kind::Split => "split",
            Kind::Music => "music",
            Kind::Reference => "reference",
            Kind::Talk => "talk",
            Kind::Voiceover => "voiceover",
            Kind::Clips => "clips",
            Kind::Audio => "audio",
            Kind::Script => "script",
            Kind::Voice => "voice",
        }
    }
    fn from_command(id: &str) -> Option<Kind> {
        Some(match id {
            "ai.split" => Kind::Split,
            "ai.music" => Kind::Music,
            "ai.reference" => Kind::Reference,
            "ai.talk" => Kind::Talk,
            "ai.voiceover" => Kind::Voiceover,
            "ai.clips" => Kind::Clips,
            "ai.audio" => Kind::Audio,
            "ai.script" => Kind::Script,
            "ai.voice" => Kind::Voice,
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
            Kind::Clips => "Make Short Clips from Long Video",
            Kind::Audio => "Extract Audio from Video",
            Kind::Script => "Write Voiceover Script",
            Kind::Voice => "Voice Clone",
        }
    }
    fn input_label(self) -> &'static str {
        match self {
            Kind::Reference => "Reference video",
            Kind::Voiceover => "Voiceover audio",
            Kind::Clips => "Long video",
            _ => "Video",
        }
    }
    fn needs_input(self) -> bool {
        !matches!(self, Kind::Music | Kind::Audio | Kind::Script | Kind::Voice)
    }
    fn needs_clips(self) -> bool {
        matches!(self, Kind::Music | Kind::Reference | Kind::Voiceover | Kind::Audio)
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
    look: &'static str,
    effects: &'static str,
    order: &'static str,
    music_start: String,
    count: f64,
    min_len: f64,
    target_len: f64,
    max_len: f64,
    content: &'static str,
    framing: &'static str,
    caption_mode: &'static str,
    hook: bool,
    title_mode: &'static str,
    caption_style: &'static str,
    llm: &'static str,
    format: &'static str,
    audio_quality: &'static str,
    start: String,
    end: String,
    mono: bool,
    normalize: bool,
    captions: bool,
    import: bool,
    error: String,
    // voiceover edit
    describe: &'static str,
    ambience: bool,
    // script writing
    topic: String,
    niche: String,
    tone: String,
    audience: String,
    notes: String,
    language: &'static str,
    seconds: f64,
    style: String,
    style_sources: Vec<String>,
    style_links: String,
    new_style: String,
    writer: &'static str,
    voice: String,
    takes: f64,
    // voice clone
    voice_name: String,
    lang_en: bool,
    lang_hi: bool,
    source: &'static str,
    rec: Recorder,
}

/// The sentence-by-sentence recording session of the Voice Clone dialog.
#[derive(Clone, Debug, Default, PartialEq)]
struct Recorder {
    prompts: Vec<Prompt>,
    loading: Option<u64>,
    mics: Vec<String>,
    mic: String,
    mics_job: Option<u64>,
    cursor: usize,
    recording: Option<u64>,
    takes: Vec<Option<Take>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Prompt {
    id: String,
    lang: String,
    text: String,
    emotion: String,
    direction: String,
    /// free talking (no fixed text): the engine transcribes it
    free: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Take {
    file: String,
    issues: String,
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
            aspect: match kind {
                Kind::Reference => "same",
                Kind::Voiceover => "horizontal",
                _ => "vertical",
            },
            pace: "medium",
            preset: if kind == Kind::Voiceover { "travel" } else { "long" },
            look: "cinematic",
            effects: "full",
            order: "time",
            music_start: String::new(),
            count: 5.0,
            min_len: 10.0,
            target_len: 48.0,
            max_len: 60.0,
            content: "auto",
            framing: "auto",
            caption_mode: "auto",
            hook: true,
            title_mode: "off",
            caption_style: if kind == Kind::Voiceover { "clean" } else { "pop" },
            llm: "default",
            format: "mp3",
            audio_quality: "high",
            start: String::new(),
            end: String::new(),
            mono: false,
            normalize: false,
            captions: false,
            import: !matches!(kind, Kind::Split | Kind::Audio | Kind::Script | Kind::Voice),
            error: String::new(),
            describe: "local",
            ambience: true,
            topic: String::new(),
            niche: String::new(),
            tone: String::new(),
            audience: String::new(),
            notes: String::new(),
            language: "hinglish",
            seconds: 60.0,
            style: String::new(),
            style_sources: Vec::new(),
            style_links: String::new(),
            new_style: String::new(),
            writer: "claude",
            voice: String::new(),
            takes: 2.0,
            voice_name: String::new(),
            lang_en: true,
            lang_hi: true,
            source: "record",
            rec: Recorder { mic: "0".into(), ..Recorder::default() },
        }
    }

    /// A draft from control-channel parameters (unknown choices fall back to the defaults).
    fn from_params(kind: Kind, p: &Value) -> Draft {
        let mut d = Draft::new(kind);
        let s = |k: &str| p.get(k).and_then(Value::as_str).map(str::to_string);
        d.input = s("input").unwrap_or_default();
        d.music = s("music").or_else(|| s("audio")).unwrap_or_default();
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
        let num = |k: &str, cur: f64| p.get(k).and_then(Value::as_f64).filter(|v| v.is_finite()).unwrap_or(cur);
        d.count = num("count", d.count);
        d.min_len = num("min", d.min_len);
        d.target_len = num("target", d.target_len);
        d.max_len = num("max", d.max_len);
        d.content = pick("content", &CONTENTS.map(|a| a.0), d.content);
        d.framing = pick("framing", &FRAMINGS.map(|a| a.0), d.framing);
        d.caption_mode = pick("captionMode", &CAPTION_MODES.map(|a| a.0), d.caption_mode);
        d.hook = p.get("hook").and_then(Value::as_bool).unwrap_or(d.hook);
        d.title_mode = pick("titleMode", &TITLE_MODES.map(|a| a.0), d.title_mode);
        d.caption_style = pick("captionStyle", &CAPTION_STYLES.map(|a| a.0), d.caption_style);
        d.llm = pick("llm", &LLMS.map(|a| a.0), d.llm);
        d.format = pick("format", &AUDIO_FORMATS.map(|a| a.0), d.format);
        d.audio_quality = pick("audioQuality", &AUDIO_QUALITIES.map(|a| a.0), d.audio_quality);
        let txt = |k: &str| match p.get(k) {
            Some(Value::Number(n)) => n.to_string(),
            Some(Value::String(v)) => v.clone(),
            _ => String::new(),
        };
        d.start = txt("start");
        d.end = txt("end");
        d.mono = p.get("mono").and_then(Value::as_bool).unwrap_or(false);
        d.normalize = p.get("normalize").and_then(Value::as_bool).unwrap_or(false);
        d.aspect = pick("aspect", &ASPECTS.map(|a| a.0), d.aspect);
        d.pace = pick("pace", &PACES.map(|a| a.0), d.pace);
        d.preset = pick("preset", &PRESETS.map(|a| a.0), d.preset);
        d.look = pick("look", &LOOKS.map(|a| a.0), d.look);
        d.effects = pick("effects", &EFFECTS.map(|a| a.0), d.effects);
        d.order = pick("order", &ORDERS.map(|a| a.0), d.order);
        d.music_start = match p.get("musicStart") {
            Some(Value::Number(n)) => n.to_string(),
            Some(Value::String(v)) => v.clone(),
            _ => String::new(),
        };
        d.captions = p.get("captions").and_then(Value::as_bool).unwrap_or(false);
        d.import = p.get("import").and_then(Value::as_bool).unwrap_or(d.import);
        d.describe = pick("describe", &DESCRIBERS.map(|a| a.0), d.describe);
        d.ambience = p.get("ambience").and_then(Value::as_bool).unwrap_or(d.ambience);
        for (k, f) in [
            ("topic", &mut d.topic),
            ("niche", &mut d.niche),
            ("tone", &mut d.tone),
            ("audience", &mut d.audience),
            ("notes", &mut d.notes),
            ("style", &mut d.style),
            ("voice", &mut d.voice),
            ("name", &mut d.voice_name),
            ("newStyle", &mut d.new_style),
        ] {
            if let Some(v) = p.get(k).and_then(Value::as_str) {
                *f = v.to_string();
            }
        }
        d.language = pick("language", &SCRIPT_LANGUAGES.map(|a| a.0), d.language);
        d.writer = pick("writer", &WRITERS.map(|a| a.0), d.writer);
        d.source = pick("source", &VOICE_SOURCES.map(|a| a.0), d.source);
        d.seconds = num("seconds", d.seconds);
        d.takes = num("takes", d.takes);
        if let Some(l) = p.get("languages").and_then(Value::as_array) {
            let has = |c: &str| l.iter().any(|v| v.as_str() == Some(c));
            d.lang_en = has("en");
            d.lang_hi = has("hi");
        }
        if let Some(a) = p.get("styleSources").and_then(Value::as_array) {
            d.style_sources = a.iter().filter_map(Value::as_str).map(str::to_string).collect();
        }
        if let Some(a) = p.get("takesFile").and_then(Value::as_str) {
            d.script = a.to_string();
        }
        d
    }

    fn style_args(&self, a: &mut Vec<String>) {
        for (k, v) in [("--look", self.look), ("--effects", self.effects), ("--order", self.order)] {
            a.push(k.into());
            a.push(v.into());
        }
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
                let start = self.music_start.trim();
                if !start.is_empty() && start != "auto" {
                    match start.parse::<f64>() {
                        Ok(v) if v.is_finite() && v >= 0.0 => push(&mut a, "--music-start", &format!("{v}")),
                        _ => return Err("Music start must be a number of seconds (or empty for automatic).".into()),
                    }
                }
                self.style_args(&mut a);
            }
            Kind::Reference => {
                a.push(self.input.trim().into());
                a.push("--clips".into());
                a.extend(self.clips.iter().cloned());
                push(&mut a, "--out", self.out.trim());
                push(&mut a, "--aspect", self.aspect);
                if !self.music.trim().is_empty() {
                    push(&mut a, "--audio", self.music.trim());
                }
                self.style_args(&mut a);
            }
            Kind::Clips => {
                let (n, lo, t, hi) = (self.count, self.min_len, self.target_len, self.max_len);
                if !(1.0..=50.0).contains(&n) || n.fract() != 0.0 {
                    return Err("Number of clips must be a whole number from 1 to 50.".into());
                }
                if !(3.0 <= lo && lo <= t && t <= hi && hi <= 600.0) {
                    return Err("Lengths must satisfy 3 ≤ shortest ≤ typical ≤ longest ≤ 600 seconds.".into());
                }
                a.push(self.input.trim().into());
                push(&mut a, "--out-dir", self.out.trim());
                push(&mut a, "--count", &format!("{n}"));
                push(&mut a, "--min", &format!("{lo}"));
                push(&mut a, "--target", &format!("{t}"));
                push(&mut a, "--max", &format!("{hi}"));
                push(&mut a, "--aspect", self.aspect);
                push(&mut a, "--content", self.content);
                push(&mut a, "--framing", self.framing);
                push(&mut a, "--captions", self.caption_mode);
                push(&mut a, "--caption-style", self.caption_style);
                push(&mut a, "--title", self.title_mode);
                push(&mut a, "--pace", self.pace);
                if !self.music.trim().is_empty() {
                    push(&mut a, "--music", self.music.trim());
                }
                push(&mut a, "--llm", self.llm);
                if !self.hook {
                    a.push("--no-hook".into());
                }
            }
            Kind::Audio => {
                a.extend(self.clips.iter().cloned());
                push(&mut a, "--out-dir", self.out.trim());
                push(&mut a, "--format", self.format);
                push(&mut a, "--quality", self.audio_quality);
                for (flag, v) in [("--start", self.start.trim()), ("--end", self.end.trim())] {
                    if !v.is_empty() {
                        match v.parse::<f64>() {
                            Ok(x) if x.is_finite() && x >= 0.0 => push(&mut a, flag, &format!("{x}")),
                            _ => return Err("Start and end must be numbers of seconds (or empty).".into()),
                        }
                    }
                }
                if self.mono {
                    a.push("--mono".into());
                }
                if self.normalize {
                    a.push("--normalize".into());
                }
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
                push(&mut a, "--aspect", if self.aspect == "same" { "horizontal" } else { self.aspect });
                push(&mut a, "--pace", self.pace);
                push(&mut a, "--describe", self.describe);
                push(&mut a, "--llm", self.llm);
                if self.captions {
                    a.push("--captions".into());
                    push(&mut a, "--caption-style", self.caption_style);
                }
                if !self.ambience {
                    push(&mut a, "--ambience-db", "-60");
                }
                if !self.music.trim().is_empty() {
                    push(&mut a, "--music", self.music.trim());
                }
                let script = self.script.trim();
                if !script.is_empty() {
                    push(&mut a, if script.ends_with(".narration.json") { "--narration" } else { "--script" }, script);
                }
            }
            Kind::Script => {
                if self.topic.trim().is_empty() {
                    return Err("Write what the video is about (the topic).".into());
                }
                if !(5.0..=1800.0).contains(&self.seconds) {
                    return Err("Length must be 5 seconds to 30 minutes.".into());
                }
                if !(1.0..=4.0).contains(&self.takes) {
                    return Err("Takes per line must be 1 to 4.".into());
                }
                a = vec!["script-write".into()];
                push(&mut a, "--topic", self.topic.trim());
                push(&mut a, "--out", self.out.trim());
                push(&mut a, "--seconds", &format!("{}", self.seconds));
                push(&mut a, "--language", self.language);
                push(&mut a, "--backend", self.writer);
                for (k, v) in
                    [("--niche", &self.niche), ("--tone", &self.tone), ("--audience", &self.audience), ("--notes", &self.notes), ("--style", &self.style)]
                {
                    if !v.trim().is_empty() {
                        push(&mut a, k, v.trim());
                    }
                }
                if !self.voice.trim().is_empty() {
                    push(&mut a, "--voice", self.voice.trim());
                    push(&mut a, "--takes", &format!("{}", self.takes.round()));
                }
                if !self.clips.is_empty() {
                    a.push("--footage".into());
                    a.extend(self.clips.iter().cloned());
                }
            }
            Kind::Voice => {
                let name = self.voice_name.trim();
                if name.is_empty() {
                    return Err("Give the voice a name.".into());
                }
                if !self.lang_en && !self.lang_hi {
                    return Err("Choose at least one language.".into());
                }
                a = vec!["voice-enroll".into()];
                push(&mut a, "--name", name);
                a.push("--languages".into());
                if self.lang_en {
                    a.push("en".into());
                }
                if self.lang_hi {
                    a.push("hi".into());
                }
                if self.source == "files" {
                    if self.clips.is_empty() {
                        return Err("Add recordings of your voice (audio or video files).".into());
                    }
                    a.push("--files".into());
                    a.extend(self.clips.iter().cloned());
                } else {
                    let n = self.rec.takes.iter().flatten().count();
                    if n < 6 {
                        return Err(format!("Record at least 6 sentences first ({n} so far); 20+ per language gives a much more faithful voice."));
                    }
                    let file = write_takes(&self.voice_name, &self.rec)?;
                    push(&mut a, "--takes", &file);
                    let free: Vec<String> = self
                        .rec
                        .prompts
                        .iter()
                        .zip(self.rec.takes.iter())
                        .filter(|(p, _)| p.free)
                        .filter_map(|(_, t)| t.as_ref().map(|t| t.file.clone()))
                        .collect();
                    if !free.is_empty() {
                        a.push("--files".into());
                        a.extend(free);
                    }
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
const LOOKS: [(&str, &str); 6] = [
    ("cinematic", "Cinematic (contrast, teal & orange, vignette)"),
    ("natural", "Natural (true colours, matched exposure)"),
    ("vivid", "Vivid (rich colour)"),
    ("warm", "Warm film"),
    ("mono", "Black & white"),
    ("match", "Match the reference's colours"),
];
const EFFECTS: [(&str, &str); 3] =
    [("full", "Full: speed ramps, punch-ins, flashes, transitions"), ("subtle", "Subtle: slow motion, push-ins, dissolves"), ("none", "Clean cuts only")];
const ORDERS: [(&str, &str); 2] = [("time", "Follow the trip (shooting order)"), ("free", "Best shots first")];

const CLIP_ASPECTS: [(&str, &str); 4] =
    [("vertical", "Vertical 9:16"), ("square", "Square 1:1"), ("horizontal", "Horizontal 16:9"), ("same", "Same as the video")];
const CONTENTS: [(&str, &str); 3] = [
    ("auto", "Automatic (talk or gameplay, from how much is spoken)"),
    ("talk", "Talk: podcast, vlog, interview (built from sentences)"),
    ("gameplay", "Gameplay / action (built from the most intense moments)"),
];
const FRAMINGS: [(&str, &str); 4] = [
    ("auto", "Fill the shape, following the action and faces"),
    ("center", "Fill the shape, centred crop"),
    ("blur", "Whole picture over a blurred copy"),
    ("bars", "Whole picture with black bars"),
];
const CAPTION_MODES: [(&str, &str); 3] = [("auto", "Automatic (when the clip has clear speech)"), ("on", "On"), ("off", "Off")];
const CAPTION_STYLES: [(&str, &str); 3] = [("pop", "Bold, word by word (Shorts / TikTok)"), ("clean", "Clean sentence case"), ("boxed", "Text on a dark box")];
const TITLE_MODES: [(&str, &str); 3] = [("off", "Off (the title is in the file name)"), ("intro", "First 3 seconds"), ("always", "Whole clip")];
const LLMS: [(&str, &str); 6] = [
    ("default", "Default (set in AI Engine Setup)"),
    ("qwen3-4b", "Qwen3 4B: best for 8 GB (2.3 GB)"),
    ("qwen2.5-7b", "Qwen2.5 7B: largest (4.3 GB, 16 GB Macs)"),
    ("qwen2.5-3b", "Qwen2.5 3B (1.8 GB, research licence)"),
    ("qwen2.5-1.5b", "Qwen2.5 1.5B: fastest (0.9 GB)"),
    ("none", "None: rules only"),
];
const AUDIO_FORMATS: [(&str, &str); 8] = [
    ("mp3", "MP3"),
    ("m4a", "M4A (AAC)"),
    ("wav", "WAV (uncompressed)"),
    ("aiff", "AIFF (uncompressed)"),
    ("flac", "FLAC (lossless)"),
    ("alac", "Apple Lossless (.m4a)"),
    ("opus", "Opus"),
    ("ogg", "Ogg Vorbis"),
];
const AUDIO_QUALITIES: [(&str, &str); 5] = [
    ("original", "Original (no re-encode when the format allows)"),
    ("best", "Best (320 kb/s)"),
    ("high", "High (256 kb/s)"),
    ("medium", "Medium (192 kb/s)"),
    ("small", "Small (128 kb/s)"),
];

const DESCRIBERS: [(&str, &str); 3] = [
    ("local", "Local model writes a shot idea per line (offline)"),
    ("claude", "Claude writes a shot idea per line (best for Hindi)"),
    ("none", "Match on the words only"),
];
const SCRIPT_LANGUAGES: [(&str, &str); 3] = [("hinglish", "Hinglish (Hindi + English words)"), ("hi", "Hindi"), ("en", "English")];
const WRITERS: [(&str, &str); 2] = [("claude", "Claude (through Claude Code; best quality)"), ("local", "Local model (offline)")];
const VOICE_SOURCES: [(&str, &str); 2] =
    [("record", "Record with the microphone (EI gives the sentences)"), ("files", "Recordings I already have (audio or video)")];
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
        Kind::Clips => free_path(&parent_dir(&d.input), &format!("{} - EI clips", file_stem(&d.input)), ""),
        Kind::Audio => d.clips.first().map(|c| parent_dir(c).to_string_lossy().to_string()).unwrap_or_else(|| movies_dir().to_string_lossy().to_string()),
        Kind::Script => movies_dir().join("Scripts").join(slug(if d.topic.trim().is_empty() { "script" } else { &d.topic })).to_string_lossy().to_string(),
        Kind::Voice => voices_dir().join(slug(&d.voice_name)).to_string_lossy().to_string(),
    }
}

/// A file-name-safe version of a name (as the engine's `slug`).
fn slug(name: &str) -> String {
    let mut out = String::new();
    for c in name.trim().chars() {
        if c.is_alphanumeric() || c == '-' || c == '_' {
            out.push(c);
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    let s: String = out.trim_matches('-').chars().take(60).collect();
    if s.is_empty() { "voice".into() } else { s }
}

fn voices_dir() -> PathBuf {
    settings_path().parent().map(Path::to_path_buf).unwrap_or_else(support_dir).join("voices")
}

fn styles_dir() -> PathBuf {
    settings_path().parent().map(Path::to_path_buf).unwrap_or_else(support_dir).join("styles")
}

/// Saved voices (folders with a profile) and talking styles, by name.
fn saved_voices() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(voices_dir())
        .map(|it| {
            it.filter_map(Result::ok)
                .filter(|e| e.path().join("profile.json").is_file())
                .map(|e| {
                    std::fs::read_to_string(e.path().join("profile.json"))
                        .ok()
                        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
                        .and_then(|p| p.get("name").and_then(Value::as_str).map(str::to_string))
                        .unwrap_or_else(|| e.file_name().to_string_lossy().to_string())
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn saved_styles() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(styles_dir())
        .map(|it| {
            it.filter_map(Result::ok)
                .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
                .filter_map(|e| e.path().file_stem().map(|s| s.to_string_lossy().to_string()))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// The recorded takes as the engine's `--takes` JSON (next to the takes).
fn write_takes(name: &str, rec: &Recorder) -> Result<String, String> {
    let dir = voices_dir().join(slug(name)).join("takes");
    std::fs::create_dir_all(&dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    let list: Vec<Value> = rec
        .prompts
        .iter()
        .zip(rec.takes.iter())
        .filter(|(p, _)| !p.free)
        .filter_map(|(p, t)| t.as_ref().map(|t| json!({"file": t.file, "text": p.text, "lang": p.lang, "emotion": p.emotion})))
        .collect();
    let file = dir.join("takes.json");
    std::fs::write(&file, serde_json::to_string_pretty(&list).map_err(|e| e.to_string())?).map_err(|e| format!("could not save the takes: {e}"))?;
    Ok(file.to_string_lossy().to_string())
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

/// EI's settings file (shared with the engine: `ai-engine/llm.py`).
fn settings_path() -> PathBuf {
    std::env::var_os("EI_SETTINGS").map(PathBuf::from).unwrap_or_else(|| support_dir().join("settings.json"))
}

/// The saved default language model (`llm_model`), or Qwen3 4B.
pub fn default_llm() -> &'static str {
    default_llm_at(&settings_path())
}

fn default_llm_at(path: &Path) -> &'static str {
    let saved = std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok());
    let key = saved.as_ref().and_then(|v| v.get("llm_model")).and_then(Value::as_str).unwrap_or("qwen3-4b");
    LLMS.iter().skip(1).map(|m| m.0).find(|k| *k == key).unwrap_or("qwen3-4b")
}

/// Save the default language model, keeping the other settings.
pub fn save_default_llm(key: &str) -> Result<(), String> {
    save_default_llm_at(&settings_path(), key)
}

fn save_default_llm_at(path: &Path, key: &str) -> Result<(), String> {
    if !LLMS.iter().skip(1).any(|m| m.0 == key) {
        return Err(format!("unknown language model `{key}`"));
    }
    let mut v = std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()).filter(Value::is_object).unwrap_or_else(|| json!({}));
    if let Some(o) = v.as_object_mut() {
        o.insert("llm_model".into(), json!(key));
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("could not save settings: {e}"))?;
    }
    let text = serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| format!("could not save settings: {e}"))
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
        Kind::Audio => format!("{}: {} file(s)", d.kind.title(), d.clips.len()),
        Kind::Split | Kind::Reference | Kind::Talk | Kind::Voiceover | Kind::Clips => format!("{}: {}", d.kind.title(), file_stem(&d.input)),
        Kind::Script => format!("{}: {}", d.kind.title(), d.topic.trim().chars().take(40).collect::<String>()),
        Kind::Voice => format!("{}: {}", d.kind.title(), d.voice_name.trim()),
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
            if let Some(k) = params.get("llm").and_then(Value::as_str) {
                save_default_llm(k).map(|()| json!({"llm": default_llm()}))
            } else if params.get("install").and_then(Value::as_bool).unwrap_or(false) {
                install_steps().map(|steps| json!({"job": start_job("Install AI engine".into(), None, steps, false, None)}))
            } else {
                let engine = engine_dir();
                Ok(json!({
                    "engine": engine.as_ref().map(|e| e.to_string_lossy().to_string()),
                    "python": python(engine.as_deref()).to_string_lossy().to_string(),
                    "venv": support_dir().join("venv").to_string_lossy().to_string(),
                    "llm": default_llm(),
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
    StyleSources,
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

fn combo_row(ui: &mut egui::Ui, value: &mut &'static str, options: &[(&'static str, &'static str)], id: &str, elems: &mut Vec<(String, egui::Rect, String)>) {
    let shown = options.iter().find(|o| o.0 == *value).map(|o| o.1).unwrap_or(*value);
    let r = egui::ComboBox::from_id_salt(id).selected_text(shown).width(330.0).show_ui(ui, |ui| {
        for &(k, label) in options {
            let o = ui.selectable_value(value, k, label);
            elems.push((format!("{id}.{k}"), o.rect, label.to_string()));
        }
    });
    elems.push((id.to_string(), r.response.rect, shown.to_string()));
}

fn style_rows(ui: &mut egui::Ui, d: &mut Draft, p: &str, reference: bool, elems: &mut Vec<(String, egui::Rect, String)>) {
    ui.label("Look");
    let looks: &[(&'static str, &'static str)] = if reference { &LOOKS } else { &LOOKS[..5] };
    combo_row(ui, &mut d.look, looks, &format!("{p}.look"), elems);
    ui.end_row();
    ui.label("Effects");
    combo_row(ui, &mut d.effects, &EFFECTS, &format!("{p}.effects"), elems);
    ui.end_row();
    ui.label("Story order");
    combo_row(ui, &mut d.order, &ORDERS, &format!("{p}.order"), elems);
    ui.end_row();
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
    let mut learn_style = false;
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
                if d.kind.needs_clips() || d.kind == Kind::Script || (d.kind == Kind::Voice && d.source == "files") {
                    ui.label(match d.kind {
                        Kind::Audio => "Videos",
                        Kind::Script => "Footage (optional)",
                        Kind::Voice => "Your recordings",
                        Kind::Voiceover => "Footage",
                        _ => "Clips",
                    });
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
                if matches!(d.kind, Kind::Music | Kind::Talk | Kind::Voiceover | Kind::Reference | Kind::Clips) {
                    ui.label(match d.kind {
                        Kind::Music => "Music",
                        Kind::Reference => "Other soundtrack (optional)",
                        _ => "Music (optional)",
                    });
                    let hint = if d.kind == Kind::Reference { "empty: the reference's own sound" } else { "a song, or any video: its sound is used" };
                    if path_row(ui, &mut d.music, &format!("{p}.music"), hint, &mut elems) {
                        pick = Some(Pick::Music);
                    }
                    ui.end_row();
                }
                if d.kind == Kind::Voiceover {
                    ui.label("Script (optional)");
                    if path_row(ui, &mut d.script, &format!("{p}.script"), "the .script.json / .txt it was read from", &mut elems) {
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
                        ui.label("Music start (s)");
                        let r = ui.add(egui::TextEdit::singleline(&mut d.music_start).hint_text("automatic: the best part of the song").desired_width(200.0));
                        elems.push((format!("{p}.musicStart"), r.rect, d.music_start.clone()));
                        ui.end_row();
                        style_rows(ui, &mut d, &p, false, &mut elems);
                    }
                    Kind::Reference => {
                        ui.label("Aspect ratio");
                        radio_row(ui, &mut d.aspect, &ASPECTS, &format!("{p}.aspect"), &mut elems);
                        ui.end_row();
                        style_rows(ui, &mut d, &p, true, &mut elems);
                    }
                    Kind::Clips => {
                        ui.label("Number of clips");
                        let r = ui.add(egui::DragValue::new(&mut d.count).range(1.0..=50.0).speed(0.1).max_decimals(0));
                        elems.push((format!("{p}.count"), r.rect, d.count.to_string()));
                        ui.end_row();
                        ui.label("Clip length (s)");
                        ui.horizontal(|ui| {
                            for (key, label, v) in
                                [("min", "shortest", &mut d.min_len), ("target", "mostly", &mut d.target_len), ("max", "longest", &mut d.max_len)]
                            {
                                ui.label(label);
                                let r = ui.add(egui::DragValue::new(v).range(3.0..=600.0).speed(1.0).max_decimals(0));
                                elems.push((format!("{p}.{key}"), r.rect, v.to_string()));
                            }
                        });
                        ui.end_row();
                        ui.label("Shape");
                        radio_row(ui, &mut d.aspect, &CLIP_ASPECTS, &format!("{p}.aspect"), &mut elems);
                        ui.end_row();
                        ui.label("Content");
                        combo_row(ui, &mut d.content, &CONTENTS, &format!("{p}.content"), &mut elems);
                        ui.end_row();
                        ui.label("Framing");
                        combo_row(ui, &mut d.framing, &FRAMINGS, &format!("{p}.framing"), &mut elems);
                        ui.end_row();
                        ui.label("Pace");
                        radio_row(ui, &mut d.pace, &PACES, &format!("{p}.pace"), &mut elems);
                        ui.end_row();
                        ui.label("Captions");
                        combo_row(ui, &mut d.caption_mode, &CAPTION_MODES, &format!("{p}.captions"), &mut elems);
                        ui.end_row();
                        ui.label("Caption style");
                        combo_row(ui, &mut d.caption_style, &CAPTION_STYLES, &format!("{p}.captionStyle"), &mut elems);
                        ui.end_row();
                        ui.label("Title in the video");
                        combo_row(ui, &mut d.title_mode, &TITLE_MODES, &format!("{p}.titleMode"), &mut elems);
                        ui.end_row();
                        ui.label("Language model");
                        combo_row(ui, &mut d.llm, &LLMS, &format!("{p}.llm"), &mut elems);
                        ui.end_row();
                        ui.label("");
                        ui.vertical(|ui| {
                            let r = ui.checkbox(&mut d.hook, "Open each clip with a teaser of its best moment");
                            elems.push((format!("{p}.hook"), r.rect, d.hook.to_string()));
                        });
                        ui.end_row();
                    }
                    Kind::Audio => {
                        ui.label("Format");
                        combo_row(ui, &mut d.format, &AUDIO_FORMATS, &format!("{p}.format"), &mut elems);
                        ui.end_row();
                        ui.label("Quality");
                        combo_row(ui, &mut d.audio_quality, &AUDIO_QUALITIES, &format!("{p}.quality"), &mut elems);
                        ui.end_row();
                        ui.label("Range (s)");
                        ui.horizontal(|ui| {
                            ui.label("from");
                            let r = ui.add(egui::TextEdit::singleline(&mut d.start).hint_text("start").desired_width(70.0));
                            elems.push((format!("{p}.start"), r.rect, d.start.clone()));
                            ui.label("to");
                            let r = ui.add(egui::TextEdit::singleline(&mut d.end).hint_text("end").desired_width(70.0));
                            elems.push((format!("{p}.end"), r.rect, d.end.clone()));
                        });
                        ui.end_row();
                        ui.label("");
                        ui.horizontal(|ui| {
                            let r = ui.checkbox(&mut d.mono, "Mono");
                            elems.push((format!("{p}.mono"), r.rect, d.mono.to_string()));
                            let r = ui.checkbox(&mut d.normalize, "Normalise loudness (-14 LUFS)");
                            elems.push((format!("{p}.normalize"), r.rect, d.normalize.to_string()));
                        });
                        ui.end_row();
                    }
                    Kind::Voiceover => {
                        ui.label("Shape");
                        radio_row(ui, &mut d.aspect, &ASPECTS[1..], &format!("{p}.aspect"), &mut elems);
                        ui.end_row();
                        ui.label("Pace");
                        radio_row(ui, &mut d.pace, &PACES, &format!("{p}.pace"), &mut elems);
                        ui.end_row();
                        ui.label("Shot ideas");
                        combo_row(ui, &mut d.describe, &DESCRIBERS, &format!("{p}.describe"), &mut elems);
                        ui.end_row();
                        ui.label("Captions");
                        ui.horizontal(|ui| {
                            let r = ui.checkbox(&mut d.captions, "Burn in");
                            elems.push((format!("{p}.captions"), r.rect, d.captions.to_string()));
                            combo_row(ui, &mut d.caption_style, &CAPTION_STYLES, &format!("{p}.captionStyle"), &mut elems);
                        });
                        ui.end_row();
                        ui.label("");
                        let r = ui.checkbox(&mut d.ambience, "Keep a little of each clip's own sound under the voice");
                        elems.push((format!("{p}.ambience"), r.rect, d.ambience.to_string()));
                        ui.end_row();
                    }
                    Kind::Script => {
                        ui.label("Topic");
                        let r = ui.add(egui::TextEdit::singleline(&mut d.topic).hint_text("what the video is about").desired_width(380.0));
                        elems.push((format!("{p}.topic"), r.rect, d.topic.clone()));
                        ui.end_row();
                        for (key, label, hint, v) in [
                            ("niche", "Niche", "e.g. travel vlog, tech review, finance", &mut d.niche),
                            ("tone", "Tone", "e.g. excited, calm documentary, funny", &mut d.tone),
                            ("audience", "Audience", "who is watching", &mut d.audience),
                        ] {
                            ui.label(label);
                            let r = ui.add(egui::TextEdit::singleline(v).hint_text(hint).desired_width(380.0));
                            elems.push((format!("{p}.{key}"), r.rect, v.clone()));
                            ui.end_row();
                        }
                        ui.label("Notes");
                        let r = ui.add(
                            egui::TextEdit::multiline(&mut d.notes)
                                .hint_text("facts, places, what happened, what to mention")
                                .desired_rows(3)
                                .desired_width(380.0),
                        );
                        elems.push((format!("{p}.notes"), r.rect, d.notes.clone()));
                        ui.end_row();
                        ui.label("Length (seconds)");
                        let r = ui.add(egui::DragValue::new(&mut d.seconds).range(5.0..=1800.0).speed(1.0).max_decimals(0));
                        elems.push((format!("{p}.seconds"), r.rect, d.seconds.to_string()));
                        ui.end_row();
                        ui.label("Language");
                        radio_row(ui, &mut d.language, &SCRIPT_LANGUAGES, &format!("{p}.language"), &mut elems);
                        ui.end_row();
                        ui.label("Talking style");
                        let styles = saved_styles();
                        let shown = if d.style.is_empty() { "None (a natural creator style)".to_string() } else { d.style.clone() };
                        let r = egui::ComboBox::from_id_salt(format!("{p}.style")).selected_text(shown.clone()).width(330.0).show_ui(ui, |ui| {
                            let o = ui.selectable_value(&mut d.style, String::new(), "None (a natural creator style)");
                            elems.push((format!("{p}.style.none"), o.rect, "None".into()));
                            for st in &styles {
                                let o = ui.selectable_value(&mut d.style, st.clone(), st);
                                elems.push((format!("{p}.style.{st}"), o.rect, st.clone()));
                            }
                        });
                        elems.push((format!("{p}.style"), r.response.rect, shown));
                        ui.end_row();
                        ui.label("Learn a style");
                        ui.vertical(|ui| {
                            ui.label(
                                egui::RichText::new(
                                    "From transcripts (.txt, .srt, .vtt) and/or YouTube links (one per line): how you or a creator you like talks.",
                                )
                                .weak(),
                            );
                            ui.horizontal(|ui| {
                                let b = ui.button("Add Transcripts…");
                                elems.push((format!("{p}.styleSources.add"), b.rect, "Add Transcripts…".into()));
                                if b.clicked() {
                                    pick = Some(Pick::StyleSources);
                                }
                                ui.label(format!("{} file(s)", d.style_sources.len()));
                            });
                            let r = ui.add(
                                egui::TextEdit::multiline(&mut d.style_links)
                                    .hint_text("https://www.youtube.com/watch?v=…")
                                    .desired_rows(2)
                                    .desired_width(380.0),
                            );
                            elems.push((format!("{p}.styleLinks"), r.rect, d.style_links.clone()));
                            ui.horizontal(|ui| {
                                let r = ui.add(egui::TextEdit::singleline(&mut d.new_style).hint_text("style name").desired_width(160.0));
                                elems.push((format!("{p}.newStyle"), r.rect, d.new_style.clone()));
                                let b = ui.button("Learn Style");
                                elems.push((format!("{p}.learnStyle"), b.rect, "Learn Style".into()));
                                if b.clicked() {
                                    learn_style = true;
                                }
                            });
                        });
                        ui.end_row();
                        ui.label("Written by");
                        combo_row(ui, &mut d.writer, &WRITERS, &format!("{p}.writer"), &mut elems);
                        ui.end_row();
                        ui.label("Speak it with");
                        let voices = saved_voices();
                        let shown = if d.voice.is_empty() { "Don't speak (script only)".to_string() } else { d.voice.clone() };
                        let r = egui::ComboBox::from_id_salt(format!("{p}.voice")).selected_text(shown.clone()).width(330.0).show_ui(ui, |ui| {
                            let o = ui.selectable_value(&mut d.voice, String::new(), "Don't speak (script only)");
                            elems.push((format!("{p}.voice.none"), o.rect, "none".into()));
                            for v in &voices {
                                let o = ui.selectable_value(&mut d.voice, v.clone(), v);
                                elems.push((format!("{p}.voice.{v}"), o.rect, v.clone()));
                            }
                        });
                        elems.push((format!("{p}.voice"), r.response.rect, shown));
                        ui.end_row();
                        if !d.voice.is_empty() {
                            ui.label("Takes per line");
                            let r = ui.add(egui::DragValue::new(&mut d.takes).range(1.0..=4.0).speed(0.05).max_decimals(0));
                            elems.push((format!("{p}.takes"), r.rect, d.takes.to_string()));
                            ui.end_row();
                        }
                    }
                    Kind::Voice => {
                        ui.label("Voice name");
                        let r = ui.add(egui::TextEdit::singleline(&mut d.voice_name).hint_text("e.g. my voice").desired_width(240.0));
                        elems.push((format!("{p}.name"), r.rect, d.voice_name.clone()));
                        ui.end_row();
                        ui.label("Languages");
                        ui.horizontal(|ui| {
                            let r = ui.checkbox(&mut d.lang_hi, "Hindi (and Hinglish)");
                            elems.push((format!("{p}.lang.hi"), r.rect, d.lang_hi.to_string()));
                            let r = ui.checkbox(&mut d.lang_en, "English");
                            elems.push((format!("{p}.lang.en"), r.rect, d.lang_en.to_string()));
                        });
                        ui.end_row();
                        ui.label("Source");
                        radio_row(ui, &mut d.source, &VOICE_SOURCES, &format!("{p}.source"), &mut elems);
                        ui.end_row();
                    }
                    Kind::Talk => {
                        ui.label("Format");
                        radio_row(ui, &mut d.preset, &PRESETS, &format!("{p}.preset"), &mut elems);
                        ui.end_row();
                        ui.label("");
                        let r = ui.checkbox(&mut d.captions, "Burn in captions (speech to text, on this Mac)");
                        elems.push((format!("{p}.captions"), r.rect, d.captions.to_string()));
                        ui.end_row();
                    }
                }
                if d.kind != Kind::Voice {
                    let folder_out = matches!(d.kind, Kind::Split | Kind::Clips | Kind::Audio);
                    ui.label(if folder_out { "Output folder" } else { "Save as" });
                    let hint = if d.kind == Kind::Script { "automatic (Movies/EI/Scripts)" } else { "automatic (next to the source)" };
                    if path_row(ui, &mut d.out, &format!("{p}.out"), hint, &mut elems) {
                        pick = Some(Pick::Out);
                    }
                    ui.end_row();
                }
                if !matches!(d.kind, Kind::Script | Kind::Voice) {
                    ui.label("");
                    let label = match d.kind {
                        Kind::Split | Kind::Clips => "Import the clips into the project",
                        Kind::Audio => "Import the audio into the project",
                        _ => "Open the edit in the timeline when done",
                    };
                    let r = ui.checkbox(&mut d.import, label);
                    elems.push((format!("{p}.import"), r.rect, d.import.to_string()));
                    ui.end_row();
                }
            });
            if d.kind == Kind::Voice && d.source == "record" {
                ui.separator();
                recorder_ui(ui, &mut d, &p, &mut elems);
            }
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
                let run_label = match d.kind {
                    Kind::Voice => "Create Voice",
                    Kind::Script if !d.voice.is_empty() => "Write and Speak",
                    Kind::Script => "Write Script",
                    _ => "Run",
                };
                let o = ui.add(egui::Button::new(egui::RichText::new(run_label).color(egui::Color32::WHITE)).fill(app.tokens.accent));
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
            Pick::StyleSources => {
                if let Some(f) = app.hooks.pick_files.as_mut() {
                    for c in f(&["txt", "srt", "vtt"]) {
                        if !d.style_sources.contains(&c) {
                            d.style_sources.push(c);
                        }
                    }
                }
            }
            Pick::ClipsAdd => {
                let media: Vec<&str> = if d.kind == Kind::Voice { AUDIO_EXT.iter().chain(VIDEO_EXT.iter()).copied().collect() } else { video.clone() };
                if let Some(f) = app.hooks.pick_files.as_mut() {
                    for c in f(&media) {
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
                if let Some(f) = app.hooks.pick_open_file.as_mut().and_then(|f| f("Audio or video", &exts)) {
                    d.music = f;
                }
            }
            Pick::Script => {
                if let Some(f) = app.hooks.pick_open_file.as_mut().and_then(|f| f("Script", &["json", "txt", "md"])) {
                    d.script = f;
                }
            }
            Pick::Out => {
                if matches!(d.kind, Kind::Split | Kind::Clips | Kind::Audio) {
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
    if learn_style {
        match learn_style_job(&d) {
            Ok(_) => {
                d.error.clear();
                app.ui.status = "Learning the talking style (AI ▸ AI Jobs shows progress)".into();
            }
            Err(e) => d.error = e,
        }
    }
    poll_recorder(&mut d);
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

/// The engine command line for `args` (python, ei_engine.py, args…) and the engine folder.
fn engine_step(args: Vec<String>) -> Result<(Vec<String>, PathBuf), String> {
    let engine = engine_dir().ok_or("The AI engine was not found. Reinstall EI, or set EI_ENGINE_DIR to the ai-engine folder.")?;
    let mut step = vec![python(Some(&engine)).to_string_lossy().to_string(), engine.join("ei_engine.py").to_string_lossy().to_string()];
    step.extend(args);
    Ok((step, engine))
}

fn job_state(id: u64) -> Option<(Status, Value, String)> {
    with_job(id, |j| (j.status, j.result.clone(), j.error.clone()))
}

fn learn_style_job(d: &Draft) -> Result<u64, String> {
    let name = d.new_style.trim();
    if name.is_empty() {
        return Err("Give the style a name.".into());
    }
    let mut sources: Vec<String> = d.style_sources.clone();
    sources.extend(d.style_links.lines().map(str::trim).filter(|l| l.starts_with("http")).map(str::to_string));
    if sources.is_empty() {
        return Err("Add transcript files or YouTube links to learn from.".into());
    }
    let mut args = vec!["script-style".to_string(), "--name".into(), name.to_string()];
    args.extend(sources);
    let (step, engine) = engine_step(args)?;
    Ok(start_job(format!("Learn talking style: {name}"), None, vec![step], false, Some(engine)))
}

/// Sentence-by-sentence recording for a voice clone.
fn recorder_ui(ui: &mut egui::Ui, d: &mut Draft, p: &str, elems: &mut Vec<(String, egui::Rect, String)>) {
    let mut langs = Vec::new();
    if d.lang_hi {
        langs.push("hi".to_string());
    }
    if d.lang_en {
        langs.push("en".to_string());
    }
    if d.rec.prompts.is_empty() && d.rec.loading.is_none() && !langs.is_empty() {
        let mut args = vec!["voice-prompts".to_string(), "--languages".into()];
        args.extend(langs.iter().cloned());
        match engine_step(args) {
            Ok((step, engine)) => d.rec.loading = Some(start_job("Voice clone: sentences".into(), None, vec![step], false, Some(engine))),
            Err(e) => d.error = e,
        }
    }
    if d.rec.mics.is_empty()
        && d.rec.mics_job.is_none()
        && let Ok((step, engine)) = engine_step(vec!["voice-mics".into()])
    {
        d.rec.mics_job = Some(start_job("Voice clone: microphones".into(), None, vec![step], false, Some(engine)));
    }
    if d.rec.prompts.is_empty() {
        ui.label(if langs.is_empty() { "Choose a language." } else { "Loading the sentences…" });
        return;
    }
    let n = d.rec.prompts.len();
    d.rec.cursor = d.rec.cursor.min(n.saturating_sub(1));
    let done = d.rec.takes.iter().flatten().count();
    ui.horizontal(|ui| {
        ui.strong(format!("Sentence {} of {n}", d.rec.cursor + 1));
        ui.weak(format!("· {done} recorded"));
        if !d.rec.mics.is_empty() {
            ui.add_space(12.0);
            ui.label("Microphone");
            let shown = d.rec.mics.iter().find(|m| m.split(':').next() == Some(d.rec.mic.as_str())).cloned().unwrap_or_else(|| d.rec.mic.clone());
            let r = egui::ComboBox::from_id_salt(format!("{p}.mic")).selected_text(shown.clone()).width(200.0).show_ui(ui, |ui| {
                for m in d.rec.mics.clone() {
                    let idx = m.split(':').next().unwrap_or("0").trim().to_string();
                    let o = ui.selectable_value(&mut d.rec.mic, idx.clone(), &m);
                    elems.push((format!("{p}.mic.{idx}"), o.rect, m.clone()));
                }
            });
            elems.push((format!("{p}.mic"), r.response.rect, shown));
        }
    });
    let Some(pr) = d.rec.prompts.get(d.rec.cursor).cloned() else { return };
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(if pr.lang == "hi" { "हिन्दी" } else { "English" }).strong());
        ui.label(egui::RichText::new(format!("· {}", pr.emotion)).weak());
    });
    if pr.direction != "free" {
        ui.label(egui::RichText::new(&pr.direction).italics().weak());
    }
    let t = ui.add(egui::Label::new(egui::RichText::new(&pr.text).size(20.0)).wrap());
    elems.push((format!("{p}.prompt"), t.rect, pr.text.clone()));
    ui.add_space(4.0);
    let take = d.rec.takes.get(d.rec.cursor).cloned().flatten();
    let recording = d.rec.recording.is_some();
    ui.horizontal(|ui| {
        let b = ui.add_enabled(d.rec.cursor > 0 && !recording, egui::Button::new("‹ Back"));
        elems.push((format!("{p}.back"), b.rect, "Back".into()));
        if b.clicked() {
            d.rec.cursor -= 1;
        }
        let label = if recording {
            "● Recording… (stops when you stop talking)"
        } else if take.is_some() {
            "● Record Again"
        } else {
            "● Record"
        };
        let r = ui.add_enabled(!recording, egui::Button::new(egui::RichText::new(label).color(egui::Color32::from_rgb(0xE5, 0x6B, 0x6B))));
        elems.push((format!("{p}.record"), r.rect, label.into()));
        if r.clicked() {
            if d.voice_name.trim().is_empty() {
                d.error = "Give the voice a name first.".into();
            } else {
                let file = voices_dir().join(slug(&d.voice_name)).join("takes").join(format!("{}.wav", pr.id));
                let max = if pr.free { "75" } else { "30" };
                let args = vec![
                    "voice-record".into(),
                    "--out".into(),
                    file.to_string_lossy().to_string(),
                    "--mic".into(),
                    d.rec.mic.clone(),
                    "--max".into(),
                    max.into(),
                ];
                match engine_step(args) {
                    Ok((step, engine)) => {
                        d.error.clear();
                        d.rec.recording = Some(start_job(format!("Recording {}", pr.id), None, vec![step], false, Some(engine)));
                    }
                    Err(e) => d.error = e,
                }
            }
        }
        if recording {
            let s = ui.button("Stop");
            elems.push((format!("{p}.stop"), s.rect, "Stop".into()));
            if s.clicked()
                && let Some(j) = d.rec.recording.take()
            {
                cancel_job(j);
            }
        }
        let pl = ui.add_enabled(take.is_some() && !recording, egui::Button::new("▶ Play"));
        elems.push((format!("{p}.play"), pl.rect, "Play".into()));
        if pl.clicked()
            && let Some(t) = &take
        {
            let _ = Command::new("afplay").arg(&t.file).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn();
        }
        let nx = ui.add_enabled(d.rec.cursor + 1 < n && !recording, egui::Button::new(if take.is_some() { "Next ›" } else { "Skip ›" }));
        elems.push((format!("{p}.next"), nx.rect, "Next".into()));
        if nx.clicked() {
            d.rec.cursor += 1;
        }
    });
    match &take {
        Some(t) if !t.issues.is_empty() => {
            ui.colored_label(egui::Color32::from_rgb(0xE0, 0xB0, 0x40), format!("Recorded, but: {}", t.issues));
        }
        Some(_) => {
            ui.colored_label(egui::Color32::from_rgb(0x5C, 0xC0, 0x7A), "✔ Recorded");
        }
        None => {
            ui.label(egui::RichText::new("A quiet room, 15–20 cm from the microphone, and say it the way you talk in your videos.").weak());
        }
    }
}

/// Pick up finished recorder jobs (sentences loaded, microphones listed, a take recorded).
fn poll_recorder(d: &mut Draft) {
    if let Some(j) = d.rec.loading
        && let Some((st, res, err)) = job_state(j)
        && st != Status::Running
    {
        d.rec.loading = None;
        if st == Status::Done {
            d.rec.prompts = res
                .get("prompts")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .map(|v| {
                            let s = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
                            Prompt {
                                id: s("id"),
                                lang: s("lang"),
                                text: s("text"),
                                emotion: s("emotion"),
                                direction: s("direction"),
                                free: v.get("free").and_then(Value::as_bool).unwrap_or(false),
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            d.rec.takes = vec![None; d.rec.prompts.len()];
            // takes already on disk from an earlier session count as recorded
            if !d.voice_name.trim().is_empty() {
                let dir = voices_dir().join(slug(&d.voice_name)).join("takes");
                for (k, p) in d.rec.prompts.iter().enumerate() {
                    let f = dir.join(format!("{}.wav", p.id));
                    if f.is_file()
                        && let Some(t) = d.rec.takes.get_mut(k)
                    {
                        *t = Some(Take { file: f.to_string_lossy().to_string(), issues: String::new() });
                    }
                }
            }
        } else {
            d.error = format!("Could not load the sentences: {err}");
        }
    }
    if let Some(j) = d.rec.mics_job
        && let Some((st, res, _)) = job_state(j)
        && st != Status::Running
    {
        d.rec.mics = res.get("mics").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
        if d.rec.mics.is_empty() {
            d.rec.mics = vec!["0: Default microphone".into()];
        }
    }
    if let Some(j) = d.rec.recording
        && let Some((st, res, err)) = job_state(j)
        && st != Status::Running
    {
        d.rec.recording = None;
        match st {
            Status::Done => {
                let file = res.get("file").and_then(Value::as_str).unwrap_or_default().to_string();
                let issues =
                    res.get("issues").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join("; ")).unwrap_or_default();
                if let Some(t) = d.rec.takes.get_mut(d.rec.cursor) {
                    *t = Some(Take { file, issues: issues.clone() });
                }
                if issues.is_empty() && d.rec.cursor + 1 < d.rec.prompts.len() {
                    d.rec.cursor += 1; // a good take: straight to the next sentence
                }
            }
            Status::Cancelled => {}
            _ => d.error = err,
        }
    }
}

fn description(kind: Kind) -> &'static str {
    match kind {
        Kind::Split => "Cuts a long video into clips of equal length, saved in a folder. The original is not changed.",
        Kind::Music => {
            "Edits your clips like an editor would: long slow-motion shots in calm parts, quick cuts and punch-ins on the drop, the best moments of each clip, the trip in order, one colour grade. Any video can be the music source."
        }
        Kind::Reference => {
            "Copies the reference's style, not its content: its cut timing, transitions, flashes, zoom punches and (optionally) colour, filled with your best matching moments. Keeps its soundtrack unless you choose another (any audio or video file)."
        }
        Kind::Talk => "Removes the pauses from a talking video (jump cuts), with optional captions and music that ducks under your voice.",
        Kind::Voiceover => {
            "Cuts your footage to a voiceover like an editor: every line gets the shot that shows what is being said (works for Hindi and English), the best-looking moment of it, camera energy that fits the delivery, no repeats, varied shot sizes, cuts just before the words. Use a cloned-voice narration (its .narration.json is found automatically) or any recording."
        }
        Kind::Script => {
            "Claude (or the local model) writes a voiceover script in your talking style, line by line with emotion, pace and a shot idea for each line, ready for your cloned voice. Optionally speaks it right away."
        }
        Kind::Voice => {
            "Clones your voice on this Mac (Chatterbox Multilingual, Hindi + English). Read the sentences EI shows (they cover every sound and the deliveries a video needs), or use recordings you already have. Use only your own voice, or a voice you have permission to use."
        }
        Kind::Clips => {
            "Edits each clip from a whole scene of a long video: only its best beats, cut on the action, a teaser first, slow motion on the climax, punch-ins on the big hits. Reframed to truly fill the shape (following the action and faces). Talk clips are built from sentences and topics. Optional captions, title card and music (cuts move onto its beats)."
        }
        Kind::Audio => "Saves the sound of videos as audio files in the format you choose. The videos are not changed.",
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
            Kind::Split | Kind::Clips | Kind::Audio => strs("outputs"),
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
    let mut llm_error: Option<String> = None;
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
            ui.label("Language model");
            let mut cur = default_llm();
            let before = cur;
            combo_row(ui, &mut cur, &LLMS[1..], "ai.setup.llm", &mut elems);
            if cur != before
                && let Err(e) = save_default_llm(cur)
            {
                llm_error = Some(e);
            }
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
    if let Some(e) = llm_error {
        app.ui.status = e;
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
        assert!(a.windows(2).any(|w| w == ["--look", "cinematic"]) && a.windows(2).any(|w| w == ["--order", "time"]));
        assert!(music.out.ends_with(".mp4"));
        let mut styled = Draft::from_params(
            Kind::Music,
            &json!({"music": video.to_string_lossy(), "clips": [dir.to_string_lossy()], "look": "warm", "effects": "none", "musicStart": 12.5}),
        );
        let a = styled.engine_args().unwrap();
        assert!(
            a.windows(2).any(|w| w == ["--look", "warm"])
                && a.windows(2).any(|w| w == ["--effects", "none"])
                && a.windows(2).any(|w| w == ["--music-start", "12.5"])
        );
        let mut bad_start = Draft::from_params(Kind::Music, &json!({"music": video.to_string_lossy(), "clips": [dir.to_string_lossy()], "musicStart": "soon"}));
        assert!(bad_start.engine_args().is_err());
        let mut other = Draft::from_params(
            Kind::Reference,
            &json!({"input": video.to_string_lossy(), "clips": [dir.to_string_lossy()], "audio": video.to_string_lossy(), "look": "match"}),
        );
        let a = other.engine_args().unwrap();
        assert!(a.windows(2).any(|w| w[0] == "--audio") && a.windows(2).any(|w| w == ["--look", "match"]));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn clips_and_audio_arguments() {
        let dir = std::env::temp_dir().join(format!("ei-ai-test2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let video = dir.join("long.mov");
        std::fs::write(&video, b"x").unwrap();
        let mut c = Draft::from_params(
            Kind::Clips,
            &json!({"input": video.to_string_lossy(), "count": 10, "llm": "qwen2.5-7b", "framing": "blur", "titleMode": "intro", "captionStyle": "clean", "music": video.to_string_lossy()}),
        );
        let a = c.engine_args().unwrap();
        assert_eq!(a.first().map(String::as_str), Some("clips"));
        for pair in [
            ["--count", "10"],
            ["--min", "10"],
            ["--target", "48"],
            ["--max", "60"],
            ["--llm", "qwen2.5-7b"],
            ["--framing", "blur"],
            ["--title", "intro"],
            ["--caption-style", "clean"],
            ["--pace", "medium"],
        ] {
            assert!(a.windows(2).any(|w| w == pair), "{pair:?} in {a:?}");
        }
        assert!(a.windows(2).any(|w| w[0] == "--music" && w[1].ends_with("long.mov")), "{a:?}");
        assert!(c.out.ends_with("long - EI clips"), "{}", c.out);
        for bad in [json!({"count": 0}), json!({"count": 2.5}), json!({"min": 50, "target": 40}), json!({"max": 9000})] {
            let mut p = bad.clone();
            p["input"] = json!(video.to_string_lossy());
            assert!(Draft::from_params(Kind::Clips, &p).engine_args().is_err(), "{bad}");
        }
        let mut au =
            Draft::from_params(Kind::Audio, &json!({"clips": [video.to_string_lossy()], "format": "flac", "start": 5, "end": "20.5", "normalize": true}));
        let a = au.engine_args().unwrap();
        assert!(a.windows(2).any(|w| w == ["--format", "flac"]) && a.windows(2).any(|w| w == ["--end", "20.5"]) && a.contains(&"--normalize".to_string()));
        assert!(Draft::from_params(Kind::Audio, &json!({})).engine_args().is_err());
        assert!(Draft::from_params(Kind::Audio, &json!({"clips": [video.to_string_lossy()], "start": "soon"})).engine_args().is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn voiceover_script_and_voice_arguments() {
        let dir = std::env::temp_dir().join(format!("ei-ai-test3-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("narr.wav");
        std::fs::write(&wav, b"x").unwrap();
        let nar = dir.join("narr.narration.json");
        std::fs::write(&nar, b"{}").unwrap();
        let p = json!({"input": wav.to_string_lossy(), "clips": [dir.to_string_lossy()], "aspect": "vertical", "pace": "fast", "captions": true,
            "describe": "claude", "ambience": false, "script": nar.to_string_lossy()});
        let a = Draft::from_params(Kind::Voiceover, &p).engine_args().unwrap();
        for pair in [["--aspect", "vertical"], ["--pace", "fast"], ["--describe", "claude"], ["--ambience-db", "-60"], ["--caption-style", "clean"]] {
            assert!(a.windows(2).any(|w| w == pair), "{pair:?} in {a:?}");
        }
        assert!(a.windows(2).any(|w| w[0] == "--narration"), "{a:?}");
        let mut s = Draft::from_params(Kind::Script, &json!({"topic": "Spiti road trip", "language": "hi", "seconds": 90, "voice": "me", "writer": "local"}));
        let a = s.engine_args().unwrap();
        assert_eq!(a.first().map(String::as_str), Some("script-write"));
        for pair in [["--topic", "Spiti road trip"], ["--language", "hi"], ["--seconds", "90"], ["--voice", "me"], ["--backend", "local"], ["--takes", "2"]] {
            assert!(a.windows(2).any(|w| w == pair), "{pair:?} in {a:?}");
        }
        assert!(s.out.contains("Spiti-road-trip"), "{}", s.out);
        assert!(Draft::from_params(Kind::Script, &json!({"topic": " "})).engine_args().is_err());
        let a = Draft::from_params(Kind::Voice, &json!({"name": "My Voice", "source": "files", "languages": ["hi"], "clips": [wav.to_string_lossy()]}))
            .engine_args()
            .unwrap();
        assert_eq!(a.first().map(String::as_str), Some("voice-enroll"));
        assert!(a.windows(2).any(|w| w == ["--name", "My Voice"]) && a.windows(2).any(|w| w == ["--languages", "hi"]) && a.contains(&"--files".to_string()));
        assert!(!a.contains(&"en".to_string()));
        // recording mode needs takes first
        assert!(Draft::from_params(Kind::Voice, &json!({"name": "v", "source": "record"})).engine_args().is_err());
        assert_eq!(slug("  My Voice / Hindi "), "My-Voice-Hindi");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_model_is_saved_and_validated() {
        let f = std::env::temp_dir().join(format!("ei-settings-{}.json", std::process::id()));
        assert_eq!(default_llm_at(&f), "qwen3-4b");
        std::fs::write(&f, r#"{"other": 1}"#).unwrap();
        save_default_llm_at(&f, "qwen2.5-1.5b").unwrap();
        assert_eq!(default_llm_at(&f), "qwen2.5-1.5b");
        assert!(std::fs::read_to_string(&f).unwrap().contains("other"), "other settings kept");
        assert!(save_default_llm_at(&f, "gpt-9").is_err());
        std::fs::write(&f, "not json").unwrap();
        assert_eq!(default_llm_at(&f), "qwen3-4b");
        let _ = std::fs::remove_file(&f);
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
