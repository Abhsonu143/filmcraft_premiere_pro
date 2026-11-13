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
//! - `ai.clips {input, outDir?, clipType?=edited|sections, count?=5, min?=10, target?=48, max?=60,
//!   aspect?=vertical|square|horizontal|same, content?=auto|talk|gameplay, framing?=auto|center|blur|bars, captionMode?=auto|on|off, captionStyle?=pop|clean|boxed,
//!   titleMode?=off|intro|always, pace?=fast|medium|slow, music?, llm?=default|<model>|none, hook?, import?}`
//! - `ai.speak {script (file) or notes (text), voice, takes?, out?}`: a script spoken (audio only).
//! - sound design (speak, voiceover, topic, clips, music): `soundDesign?, musicSource?=internet|folder|ai|none,
//!   musicFolder?, licence?=monetized|personal, soundEffects?, musicLevel?` (dB).
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
    Topic,
    Tv,
    Speak,
    Hooks,
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
            Kind::Topic => "topic",
            Kind::Tv => "tv",
            Kind::Speak => "speak",
            Kind::Hooks => "hooks",
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
            "ai.topic" => Kind::Topic,
            "ai.tv" => Kind::Tv,
            "ai.speak" => Kind::Speak,
            "ai.hooks" => Kind::Hooks,
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
            Kind::Topic => "Video from Topic",
            Kind::Tv => "TV Light Sync",
            Kind::Speak => "Speak Script",
            Kind::Hooks => "Hook Library",
        }
    }
    fn input_label(self) -> &'static str {
        match self {
            Kind::Reference => "Reference video",
            Kind::Voiceover => "Voiceover audio",
            Kind::Clips => "Long video",
            Kind::Tv => "Your clip",
            _ => "Video",
        }
    }
    fn needs_input(self) -> bool {
        !matches!(self, Kind::Music | Kind::Audio | Kind::Script | Kind::Voice | Kind::Topic | Kind::Tv | Kind::Speak | Kind::Hooks)
    }
    /// Tools that render a video in a chosen shape (and so offer 1080p or 4K).
    fn has_resolution(self) -> bool {
        matches!(self, Kind::Music | Kind::Reference | Kind::Talk | Kind::Voiceover | Kind::Clips | Kind::Topic | Kind::Tv)
    }
    /// Tools that can add the optional emotional sound design.
    fn has_sound_design(self) -> bool {
        matches!(self, Kind::Speak | Kind::Voiceover | Kind::Topic | Kind::Clips | Kind::Music)
    }
    fn needs_clips(self) -> bool {
        matches!(self, Kind::Music | Kind::Reference | Kind::Voiceover | Kind::Audio | Kind::Tv)
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
    lang_mix: bool,
    source: &'static str,
    rec: Recorder,
    // video from topic
    make_video: bool,
    src_free: bool,
    src_cards: bool,
    src_youtube: bool,
    fresh: bool,
    // TV light sync
    tv_folder: String,
    relight: f64,
    tv_bezel: bool,
    // short clips: "edited" (best moments cut together) or "sections" (one continuous part each)
    clip_type: &'static str,
    /// "1080p" or "4k" (in every shape)
    resolution: &'static str,
    // hooks: the voiceover edit's opening, and the hook library's sync
    vo_hook: &'static str,
    hook_niches: String,
    hook_geos: String,
    hook_claude: bool,
    // video from topic: your own first line, or hooks looked up online for the topic
    hook_line: String,
    find_hook: bool,
    /// Editing this waiting queue item (its id) instead of starting a new job.
    queue_edit: Option<u64>,
    // optional emotional sound design (music per mood, hits, risers, pauses)
    sound: bool,
    music_source: &'static str,
    music_folder: String,
    licence: &'static str,
    sound_effects: bool,
    music_level: f64,
    emo: EmotionEditor,
    tv: TvState,
}

/// TV Light Sync: the recordings of the room and the corner check of one of them.
#[derive(Clone, Debug, Default, PartialEq)]
struct TvState {
    scan_job: Option<u64>,
    scanned: String,
    recs: Vec<TvRec>,
    calib: Option<TvCalib>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct TvRec {
    path: String,
    orientation: String,
    calibrated: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct TvCalib {
    path: String,
    job: Option<u64>,
    image: String,
    ref_t: f64,
    /// corners, normalised: top-left, top-right, bottom-right, bottom-left
    quad: [[f32; 2]; 4],
    loaded: bool,
    saving: Option<u64>,
    /// recording length, the moment shown, and where its corners came from (suggested/saved/tracked)
    duration: f64,
    at: f64,
    source: String,
    /// the corner last touched (arrow keys nudge it)
    active: Option<usize>,
    preview_job: Option<u64>,
    preview_image: String,
}

/// The sentence-by-sentence recording session of the Voice Clone dialog.
#[derive(Clone, Debug, Default, PartialEq)]
struct Recorder {
    prompts: Vec<Prompt>,
    loading: Option<u64>,
    mics: Vec<String>,
    mic: String,
    mics_job: Option<u64>,
    /// the language sets the sentences were loaded for (a change reloads them)
    loaded: Vec<String>,
    cursor: usize,
    recording: Option<u64>,
    /// what went wrong with the last take (shown right under the Record button)
    message: String,
    takes: Vec<Option<Take>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Prompt {
    id: String,
    lang: String,
    text: String,
    emotion: String,
    direction: String,
    /// the section it belongs to: "hi", "mix" (Hinglish) or "en"
    set: String,
    /// free talking (no fixed text): the engine transcribes it
    free: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Take {
    file: String,
    issues: String,
    /// length of the take in seconds (0 when found on disk from an earlier session)
    seconds: f64,
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
            captions: kind == Kind::Topic,
            import: !matches!(kind, Kind::Split | Kind::Audio | Kind::Script | Kind::Voice | Kind::Topic),
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
            voice_name: if kind == Kind::Voice { "My voice".into() } else { String::new() },
            lang_en: true,
            lang_hi: true,
            lang_mix: true,
            source: "record",
            rec: Recorder { mic: "0".into(), ..Recorder::default() },
            make_video: true,
            src_free: true,
            src_cards: true,
            src_youtube: false,
            fresh: false,
            tv_folder: setting_str("tv_recordings").unwrap_or_default(),
            relight: 1.0,
            tv_bezel: false,
            clip_type: "edited",
            resolution: "1080p",
            vo_hook: "auto",
            hook_niches: setting_str("hook_niches")
                .unwrap_or_else(|| "gaming, tech, travel, finance, fitness, food, motivation, movies, news, education".into()),
            hook_geos: setting_str("hook_geos").unwrap_or_else(|| "IN, US".into()),
            hook_claude: true,
            hook_line: String::new(),
            find_hook: false,
            queue_edit: None,
            sound: false,
            music_source: "internet",
            music_folder: setting_str("music_folder").unwrap_or_default(),
            licence: "monetized",
            sound_effects: true,
            music_level: -12.0,
            emo: EmotionEditor::default(),
            tv: TvState::default(),
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
        d.captions = p.get("captions").and_then(Value::as_bool).unwrap_or(d.captions);
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
            d.lang_mix = has("mix");
        }
        if let Some(a) = p.get("styleSources").and_then(Value::as_array) {
            d.style_sources = a.iter().filter_map(Value::as_str).map(str::to_string).collect();
        }
        let flag = |k: &str, cur: bool| p.get(k).and_then(Value::as_bool).unwrap_or(cur);
        d.make_video = flag("video", d.make_video);
        d.src_free = flag("free", d.src_free);
        d.src_cards = flag("cards", d.src_cards);
        d.src_youtube = flag("youtube", d.src_youtube);
        d.fresh = flag("fresh", d.fresh);
        if let Some(v) = p.get("recordings").and_then(Value::as_str) {
            d.tv_folder = v.to_string();
        }
        if kind == Kind::Tv && d.clips.is_empty() && !d.input.trim().is_empty() {
            d.clips.push(d.input.trim().to_string()); // one clip given as "input"
        }
        d.relight = num("relight", d.relight);
        d.tv_bezel = flag("bezel", d.tv_bezel);
        d.clip_type = pick("clipType", &CLIP_TYPES.map(|a| a.0), d.clip_type);
        d.sound = flag("soundDesign", d.sound);
        d.resolution = pick("resolution", &RESOLUTIONS.map(|a| a.0), d.resolution);
        d.vo_hook = pick("hook", &HOOK_MODES.map(|a| a.0), d.vo_hook);
        d.hook_claude = flag("hookClaude", d.hook_claude);
        d.find_hook = flag("findHook", d.find_hook);
        if let Some(v) = s("hookLine") {
            d.hook_line = v;
        }
        if let Some(v) = s("niches") {
            d.hook_niches = v;
        }
        if let Some(v) = s("countries") {
            d.hook_geos = v;
        }
        d.music_source = pick("musicSource", &MUSIC_SOURCES.map(|a| a.0), d.music_source);
        d.licence = pick("licence", &LICENCES.map(|a| a.0), d.licence);
        d.sound_effects = flag("soundEffects", d.sound_effects);
        d.music_level = num("musicLevel", d.music_level);
        if let Some(f) = s("musicFolder") {
            d.music_folder = f;
        }
        if d.clip_type == "sections" {
            d.min_len = num("min", SECTION_LENGTHS.0);
            d.target_len = num("target", SECTION_LENGTHS.1);
            d.max_len = num("max", SECTION_LENGTHS.2);
        }
        if let Some(a) = p.get("takesFile").and_then(Value::as_str) {
            d.script = a.to_string();
        }
        d
    }

    /// The draft as control-channel parameters (`Draft::from_params` reads them back): what the
    /// queue keeps, edits and exports.
    fn to_params(&self) -> Value {
        let mut m = serde_json::Map::new();
        m.insert("input".into(), json!(self.input));
        m.insert("music".into(), json!(self.music));
        m.insert("script".into(), json!(self.script));
        m.insert("out".into(), json!(self.out));
        m.insert("clips".into(), json!(self.clips));
        m.insert("clipLength".into(), json!(self.clip_len));
        m.insert("length".into(), json!(self.length));
        m.insert("quality".into(), json!(self.quality));
        m.insert("count".into(), json!(self.count));
        m.insert("min".into(), json!(self.min_len));
        m.insert("target".into(), json!(self.target_len));
        m.insert("max".into(), json!(self.max_len));
        m.insert("content".into(), json!(self.content));
        m.insert("framing".into(), json!(self.framing));
        m.insert("captionMode".into(), json!(self.caption_mode));
        m.insert("titleMode".into(), json!(self.title_mode));
        m.insert("captionStyle".into(), json!(self.caption_style));
        m.insert("llm".into(), json!(self.llm));
        m.insert("format".into(), json!(self.format));
        m.insert("audioQuality".into(), json!(self.audio_quality));
        m.insert("start".into(), json!(self.start));
        m.insert("end".into(), json!(self.end));
        m.insert("mono".into(), json!(self.mono));
        m.insert("normalize".into(), json!(self.normalize));
        m.insert("aspect".into(), json!(self.aspect));
        m.insert("pace".into(), json!(self.pace));
        m.insert("preset".into(), json!(self.preset));
        m.insert("look".into(), json!(self.look));
        m.insert("effects".into(), json!(self.effects));
        m.insert("order".into(), json!(self.order));
        m.insert("musicStart".into(), json!(self.music_start));
        m.insert("captions".into(), json!(self.captions));
        m.insert("import".into(), json!(self.import));
        m.insert("describe".into(), json!(self.describe));
        m.insert("ambience".into(), json!(self.ambience));
        m.insert("topic".into(), json!(self.topic));
        m.insert("niche".into(), json!(self.niche));
        m.insert("tone".into(), json!(self.tone));
        m.insert("audience".into(), json!(self.audience));
        m.insert("notes".into(), json!(self.notes));
        m.insert("style".into(), json!(self.style));
        m.insert("voice".into(), json!(self.voice));
        m.insert("name".into(), json!(self.voice_name));
        m.insert("newStyle".into(), json!(self.new_style));
        m.insert("language".into(), json!(self.language));
        m.insert("writer".into(), json!(self.writer));
        m.insert("source".into(), json!(self.source));
        m.insert("seconds".into(), json!(self.seconds));
        m.insert("takes".into(), json!(self.takes));
        m.insert("styleSources".into(), json!(self.style_sources));
        m.insert("video".into(), json!(self.make_video));
        m.insert("free".into(), json!(self.src_free));
        m.insert("cards".into(), json!(self.src_cards));
        m.insert("youtube".into(), json!(self.src_youtube));
        m.insert("fresh".into(), json!(self.fresh));
        m.insert("recordings".into(), json!(self.tv_folder));
        m.insert("relight".into(), json!(self.relight));
        m.insert("bezel".into(), json!(self.tv_bezel));
        m.insert("clipType".into(), json!(self.clip_type));
        m.insert("soundDesign".into(), json!(self.sound));
        m.insert("resolution".into(), json!(self.resolution));
        m.insert("hookClaude".into(), json!(self.hook_claude));
        m.insert("findHook".into(), json!(self.find_hook));
        m.insert("hookLine".into(), json!(self.hook_line));
        m.insert("niches".into(), json!(self.hook_niches));
        m.insert("countries".into(), json!(self.hook_geos));
        m.insert("musicSource".into(), json!(self.music_source));
        m.insert("licence".into(), json!(self.licence));
        m.insert("soundEffects".into(), json!(self.sound_effects));
        m.insert("musicLevel".into(), json!(self.music_level));
        m.insert("musicFolder".into(), json!(self.music_folder));
        let mut p = Value::Object(m);
        let mut langs = Vec::new();
        for (on, c) in [(self.lang_en, "en"), (self.lang_hi, "hi"), (self.lang_mix, "mix")] {
            if on {
                langs.push(c);
            }
        }
        p["languages"] = json!(langs);
        // "hook" is the teaser checkbox for short clips, and the opening style for a voiceover edit
        p["hook"] = if self.kind == Kind::Clips { json!(self.hook) } else { json!(self.vo_hook) };
        p
    }

    fn style_args(&self, a: &mut Vec<String>) {
        for (k, v) in [("--look", self.look), ("--effects", self.effects), ("--order", self.order)] {
            a.push(k.into());
            a.push(v.into());
        }
    }

    /// Fill defaults (output paths) and check that the job can run; returns the engine arguments.
    /// The optional sound design: a music edit only gets effects (the song is its music).
    fn sound_args(&self, a: &mut Vec<String>) -> Result<(), String> {
        a.push("--sound-design".into());
        let source = if self.kind == Kind::Music { "none" } else { self.music_source };
        if source == "folder" {
            if self.music_folder.trim().is_empty() || !Path::new(self.music_folder.trim()).is_dir() {
                return Err("Choose your music folder (or another music source).".into());
            }
            a.extend(["--music-folder".into(), self.music_folder.trim().into()]);
        }
        a.extend(["--music-source".into(), source.into()]);
        if source == "internet" {
            a.extend(["--licence".into(), self.licence.into()]);
        }
        if !self.sound_effects {
            if source == "none" {
                return Err("Sound design without music needs the sound effects.".into());
            }
            a.push("--no-effects".into());
        }
        if !(-30.0..=0.0).contains(&self.music_level) {
            return Err("Music level must be between -30 and 0 dB.".into());
        }
        a.extend(["--music-level".into(), format!("{}", self.music_level)]);
        Ok(())
    }

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
                let sections = self.clip_type == "sections";
                if !(1.0..=50.0).contains(&n) || n.fract() != 0.0 {
                    return Err("Number of clips must be a whole number from 1 to 50.".into());
                }
                let shortest = if sections { 5.0 } else { 3.0 };
                if !(shortest <= lo && lo <= t && t <= hi && hi <= 600.0) {
                    return Err(format!("Lengths must satisfy {shortest} ≤ shortest ≤ typical ≤ longest ≤ 600 seconds."));
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
                push(&mut a, "--llm", self.llm);
                if sections {
                    a.push("--sections".into());
                } else {
                    push(&mut a, "--pace", self.pace);
                    if !self.music.trim().is_empty() && !self.sound {
                        push(&mut a, "--music", self.music.trim());
                    }
                    if !self.hook {
                        a.push("--no-hook".into());
                    }
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
                if self.vo_hook != "auto" {
                    push(&mut a, "--hook", self.vo_hook);
                }
                if !self.niche.trim().is_empty() {
                    push(&mut a, "--niche", self.niche.trim());
                }
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
                if !self.music.trim().is_empty() && !self.sound {
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
            Kind::Topic => {
                if self.topic.trim().is_empty() {
                    return Err("Write a topic, or paste a link to a post or article.".into());
                }
                if !(5.0..=1800.0).contains(&self.seconds) {
                    return Err("Length must be 5 seconds to 30 minutes.".into());
                }
                if self.make_video && self.voice.trim().is_empty() {
                    return Err("Choose the voice for the video (or untick “Make the video” for the script only).".into());
                }
                if self.make_video && !self.src_free && !self.src_cards && !self.src_youtube && self.clips.is_empty() {
                    return Err("Allow at least one source of pictures, or add your own footage.".into());
                }
                a = vec!["topic".into(), self.topic.trim().into()];
                push(&mut a, "--seconds", &format!("{}", self.seconds));
                push(&mut a, "--language", self.language);
                push(&mut a, "--aspect", if self.aspect == "same" { "horizontal" } else { self.aspect });
                push(&mut a, "--writer", self.writer);
                if !self.style.trim().is_empty() {
                    push(&mut a, "--style", self.style.trim());
                }
                if !self.out.trim().is_empty() {
                    push(&mut a, "--out-dir", self.out.trim());
                }
                if self.fresh {
                    a.push("--fresh".into());
                }
                if !self.hook_line.trim().is_empty() {
                    push(&mut a, "--hook-line", self.hook_line.trim()); // yours wins over a search
                } else if self.find_hook {
                    a.push("--find-hook".into());
                }
                if self.make_video {
                    push(&mut a, "--voice", self.voice.trim());
                    push(&mut a, "--takes", &format!("{}", self.takes.round()));
                    push(&mut a, "--pace", self.pace);
                    a.push("--sources".into());
                    for (on, k) in [(self.src_free, "free"), (self.src_cards, "cards"), (self.src_youtube, "youtube")] {
                        if on {
                            a.push(k.into());
                        }
                    }
                    if !self.clips.is_empty() {
                        a.push("--footage".into());
                        a.extend(self.clips.iter().cloned());
                    }
                    if !self.music.trim().is_empty() && !self.sound {
                        push(&mut a, "--music", self.music.trim());
                    }
                    if !self.captions {
                        a.push("--no-captions".into());
                    }
                } else {
                    a.push("--script-only".into());
                }
            }
            Kind::Hooks => {
                let list = |v: &str| v.split([',', ' ']).map(str::trim).filter(|x| !x.is_empty()).map(str::to_string).collect::<Vec<_>>();
                let niches = list(&self.hook_niches);
                if niches.is_empty() {
                    return Err("Write at least one niche (e.g. gaming, travel).".into());
                }
                a = vec!["hooks-sync".into(), "--niches".into()];
                a.extend(niches);
                let geos = list(&self.hook_geos);
                if !geos.is_empty() {
                    a.push("--geo".into());
                    a.extend(geos);
                }
                if !self.hook_claude {
                    a.push("--no-claude".into());
                }
            }
            Kind::Speak => {
                let text = self.notes.trim();
                if self.script.trim().is_empty() && text.is_empty() {
                    return Err("Choose a script file, or type the text to speak.".into());
                }
                if self.voice.trim().is_empty() {
                    return Err("Choose the voice.".into());
                }
                if !(1.0..=4.0).contains(&self.takes) {
                    return Err("Takes per line must be 1 to 4.".into());
                }
                push(&mut a, "--voice", self.voice.trim());
                push(&mut a, "--script", if self.script.trim().is_empty() { text } else { self.script.trim() });
                push(&mut a, "--out", self.out.trim());
                if !self.voice.starts_with("mac") {
                    push(&mut a, "--takes", &format!("{}", self.takes.round()));
                }
            }
            Kind::Tv => {
                if !exists(&self.tv_folder) {
                    return Err("Choose the folder with the recordings of your room.".into());
                }
                if !(0.0..=1.0).contains(&self.relight) {
                    return Err("Backlight matching must be between 0 and 1.".into());
                }
                let orientation = if self.aspect == "vertical" { "vertical" } else { "horizontal" };
                a = vec!["tv-make".into()];
                a.extend(self.clips.iter().cloned());
                push(&mut a, "--recordings", self.tv_folder.trim());
                push(&mut a, "--orientation", orientation);
                push(&mut a, "--relight", &format!("{}", self.relight));
                if self.tv_bezel {
                    push(&mut a, "--bezel", "0.012"); // thin; the bottom is twice as wide
                }
                push(&mut a, "--out-dir", self.out.trim());
            }
            Kind::Voice => {
                let name = self.voice_name.trim();
                if name.is_empty() {
                    return Err("Give the voice a name.".into());
                }
                if !self.lang_en && !self.lang_hi && !self.lang_mix {
                    return Err("Choose at least one language.".into());
                }
                a = vec!["voice-enroll".into()];
                push(&mut a, "--name", name);
                a.push("--languages".into());
                if self.lang_en {
                    a.push("en".into());
                }
                if self.lang_hi || self.lang_mix {
                    a.push("hi".into()); // Hinglish is spoken by the Hindi voice
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
        if self.kind.has_resolution() && self.resolution != "1080p" && !(self.kind == Kind::Topic && !self.make_video) {
            a.extend(["--resolution".into(), self.resolution.into()]);
        }
        if self.kind.has_sound_design() && self.sound && !(self.kind == Kind::Topic && !self.make_video) {
            self.sound_args(&mut a)?;
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

const HOOK_MODES: [(&str, &str); 5] = [
    ("auto", "Automatic: open on the strongest line, on-screen hook, fast first cuts"),
    ("teaser", "Open on the strongest line (cold open), then the start"),
    ("text", "On-screen hook text over the first seconds"),
    ("both", "Cold open and on-screen hook"),
    ("off", "No hook (start as recorded)"),
];
const RESOLUTIONS: [(&str, &str); 2] = [("1080p", "1080p (Full HD)"), ("4k", "4K (Ultra HD: 3840 on the long side; slower, bigger files)")];
const MUSIC_SOURCES: [(&str, &str); 4] = [
    ("internet", "Free music from the internet, chosen by mood (credited in the description)"),
    ("folder", "My music folder (EI sorts it by mood)"),
    ("ai", "AI-composed score on this Mac (slow; personal use only)"),
    ("none", "No music, sound effects only"),
];
const LICENCES: [(&str, &str); 2] =
    [("monetized", "Safe for monetized videos (CC0, CC BY, CC BY-SA)"), ("personal", "Personal use (more music, incl. non-commercial licences)")];
/// The emotions of the sound design (`ai-engine/emotions.py`), grouped by mood.
const EMOTION_NAMES: [&str; 43] = [
    "Neutral",
    "Joy",
    "Happiness",
    "Cheerful",
    "Delight",
    "Bliss",
    "Euphoria",
    "Excitement",
    "Thrill",
    "Energy",
    "Love",
    "Passion",
    "Affection",
    "Romance",
    "Nostalgia",
    "Melancholy",
    "Sorrow",
    "Grief",
    "Fear",
    "Dread",
    "Anxiety",
    "Suspense",
    "Tension",
    "Anticipation",
    "Mystery",
    "Curiosity",
    "Fascination",
    "Wonder",
    "Awe",
    "Amazement",
    "Wonderment",
    "Surprise",
    "Serenity",
    "Tranquility",
    "Peace",
    "Calmness",
    "Courage",
    "Triumph",
    "Pride",
    "Hope",
    "Freedom",
    "Anger",
    "Rage",
];

/// One line of a script with its emotion (the per-line override of the sound design).
#[derive(Clone, Debug, PartialEq)]
struct EmoLine {
    text: String,
    emotion: String,
    intensity: f64,
}

/// The emotions of a script file, editable line by line; changes are written back to the file.
#[derive(Clone, Debug, Default, PartialEq)]
struct EmotionEditor {
    path: String,
    lines: Vec<EmoLine>,
    error: String,
}

/// The canonical emotion name for a tag ("hook", "excited" and other old tags included).
fn emotion_name(tag: &str) -> String {
    let t = tag.trim();
    if let Some(n) = EMOTION_NAMES.iter().find(|n| n.eq_ignore_ascii_case(t)) {
        return (*n).to_string();
    }
    let alias = match t.to_lowercase().as_str() {
        "calm" => "Calmness",
        "warm" => "Affection",
        "story" => "Nostalgia",
        "excited" => "Excitement",
        "curious" => "Curiosity",
        "serious" => "Tension",
        "suspense" => "Suspense",
        "hook" => "Surprise",
        "cta" => "Hope",
        "happy" => "Happiness",
        "sad" => "Sorrow",
        "angry" => "Anger",
        _ => "Neutral",
    };
    alias.to_string()
}

/// "[Awe 0.8] text" -> (Some(("Awe", Some(0.8))), "text"); plain text -> (None, text).
fn parse_tagged(line: &str) -> (Option<(String, Option<f64>)>, String) {
    let l = line.trim();
    if let Some(rest) = l.strip_prefix('[')
        && let Some(end) = rest.find(']')
    {
        let inside = rest.get(..end).unwrap_or_default();
        let text = rest.get(end + 1..).unwrap_or_default().trim().to_string();
        let mut parts = inside.split_whitespace();
        let name = parts.next().unwrap_or_default().to_string();
        let inten = parts.next().and_then(|v| v.parse::<f64>().ok()).filter(|v| v.is_finite()).map(|v| v.clamp(0.0, 1.0));
        if !name.is_empty() {
            return (Some((name, inten)), text);
        }
    }
    (None, l.to_string())
}

impl EmotionEditor {
    /// Loads the lines of a `.script.json` (segments) or a tagged `.txt` script.
    fn load(path: &str) -> EmotionEditor {
        let mut ed = EmotionEditor { path: path.to_string(), ..EmotionEditor::default() };
        let p = Path::new(path.trim());
        let Ok(text) = std::fs::read_to_string(p) else {
            ed.error = "The script could not be read.".into();
            return ed;
        };
        if path.trim().to_lowercase().ends_with(".json") {
            match serde_json::from_str::<Value>(&text) {
                Ok(v) => {
                    for seg in v.get("segments").and_then(Value::as_array).into_iter().flatten() {
                        let t = seg.get("text").and_then(Value::as_str).unwrap_or_default().trim().to_string();
                        if t.is_empty() {
                            continue;
                        }
                        let emotion = emotion_name(seg.get("emotion").and_then(Value::as_str).unwrap_or("Neutral"));
                        let intensity =
                            seg.get("intensity").or_else(|| seg.get("energy")).and_then(Value::as_f64).filter(|v| v.is_finite()).unwrap_or(0.6).clamp(0.0, 1.0);
                        ed.lines.push(EmoLine { text: t, emotion, intensity });
                    }
                }
                Err(_) => ed.error = "The script is not valid JSON.".into(),
            }
        } else {
            for l in text.lines().filter(|l| !l.trim().is_empty()) {
                let (tag, t) = parse_tagged(l);
                let (emotion, intensity) = match tag {
                    Some((n, i)) => (emotion_name(&n), i.unwrap_or(0.6)),
                    None => ("Neutral".to_string(), 0.6),
                };
                ed.lines.push(EmoLine { text: t, emotion, intensity });
            }
        }
        ed
    }

    /// Writes the emotions back into the script file (only the emotion and intensity change).
    fn save(&mut self) {
        let p = Path::new(self.path.trim());
        let result = if self.path.trim().to_lowercase().ends_with(".json") {
            std::fs::read_to_string(p).map_err(|e| e.to_string()).and_then(|text| {
                let mut v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
                let mut k = 0usize;
                if let Some(segs) = v.get_mut("segments").and_then(Value::as_array_mut) {
                    for seg in segs.iter_mut() {
                        if seg.get("text").and_then(Value::as_str).unwrap_or_default().trim().is_empty() {
                            continue;
                        }
                        if let (Some(line), Some(obj)) = (self.lines.get(k), seg.as_object_mut()) {
                            obj.insert("emotion".into(), json!(line.emotion));
                            obj.insert("intensity".into(), json!((line.intensity * 100.0).round() / 100.0));
                        }
                        k += 1;
                    }
                }
                serde_json::to_string_pretty(&v).map_err(|e| e.to_string())
            })
        } else {
            Ok(self.lines.iter().map(|l| format!("[{} {:.1}] {}", l.emotion, l.intensity, l.text)).collect::<Vec<_>>().join("\n") + "\n")
        };
        match result.and_then(|out| std::fs::write(p, out).map_err(|e| e.to_string())) {
            Ok(()) => self.error.clear(),
            Err(e) => self.error = format!("Could not save the emotions: {e}"),
        }
    }
}

const CLIP_TYPES: [(&str, &str); 2] = [
    ("edited", "Edited short clips: the best moments cut together (teaser, slow motion, punch-ins)"),
    ("sections", "Continuous sections: one uninterrupted part of the video each, cut where it starts and ends naturally"),
];
/// Shortest, usual and longest length of a continuous section (seconds).
const SECTION_LENGTHS: (f64, f64, f64) = (15.0, 60.0, 180.0);
/// The same for edited clips.
const EDITED_LENGTHS: (f64, f64, f64) = (10.0, 48.0, 60.0);
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
const TV_SHAPES: [(&str, &str); 2] =
    [("horizontal", "Horizontal (16:9, from the horizontal recordings)"), ("vertical", "Vertical (9:16, from the vertical recordings)")];
const MAC_VOICES: [(&str, &str); 5] = [
    ("mac:Rishi", "Rishi: Indian English (macOS voice, quick preview)"),
    ("mac:Tara", "Tara: Indian English (macOS voice)"),
    ("mac:Aman", "Aman: Indian English (macOS voice)"),
    ("mac:Lekha", "Lekha: Hindi (macOS voice)"),
    ("mac", "macOS voice for the script's language"),
];
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
        Kind::Clips if d.clip_type == "sections" => free_path(&parent_dir(&d.input), &format!("{} - EI sections", file_stem(&d.input)), ""),
        Kind::Clips => free_path(&parent_dir(&d.input), &format!("{} - EI clips", file_stem(&d.input)), ""),
        Kind::Audio => d.clips.first().map(|c| parent_dir(c).to_string_lossy().to_string()).unwrap_or_else(|| movies_dir().to_string_lossy().to_string()),
        Kind::Script => movies_dir().join("Scripts").join(slug(if d.topic.trim().is_empty() { "script" } else { &d.topic })).to_string_lossy().to_string(),
        Kind::Voice => voices_dir().join(slug(&d.voice_name)).to_string_lossy().to_string(),
        Kind::Topic => String::new(), // the engine names the folder after the topic (Movies/EI/Topics/…)
        Kind::Hooks => String::new(), // the library has one place (Application Support/EI/hooks)
        Kind::Speak => {
            let stem = if d.script.trim().is_empty() { "spoken".to_string() } else { file_stem(&d.script).trim_end_matches(".script").to_string() };
            free_path(&movies_dir().join("Sound"), &stem, ".wav")
        }
        Kind::Tv => d
            .clips
            .first()
            .map(|c| parent_dir(c).join("EI on TV").to_string_lossy().to_string())
            .unwrap_or_else(|| movies_dir().join("TV").to_string_lossy().to_string()),
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

/// A string setting from EI's settings file.
fn setting_str(key: &str) -> Option<String> {
    let saved = std::fs::read_to_string(settings_path()).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok())?;
    saved.get(key).and_then(Value::as_str).map(str::to_string)
}

/// Save one string setting, keeping the others.
fn save_setting(key: &str, value: &str) -> Result<(), String> {
    let path = settings_path();
    let mut v = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()).filter(Value::is_object).unwrap_or_else(|| json!({}));
    if let Some(o) = v.as_object_mut() {
        o.insert(key.into(), json!(value));
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("could not save settings: {e}"))?;
    }
    std::fs::write(&path, serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?).map_err(|e| format!("could not save settings: {e}"))
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
    /// What ran (for Retry and the queue) and where.
    steps: Vec<Vec<String>>,
    cwd: Option<PathBuf>,
    /// The engine process (its process group holds ffmpeg and model workers too).
    pid: Option<u32>,
    /// The engine's own estimate: (fraction done, current stage).
    progress: Option<(f32, String)>,
    paused_since: Option<Instant>,
    paused_total: Duration,
    /// Started from the queue (the next queued job waits for it).
    queued: bool,
}

impl Job {
    /// Time spent working (pauses not counted).
    fn active(&self) -> Duration {
        let total = self.elapsed.unwrap_or_else(|| self.started.elapsed());
        let paused = self.paused_total + self.paused_since.map(|t| t.elapsed()).unwrap_or_default();
        total.saturating_sub(paused)
    }

    fn summary(&self) -> Value {
        json!({
            "job": self.id,
            "title": self.title,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "seconds": self.elapsed.unwrap_or_else(|| self.started.elapsed()).as_secs_f64(),
            "lastLog": self.log.last(),
            "progress": self.progress.as_ref().map(|p| p.0),
            "stage": self.progress.as_ref().map(|p| p.1.clone()),
            "paused": self.paused_since.is_some(),
            "queued": self.queued,
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
    start_job_ex(title, kind, steps, import, cwd, false)
}

fn start_job_ex(title: String, kind: Option<Kind>, steps: Vec<Vec<String>>, import: bool, cwd: Option<PathBuf>, queued: bool) -> u64 {
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
            steps: steps.clone(),
            cwd: cwd.clone(),
            pid: None,
            progress: None,
            paused_since: None,
            paused_total: Duration::ZERO,
            queued,
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
            j.pid = None;
            if let Some(t) = j.paused_since.take() {
                j.paused_total += t.elapsed();
            }
            j.elapsed = Some(j.started.elapsed());
            if status == Status::Done {
                j.progress = Some((1.0, String::new()));
            }
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
        #[cfg(unix)]
        {
            // its own process group: pause, resume and cancel reach ffmpeg and model workers too
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        let mut child = cmd.spawn().map_err(|e| format!("could not run {prog}: {e}"))?;
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let pid = child.id();
        let child = Arc::new(Mutex::new(child));
        with_job(id, |j| {
            j.child = Some(child.clone());
            j.pid = Some(pid);
        });
        let err_thread = stderr.map(|s| {
            std::thread::spawn(move || {
                for line in BufReader::new(s).lines().map_while(Result::ok) {
                    if let Some(p) = parse_progress(&line) {
                        with_job(id, |j| j.progress = Some(p));
                        continue;
                    }
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

/// Stop every running AI job (EI is quitting: no engine process, recording or model keeps running).
pub fn cancel_all() {
    requeue_running();
    let ids: Vec<u64> = jobs().lock().unwrap_or_else(PoisonError::into_inner).iter().filter(|j| j.status == Status::Running).map(|j| j.id).collect();
    for id in ids {
        cancel_job(id);
    }
}

fn cancel_job(id: u64) -> bool {
    with_job(id, |j| {
        j.cancel = true;
        if let Some(pid) = j.pid {
            signal_group(pid, "TERM");
            if j.paused_since.is_some() {
                signal_group(pid, "CONT"); // a stopped process can only exit once it runs again
            }
        }
        if let Some(c) = &j.child {
            let _ = c.lock().unwrap_or_else(PoisonError::into_inner).kill();
        }
        j.status == Status::Running
    })
    .unwrap_or(false)
}

/// `EI_PROGRESS {"fraction": 0.42, "stage": "voice"}` -> (0.42, "voice").
fn parse_progress(line: &str) -> Option<(f32, String)> {
    let v: Value = serde_json::from_str(line.strip_prefix("EI_PROGRESS ")?).ok()?;
    let f = v.get("fraction").and_then(Value::as_f64).filter(|f| f.is_finite())?.clamp(0.0, 1.0) as f32;
    Some((f, v.get("stage").and_then(Value::as_str).unwrap_or_default().to_string()))
}

/// Send a signal to a job's whole process group (the engine, ffmpeg, model workers).
fn signal_group(pid: u32, sig: &str) {
    if pid == 0 {
        return;
    }
    let _ = Command::new("/bin/kill")
        .args([format!("-{sig}"), "--".to_string(), format!("-{pid}")])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

/// Pause (true) or resume (false) a running job. Returns whether anything changed.
fn pause_job(id: u64, pause: bool) -> bool {
    with_job(id, |j| {
        let Some(pid) = j.pid.filter(|_| j.status == Status::Running) else { return false };
        match (pause, j.paused_since) {
            (true, None) => {
                signal_group(pid, "STOP");
                j.paused_since = Some(Instant::now());
                push_log_locked(j, "Paused".into());
                true
            }
            (false, Some(t)) => {
                signal_group(pid, "CONT");
                j.paused_total += t.elapsed();
                j.paused_since = None;
                push_log_locked(j, "Resumed".into());
                true
            }
            _ => false,
        }
    })
    .unwrap_or(false)
}

fn push_log_locked(j: &mut Job, line: String) {
    j.log.push(line);
    if j.log.len() > MAX_LOG {
        let extra = j.log.len() - MAX_LOG;
        j.log.drain(..extra);
    }
}

/// Run a finished, failed or cancelled job again. The engine keeps what each stage made (research,
/// script, voice takes, media, rendered shots), so a retry continues where the job stopped.
fn retry_job(id: u64) -> Option<u64> {
    let (title, kind, steps, import, cwd) =
        with_job(id, |j| (j.status != Status::Running).then(|| (j.title.clone(), j.kind, j.steps.clone(), j.import, j.cwd.clone())))??;
    if steps.is_empty() {
        return None;
    }
    let title = if title.ends_with(" (retry)") { title } else { format!("{title} (retry)") };
    Some(start_job(title, kind, steps, import, cwd))
}

// ---------------------------------------------------------------------------------------------
// Queue (Video from Topic): one job at a time, kept on disk so it survives quitting EI
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct QueueItem {
    title: String,
    kind: String,
    steps: Vec<Vec<String>>,
    import: bool,
    cwd: Option<String>,
    /// The dialog's settings (control-channel parameters): what Edit reopens and Export writes.
    #[serde(default)]
    params: Option<Value>,
    #[serde(default)]
    uid: u64,
}

fn next_uid() -> u64 {
    static N: OnceLock<Mutex<u64>> = OnceLock::new();
    let mut n = N.get_or_init(|| Mutex::new(0)).lock().unwrap_or_else(PoisonError::into_inner);
    *n += 1;
    *n
}

#[derive(Default)]
struct QueueState {
    items: Vec<QueueItem>,
    paused: bool,
    loaded: bool,
    note: String,
}

fn queue() -> &'static Mutex<QueueState> {
    static Q: OnceLock<Mutex<QueueState>> = OnceLock::new();
    Q.get_or_init(|| Mutex::new(QueueState::default()))
}

fn queue_path() -> PathBuf {
    settings_path().parent().map(Path::to_path_buf).unwrap_or_else(support_dir).join("queue.json")
}

fn save_queue(q: &QueueState) {
    let path = queue_path();
    if q.items.is_empty() {
        let _ = std::fs::remove_file(&path);
        return;
    }
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(t) = serde_json::to_string_pretty(&json!({"items": q.items, "paused": q.paused})) {
        let tmp = path.with_extension("json.part");
        if std::fs::write(&tmp, t).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }
}

/// The queue saved when EI last quit (paused: it waits for Resume).
fn load_queue(q: &mut QueueState) {
    if q.loaded {
        return;
    }
    q.loaded = true;
    let Ok(t) = std::fs::read_to_string(queue_path()) else { return };
    let Ok(v) = serde_json::from_str::<Value>(&t) else { return };
    let items: Vec<QueueItem> = v.get("items").and_then(|i| serde_json::from_value(i.clone()).ok()).unwrap_or_default();
    if !items.is_empty() {
        q.note = format!("{} task(s) waiting from last time: press Resume Queue to continue", items.len());
        q.items = items
            .into_iter()
            .map(|mut it| {
                it.uid = next_uid();
                if it.params.is_none() {
                    it.params = params_from_args(&it.kind, it.steps.first().map(|s| s.get(2..).unwrap_or_default()).unwrap_or_default());
                }
                it
            })
            .collect();
        q.paused = true;
    }
}

fn enqueue(title: String, kind: Kind, steps: Vec<Vec<String>>, import: bool, cwd: Option<PathBuf>, params: Option<Value>) -> usize {
    let mut q = queue().lock().unwrap_or_else(PoisonError::into_inner);
    load_queue(&mut q);
    q.items.push(QueueItem { title, kind: kind.id().to_string(), steps, import, cwd: cwd.map(|c| c.to_string_lossy().to_string()), params, uid: next_uid() });
    save_queue(&q);
    q.items.len()
}

/// Settings from an older queue item's command line (Video from Topic), so it can be edited.
fn params_from_args(kind: &str, args: &[String]) -> Option<Value> {
    if kind != "topic" || args.first().map(String::as_str) != Some("topic") {
        return None;
    }
    let mut p = json!({"topic": args.get(1)?, "video": true});
    let mut i = 2;
    while let Some(a) = args.get(i) {
        let val = args.get(i + 1).cloned().unwrap_or_default();
        let num = |v: &str| v.parse::<f64>().ok().filter(|x| x.is_finite()).map(|x| json!(x)).unwrap_or(Value::Null);
        let (key, value, used) = match a.as_str() {
            "--fresh" => ("fresh", json!(true), 1),
            "--find-hook" => ("findHook", json!(true), 1),
            "--no-captions" => ("captions", json!(false), 1),
            "--script-only" => ("video", json!(false), 1),
            "--sound-design" => ("soundDesign", json!(true), 1),
            "--no-effects" => ("soundEffects", json!(false), 1),
            "--seconds" => ("seconds", num(&val), 2),
            "--takes" => ("takes", num(&val), 2),
            "--music-level" => ("musicLevel", num(&val), 2),
            "--language" => ("language", json!(val), 2),
            "--aspect" => ("aspect", json!(val), 2),
            "--writer" => ("writer", json!(val), 2),
            "--style" => ("style", json!(val), 2),
            "--out-dir" => ("out", json!(val), 2),
            "--hook-line" => ("hookLine", json!(val), 2),
            "--voice" => ("voice", json!(val), 2),
            "--pace" => ("pace", json!(val), 2),
            "--music" => ("music", json!(val), 2),
            "--resolution" => ("resolution", json!(val), 2),
            "--music-source" => ("musicSource", json!(val), 2),
            "--music-folder" => ("musicFolder", json!(val), 2),
            "--licence" => ("licence", json!(val), 2),
            "--niche" => ("niche", json!(val), 2),
            "--sources" | "--footage" => {
                let vals: Vec<String> = args.get(i + 1..).unwrap_or_default().iter().take_while(|v| !v.starts_with("--")).cloned().collect();
                let n = vals.len();
                if a == "--sources" {
                    for s in ["free", "cards", "youtube"] {
                        p[s] = json!(vals.iter().any(|v| v == s));
                    }
                } else {
                    p["clips"] = json!(vals);
                }
                i += 1 + n;
                continue;
            }
            _ => return None, // an option this version does not know: not editable, still runnable
        };
        p[key] = value;
        i += used;
    }
    Some(p)
}

/// The engine arguments of a waiting item (to recognise the same task twice).
fn item_args(it: &QueueItem) -> Vec<String> {
    it.steps.first().map(|s| s.get(2..).unwrap_or_default().to_vec()).unwrap_or_default()
}

/// Add tasks from a queue file (`{"tasks": [{"command": "ai.topic", "params": {...}}]}`, or a list
/// of tasks). Each is checked like its dialog would; a task already waiting is skipped.
/// Returns (added, already waiting, problems).
fn import_tasks(v: &Value) -> (usize, usize, Vec<String>) {
    let (mut added, mut dup, mut errors) = (0, 0, Vec::new());
    for t in check_tasks(v) {
        match t {
            Ok((title, kind, steps, engine, d)) => {
                let args = steps.first().map(|s| s.get(2..).unwrap_or_default().to_vec()).unwrap_or_default();
                let waiting = {
                    let mut q = queue().lock().unwrap_or_else(PoisonError::into_inner);
                    load_queue(&mut q);
                    q.items.iter().any(|it| item_args(it) == args)
                };
                if waiting {
                    dup += 1;
                } else {
                    enqueue(title, kind, steps, d.import, Some(engine), Some(d.to_params()));
                    added += 1;
                }
            }
            Err(e) => errors.push(e),
        }
    }
    (added, dup, errors)
}

type CheckedTask = (String, Kind, Vec<Vec<String>>, PathBuf, Draft);

/// Every task of a queue file checked like its dialog would (no queue touched).
fn check_tasks(v: &Value) -> Vec<Result<CheckedTask, String>> {
    let list = v.get("tasks").and_then(Value::as_array).or_else(|| v.as_array()).cloned().unwrap_or_default();
    if list.is_empty() {
        return vec![Err("no tasks in the file (expected {\"tasks\": [...]})".to_string())];
    }
    list.iter()
        .enumerate()
        .map(|(k, t)| {
            let cmd = t.get("command").or_else(|| t.get("kind")).and_then(Value::as_str).unwrap_or_default();
            let id = if cmd.starts_with("ai.") { cmd.to_string() } else { format!("ai.{cmd}") };
            let label = t.get("title").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| format!("task {}", k + 1));
            let kind = Kind::from_command(&id).filter(|k| *k != Kind::Voice).ok_or_else(|| format!("{label}: unknown command “{cmd}”"))?;
            // settings, or (an older export) the engine command line
            let params = match t.get("params").filter(|p| p.is_object()) {
                Some(p) => p.clone(),
                None => t
                    .get("args")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect::<Vec<_>>())
                    .and_then(|a| params_from_args(kind.id(), &a))
                    .ok_or_else(|| format!("{label}: no settings"))?,
            };
            let mut d = Draft::from_params(kind, &params);
            let (title, steps, engine) = draft_job(&mut d).map_err(|e| format!("{label}: {e}"))?;
            let title = t.get("title").and_then(Value::as_str).map(str::to_string).unwrap_or(title);
            Ok((title, kind, steps, engine, d))
        })
        .collect()
}

/// The waiting tasks as a queue file (the format `import_tasks` reads).
fn export_tasks() -> Value {
    let mut q = queue().lock().unwrap_or_else(PoisonError::into_inner);
    load_queue(&mut q);
    let tasks: Vec<Value> = q
        .items
        .iter()
        .map(|it| json!({"command": format!("ai.{}", it.kind), "title": it.title, "params": it.params.clone().unwrap_or(Value::Null), "args": item_args(it)}))
        .collect();
    json!({"ei_queue": 1, "tasks": tasks})
}

/// Move a waiting task: `to` is its new position (0 = next to run).
fn move_waiting(from: usize, to: usize) -> bool {
    let mut q = queue().lock().unwrap_or_else(PoisonError::into_inner);
    let ok = move_in(&mut q.items, from, to);
    if ok {
        save_queue(&q);
    }
    ok
}

fn move_in<T>(v: &mut Vec<T>, from: usize, to: usize) -> bool {
    if from >= v.len() {
        return false;
    }
    let it = v.remove(from);
    let to = to.min(v.len());
    v.insert(to, it);
    true
}

/// Start the next queued job when nothing from the queue is running (called every frame).
fn tick_queue() {
    let busy = jobs().lock().unwrap_or_else(PoisonError::into_inner).iter().any(|j| j.queued && j.status == Status::Running);
    let next = {
        let mut q = queue().lock().unwrap_or_else(PoisonError::into_inner);
        load_queue(&mut q);
        if busy || q.paused || q.items.is_empty() {
            return;
        }
        let it = q.items.remove(0);
        save_queue(&q);
        it
    };
    let kind = Kind::from_command(&format!("ai.{}", next.kind));
    let _ = start_job_ex(next.title, kind, next.steps, next.import, next.cwd.map(PathBuf::from), true);
}

/// Cancel what the queue is running and everything waiting in it.
fn cancel_queue(include_current: bool) -> usize {
    let n = {
        let mut q = queue().lock().unwrap_or_else(PoisonError::into_inner);
        let n = q.items.len();
        q.items.clear();
        q.note.clear();
        save_queue(&q);
        n
    };
    if include_current {
        let ids: Vec<u64> =
            jobs().lock().unwrap_or_else(PoisonError::into_inner).iter().filter(|j| j.queued && j.status == Status::Running).map(|j| j.id).collect();
        for id in ids {
            cancel_job(id);
        }
    }
    n
}

/// EI is quitting: a queued job that is still running goes back to the front of the saved queue
/// (it continues from its saved stages next time).
fn requeue_running() {
    let running: Vec<QueueItem> = jobs()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .filter(|j| j.queued && j.status == Status::Running)
        .map(|j| QueueItem {
            title: j.title.clone(),
            kind: j.kind.map(Kind::id).unwrap_or("topic").to_string(),
            steps: j.steps.clone(),
            import: j.import,
            cwd: j.cwd.as_ref().map(|c| c.to_string_lossy().to_string()),
            params: j.kind.and_then(|k| params_from_args(k.id(), j.steps.first().map(|s| s.get(2..).unwrap_or_default()).unwrap_or_default())),
            uid: next_uid(),
        })
        .collect();
    if running.is_empty() {
        return;
    }
    let mut q = queue().lock().unwrap_or_else(PoisonError::into_inner);
    load_queue(&mut q);
    for (k, it) in running.into_iter().enumerate() {
        q.items.insert(k, it);
    }
    save_queue(&q);
}

/// A draft's job: (title, steps, engine folder), after checking it can run.
fn draft_job(d: &mut Draft) -> Result<(String, Vec<Vec<String>>, PathBuf), String> {
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
        Kind::Topic => format!("{}: {}", d.kind.title(), d.topic.trim().chars().take(40).collect::<String>()),
        Kind::Speak => format!("{}: {}", d.kind.title(), file_stem(&d.out)),
        Kind::Hooks => format!("{}: sync {}", d.kind.title(), d.hook_niches.trim().chars().take(40).collect::<String>()),
        Kind::Tv => match d.clips.len() {
            1 => format!("{}: {}", d.kind.title(), d.clips.first().map(|c| file_stem(c)).unwrap_or_default()),
            n => format!("{}: {n} clips", d.kind.title()),
        },
    };
    Ok((title, vec![step], engine))
}

/// Run a draft: returns the job id.
fn run_draft(d: &mut Draft) -> Result<u64, String> {
    let (title, steps, engine) = draft_job(d)?;
    Ok(start_job(title, Some(d.kind), steps, d.import, Some(engine)))
}

/// Put a draft in the queue instead of running it now. Returns the number waiting.
fn queue_draft(d: &mut Draft) -> Result<usize, String> {
    if d.kind == Kind::Voice {
        return Err("Voice Clone records with you there, so it cannot wait in the queue.".into());
    }
    let (title, steps, engine) = draft_job(d)?;
    Ok(enqueue(title, d.kind, steps, d.import, Some(engine), Some(d.to_params())))
}

/// Save an edited waiting task back into its place in the queue.
fn save_queue_edit(d: &mut Draft, uid: u64) -> Result<(), String> {
    let (title, steps, engine) = draft_job(d)?;
    let mut q = queue().lock().unwrap_or_else(PoisonError::into_inner);
    let it = q.items.iter_mut().find(|it| it.uid == uid).ok_or("That task is no longer waiting (it started, or was removed).")?;
    it.title = title;
    it.steps = steps;
    it.import = d.import;
    it.cwd = Some(engine.to_string_lossy().to_string());
    it.params = Some(d.to_params());
    save_queue(&q);
    Ok(())
}

/// Open a waiting task's dialog with its settings, to edit it.
fn edit_waiting(ctx: &egui::Context, uid: u64) -> Result<(), String> {
    let (kind, params) = {
        let mut q = queue().lock().unwrap_or_else(PoisonError::into_inner);
        load_queue(&mut q);
        let it = q.items.iter().find(|it| it.uid == uid).ok_or("That task is no longer waiting.")?;
        let kind = Kind::from_command(&format!("ai.{}", it.kind)).ok_or("Unknown task kind.")?;
        (kind, it.params.clone().ok_or("This task was queued by an older EI and cannot be edited: remove it and queue it again.")?)
    };
    let mut d = Draft::from_params(kind, &params);
    d.queue_edit = Some(uid);
    ctx.data_mut(|m| m.insert_temp(draft_id(), Some(d)));
    Ok(())
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
        "ai.pause" | "ai.resume" => match params.get("job").and_then(Value::as_u64) {
            Some(j) => Ok(json!({"changed": pause_job(j, id == "ai.pause")})),
            None => Err(format!("{id} needs {{job}}")),
        },
        "ai.retry" => match params.get("job").and_then(Value::as_u64) {
            Some(j) => retry_job(j).map(|n| json!({"job": n})).ok_or_else(|| "that job cannot be run again (still running, or unknown)".to_string()),
            None => Err("ai.retry needs {job}".into()),
        },
        "ai.queue" => {
            let mut q = queue().lock().unwrap_or_else(PoisonError::into_inner);
            load_queue(&mut q);
            if let Some(p) = params.get("paused").and_then(Value::as_bool) {
                q.paused = p;
                q.note.clear();
                save_queue(&q);
            }
            Ok(json!({"waiting": q.items.iter().map(|i| i.title.clone()).collect::<Vec<_>>(), "paused": q.paused}))
        }
        "ai.queueImport" => {
            let v = match params.get("path").and_then(Value::as_str) {
                Some(path) => {
                    std::fs::read_to_string(path).map_err(|e| e.to_string()).and_then(|t| serde_json::from_str::<Value>(&t).map_err(|e| e.to_string()))
                }
                None => Ok(params.clone()),
            };
            v.map(|v| {
                let (added, dup, errors) = import_tasks(&v);
                json!({"added": added, "alreadyWaiting": dup, "errors": errors})
            })
        }
        "ai.queueExport" => {
            let v = export_tasks();
            match params.get("path").and_then(Value::as_str) {
                Some(path) => serde_json::to_string_pretty(&v)
                    .map_err(|e| e.to_string())
                    .and_then(|t| std::fs::write(path, t).map_err(|e| e.to_string()))
                    .map(|()| json!({"path": path, "tasks": v["tasks"].as_array().map(Vec::len)})),
                None => Ok(v),
            }
        }
        "ai.queueMove" => match (params.get("index").and_then(Value::as_u64), params.get("to").and_then(Value::as_u64)) {
            (Some(i), Some(t)) => Ok(json!({"moved": move_waiting(i as usize, t as usize)})),
            _ => Err("ai.queueMove needs {index, to} (0 = next to run)".into()),
        },
        "ai.queueCancel" => {
            let all = params.get("current").and_then(Value::as_bool).unwrap_or(false);
            Ok(json!({"removed": cancel_queue(all)}))
        }
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
            if has_params && params.get("queue").and_then(Value::as_bool).unwrap_or(false) {
                let mut d = Draft::from_params(kind, params);
                queue_draft(&mut d).map(|n| {
                    ctx.data_mut(|m| m.insert_temp(jobs_open_id(), true));
                    json!({"queued": n})
                })
            } else if has_params {
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
    TvFolder,
    MusicFolder,
}

pub fn show(app: &mut FilmcraftApp, ctx: &egui::Context) {
    tick_queue();
    show_draft(app, ctx);
    show_jobs(app, ctx);
    show_setup(app, ctx);
    import_finished(app);
    if jobs().lock().unwrap_or_else(PoisonError::into_inner).iter().any(|j| j.status == Status::Running) {
        ctx.request_repaint_after(Duration::from_millis(300));
    }
}

/// The hook library: its state, and what the next sync fetches.
fn hooks_rows(ui: &mut egui::Ui, d: &mut Draft, p: &str, elems: &mut Vec<(String, egui::Rect, String)>) {
    let lib = settings_path().parent().map(Path::to_path_buf).unwrap_or_else(support_dir).join("hooks").join("library.json");
    let data: Value = std::fs::read_to_string(&lib).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
    let hooks = data.get("hooks").and_then(Value::as_array).cloned().unwrap_or_default();
    let last = data.get("syncs").and_then(Value::as_array).and_then(|v| v.last()).and_then(|v| v.get("at")).and_then(Value::as_str).unwrap_or("never");
    ui.label("Library");
    ui.vertical(|ui| {
        let found = hooks.iter().filter(|h| h.get("source").and_then(Value::as_str) != Some("built in")).count();
        ui.label(format!(
            "{} hooks ({} built in, {} found online) · last sync: {}",
            hooks.len(),
            hooks.len() - found,
            found,
            last.replace('T', " ").trim_end_matches('Z')
        ));
        let mut fresh: Vec<&Value> = hooks.iter().filter(|h| h.get("source").and_then(Value::as_str) != Some("built in")).collect();
        fresh.sort_by(|a, b| {
            let k = |h: &Value| (h.get("added").and_then(Value::as_str).unwrap_or_default().to_string(), h.get("hot").and_then(Value::as_f64).unwrap_or(0.0));
            let (ka, kb) = (k(a), k(b));
            kb.0.cmp(&ka.0).then(kb.1.partial_cmp(&ka.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        if !fresh.is_empty() {
            egui::CollapsingHeader::new("Newest hooks").id_salt(format!("{p}.newest")).show(ui, |ui| {
                for h in fresh.iter().take(12) {
                    let text = h.get("text").and_then(Value::as_str).unwrap_or_default();
                    let niche = h.get("niche").and_then(Value::as_str).unwrap_or_default();
                    let trend = h.get("trend").and_then(Value::as_str).unwrap_or_default();
                    let r = ui.label(format!("• {text}  [{niche}]"));
                    if !trend.is_empty() {
                        r.on_hover_text(format!("rides: {trend}"));
                    }
                }
            });
        }
        if let Some(formats) = data.get("formats").and_then(Value::as_array).filter(|f| !f.is_empty()) {
            egui::CollapsingHeader::new("Hook formats trending this month").id_salt(format!("{p}.formats")).show(ui, |ui| {
                for f in formats.iter().filter_map(Value::as_str).take(8) {
                    ui.label(format!("• {f}"));
                }
            });
        }
    });
    ui.end_row();
    ui.label("Niches");
    let r = ui.add(egui::TextEdit::singleline(&mut d.hook_niches).hint_text("gaming, travel, tech …").desired_width(380.0));
    elems.push((format!("{p}.niches"), r.rect, d.hook_niches.clone()));
    if r.changed() {
        let _ = save_setting("hook_niches", d.hook_niches.trim());
    }
    ui.end_row();
    ui.label("Countries");
    let r = ui.add(egui::TextEdit::singleline(&mut d.hook_geos).hint_text("IN, US (trending searches)").desired_width(380.0));
    elems.push((format!("{p}.countries"), r.rect, d.hook_geos.clone()));
    if r.changed() {
        let _ = save_setting("hook_geos", d.hook_geos.trim());
    }
    ui.end_row();
    ui.label("");
    let r = ui.checkbox(&mut d.hook_claude, "Claude writes new hooks from the trends (with web search); off: only the top video titles");
    elems.push((format!("{p}.claude"), r.rect, d.hook_claude.to_string()));
    ui.end_row();
}

/// The optional sound design rows. Returns true when the music folder should be chosen.
fn sound_rows(ui: &mut egui::Ui, d: &mut Draft, p: &str, elems: &mut Vec<(String, egui::Rect, String)>) -> bool {
    let mut choose = false;
    ui.label("Sound design");
    let label = if d.kind == Kind::Music {
        "Add sound effects synced to the song (a riser into the drop, an impact on it, whooshes on fast cuts)"
    } else {
        "Score the emotions: music that changes with the mood, hits, risers, whooshes, silence before reveals"
    };
    let r = ui.checkbox(&mut d.sound, label);
    elems.push((format!("{p}.sound"), r.rect, d.sound.to_string()));
    ui.end_row();
    if !d.sound || d.kind == Kind::Music {
        return false;
    }
    ui.label("Music from");
    combo_row(ui, &mut d.music_source, &MUSIC_SOURCES, &format!("{p}.musicSource"), elems);
    ui.end_row();
    if d.music_source == "folder" {
        ui.label("Music folder");
        choose = path_row(ui, &mut d.music_folder, &format!("{p}.musicFolder"), "your songs and scores", elems);
        ui.end_row();
    }
    if d.music_source == "internet" {
        ui.label("Licence");
        combo_row(ui, &mut d.licence, &LICENCES, &format!("{p}.licence"), elems);
        ui.end_row();
    }
    ui.label("Music level");
    let r = ui.add(egui::Slider::new(&mut d.music_level, -24.0..=0.0).suffix(" dB").step_by(1.0));
    elems.push((format!("{p}.musicLevel"), r.rect, d.music_level.to_string()));
    ui.end_row();
    ui.label("");
    let r = ui.checkbox(&mut d.sound_effects, "Sound effects (impacts, risers, whooshes, shimmer, heartbeat)");
    elems.push((format!("{p}.soundEffects"), r.rect, d.sound_effects.to_string()));
    ui.end_row();
    choose
}

/// Emotions per line of the script (Speak, Voiceover): chosen by Claude when the script was
/// written; change any of them here (saved into the script file at once).
fn emotion_rows(ui: &mut egui::Ui, d: &mut Draft, p: &str, elems: &mut Vec<(String, egui::Rect, String)>) {
    let path = d.script.trim().to_string();
    let lower = path.to_lowercase();
    let usable = (lower.ends_with(".script.json") || lower.ends_with(".txt")) && Path::new(&path).is_file();
    if !usable {
        return;
    }
    if d.emo.path != path {
        d.emo = EmotionEditor::load(&path);
    }
    ui.label("Emotions");
    let mut changed = false;
    ui.vertical(|ui| {
        let id = format!("{p}.emotions");
        egui::CollapsingHeader::new(format!("{} lines: change the emotion of any line", d.emo.lines.len())).id_salt(&id).show(ui, |ui| {
            egui::ScrollArea::vertical().max_height(260.0).id_salt(format!("{id}.scroll")).show(ui, |ui| {
                egui::Grid::new(format!("{id}.grid")).num_columns(3).spacing([8.0, 4.0]).show(ui, |ui| {
                    for (k, line) in d.emo.lines.iter_mut().enumerate() {
                        let short: String = line.text.chars().take(46).collect();
                        let more = if line.text.chars().count() > 46 { "…" } else { "" };
                        ui.label(format!("{short}{more}")).on_hover_text(&line.text);
                        let before = line.emotion.clone();
                        let r = egui::ComboBox::from_id_salt(format!("{id}.{k}")).selected_text(line.emotion.clone()).width(120.0).show_ui(ui, |ui| {
                            for n in EMOTION_NAMES {
                                ui.selectable_value(&mut line.emotion, n.to_string(), n);
                            }
                        });
                        elems.push((format!("{id}.{k}"), r.response.rect, line.emotion.clone()));
                        let r2 = ui.add(egui::DragValue::new(&mut line.intensity).range(0.0..=1.0).speed(0.01).max_decimals(2));
                        elems.push((format!("{id}.{k}.intensity"), r2.rect, line.intensity.to_string()));
                        changed |= line.emotion != before || r2.changed();
                        ui.end_row();
                    }
                });
            });
        });
        if !d.emo.error.is_empty() {
            ui.colored_label(egui::Color32::from_rgb(220, 90, 80), &d.emo.error);
        }
    });
    ui.end_row();
    if changed {
        d.emo.save();
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
    let mut add_to_queue = false;
    let mut save_edit = false;
    let mut pick: Option<Pick> = None;
    let mut learn_style = false;
    let mut elems: Vec<(String, egui::Rect, String)> = Vec::new();
    let title = if d.queue_edit.is_some() { format!("{} (queued task)", d.kind.title()) } else { d.kind.title().to_string() };
    egui::Window::new(title)
        .id(egui::Id::new(("ei-ai-dialog", d.kind.id())))
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.set_min_width(520.0);
            ui.label(egui::RichText::new(description(d.kind)).weak());
            ui.add_space(6.0);
            // the dialog never grows past the screen: its content scrolls, the buttons stay visible
            let max_h = (ui.ctx().content_rect().height() - 190.0).max(240.0);
            egui::ScrollArea::vertical().max_height(max_h).auto_shrink([true, true]).show(ui, |ui| {
                egui::Grid::new("ei-ai-grid").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                    if d.kind.needs_input() {
                        ui.label(d.kind.input_label());
                        if path_row(ui, &mut d.input, &format!("{p}.input"), "a file", &mut elems) {
                            pick = Some(Pick::Input);
                        }
                        ui.end_row();
                    }
                    if d.kind.needs_clips()
                        || d.kind == Kind::Script
                        || (d.kind == Kind::Voice && d.source == "files")
                        || (d.kind == Kind::Topic && d.make_video)
                    {
                        ui.label(match d.kind {
                            Kind::Audio => "Videos",
                            Kind::Script | Kind::Topic => "Your footage (optional)",
                            Kind::Voice => "Your recordings",
                            Kind::Voiceover => "Footage",
                            Kind::Tv => "Your clips",
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
                    let scored = d.sound && d.kind != Kind::Music; // the sound design brings its own music
                    if matches!(d.kind, Kind::Music | Kind::Talk | Kind::Reference)
                        || (d.kind == Kind::Voiceover && !scored)
                        || (d.kind == Kind::Clips && d.clip_type != "sections" && !scored)
                        || (d.kind == Kind::Topic && d.make_video && !scored)
                    {
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
                            let r =
                                ui.add(egui::TextEdit::singleline(&mut d.music_start).hint_text("automatic: the best part of the song").desired_width(200.0));
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
                            ui.label("Clip type");
                            let before = d.clip_type;
                            radio_row(ui, &mut d.clip_type, &CLIP_TYPES, &format!("{p}.clipType"), &mut elems);
                            if d.clip_type != before {
                                // switch the lengths too, unless they were changed by hand
                                let (from, to) = if d.clip_type == "sections" { (EDITED_LENGTHS, SECTION_LENGTHS) } else { (SECTION_LENGTHS, EDITED_LENGTHS) };
                                if (d.min_len, d.target_len, d.max_len) == from {
                                    (d.min_len, d.target_len, d.max_len) = to;
                                }
                            }
                            ui.end_row();
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
                            if d.clip_type != "sections" {
                                ui.label("Pace");
                                radio_row(ui, &mut d.pace, &PACES, &format!("{p}.pace"), &mut elems);
                                ui.end_row();
                            }
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
                            if d.clip_type != "sections" {
                                ui.label("");
                                ui.vertical(|ui| {
                                    let r = ui.checkbox(&mut d.hook, "Open each clip with a teaser of its best moment");
                                    elems.push((format!("{p}.hook"), r.rect, d.hook.to_string()));
                                });
                                ui.end_row();
                            }
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
                            ui.label("Hook");
                            combo_row(ui, &mut d.vo_hook, &HOOK_MODES, &format!("{p}.hook"), &mut elems);
                            ui.end_row();
                            if d.vo_hook != "off" {
                                ui.label("Niche");
                                let r = ui.add(egui::TextEdit::singleline(&mut d.niche).hint_text("e.g. travel (helps pick hooks)").desired_width(330.0));
                                elems.push((format!("{p}.niche"), r.rect, d.niche.clone()));
                                ui.end_row();
                            }
                        }
                        Kind::Hooks => hooks_rows(ui, &mut d, &p, &mut elems),
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
                        Kind::Tv => {
                            ui.label("Recordings folder");
                            if path_row(ui, &mut d.tv_folder, &format!("{p}.recordings"), "videos of your room with the TV", &mut elems) {
                                pick = Some(Pick::TvFolder);
                            }
                            ui.end_row();
                            ui.label("Final video");
                            radio_row(ui, &mut d.aspect, &TV_SHAPES, &format!("{p}.aspect"), &mut elems);
                            ui.end_row();
                            ui.label("Backlight matching");
                            let r = ui.add(egui::Slider::new(&mut d.relight, 0.0..=1.0).text("recolour the glow"));
                            elems.push((format!("{p}.relight"), r.rect, d.relight.to_string()));
                            ui.end_row();
                            ui.label("TV bezel");
                            let r = ui
                                .checkbox(&mut d.tv_bezel, "Draw a thin bezel around the clip (wider at the bottom); place the corners on the TV's outer edge");
                            elems.push((format!("{p}.bezel"), r.rect, d.tv_bezel.to_string()));
                            ui.end_row();
                            ui.label("Recordings");
                            ui.vertical(|ui| tv_recordings_ui(ui, &mut d, &p, &mut elems));
                            ui.end_row();
                        }
                        Kind::Topic => {
                            ui.label("Topic or link");
                            let r = ui.add(
                                egui::TextEdit::multiline(&mut d.topic)
                                    .hint_text("e.g. GTA 6 release, or a link to a post/article")
                                    .desired_rows(2)
                                    .desired_width(380.0),
                            );
                            elems.push((format!("{p}.topic"), r.rect, d.topic.clone()));
                            ui.end_row();
                            ui.label("Length (seconds)");
                            ui.horizontal(|ui| {
                                let r = ui.add(egui::DragValue::new(&mut d.seconds).range(5.0..=1800.0).speed(1.0).max_decimals(0));
                                elems.push((format!("{p}.seconds"), r.rect, d.seconds.to_string()));
                                for (label, v) in [("Short 30 s", 30.0), ("60 s", 60.0), ("3 min", 180.0), ("8 min", 480.0)] {
                                    let b = ui.small_button(label);
                                    elems.push((format!("{p}.seconds.{}", v as u32), b.rect, label.into()));
                                    if b.clicked() {
                                        d.seconds = v;
                                    }
                                }
                            });
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
                            ui.label("Research & writing");
                            combo_row(ui, &mut d.writer, &WRITERS, &format!("{p}.writer"), &mut elems);
                            ui.end_row();
                            ui.label("");
                            ui.vertical(|ui| {
                                let r = ui.checkbox(&mut d.make_video, "Make the video (untick for the script and sources only)");
                                elems.push((format!("{p}.video"), r.rect, d.make_video.to_string()));
                                let r = ui.checkbox(&mut d.fresh, "Research again (otherwise the last research of this topic is reused)");
                                elems.push((format!("{p}.fresh"), r.rect, d.fresh.to_string()));
                            });
                            ui.end_row();
                            ui.label("Hook (optional)");
                            ui.vertical(|ui| {
                                let r = ui.add(
                                    egui::TextEdit::singleline(&mut d.hook_line).hint_text("your own first line, spoken word for word").desired_width(380.0),
                                );
                                elems.push((format!("{p}.hookLine"), r.rect, d.hook_line.clone()));
                                ui.add_enabled_ui(d.hook_line.trim().is_empty(), |ui| {
                                    let r = ui.checkbox(&mut d.find_hook, "Search online for a hook for this topic (its top videos, Claude with web search)");
                                    elems.push((format!("{p}.findHook"), r.rect, d.find_hook.to_string()));
                                });
                                if d.hook_line.trim().is_empty() {
                                    ui.label(egui::RichText::new("Empty: the hook is written from your hook library.").small().weak());
                                } else {
                                    ui.label(egui::RichText::new("The video opens with your line (no cold open before it).").small().weak());
                                }
                            });
                            ui.end_row();
                            if d.make_video {
                                ui.label("Voiceover");
                                let mut voices: Vec<(String, String)> = saved_voices().into_iter().map(|v| (v.clone(), format!("{v} (your clone)"))).collect();
                                for (k, l) in MAC_VOICES {
                                    voices.push((k.to_string(), l.to_string()));
                                }
                                let shown = voices.iter().find(|v| v.0 == d.voice).map(|v| v.1.clone()).unwrap_or_else(|| "Choose a voice…".into());
                                let r = egui::ComboBox::from_id_salt(format!("{p}.voice")).selected_text(shown.clone()).width(330.0).show_ui(ui, |ui| {
                                    for (k, l) in &voices {
                                        let o = ui.selectable_value(&mut d.voice, k.clone(), l);
                                        elems.push((format!("{p}.voice.{k}"), o.rect, l.clone()));
                                    }
                                });
                                elems.push((format!("{p}.voice"), r.response.rect, shown));
                                ui.end_row();
                                if !d.voice.starts_with("mac") && !d.voice.is_empty() {
                                    ui.label("Takes per line");
                                    let r = ui.add(egui::DragValue::new(&mut d.takes).range(1.0..=4.0).speed(0.05).max_decimals(0));
                                    elems.push((format!("{p}.takes"), r.rect, d.takes.to_string()));
                                    ui.end_row();
                                }
                                ui.label("Shape");
                                radio_row(ui, &mut d.aspect, &ASPECTS[1..], &format!("{p}.aspect"), &mut elems);
                                ui.end_row();
                                ui.label("Pace");
                                radio_row(ui, &mut d.pace, &PACES, &format!("{p}.pace"), &mut elems);
                                ui.end_row();
                                ui.label("Pictures from");
                                ui.vertical(|ui| {
                                    let r = ui.checkbox(&mut d.src_free, "Free-licensed photos and videos (Wikimedia Commons, Openverse; credited)");
                                    elems.push((format!("{p}.free"), r.rect, d.src_free.to_string()));
                                    let r = ui.checkbox(&mut d.src_cards, "Headline and fact cards from the research");
                                    elems.push((format!("{p}.cards"), r.rect, d.src_cards.to_string()));
                                    let r = ui.checkbox(&mut d.src_youtube, "Short clips from YouTube videos the research finds");
                                    elems.push((format!("{p}.youtube"), r.rect, d.src_youtube.to_string()));
                                    if d.src_youtube {
                                        ui.label(
                                            egui::RichText::new(
                                                "Only where you have the right to use them (e.g. commentary or news). Every source is listed in sources.md.",
                                            )
                                            .weak(),
                                        );
                                    }
                                });
                                ui.end_row();
                                ui.label("");
                                let r = ui.checkbox(&mut d.captions, "Captions");
                                elems.push((format!("{p}.captions"), r.rect, d.captions.to_string()));
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
                                let r = ui.checkbox(&mut d.lang_hi, "Hindi");
                                elems.push((format!("{p}.lang.hi"), r.rect, d.lang_hi.to_string()));
                                let r = ui.checkbox(&mut d.lang_mix, "Hinglish");
                                elems.push((format!("{p}.lang.mix"), r.rect, d.lang_mix.to_string()));
                                let r = ui.checkbox(&mut d.lang_en, "English");
                                elems.push((format!("{p}.lang.en"), r.rect, d.lang_en.to_string()));
                            });
                            ui.end_row();
                            ui.label("Source");
                            radio_row(ui, &mut d.source, &VOICE_SOURCES, &format!("{p}.source"), &mut elems);
                            ui.end_row();
                        }
                        Kind::Speak => {
                            ui.label("Script");
                            if path_row(ui, &mut d.script, &format!("{p}.script"), "a .script.json or .txt (from Write Voiceover Script)", &mut elems) {
                                pick = Some(Pick::Script);
                            }
                            ui.end_row();
                            if d.script.trim().is_empty() {
                                ui.label("Or the text");
                                let r = ui.add(
                                    egui::TextEdit::multiline(&mut d.notes)
                                        .hint_text("[Awe 0.8] Twelve years. [Triumph 0.9] It's finally here!")
                                        .desired_rows(4)
                                        .desired_width(330.0),
                                );
                                elems.push((format!("{p}.text"), r.rect, String::new()));
                                ui.end_row();
                            }
                            ui.label("Voice");
                            let mut voices: Vec<(String, String)> = saved_voices().into_iter().map(|v| (v.clone(), format!("{v} (your clone)"))).collect();
                            for (k, l) in MAC_VOICES {
                                voices.push((k.to_string(), l.to_string()));
                            }
                            let shown = voices.iter().find(|v| v.0 == d.voice).map(|v| v.1.clone()).unwrap_or_else(|| "Choose a voice…".into());
                            let r = egui::ComboBox::from_id_salt(format!("{p}.voice")).selected_text(shown.clone()).width(330.0).show_ui(ui, |ui| {
                                for (k, l) in &voices {
                                    let o = ui.selectable_value(&mut d.voice, k.clone(), l);
                                    elems.push((format!("{p}.voice.{k}"), o.rect, l.clone()));
                                }
                            });
                            elems.push((format!("{p}.voice"), r.response.rect, shown));
                            ui.end_row();
                            if !d.voice.starts_with("mac") && !d.voice.is_empty() {
                                ui.label("Takes per line");
                                let r = ui.add(egui::DragValue::new(&mut d.takes).range(1.0..=4.0).speed(0.05).max_decimals(0));
                                elems.push((format!("{p}.takes"), r.rect, d.takes.to_string()));
                                ui.end_row();
                            }
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
                    if matches!(d.kind, Kind::Speak | Kind::Voiceover) {
                        emotion_rows(ui, &mut d, &p, &mut elems);
                    }
                    if d.kind.has_resolution() && !(d.kind == Kind::Topic && !d.make_video) {
                        ui.label("Resolution");
                        radio_row(ui, &mut d.resolution, &RESOLUTIONS, &format!("{p}.resolution"), &mut elems);
                        ui.end_row();
                    }
                    if d.kind.has_sound_design() && !(d.kind == Kind::Topic && !d.make_video) && sound_rows(ui, &mut d, &p, &mut elems) {
                        pick = Some(Pick::MusicFolder);
                    }
                    if !matches!(d.kind, Kind::Voice | Kind::Hooks) {
                        let folder_out = matches!(d.kind, Kind::Split | Kind::Clips | Kind::Audio | Kind::Topic | Kind::Tv);
                        ui.label(if folder_out { "Output folder" } else { "Save as" });
                        let hint = match d.kind {
                            Kind::Script => "automatic (Movies/EI/Scripts)",
                            Kind::Topic => "automatic (Movies/EI/Topics/<topic>)",
                            Kind::Tv => "automatic (“EI on TV” next to the clips)",
                            _ => "automatic (next to the source)",
                        };
                        if path_row(ui, &mut d.out, &format!("{p}.out"), hint, &mut elems) {
                            pick = Some(Pick::Out);
                        }
                        ui.end_row();
                    }
                    if !matches!(d.kind, Kind::Script | Kind::Voice | Kind::Topic | Kind::Hooks) {
                        ui.label("");
                        let label = match d.kind {
                            Kind::Split | Kind::Clips | Kind::Tv => "Import the videos into the project",
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
                let run_label = match d.kind {
                    Kind::Voice => "Create Voice",
                    Kind::Script if !d.voice.is_empty() => "Write and Speak",
                    Kind::Script => "Write Script",
                    Kind::Topic if d.make_video => "Make Video",
                    Kind::Tv => "Make Video",
                    Kind::Topic => "Research and Write",
                    Kind::Hooks => "Sync Hooks",
                    Kind::Speak => "Speak",
                    _ => "Run",
                };
                if d.queue_edit.is_some() {
                    // editing a waiting task: save it back into its place in the queue
                    let o = ui.add(egui::Button::new(egui::RichText::new("Save Changes").color(egui::Color32::WHITE)).fill(app.tokens.accent));
                    elems.push((format!("{p}.saveQueued"), o.rect, "Save Changes".into()));
                    if o.clicked() {
                        save_edit = true;
                    }
                } else {
                    if d.kind != Kind::Voice {
                        let q = ui.button("Add to Queue").on_hover_text("Wait in the queue (AI ▸ AI Jobs) and run after the tasks before it");
                        elems.push((format!("{p}.queue"), q.rect, "Add to Queue".into()));
                        if q.clicked() {
                            add_to_queue = true;
                        }
                    }
                    let o = ui.add(egui::Button::new(egui::RichText::new(run_label).color(egui::Color32::WHITE)).fill(app.tokens.accent));
                    elems.push((format!("{p}.run"), o.rect, "Run".into()));
                    if o.clicked() {
                        run = true;
                    }
                }
            });
        });
    if d.kind == Kind::Tv && d.tv.calib.is_some() {
        // the corner check: its own window on top, everything visible at once
        egui::Window::new("Check TV Corners")
            .id(egui::Id::new("ei-ai-tv-corners"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .order(egui::Order::Foreground)
            .show(ctx, |ui| tv_calib_ui(ui, &mut d, &p, &mut elems));
    }
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
            Pick::MusicFolder => {
                if let Some(dir) = app.hooks.pick_folder.as_mut().and_then(|f| f()) {
                    let _ = save_setting("music_folder", &dir);
                    d.music_folder = dir;
                }
            }
            Pick::TvFolder => {
                if let Some(dir) = app.hooks.pick_folder.as_mut().and_then(|f| f()) {
                    d.tv_folder = dir;
                    d.tv.recs.clear();
                    d.tv.scanned.clear();
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
                if matches!(d.kind, Kind::Split | Kind::Clips | Kind::Audio | Kind::Topic) {
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
    poll_tv(&mut d);
    if run && d.kind == Kind::Tv && !d.tv_folder.trim().is_empty() {
        let _ = save_setting("tv_recordings", d.tv_folder.trim());
    }
    if add_to_queue {
        match queue_draft(&mut d) {
            Ok(n) => {
                app.ui.status = format!("Added to the queue ({n} waiting; AI ▸ AI Jobs shows it)");
                ctx.data_mut(|m| m.insert_temp(jobs_open_id(), true));
                d.error.clear();
                if d.kind == Kind::Topic {
                    d.topic.clear(); // ready for the next topic
                    d.hook_line.clear();
                }
                d.out.clear(); // (a fresh automatic output name for the next one)
            }
            Err(e) => d.error = e,
        }
    }
    if save_edit && let Some(uid) = d.queue_edit {
        match save_queue_edit(&mut d, uid) {
            Ok(()) => {
                app.ui.status = "Queued task updated".into();
                ctx.data_mut(|m| m.insert_temp(jobs_open_id(), true));
                close = true;
            }
            Err(e) => d.error = e,
        }
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
    let langs = selected_sets(d);
    if !d.rec.prompts.is_empty() && d.rec.loaded != langs && d.rec.recording.is_none() {
        // languages changed: load the matching sentences (takes on disk are found again)
        d.rec.prompts.clear();
        d.rec.takes.clear();
        d.rec.cursor = 0;
    }
    if d.rec.prompts.is_empty() && d.rec.loading.is_none() && !langs.is_empty() {
        let mut args = vec!["voice-prompts".to_string(), "--languages".into()];
        args.extend(langs.iter().cloned());
        match engine_step(args) {
            Ok((step, engine)) => {
                d.rec.loaded = langs.clone();
                d.rec.loading = Some(start_job("Voice clone: sentences".into(), None, vec![step], false, Some(engine)));
            }
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
    ui.horizontal(|ui| {
        for (set, label) in [("hi", "हिन्दी"), ("mix", "Hinglish"), ("en", "English")] {
            let idx: Vec<usize> = d.rec.prompts.iter().enumerate().filter(|(_, q)| q.set == set).map(|(i, _)| i).collect();
            let Some(&first) = idx.first() else { continue };
            let got = idx.iter().filter(|&&i| d.rec.takes.get(i).is_some_and(Option::is_some)).count();
            let here = d.rec.prompts.get(d.rec.cursor).is_some_and(|q| q.set == set);
            let text = format!("{label} {got}/{}", idx.len());
            let b = ui.add_enabled(d.rec.recording.is_none(), egui::Button::new(text.clone()).selected(here));
            elems.push((format!("{p}.section.{set}"), b.rect, text));
            if b.clicked() {
                // jump to the first sentence of the section not yet recorded
                d.rec.cursor = idx.iter().copied().find(|&i| d.rec.takes.get(i).is_none_or(Option::is_none)).unwrap_or(first);
            }
        }
    });
    let Some(pr) = d.rec.prompts.get(d.rec.cursor).cloned() else { return };
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(match pr.set.as_str() {
                "mix" => "Hinglish",
                "hi" => "हिन्दी",
                _ => "English",
            })
            .strong(),
        );
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
                d.rec.message = "Give the voice a name first (top of this window).".into();
            } else {
                let takes_dir = voices_dir().join(slug(&d.voice_name)).join("takes");
                let file = takes_dir.join(format!("{}.wav", pr.id));
                let max = if pr.free { "75" } else { "30" };
                let args = vec![
                    "voice-record".into(),
                    "--out".into(),
                    file.to_string_lossy().to_string(),
                    "--mic".into(),
                    d.rec.mic.clone(),
                    "--max".into(),
                    max.into(),
                    "--stop-file".into(),
                    takes_dir.join(".stop").to_string_lossy().to_string(),
                ];
                match engine_step(args) {
                    Ok((step, engine)) => {
                        d.rec.message.clear();
                        d.rec.recording = Some(start_job(format!("Recording {}", pr.id), None, vec![step], false, Some(engine)));
                    }
                    Err(e) => d.rec.message = e,
                }
            }
        }
        if recording {
            let s = ui.button("■ Stop");
            elems.push((format!("{p}.stop"), s.rect, "Stop".into()));
            if s.clicked() {
                // the engine finishes the take and keeps it when this file appears
                let stop = voices_dir().join(slug(&d.voice_name)).join("takes").join(".stop");
                if std::fs::write(&stop, b"stop").is_err()
                    && let Some(j) = d.rec.recording.take()
                {
                    cancel_job(j);
                }
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
    if let Some(j) = d.rec.recording {
        // live status from the engine: "listening (-52 dB)", "hearing you (-28 dB)"
        let last =
            with_job(j, |job| job.log.iter().rev().find(|l| !l.starts_with("$ ")).cloned()).flatten().unwrap_or_else(|| "starting the microphone…".into());
        ui.label(egui::RichText::new(format!("🎙 {last}")).strong());
        ui.ctx().request_repaint_after(Duration::from_millis(150));
        return;
    }
    if !d.rec.message.is_empty() {
        ui.colored_label(egui::Color32::from_rgb(0xE5, 0x6B, 0x6B), &d.rec.message);
    }
    match &take {
        Some(t) if !t.issues.is_empty() => {
            let len = if t.seconds > 0.0 { format!(" ({:.1} s)", t.seconds) } else { String::new() };
            ui.colored_label(egui::Color32::from_rgb(0xE0, 0xB0, 0x40), format!("✔ Saved{len}, but: {}. Play it, then Record Again or Next.", t.issues));
        }
        Some(t) => {
            let len = if t.seconds > 0.0 { format!(" ({:.1} s)", t.seconds) } else { String::new() };
            let l = ui.colored_label(egui::Color32::from_rgb(0x5C, 0xC0, 0x7A), format!("✔ Saved{len}. Play it to check, then Next ›"));
            elems.push((format!("{p}.saved"), l.rect, "saved".into()));
        }
        None => {
            ui.label(egui::RichText::new("A quiet room, 15–20 cm from the microphone, and say it the way you talk in your videos.").weak());
        }
    }
}

/// The sentence sets to read, in reading order (Hindi, Hinglish, English).
fn selected_sets(d: &Draft) -> Vec<String> {
    [("hi", d.lang_hi), ("mix", d.lang_mix), ("en", d.lang_en)].iter().filter(|(_, on)| *on).map(|(k, _)| k.to_string()).collect()
}

/// The recordings of the room for TV Light Sync: orientation and whether the TV corners were checked.
fn tv_recordings_ui(ui: &mut egui::Ui, d: &mut Draft, p: &str, elems: &mut Vec<(String, egui::Rect, String)>) {
    let folder = d.tv_folder.trim().to_string();
    if folder.is_empty() || !Path::new(&folder).is_dir() {
        ui.label(egui::RichText::new("Choose the folder above.").weak());
        return;
    }
    if d.tv.scanned != folder && d.tv.scan_job.is_none() {
        match engine_step(vec!["tv-scan".into(), folder.clone()]) {
            Ok((step, engine)) => {
                d.tv.scan_job = Some(start_job("TV Light Sync: recordings".into(), None, vec![step], false, Some(engine)));
                d.tv.scanned = folder.clone();
            }
            Err(e) => d.error = e,
        }
    }
    if d.tv.scan_job.is_some() {
        ui.label("Looking at the recordings…");
        return;
    }
    let want = if d.aspect == "vertical" { "vertical" } else { "horizontal" };
    let mut open: Option<String> = None;
    let busy = d.tv.calib.is_some();
    for r in d.tv.recs.iter().filter(|r| r.orientation == want) {
        let name = Path::new(&r.path).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        ui.horizontal(|ui| {
            ui.label(&name);
            if r.calibrated {
                ui.colored_label(egui::Color32::from_rgb(0x5C, 0xC0, 0x7A), "✔ corners checked");
            } else {
                ui.colored_label(egui::Color32::from_rgb(0xE0, 0xB0, 0x40), "corners not checked yet");
            }
            let b = ui.add_enabled(!busy, egui::Button::new(if r.calibrated { "Re-check" } else { "Check corners" }).small());
            elems.push((format!("{p}.calibrate.{name}"), b.rect, name.clone()));
            if b.clicked() {
                open = Some(r.path.clone());
            }
        });
    }
    let other = d.tv.recs.iter().filter(|r| r.orientation != want).count();
    if d.tv.recs.iter().all(|r| r.orientation != want) {
        ui.colored_label(egui::Color32::from_rgb(0xE5, 0x6B, 0x6B), format!("No {want} recordings in this folder."));
    } else if other > 0 {
        ui.label(
            egui::RichText::new(format!("({other} {} recording(s) are used for the other shape)", if want == "vertical" { "horizontal" } else { "vertical" }))
                .weak(),
        );
    }
    ui.label(egui::RichText::new("Unchecked recordings use EI's own guess of the TV corners; checking them once (10 s each) gives a perfect fit.").weak());
    if let Some(path) = open {
        let png = support_dir().join("tvlight").join(format!("{}.png", file_stem(&path)));
        let _ = std::fs::create_dir_all(support_dir().join("tvlight"));
        match engine_step(vec!["tv-frame".into(), path.clone(), "--out".into(), png.to_string_lossy().to_string()]) {
            Ok((step, engine)) => {
                let job = start_job(format!("TV Light Sync: frame of {}", file_stem(&path)), None, vec![step], false, Some(engine));
                d.tv.calib = Some(TvCalib { path, job: Some(job), ..TvCalib::default() });
            }
            Err(e) => d.error = e,
        }
    }
}

/// The corner check: a frame of the recording with four draggable corners on the TV screen.
fn tv_calib_ui(ui: &mut egui::Ui, d: &mut Draft, p: &str, elems: &mut Vec<(String, egui::Rect, String)>) {
    let Some(c) = d.tv.calib.as_mut() else { return };
    ui.strong(format!("TV corners: {}", Path::new(&c.path).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()));
    if !c.loaded {
        ui.label("Finding the TV in this recording (EI suggests the corners)…");
        return;
    }
    ui.label(
        egui::RichText::new(
            "Drag the four green corners onto the corners of the TV's picture (inside the black frame); arrow keys nudge the last corner you touched. Check other moments with the slider: if the outline drifts there, fix it at that moment and save.",
        )
        .weak(),
    );
    let mut load_at: Option<f64> = None;
    ui.horizontal(|ui| {
        ui.label("Moment");
        let dur = c.duration.max(1.0);
        let r = ui.add(egui::Slider::new(&mut c.at, 0.0..=dur).suffix(" s").max_decimals(1));
        elems.push((format!("{p}.moment"), r.rect, format!("{:.1}", c.at)));
        let b = ui.add_enabled(c.job.is_none(), egui::Button::new("Show This Moment"));
        elems.push((format!("{p}.moment.show"), b.rect, "Show This Moment".into()));
        if b.clicked() {
            load_at = Some(c.at);
        }
        ui.weak(match c.source.as_str() {
            "tracked" => "corners: where EI tracked the TV here",
            "saved" => "corners: your saved ones",
            _ => "corners: EI's suggestion",
        });
    });
    if let Some(t) = load_at {
        let png = support_dir().join("tvlight").join(format!("{}-{:.0}.png", file_stem(&c.path), t * 10.0));
        match engine_step(vec!["tv-frame".into(), c.path.clone(), "--out".into(), png.to_string_lossy().to_string(), "--t".into(), format!("{t:.2}")]) {
            Ok((step, engine)) => {
                c.job = Some(start_job(format!("TV Light Sync: moment {t:.0}s"), None, vec![step], false, Some(engine)));
                c.loaded = false;
            }
            Err(e) => d.error = e,
        }
        return;
    }
    let tex_id = egui::Id::new(("ei-tv-calib-tex", c.image.clone()));
    let tex = ui.ctx().data(|m| m.get_temp::<egui::TextureHandle>(tex_id));
    let tex = match tex {
        Some(t) => t,
        None => {
            let Ok(img) = image::open(&c.image) else {
                ui.colored_label(egui::Color32::from_rgb(0xE5, 0x6B, 0x6B), "Could not open the frame.");
                return;
            };
            let rgba = img.to_rgba8();
            let size = [rgba.width() as usize, rgba.height() as usize];
            let t = ui.ctx().load_texture("ei-tv-calib", egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw()), egui::TextureOptions::LINEAR);
            ui.ctx().data_mut(|m| m.insert_temp(tex_id, t.clone()));
            t
        }
    };
    let [tw, th] = tex.size();
    // sized to the screen: the corners must be reachable on a laptop screen too
    let max = egui::vec2(720.0, (ui.ctx().content_rect().height() - 300.0).clamp(220.0, 560.0));
    let scale = (max.x / tw.max(1) as f32).min(max.y / th.max(1) as f32);
    let size = egui::vec2(tw as f32 * scale, th as f32 * scale);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    ui.painter().image(tex.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
    let to_screen = |q: [f32; 2]| rect.min + egui::vec2(q[0] * size.x, q[1] * size.y);
    let pts: Vec<egui::Pos2> = c.quad.iter().map(|q| to_screen(*q)).collect();
    let green = egui::Color32::from_rgb(0x4C, 0xE0, 0x6C);
    for k in 0..4 {
        if let (Some(a), Some(b)) = (pts.get(k), pts.get((k + 1) % 4)) {
            ui.painter().line_segment([*a, *b], egui::Stroke::new(2.0, green));
        }
    }
    for (k, label) in ["top-left", "top-right", "bottom-right", "bottom-left"].iter().enumerate() {
        let Some(pt) = pts.get(k).copied() else { continue };
        let hr = egui::Rect::from_center_size(pt, egui::vec2(18.0, 18.0));
        let r = ui.interact(hr, egui::Id::new(("ei-tv-corner", k)), egui::Sense::drag());
        if r.dragged()
            && let Some(q) = c.quad.get_mut(k)
        {
            let dlt = r.drag_delta();
            q[0] = (q[0] + dlt.x / size.x.max(1.0)).clamp(-0.2, 1.2);
            q[1] = (q[1] + dlt.y / size.y.max(1.0)).clamp(-0.2, 1.2);
        }
        if r.dragged() || r.clicked() || r.drag_started() {
            c.active = Some(k);
        }
        let col = if r.hovered() || r.dragged() || c.active == Some(k) { egui::Color32::WHITE } else { green };
        ui.painter().circle_stroke(pt, 7.0, egui::Stroke::new(2.0, col));
        ui.painter().circle_filled(pt, 2.0, col);
        elems.push((format!("{p}.corner.{k}"), hr, (*label).to_string()));
    }
    // arrow keys nudge the active corner by one pixel of the frame
    if let Some(k) = c.active
        && let Some(q) = c.quad.get_mut(k)
    {
        let (dx, dy) = ui.input(|i| {
            let mut v = (0.0f32, 0.0f32);
            if i.key_pressed(egui::Key::ArrowLeft) {
                v.0 -= 1.0;
            }
            if i.key_pressed(egui::Key::ArrowRight) {
                v.0 += 1.0;
            }
            if i.key_pressed(egui::Key::ArrowUp) {
                v.1 -= 1.0;
            }
            if i.key_pressed(egui::Key::ArrowDown) {
                v.1 += 1.0;
            }
            v
        });
        q[0] = (q[0] + dx / tw.max(1) as f32).clamp(-0.2, 1.2);
        q[1] = (q[1] + dy / th.max(1) as f32).clamp(-0.2, 1.2);
        // magnifier: the area around the active corner, enlarged, in the image's top-right
        let zoom = 0.05f32;
        let aspect = tw as f32 / th.max(1) as f32;
        let (cx, cy) = (q[0].clamp(0.0, 1.0), q[1].clamp(0.0, 1.0));
        let uv = egui::Rect::from_min_max(egui::pos2(cx - zoom * 0.5, cy - zoom * 0.5 * aspect), egui::pos2(cx + zoom * 0.5, cy + zoom * 0.5 * aspect));
        let side = 130.0;
        let inset = egui::Rect::from_min_size(egui::pos2(rect.max.x - side - 6.0, rect.min.y + 6.0), egui::vec2(side, side));
        ui.painter().rect_filled(inset.expand(2.0), 2.0, egui::Color32::BLACK);
        ui.painter().image(tex.id(), inset, uv, egui::Color32::WHITE);
        let mid = inset.center();
        ui.painter().line_segment([egui::pos2(mid.x - 10.0, mid.y), egui::pos2(mid.x + 10.0, mid.y)], egui::Stroke::new(1.0, green));
        ui.painter().line_segment([egui::pos2(mid.x, mid.y - 10.0), egui::pos2(mid.x, mid.y + 10.0)], egui::Stroke::new(1.0, green));
    }
    let (mut save_clicked, mut close, mut want_preview) = (false, false, false);
    ui.horizontal(|ui| {
        let save = ui.add_enabled(c.saving.is_none(), egui::Button::new("Save Corners"));
        elems.push((format!("{p}.corners.save"), save.rect, "Save Corners".into()));
        save_clicked = save.clicked();
        let pv = ui.add_enabled(c.preview_job.is_none(), egui::Button::new(if c.preview_job.is_some() { "Checking alignment…" } else { "Preview Alignment" }));
        elems.push((format!("{p}.corners.preview"), pv.rect, "Preview Alignment".into()));
        want_preview = pv.clicked();
        let cancel = ui.button("Close");
        elems.push((format!("{p}.corners.close"), cancel.rect, "Close".into()));
        close = cancel.clicked();
    });
    if !c.preview_image.is_empty() {
        let pid = egui::Id::new(("ei-tv-preview-tex", c.preview_image.clone()));
        let ptex = ui.ctx().data(|m| m.get_temp::<egui::TextureHandle>(pid)).or_else(|| {
            let img = image::open(&c.preview_image).ok()?.to_rgba8();
            let size = [img.width() as usize, img.height() as usize];
            let t = ui.ctx().load_texture("ei-tv-preview", egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw()), egui::TextureOptions::LINEAR);
            ui.ctx().data_mut(|m| m.insert_temp(pid, t.clone()));
            Some(t)
        });
        if let Some(t) = ptex {
            let [pw, ph] = t.size();
            let sc = (640.0 / pw.max(1) as f32).min(360.0 / ph.max(1) as f32);
            ui.label(egui::RichText::new("The tracked outline across the recording (saved corners):").weak());
            ui.image((t.id(), egui::vec2(pw as f32 * sc, ph as f32 * sc)));
        }
    }
    if want_preview {
        let png = support_dir().join("tvlight").join(format!("{}-preview.png", file_stem(&c.path)));
        match engine_step(vec!["tv-preview".into(), c.path.clone(), "--out".into(), png.to_string_lossy().to_string()]) {
            Ok((step, engine)) => c.preview_job = Some(start_job("TV Light Sync: alignment preview".into(), None, vec![step], false, Some(engine))),
            Err(e) => d.error = e,
        }
    }
    if save_clicked {
        let q: Vec<String> = c.quad.iter().flat_map(|v| [format!("{:.4}", v[0]), format!("{:.4}", v[1])]).collect();
        match engine_step(vec!["tv-calibrate".into(), c.path.clone(), "--quad".into(), q.join(","), "--ref-t".into(), format!("{}", c.ref_t)]) {
            Ok((step, engine)) => c.saving = Some(start_job("TV Light Sync: save corners".into(), None, vec![step], false, Some(engine))),
            Err(e) => d.error = e,
        }
    }
    if close {
        d.tv.calib = None;
    }
}

/// Pick up finished TV Light Sync jobs (recordings listed, a frame ready, corners saved).
fn poll_tv(d: &mut Draft) {
    if let Some(j) = d.tv.scan_job
        && let Some((st, res, err)) = job_state(j)
        && st != Status::Running
    {
        d.tv.scan_job = None;
        if st == Status::Done {
            d.tv.recs = res
                .get("recordings")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .map(|r| TvRec {
                            path: r.get("path").and_then(Value::as_str).unwrap_or_default().to_string(),
                            orientation: r.get("orientation").and_then(Value::as_str).unwrap_or_default().to_string(),
                            calibrated: r.get("calibrated").and_then(Value::as_bool).unwrap_or(false),
                        })
                        .collect()
                })
                .unwrap_or_default();
        } else {
            d.error = format!("Could not read the recordings: {err}");
        }
    }
    let Some(c) = d.tv.calib.as_mut() else { return };
    if let Some(j) = c.job
        && let Some((st, res, err)) = job_state(j)
        && st != Status::Running
    {
        c.job = None;
        if st == Status::Done {
            c.image = res.get("image").and_then(Value::as_str).unwrap_or_default().to_string();
            c.ref_t = res.get("ref_t").and_then(Value::as_f64).unwrap_or(0.0);
            c.at = c.ref_t;
            c.duration = res.get("duration").and_then(Value::as_f64).unwrap_or(c.duration);
            c.source = res.get("source").and_then(Value::as_str).unwrap_or("suggested").to_string();
            if let Some(q) = res.get("quad").and_then(Value::as_array) {
                for (k, pt) in q.iter().take(4).enumerate() {
                    let x = pt.get(0).and_then(Value::as_f64).unwrap_or(0.5) as f32;
                    let y = pt.get(1).and_then(Value::as_f64).unwrap_or(0.5) as f32;
                    if let Some(slot) = c.quad.get_mut(k) {
                        *slot = [x, y];
                    }
                }
            }
            c.loaded = true;
        } else {
            d.error = format!("Could not read the recording: {err}");
            d.tv.calib = None;
            return;
        }
    }
    if let Some(j) = c.preview_job
        && let Some((st, res, err)) = job_state(j)
        && st != Status::Running
    {
        c.preview_job = None;
        if st == Status::Done {
            c.preview_image = res.get("image").and_then(Value::as_str).unwrap_or_default().to_string();
        } else {
            d.error = format!("Could not check the alignment: {err}");
            return;
        }
    }
    if let Some(j) = c.saving
        && let Some((st, _, err)) = job_state(j)
        && st != Status::Running
    {
        c.saving = None;
        if st == Status::Done {
            let path = c.path.clone();
            if let Some(r) = d.tv.recs.iter_mut().find(|r| r.path == path) {
                r.calibrated = true;
            }
            d.tv.calib = None;
        } else {
            d.error = format!("Could not save the corners: {err}");
        }
    }
}

/// Pick up finished recorder jobs/// Pick up finished recorder jobs (sentences loaded, microphones listed, a take recorded).
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
                                set: s("set"),
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
                        *t = Some(Take { file: f.to_string_lossy().to_string(), issues: String::new(), seconds: 0.0 });
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
                    *t = Some(Take { file, issues, seconds: res.get("seconds").and_then(Value::as_f64).unwrap_or(0.0) });
                }
                // stay on the sentence: you see it was saved and can play it; Next moves on
            }
            Status::Cancelled => {}
            _ => d.rec.message = if err.is_empty() { "The recording failed (AI ▸ AI Jobs has the details).".into() } else { err },
        }
    }
}

fn description(kind: Kind) -> &'static str {
    match kind {
        Kind::Hooks => {
            "Your hook library: proven opening lines (the first 3 seconds that keep viewers watching), kept on this Mac so the script writer, Video from Topic and the voiceover edit can use them offline. Sync fetches what is hot now (Google Trends for your countries, this week's most-viewed YouTube videos in your niches), and Claude (with web search) turns it into new hooks in English, Hindi and Hinglish."
        }
        Kind::Speak => {
            "Speaks a script in your cloned voice (or a macOS voice) and saves the audio, so you can hear it without making a video. Every line is spoken with its emotion (set them below). With sound design, a second file also has music that follows the emotions, hits, risers and pauses."
        }
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
        Kind::Tv => {
            "Plays your clip on the big TV in a recording of your room, with the TV backlight matching your clip's colours in software (no HDMI sync box): EI picks the moments of your recordings whose light matches each part of your clip and recolours the glow for the rest. The monitor and laptop stay as recorded. Each recording is analysed once (a few minutes) and remembered."
        }
        Kind::Topic => {
            "Give a topic or a link to a post. Claude researches it on the web (facts with sources), writes the script, and EI makes the video: your cloned voice (or a macOS voice), credited free-licensed photos and videos, fact cards, your own footage, optional YouTube clips. Or make only the script with its sources."
        }
        Kind::Voice => {
            "Clones your voice on this Mac (Chatterbox Multilingual, Hindi + English). Read the sentences EI shows (they cover every sound and the deliveries a video needs), or use recordings you already have. Use only your own voice, or a voice you have permission to use."
        }
        Kind::Clips => {
            "Edits each clip from a whole scene of a long video: only its best beats, cut on the action, a teaser first, slow motion on the climax, punch-ins on the big hits. Reframed to truly fill the shape (following the action and faces). Talk clips are built from sentences and topics. Optional captions, title card and music (cuts move onto its beats).\n\nContinuous sections: each output is one uninterrupted part of the video (15 s – 3 min). EI chooses where it starts and ends: on a cut of the video, between sentences, before the action builds and after it settles, never mid-sentence or on black."
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
        progress: Option<(f32, String)>,
        paused: bool,
        queued: bool,
        retry: bool,
        outputs: Vec<String>,
        out_dir: Option<String>,
        importable: bool,
        log: String,
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
            last: match j.log.iter().rev().find(|l| l.as_str() != "Paused" && l.as_str() != "Resumed") {
                Some(l) if l.starts_with("$ ") => "Starting…".to_string(),
                Some(l) => l.clone(),
                None => "Starting…".to_string(),
            },
            error: j.error.clone(),
            secs: j.active().as_secs_f64(),
            progress: j.progress.clone(),
            paused: j.paused_since.is_some(),
            queued: j.queued,
            retry: j.status != Status::Running && !j.steps.is_empty(),
            outputs: j
                .result
                .get("outputs")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
                .unwrap_or_default(),
            out_dir: j.result.get("out_dir").and_then(Value::as_str).map(str::to_string),
            importable: j.kind.is_some() && j.status == Status::Done,
            log: j.log.join("\n"),
        })
        .collect();
    let (waiting, queue_paused, queue_note) = {
        let mut q = queue().lock().unwrap_or_else(PoisonError::into_inner);
        load_queue(&mut q);
        (q.items.iter().map(|i| (i.title.clone(), i.uid, i.params.is_some())).collect::<Vec<_>>(), q.paused, q.note.clone())
    };
    let mut elems: Vec<(String, egui::Rect, String)> = Vec::new();
    let mut cancel: Option<u64> = None;
    let mut pause: Option<(u64, bool)> = None;
    let mut retry: Option<u64> = None;
    let mut reveal: Option<String> = None;
    let mut open_file: Option<String> = None;
    let mut import: Option<u64> = None;
    let mut copy: Option<String> = None;
    let mut queue_action: Option<&str> = None;
    let mut remove_waiting: Option<usize> = None;
    let mut move_item: Option<(usize, usize)> = None;
    let mut edit_item: Option<u64> = None;
    let mut import_file = false;
    let mut export_file = false;
    let mut clear = false;
    egui::Window::new("AI Jobs").id(egui::Id::new("ei-ai-jobs")).open(&mut open).default_width(600.0).resizable(true).show(ctx, |ui| {
        // the queue: what waits (in the order it will run), and its controls
        ui.horizontal(|ui| {
            let i = ui.button("Import Tasks…").on_hover_text("Add tasks from a queue file (.json)");
            elems.push(("ai.jobs.queue.import".into(), i.rect, "Import Tasks…".into()));
            import_file = i.clicked();
            if !waiting.is_empty() {
                let e = ui.button("Export Queue…").on_hover_text("Save the waiting tasks as a queue file (.json)");
                elems.push(("ai.jobs.queue.export".into(), e.rect, "Export Queue…".into()));
                export_file = e.clicked();
            }
        });
        if !waiting.is_empty() || queue_paused {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.strong(format!("Queue: {} waiting{}", waiting.len(), if queue_paused { " (paused)" } else { "" }));
                    let b = ui.button(if queue_paused { "Resume Queue" } else { "Pause Queue" });
                    elems.push(("ai.jobs.queue.pause".into(), b.rect, b.clicked().to_string()));
                    if b.clicked() {
                        queue_action = Some(if queue_paused { "resume" } else { "pause" });
                    }
                    let c = ui.button("Cancel All Queued");
                    elems.push(("ai.jobs.queue.cancelAll".into(), c.rect, "Cancel All Queued".into()));
                    if c.clicked() {
                        queue_action = Some("clear");
                    }
                    let c2 = ui.button("Cancel Current and Queued");
                    elems.push(("ai.jobs.queue.cancelEverything".into(), c2.rect, "Cancel Current and Queued".into()));
                    if c2.clicked() {
                        queue_action = Some("everything");
                    }
                });
                if !queue_note.is_empty() {
                    ui.weak(&queue_note);
                }
                let n = waiting.len();
                for (k, (t, uid, editable)) in waiting.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("{}. {t}", k + 1));
                        let id = format!("ai.jobs.queue.{k}");
                        let up = ui.add_enabled(k > 0, egui::Button::new("↑").small()).on_hover_text("Run earlier");
                        elems.push((format!("{id}.up"), up.rect, "Up".into()));
                        if up.clicked() {
                            move_item = Some((k, k - 1));
                        }
                        let down = ui.add_enabled(k + 1 < n, egui::Button::new("↓").small()).on_hover_text("Run later");
                        elems.push((format!("{id}.down"), down.rect, "Down".into()));
                        if down.clicked() {
                            move_item = Some((k, k + 1));
                        }
                        let top = ui.add_enabled(k > 0, egui::Button::new("Top").small()).on_hover_text("Run next");
                        elems.push((format!("{id}.top"), top.rect, "Top".into()));
                        if top.clicked() {
                            move_item = Some((k, 0));
                        }
                        let e = ui
                            .add_enabled(*editable, egui::Button::new("Edit").small())
                            .on_hover_text("Change its settings")
                            .on_disabled_hover_text("Queued by an older EI: remove it and queue it again to edit");
                        elems.push((format!("{id}.edit"), e.rect, "Edit".into()));
                        if e.clicked() {
                            edit_item = Some(*uid);
                        }
                        let r = ui.small_button("Remove");
                        elems.push((format!("{id}.remove"), r.rect, "Remove".into()));
                        if r.clicked() {
                            remove_waiting = Some(k);
                        }
                    });
                }
            });
            ui.add_space(4.0);
        }
        if rows.is_empty() && waiting.is_empty() {
            ui.label("No AI jobs yet. Start one from the AI menu.");
        }
        egui::ScrollArea::vertical().max_height(520.0).show(ui, |ui| {
            for r in &rows {
                let (icon, color) = match r.status {
                    Status::Running if r.paused => ("⏸", egui::Color32::from_rgb(0x9A, 0x9A, 0xA8)),
                    Status::Running => ("⏳", egui::Color32::from_rgb(0xE0, 0xB0, 0x40)),
                    Status::Done => ("✔", egui::Color32::from_rgb(0x5C, 0xC0, 0x7A)),
                    Status::Failed => ("✖", egui::Color32::from_rgb(0xE5, 0x6B, 0x6B)),
                    Status::Cancelled => ("■", egui::Color32::GRAY),
                };
                ui.horizontal(|ui| {
                    ui.colored_label(color, icon);
                    ui.strong(&r.title);
                    if r.queued {
                        ui.weak("(queue)");
                    }
                    ui.weak(fmt_duration(r.secs));
                });
                match r.status {
                    Status::Running => {
                        let (frac, stage) = r.progress.clone().unwrap_or((0.0, String::new()));
                        let left = remaining(r.secs, frac).map(|t| format!(" · about {} left", fmt_duration(t))).unwrap_or_default();
                        let what = if r.paused {
                            "paused".to_string()
                        } else if stage.is_empty() {
                            "working".to_string()
                        } else {
                            stage.clone()
                        };
                        let bar =
                            egui::ProgressBar::new(frac).text(format!("{:.0}% · {what}{left}", frac * 100.0)).desired_width(ui.available_width().min(560.0));
                        let br = ui.add(if r.progress.is_none() && !r.paused { bar.animate(true) } else { bar });
                        elems.push((format!("ai.jobs.{}.progress", r.id), br.rect, format!("{frac:.3}")));
                        ui.weak(&r.last);
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
                        let pb = ui.button(if r.paused { "Resume" } else { "Pause" });
                        elems.push((format!("{pid}.pause"), pb.rect, if r.paused { "Resume" } else { "Pause" }.into()));
                        if pb.clicked() {
                            pause = Some((r.id, !r.paused));
                        }
                        let b = ui.button("Cancel");
                        elems.push((format!("{pid}.cancel"), b.rect, "Cancel".into()));
                        if b.clicked() {
                            cancel = Some(r.id);
                        }
                    }
                    if r.retry && r.status != Status::Done {
                        let b = ui.button("Retry").on_hover_text("Run it again: it continues from what was already made");
                        elems.push((format!("{pid}.retry"), b.rect, "Retry".into()));
                        if b.clicked() {
                            retry = Some(r.id);
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
                    let c = ui.small_button("Copy Log");
                    elems.push((format!("ai.jobs.{}.copyLog", r.id), c.rect, "Copy Log".into()));
                    if c.clicked() {
                        copy = Some(r.log.clone());
                    }
                    // selectable text: drag to select, ⌘C to copy, ⌘A to select all
                    let mut text: &str = &r.log;
                    egui::ScrollArea::vertical().id_salt(("ei-ai-log-scroll", r.id)).max_height(220.0).stick_to_bottom(true).show(ui, |ui| {
                        let t = ui.add(egui::TextEdit::multiline(&mut text).font(egui::TextStyle::Monospace).desired_width(f32::INFINITY).desired_rows(4));
                        elems.push((format!("ai.jobs.{}.log", r.id), t.rect, String::new()));
                    });
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
    if let Some(t) = copy {
        ctx.copy_text(t);
        app.ui.status = "Log copied".into();
    }
    if let Some(j) = cancel {
        cancel_job(j);
    }
    if let Some((j, p)) = pause {
        pause_job(j, p);
    }
    if let Some(j) = retry
        && retry_job(j).is_some()
    {
        app.ui.status = "Running it again: it continues from what was already made".into();
    }
    if let Some((from, to)) = move_item {
        move_waiting(from, to);
    }
    if let Some(uid) = edit_item
        && let Err(e) = edit_waiting(ctx, uid)
    {
        app.ui.status = e;
    }
    if import_file && let Some(path) = app.hooks.pick_open_file.as_mut().and_then(|f| f("Queue file", &["json"])) {
        app.ui.status =
            match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|t| serde_json::from_str::<Value>(&t).map_err(|e| e.to_string())) {
                Ok(v) => import_summary(import_tasks(&v)),
                Err(e) => format!("Could not read the queue file: {e}"),
            };
    }
    if export_file {
        let name = format!("EI queue {}.json", waiting.len());
        if let Some(path) = app.hooks.pick_save_as.as_mut().and_then(|f| f("Queue file", &["json"], &name)) {
            app.ui.status = match serde_json::to_string_pretty(&export_tasks())
                .map_err(|e| e.to_string())
                .and_then(|t| std::fs::write(&path, t).map_err(|e| e.to_string()))
            {
                Ok(()) => format!("Queue saved: {path}"),
                Err(e) => format!("Could not save the queue: {e}"),
            };
        }
    }
    if let Some(k) = remove_waiting {
        let mut q = queue().lock().unwrap_or_else(PoisonError::into_inner);
        if k < q.items.len() {
            q.items.remove(k);
            save_queue(&q);
        }
    }
    match queue_action {
        Some("pause") | Some("resume") => {
            let mut q = queue().lock().unwrap_or_else(PoisonError::into_inner);
            q.paused = queue_action == Some("pause");
            q.note.clear();
            save_queue(&q);
        }
        Some("clear") => {
            cancel_queue(false);
        }
        Some("everything") => {
            cancel_queue(true);
        }
        _ => {}
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

/// The result of an import, for the status line.
fn import_summary((added, dup, errors): (usize, usize, Vec<String>)) -> String {
    let mut out = format!("Queue: {added} task(s) added");
    if dup > 0 {
        out += &format!(", {dup} already waiting");
    }
    if !errors.is_empty() {
        out += &format!("; {} not added: {}", errors.len(), errors.join("; "));
    }
    out
}

/// "45s", "3m 20s", "1h 05m".
fn fmt_duration(secs: f64) -> String {
    let s = if secs.is_finite() { secs.max(0.0) as u64 } else { 0 };
    match s {
        0..=59 => format!("{s}s"),
        60..=3599 => format!("{}m {:02}s", s / 60, s % 60),
        _ => format!("{}h {:02}m", s / 3600, (s % 3600) / 60),
    }
}

/// Time left from the time spent and the fraction done (None until there is enough to go on).
fn remaining(spent: f64, frac: f32) -> Option<f64> {
    let f = f64::from(frac);
    ((0.03..0.999).contains(&f) && spent >= 5.0).then(|| spent * (1.0 - f) / f)
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
            Kind::Split | Kind::Clips | Kind::Audio | Kind::Speak => strs("outputs"),
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
        // continuous sections: their own lengths and folder, no edited-clip options
        let mut sec = Draft::from_params(Kind::Clips, &json!({"input": video.to_string_lossy(), "clipType": "sections", "count": 2, "aspect": "same"}));
        let a = sec.engine_args().unwrap();
        for pair in [["--min", "15"], ["--target", "60"], ["--max", "180"], ["--aspect", "same"], ["--count", "2"]] {
            assert!(a.windows(2).any(|w| w == pair), "{pair:?} in {a:?}");
        }
        assert!(a.contains(&"--sections".to_string()) && !a.contains(&"--pace".to_string()) && !a.contains(&"--no-hook".to_string()), "{a:?}");
        assert!(sec.out.ends_with("long - EI sections"), "{}", sec.out);
        let p = json!({"input": video.to_string_lossy(), "clipType": "sections", "min": 4, "target": 30, "max": 90});
        assert!(Draft::from_params(Kind::Clips, &p).engine_args().is_err());
        let mut au =
            Draft::from_params(Kind::Audio, &json!({"clips": [video.to_string_lossy()], "format": "flac", "start": 5, "end": "20.5", "normalize": true}));
        let a = au.engine_args().unwrap();
        assert!(a.windows(2).any(|w| w == ["--format", "flac"]) && a.windows(2).any(|w| w == ["--end", "20.5"]) && a.contains(&"--normalize".to_string()));
        assert!(Draft::from_params(Kind::Audio, &json!({})).engine_args().is_err());
        assert!(Draft::from_params(Kind::Audio, &json!({"clips": [video.to_string_lossy()], "start": "soon"})).engine_args().is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sound_design_and_speak_arguments() {
        let dir = std::env::temp_dir().join(format!("ei-ai-sound-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("gta.script.json");
        std::fs::write(
            &script,
            r#"{"title": "T", "segments": [{"text": "Twelve years.", "emotion": "awe", "intensity": 0.8}, {"text": " "}, {"text": "It's here!", "emotion": "excited"}]}"#,
        )
        .unwrap();
        let sp = script.to_string_lossy().to_string();
        let mut d = Draft::from_params(Kind::Speak, &json!({"script": sp, "voice": "test1", "soundDesign": true, "licence": "personal"}));
        let a = d.engine_args().unwrap();
        assert_eq!(a.first().map(String::as_str), Some("speak"));
        for pair in [["--voice", "test1"], ["--music-source", "internet"], ["--licence", "personal"], ["--music-level", "-12"], ["--takes", "2"]] {
            assert!(a.windows(2).any(|w| w == pair), "{pair:?} in {a:?}");
        }
        assert!(a.contains(&"--sound-design".to_string()) && d.out.ends_with(".wav"), "{a:?} {}", d.out);
        // typed text instead of a file; a macOS voice has no takes
        let mut t = Draft::from_params(Kind::Speak, &json!({"notes": "[Awe 0.8] Look up.", "voice": "mac:Rishi"}));
        let a = t.engine_args().unwrap();
        assert!(
            a.windows(2).any(|w| w == ["--script", "[Awe 0.8] Look up."]) && !a.contains(&"--takes".to_string()) && !a.contains(&"--sound-design".to_string())
        );
        assert!(Draft::from_params(Kind::Speak, &json!({"voice": "test1"})).engine_args().is_err());
        assert!(Draft::from_params(Kind::Speak, &json!({"script": sp})).engine_args().is_err());
        // a music folder must exist; a music edit only gets effects; no music needs effects
        assert!(Draft::from_params(Kind::Speak, &json!({"script": sp, "voice": "v", "soundDesign": true, "musicSource": "folder"})).engine_args().is_err());
        let mut f = Draft::from_params(
            Kind::Speak,
            &json!({"script": sp, "voice": "v", "soundDesign": true, "musicSource": "folder", "musicFolder": dir.to_string_lossy()}),
        );
        assert!(f.engine_args().unwrap().windows(2).any(|w| w[0] == "--music-folder"));
        assert!(
            Draft::from_params(Kind::Speak, &json!({"script": sp, "voice": "v", "soundDesign": true, "musicSource": "none", "soundEffects": false}))
                .engine_args()
                .is_err()
        );
        let song = dir.join("song.mp3");
        std::fs::write(&song, b"x").unwrap();
        let clip = dir.join("a.mp4");
        std::fs::write(&clip, b"x").unwrap();
        let mut m = Draft::from_params(
            Kind::Music,
            &json!({"music": song.to_string_lossy(), "clips": [clip.to_string_lossy()], "soundDesign": true, "musicSource": "ai"}),
        );
        let a = m.engine_args().unwrap();
        assert!(a.windows(2).any(|w| w == ["--music-source", "none"]), "{a:?}");
        // clips: the score replaces the single music file
        let mut c = Draft::from_params(Kind::Clips, &json!({"input": clip.to_string_lossy(), "music": song.to_string_lossy(), "soundDesign": true}));
        let a = c.engine_args().unwrap();
        assert!(a.contains(&"--sound-design".to_string()) && !a.contains(&"--music".to_string()), "{a:?}");
        let mut off = Draft::from_params(Kind::Clips, &json!({"input": clip.to_string_lossy()}));
        assert!(!off.engine_args().unwrap().contains(&"--sound-design".to_string()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn progress_lines_and_times() {
        assert_eq!(parse_progress(r#"EI_PROGRESS {"fraction": 0.42, "stage": "voice"}"#), Some((0.42, "voice".to_string())));
        assert_eq!(parse_progress(r#"EI_PROGRESS {"fraction": 7, "stage": ""}"#), Some((1.0, String::new())));
        assert_eq!(parse_progress("EI_PROGRESS not json"), None);
        assert_eq!(parse_progress("progress 3/10 shots"), None);
        assert_eq!(fmt_duration(42.4), "42s");
        assert_eq!(fmt_duration(200.0), "3m 20s");
        assert_eq!(fmt_duration(3751.0), "1h 02m");
        assert_eq!(fmt_duration(f64::NAN), "0s");
        assert_eq!(remaining(60.0, 0.25), Some(180.0));
        assert_eq!(remaining(60.0, 0.01), None); // too early to tell
        assert_eq!(remaining(2.0, 0.5), None);
    }

    #[test]
    fn settings_round_trip_for_every_tool() {
        let dir = std::env::temp_dir().join(format!("ei-ai-rt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let v = dir.join("long.mov");
        std::fs::write(&v, b"x").unwrap();
        let vs = v.to_string_lossy().to_string();
        let cases = [
            (
                Kind::Topic,
                json!({"topic": "GTA 6 map", "voice": "nxstory-a1", "seconds": 45, "language": "hinglish", "aspect": "vertical", "hookLine": "Big news.",
                "youtube": true, "soundDesign": true, "musicSource": "ai", "resolution": "4k", "takes": 3}),
            ),
            (Kind::Clips, json!({"input": vs, "clipType": "sections", "count": 3, "aspect": "same", "hook": false, "captionMode": "off"})),
            (Kind::Clips, json!({"input": vs, "count": 7, "pace": "fast", "titleMode": "intro", "llm": "none"})),
            (Kind::Voiceover, json!({"input": vs, "clips": [vs], "hook": "teaser", "niche": "travel", "captions": true, "pace": "slow"})),
            (Kind::Music, json!({"music": vs, "clips": [vs], "length": 20, "look": "warm", "soundDesign": true})),
            (Kind::Speak, json!({"notes": "[Awe 0.8] Hi.", "voice": "mac:Rishi", "soundDesign": true, "licence": "personal"})),
            (Kind::Split, json!({"input": vs, "clipLength": 30, "quality": "high"})),
            (Kind::Audio, json!({"clips": [vs], "format": "flac", "start": "5", "normalize": true})),
            (Kind::Hooks, json!({"niches": "gaming, travel", "countries": "IN", "hookClaude": false})),
        ];
        for (kind, params) in cases {
            let mut a = Draft::from_params(kind, &params);
            let first = a.engine_args().unwrap_or_else(|e| panic!("{kind:?}: {e}"));
            let mut b = Draft::from_params(kind, &a.to_params());
            assert_eq!(b.engine_args().unwrap(), first, "{kind:?} keeps every setting through the queue");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn older_queue_items_become_editable() {
        // a task exactly as the previous EI saved it
        let args: Vec<String> = [
            "topic",
            "gta 6 Pre-order records",
            "--seconds",
            "60",
            "--language",
            "en",
            "--aspect",
            "vertical",
            "--writer",
            "claude",
            "--fresh",
            "--hook-line",
            "GTA 6 just broke a record no game has ever touched",
            "--voice",
            "nxstory-a1",
            "--takes",
            "2",
            "--pace",
            "medium",
            "--sources",
            "free",
            "cards",
            "youtube",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let p = params_from_args("topic", &args).unwrap();
        let mut d = Draft::from_params(Kind::Topic, &p);
        assert_eq!(d.engine_args().unwrap(), args, "the same command after editing nothing");
        assert!(params_from_args("topic", &["topic".into(), "x".into(), "--unknown".into()]).is_none());
        assert!(params_from_args("clips", &args).is_none());
        // an export file is checked task by task
        let file = json!({"tasks": [
            {"command": "ai.topic", "title": "Pre-order records", "params": p},
            {"command": "ai.topic", "args": args},
            {"command": "ai.nothing", "params": {}},
            {"command": "ai.topic", "params": {"topic": ""}},
            {"command": "ai.voice", "params": {"name": "x"}},
        ]});
        let r = check_tasks(&file);
        assert!(
            r[0].is_ok() && r[1].is_ok() && r[2].is_err() && r[3].is_err() && r[4].is_err(),
            "{:?}",
            r.iter().map(|x| x.as_ref().err()).collect::<Vec<_>>()
        );
        assert_eq!(r[0].as_ref().map(|t| t.0.clone()).unwrap(), "Pre-order records");
        assert!(check_tasks(&json!({}))[0].is_err());
        let mut v = vec![1, 2, 3, 4];
        assert!(move_in(&mut v, 3, 0) && v == [4, 1, 2, 3]);
        assert!(move_in(&mut v, 0, 9) && v == [1, 2, 3, 4]);
        assert!(!move_in(&mut v, 7, 0));
    }

    /// Opt-in: EI_QUEUE_FILE=/path/to/queue.json checks that every task in a queue file imports.
    #[test]
    fn a_queue_file_imports() {
        let Ok(path) = std::env::var("EI_QUEUE_FILE") else { return };
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for r in check_tasks(&v) {
            match r {
                Ok(t) => eprintln!("ok: {}", t.0),
                Err(e) => panic!("{e}"),
            }
        }
    }

    #[test]
    fn queue_items_round_trip() {
        let it = QueueItem {
            title: "Video from Topic: GTA 6".into(),
            kind: "topic".into(),
            steps: vec![vec!["python".into(), "ei_engine.py".into(), "topic".into(), "GTA 6".into()]],
            import: false,
            cwd: Some("/tmp".into()),
            params: None,
            uid: 1,
        };
        let v = json!({"items": [it.clone()], "paused": true});
        let back: Vec<QueueItem> = serde_json::from_value(v["items"].clone()).unwrap();
        assert_eq!(back, vec![it]);
        assert_eq!(Kind::from_command(&format!("ai.{}", back[0].kind)), Some(Kind::Topic));
    }

    #[cfg(unix)]
    #[test]
    fn pause_resume_and_cancel_reach_the_whole_process_group() {
        use std::os::unix::process::CommandExt;
        // a shell with a child of its own (like the engine with ffmpeg)
        let mut child = Command::new("/bin/sh").args(["-c", "sleep 30 & wait"]).process_group(0).spawn().unwrap();
        let pid = child.id();
        std::thread::sleep(Duration::from_millis(300));
        let state = |p: u32| {
            let out = Command::new("/bin/ps").args(["-o", "stat=", "-g", &p.to_string()]).output().unwrap();
            String::from_utf8_lossy(&out.stdout).lines().map(|l| l.trim().chars().next().unwrap_or(' ')).collect::<String>()
        };
        signal_group(pid, "STOP");
        std::thread::sleep(Duration::from_millis(200));
        let s = state(pid);
        assert!(!s.is_empty() && s.chars().all(|c| c == 'T'), "every process stopped: {s}");
        signal_group(pid, "CONT");
        std::thread::sleep(Duration::from_millis(200));
        assert!(!state(pid).contains('T'), "running again: {}", state(pid));
        signal_group(pid, "TERM");
        let status = child.wait().unwrap();
        assert!(!status.success());
        std::thread::sleep(Duration::from_millis(200));
        assert!(state(pid).is_empty(), "the child sleep is gone too: {}", state(pid));
    }

    #[test]
    fn topic_hook_line_and_search() {
        let a = Draft::from_params(Kind::Topic, &json!({"topic": "GTA 6", "voice": "mac:Rishi", "hookLine": "  GTA 6 is finally here!  ", "findHook": true}))
            .engine_args()
            .unwrap();
        assert!(a.windows(2).any(|w| w == ["--hook-line", "GTA 6 is finally here!"]) && !a.contains(&"--find-hook".to_string()), "{a:?}");
        let a = Draft::from_params(Kind::Topic, &json!({"topic": "GTA 6", "voice": "mac:Rishi", "findHook": true})).engine_args().unwrap();
        assert!(a.contains(&"--find-hook".to_string()) && !a.contains(&"--hook-line".to_string()), "{a:?}");
        let a = Draft::from_params(Kind::Topic, &json!({"topic": "GTA 6", "voice": "mac:Rishi"})).engine_args().unwrap();
        assert!(!a.contains(&"--find-hook".to_string()) && !a.contains(&"--hook-line".to_string()), "{a:?}");
    }

    #[test]
    fn hook_library_and_voiceover_hook_arguments() {
        let mut h = Draft::from_params(Kind::Hooks, &json!({"niches": "gaming, travel", "countries": "IN US", "hookClaude": false}));
        h.hook_niches = "gaming, travel".into(); // (without touching the saved settings in this test)
        let a = h.engine_args().unwrap_or_default();
        assert_eq!(a.first().map(String::as_str), Some("hooks-sync"), "{a:?}");
        assert!(a.windows(3).any(|w| w == ["--niches", "gaming", "travel"]) && a.windows(3).any(|w| w == ["--geo", "IN", "US"]), "{a:?}");
        assert!(a.contains(&"--no-claude".to_string()));
        let mut empty = Draft::from_params(Kind::Hooks, &json!({}));
        empty.hook_niches = " , ".into();
        assert!(empty.engine_args().is_err());
        let dir = std::env::temp_dir().join(format!("ei-ai-hook-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let v = dir.join("v.wav");
        std::fs::write(&v, b"x").unwrap();
        let vs = v.to_string_lossy().to_string();
        let a = Draft::from_params(Kind::Voiceover, &json!({"input": vs, "clips": [vs], "hook": "teaser", "niche": "travel"})).engine_args().unwrap();
        assert!(a.windows(2).any(|w| w == ["--hook", "teaser"]) && a.windows(2).any(|w| w == ["--niche", "travel"]), "{a:?}");
        let a = Draft::from_params(Kind::Voiceover, &json!({"input": vs, "clips": [vs]})).engine_args().unwrap();
        assert!(!a.contains(&"--hook".to_string()), "auto is the default: {a:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolution_4k_in_every_video_tool() {
        let dir = std::env::temp_dir().join(format!("ei-ai-4k-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let v = dir.join("long.mov");
        std::fs::write(&v, b"x").unwrap();
        let vs = v.to_string_lossy().to_string();
        for (kind, extra) in [
            (Kind::Clips, json!({"input": vs})),
            (Kind::Talk, json!({"input": vs})),
            (Kind::Voiceover, json!({"input": vs, "clips": [vs]})),
            (Kind::Music, json!({"music": vs, "clips": [vs]})),
            (Kind::Reference, json!({"input": vs, "clips": [vs]})),
        ] {
            let mut p = extra.clone();
            p["resolution"] = json!("4k");
            let a = Draft::from_params(kind, &p).engine_args().unwrap();
            assert!(a.windows(2).any(|w| w == ["--resolution", "4k"]), "{kind:?}: {a:?}");
            let a = Draft::from_params(kind, &extra).engine_args().unwrap();
            assert!(!a.contains(&"--resolution".to_string()), "{kind:?} 1080p is the default: {a:?}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn emotion_editor_reads_and_writes_scripts() {
        let dir = std::env::temp_dir().join(format!("ei-ai-emo-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let js = dir.join("a.script.json");
        std::fs::write(
            &js,
            r#"{"segments": [{"text": "One.", "emotion": "hook", "visual": "v"}, {"text": ""}, {"text": "Two.", "emotion": "Grief", "intensity": 0.9}]}"#,
        )
        .unwrap();
        let mut ed = EmotionEditor::load(&js.to_string_lossy());
        assert_eq!(ed.lines.iter().map(|l| l.emotion.as_str()).collect::<Vec<_>>(), ["Surprise", "Grief"]);
        assert!((ed.lines[1].intensity - 0.9).abs() < 1e-9);
        ed.lines[0].emotion = "Mystery".into();
        ed.lines[0].intensity = 0.7;
        ed.save();
        assert!(ed.error.is_empty(), "{}", ed.error);
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&js).unwrap()).unwrap();
        assert_eq!(v["segments"][0]["emotion"], "Mystery");
        assert_eq!(v["segments"][0]["intensity"], 0.7);
        assert_eq!(v["segments"][0]["visual"], "v"); // everything else is kept
        assert_eq!(v["segments"][2]["emotion"], "Grief");
        let tx = dir.join("b.txt");
        std::fs::write(&tx, "[Awe 0.8] Look up.\n\nPlain line.\n[excited] Go!\n").unwrap();
        let mut ed = EmotionEditor::load(&tx.to_string_lossy());
        assert_eq!(
            ed.lines.iter().map(|l| (l.emotion.as_str(), l.text.as_str())).collect::<Vec<_>>(),
            [("Awe", "Look up."), ("Neutral", "Plain line."), ("Excitement", "Go!")]
        );
        ed.lines[1].emotion = "Hope".into();
        ed.save();
        assert_eq!(std::fs::read_to_string(&tx).unwrap(), "[Awe 0.8] Look up.\n[Hope 0.6] Plain line.\n[Excitement 0.6] Go!\n");
        assert_eq!(parse_tagged("[Rage 7] x").0, Some(("Rage".to_string(), Some(1.0))));
        assert_eq!(parse_tagged("no [tag] here").0, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn emotion_names_match_the_engine() {
        // the dialog's list must be the engine's (ai-engine/emotions.py)
        let py = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ai-engine/emotions.py");
        let Ok(src) = std::fs::read_to_string(&py) else { return };
        for n in EMOTION_NAMES {
            assert!(src.contains(&format!("\"{n}\": (")), "{n} is not in emotions.py");
        }
        let engine = src.matches("\": (\"").count(); // entries like "Joy": ("joy", …)
        assert_eq!(engine, EMOTION_NAMES.len(), "emotions.py has {engine} emotions");
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
        // Hinglish alone is spoken by the Hindi voice; the reading order is Hindi, Hinglish, English
        let mix = Draft::from_params(Kind::Voice, &json!({"name": "v", "source": "files", "languages": ["mix"], "clips": [wav.to_string_lossy()]}));
        assert!(mix.clone().engine_args().unwrap().windows(2).any(|w| w == ["--languages", "hi"]));
        assert_eq!(selected_sets(&mix), vec!["mix".to_string()]);
        let all = Draft::from_params(Kind::Voice, &json!({"languages": ["en", "mix", "hi"]}));
        assert_eq!(selected_sets(&all), vec!["hi".to_string(), "mix".into(), "en".into()]);
        assert!(
            Draft::from_params(Kind::Voice, &json!({"name": "v", "source": "files", "languages": [], "clips": [wav.to_string_lossy()]})).engine_args().is_err()
        );
        // recording mode needs takes first
        assert!(Draft::from_params(Kind::Voice, &json!({"name": "v", "source": "record"})).engine_args().is_err());
        assert_eq!(slug("  My Voice / Hindi "), "My-Voice-Hindi");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn topic_video_arguments() {
        let mut d = Draft::from_params(Kind::Topic, &json!({"topic": "GTA 6 release", "seconds": 45, "voice": "mac:Rishi", "youtube": true, "cards": false}));
        let a = d.engine_args().unwrap();
        assert_eq!(a.get(..2), Some(&["topic".to_string(), "GTA 6 release".to_string()][..]));
        for pair in [["--seconds", "45"], ["--voice", "mac:Rishi"], ["--writer", "claude"], ["--aspect", "vertical"]] {
            assert!(a.windows(2).any(|w| w == pair), "{pair:?} in {a:?}");
        }
        let src = a.iter().position(|x| x == "--sources").unwrap();
        assert_eq!(a.get(src + 1..src + 3), Some(&["free".to_string(), "youtube".to_string()][..]));
        assert!(!a.contains(&"--no-captions".to_string()), "captions on by default");
        // script only: no voice needed
        let a = Draft::from_params(Kind::Topic, &json!({"topic": "https://example.com/post", "video": false})).engine_args().unwrap();
        assert!(a.contains(&"--script-only".to_string()) && !a.contains(&"--voice".to_string()));
        // a video needs a voice and a picture source
        assert!(Draft::from_params(Kind::Topic, &json!({"topic": "x"})).engine_args().is_err());
        assert!(Draft::from_params(Kind::Topic, &json!({"topic": "x", "voice": "v", "free": false, "cards": false})).engine_args().is_err());
        assert!(Draft::from_params(Kind::Topic, &json!({"topic": " "})).engine_args().is_err());
    }

    #[test]
    fn tv_light_arguments() {
        let dir = std::env::temp_dir().join(format!("ei-ai-test4-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (c1, c2) = (dir.join("a.mp4"), dir.join("b.mp4"));
        std::fs::write(&c1, b"x").unwrap();
        std::fs::write(&c2, b"x").unwrap();
        // a list of clips: one background job, output folder
        let mut d = Draft::from_params(
            Kind::Tv,
            &json!({"clips": [c1.to_string_lossy(), c2.to_string_lossy()], "recordings": dir.to_string_lossy(), "aspect": "horizontal", "relight": 0.6}),
        );
        let a = d.engine_args().unwrap();
        assert_eq!(a.first().map(String::as_str), Some("tv-make"));
        assert!(a.contains(&c1.to_string_lossy().to_string()) && a.contains(&c2.to_string_lossy().to_string()));
        for pair in [["--orientation", "horizontal"], ["--relight", "0.6"]] {
            assert!(a.windows(2).any(|w| w == pair), "{pair:?} in {a:?}");
        }
        assert!(a.windows(2).any(|w| w[0] == "--out-dir" && w[1].ends_with("EI on TV")), "{a:?}");
        assert!(d.out.ends_with("EI on TV"), "{}", d.out);
        assert!(!a.contains(&"--bezel".to_string()), "no bezel unless asked");
        let with_bezel =
            Draft::from_params(Kind::Tv, &json!({"clips": [c1.to_string_lossy()], "recordings": dir.to_string_lossy(), "bezel": true})).engine_args().unwrap();
        assert!(with_bezel.windows(2).any(|w| w == ["--bezel", "0.012"]));
        // one clip given the old way ("input") still works
        let a = Draft::from_params(Kind::Tv, &json!({"input": c1.to_string_lossy(), "recordings": dir.to_string_lossy(), "aspect": "vertical"}))
            .engine_args()
            .unwrap();
        assert!(a.windows(2).any(|w| w == ["--orientation", "vertical"]) && a.contains(&c1.to_string_lossy().to_string()));
        assert!(Draft::from_params(Kind::Tv, &json!({"clips": [c1.to_string_lossy()], "recordings": "/no/such"})).engine_args().is_err());
        assert!(Draft::from_params(Kind::Tv, &json!({"recordings": dir.to_string_lossy()})).engine_args().is_err());
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
