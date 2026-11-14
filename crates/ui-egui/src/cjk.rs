//! Chinese fallback fonts for document text in the interface, independent of UI language.
//! Optional craft-fonts are preferred; native builds without them reuse one installed face.

use std::sync::{Arc, OnceLock};

use egui::{FontData, FontDefinitions, FontFamily};

const SAMPLE: &str = "中文轨道配乐音效简体汉语繁體漢語车";
const PREFERRED: &[&str] = &[
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "PingFang SC",
    // macOS: in /System/Library/Fonts (PingFang is often only an on-demand system asset)
    "Hiragino Sans GB",
    "Heiti SC",
    "STHeiti",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Micro Hei",
    "Microsoft JhengHei UI",
    "Microsoft JhengHei",
    "Songti SC",
    "Arial Unicode MS",
];

/// Family names worth checking first when none of [`PREFERRED`] is installed.
fn likely_chinese(family: &str) -> bool {
    let f = family.to_ascii_lowercase();
    ["cjk", " sc", " tc", " gb", "hei", "song", "kai", "ming", "yuan", "fang", "han ", "hans", "hant", "chinese", "unicode"].iter().any(|k| f.contains(k))
}

pub(crate) fn craft_fonts() -> impl Iterator<Item = &'static filmcraft_text::fonts::CraftFont> {
    filmcraft_text::fonts::CRAFT_FONTS.iter().filter(|f| f.scripts.iter().any(|s| matches!(*s, "Hans" | "Hant")))
}

/// A font collection's face index matters: Windows Chinese UI fonts commonly live in TTC files.
/// Keep the owned bytes behind an Arc so theme reinstalls do not reread or leak a font file.
fn system_font() -> Option<Arc<FontData>> {
    static FONT: OnceLock<Option<Arc<FontData>>> = OnceLock::new();
    FONT.get_or_init(|| {
        if cfg!(target_arch = "wasm32") {
            return None;
        }
        filmcraft_text::fonts::scan_system();
        let faces: Vec<_> = filmcraft_text::fonts::all_faces().into_iter().filter(|f| f.info.origin == "system" && !f.info.italic).collect();
        // checked without keeping the files: testing every installed font once held all of them in
        // memory for as long as the app ran (3 GB on a Mac with the standard fonts)
        let covers = |f: &filmcraft_text::fonts::Face| f.covers_unloaded(SAMPLE);
        let by_weight = |f: &&Arc<filmcraft_text::fonts::Face>| f.info.weight.abs_diff(400);
        let preferred =
            PREFERRED.iter().find_map(|name| faces.iter().filter(|f| f.info.family.eq_ignore_ascii_case(name)).min_by_key(by_weight).filter(|f| covers(f)));
        let face = preferred
            .or_else(|| faces.iter().filter(|f| likely_chinese(&f.info.family)).filter(|f| f.info.weight.abs_diff(400) <= 100).find(|f| covers(f)))
            .or_else(|| faces.iter().find(|f| covers(f)))?;
        Some(Arc::new(FontData { font: std::borrow::Cow::Owned(face.data_unloaded()?), index: face.info.index, tweak: Default::default() }))
    })
    .clone()
}

/// Append Chinese faces after the Latin UI fonts and the Japanese fallbacks. This keeps the
/// interface's Latin typography and Japanese glyph forms, and covers the simplified and
/// traditional hanzi in media and track names that Japanese fonts lack.
pub(crate) fn install(fonts: &mut FontDefinitions) {
    let chinese: Vec<_> = craft_fonts().collect();
    if chinese.is_empty() {
        if let Some(font) = system_font() {
            fonts.font_data.insert("system-chinese".into(), font);
            for stack in fonts.families.values_mut() {
                stack.push("system-chinese".into());
            }
        }
        return;
    }
    let name = |f: &filmcraft_text::fonts::CraftFont| format!("craft:{} {}", f.family, f.style);
    for f in &chinese {
        fonts.font_data.insert(name(f), Arc::new(FontData::from_static(f.bytes)));
    }
    for (family, stack) in &mut fonts.families {
        let heavy = matches!(family, FontFamily::Name(n) if matches!(n.as_ref(), "semibold" | "medium"));
        let mut order = chinese.clone();
        order.sort_by_key(|f| (f.family.contains("Serif") && !f.family.contains("Sans"), (f.style == "Bold") != heavy));
        stack.extend(order.iter().map(|f| name(f)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: finding the Chinese fallback tested every installed font and kept each file in
    /// memory (3.2 GB of 678 faces on macOS), for as long as the app ran.
    #[test]
    fn finding_the_chinese_font_keeps_no_font_files() {
        let font = system_font();
        // a Mac always has one (Hiragino Sans GB, Heiti, Songti, Arial Unicode)
        assert!(!cfg!(target_os = "macos") || font.is_some());
        let loaded: Vec<String> = filmcraft_text::fonts::all_faces()
            .iter()
            .filter(|f| f.info.origin == "system" && f.is_loaded())
            .map(|f| format!("{} {}", f.info.family, f.info.style))
            .collect();
        assert!(loaded.len() <= 4, "system fonts left in memory: {loaded:?}");
        assert!(likely_chinese("Hiragino Sans GB") && likely_chinese("Noto Sans CJK SC") && !likely_chinese("Helvetica Neue"));
    }
}
