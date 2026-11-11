//! The EI wordmark, drawn in code (no image asset): "EI" in heavy type on a rounded tile in the
//! app's accent colour, the same look as the app icon.

/// Draw the wordmark `height` points tall with its left edge at `left_center` (vertically centred).
/// Returns the drawn rect. `dark` is kept for callers that pick ink by theme (the tile reads on both).
pub fn paint_wordmark(ui: &egui::Ui, left_center: egui::Pos2, height: f32, dark: bool) -> Option<egui::Rect> {
    let _ = dark;
    let h = height.max(1.0) * 1.5;
    let font = egui::FontId::new(h * 0.62, egui::FontFamily::Proportional);
    let galley = ui.painter().layout_no_wrap("EI".to_string(), font, egui::Color32::WHITE);
    let w = galley.size().x + h * 0.5;
    let r = egui::Rect::from_min_size(egui::pos2(left_center.x, left_center.y - h / 2.0), egui::vec2(w, h));
    ui.painter().rect_filled(r, h * 0.22, ACCENT);
    let pos = egui::pos2(r.center().x - galley.size().x / 2.0, r.center().y - galley.size().y / 2.0);
    // Draw twice, offset by a hair, for a heavier weight than the bundled UI font's regular.
    ui.painter().galley(pos, galley.clone(), egui::Color32::WHITE);
    ui.painter().galley(pos + egui::vec2(h * 0.02, 0.0), galley, egui::Color32::WHITE);
    Some(r)
}

/// EI's brand colour (also the app icon's background).
pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(0x25, 0x63, 0xEB);
