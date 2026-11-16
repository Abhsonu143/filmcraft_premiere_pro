//! Project links (Help menu, About dialog).
//!
//! EI is a fork of the open-source FilmCraft editor; the About dialog and Help menu credit it.

/// EI's source repository.
pub const GITHUB: &str = "https://github.com/Abhsonu143/filmcraft_premiere_pro";
/// New issue on EI's repository.
pub const ISSUES: &str = "https://github.com/Abhsonu143/filmcraft_premiere_pro/issues";
/// The upstream FilmCraft project EI is based on.
pub const UPSTREAM: &str = "https://github.com/storytold/filmcraft";

/// (command id, label, url) for every link, in menu order.
pub const ALL: [(&str, &str, &str); 3] =
    [("help.github", "EI on GitHub", GITHUB), ("help.reportIssue", "Report an Issue…", ISSUES), ("help.upstream", "Based on FilmCraft…", UPSTREAM)];

/// The URL a `help.*` link command opens.
pub fn url_for(command: &str) -> Option<&'static str> {
    ALL.iter().find(|(id, _, _)| *id == command).map(|(_, _, u)| *u)
}

/// Open `url` in the system browser (a new tab on the web).
pub fn open(ctx: &egui::Context, url: &str) {
    ctx.open_url(egui::OpenUrl::new_tab(url));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn links_are_https_help_commands() {
        assert!(ALL.iter().all(|(id, _, u)| id.starts_with("help.") && u.starts_with("https://")));
        assert_eq!(url_for("help.github"), Some(GITHUB));
        assert_eq!(url_for("help.discord"), None);
        assert_eq!(url_for("help.nope"), None);
        assert!(ALL.iter().all(|(_, label, url)| !label.contains("ArtCraft") && !url.contains("artcraft")));
    }
}
