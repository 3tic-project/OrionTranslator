use std::path::PathBuf;
use std::time::Duration;

use gpui::*;

use crate::app::OrionApp;

// ============================================================================
// Utility Functions
// ============================================================================

impl OrionApp {
    pub fn format_duration_hms(duration: Duration) -> String {
        let secs = duration.as_secs();
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        let s = secs % 60;
        if h > 0 {
            format!("{:02}:{:02}:{:02}", h, m, s)
        } else {
            format!("{:02}:{:02}", m, s)
        }
    }

    pub fn add_log(&mut self, msg: &str) {
        let s: SharedString = msg.to_string().into();
        self.log_messages.push(s);
        if self.log_messages.len() > 200 {
            self.log_messages.drain(0..self.log_messages.len() - 200);
        }
        if self.auto_scroll_log {
            self.scroll_handle.scroll_to_bottom();
        }
    }

    pub fn joined_log_text(&self) -> String {
        self.log_messages
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn compute_output_paths(input: &PathBuf) -> (PathBuf, PathBuf) {
        let ext = input
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        let is_txt = ext == "txt";
        let stem = input.file_stem().unwrap_or_default().to_string_lossy();
        let parent = input.parent().unwrap_or(std::path::Path::new("."));

        if is_txt {
            (
                parent.join(format!("{}.ja-zh.txt", stem)),
                parent.join(format!("{}.zh.txt", stem)),
            )
        } else {
            (
                parent.join(format!("{}.ja-zh.epub", stem)),
                parent.join(format!("{}.zh.epub", stem)),
            )
        }
    }
}
