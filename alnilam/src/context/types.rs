use std::collections::HashMap;

/// Line classification type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LineType {
    Blank,
    Metadata,
    ChapterHeading,
    SceneBreak,
    Dialogue,
    Narration,
}

impl std::fmt::Display for LineType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LineType::Blank => write!(f, "blank"),
            LineType::Metadata => write!(f, "metadata"),
            LineType::ChapterHeading => write!(f, "chapter_heading"),
            LineType::SceneBreak => write!(f, "scene_break"),
            LineType::Dialogue => write!(f, "dialogue"),
            LineType::Narration => write!(f, "narration"),
        }
    }
}

/// Result of detecting context needs for a line
#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub line_type: LineType,
    pub dominant_category: Option<String>,
    pub matches: Vec<crate::context::trie::KeywordMatch>,
    pub actions: HashMap<String, serde_json::Value>,
}

impl DetectionResult {
    pub fn reset_context(&self) -> bool {
        self.actions
            .get("reset_context")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }
}
