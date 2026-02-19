use gpui::*;

// ============================================================================
// Actions
// ============================================================================

actions!(alnitak, [Quit]);

// ============================================================================
// Translation State
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum TranslationStatus {
    Idle,
    Running,
    Completed,
    Cancelled,
    Failed,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GlossaryGenStatus {
    Idle,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelPreset {
    DeepSeek,
    OrionHYMT,
    OrionQwen,
}

impl ModelPreset {
    pub const ALL: [ModelPreset; 3] = [
        ModelPreset::DeepSeek,
        ModelPreset::OrionHYMT,
        ModelPreset::OrionQwen,
    ];

    pub fn index(self) -> usize {
        match self {
            ModelPreset::DeepSeek => 0,
            ModelPreset::OrionHYMT => 1,
            ModelPreset::OrionQwen => 2,
        }
    }

    pub fn from_index(i: usize) -> Self {
        match i {
            1 => ModelPreset::OrionHYMT,
            2 => ModelPreset::OrionQwen,
            _ => ModelPreset::DeepSeek,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ModelPreset::DeepSeek => "deepseek-chat",
            ModelPreset::OrionHYMT => "Orion-HYMT",
            ModelPreset::OrionQwen => "Orion-Qwen",
        }
    }

    pub fn llm_url(self) -> &'static str {
        match self {
            ModelPreset::DeepSeek => "https://api.deepseek.com",
            ModelPreset::OrionHYMT => "http://127.0.0.1:9633",
            ModelPreset::OrionQwen => "http://127.0.0.1:9633",
        }
    }

    pub fn model_name(self) -> &'static str {
        match self {
            ModelPreset::DeepSeek => "deepseek-chat",
            ModelPreset::OrionHYMT => "Orion-HYMT1.5-1.8B-SFT-v2601",
            ModelPreset::OrionQwen => "Orion-Qwen3-1.7B-SFT-v2601",
        }
    }

    pub fn batch_size(self) -> usize {
        match self {
            ModelPreset::DeepSeek => 20,
            ModelPreset::OrionHYMT | ModelPreset::OrionQwen => 10,
        }
    }

    pub fn workers(self) -> usize {
        match self {
            ModelPreset::DeepSeek => 16,
            ModelPreset::OrionHYMT | ModelPreset::OrionQwen => 16,
        }
    }

    pub fn context_lines(self) -> usize {
        match self {
            ModelPreset::DeepSeek => 10,
            ModelPreset::OrionHYMT | ModelPreset::OrionQwen => 5,
        }
    }
}
