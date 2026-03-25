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
    Orion,
}

impl ModelPreset {
    pub const ALL: [ModelPreset; 2] = [
        ModelPreset::DeepSeek,
        ModelPreset::Orion,
    ];

    pub fn index(self) -> usize {
        match self {
            ModelPreset::DeepSeek => 0,
            ModelPreset::Orion => 1,
        }
    }

    pub fn from_index(i: usize) -> Self {
        match i {
            1 => ModelPreset::Orion,
            _ => ModelPreset::DeepSeek,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ModelPreset::DeepSeek => "deepseek-chat",
            ModelPreset::Orion => "Orion-Qwen3.5",
        }
    }

    pub fn llm_url(self) -> &'static str {
        match self {
            ModelPreset::DeepSeek => "https://api.deepseek.com",
            ModelPreset::Orion => "http://127.0.0.1:9633",
        }
    }

    pub fn model_name(self) -> &'static str {
        match self {
            ModelPreset::DeepSeek => "deepseek-chat",
            ModelPreset::Orion => "Orion-Qwen3.5_SFT_v2603",
        }
    }

    pub fn batch_size(self) -> usize {
        match self {
            ModelPreset::DeepSeek => 20,
            ModelPreset::Orion => 10,
        }
    }

    pub fn workers(self) -> usize {
        match self {
            ModelPreset::DeepSeek => 16,
            ModelPreset::Orion => 16,
        }
    }

    pub fn context_lines(self) -> usize {
        match self {
            ModelPreset::DeepSeek => 10,
            ModelPreset::Orion => 5,
        }
    }
}
