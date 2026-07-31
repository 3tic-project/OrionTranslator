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
    CompletedWithErrors,
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
    Volcengine,
    Orion,
}

impl ModelPreset {
    pub const ALL: [ModelPreset; 3] = [
        ModelPreset::DeepSeek,
        ModelPreset::Volcengine,
        ModelPreset::Orion,
    ];

    pub fn index(self) -> usize {
        match self {
            ModelPreset::DeepSeek => 0,
            ModelPreset::Volcengine => 1,
            ModelPreset::Orion => 2,
        }
    }

    pub fn from_index(i: usize) -> Self {
        match i {
            1 => ModelPreset::Volcengine,
            2 => ModelPreset::Orion,
            _ => ModelPreset::DeepSeek,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ModelPreset::DeepSeek => "Deepseek",
            ModelPreset::Volcengine => "火山引擎",
            ModelPreset::Orion => "Orion",
        }
    }

    pub fn llm_url(self) -> &'static str {
        match self {
            ModelPreset::DeepSeek => "https://api.deepseek.com/",
            ModelPreset::Volcengine => "https://ark.cn-beijing.volces.com/api/v3",
            ModelPreset::Orion => "http://127.0.0.1:9633/v1",
        }
    }

    pub fn model_name(self) -> &'static str {
        match self {
            ModelPreset::DeepSeek => "deepseek-v4-flash",
            ModelPreset::Volcengine => "",
            ModelPreset::Orion => "Orion-Qwen3-1.7B-SFT",
        }
    }

    pub fn batch_size(self) -> usize {
        match self {
            ModelPreset::DeepSeek | ModelPreset::Volcengine => 15,
            ModelPreset::Orion => 10,
        }
    }

    pub fn workers(self) -> usize {
        match self {
            ModelPreset::DeepSeek | ModelPreset::Volcengine => 32,
            ModelPreset::Orion => 16,
        }
    }

    pub fn context_lines(self) -> usize {
        match self {
            ModelPreset::DeepSeek | ModelPreset::Volcengine => 10,
            ModelPreset::Orion => 5,
        }
    }

    pub fn temperature(self) -> f64 {
        match self {
            ModelPreset::DeepSeek | ModelPreset::Volcengine => 0.7,
            ModelPreset::Orion => 0.8,
        }
    }

    /// 持久化文件中的预设键名（见 `credentials` 模块）。
    pub fn storage_key(self) -> &'static str {
        match self {
            ModelPreset::DeepSeek => "deepseek",
            ModelPreset::Volcengine => "volcengine",
            ModelPreset::Orion => "orion",
        }
    }

    pub fn from_storage_key(key: &str) -> Self {
        match key.trim().to_ascii_lowercase().as_str() {
            "volcengine" | "volc" | "ark" => ModelPreset::Volcengine,
            "orion" => ModelPreset::Orion,
            _ => ModelPreset::DeepSeek,
        }
    }
}
