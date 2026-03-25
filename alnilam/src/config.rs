use std::path::PathBuf;

// ============================================================================
// Default Configuration Values
// ============================================================================

pub const DEFAULT_LLM_URL: &str = "https://api.deepseek.com/v1";
pub const DEFAULT_MODEL: &str = "deepseek-chat";
pub const DEFAULT_BATCH_SIZE: usize = 20;
pub const DEFAULT_CONTEXT_LINES: usize = 10;
pub const DEFAULT_WORKERS: usize = 16;
pub const DEFAULT_MAX_RETRY: usize = 3;
pub const DEFAULT_CONTEXT_WINDOW_MULTIPLIER: usize = 3;
pub const DEFAULT_CONTEXT_WINDOW_MIN: usize = 30;
pub const TARGET_LANG: &str = "简体中文";
pub const DEFAULT_TRANSLATION_GAP: &str = "1rem";
pub const DEFAULT_TEMPERATURE: f64 = 0.8;
pub const DEFAULT_TOP_P: f64 = 0.95;
pub const DEFAULT_TOP_K: u32 = 20;

// ============================================================================
// Translation Mode
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranslationMode {
    Bilingual,
    Replace,
}

impl std::fmt::Display for TranslationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TranslationMode::Bilingual => write!(f, "bilingual"),
            TranslationMode::Replace => write!(f, "replace"),
        }
    }
}

// ============================================================================
// Pipeline Configuration
// ============================================================================

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub llm_url: String,
    pub model: String,
    pub batch_size: usize,
    pub context_lines: usize,
    pub workers: usize,
    pub max_retry: usize,
    pub mode: TranslationMode,
    pub apply_fixes: bool,
    pub rules_path: Option<PathBuf>,
    /// Margin-bottom gap for bilingual translation paragraphs.
    /// None = no gap; Some("1rem") etc. = add inline style
    pub translation_gap: Option<String>,
    /// LLM generation temperature (0.0–2.0)
    pub temperature: f64,
    /// Top-P (nucleus sampling) threshold
    pub top_p: f64,
    /// Top-K sampling
    pub top_k: u32,
    /// 术语表 JSON 文件路径（通用模型使用）
    pub glossary_path: Option<PathBuf>,
    /// API 密钥（用于需要鉴权的 LLM 服务）
    pub api_key: Option<String>,
}

/// 将用户输入的 LLM 地址解析为最终的 chat completions endpoint。
///
/// 新格式推荐直接填写 BASE_URL，例如：
/// - https://api.deepseek.com/v1
/// - https://ark.cn-beijing.volces.com/api/v3
///
/// 同时保留对旧格式域名前缀和完整 endpoint 的兼容：
/// - https://api.deepseek.com               -> https://api.deepseek.com/v1/chat/completions
/// - https://api.deepseek.com/v1/chat/completions -> 原样返回
pub fn resolve_chat_completions_endpoint(raw_url: &str) -> String {
    let trimmed = raw_url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return format!("{}/chat/completions", DEFAULT_LLM_URL);
    }

    if trimmed.ends_with("/chat/completions") {
        return trimmed.to_string();
    }

    if let Ok(url) = reqwest::Url::parse(trimmed) {
        let path = url.path().trim_matches('/');
        if path.is_empty() {
            return format!("{}/v1/chat/completions", trimmed);
        }
    }

    format!("{}/chat/completions", trimmed)
}

#[cfg(test)]
mod tests {
    use super::resolve_chat_completions_endpoint;

    #[test]
    fn resolves_base_url_to_chat_endpoint() {
        assert_eq!(
            resolve_chat_completions_endpoint("https://api.deepseek.com/v1"),
            "https://api.deepseek.com/v1/chat/completions"
        );
        assert_eq!(
            resolve_chat_completions_endpoint("https://ark.cn-beijing.volces.com/api/v3"),
            "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        );
    }

    #[test]
    fn keeps_full_chat_endpoint_unchanged() {
        assert_eq!(
            resolve_chat_completions_endpoint("https://api.deepseek.com/v1/chat/completions"),
            "https://api.deepseek.com/v1/chat/completions"
        );
    }

    #[test]
    fn keeps_legacy_host_prefix_compatible() {
        assert_eq!(
            resolve_chat_completions_endpoint("https://api.deepseek.com"),
            "https://api.deepseek.com/v1/chat/completions"
        );
    }
}
