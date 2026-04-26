use std::path::PathBuf;

use anyhow::{bail, Result};

// ============================================================================
// Default Configuration Values
// ============================================================================

pub const DEFAULT_LLM_URL: &str = "https://api.deepseek.com/";
pub const DEFAULT_MODEL: &str = "deepseek-v4-flash";
pub const DEFAULT_BATCH_SIZE: usize = 20;
pub const DEFAULT_CONTEXT_LINES: usize = 10;
pub const DEFAULT_WORKERS: usize = 16;
pub const DEFAULT_MAX_RETRY: usize = 3;
pub const DEFAULT_CONTEXT_WINDOW_MULTIPLIER: usize = 3;
pub const DEFAULT_CONTEXT_WINDOW_MIN: usize = 30;
pub const TARGET_LANG: &str = "简体中文";
pub const DEFAULT_TRANSLATION_GAP: &str = "1rem";
pub const DEFAULT_TEMPERATURE: f64 = 0.8;
pub const DEFAULT_TOP_P: f64 = 0.9;
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

impl PipelineConfig {
    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0 || self.batch_size > 100 {
            bail!("批次大小必须在 1..=100 之间，当前为 {}", self.batch_size);
        }
        if self.workers == 0 || self.workers > 64 {
            bail!("并行数必须在 1..=64 之间，当前为 {}", self.workers);
        }
        if self.context_lines > 50 {
            bail!(
                "上下文行数必须在 0..=50 之间，当前为 {}",
                self.context_lines
            );
        }
        if self.max_retry > 10 {
            bail!("最大重试次数必须在 0..=10 之间，当前为 {}", self.max_retry);
        }
        if !(0.0..=2.0).contains(&self.temperature) || !self.temperature.is_finite() {
            bail!(
                "temperature 必须在 0.0..=2.0 之间，当前为 {}",
                self.temperature
            );
        }
        if !(0.0..=1.0).contains(&self.top_p) || !self.top_p.is_finite() {
            bail!("top_p 必须在 0.0..=1.0 之间，当前为 {}", self.top_p);
        }
        if self.top_k == 0 || self.top_k > 10_000 {
            bail!("top_k 必须在 1..=10000 之间，当前为 {}", self.top_k);
        }
        if reqwest::Url::parse(self.llm_url.trim()).is_err() {
            bail!("LLM API URL 无效: {}", self.llm_url);
        }
        if let Some(gap) = self.translation_gap.as_deref() {
            validate_css_length(gap)?;
        }

        Ok(())
    }

    pub fn api_security_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        let Ok(url) = reqwest::Url::parse(self.llm_url.trim()) else {
            return warnings;
        };

        if url.scheme() == "http" && !is_local_host(url.host_str()) {
            warnings.push(
                "API URL 使用明文 HTTP 且不是本机地址，正文和 API Key 可能被网络中间人读取"
                    .to_string(),
            );
        }

        if self
            .api_key
            .as_ref()
            .is_some_and(|key| !key.trim().is_empty())
            && url.scheme() != "https"
            && !is_local_host(url.host_str())
        {
            warnings
                .push("API Key 将发送到非 HTTPS 的远程地址，请确认这是可信内网服务".to_string());
        }

        warnings
    }
}

pub fn validate_css_length(value: &str) -> Result<()> {
    let trimmed = value.trim();
    if trimmed == "0" {
        return Ok(());
    }

    let allowed_units = ["rem", "px", "em", "%"];
    let Some(unit) = allowed_units.iter().find(|unit| trimmed.ends_with(**unit)) else {
        bail!("译文间距只允许 0 或 px/em/rem/% 长度，当前为 {}", value);
    };
    let number = trimmed[..trimmed.len() - unit.len()].trim();
    if number.is_empty() {
        bail!("译文间距缺少数值: {}", value);
    }
    let parsed: f64 = number
        .parse()
        .map_err(|_| anyhow::anyhow!("译文间距数值无效: {}", value))?;
    if !parsed.is_finite() || parsed < 0.0 || parsed > 20.0 {
        bail!("译文间距数值必须在 0..=20 之间，当前为 {}", value);
    }
    Ok(())
}

fn is_local_host(host: Option<&str>) -> bool {
    matches!(host, Some("localhost" | "127.0.0.1" | "::1" | "[::1]"))
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
    use super::{resolve_chat_completions_endpoint, validate_css_length};

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

    #[test]
    fn validates_safe_css_lengths() {
        assert!(validate_css_length("0").is_ok());
        assert!(validate_css_length("1rem").is_ok());
        assert!(validate_css_length("8px").is_ok());
        assert!(validate_css_length("0.5em").is_ok());
        assert!(validate_css_length("10%").is_ok());
        assert!(validate_css_length("1rem; color:red").is_err());
        assert!(validate_css_length("url(test)").is_err());
    }
}
