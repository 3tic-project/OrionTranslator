use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::config;

use super::parser::{parse_jsonl_response, parse_jsonl_response_detailed, ParseDiagnostics};
use super::prompt;

const DEFAULT_MAX_TOKENS: u32 = 3200;
const MIN_TRANSLATION_MAX_TOKENS: u32 = 1024;
const MAX_TRANSLATION_MAX_TOKENS: u32 = 12_000;

// ── API Types ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f64,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct ChatChoiceMessage {
    #[serde(default)]
    content: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

// ── LLM Client ───────────────────────────────────────────────────────────

pub struct LlmClient {
    client: reqwest::Client,
    llm_url: String,
    model: String,
    max_retries: usize,
    temperature: f64,
    top_p: Option<f64>,
    top_k: Option<u32>,
    glossary_text: String,
    /// Orion 模型专用术语表（与 SFT 训练格式一致：术语表：\nJA→ZH\n）
    orion_glossary_text: Option<String>,
    api_key: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BatchTranslationResponse {
    pub translations: HashMap<usize, String>,
    pub diagnostics: ParseDiagnostics,
}

impl LlmClient {
    pub fn new(llm_url: &str, model: &str, max_retries: usize) -> Result<Self> {
        Self::with_params(
            llm_url,
            model,
            max_retries,
            0.8,
            None,
            None,
            String::new(),
            None,
            None,
        )
    }

    pub fn with_params(
        llm_url: &str,
        model: &str,
        max_retries: usize,
        temperature: f64,
        top_p: Option<f64>,
        top_k: Option<u32>,
        glossary_text: String,
        orion_glossary_text: Option<String>,
        api_key: Option<String>,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            llm_url: llm_url.to_string(),
            model: model.to_string(),
            max_retries,
            temperature,
            top_p,
            top_k,
            glossary_text,
            orion_glossary_text,
            api_key,
        })
    }

    pub fn is_orion_model(&self) -> bool {
        self.model.to_lowercase().contains("orion")
    }

    pub fn llm_url(&self) -> &str {
        &self.llm_url
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    pub fn top_p(&self) -> Option<f64> {
        self.top_p
    }

    pub fn top_k(&self) -> Option<u32> {
        self.top_k
    }

    pub fn glossary_text(&self) -> &str {
        &self.glossary_text
    }

    pub fn orion_glossary_text(&self) -> Option<&str> {
        self.orion_glossary_text.as_deref()
    }

    pub fn api_key(&self) -> Option<&String> {
        self.api_key.as_ref()
    }

    fn redact_secrets(&self, text: String) -> String {
        if let Some(key) = self.api_key.as_deref() {
            let key = key.trim();
            if !key.is_empty() {
                return text.replace(key, "[REDACTED_API_KEY]");
            }
        }
        text
    }

    fn estimate_translation_max_tokens(&self, texts: &[String], context: &[String]) -> u32 {
        let source_chars: usize = texts.iter().map(|text| text.chars().count()).sum();
        let context_chars: usize = context.iter().map(|text| text.chars().count()).sum();
        let glossary_chars = self.glossary_text.chars().count()
            + self
                .orion_glossary_text
                .as_deref()
                .map(|text| text.chars().count())
                .unwrap_or(0);

        let source_budget = (source_chars as u32).saturating_mul(2);
        let structure_budget = (texts.len() as u32).saturating_mul(48).saturating_add(512);
        let context_margin = ((context_chars as u32) / 4).min(1_500);
        let glossary_margin = ((glossary_chars as u32) / 8).min(1_000);

        source_budget
            .saturating_add(structure_budget)
            .saturating_add(context_margin)
            .saturating_add(glossary_margin)
            .clamp(MIN_TRANSLATION_MAX_TOKENS, MAX_TRANSLATION_MAX_TOKENS)
    }

    /// Call the LLM API and return the raw response text
    pub async fn call(&self, prompt: &str, batch_id: &str) -> Result<Option<String>> {
        self.call_with_max_tokens(prompt, batch_id, DEFAULT_MAX_TOKENS)
            .await
    }

    async fn call_with_max_tokens(
        &self,
        prompt: &str,
        batch_id: &str,
        max_tokens: u32,
    ) -> Result<Option<String>> {
        let endpoint = config::resolve_chat_completions_endpoint(&self.llm_url);

        let payload = ChatRequest {
            model: self.model.clone(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: self.temperature,
            max_tokens,
            top_p: self.top_p,
            top_k: self.top_k,
        };

        debug!(
            "REQUEST [Batch {}]: endpoint={}, model={}, max_tokens={}",
            batch_id, endpoint, self.model, max_tokens
        );

        for attempt in 0..self.max_retries {
            match self.send_request(&endpoint, &payload).await {
                Ok(response_text) => {
                    debug!("RESPONSE [Batch {}]: len={}", batch_id, response_text.len());
                    return Ok(Some(response_text));
                }
                Err(e) => {
                    let err_str = e.to_string();
                    let is_rate_limit = err_str.contains("429") || err_str.contains("rate");
                    let is_timeout = err_str.contains("timed out") || err_str.contains("timeout");
                    let label = if is_rate_limit {
                        "API 限流"
                    } else if is_timeout {
                        "请求超时"
                    } else {
                        "请求失败"
                    };
                    warn!(
                        "[{}] Attempt {}/{} [Batch {}]: {}",
                        label,
                        attempt + 1,
                        self.max_retries,
                        batch_id,
                        e
                    );
                    if attempt + 1 < self.max_retries {
                        // 限流时使用更长的退避时间
                        let base_ms = if is_rate_limit { 3000 } else { 1000 };
                        let delay = Duration::from_millis(base_ms * 2u64.pow(attempt as u32));
                        warn!(
                            "[Batch {}] 等待 {:.1}s 后重试...",
                            batch_id,
                            delay.as_secs_f64()
                        );
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        Ok(None)
    }

    async fn send_request(&self, endpoint: &str, payload: &ChatRequest) -> Result<String> {
        let mut req = self.client.post(endpoint).json(payload);
        if let Some(key) = &self.api_key {
            if !key.is_empty() {
                req = req.header("Authorization", format!("Bearer {}", key));
            }
        }
        let response = req.send().await.context("Failed to send request to LLM")?;

        let status = response.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<failed to read body>".to_string());
            let body = self.redact_secrets(body);
            anyhow::bail!("API 限流 (429): {}", body);
        }
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<failed to read body>".to_string());
            let body = self.redact_secrets(body);
            anyhow::bail!("HTTP error {}: {}", status, body);
        }

        let data: ChatResponse = response
            .json()
            .await
            .context("Failed to parse LLM response")?;

        data.choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No content in LLM response"))
    }

    /// 测试模型：发送一条真实翻译格式的 prompt，验证返回是否可正常解析
    pub async fn test_translation(&self) -> Result<String> {
        let test_texts = vec!["今日はいい天気ですね。".to_string()];
        let context: Vec<String> = vec![];

        let prompt_text = if self.is_orion_model() {
            prompt::build_prompt_with_context(
                &test_texts,
                &context,
                self.orion_glossary_text.as_deref(),
            )
        } else {
            prompt::build_common_prompt_with_context(&test_texts, &context, &self.glossary_text)
        };

        let response = self.call(&prompt_text, "model-test").await?;
        match response {
            Some(text) => {
                let parsed = parse_jsonl_response(&text, 1);
                if let Some(translated) = parsed.get(&1) {
                    if !translated.is_empty() {
                        Ok(translated.clone())
                    } else {
                        anyhow::bail!("解析成功但译文为空")
                    }
                } else {
                    anyhow::bail!("无法解析 JSONL 响应: {}", text)
                }
            }
            None => anyhow::bail!("模型未返回结果"),
        }
    }

    /// Translate a batch of texts with context
    pub async fn translate_batch(
        &self,
        texts: &[String],
        context: &[String],
        batch_id: &str,
    ) -> Result<HashMap<usize, String>> {
        Ok(self
            .translate_batch_detailed(texts, context, batch_id)
            .await?
            .translations)
    }

    /// Translate a batch of texts with parse diagnostics.
    pub async fn translate_batch_detailed(
        &self,
        texts: &[String],
        context: &[String],
        batch_id: &str,
    ) -> Result<BatchTranslationResponse> {
        let prompt_text = if self.is_orion_model() {
            prompt::build_prompt_with_context(texts, context, self.orion_glossary_text.as_deref())
        } else {
            prompt::build_common_prompt_with_context(texts, context, &self.glossary_text)
        };
        let max_tokens = self.estimate_translation_max_tokens(texts, context);
        let response = self
            .call_with_max_tokens(&prompt_text, batch_id, max_tokens)
            .await?;

        match response {
            Some(text) => {
                let parsed = parse_jsonl_response_detailed(&text, texts.len());
                Ok(BatchTranslationResponse {
                    translations: parsed.translations,
                    diagnostics: parsed.diagnostics,
                })
            }
            None => {
                warn!("Failed to get response for batch {}", batch_id);
                Ok(BatchTranslationResponse {
                    translations: HashMap::new(),
                    diagnostics: ParseDiagnostics {
                        missing_indices: (1..=texts.len()).collect(),
                        ..ParseDiagnostics::default()
                    },
                })
            }
        }
    }

    /// Translate a single text with context
    pub async fn translate_single(
        &self,
        text: &str,
        context: &[String],
        batch_id: &str,
    ) -> Result<Option<String>> {
        let prompt_text = if self.is_orion_model() {
            prompt::build_single_prompt_with_context(
                text,
                context,
                self.orion_glossary_text.as_deref(),
            )
        } else {
            prompt::build_common_single_prompt_with_context(text, context, &self.glossary_text)
        };
        let max_tokens = self.estimate_translation_max_tokens(&[text.to_string()], context);
        let response = self
            .call_with_max_tokens(&prompt_text, batch_id, max_tokens)
            .await?;
        // Orion 模型现在也输出 JSONL 格式（与 SFT 训练一致），统一用 JSONL 解析
        match response {
            Some(text) => {
                let parsed = parse_jsonl_response(&text, 1);
                Ok(parsed.get(&1).cloned())
            }
            None => Ok(None),
        }
    }
}
