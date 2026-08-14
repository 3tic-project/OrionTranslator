use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::header::RETRY_AFTER;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use tracing::{debug, warn};

use crate::config;

use super::glossary::{self, GlossaryEntry};
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

/// DeepSeek / 火山方舟思考模式开关。
/// 文档：`{"thinking": {"type": "enabled"|"disabled"}}`
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct ThinkingConfig {
    #[serde(rename = "type")]
    thinking_type: &'static str,
}

impl ThinkingConfig {
    const DISABLED: Self = Self {
        thinking_type: "disabled",
    };
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
    /// 仅对 DeepSeek / 火山方舟等支持该字段的提供商发送，默认关闭思考以降低延迟与成本。
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
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

#[derive(Debug, thiserror::Error)]
enum RequestError {
    #[error("HTTP error {status}: {body}")]
    Http {
        status: StatusCode,
        body: String,
        retry_after: Option<Duration>,
    },
    #[error("LLM transport error: {0}")]
    Transport(#[source] reqwest::Error),
    #[error("LLM response protocol error: {0}")]
    Protocol(String),
}

impl RequestError {
    fn is_retryable(&self) -> bool {
        match self {
            Self::Http { status, .. } => {
                *status == StatusCode::REQUEST_TIMEOUT
                    || *status == StatusCode::TOO_EARLY
                    || *status == StatusCode::TOO_MANY_REQUESTS
                    || status.is_server_error()
            }
            Self::Transport(error) => {
                error.is_timeout() || error.is_connect() || error.is_request()
            }
            Self::Protocol(_) => true,
        }
    }

    fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::Http { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Http { status, .. } if *status == StatusCode::TOO_MANY_REQUESTS => "API 限流",
            Self::Transport(error) if error.is_timeout() => "请求超时",
            Self::Protocol(_) => "响应协议错误",
            _ => "请求失败",
        }
    }
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
    glossary_entries: Vec<GlossaryEntry>,
    api_key: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BatchTranslationResponse {
    pub translations: HashMap<usize, String>,
    pub diagnostics: ParseDiagnostics,
    pub contract: BatchContract,
}

/// 本地批次提交契约。模型仍使用兼容的 1..N 数字 JSONL key；该映射用于确保
/// 返回对象只会提交到创建它的稳定 UnitId 序列。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchContract {
    pub revision: String,
    pub unit_ids: Vec<String>,
}

impl BatchContract {
    pub fn new(unit_ids: Vec<String>, source_hashes: &[String]) -> Result<Self> {
        if unit_ids.is_empty() || unit_ids.len() != source_hashes.len() {
            anyhow::bail!(
                "批次契约单元数量无效: unit_ids={}, source_hashes={}",
                unit_ids.len(),
                source_hashes.len()
            );
        }
        let mut seen = HashSet::with_capacity(unit_ids.len());
        let mut hasher = Sha256::new();
        hasher.update(b"orion-batch-v1");
        for (unit_id, source_hash) in unit_ids.iter().zip(source_hashes) {
            if unit_id.is_empty() || source_hash.is_empty() {
                anyhow::bail!("批次契约含空 UnitId/source hash");
            }
            if !seen.insert(unit_id.as_str()) {
                anyhow::bail!("批次契约含重复 UnitId: {}", unit_id);
            }
            for value in [unit_id, source_hash] {
                hasher.update((value.len() as u64).to_be_bytes());
                hasher.update(value.as_bytes());
            }
        }
        Ok(Self {
            revision: format!("batch-v1:{:x}", hasher.finalize()),
            unit_ids,
        })
    }

    fn legacy(texts: &[String]) -> Result<Self> {
        let unit_ids: Vec<String> = (1..=texts.len())
            .map(|index| format!("legacy-position-{index}"))
            .collect();
        let source_hashes: Vec<String> = texts
            .iter()
            .map(|text| crate::unit_identity::source_sha256(text))
            .collect();
        Self::new(unit_ids, &source_hashes)
    }
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

    #[allow(clippy::too_many_arguments)]
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
        Self::with_params_and_glossary_entries(
            llm_url,
            model,
            max_retries,
            temperature,
            top_p,
            top_k,
            glossary_text,
            orion_glossary_text,
            api_key,
            Vec::new(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_params_and_glossary_entries(
        llm_url: &str,
        model: &str,
        max_retries: usize,
        temperature: f64,
        top_p: Option<f64>,
        top_k: Option<u32>,
        glossary_text: String,
        orion_glossary_text: Option<String>,
        api_key: Option<String>,
        glossary_entries: Vec<GlossaryEntry>,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            llm_url: llm_url.trim().to_string(),
            model: model.trim().to_string(),
            max_retries,
            temperature,
            top_p,
            top_k,
            glossary_text,
            orion_glossary_text,
            glossary_entries,
            api_key: api_key.and_then(|key| {
                let key = key.trim().to_string();
                if key.is_empty() {
                    None
                } else {
                    Some(key)
                }
            }),
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

    pub fn glossary_entries(&self) -> &[GlossaryEntry] {
        &self.glossary_entries
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

    fn request_glossary_scope(texts: &[String], context: &[String]) -> Vec<String> {
        let mut scope = Vec::with_capacity(context.len() + texts.len());
        scope.extend(context.iter().cloned());
        scope.extend(texts.iter().cloned());
        scope
    }

    fn common_glossary_for_request(&self, texts: &[String], context: &[String]) -> String {
        if self.glossary_entries.is_empty() {
            return self.glossary_text.clone();
        }

        let scope = Self::request_glossary_scope(texts, context);
        glossary::format_matched_glossary(&self.glossary_entries, &scope)
    }

    fn orion_glossary_for_request(&self, texts: &[String], context: &[String]) -> Option<String> {
        if self.glossary_entries.is_empty() {
            return self.orion_glossary_text.clone();
        }

        let scope = Self::request_glossary_scope(texts, context);
        glossary::format_matched_glossary_for_orion(&self.glossary_entries, &scope)
    }

    fn estimate_translation_max_tokens(
        &self,
        texts: &[String],
        context: &[String],
        glossary_chars: usize,
    ) -> u32 {
        let source_chars: usize = texts.iter().map(|text| text.chars().count()).sum();
        let context_chars: usize = context.iter().map(|text| text.chars().count()).sum();

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

    fn thinking_for_request(&self) -> Option<ThinkingConfig> {
        if config::should_disable_thinking(&self.llm_url, &self.model) {
            Some(ThinkingConfig::DISABLED)
        } else {
            None
        }
    }

    async fn call_with_max_tokens(
        &self,
        prompt: &str,
        batch_id: &str,
        max_tokens: u32,
    ) -> Result<Option<String>> {
        let endpoint = config::resolve_chat_completions_endpoint(&self.llm_url);
        let thinking = self.thinking_for_request();

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
            thinking,
        };

        debug!(
            "REQUEST [Batch {}]: endpoint={}, model={}, max_tokens={}, thinking={:?}",
            batch_id,
            endpoint,
            self.model,
            max_tokens,
            payload.thinking.as_ref().map(|t| t.thinking_type)
        );

        let mut last_error = None;
        let total_attempts = network_attempts(self.max_retries);
        for attempt in 0..total_attempts {
            match self.send_request(&endpoint, &payload).await {
                Ok(response_text) => {
                    debug!("RESPONSE [Batch {}]: len={}", batch_id, response_text.len());
                    return Ok(Some(response_text));
                }
                Err(error) => {
                    if !error.is_retryable() {
                        warn!("[永久请求错误] 不再重试 [Batch {}]: {}", batch_id, error);
                        return Err(error.into());
                    }
                    warn!(
                        "[{}] Attempt {}/{} [Batch {}]: {}",
                        error.label(),
                        attempt + 1,
                        total_attempts,
                        batch_id,
                        error
                    );
                    if attempt + 1 < total_attempts {
                        let delay = retry_delay(&error, attempt);
                        warn!(
                            "[Batch {}] 等待 {:.1}s 后重试...",
                            batch_id,
                            delay.as_secs_f64()
                        );
                        tokio::time::sleep(delay).await;
                    }
                    last_error = Some(error);
                }
            }
        }

        match last_error {
            Some(error) => Err(error.into()),
            None => anyhow::bail!("LLM 请求未执行"),
        }
    }

    async fn send_request(
        &self,
        endpoint: &str,
        payload: &ChatRequest,
    ) -> std::result::Result<String, RequestError> {
        let mut req = self.client.post(endpoint).json(payload);
        if let Some(key) = &self.api_key {
            if !key.is_empty() {
                req = req.header("Authorization", format!("Bearer {}", key));
            }
        }
        let response = req.send().await.map_err(RequestError::Transport)?;

        let status = response.status();
        if !status.is_success() {
            let retry_after = response
                .headers()
                .get(RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.trim().parse::<u64>().ok())
                .map(Duration::from_secs);
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<failed to read body>".to_string());
            let body = self.redact_secrets(body);
            return Err(RequestError::Http {
                status,
                body,
                retry_after,
            });
        }

        let data: ChatResponse = response
            .json()
            .await
            .map_err(|error| RequestError::Protocol(format!("JSON 解析失败: {error}")))?;

        data.choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .cloned()
            .ok_or_else(|| RequestError::Protocol("choices 为空或缺少 message.content".to_string()))
    }

    /// 测试模型：发送一条真实翻译格式的 prompt，验证返回是否可正常解析
    pub async fn test_translation(&self) -> Result<String> {
        let test_texts = vec!["今日はいい天気ですね。".to_string()];
        let context: Vec<String> = vec![];

        let prompt_text = if self.is_orion_model() {
            let glossary_text = self.orion_glossary_for_request(&test_texts, &context);
            prompt::build_prompt_with_context(&test_texts, &context, glossary_text.as_deref())
        } else {
            let glossary_text = self.common_glossary_for_request(&test_texts, &context);
            prompt::build_common_prompt_with_context(&test_texts, &context, &glossary_text)
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
        let contract = BatchContract::legacy(texts)?;
        self.translate_batch_detailed_with_contract(texts, context, batch_id, contract)
            .await
    }

    pub async fn translate_batch_detailed_with_contract(
        &self,
        texts: &[String],
        context: &[String],
        batch_id: &str,
        contract: BatchContract,
    ) -> Result<BatchTranslationResponse> {
        if texts.len() != contract.unit_ids.len() {
            anyhow::bail!(
                "批次文本与契约数量不一致: texts={}, units={}",
                texts.len(),
                contract.unit_ids.len()
            );
        }
        let (prompt_text, glossary_chars) = if self.is_orion_model() {
            let glossary_text = self.orion_glossary_for_request(texts, context);
            let glossary_chars = glossary_text
                .as_deref()
                .map(|text| text.chars().count())
                .unwrap_or(0);
            (
                prompt::build_prompt_with_context(texts, context, glossary_text.as_deref()),
                glossary_chars,
            )
        } else {
            let glossary_text = self.common_glossary_for_request(texts, context);
            let glossary_chars = glossary_text.chars().count();
            (
                prompt::build_common_prompt_with_context(texts, context, &glossary_text),
                glossary_chars,
            )
        };
        let max_tokens = self.estimate_translation_max_tokens(texts, context, glossary_chars);
        let response = self
            .call_with_max_tokens(&prompt_text, batch_id, max_tokens)
            .await?;

        match response {
            Some(text) => {
                let parsed = parse_jsonl_response_detailed(&text, texts.len());
                Ok(BatchTranslationResponse {
                    translations: parsed.translations,
                    diagnostics: parsed.diagnostics,
                    contract,
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
                    contract,
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
        let texts = vec![text.to_string()];
        let (prompt_text, glossary_chars) = if self.is_orion_model() {
            let glossary_text = self.orion_glossary_for_request(&texts, context);
            let glossary_chars = glossary_text
                .as_deref()
                .map(|text| text.chars().count())
                .unwrap_or(0);
            (
                prompt::build_single_prompt_with_context(text, context, glossary_text.as_deref()),
                glossary_chars,
            )
        } else {
            let glossary_text = self.common_glossary_for_request(&texts, context);
            let glossary_chars = glossary_text.chars().count();
            (
                prompt::build_common_single_prompt_with_context(text, context, &glossary_text),
                glossary_chars,
            )
        };
        let max_tokens = self.estimate_translation_max_tokens(&texts, context, glossary_chars);
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

fn retry_delay(error: &RequestError, attempt: usize) -> Duration {
    if let Some(delay) = error.retry_after() {
        return delay.min(Duration::from_secs(30));
    }
    let base_ms: u64 = match error {
        RequestError::Http { status, .. } if *status == StatusCode::TOO_MANY_REQUESTS => 3000,
        _ => 1000,
    };
    let exponent = (attempt as u32).min(5);
    Duration::from_millis(base_ms.saturating_mul(2u64.pow(exponent)).min(30_000))
}

fn network_attempts(max_retries: usize) -> usize {
    max_retries.saturating_add(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn spawn_mock_server(
        responses: Vec<String>,
    ) -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(AtomicUsize::new(0));
        let request_count = requests.clone();
        let task = tokio::spawn(async move {
            for response in responses {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let mut buffer = [0u8; 4096];
                loop {
                    let read = stream.read(&mut buffer).await.unwrap();
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&buffer[..read]);
                    if let Some(header_start) =
                        request.windows(4).position(|part| part == b"\r\n\r\n")
                    {
                        let body_start = header_start + 4;
                        let headers = String::from_utf8_lossy(&request[..header_start]);
                        let content_length = headers
                            .lines()
                            .find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().ok())
                                    .flatten()
                            })
                            .unwrap_or(0);
                        if request.len() >= body_start + content_length {
                            break;
                        }
                    }
                }
                request_count.fetch_add(1, Ordering::SeqCst);
                stream.write_all(response.as_bytes()).await.unwrap();
                stream.shutdown().await.unwrap();
            }
        });
        (
            format!("http://{address}/v1/chat/completions"),
            requests,
            task,
        )
    }

    fn mock_http_response(status: &str, extra_headers: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{extra_headers}\r\n{body}",
            body.len()
        )
    }

    fn successful_chat_response(content: &str) -> String {
        let body = serde_json::json!({
            "choices": [{"message": {"content": content}}]
        })
        .to_string();
        mock_http_response("200 OK", "", &body)
    }

    fn mock_client(url: &str, max_retries: usize, api_key: Option<String>) -> LlmClient {
        LlmClient::with_params(
            url,
            "mock-model",
            max_retries,
            0.0,
            None,
            None,
            String::new(),
            None,
            api_key,
        )
        .unwrap()
    }

    fn glossary_entry(src: &str, dst: &str) -> GlossaryEntry {
        GlossaryEntry {
            src: src.to_string(),
            dst: dst.to_string(),
            info: String::new(),
        }
    }

    #[test]
    fn classifies_http_status_without_parsing_error_strings() {
        let error = |status| RequestError::Http {
            status,
            body: String::new(),
            retry_after: None,
        };
        for status in [
            StatusCode::BAD_REQUEST,
            StatusCode::UNAUTHORIZED,
            StatusCode::FORBIDDEN,
            StatusCode::UNPROCESSABLE_ENTITY,
        ] {
            assert!(!error(status).is_retryable(), "{status}");
        }
        for status in [
            StatusCode::REQUEST_TIMEOUT,
            StatusCode::TOO_EARLY,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::INTERNAL_SERVER_ERROR,
        ] {
            assert!(error(status).is_retryable(), "{status}");
        }
    }

    #[test]
    fn retry_after_seconds_is_capped() {
        let error = RequestError::Http {
            status: StatusCode::TOO_MANY_REQUESTS,
            body: String::new(),
            retry_after: Some(Duration::from_secs(300)),
        };
        assert_eq!(retry_delay(&error, 0), Duration::from_secs(30));
    }

    #[test]
    fn zero_retry_budget_still_allows_the_initial_request() {
        assert_eq!(network_attempts(0), 1);
        assert_eq!(network_attempts(3), 4);
    }

    #[test]
    fn batch_contract_is_order_sensitive_and_rejects_duplicates() {
        let hashes = vec![
            crate::unit_identity::source_sha256("A"),
            crate::unit_identity::source_sha256("B"),
        ];
        let first = BatchContract::new(vec!["unit-a".into(), "unit-b".into()], &hashes).unwrap();
        let same = BatchContract::new(vec!["unit-a".into(), "unit-b".into()], &hashes).unwrap();
        let reordered =
            BatchContract::new(vec!["unit-b".into(), "unit-a".into()], &hashes).unwrap();

        assert_eq!(first, same);
        assert_ne!(first.revision, reordered.revision);
        assert!(BatchContract::new(vec!["dup".into(), "dup".into()], &hashes).is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn mock_unauthorized_is_not_retried_and_redacts_key() {
        let secret = "mock-secret-key";
        let body = format!(r#"{{"error":"invalid {secret}"}}"#);
        let (url, requests, server) =
            spawn_mock_server(vec![mock_http_response("401 Unauthorized", "", &body)]).await;
        let client = mock_client(&url, 3, Some(secret.to_string()));

        let error = client.call("test", "unauthorized").await.unwrap_err();

        server.await.unwrap();
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        let message = error.to_string();
        assert!(message.contains("401"));
        assert!(message.contains("[REDACTED_API_KEY]"));
        assert!(!message.contains(secret));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn mock_server_error_retries_once_then_succeeds() {
        let (url, requests, server) = spawn_mock_server(vec![
            mock_http_response("500 Internal Server Error", "", r#"{"error":"temporary"}"#),
            successful_chat_response(r#"{"1":"你好"}"#),
        ])
        .await;
        let client = mock_client(&url, 1, None);

        let response = client.call("test", "server-error").await.unwrap();

        server.await.unwrap();
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        assert_eq!(response.as_deref(), Some(r#"{"1":"你好"}"#));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn mock_rate_limit_honors_zero_retry_after_then_succeeds() {
        let (url, requests, server) = spawn_mock_server(vec![
            mock_http_response(
                "429 Too Many Requests",
                "Retry-After: 0\r\n",
                r#"{"error":"slow down"}"#,
            ),
            successful_chat_response(r#"{"1":"你好"}"#),
        ])
        .await;
        let client = mock_client(&url, 1, None);

        let response = client.call("test", "rate-limit").await.unwrap();

        server.await.unwrap();
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        assert_eq!(response.as_deref(), Some(r#"{"1":"你好"}"#));
    }

    #[test]
    fn orion_glossary_for_request_filters_to_current_scope() {
        let entries = vec![
            glossary_entry("ネギ", "涅吉"),
            glossary_entry("茶々丸", "茶茶丸"),
            glossary_entry("なのは", "奈叶"),
        ];
        let llm = LlmClient::with_params_and_glossary_entries(
            "http://127.0.0.1:9633/v1",
            "Orion-Qwen3-1.7B_SFT_v2605",
            1,
            0.3,
            Some(0.9),
            Some(20),
            glossary::format_glossary(&entries),
            glossary::format_glossary_for_orion(&entries),
            None,
            entries,
        )
        .unwrap();
        let texts = vec!["「茶々丸か？」".to_string()];
        let context = vec!["なぜネギとニンニク？".to_string()];

        let glossary_text = llm.orion_glossary_for_request(&texts, &context).unwrap();

        assert!(glossary_text.contains("ネギ→涅吉\n"));
        assert!(glossary_text.contains("茶々丸→茶茶丸\n"));
        assert!(!glossary_text.contains("なのは"));
    }

    #[test]
    fn deepseek_request_disables_thinking_by_default() {
        let llm = LlmClient::new("https://api.deepseek.com/v1", "deepseek-v4-flash", 1).unwrap();
        assert_eq!(llm.thinking_for_request(), Some(ThinkingConfig::DISABLED));

        let payload = ChatRequest {
            model: llm.model().to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            temperature: 0.8,
            max_tokens: 100,
            top_p: None,
            top_k: None,
            thinking: llm.thinking_for_request(),
        };
        let json = serde_json::to_value(&payload).unwrap();
        assert_eq!(json["thinking"]["type"], "disabled");
    }

    #[test]
    fn volcengine_request_disables_thinking_by_default() {
        let llm = LlmClient::new(
            "https://ark.cn-beijing.volces.com/api/v3",
            "ep-20250101-xxxxx",
            1,
        )
        .unwrap();
        assert_eq!(llm.thinking_for_request(), Some(ThinkingConfig::DISABLED));
    }

    #[test]
    fn orion_request_omits_thinking_field() {
        let llm = LlmClient::new("http://127.0.0.1:9633/v1", "Orion-Qwen3-1.7B-SFT", 1).unwrap();
        assert_eq!(llm.thinking_for_request(), None);

        let payload = ChatRequest {
            model: llm.model().to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            }],
            temperature: 0.3,
            max_tokens: 100,
            top_p: None,
            top_k: None,
            thinking: llm.thinking_for_request(),
        };
        let json = serde_json::to_value(&payload).unwrap();
        assert!(json.get("thinking").is_none());
    }
}
