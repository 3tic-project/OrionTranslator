use crate::detector::{CharacterInfo, Mention};
use crate::{emit, GlossaryProgressCallback, GlossaryProgressEvent};
use anyhow::Result;
use log::warn;
use reqwest::Client;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

const SYSTEM_PROMPT: &str = r#"你是一个轻小说翻译助手。现在给出一段通过NER识别出的"候选人物称呼/人名"，请你根据上下文判断它是否为人物名（不是地名/组织/家庭/职务称谓等）。

你必须只输出一个JSON对象（不要Markdown、不要解释），格式如下：
{"is_name": true|false, "gender": "男性"|"女性"|"动物"|null, "full_name": string|null, "translated_chinese_name": string|null}

规则（必须严格遵守）：
1) 如果不是人物名：is_name=false，其他字段全部为null。
2) 如果是"纯称谓/职务/关系称呼"且不包含具体人名（如：先生/部長/先輩/お兄様等）：视为非人物名，is_name=false。
3) translated_chinese_name 必须是简体中文或常用汉字（允许保留原本为汉字的人名写法），但禁止包含任何平假名/片假名/半角片假名；禁止包含空格。
4) 如果无法确定性别或全名：对应字段返回null，不要猜。
5) full_name 仅在上下文出现明确全名或强证据时填写，否则为null。
6) translated_chinese_name 必须针对"输入的称呼/人名本身"给出翻译（不要把 full_name 直接当作翻译名）。
7) 输入已经去除了さん/ちゃん/先生/先輩等称谓；translated_chinese_name 不得自行补回老师、前辈、小姐等称谓。"#;

const FINAL_REVIEW_SYSTEM_PROMPT: &str = r#"你是轻小说人物术语表的总审校。输入是一批初步NER+LLM结果及上下文，你需要在整批范围内消除误识别、边界污染、重复人物和译名不一致。

只输出一个JSON对象（不要Markdown、不要解释）：
{"decisions":[{"src":"必须原样复制输入src","keep":true|false,"dst":"保留时的最终简体中文译名或null","gender":"男性"|"女性"|"动物"|null,"full_name":"明确全名或null","reason":"简短原因"}]}

规则：
1) 每个输入src必须恰好返回一次，不得新增、合并或遗漏src。
2) NER边界沾上助词/副词/上下文片段、普通词、地点、组织、宫殿、物品、目录项、纯职务/身份称谓时 keep=false。
3) 姓名加先生/先輩/様等仍可保留，但dst称谓只能出现一次；同一人物的全名、姓、名、昵称和带敬称形式必须使用一致的核心译名。
4) 不要仅因短名就删除；结合上下文和同批长名判断。证据不足时保持初步结果，不要凭空改名。
5) dst只能是简体中文/常用汉字，不得含假名、空白，不得出现“老师老师”等重复称谓。
6) gender/full_name只有明确证据时填写；无法确定返回null。"#;

const MAX_CONTEXT_ITEMS: usize = 10;
const MAX_CONTEXT_CHARS_PER_ITEM: usize = 220;
const FINAL_REVIEW_BATCH_SIZE: usize = 24;
const FINAL_REVIEW_CONTEXT_ITEMS: usize = 2;
const FINAL_REVIEW_CONTEXT_CHARS: usize = 180;

const HONORIFIC_SUFFIXES: &[&str] = &[
    "さん",
    "ちゃん",
    "くん",
    "君",
    "様",
    "さま",
    "殿",
    "どの",
    "先輩",
    "先生",
    "部長",
    "会長",
    "委員長",
    "店長",
    "課長",
    "社長",
    "監督",
    "姉",
    "兄",
    "妹",
    "弟",
    "姉さん",
    "兄さん",
    "姉ちゃん",
    "兄ちゃん",
];

const HONORIFIC_PREFIXES: &[&str] = &["お", "ご"];

const PURE_TITLE_CORES: &[&str] = &[
    "先生",
    "部長",
    "先輩",
    "会長",
    "委員長",
    "店長",
    "課長",
    "社長",
    "監督",
    "校長",
    "副会長",
    "副部長",
    "王様",
    "お兄様",
    "お姉様",
    "お兄さん",
    "お姉さん",
    "お兄ちゃん",
    "お姉ちゃん",
];

/// DeepSeek / 火山方舟思考模式开关：`{"thinking":{"type":"disabled"}}`
#[derive(Debug, Clone, Serialize)]
struct ThinkingConfig {
    #[serde(rename = "type")]
    thinking_type: &'static str,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    /// 仅对 DeepSeek / 火山方舟发送，默认关闭思考。
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
}

fn thinking_for_provider(api_url: &str, model: &str) -> Option<ThinkingConfig> {
    let model_l = model.trim().to_ascii_lowercase();
    let url_l = api_url.trim().to_ascii_lowercase();
    let disable = model_l.contains("deepseek")
        || url_l.contains("deepseek.com")
        || url_l.contains("volces.com")
        || url_l.contains("volcengine.com");
    if disable {
        Some(ThinkingConfig {
            thinking_type: "disabled",
        })
    } else {
        None
    }
}

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: String,
}

fn first_response_content(response: &ChatResponse) -> Result<&str> {
    response
        .choices
        .first()
        .map(|choice| choice.message.content.as_str())
        .ok_or_else(|| anyhow::anyhow!("LLM response choices 为空"))
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlmResult {
    #[serde(default)]
    pub is_name: bool,
    pub gender: Option<String>,
    pub full_name: Option<String>,
    pub translated_chinese_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationEntry {
    pub src: String,
    pub dst: String,
    pub info: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GlossaryIssueKind {
    Rejected,
    Unresolved,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GlossaryGenerationIssue {
    pub source: String,
    pub aliases: Vec<String>,
    pub kind: GlossaryIssueKind,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GlossaryReviewChange {
    pub source: String,
    pub before_dst: String,
    pub after_dst: Option<String>,
    pub before_info: String,
    pub after_info: Option<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GlossaryTranslationReport {
    pub entries: Vec<TranslationEntry>,
    pub issues: Vec<GlossaryGenerationIssue>,
    pub review_changes: Vec<GlossaryReviewChange>,
    pub review_batches: usize,
    pub review_failures: usize,
}

impl GlossaryTranslationReport {
    pub fn unresolved_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|issue| issue.kind == GlossaryIssueKind::Unresolved)
            .count()
    }

    pub fn rejected_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|issue| issue.kind == GlossaryIssueKind::Rejected)
            .count()
    }
}

pub struct LlmClient {
    client: Client,
    api_url: String,
    api_key: String,
    model: String,
}

#[derive(Clone)]
struct AliasInfo {
    name: String,
    count: usize,
    mentions: Vec<Mention>,
}

#[derive(Clone)]
struct NameCluster {
    key: String,
    aliases: Vec<AliasInfo>,
    primary: String,
}

impl LlmClient {
    pub fn new(api_url: &str, api_key: &str, model: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            api_url: normalize_chat_completions_endpoint(api_url),
            api_key: api_key.trim().to_string(),
            model: model.trim().to_string(),
        }
    }

    /// Translate all character names concurrently
    pub async fn translate_all(
        &self,
        characters: &HashMap<String, CharacterInfo>,
        max_concurrent: usize,
        progress: GlossaryProgressCallback,
    ) -> Vec<TranslationEntry> {
        self.translate_all_detailed(characters, max_concurrent, progress)
            .await
            .entries
    }

    pub async fn translate_all_detailed(
        &self,
        characters: &HashMap<String, CharacterInfo>,
        max_concurrent: usize,
        progress: GlossaryProgressCallback,
    ) -> GlossaryTranslationReport {
        // 0 permits 会导致 acquire 永久阻塞
        let max_concurrent = max_concurrent.max(1);
        let (clusters, preprocessing_issues) = build_name_clusters(characters);
        let total = clusters.len();
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent));
        let client = self.client.clone();
        let api_url = self.api_url.clone();
        let api_key = self.api_key.clone();
        let model = self.model.clone();

        let completed_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let mut handles = Vec::new();
        for cluster in clusters {
            let task_source = cluster.key.clone();
            let task_aliases = cluster
                .aliases
                .iter()
                .map(|alias| alias.name.clone())
                .collect::<Vec<_>>();
            let sem = semaphore.clone();
            let client = client.clone();
            let api_url = api_url.clone();
            let api_key = api_key.clone();
            let model = model.clone();
            let progress = progress.clone();
            let completed = completed_count.clone();

            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                let result = translate_cluster(&client, &api_url, &api_key, &model, cluster).await;
                let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                emit(
                    &progress,
                    GlossaryProgressEvent::LlmProgress {
                        completed: done,
                        total,
                    },
                );
                result
            });
            handles.push((task_source, task_aliases, handle));
        }

        let mut report = GlossaryTranslationReport {
            issues: preprocessing_issues,
            ..GlossaryTranslationReport::default()
        };
        for (source, aliases, handle) in handles {
            match handle.await {
                Ok(mut outcome) => {
                    report.entries.append(&mut outcome.entries);
                    if let Some(issue) = outcome.issue {
                        report.issues.push(issue);
                    }
                }
                Err(error) => {
                    warn!("LLM任务失败: {}", error);
                    report.issues.push(GlossaryGenerationIssue {
                        source,
                        aliases,
                        kind: GlossaryIssueKind::Unresolved,
                        reason: format!("task join failure: {error}"),
                    });
                }
            }
        }

        propagate_gender_within_canonical(&mut report.entries);
        self.final_review(characters, max_concurrent, &mut report, progress)
            .await;
        report
    }

    async fn final_review(
        &self,
        characters: &HashMap<String, CharacterInfo>,
        max_concurrent: usize,
        report: &mut GlossaryTranslationReport,
        progress: GlossaryProgressCallback,
    ) {
        if report.entries.is_empty() {
            return;
        }

        let batches = build_final_review_batches(&report.entries);
        report.review_batches = batches.len();
        emit(
            &progress,
            GlossaryProgressEvent::StageStarted {
                stage: "术语总审校".to_string(),
                detail: format!(
                    "分 {} 个关联批次复核误识别、边界和跨簇译名一致性...",
                    batches.len()
                ),
            },
        );

        let semaphore =
            std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent.clamp(1, 3)));
        let mut handles = Vec::new();
        for (index, batch) in batches.into_iter().enumerate() {
            let payload = build_final_review_payload(&batch, characters);
            let expected_sources = batch
                .iter()
                .map(|entry| entry.src.clone())
                .collect::<Vec<_>>();
            let sem = semaphore.clone();
            let client = self.client.clone();
            let api_url = self.api_url.clone();
            let api_key = self.api_key.clone();
            let model = self.model.clone();
            let task_sources = expected_sources.clone();
            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.expect("review semaphore closed");
                review_batch(&client, &api_url, &api_key, &model, &payload, &task_sources).await
            });
            handles.push((index, expected_sources, handle));
        }

        let mut decisions = HashMap::new();
        for (index, sources, handle) in handles {
            match handle.await {
                Ok(Ok(batch_decisions)) => {
                    for decision in batch_decisions {
                        decisions.insert(decision.src.clone(), decision);
                    }
                }
                Ok(Err(error)) => {
                    report.review_failures += 1;
                    report.issues.push(GlossaryGenerationIssue {
                        source: format!("final_review_batch_{}", index + 1),
                        aliases: sources,
                        kind: GlossaryIssueKind::Unresolved,
                        reason: format!("总审校失败，已保留初步结果: {error}"),
                    });
                }
                Err(error) => {
                    report.review_failures += 1;
                    report.issues.push(GlossaryGenerationIssue {
                        source: format!("final_review_batch_{}", index + 1),
                        aliases: sources,
                        kind: GlossaryIssueKind::Unresolved,
                        reason: format!("总审校任务失败，已保留初步结果: {error}"),
                    });
                }
            }
        }

        let (reviewed, changes) =
            apply_final_review_decisions(std::mem::take(&mut report.entries), decisions);
        report.entries = reviewed;
        report.review_changes.extend(changes);
        propagate_gender_within_canonical(&mut report.entries);

        emit(
            &progress,
            GlossaryProgressEvent::Log {
                message: format!(
                    "术语总审校完成：{} 个批次，{} 处调整，{} 个批次回退初步结果",
                    report.review_batches,
                    report.review_changes.len(),
                    report.review_failures
                ),
            },
        );
    }
}

#[derive(Debug, Serialize)]
struct FinalReviewItem {
    src: String,
    initial_dst: String,
    initial_info: String,
    canonical_hint: String,
    count: usize,
    contexts: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct FinalReviewResponse {
    decisions: Vec<FinalReviewDecision>,
}

#[derive(Debug, Clone, Deserialize)]
struct FinalReviewDecision {
    src: String,
    keep: bool,
    dst: Option<String>,
    gender: Option<String>,
    full_name: Option<String>,
    reason: String,
}

async fn review_batch(
    client: &Client,
    api_url: &str,
    api_key: &str,
    model: &str,
    payload: &[FinalReviewItem],
    expected_sources: &[String],
) -> Result<Vec<FinalReviewDecision>> {
    let user_content = serde_json::to_string(payload)?;
    let mut last_error = None;
    for attempt in 1..=2 {
        let request = ChatRequest {
            model: model.to_string(),
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: FINAL_REVIEW_SYSTEM_PROMPT.to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: user_content.clone(),
                },
            ],
            temperature: 0.0,
            thinking: thinking_for_provider(api_url, model),
        };

        let outcome = async {
            let response = client
                .post(api_url)
                .header("Authorization", format!("Bearer {api_key}"))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await?;
            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                anyhow::bail!("LLM API returned {status}: {body}");
            }
            let chat: ChatResponse = response.json().await?;
            let parsed: FinalReviewResponse = parse_json_object(first_response_content(&chat)?)?;
            validate_review_decisions(parsed.decisions, expected_sources)
        }
        .await;

        match outcome {
            Ok(decisions) => return Ok(decisions),
            Err(error) => {
                last_error = Some(error);
                if attempt < 2 {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            }
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("术语总审校失败")))
}

fn validate_review_decisions(
    decisions: Vec<FinalReviewDecision>,
    expected_sources: &[String],
) -> Result<Vec<FinalReviewDecision>> {
    let expected: HashSet<&str> = expected_sources.iter().map(String::as_str).collect();
    let mut seen = HashSet::new();
    for decision in &decisions {
        if !expected.contains(decision.src.as_str()) {
            anyhow::bail!("总审校返回未知 src: {}", decision.src);
        }
        if !seen.insert(decision.src.as_str()) {
            anyhow::bail!("总审校重复返回 src: {}", decision.src);
        }
        if decision.reason.trim().is_empty() {
            anyhow::bail!("总审校缺少 reason: {}", decision.src);
        }
        if decision.keep {
            let dst = decision
                .dst
                .as_deref()
                .and_then(normalize_text)
                .ok_or_else(|| anyhow::anyhow!("总审校保留项缺少 dst: {}", decision.src))?;
            if contains_kana(&dst)
                || dst.chars().any(char::is_whitespace)
                || contains_traditional_hint(&dst)
                || has_repeated_title(&dst)
            {
                anyhow::bail!("总审校 dst 非法: {} -> {}", decision.src, dst);
            }
        }
    }
    if seen.len() != expected.len() {
        let missing = expected
            .into_iter()
            .filter(|source| !seen.contains(source))
            .collect::<Vec<_>>();
        anyhow::bail!("总审校遗漏 src: {}", missing.join(" / "));
    }
    Ok(decisions)
}

fn apply_final_review_decisions(
    entries: Vec<TranslationEntry>,
    mut decisions: HashMap<String, FinalReviewDecision>,
) -> (Vec<TranslationEntry>, Vec<GlossaryReviewChange>) {
    let mut reviewed = Vec::with_capacity(entries.len());
    let mut changes = Vec::new();
    for entry in entries {
        let Some(decision) = decisions.remove(&entry.src) else {
            reviewed.push(entry);
            continue;
        };

        if !decision.keep {
            changes.push(GlossaryReviewChange {
                source: entry.src,
                before_dst: entry.dst,
                after_dst: None,
                before_info: entry.info,
                after_info: None,
                reason: decision.reason,
            });
            continue;
        }

        let dst = decision
            .dst
            .as_deref()
            .and_then(normalize_text)
            .expect("validated review decision must have dst");
        let gender = normalize_gender(decision.gender);
        let full_name = decision.full_name.and_then(|name| normalize_text(&name));
        let info = build_info(&full_name, &gender, &canonical_key(&entry.src));
        if dst != entry.dst || info != entry.info {
            changes.push(GlossaryReviewChange {
                source: entry.src.clone(),
                before_dst: entry.dst.clone(),
                after_dst: Some(dst.clone()),
                before_info: entry.info.clone(),
                after_info: Some(info.clone()),
                reason: decision.reason,
            });
        }
        reviewed.push(TranslationEntry {
            src: entry.src,
            dst,
            info,
        });
    }
    (reviewed, changes)
}

fn parse_json_object<T: DeserializeOwned>(content: &str) -> Result<T> {
    let content = strip_leading_thinking_content(content);

    if let Ok(result) = serde_json::from_str::<T>(content) {
        return Ok(result);
    }

    let json_str = if content.contains("```json") {
        content
            .split("```json")
            .nth(1)
            .and_then(|s| s.split("```").next())
            .unwrap_or(content)
            .trim()
    } else if content.contains("```") {
        content.split("```").nth(1).unwrap_or(content).trim()
    } else {
        content
    };

    if let Some(start) = json_str.find('{') {
        if let Some(end) = json_str.rfind('}') {
            let json_slice = &json_str[start..=end];
            if let Ok(result) = serde_json::from_str::<T>(json_slice) {
                return Ok(result);
            }
        }
    }

    anyhow::bail!("Failed to parse LLM response as JSON: {}", content)
}

fn parse_json_from_llm(content: &str) -> Result<LlmResult> {
    parse_json_object(content)
}

fn normalize_chat_completions_endpoint(raw_url: &str) -> String {
    let trimmed = raw_url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return String::new();
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

fn strip_leading_thinking_content(content: &str) -> &str {
    let mut remaining = content.trim_start();

    loop {
        if let Some(rest) = remaining.strip_prefix("<think>") {
            if let Some(end) = rest.find("</think>") {
                remaining = rest[end + "</think>".len()..].trim_start();
                continue;
            }
        }

        if let Some(rest) = remaining.strip_prefix("<thinking>") {
            if let Some(end) = rest.find("</thinking>") {
                remaining = rest[end + "</thinking>".len()..].trim_start();
                continue;
            }
        }

        break;
    }

    remaining
}

fn build_name_clusters(
    characters: &HashMap<String, CharacterInfo>,
) -> (Vec<NameCluster>, Vec<GlossaryGenerationIssue>) {
    let mut grouped: HashMap<String, Vec<AliasInfo>> = HashMap::new();
    let mut issues = Vec::new();

    for (name, info) in characters {
        if is_structurally_invalid_candidate(name) {
            issues.push(GlossaryGenerationIssue {
                source: name.trim().to_string(),
                aliases: vec![name.clone()],
                kind: GlossaryIssueKind::Rejected,
                reason: "NER候选为空、纯符号或单个假名，已在LLM前过滤".to_string(),
            });
            continue;
        }
        if let Some(target) = attached_fragment_target(name, info.count, characters) {
            issues.push(GlossaryGenerationIssue {
                source: name.clone(),
                aliases: vec![name.clone()],
                kind: GlossaryIssueKind::Rejected,
                reason: format!("NER边界污染：前缀片段附着到已识别称呼 {target}"),
            });
            continue;
        }
        let key = canonical_key(name);
        let key = if key.is_empty() {
            name.trim().to_string()
        } else {
            key
        };

        let alias = AliasInfo {
            name: name.clone(),
            count: info.count,
            mentions: info.content.clone(),
        };

        grouped.entry(key).or_default().push(alias);
    }

    let mut clusters: Vec<NameCluster> = Vec::new();
    for (key, aliases) in grouped {
        let parts = split_aliases_by_gender_hints(&aliases);
        for mut part in parts {
            let primary = pick_primary_alias(&key, &part);
            part.sort_by(|a, b| b.count.cmp(&a.count).then(a.name.cmp(&b.name)));
            clusters.push(NameCluster {
                key: key.clone(),
                aliases: part,
                primary,
            });
        }
    }

    clusters.sort_by(|a, b| {
        let ac = a.aliases.iter().map(|x| x.count).max().unwrap_or(0);
        let bc = b.aliases.iter().map(|x| x.count).max().unwrap_or(0);
        bc.cmp(&ac)
            .then(a.key.cmp(&b.key))
            .then(a.primary.cmp(&b.primary))
    });
    (clusters, issues)
}

const ATTACHED_FRAGMENT_PREFIXES: &[&str] = &[
    "か",
    "が",
    "は",
    "を",
    "に",
    "と",
    "で",
    "へ",
    "の",
    "も",
    "や",
    "より",
    "ただ",
    "また",
    "でも",
    "もし",
    "まさか",
    "そして",
    "むしろ",
];

fn is_structurally_invalid_candidate(name: &str) -> bool {
    let normalized = remove_name_whitespace(name);
    if normalized.is_empty() {
        return true;
    }
    if normalized.chars().count() == 1 && contains_kana(&normalized) {
        return true;
    }
    normalized.chars().all(|ch| {
        ch.is_ascii_punctuation()
            || matches!(
                ch,
                '。' | '、'
                    | '！'
                    | '？'
                    | '「'
                    | '」'
                    | '『'
                    | '』'
                    | '（'
                    | '）'
                    | '【'
                    | '】'
                    | '・'
                    | '…'
            )
    })
}

fn attached_fragment_target<'a>(
    name: &str,
    count: usize,
    characters: &'a HashMap<String, CharacterInfo>,
) -> Option<&'a str> {
    let normalized = remove_name_whitespace(name);
    ATTACHED_FRAGMENT_PREFIXES.iter().find_map(|prefix| {
        let target = normalized.strip_prefix(prefix)?;
        if target.chars().count() < 2 {
            return None;
        }
        characters
            .get_key_value(target)
            .filter(|(_, info)| info.count >= count)
            .map(|(existing, _)| existing.as_str())
    })
}

fn build_final_review_payload(
    entries: &[TranslationEntry],
    characters: &HashMap<String, CharacterInfo>,
) -> Vec<FinalReviewItem> {
    entries
        .iter()
        .map(|entry| {
            let info = characters.get(&entry.src);
            let mut mentions = info
                .map(|character| character.content.iter().collect::<Vec<_>>())
                .unwrap_or_default();
            mentions.sort_by(|a, b| {
                mention_score(b)
                    .cmp(&mention_score(a))
                    .then(a.line.cmp(&b.line))
            });
            let contexts = mentions
                .into_iter()
                .take(FINAL_REVIEW_CONTEXT_ITEMS)
                .map(|mention| {
                    truncate_context_item(&context_item_text(mention), FINAL_REVIEW_CONTEXT_CHARS)
                })
                .collect();
            FinalReviewItem {
                src: entry.src.clone(),
                initial_dst: entry.dst.clone(),
                initial_info: entry.info.clone(),
                canonical_hint: canonical_key(&entry.src),
                count: info.map(|character| character.count).unwrap_or(0),
                contexts,
            }
        })
        .collect()
}

fn build_final_review_batches(entries: &[TranslationEntry]) -> Vec<Vec<TranslationEntry>> {
    if entries.is_empty() {
        return Vec::new();
    }

    // Connected components keep short/full names and provisional aliases in
    // the same LLM request. This is what lets the reviewer reconcile, e.g.,
    // 慧月 / 朱慧月 / 朱 慧月 instead of judging each cluster independently.
    let mut visited = vec![false; entries.len()];
    let mut components: Vec<Vec<usize>> = Vec::new();
    for start in 0..entries.len() {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut queue = vec![start];
        let mut component = Vec::new();
        while let Some(index) = queue.pop() {
            component.push(index);
            for candidate in 0..entries.len() {
                if !visited[candidate]
                    && review_entries_related(&entries[index], &entries[candidate])
                {
                    visited[candidate] = true;
                    queue.push(candidate);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components.sort_by_key(|component| component[0]);

    let mut batches = Vec::new();
    let mut current = Vec::new();
    for component in components {
        if component.len() > FINAL_REVIEW_BATCH_SIZE {
            if !current.is_empty() {
                batches.push(std::mem::take(&mut current));
            }
            for chunk in component.chunks(FINAL_REVIEW_BATCH_SIZE) {
                batches.push(chunk.iter().map(|&index| entries[index].clone()).collect());
            }
            continue;
        }
        if !current.is_empty() && current.len() + component.len() > FINAL_REVIEW_BATCH_SIZE {
            batches.push(std::mem::take(&mut current));
        }
        current.extend(component.into_iter().map(|index| entries[index].clone()));
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

fn review_entries_related(a: &TranslationEntry, b: &TranslationEntry) -> bool {
    let a_tokens = review_relation_tokens(a);
    let b_tokens = review_relation_tokens(b);
    a_tokens.iter().any(|left| {
        b_tokens.iter().any(|right| {
            left == right
                || (left.chars().count() >= 2
                    && right.chars().count() >= 2
                    && (left.contains(right) || right.contains(left)))
        })
    })
}

fn review_relation_tokens(entry: &TranslationEntry) -> Vec<String> {
    let mut tokens = vec![
        comparison_form(&canonical_key(&entry.src)),
        comparison_form(&entry.dst),
    ];
    if let Some((Some(full_name), _)) = parse_info(&entry.info) {
        tokens.push(comparison_form(&full_name));
    }
    tokens.retain(|token| token.chars().count() >= 2);
    tokens.sort();
    tokens.dedup();
    tokens
}

fn comparison_form(value: &str) -> String {
    value.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn pick_primary_alias(key: &str, aliases: &[AliasInfo]) -> String {
    let mut hinted: Vec<&AliasInfo> = aliases
        .iter()
        .filter(|a| {
            let (m, f) = gender_hint_from_alias_name(&a.name);
            (m >= 4 && f == 0) || (f >= 4 && m == 0)
        })
        .collect();
    if !hinted.is_empty() {
        hinted.sort_by_key(|alias| std::cmp::Reverse(hinted_primary_score(alias)));
        return hinted[0].name.clone();
    }

    aliases
        .iter()
        .max_by(|a, b| {
            alias_primary_score(a)
                .cmp(&alias_primary_score(b))
                .then(a.count.cmp(&b.count))
                .then_with(|| b.name.cmp(&a.name))
        })
        .map(|a| a.name.clone())
        .unwrap_or_else(|| key.to_string())
}

fn hinted_primary_score(a: &AliasInfo) -> i32 {
    let (m_hint, f_hint) = gender_hint_from_alias_name(&a.name);
    let hint = m_hint.max(f_hint);
    let max_mention = a.mentions.iter().map(mention_score).max().unwrap_or(0);
    hint * 100 + max_mention + (a.count.min(999) as i32)
}

fn alias_primary_score(a: &AliasInfo) -> i32 {
    let mut score = 0i32;
    score += (a.count.min(999) as i32) * 5;

    let (m_hint, f_hint) = gender_hint_from_alias_name(&a.name);
    score += (m_hint + f_hint) * 4;

    let max_mention = a.mentions.iter().map(mention_score).max().unwrap_or(0);
    score += max_mention;

    let mut m = 0i32;
    let mut f = 0i32;
    for mention in &a.mentions {
        let t = context_item_text(mention);
        let (mm, ff) = gender_evidence_score(&t);
        m += mm;
        f += ff;
    }
    score += (m.max(f)).min(50);

    score
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GenderHint {
    Male,
    Female,
}

fn split_aliases_by_gender_hints(aliases: &[AliasInfo]) -> Vec<Vec<AliasInfo>> {
    let mut male = Vec::new();
    let mut female = Vec::new();
    let mut neutral = Vec::new();

    for a in aliases {
        match infer_gender_hint_from_alias(a) {
            Some(GenderHint::Male) => male.push(a.clone()),
            Some(GenderHint::Female) => female.push(a.clone()),
            None => neutral.push(a.clone()),
        }
    }

    if !male.is_empty() && !female.is_empty() {
        let male_weight: usize = male.iter().map(|a| a.count).sum();
        let female_weight: usize = female.iter().map(|a| a.count).sum();
        if male_weight >= female_weight {
            male.extend(neutral);
            vec![male, female]
        } else {
            female.extend(neutral);
            vec![female, male]
        }
    } else {
        vec![aliases.to_vec()]
    }
}

fn infer_gender_hint_from_alias(a: &AliasInfo) -> Option<GenderHint> {
    let mut male = 0i32;
    let mut female = 0i32;

    let (m_hint, f_hint) = gender_hint_from_alias_name(&a.name);
    male += m_hint;
    female += f_hint;

    for m in &a.mentions {
        let t = context_item_text(m);
        let (m_score, f_score) = gender_evidence_score(&t);
        male += m_score;
        female += f_score;
    }

    if male >= 6 && male >= female + 3 {
        Some(GenderHint::Male)
    } else if female >= 6 && female >= male + 3 {
        Some(GenderHint::Female)
    } else {
        None
    }
}

fn gender_evidence_score(t: &str) -> (i32, i32) {
    let mut male = 0i32;
    let mut female = 0i32;

    let male_keys = [("男性", 6), ("男子", 4), ("彼氏", 3), ("男", 1)];
    let female_keys = [("女性", 6), ("女子", 4), ("彼女", 2), ("女", 1)];

    for (k, w) in male_keys {
        if t.contains(k) {
            male += w;
        }
    }
    for (k, w) in female_keys {
        if t.contains(k) {
            female += w;
        }
    }

    (male, female)
}

fn gender_evidence_score_from_context(context: &serde_json::Value, alias_name: &str) -> (i32, i32) {
    let mut male = 0i32;
    let mut female = 0i32;
    let (m_hint, f_hint) = gender_hint_from_alias_name(alias_name);
    male += m_hint;
    female += f_hint;
    if let Some(arr) = context.as_array() {
        for item in arr {
            if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                let (m, f) = gender_evidence_score(text);
                male += m;
                female += f;
            }
        }
    } else if let Some(text) = context.as_str() {
        let (m, f) = gender_evidence_score(text);
        male += m;
        female += f;
    }
    (male, female)
}

fn gender_hint_from_alias_name(name: &str) -> (i32, i32) {
    let s = name.replace('\u{3000}', "").trim().to_string();
    if s.is_empty() {
        return (0, 0);
    }

    let mut male = 0i32;
    let mut female = 0i32;

    if s.ends_with("君") || s.ends_with("くん") || s.ends_with("クン") {
        male += 4;
    }
    if s.contains("姉") || s.contains("お姉") || s.contains("姉さん") || s.contains("姉ちゃん")
    {
        female += 4;
    }
    if s.contains("兄") || s.contains("お兄") || s.contains("兄さん") || s.contains("兄ちゃん")
    {
        male += 2;
    }

    (male, female)
}

struct ClusterTranslationOutcome {
    entries: Vec<TranslationEntry>,
    issue: Option<GlossaryGenerationIssue>,
}

impl ClusterTranslationOutcome {
    fn rejected(cluster: &NameCluster, reason: impl Into<String>) -> Self {
        Self {
            entries: Vec::new(),
            issue: Some(cluster_issue(cluster, GlossaryIssueKind::Rejected, reason)),
        }
    }

    fn unresolved(cluster: &NameCluster, reason: impl Into<String>) -> Self {
        Self {
            entries: Vec::new(),
            issue: Some(cluster_issue(
                cluster,
                GlossaryIssueKind::Unresolved,
                reason,
            )),
        }
    }
}

fn cluster_issue(
    cluster: &NameCluster,
    kind: GlossaryIssueKind,
    reason: impl Into<String>,
) -> GlossaryGenerationIssue {
    GlossaryGenerationIssue {
        source: cluster.key.clone(),
        aliases: cluster
            .aliases
            .iter()
            .map(|alias| alias.name.clone())
            .collect(),
        kind,
        reason: reason.into(),
    }
}

async fn translate_cluster(
    client: &Client,
    api_url: &str,
    api_key: &str,
    model: &str,
    cluster: NameCluster,
) -> ClusterTranslationOutcome {
    if is_pure_title(&cluster.key) {
        return ClusterTranslationOutcome::rejected(&cluster, "纯称谓，不作为实体术语");
    }
    if is_family_like(&cluster.key) {
        return ClusterTranslationOutcome::rejected(&cluster, "家庭/群体称呼，不作为人物实体");
    }

    let context = build_context_for_cluster(&cluster);
    let inferred = match infer_base_name(
        client,
        api_url,
        api_key,
        model,
        &cluster.key,
        cluster.primary.as_str(),
        &context,
    )
    .await
    {
        Ok(Some(v)) => v,
        Ok(None) => {
            return ClusterTranslationOutcome::rejected(
                &cluster,
                "模型判定不是人物名或无法给出合法译名",
            )
        }
        Err(error) => {
            warn!("LLM翻译失败 {}: {}", cluster.key, error);
            return ClusterTranslationOutcome::unresolved(
                &cluster,
                format!("LLM/protocol failure: {error}"),
            );
        }
    };

    let mut out = Vec::new();
    let mut unresolved_aliases = Vec::new();
    for alias in &cluster.aliases {
        if is_pure_title(&alias.name) {
            continue;
        }
        if alias.count == 0 {
            continue;
        }
        let dst = build_alias_dst(&cluster.key, &inferred.base_dst, &alias.name);
        if contains_kana(&dst) || dst.contains(' ') {
            warn!("dst含假名或空格，跳过: {} -> {}", alias.name, dst);
            unresolved_aliases.push(alias.name.clone());
            continue;
        }

        let info = build_info(&inferred.full_name, &inferred.gender, &cluster.key);
        out.push(TranslationEntry {
            src: alias.name.clone(),
            dst,
            info,
        });
    }

    let issue = if unresolved_aliases.is_empty() {
        None
    } else {
        Some(cluster_issue(
            &cluster,
            GlossaryIssueKind::Unresolved,
            format!(
                "alias 目标仍含假名或空格: {}",
                unresolved_aliases.join(" / ")
            ),
        ))
    };
    ClusterTranslationOutcome {
        entries: out,
        issue,
    }
}

struct InferredBase {
    base_dst: String,
    gender: Option<String>,
    full_name: Option<String>,
}

async fn infer_base_name(
    client: &Client,
    api_url: &str,
    api_key: &str,
    model: &str,
    name: &str,
    hint_name: &str,
    context: &serde_json::Value,
) -> Result<Option<InferredBase>> {
    if is_pure_title(name) {
        return Ok(None);
    }

    let mut attempt = 0usize;

    loop {
        attempt += 1;
        let extra = match attempt {
            1 => "",
            2 => "\n补充约束：translated_chinese_name 禁止包含任何平假名/片假名/半角片假名；必须使用简体中文，不要繁体。",
            _ => "\n再次强调：严格输出JSON对象；translated_chinese_name 只能用简体中文/常用汉字，禁止任何假名与空格。",
        };

        let user_content = format!("文中的姓名：{}  上下文：{}{}", name, context, extra);
        let request = ChatRequest {
            model: model.to_string(),
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: SYSTEM_PROMPT.to_string(),
                },
                Message {
                    role: "user".to_string(),
                    content: user_content,
                },
            ],
            temperature: 0.0,
            thinking: thinking_for_provider(api_url, model),
        };

        let resp = client
            .post(api_url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("LLM API returned {}: {}", status, body);
        }

        let chat_resp: ChatResponse = resp.json().await?;
        let content = first_response_content(&chat_resp)?;
        let data = match parse_json_from_llm(content) {
            Ok(v) => v,
            Err(e) if attempt < 3 => {
                warn!("LLM响应解析失败（第{}次），将重试: {}", attempt, e);
                continue;
            }
            Err(e) => return Err(e),
        };

        if !data.is_name {
            return Ok(None);
        }

        let mut gender = normalize_gender(data.gender);
        let full_name = data.full_name.and_then(|s| normalize_text(&s));
        let translated = data
            .translated_chinese_name
            .and_then(|s| normalize_text(&s))
            .unwrap_or_else(|| name.to_string());

        let (m_hint, f_hint) = gender_hint_from_alias_name(hint_name);
        if gender.is_none() {
            if m_hint >= 4 && f_hint == 0 {
                gender = Some("男性".to_string());
            } else if f_hint >= 4 && m_hint == 0 {
                gender = Some("女性".to_string());
            }
        }

        if let Some(g) = &gender {
            let (m_score, f_score) = gender_evidence_score_from_context(context, hint_name);
            let min_score = if (m_hint >= 4 && f_hint == 0) || (f_hint >= 4 && m_hint == 0) {
                4
            } else {
                6
            };
            let min_gap = if min_score == 4 { 2 } else { 3 };
            let ok = match g.as_str() {
                "男性" => m_score >= min_score && m_score >= f_score + min_gap,
                "女性" => f_score >= min_score && f_score >= m_score + min_gap,
                _ => true,
            };
            if !ok {
                gender = None;
            }
        }

        if translated.contains(' ') || contains_kana(&translated) {
            if attempt < 3 {
                continue;
            }
            return Ok(None);
        }

        if contains_traditional_hint(&translated) && attempt < 3 {
            continue;
        }

        return Ok(Some(InferredBase {
            base_dst: translated,
            gender,
            full_name,
        }));
    }
}

fn normalize_text(s: &str) -> Option<String> {
    let t = s.replace('\u{3000}', "").trim().to_string();
    if t.is_empty() || t.eq_ignore_ascii_case("null") {
        None
    } else {
        Some(t)
    }
}

fn normalize_gender(g: Option<String>) -> Option<String> {
    let g = g.and_then(|s| normalize_text(&s))?;
    match g.as_str() {
        "男性" | "男" => Some("男性".to_string()),
        "女性" | "女" => Some("女性".to_string()),
        "动物" | "動物" => Some("动物".to_string()),
        _ => None,
    }
}

fn contains_traditional_hint(s: &str) -> bool {
    const HINTS: &[char] = &[
        '為', '國', '學', '體', '發', '會', '對', '這', '說', '嗎', '麼', '後', '於', '與', '過',
        '還', '點', '當', '場', '歲', '裡', '與', '總', '劃', '顏', '髮', '聲', '覺', '親', '願',
    ];
    s.chars().any(|c| HINTS.contains(&c))
}

pub fn canonical_key(name: &str) -> String {
    strip_affixes(name)
}

fn remove_name_whitespace(name: &str) -> String {
    name.chars().filter(|ch| !ch.is_whitespace()).collect()
}

pub fn strip_affixes(name: &str) -> String {
    let mut s = remove_name_whitespace(name);
    for p in HONORIFIC_PREFIXES {
        if s.starts_with(p) && s.chars().count() > p.chars().count() + 1 {
            s = s.strip_prefix(p).unwrap_or(&s).to_string();
            break;
        }
    }

    loop {
        let mut changed = false;
        for suf in HONORIFIC_SUFFIXES {
            if s.ends_with(suf) && s.chars().count() > suf.chars().count() {
                s = s.strip_suffix(suf).unwrap_or(&s).to_string();
                changed = true;
                break;
            }
        }
        if !changed {
            break;
        }
    }

    s.trim().to_string()
}

pub fn is_pure_title(name: &str) -> bool {
    let raw = name.replace('\u{3000}', "").trim().to_string();
    if raw.is_empty() {
        return true;
    }
    if PURE_TITLE_CORES.contains(&raw.as_str()) {
        return true;
    }
    let core = strip_affixes(&raw);
    if core.is_empty() {
        return true;
    }
    PURE_TITLE_CORES.contains(&core.as_str())
}

pub fn contains_kana(s: &str) -> bool {
    s.chars().any(|c| {
        let u = c as u32;
        (0x3040..=0x309F).contains(&u)
            || (0x30A0..=0x30FF).contains(&u)
            || (0xFF66..=0xFF9D).contains(&u)
    })
}

fn build_context_for_cluster(cluster: &NameCluster) -> serde_json::Value {
    let primary_name = cluster.primary.as_str();
    let primary = cluster
        .aliases
        .iter()
        .find(|a| a.name == primary_name)
        .unwrap_or_else(|| {
            cluster
                .aliases
                .iter()
                .max_by(|a, b| a.count.cmp(&b.count).then(a.name.cmp(&b.name)))
                .expect("cluster.aliases is empty")
        });

    let mut scored: Vec<(i32, &Mention)> = primary
        .mentions
        .iter()
        .map(|m| (mention_score(m), m))
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.line.cmp(&b.1.line)));

    let mut out = Vec::new();
    for (_score, m) in scored.into_iter().take(MAX_CONTEXT_ITEMS) {
        let item = serde_json::json!({
            "line": m.line,
            "text": truncate_context_item(&context_item_text(m), MAX_CONTEXT_CHARS_PER_ITEM),
        });
        out.push(item);
    }

    serde_json::Value::Array(out)
}

fn context_item_text(m: &Mention) -> String {
    let mut parts = Vec::new();
    let above: Vec<String> = m
        .above
        .iter()
        .filter(|s| !s.trim().is_empty())
        .rev()
        .take(2)
        .cloned()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let follow: Vec<String> = m
        .follow
        .iter()
        .filter(|s| !s.trim().is_empty())
        .take(2)
        .cloned()
        .collect();
    parts.extend(above);
    parts.push(m.line_text.trim().to_string());
    parts.extend(follow);
    parts.join("\n")
}

fn is_family_like(name: &str) -> bool {
    let raw = name.replace('\u{3000}', "").trim().to_string();
    raw.chars().count() >= 2 && raw.ends_with('家')
}

fn truncate_context_item(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    s.chars().take(max_chars).collect()
}

fn mention_score(m: &Mention) -> i32 {
    let t = context_item_text(m);
    let mut score = 0i32;

    let strong = [
        ("と呼ばれ", 40),
        ("呼ばれた", 25),
        ("本名", 25),
        ("フルネーム", 25),
        ("──", 10),
    ];
    for (k, w) in strong {
        if t.contains(k) {
            score += w;
        }
    }

    let gender_keys = [
        ("女性", 25),
        ("男子", 18),
        ("女子", 18),
        ("彼女", 12),
        ("彼", 10),
        ("男", 6),
        ("女", 6),
    ];
    for (k, w) in gender_keys {
        if t.contains(k) {
            score += w;
        }
    }

    if t.contains("「") || t.contains("『") {
        score += 4;
    }
    if t.contains("こと") {
        score += 2;
    }

    let line_bonus = (1_000i32 - (m.line as i32).min(1_000)) / 100;
    score + line_bonus
}

fn build_alias_dst(cluster_key: &str, base_dst: &str, alias: &str) -> String {
    let alias = remove_name_whitespace(alias);
    if alias == cluster_key {
        return base_dst.to_string();
    }

    if let Some(rest) = alias.strip_prefix(cluster_key) {
        let suffix = rest.replace('\u{3000}', "").trim().to_string();
        if suffix.is_empty() {
            return base_dst.to_string();
        }
        if let Some(mapped) = map_suffix(&suffix) {
            if base_dst.ends_with(mapped) {
                return base_dst.to_string();
            }
            return format!("{}{}", base_dst, mapped);
        }
    }

    base_dst.to_string()
}

fn has_repeated_title(value: &str) -> bool {
    ["老师", "部长", "会长", "委员长", "店长", "课长", "社长"]
        .iter()
        .any(|title| value.contains(&format!("{title}{title}")))
}

fn map_suffix(s: &str) -> Option<&'static str> {
    match s {
        "くん" | "クン" | "君" => Some("君"),
        "先生" => Some("老师"),
        "部長" => Some("部长"),
        "会長" => Some("会长"),
        "委員長" => Some("委员长"),
        "店長" => Some("店长"),
        "課長" => Some("课长"),
        "社長" => Some("社长"),
        _ => None,
    }
}

fn build_info(full_name: &Option<String>, gender: &Option<String>, key: &str) -> String {
    match (full_name, gender) {
        (Some(f), Some(g)) if f != key => format!("{},{}", f, g),
        (_, Some(g)) => g.clone(),
        (Some(f), None) if f != key => f.clone(),
        _ => String::new(),
    }
}

fn propagate_gender_within_canonical(entries: &mut [TranslationEntry]) {
    let mut genders_by_key: HashMap<String, String> = HashMap::new();
    for e in entries.iter() {
        let key = canonical_key(&e.src);
        if key.trim().is_empty() {
            continue;
        }
        if let Some((_full, Some(g))) = parse_info(&e.info) {
            genders_by_key.entry(key).or_insert(g);
        }
    }

    for e in entries.iter_mut() {
        let key = canonical_key(&e.src);
        if let Some(g) = genders_by_key.get(&key).cloned() {
            let (full, gender) = parse_info(&e.info).unwrap_or((None, None));
            if gender.is_none() {
                e.info = format_info(full.as_deref(), Some(g.as_str()));
            }
        }
    }
}

fn parse_info(info: &str) -> Option<(Option<String>, Option<String>)> {
    let s = info.trim();
    if s.is_empty() {
        return Some((None, None));
    }
    if let Some((a, b)) = s.split_once(',') {
        let a = a.trim();
        let b = b.trim();
        if b == "男性" || b == "女性" || b == "动物" {
            let full = if a.is_empty() {
                None
            } else {
                Some(a.to_string())
            };
            return Some((full, Some(b.to_string())));
        }
    }
    if s == "男性" || s == "女性" || s == "动物" {
        return Some((None, Some(s.to_string())));
    }
    Some((Some(s.to_string()), None))
}

fn format_info(full_name: Option<&str>, gender: Option<&str>) -> String {
    match (full_name, gender) {
        (Some(f), Some(g)) if !f.trim().is_empty() => format!("{},{}", f.trim(), g),
        (None, Some(g)) => g.to_string(),
        (Some(f), None) => f.trim().to_string(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_alias_dst_keeps_only_low_risk_suffixes() {
        assert_eq!(build_alias_dst("志喜屋", "志喜屋", "志喜屋先輩"), "志喜屋");
        assert_eq!(build_alias_dst("佳樹", "佳树", "佳樹さん"), "佳树");
        assert_eq!(build_alias_dst("佳樹", "佳树", "佳樹ちゃん"), "佳树");
        assert_eq!(build_alias_dst("姫乃", "姬乃", "姫乃様"), "姬乃");
        assert_eq!(build_alias_dst("温水", "温水", "温水君"), "温水君");
        assert_eq!(build_alias_dst("星野", "星野", "星野先生"), "星野老师");
        assert_eq!(build_alias_dst("森", "森老师", "森先生"), "森老师");
        assert_eq!(build_alias_dst("美波", "美波", "美波監督"), "美波");
        assert_eq!(build_alias_dst("ゆづ", "柚", "ゆづ姉"), "柚");
    }

    #[test]
    fn canonical_key_removes_internal_name_whitespace() {
        assert_eq!(canonical_key("黄 玲琳様"), "黄玲琳");
        assert_eq!(canonical_key("朱\u{3000}慧月"), "朱慧月");
    }

    #[test]
    fn cluster_prepass_rejects_attached_sentence_fragments() {
        fn character(name: &str, count: usize) -> CharacterInfo {
            CharacterInfo {
                name: name.to_string(),
                count,
                content: Vec::new(),
            }
        }

        let characters = HashMap::from([
            ("めぐるちゃん".to_string(), character("めぐるちゃん", 30)),
            ("かめぐるちゃん".to_string(), character("かめぐるちゃん", 2)),
            (
                "むしろめぐるちゃん".to_string(),
                character("むしろめぐるちゃん", 1),
            ),
        ]);
        let (clusters, issues) = build_name_clusters(&characters);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].aliases[0].name, "めぐるちゃん");
        assert_eq!(issues.len(), 2);
        assert!(issues.iter().all(|issue| issue.reason.contains("边界污染")));
    }

    #[test]
    fn cluster_prepass_rejects_single_kana_and_blank_candidates() {
        fn character(name: &str) -> CharacterInfo {
            CharacterInfo {
                name: name.to_string(),
                count: 2,
                content: Vec::new(),
            }
        }
        let characters = HashMap::from([
            ("ん".to_string(), character("ん")),
            ("ゃ".to_string(), character("ゃ")),
            (" ".to_string(), character(" ")),
            ("月".to_string(), character("月")),
        ]);
        let (clusters, issues) = build_name_clusters(&characters);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].key, "月");
        assert_eq!(issues.len(), 3);
    }

    #[test]
    fn final_review_batches_keep_related_names_together() {
        let entries = vec![
            TranslationEntry {
                src: "慧月".into(),
                dst: "朱慧月".into(),
                info: "朱慧月,女性".into(),
            },
            TranslationEntry {
                src: "无关人物".into(),
                dst: "无关人物".into(),
                info: String::new(),
            },
            TranslationEntry {
                src: "朱 慧月様".into(),
                dst: "朱慧月".into(),
                info: "女性".into(),
            },
            TranslationEntry {
                src: "月".into(),
                dst: "朱慧月".into(),
                info: "朱慧月".into(),
            },
        ];
        let batches = build_final_review_batches(&entries);
        let related_batch = batches
            .iter()
            .find(|batch| batch.iter().any(|entry| entry.src == "慧月"))
            .unwrap();
        assert!(related_batch.iter().any(|entry| entry.src == "朱 慧月様"));
        assert!(related_batch.iter().any(|entry| entry.src == "月"));
    }

    #[test]
    fn final_review_protocol_rejects_duplicate_titles_and_missing_sources() {
        let expected = vec!["森先生".to_string()];
        let duplicate_title = FinalReviewDecision {
            src: "森先生".into(),
            keep: true,
            dst: Some("森老师老师".into()),
            gender: None,
            full_name: None,
            reason: "人物".into(),
        };
        assert!(validate_review_decisions(vec![duplicate_title], &expected).is_err());
        assert!(validate_review_decisions(Vec::new(), &expected).is_err());
    }

    #[test]
    fn final_review_applies_rejection_and_cross_cluster_consistency() {
        let entries = vec![
            TranslationEntry {
                src: "慧月".into(),
                dst: "慧月".into(),
                info: String::new(),
            },
            TranslationEntry {
                src: "朱 慧月".into(),
                dst: "朱慧月".into(),
                info: "女性".into(),
            },
            TranslationEntry {
                src: "黄麒宮".into(),
                dst: "黄麒宫".into(),
                info: String::new(),
            },
        ];
        let decisions = HashMap::from([
            (
                "慧月".into(),
                FinalReviewDecision {
                    src: "慧月".into(),
                    keep: true,
                    dst: Some("朱慧月".into()),
                    gender: Some("女性".into()),
                    full_name: Some("朱慧月".into()),
                    reason: "与全名一致".into(),
                },
            ),
            (
                "朱 慧月".into(),
                FinalReviewDecision {
                    src: "朱 慧月".into(),
                    keep: true,
                    dst: Some("朱慧月".into()),
                    gender: Some("女性".into()),
                    full_name: Some("朱慧月".into()),
                    reason: "全名".into(),
                },
            ),
            (
                "黄麒宮".into(),
                FinalReviewDecision {
                    src: "黄麒宮".into(),
                    keep: false,
                    dst: None,
                    gender: None,
                    full_name: None,
                    reason: "宫殿名".into(),
                },
            ),
        ]);
        let (reviewed, changes) = apply_final_review_decisions(entries, decisions);
        assert_eq!(reviewed.len(), 2);
        assert!(reviewed.iter().all(|entry| entry.dst == "朱慧月"));
        assert!(reviewed.iter().all(|entry| entry.info.contains("女性")));
        assert_eq!(changes.len(), 2);
        assert!(changes
            .iter()
            .any(|change| change.source == "黄麒宮" && change.after_dst.is_none()));
    }

    #[test]
    fn empty_choices_is_a_protocol_error_not_a_panic() {
        let response = ChatResponse { choices: vec![] };
        let error = first_response_content(&response).unwrap_err();
        assert!(error.to_string().contains("choices 为空"));
    }

    #[test]
    fn glossary_report_counts_visible_outcomes() {
        let report = GlossaryTranslationReport {
            entries: vec![],
            issues: vec![
                GlossaryGenerationIssue {
                    source: "候補A".to_string(),
                    aliases: vec!["候補A".to_string()],
                    kind: GlossaryIssueKind::Unresolved,
                    reason: "protocol".to_string(),
                },
                GlossaryGenerationIssue {
                    source: "先生".to_string(),
                    aliases: vec!["先生".to_string()],
                    kind: GlossaryIssueKind::Rejected,
                    reason: "title".to_string(),
                },
            ],
            ..GlossaryTranslationReport::default()
        };
        assert_eq!(report.unresolved_count(), 1);
        assert_eq!(report.rejected_count(), 1);
    }
}
