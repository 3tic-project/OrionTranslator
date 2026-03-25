use crate::detector::{CharacterInfo, Mention};
use crate::{emit, GlossaryProgressCallback, GlossaryProgressEvent};
use anyhow::Result;
use log::warn;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
6) translated_chinese_name 必须针对"输入的称呼/人名本身"给出翻译（不要把 full_name 直接当作翻译名）。"#;

const MAX_CONTEXT_ITEMS: usize = 10;
const MAX_CONTEXT_CHARS_PER_ITEM: usize = 220;

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

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
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
            api_key: api_key.to_string(),
            model: model.to_string(),
        }
    }

    /// Translate all character names concurrently
    pub async fn translate_all(
        &self,
        characters: &HashMap<String, CharacterInfo>,
        max_concurrent: usize,
        progress: GlossaryProgressCallback,
    ) -> Vec<TranslationEntry> {
        let clusters = build_name_clusters(characters);
        let total = clusters.len();
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent));
        let client = self.client.clone();
        let api_url = self.api_url.clone();
        let api_key = self.api_key.clone();
        let model = self.model.clone();

        let completed_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let mut handles = Vec::new();
        for cluster in clusters {
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
            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(mut entries) => results.append(&mut entries),
                Err(e) => warn!("LLM任务失败: {}", e),
            }
        }

        propagate_gender_within_canonical(&mut results);
        results
    }
}

fn parse_json_from_llm(content: &str) -> Result<LlmResult> {
    let content = strip_leading_thinking_content(content);

    if let Ok(result) = serde_json::from_str::<LlmResult>(content) {
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
            if let Ok(result) = serde_json::from_str::<LlmResult>(json_slice) {
                return Ok(result);
            }
        }
    }

    anyhow::bail!("Failed to parse LLM response as JSON: {}", content)
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

fn build_name_clusters(characters: &HashMap<String, CharacterInfo>) -> Vec<NameCluster> {
    let mut grouped: HashMap<String, Vec<AliasInfo>> = HashMap::new();

    for (name, info) in characters {
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
    clusters
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
        hinted.sort_by(|a, b| hinted_primary_score(b).cmp(&hinted_primary_score(a)));
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
    let max_mention = a
        .mentions
        .iter()
        .map(|m| mention_score(m))
        .max()
        .unwrap_or(0);
    hint * 100 + max_mention + (a.count.min(999) as i32)
}

fn alias_primary_score(a: &AliasInfo) -> i32 {
    let mut score = 0i32;
    score += (a.count.min(999) as i32) * 5;

    let (m_hint, f_hint) = gender_hint_from_alias_name(&a.name);
    score += (m_hint + f_hint) * 4;

    let max_mention = a
        .mentions
        .iter()
        .map(|m| mention_score(m))
        .max()
        .unwrap_or(0);
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

async fn translate_cluster(
    client: &Client,
    api_url: &str,
    api_key: &str,
    model: &str,
    cluster: NameCluster,
) -> Vec<TranslationEntry> {
    if is_pure_title(&cluster.key) {
        return vec![];
    }
    if is_family_like(&cluster.key) {
        return vec![];
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
        Ok(None) => return vec![],
        Err(e) => {
            warn!("LLM翻译失败 {}: {}", cluster.key, e);
            return vec![];
        }
    };

    let mut out = Vec::new();
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
            continue;
        }

        let info = build_info(&inferred.full_name, &inferred.gender, &cluster.key);
        out.push(TranslationEntry {
            src: alias.name.clone(),
            dst,
            info,
        });
    }

    out
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
        let content = &chat_resp.choices[0].message.content;
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

        if contains_traditional_hint(&translated) {
            if attempt < 3 {
                continue;
            }
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

pub fn strip_affixes(name: &str) -> String {
    let mut s = name.replace('\u{3000}', "").trim().to_string();
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
    if alias == cluster_key {
        return base_dst.to_string();
    }

    if let Some(rest) = alias.strip_prefix(cluster_key) {
        let suffix = rest.replace('\u{3000}', "").trim().to_string();
        if suffix.is_empty() {
            return base_dst.to_string();
        }
        let mapped = map_suffix(&suffix);
        if mapped.is_empty() {
            return base_dst.to_string();
        }
        return format!("{}{}", base_dst, mapped);
    }

    base_dst.to_string()
}

fn map_suffix(s: &str) -> String {
    match s {
        "さん" => "".to_string(),
        "ちゃん" | "チャン" => "酱".to_string(),
        "くん" | "クン" => "君".to_string(),
        "君" => "君".to_string(),
        "先輩" => "前辈".to_string(),
        "先生" => "老师".to_string(),
        "部長" => "部长".to_string(),
        "会長" => "会长".to_string(),
        "委員長" => "委员长".to_string(),
        "店長" => "店长".to_string(),
        "課長" => "课长".to_string(),
        "社長" => "社长".to_string(),
        "監督" => "监督".to_string(),
        "様" | "さま" => "大人".to_string(),
        "姉" | "姉さん" | "姉ちゃん" => "姐".to_string(),
        "兄" | "兄さん" | "兄ちゃん" => "哥".to_string(),
        "妹" => "妹".to_string(),
        "弟" => "弟".to_string(),
        _ => "".to_string(),
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
        if let Some((_full, gender)) = parse_info(&e.info) {
            if let Some(g) = gender {
                genders_by_key.entry(key).or_insert(g);
            }
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
