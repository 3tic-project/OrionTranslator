use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use fancy_regex::Regex;
use serde_json::Value;

use super::trie::{KeywordMatch, MatchKind, TrieMatcher};
use super::types::{DetectionResult, LineType};

// ── Constants ──────────────────────────────────────────────────────────────

const SENTENCE_INITIAL_CHARS: usize = 6;
const SENTENCE_INITIAL_BONUS: f64 = 1.4;
const SHORT_DIALOGUE_CHARS: usize = 12;
const PER_CATEGORY_SCORE_CAP: f64 = 80.0;

// ── Internal category ──────────────────────────────────────────────────────

#[derive(Debug)]
struct Category {
    id: String,
    label: String,
    priority: i32,
    actions: HashMap<String, Value>,
    keywords: Vec<(String, f64)>,
    regex_patterns: Vec<(Regex, f64)>,
}

// ── Detector ───────────────────────────────────────────────────────────────

pub struct ContextDetector {
    scene_break_re: Vec<Regex>,
    chapter_heading_re: Vec<Regex>,
    metadata_re: Vec<Regex>,
    categories: Vec<Category>,
    categories_by_id: HashMap<String, usize>, // index into categories
    trie: TrieMatcher,
}

impl ContextDetector {
    pub fn from_file(rules_path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(rules_path)
            .with_context(|| format!("Failed to read rules file: {}", rules_path.display()))?;
        let config: Value =
            serde_json::from_str(&content).with_context(|| "Failed to parse rules JSON")?;
        Self::from_config(&config)
    }

    pub fn from_config(config: &Value) -> Result<Self> {
        let boundaries = config
            .get("boundaries")
            .ok_or_else(|| anyhow::anyhow!("Missing 'boundaries' in config"))?;

        let scene_break_re = Self::compile_regex_list(boundaries, "scene_break_regex")?;
        let chapter_heading_re = Self::compile_regex_list(boundaries, "chapter_heading_regex")?;
        let metadata_re = Self::compile_regex_list(boundaries, "metadata_regex")?;

        let mut categories = Vec::new();
        let mut categories_by_id = HashMap::new();
        let mut trie = TrieMatcher::new();

        if let Some(cats) = config.get("categories").and_then(|v| v.as_array()) {
            for cat_val in cats {
                let category = Self::parse_category(cat_val)?;
                let idx = categories.len();
                categories_by_id.insert(category.id.clone(), idx);

                // Add keywords to trie
                for (kw, weight) in &category.keywords {
                    if !kw.is_empty() {
                        trie.add(kw, &category.id, *weight);
                    }
                }

                categories.push(category);
            }
        }

        Ok(Self {
            scene_break_re,
            chapter_heading_re,
            metadata_re,
            categories,
            categories_by_id,
            trie,
        })
    }

    // ── Public API ──────────────────────────────────────────────────────

    pub fn detect_line(&self, line: &str, _history: &[String]) -> DetectionResult {
        let line_type = self.classify_line(line);

        // Blank / metadata → no context needed
        if matches!(line_type, LineType::Blank | LineType::Metadata) {
            return DetectionResult {
                line_type,
                dominant_category: None,
                matches: vec![],
                actions: HashMap::from([("reset_context".to_string(), Value::Bool(false))]),
            };
        }

        // Scene break / chapter heading → reset
        if matches!(line_type, LineType::SceneBreak | LineType::ChapterHeading) {
            return DetectionResult {
                line_type,
                dominant_category: None,
                matches: vec![],
                actions: HashMap::from([("reset_context".to_string(), Value::Bool(true))]),
            };
        }

        // ── Collect matches ─────────────────────────────────────────────
        let raw_matches = self.trie.find(line);
        let regex_matches = self.find_regex_matches(line);
        let mut all_matches =
            self.dedupe_matches(raw_matches.into_iter().chain(regex_matches).collect());

        // Filter: dialogue_reaction only for dialogue lines
        if line_type != LineType::Dialogue {
            all_matches.retain(|m| m.category_id != "dialogue_reaction");
        }

        // ── Position-aware weight adjustment ────────────────────────────
        // 句首奖励按「字符」计，不能用 UTF-8 字节偏移与 SENTENCE_INITIAL_CHARS 直接比较
        let stripped = line.trim_start_matches(|c: char| "「『【《（(\u{3000} ".contains(c));
        let prefix_byte_len = line.len() - stripped.len();
        let initial_char_offset = byte_offset_to_char_index(line, prefix_byte_len);
        all_matches = self.apply_position_bonus(line, all_matches, initial_char_offset);

        // ── Short dialogue auto-detection ───────────────────────────────
        if line_type == LineType::Dialogue {
            if let Some(content) = Self::extract_dialogue_content(line) {
                if content.chars().count() <= SHORT_DIALOGUE_CHARS {
                    let has_reaction = all_matches
                        .iter()
                        .any(|m| m.category_id == "dialogue_reaction");
                    if !has_reaction {
                        all_matches.push(KeywordMatch {
                            category_id: "dialogue_reaction".to_string(),
                            text: "(short_dialogue)".to_string(),
                            weight: 15.0,
                            start: 0,
                            end: 0,
                            kind: MatchKind::Heuristic,
                        });
                    }
                }
            }
        }

        // ── Aggregate ──────────────────────────────────────────────────
        let mut matched_categories: HashMap<String, f64> = HashMap::new();
        let mut matched_priorities: HashMap<String, i32> = HashMap::new();
        let mut actions: HashMap<String, Value> = HashMap::new();
        actions.insert("reset_context".to_string(), Value::Bool(false));
        actions.insert("needs_glossary".to_string(), Value::Bool(false));

        for m in &all_matches {
            *matched_categories
                .entry(m.category_id.clone())
                .or_insert(0.0) += m.weight;

            if let Some(&idx) = self.categories_by_id.get(&m.category_id) {
                let cat = &self.categories[idx];
                matched_priorities
                    .entry(m.category_id.clone())
                    .or_insert(cat.priority);

                if cat
                    .actions
                    .get("needs_glossary")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    actions.insert("needs_glossary".to_string(), Value::Bool(true));
                }
                if cat
                    .actions
                    .get("reset_context")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    actions.insert("reset_context".to_string(), Value::Bool(true));
                }
            }
        }

        // Apply per-category score cap
        for score in matched_categories.values_mut() {
            *score = score.min(PER_CATEGORY_SCORE_CAP);
        }

        // ── No matches → minimal context ────────────────────────────────
        if matched_categories.is_empty() {
            return DetectionResult {
                line_type,
                dominant_category: None,
                matches: vec![],
                actions,
            };
        }

        // ── Reset context action ────────────────────────────────────────
        if actions
            .get("reset_context")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            let dominant_id = Self::pick_dominant(&matched_categories, &matched_priorities);
            let dominant_label = self
                .categories_by_id
                .get(&dominant_id)
                .map(|&idx| self.categories[idx].label.clone());

            all_matches.sort_by(|a, b| {
                a.category_id
                    .cmp(&b.category_id)
                    .then(a.start.cmp(&b.start))
                    .then(a.end.cmp(&b.end))
            });

            return DetectionResult {
                line_type,
                dominant_category: dominant_label,
                matches: all_matches,
                actions,
            };
        }

        // ── Determine dominant category ─────────────────────────────────
        let dominant_id = Self::pick_dominant(&matched_categories, &matched_priorities);
        let dominant_label = self
            .categories_by_id
            .get(&dominant_id)
            .map(|&idx| self.categories[idx].label.clone());

        all_matches.sort_by(|a, b| {
            a.category_id
                .cmp(&b.category_id)
                .then(a.start.cmp(&b.start))
                .then(a.end.cmp(&b.end))
        });

        DetectionResult {
            line_type,
            dominant_category: dominant_label,
            matches: all_matches,
            actions,
        }
    }

    // ── Line classification ─────────────────────────────────────────────

    fn classify_line(&self, line: &str) -> LineType {
        let s = line.trim();
        if s.is_empty() {
            return LineType::Blank;
        }

        for p in &self.metadata_re {
            if p.is_match(s).unwrap_or(false) {
                return LineType::Metadata;
            }
        }

        for p in &self.scene_break_re {
            if p.is_match(s).unwrap_or(false) {
                return LineType::SceneBreak;
            }
        }

        for p in &self.chapter_heading_re {
            if p.is_match(s).unwrap_or(false) {
                return LineType::ChapterHeading;
            }
        }

        if s.starts_with('「') || s.starts_with('『') {
            return LineType::Dialogue;
        }

        LineType::Narration
    }

    // ── Category parsing ────────────────────────────────────────────────

    fn parse_category(cat: &Value) -> Result<Category> {
        let id = cat
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Category missing 'id'"))?
            .to_string();

        let label = cat
            .get("label")
            .and_then(|v| v.as_str())
            .unwrap_or(&id)
            .to_string();

        let priority = cat.get("priority").and_then(|v| v.as_i64()).unwrap_or(3) as i32;

        let base_weight = cat
            .get("base_weight")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let actions: HashMap<String, Value> = cat
            .get("actions")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        // Parse keywords
        let mut keywords = Vec::new();
        if let Some(match_def) = cat.get("match") {
            if let Some(kw_arr) = match_def.get("keywords").and_then(|v| v.as_array()) {
                for item in kw_arr {
                    if let Some(s) = item.as_str() {
                        keywords.push((s.to_string(), base_weight));
                    } else if let Some(obj) = item.as_object() {
                        let text = obj
                            .get("text")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let weight = obj
                            .get("weight")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(base_weight);
                        keywords.push((text, weight));
                    }
                }
            }
        }

        // Parse regex patterns
        let mut regex_patterns = Vec::new();
        if let Some(match_def) = cat.get("match") {
            if let Some(regex_arr) = match_def.get("regex").and_then(|v| v.as_array()) {
                for item in regex_arr {
                    let (pattern_str, weight) = if let Some(s) = item.as_str() {
                        (s.to_string(), base_weight)
                    } else if let Some(obj) = item.as_object() {
                        let p = obj
                            .get("pattern")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let w = obj
                            .get("weight")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(base_weight);
                        (p, w)
                    } else {
                        continue;
                    };

                    match Regex::new(&pattern_str) {
                        Ok(re) => regex_patterns.push((re, weight)),
                        Err(e) => {
                            tracing::warn!(
                                "Failed to compile regex '{}' for category '{}': {}",
                                pattern_str,
                                id,
                                e
                            );
                        }
                    }
                }
            }
        }

        Ok(Category {
            id,
            label,
            priority,
            actions,
            keywords,
            regex_patterns,
        })
    }

    // ── Matching helpers ────────────────────────────────────────────────

    fn find_regex_matches(&self, line: &str) -> Vec<KeywordMatch> {
        let mut matches = Vec::new();
        for cat in &self.categories {
            for (pattern, weight) in &cat.regex_patterns {
                for m in pattern.find_iter(line).flatten() {
                    matches.push(KeywordMatch {
                        category_id: cat.id.clone(),
                        text: m.as_str().to_string(),
                        weight: *weight,
                        start: m.start(),
                        end: m.end(),
                        kind: MatchKind::Regex,
                    });
                }
            }
        }
        matches
    }

    fn dedupe_matches(&self, matches: Vec<KeywordMatch>) -> Vec<KeywordMatch> {
        // Sort: prefer longer matches, then higher weight
        let mut sorted = matches;
        sorted.sort_by(|a, b| {
            a.category_id
                .cmp(&b.category_id)
                .then_with(|| {
                    let len_a = a.end - a.start;
                    let len_b = b.end - b.start;
                    len_b.cmp(&len_a) // longer first
                })
                .then_with(|| {
                    b.weight
                        .partial_cmp(&a.weight)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.start.cmp(&b.start))
        });

        let mut seen_texts: HashSet<(String, String)> = HashSet::new();
        let mut kept: Vec<KeywordMatch> = Vec::new();
        let mut cat_spans: HashMap<String, Vec<(usize, usize)>> = HashMap::new();

        for m in sorted {
            let key = (m.category_id.clone(), m.text.clone());
            if seen_texts.contains(&key) {
                continue;
            }

            // Check containment
            let spans = cat_spans.get(&m.category_id);
            let contained = spans
                .map(|s| s.iter().any(|&(s, e)| s <= m.start && m.end <= e))
                .unwrap_or(false);

            if contained && m.kind == MatchKind::Keyword {
                continue;
            }

            seen_texts.insert(key);
            cat_spans
                .entry(m.category_id.clone())
                .or_default()
                .push((m.start, m.end));
            kept.push(m);
        }

        kept
    }

    fn apply_position_bonus(
        &self,
        line: &str,
        matches: Vec<KeywordMatch>,
        initial_char_offset: usize,
    ) -> Vec<KeywordMatch> {
        let discourse_categories: HashSet<&str> = [
            "discourse_cause_result",
            "discourse_contrast",
            "discourse_addition",
            "discourse_choice",
            "discourse_explanatory",
            "discourse_topic_shift",
        ]
        .into_iter()
        .collect();

        matches
            .into_iter()
            .map(|mut m| {
                if discourse_categories.contains(m.category_id.as_str()) {
                    // KeywordMatch.start/end 存的是字节偏移；奖励窗口用字符序号
                    let match_char_start = byte_offset_to_char_index(line, m.start);
                    if match_char_start >= initial_char_offset {
                        let effective_chars = match_char_start - initial_char_offset;
                        if effective_chars < SENTENCE_INITIAL_CHARS {
                            m.weight = (m.weight * SENTENCE_INITIAL_BONUS * 10.0).round() / 10.0;
                        }
                    }
                }
                m
            })
            .collect()
    }

    // ── Dialogue helpers ────────────────────────────────────────────────

    fn extract_dialogue_content(line: &str) -> Option<&str> {
        let s = line.trim();
        for (open, close) in [('「', '」'), ('『', '』')] {
            if s.starts_with(open) {
                if let Some(end_idx) = s.rfind(close) {
                    if end_idx > open.len_utf8() {
                        return Some(&s[open.len_utf8()..end_idx]);
                    }
                }
            }
        }
        None
    }

    // ── Scoring and recommendation ──────────────────────────────────────

    fn pick_dominant(scores: &HashMap<String, f64>, priorities: &HashMap<String, i32>) -> String {
        scores
            .keys()
            .max_by(|a, b| {
                let pa = priorities.get(*a).copied().unwrap_or(3);
                let pb = priorities.get(*b).copied().unwrap_or(3);
                pa.cmp(&pb).then_with(|| {
                    let sa = scores.get(*a).copied().unwrap_or(0.0);
                    let sb = scores.get(*b).copied().unwrap_or(0.0);
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                })
            })
            .cloned()
            .unwrap_or_default()
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    fn compile_regex_list(boundaries: &Value, key: &str) -> Result<Vec<Regex>> {
        let mut result = Vec::new();
        if let Some(arr) = boundaries.get(key).and_then(|v| v.as_array()) {
            for item in arr {
                if let Some(pattern) = item.as_str() {
                    match Regex::new(pattern) {
                        Ok(re) => result.push(re),
                        Err(e) => {
                            tracing::warn!("Failed to compile regex '{}': {}", pattern, e);
                        }
                    }
                }
            }
        }
        Ok(result)
    }
}

/// 将字节偏移转为 Unicode 标量序号；偏移超出或落在码点中间时钳到最近合法位置。
fn byte_offset_to_char_index(s: &str, byte_offset: usize) -> usize {
    if byte_offset == 0 {
        return 0;
    }
    if byte_offset >= s.len() {
        return s.chars().count();
    }
    // floor to char boundary
    let mut end = byte_offset;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].chars().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_offset_to_char_index_handles_multibyte() {
        // 每个日文假名 3 字节
        let s = "あいうえお";
        assert_eq!(byte_offset_to_char_index(s, 0), 0);
        assert_eq!(byte_offset_to_char_index(s, 3), 1);
        assert_eq!(byte_offset_to_char_index(s, 9), 3);
        assert_eq!(byte_offset_to_char_index(s, s.len()), 5);
    }

    #[test]
    fn position_bonus_uses_char_window_not_bytes() {
        // 前缀「」各 3 字节；若误用字节比较，SENTENCE_INITIAL_CHARS=6 只能覆盖 2 个假名
        let config = serde_json::json!({
            "boundaries": {
                "scene_break_regex": [],
                "chapter_heading_regex": [],
                "metadata_regex": []
            },
            "categories": [{
                "id": "discourse_cause_result",
                "label": "因果",
                "priority": 1,
                "base_weight": 10.0,
                "actions": {},
                "match": {
                    "keywords": [{"text": "だから", "weight": 10.0}],
                    "regex": []
                }
            }]
        });
        let detector = ContextDetector::from_config(&config).expect("config");
        // 「あああだから…」：だから 在引号后第 4 字符处（字符索引 1+3=4），仍应在 6 字窗口内
        let line = "「あああだから行った」";
        let result = detector.detect_line(line, &[]);
        let hit = result
            .matches
            .iter()
            .find(|m| m.category_id == "discourse_cause_result")
            .expect("keyword should match");
        // 基础 10 * 1.4 = 14
        assert!(
            (hit.weight - 14.0).abs() < 0.01,
            "expected sentence-initial bonus, got {}",
            hit.weight
        );
    }

    #[test]
    fn test_classify_line() {
        let config = serde_json::json!({
            "boundaries": {
                "scene_break_regex": [r"^[\s　]*[◆◇■□★☆※＊*＝=─—―-]{3,}[\s　]*$"],
                "chapter_heading_regex": [r"^(?:プロローグ|エピローグ)(?:\s|　|$)"],
                "metadata_regex": []
            },
            "categories": []
        });
        let detector = ContextDetector::from_config(&config).expect("should parse config");

        assert_eq!(detector.classify_line(""), LineType::Blank);
        assert_eq!(detector.classify_line("「こんにちは」"), LineType::Dialogue);
        assert_eq!(
            detector.classify_line("普通のテキスト"),
            LineType::Narration
        );
        assert_eq!(detector.classify_line("◆◆◆"), LineType::SceneBreak);
        assert_eq!(
            detector.classify_line("プロローグ"),
            LineType::ChapterHeading
        );
    }
}
