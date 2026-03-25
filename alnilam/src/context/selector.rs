use std::collections::{HashMap, HashSet};

use regex::Regex;

use super::detector::ContextDetector;
use super::types::{DetectionResult, LineType};

// ── Regex patterns for line features ─────────────────────────────────────

macro_rules! lazy_static_regex {
    ($($name:ident = $pattern:expr;)*) => {
        $(
            fn $name() -> &'static Regex {
                static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
                RE.get_or_init(|| Regex::new($pattern).expect(concat!("Failed to compile regex: ", $pattern)))
            }
        )*
    };
}

lazy_static_regex! {
    name_hint_re = r"(?:[一-龥]{2,6}|[ァ-ヴー]{3,}|[ぁ-ん]{3,})(?:さん|様|さま|くん|君|ちゃん|先輩|先生|師匠)";
    kanji_name_re = r"[一-龥]{2,6}";
    katakana_long_re = r"[ァ-ヴー]{4,}(?:・[ァ-ヴー]{2,})*";
    time_anchor_re = r"(昨日|今日|明日|翌日|その後|その時|あの時|当時|先ほど|先程|さっき)";
    speech_verb_re = r"と(?:言っ|言われ|呟|囁|叫|怒鳴|答え|返事|尋ね|訊ね|問う|続け|付け加え)";
}

// ── Data structures ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SelectedLine {
    pub line_number: usize,
    pub text: String,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct SelectionResult {
    pub selected: Vec<SelectedLine>,
}

// ── Precomputed context ─────────────────────────────────────────────────

/// 预计算的上下文数据，包含所有行的检测结果和特征。
/// 调用一次 `precompute_context()` 后，可多次复用 `select_context_precomputed()`。
pub struct PrecomputedContext {
    /// segment_start_before[line_no] = 当前行所属段落的起始行号 (1-based)
    segment_start_before: Vec<usize>,
    /// 每行的特征集合 (0-based index)
    features: Vec<HashSet<&'static str>>,
    /// 每行的检测结果 (0-based index)
    detection_results: Vec<DetectionResult>,
}

impl PrecomputedContext {
    /// 创建一个空的预计算上下文（用于测试或无检测器时的回退）
    pub fn empty(num_lines: usize) -> Self {
        Self {
            segment_start_before: vec![1; num_lines + 1],
            features: vec![HashSet::new(); num_lines],
            detection_results: vec![DetectionResult::default(); num_lines],
        }
    }

    /// 检查指定 0-based 索引处的行是否触发上下文重置
    pub fn is_reset_at(&self, idx: usize) -> bool {
        self.detection_results
            .get(idx)
            .map_or(false, |r| r.reset_context())
    }
}

// ── Feature extraction ───────────────────────────────────────────────────

fn line_features(text: &str) -> HashSet<&'static str> {
    let mut feats = HashSet::new();
    let trimmed = text.trim();

    if trimmed.starts_with('「') || trimmed.starts_with('『') {
        feats.insert("dialogue");
    }
    if text.contains('「') || text.contains('『') {
        feats.insert("has_quote");
    }
    if name_hint_re().is_match(text) {
        feats.insert("name_hint");
    }
    if katakana_long_re().is_match(text) {
        feats.insert("katakana_long");
    }
    if kanji_name_re().is_match(text) {
        feats.insert("kanji");
    }
    if time_anchor_re().is_match(text) {
        feats.insert("time_anchor");
    }
    if speech_verb_re().is_match(text) {
        feats.insert("speech_verb");
    }

    feats
}

fn target_needs(result: &DetectionResult) -> HashSet<&'static str> {
    let mut needs = HashSet::new();

    if result.line_type == LineType::Dialogue {
        needs.insert("dialogue_prev");
    }

    if let Some(ref dom) = result.dominant_category {
        if dom.contains("指代") || dom.contains("指示词") {
            needs.insert("antecedent");
        }
        if dom.contains("显式回指") {
            needs.insert("anchor");
        }
    }

    for m in &result.matches {
        match m.category_id.as_str() {
            "quote_and_speech_verbs" => {
                needs.insert("dialogue_prev");
            }
            "hearsay_and_source" => {
                needs.insert("source");
            }
            "benefactive_construction" => {
                needs.insert("relationship");
            }
            _ => {}
        }
    }

    needs
}

// ── Precomputation ──────────────────────────────────────────────────────

/// 预计算所有行的上下文检测结果和特征。
///
/// 完成后可多次调用 `select_context_precomputed()` 而无需重复检测。
pub fn precompute_context(
    detector: &ContextDetector,
    lines: &[String],
) -> PrecomputedContext {
    let mut segment_start_before = vec![1; lines.len() + 1];
    let mut features = Vec::with_capacity(lines.len());
    let mut detection_results = Vec::with_capacity(lines.len());
    let mut segment_start = 1usize;
    let mut history: Vec<String> = Vec::new();

    for (idx, line) in lines.iter().enumerate() {
        let line_no = idx + 1;
        segment_start_before[line_no] = segment_start;
        let r = detector.detect_line(line, &history);
        detection_results.push(r.clone());
        features.push(line_features(line));
        if r.reset_context() {
            history.clear();
            segment_start = line_no + 1;
        } else {
            history.push(line.clone());
        }
    }

    PrecomputedContext {
        segment_start_before,
        features,
        detection_results,
    }
}

/// 使用预计算数据选择上下文行（高效版本，适合多次调用）。
pub fn select_context_precomputed(
    lines: &[String],
    precomputed: &PrecomputedContext,
    target_start: usize,
    target_end: usize,
    window: usize,
    max_selected_lines: usize,
) -> anyhow::Result<SelectionResult> {
    if target_start < 1 || target_end < 1 || target_end < target_start {
        anyhow::bail!("Invalid target range");
    }
    if target_end > lines.len() {
        anyhow::bail!("Target end out of file range");
    }
    if window == 0 {
        anyhow::bail!("window must be positive");
    }
    if max_selected_lines == 0 {
        anyhow::bail!("max_selected_lines must be positive");
    }

    let segment_start = precomputed
        .segment_start_before
        .get(target_start)
        .copied()
        .unwrap_or(1);
    let allowed_start = segment_start.max(target_start.saturating_sub(window));
    let allowed_end = target_start - 1;

    if allowed_end < allowed_start {
        return Ok(SelectionResult { selected: vec![] });
    }

    // Get target detection results from precomputed data
    let mut target_results: Vec<DetectionResult> = Vec::new();
    for i in target_start..=target_end {
        if let Some(r) = precomputed.detection_results.get(i - 1) {
            target_results.push(r.clone());
        }
    }

    let mut needs: HashSet<&'static str> = HashSet::new();
    for tr in &target_results {
        needs.extend(target_needs(tr));
    }

    // Use precomputed features for candidate lines
    let candidate_features: HashMap<usize, HashSet<&str>> = (allowed_start..=allowed_end)
        .map(|i| {
            let feats = precomputed
                .features
                .get(i - 1)
                .cloned()
                .unwrap_or_default();
            (i, feats)
        })
        .collect();

    let scored = score_candidates(
        &needs,
        &candidate_features,
        allowed_start,
        allowed_end,
        target_start,
        max_selected_lines,
    );

    Ok(scored)
}

// ── Segment start computation (used by select_context) ──────────────────

fn compute_segment_start(
    detector: &ContextDetector,
    lines: &[String],
    upto_line: usize,
) -> usize {
    let mut segment_start = 1;
    let mut history: Vec<String> = Vec::new();

    for i in 1..=upto_line {
        let line = &lines[i - 1];
        let r = detector.detect_line(line, &history);
        if r.reset_context() {
            history.clear();
            segment_start = i + 1;
        } else {
            history.push(line.clone());
        }
    }

    segment_start
}

// ── Main selection function (legacy, recomputes each time) ──────────────

pub fn select_context(
    detector: &ContextDetector,
    lines: &[String],
    target_start: usize,
    target_end: usize,
    window: usize,
    max_selected_lines: usize,
) -> anyhow::Result<SelectionResult> {
    if target_start < 1 || target_end < 1 || target_end < target_start {
        anyhow::bail!("Invalid target range");
    }
    if target_end > lines.len() {
        anyhow::bail!("Target end out of file range");
    }
    if window == 0 {
        anyhow::bail!("window must be positive");
    }
    if max_selected_lines == 0 {
        anyhow::bail!("max_selected_lines must be positive");
    }

    let segment_start = compute_segment_start(detector, lines, target_start - 1);
    let allowed_start = segment_start.max(target_start.saturating_sub(window));
    let allowed_end = target_start - 1;

    if allowed_end < allowed_start {
        return Ok(SelectionResult {
            selected: vec![],
        });
    }

    // Build segment history
    let mut segment_history: Vec<String> = Vec::new();
    for i in segment_start..target_start {
        let line = &lines[i - 1];
        let r = detector.detect_line(line, &segment_history);
        if r.reset_context() {
            segment_history.clear();
        } else {
            segment_history.push(line.clone());
        }
    }

    // Detect target lines
    let mut target_results: Vec<DetectionResult> = Vec::new();
    let mut history_for_targets = segment_history.clone();
    for i in target_start..=target_end {
        let line = &lines[i - 1];
        let r = detector.detect_line(line, &history_for_targets);
        target_results.push(r.clone());
        if r.reset_context() {
            history_for_targets.clear();
        } else {
            history_for_targets.push(line.clone());
        }
    }

    // Gather needs
    let mut needs: HashSet<&'static str> = HashSet::new();
    for tr in &target_results {
        needs.extend(target_needs(tr));
    }

    // Compute features for candidate lines
    let candidate_features: HashMap<usize, HashSet<&str>> = (allowed_start..=allowed_end)
        .map(|i| (i, line_features(&lines[i - 1])))
        .collect();

    let scored = score_candidates(
        &needs,
        &candidate_features,
        allowed_start,
        allowed_end,
        target_start,
        max_selected_lines,
    );

    // Convert to SelectedLine with text (we have access to lines)
    let selected: Vec<SelectedLine> = scored
        .selected
        .into_iter()
        .map(|sl| SelectedLine {
            line_number: sl.line_number,
            text: lines[sl.line_number - 1].clone(),
            score: sl.score,
        })
        .collect();

    Ok(SelectionResult { selected })
}

// ── Shared scoring logic ────────────────────────────────────────────────

fn score_candidates(
    needs: &HashSet<&'static str>,
    candidate_features: &HashMap<usize, HashSet<&str>>,
    allowed_start: usize,
    allowed_end: usize,
    target_start: usize,
    max_selected_lines: usize,
) -> SelectionResult {
    let mut scored: HashMap<usize, (f64, HashSet<String>)> = HashMap::new();

    let mut add_score = |line_no: usize, delta: f64, reason: &str| {
        if line_no < allowed_start || line_no > allowed_end {
            return;
        }
        let entry = scored
            .entry(line_no)
            .or_insert_with(|| (0.0, HashSet::new()));
        entry.0 += delta;
        entry.1.insert(reason.to_string());
    };

    // Proximity scoring
    for ln in allowed_start..=allowed_end {
        let distance = target_start - ln;
        add_score(ln, 1.0 / distance.max(1) as f64, "proximity");
    }

    // Near target bonus
    let near_start = allowed_start.max(target_start.saturating_sub(2));
    for ln in near_start..=allowed_end {
        add_score(ln, 0.6, "near_target");
    }

    // Dialogue needs
    if needs.contains("dialogue_prev") {
        // Find nearest dialogue line
        for ln in (allowed_start..=allowed_end).rev() {
            if let Some(feats) = candidate_features.get(&ln) {
                if feats.contains("dialogue") {
                    add_score(ln, 3.0, "dialogue_anchor");
                    break;
                }
            }
        }
        // Recent dialogues
        let recent_start = allowed_start.max(allowed_end.saturating_sub(6));
        for ln in (recent_start..=allowed_end).rev() {
            if let Some(feats) = candidate_features.get(&ln) {
                if feats.contains("dialogue") {
                    add_score(ln, 1.0, "dialogue_recent");
                }
            }
        }
    }

    // Antecedent needs
    if needs.contains("antecedent") {
        let search_start = allowed_start.max(allowed_end.saturating_sub(20));
        for ln in (search_start..=allowed_end).rev() {
            if let Some(feats) = candidate_features.get(&ln) {
                if feats.contains("name_hint") {
                    add_score(ln, 2.2, "name_hint");
                }
                if feats.contains("katakana_long") {
                    add_score(ln, 1.2, "katakana");
                }
            }
        }
    }

    // Anchor/source needs
    if needs.contains("anchor") || needs.contains("source") {
        let search_start = allowed_start.max(allowed_end.saturating_sub(25));
        for ln in (search_start..=allowed_end).rev() {
            if let Some(feats) = candidate_features.get(&ln) {
                if feats.contains("speech_verb") {
                    add_score(ln, 1.6, "speech_verb");
                }
                if feats.contains("time_anchor") {
                    add_score(ln, 1.4, "time_anchor");
                }
            }
        }
    }

    // Relationship needs
    if needs.contains("relationship") {
        let search_start = allowed_start.max(allowed_end.saturating_sub(15));
        for ln in (search_start..=allowed_end).rev() {
            if let Some(feats) = candidate_features.get(&ln) {
                if feats.contains("name_hint") {
                    add_score(ln, 1.0, "relationship_name");
                }
            }
        }
    }

    // Previous line bonus
    if target_start > 0 && target_start - 1 >= allowed_start {
        add_score(target_start - 1, 2.0, "prev_line");
    }

    // Rank and select
    let mut ranked: Vec<(usize, f64, HashSet<String>)> = scored
        .into_iter()
        .map(|(ln, (score, reasons))| (ln, score, reasons))
        .collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.0.cmp(&a.0))
    });

    let mut selected_ranked: Vec<(usize, f64)> = Vec::new();
    for (ln, score, _) in &ranked {
        if !selected_ranked.iter().any(|(sln, _)| sln == ln) {
            selected_ranked.push((*ln, *score));
            if selected_ranked.len() >= max_selected_lines {
                break;
            }
        }
    }

    selected_ranked.sort_by(|a, b| a.0.cmp(&b.0));

    let selected: Vec<SelectedLine> = selected_ranked
        .iter()
        .map(|&(ln, score)| SelectedLine {
            line_number: ln,
            text: String::new(), // caller fills in text if needed
            score,
        })
        .collect();

    SelectionResult { selected }
}
