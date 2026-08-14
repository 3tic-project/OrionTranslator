use std::collections::{BTreeMap, HashSet};

use regex::Regex;

use super::types::{CheckResult, ErrorType};
use crate::llm::glossary::{self, GlossaryEntry};

// ── Text helpers ─────────────────────────────────────────────────────────

fn is_hiragana(c: char) -> bool {
    ('\u{3040}'..='\u{309F}').contains(&c)
}

fn is_katakana(c: char) -> bool {
    ('\u{30A0}'..='\u{30FF}').contains(&c)
}

fn is_cjk(c: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&c)
        || ('\u{3400}'..='\u{4DBF}').contains(&c)
        || ('\u{F900}'..='\u{FAFF}').contains(&c)
}

fn has_cjk(text: &str) -> bool {
    text.chars().any(is_cjk)
}

fn is_kana(c: char) -> bool {
    is_hiragana(c) || is_katakana(c)
}

fn problematic_kana_runs<'a>(src: &str, text: &'a str) -> Vec<&'a str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut i = 0usize;
    let mut problematic = Vec::new();

    while i < chars.len() {
        if !is_kana(chars[i].1) {
            i += 1;
            continue;
        }

        let start_byte = chars[i].0;
        let mut end = i + 1;
        while end < chars.len() && is_kana(chars[end].1) {
            end += 1;
        }
        let end_byte = chars
            .get(end)
            .map(|(byte, _)| *byte)
            .unwrap_or_else(|| text.len());
        let run = &text[start_byte..end_byte];
        let prev = if i == 0 { None } else { Some(chars[i - 1].1) };
        let next = chars.get(end).map(|(_, c)| *c);

        if !is_allowed_shape_kana(run, prev, next) && !is_protected_japanese_token(src, run) {
            problematic.push(run);
        }
        i = end;
    }

    problematic
}

fn has_problematic_kana_residue(src: &str, text: &str) -> bool {
    !problematic_kana_runs(src, text).is_empty()
}

fn is_protected_japanese_token(src: &str, run: &str) -> bool {
    if run.is_empty() || !src.contains(run) {
        return false;
    }
    is_inside_source_quotes(src, run) || is_credit_line(src)
}

fn is_inside_source_quotes(src: &str, token: &str) -> bool {
    const OPEN: &[char] = &[
        '「', '『', '〝', '“', '‘', '（', '(', '【', '[', '《', '〈', '"',
    ];
    const CLOSE: &[char] = &[
        '」', '』', '〟', '”', '’', '）', ')', '】', ']', '》', '〉', '"',
    ];

    src.match_indices(token).any(|(start, matched)| {
        let prefix = &src[..start];
        let suffix = &src[start + matched.len()..];
        let last_open = prefix
            .char_indices()
            .filter(|(_, character)| OPEN.contains(character))
            .map(|(index, _)| index)
            .next_back();
        let last_close = prefix
            .char_indices()
            .filter(|(_, character)| CLOSE.contains(character))
            .map(|(index, _)| index)
            .next_back();
        last_open.is_some_and(|open| last_close.is_none_or(|close| open >= close))
            && suffix.chars().any(|character| CLOSE.contains(&character))
    })
}

fn is_credit_line(src: &str) -> bool {
    [
        "イラスト",
        "口絵",
        "挿絵",
        "装画",
        "原作",
        "著者",
        "印刷・製本",
    ]
    .iter()
    .any(|marker| src.contains(marker))
}

fn is_allowed_shape_kana(run: &str, prev: Option<char>, next: Option<char>) -> bool {
    let shape_markers = ['字', '形', '型', '状', '口', '角', '弯', '折', '线'];
    let allowed_runs = [
        "く", "へ", "し", "つ", "の", "ノ", "ハ", "コ", "ロ", "ニ", "へ", "への",
    ];
    let quoted = matches!(prev, Some('“' | '"' | '「' | '『' | '《' | '(' | '（'))
        && matches!(next, Some('”' | '"' | '」' | '』' | '》' | ')' | '）'));

    if quoted && run.chars().count() <= 2 && allowed_runs.contains(&run) {
        return true;
    }

    allowed_runs.contains(&run) && next.is_some_and(|c| shape_markers.contains(&c))
}

fn is_hangeul(c: char) -> bool {
    ('\u{AC00}'..='\u{D7AF}').contains(&c) || ('\u{1100}'..='\u{11FF}').contains(&c)
}

fn any_hangeul(text: &str) -> bool {
    text.chars().any(is_hangeul)
}

fn is_cjk_punctuation(c: char) -> bool {
    matches!(c,
        '\u{3001}'..='\u{303F}' |
        '\u{FF01}'..='\u{FF0F}' |
        '\u{FF1A}'..='\u{FF1F}' |
        '\u{FF3B}'..='\u{FF40}' |
        '\u{FF5B}'..='\u{FF65}'
    )
}

fn is_latin_punctuation(c: char) -> bool {
    matches!(c,
        '\u{0021}'..='\u{002F}' |
        '\u{003A}'..='\u{0040}' |
        '\u{005B}'..='\u{0060}' |
        '\u{007B}'..='\u{007E}'
    )
}

fn is_only_punctuation_and_space(text: &str) -> bool {
    text.chars()
        .all(|c| c.is_whitespace() || is_cjk_punctuation(c) || is_latin_punctuation(c))
}

fn jaccard_similarity(text1: &str, text2: &str) -> f64 {
    let set1: HashSet<char> = text1.chars().collect();
    let set2: HashSet<char> = text2.chars().collect();
    let union = set1.union(&set2).count();
    if union == 0 {
        return 0.0;
    }
    let intersection = set1.intersection(&set2).count();
    intersection as f64 / union as f64
}

fn fullwidth_to_halfwidth(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '０' => '0',
            '１' => '1',
            '２' => '2',
            '３' => '3',
            '４' => '4',
            '５' => '5',
            '６' => '6',
            '７' => '7',
            '８' => '8',
            '９' => '9',
            other => other,
        })
        .collect()
}

// ── Lazy regex patterns ──────────────────────────────────────────────────

/// Check for repeated substrings (replacement for backreference-based regex)
fn has_repeated_substring(
    text: &str,
    min_pattern_len: usize,
    max_pattern_len: usize,
    min_repeats: usize,
) -> bool {
    let chars: Vec<char> = text.chars().collect();
    for pat_len in min_pattern_len..=max_pattern_len.min(chars.len() / min_repeats) {
        for start in 0..chars.len().saturating_sub(pat_len * min_repeats) {
            let pattern: String = chars[start..start + pat_len].iter().collect();
            let mut count = 1;
            let mut pos = start + pat_len;
            while pos + pat_len <= chars.len() {
                let sub: String = chars[pos..pos + pat_len].iter().collect();
                if sub == pattern {
                    count += 1;
                    pos += pat_len;
                } else {
                    break;
                }
            }
            if count >= min_repeats {
                return true;
            }
        }
    }
    false
}

fn re_json_escape() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"(\\?"\\?:\\?")|(":")|(":")"#).expect("json escape regex"))
}

fn re_digits() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\d+").expect("digits regex"))
}

fn re_placeholder() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"\{\{[^{}\r\n]+\}\}|\{[A-Za-z_][A-Za-z0-9_.-]*\}|%(?:\d+\$)?[sdif]|__[A-Z][A-Z0-9_]*__",
        )
        .expect("placeholder regex")
    })
}

fn token_multiset(regex: &Regex, text: &str) -> BTreeMap<String, usize> {
    let mut tokens = BTreeMap::new();
    for matched in regex.find_iter(text) {
        *tokens.entry(matched.as_str().to_string()).or_default() += 1;
    }
    tokens
}

// ── ResponseChecker ──────────────────────────────────────────────────────

pub struct ResponseChecker {
    source_lang: String,
    target_lang: String,
    similarity_threshold: f64,
    retry_threshold: usize,
    glossary_entries: Vec<GlossaryEntry>,
}

impl ResponseChecker {
    pub fn new(
        source_lang: &str,
        target_lang: &str,
        similarity_threshold: f64,
        retry_threshold: usize,
    ) -> Self {
        Self {
            source_lang: source_lang.to_lowercase(),
            target_lang: target_lang.to_lowercase(),
            similarity_threshold,
            retry_threshold,
            glossary_entries: Vec::new(),
        }
    }

    pub fn with_glossary_entries(mut self, entries: Vec<GlossaryEntry>) -> Self {
        self.glossary_entries = entries;
        self
    }

    pub fn check(&self, srcs: &[String], dsts: &[String], retry_count: usize) -> Vec<CheckResult> {
        // 1. Data parse failure
        if dsts.is_empty() || dsts.iter().all(|d| d.trim().is_empty()) {
            return srcs
                .iter()
                .map(|_| CheckResult {
                    error: ErrorType::FailParse,
                    details: "响应为空或解析失败".to_string(),
                })
                .collect();
        }

        // 2. Line count mismatch
        if srcs.len() != dsts.len() {
            return srcs
                .iter()
                .map(|_| CheckResult {
                    error: ErrorType::FailLineCount,
                    details: format!("行数不匹配: 原文{}行, 译文{}行", srcs.len(), dsts.len()),
                })
                .collect();
        }

        // 3. Per-line check. After retry threshold, keep hard safety checks but
        // skip soft heuristics that are prone to false positives.
        let skip_soft_checks = retry_count >= self.retry_threshold;
        srcs.iter()
            .zip(dsts.iter())
            .map(|(src, dst)| self.check_line(src, dst, skip_soft_checks))
            .collect()
    }

    fn check_line(&self, src: &str, dst: &str, skip_soft_checks: bool) -> CheckResult {
        let src = src.trim();
        let dst = dst.trim();

        // Empty translation
        if !src.is_empty() && dst.is_empty() {
            return CheckResult {
                error: ErrorType::EmptyTranslation,
                details: "原文非空但译文为空".to_string(),
            };
        }

        // Confirmed glossary constraints are hard QA: retries must not disable them.
        if !self.glossary_entries.is_empty() {
            let matched =
                glossary::filter_glossary_for_texts(&self.glossary_entries, &[src.to_string()]);
            let mut missing = Vec::new();
            for entry in matched {
                let target = entry.dst.trim();
                if !target.is_empty() && !dst.contains(target) {
                    missing.push(format!("{}→{}", entry.src.trim(), target));
                }
            }
            missing.sort();
            missing.dedup();
            if !missing.is_empty() {
                return CheckResult {
                    error: ErrorType::TermMissing,
                    details: format!("已确认术语未按约束呈现: {}", missing.join(", ")),
                };
            }
        }

        // Skip pure punctuation/numbers
        if is_only_punctuation_and_space(src) {
            return CheckResult {
                error: ErrorType::None,
                details: String::new(),
            };
        }

        // JSON structure anomaly
        let json_matches: Vec<_> = re_json_escape().find_iter(dst).collect();
        if json_matches.len() >= 3 {
            return CheckResult {
                error: ErrorType::JsonStructureError,
                details: format!("译文包含 JSON 结构片段 ({} 处)", json_matches.len()),
            };
        }

        // Degradation detection (multi-char pattern repeat)
        let src_has_repeat = has_repeated_substring(src, 1, 3, 16);
        let dst_has_repeat = has_repeated_substring(dst, 1, 3, 16);
        if !src_has_repeat && dst_has_repeat {
            return CheckResult {
                error: ErrorType::Degradation,
                details: "检测到退化（重复文本）".to_string(),
            };
        }

        // Extended degradation (longer pattern repeats)
        let src_ext = has_repeated_substring(src, 4, 10, 4);
        let dst_ext = has_repeated_substring(dst, 4, 10, 4);
        if !src_ext && dst_ext {
            return CheckResult {
                error: ErrorType::Degradation,
                details: "检测到退化（片段重复）".to_string(),
            };
        }

        // Arabic numbers are hard QA. Width differences are normalized, counts are preserved.
        let src_normalized = fullwidth_to_halfwidth(src);
        let dst_normalized = fullwidth_to_halfwidth(dst);
        let src_numbers = token_multiset(re_digits(), &src_normalized);
        let dst_numbers = token_multiset(re_digits(), &dst_normalized);
        if src_numbers != dst_numbers {
            return CheckResult {
                error: ErrorType::NumberMismatch,
                details: format!("数字不一致: src={:?}, dst={:?}", src_numbers, dst_numbers),
            };
        }

        // Template/control placeholders are hard QA and must survive verbatim.
        let src_placeholders = token_multiset(re_placeholder(), src);
        let dst_placeholders = token_multiset(re_placeholder(), dst);
        if src_placeholders != dst_placeholders {
            return CheckResult {
                error: ErrorType::PlaceholderMismatch,
                details: format!(
                    "占位符不一致: src={:?}, dst={:?}",
                    src_placeholders, dst_placeholders
                ),
            };
        }

        // Kana residue (ja -> other). Quoted puzzle/name tokens copied from the
        // source and credit-line names are protected; other runs are prose.
        if self.source_lang == "ja"
            && self.target_lang != "ja"
            && has_problematic_kana_residue(src, dst)
        {
            let runs = problematic_kana_runs(src, dst);
            return CheckResult {
                error: ErrorType::KanaUntranslatedProse,
                details: format!("译文中残留未保护的假名正文: {}", runs.join(" / ")),
            };
        }

        // Hangeul residue (ko -> other)
        if self.source_lang == "ko" && self.target_lang != "ko" && any_hangeul(dst) {
            return CheckResult {
                error: ErrorType::HangeulResidue,
                details: "译文中残留谚文".to_string(),
            };
        }

        if skip_soft_checks {
            return CheckResult {
                error: ErrorType::None,
                details: String::new(),
            };
        }

        if src == dst && src.chars().count() >= 3 && has_cjk(src) {
            return CheckResult {
                error: ErrorType::HighSimilarity,
                details: "译文与原文完全相同".to_string(),
            };
        }

        // Length ratio check
        if src.chars().count() >= 10 && dst.chars().count() >= 5 {
            let ratio = dst.chars().count() as f64 / src.chars().count() as f64;
            if ratio < 0.3 {
                return CheckResult {
                    error: ErrorType::LengthMismatch,
                    details: format!("译文过短 (ratio={:.2})", ratio),
                };
            }
            if ratio > 3.0 {
                return CheckResult {
                    error: ErrorType::LengthMismatch,
                    details: format!("译文过长 (ratio={:.2})", ratio),
                };
            }
        }

        // Similarity check
        if self.check_similarity(src, dst) {
            let should_flag = if self.source_lang == "ja" && self.target_lang == "zh" {
                has_problematic_kana_residue(src, dst)
            } else if self.source_lang == "ko" && self.target_lang == "zh" {
                any_hangeul(dst)
            } else {
                true
            };

            if should_flag {
                return CheckResult {
                    error: ErrorType::HighSimilarity,
                    details: "原译文高度相似（疑似未翻译）".to_string(),
                };
            }
        }

        CheckResult {
            error: ErrorType::None,
            details: String::new(),
        }
    }

    /// Audit a single translation without short-circuiting after the first finding.
    ///
    /// The online pipeline continues to use [`Self::check`], whose first-error
    /// priority and retry behavior remain unchanged. Offline quality baselines use
    /// this method so a terminology miss cannot hide a number or placeholder loss
    /// on the same unit.
    pub fn audit_line(&self, src: &str, dst: &str, include_soft_checks: bool) -> Vec<CheckResult> {
        let src = src.trim();
        let dst = dst.trim();
        let mut findings = Vec::new();

        if !src.is_empty() && dst.is_empty() {
            findings.push(CheckResult {
                error: ErrorType::EmptyTranslation,
                details: "原文非空但译文为空".to_string(),
            });
            return findings;
        }

        if !self.glossary_entries.is_empty() {
            let matched =
                glossary::filter_glossary_for_texts(&self.glossary_entries, &[src.to_string()]);
            let mut missing = matched
                .into_iter()
                .filter_map(|entry| {
                    let target = entry.dst.trim();
                    (!target.is_empty() && !dst.contains(target))
                        .then(|| format!("{}→{}", entry.src.trim(), target))
                })
                .collect::<Vec<_>>();
            missing.sort();
            missing.dedup();
            if !missing.is_empty() {
                findings.push(CheckResult {
                    error: ErrorType::TermMissing,
                    details: format!("已确认术语未按约束呈现: {}", missing.join(", ")),
                });
            }
        }

        if is_only_punctuation_and_space(src) {
            return findings;
        }

        let json_matches: Vec<_> = re_json_escape().find_iter(dst).collect();
        if json_matches.len() >= 3 {
            findings.push(CheckResult {
                error: ErrorType::JsonStructureError,
                details: format!("译文包含 JSON 结构片段 ({} 处)", json_matches.len()),
            });
        }

        let src_has_repeat = has_repeated_substring(src, 1, 3, 16);
        let dst_has_repeat = has_repeated_substring(dst, 1, 3, 16);
        let src_ext = has_repeated_substring(src, 4, 10, 4);
        let dst_ext = has_repeated_substring(dst, 4, 10, 4);
        if !src_has_repeat && dst_has_repeat {
            findings.push(CheckResult {
                error: ErrorType::Degradation,
                details: "检测到退化（重复文本）".to_string(),
            });
        }
        if !src_ext && dst_ext {
            findings.push(CheckResult {
                error: ErrorType::Degradation,
                details: "检测到退化（片段重复）".to_string(),
            });
        }

        let src_normalized = fullwidth_to_halfwidth(src);
        let dst_normalized = fullwidth_to_halfwidth(dst);
        let src_numbers = token_multiset(re_digits(), &src_normalized);
        let dst_numbers = token_multiset(re_digits(), &dst_normalized);
        if src_numbers != dst_numbers {
            findings.push(CheckResult {
                error: ErrorType::NumberMismatch,
                details: format!("数字不一致: src={:?}, dst={:?}", src_numbers, dst_numbers),
            });
        }

        let src_placeholders = token_multiset(re_placeholder(), src);
        let dst_placeholders = token_multiset(re_placeholder(), dst);
        if src_placeholders != dst_placeholders {
            findings.push(CheckResult {
                error: ErrorType::PlaceholderMismatch,
                details: format!(
                    "占位符不一致: src={:?}, dst={:?}",
                    src_placeholders, dst_placeholders
                ),
            });
        }

        if self.source_lang == "ja"
            && self.target_lang != "ja"
            && has_problematic_kana_residue(src, dst)
        {
            findings.push(CheckResult {
                error: ErrorType::KanaUntranslatedProse,
                details: format!(
                    "译文中残留未保护的假名正文: {}",
                    problematic_kana_runs(src, dst).join(" / ")
                ),
            });
        }

        if self.source_lang == "ko" && self.target_lang != "ko" && any_hangeul(dst) {
            findings.push(CheckResult {
                error: ErrorType::HangeulResidue,
                details: "译文中残留谚文".to_string(),
            });
        }

        if !include_soft_checks {
            return findings;
        }

        if src == dst && src.chars().count() >= 3 && has_cjk(src) {
            findings.push(CheckResult {
                error: ErrorType::HighSimilarity,
                details: "译文与原文完全相同".to_string(),
            });
        }

        if src.chars().count() >= 10 && dst.chars().count() >= 5 {
            let ratio = dst.chars().count() as f64 / src.chars().count() as f64;
            if ratio < 0.3 {
                findings.push(CheckResult {
                    error: ErrorType::LengthMismatch,
                    details: format!("译文过短 (ratio={:.2})", ratio),
                });
            } else if ratio > 3.0 {
                findings.push(CheckResult {
                    error: ErrorType::LengthMismatch,
                    details: format!("译文过长 (ratio={:.2})", ratio),
                });
            }
        }

        let similarity_should_flag = if self.source_lang == "ja" && self.target_lang == "zh" {
            has_problematic_kana_residue(src, dst)
        } else if self.source_lang == "ko" && self.target_lang == "zh" {
            any_hangeul(dst)
        } else {
            true
        };
        if self.check_similarity(src, dst)
            && similarity_should_flag
            && !findings
                .iter()
                .any(|finding| finding.error == ErrorType::HighSimilarity)
        {
            findings.push(CheckResult {
                error: ErrorType::HighSimilarity,
                details: "原译文高度相似（疑似未翻译）".to_string(),
            });
        }

        findings
    }

    fn check_similarity(&self, src: &str, dst: &str) -> bool {
        if src.contains(dst) || dst.contains(src) {
            return true;
        }
        jaccard_similarity(src, dst) > self.similarity_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_normal() {
        let checker = ResponseChecker::new("ja", "zh", 0.80, 2);
        let srcs = vec!["こんにちは".to_string()];
        let dsts = vec!["你好".to_string()];
        let results = checker.check(&srcs, &dsts, 0);
        assert_eq!(results[0].error, ErrorType::None);
    }

    #[test]
    fn test_check_empty() {
        let checker = ResponseChecker::new("ja", "zh", 0.80, 2);
        let srcs = vec!["テスト".to_string()];
        // All-empty dsts triggers FailParse (response entirely empty)
        let dsts = vec!["".to_string()];
        let results = checker.check(&srcs, &dsts, 0);
        assert_eq!(results[0].error, ErrorType::FailParse);
    }

    #[test]
    fn test_jaccard() {
        assert!(jaccard_similarity("abc", "abc") > 0.99);
        assert!(jaccard_similarity("abc", "xyz") < 0.01);
    }

    #[test]
    fn allows_kana_shape_markers() {
        let checker = ResponseChecker::new("ja", "zh", 0.80, 2);
        let srcs = vec!["くの字に曲がる".to_string()];
        let dsts = vec!["身体弯成く字形".to_string()];
        let results = checker.check(&srcs, &dsts, 0);
        assert_eq!(results[0].error, ErrorType::None);
    }

    #[test]
    fn still_flags_substantial_kana_residue_after_retries() {
        let checker = ResponseChecker::new("ja", "zh", 0.80, 1);
        let srcs = vec!["今日はいい天気ですね".to_string()];
        let dsts = vec!["今天はいい天気ですね".to_string()];
        let results = checker.check(&srcs, &dsts, 1);
        assert_eq!(results[0].error, ErrorType::KanaUntranslatedProse);
    }

    #[test]
    fn flags_unchanged_cjk_text_before_retry_threshold() {
        let checker = ResponseChecker::new("ja", "zh", 0.80, 2);
        let srcs = vec!["第一章".to_string()];
        let dsts = vec!["第一章".to_string()];
        let results = checker.check(&srcs, &dsts, 0);
        assert_eq!(results[0].error, ErrorType::HighSimilarity);
    }

    #[test]
    fn confirmed_term_is_a_hard_check_even_after_retry_threshold() {
        let checker =
            ResponseChecker::new("ja", "zh", 0.80, 0).with_glossary_entries(vec![GlossaryEntry {
                src: "シラジノオト".to_string(),
                dst: "白地野音".to_string(),
                info: "confirmed ruby alias".to_string(),
            }]);
        let srcs = vec!["シラジノオトが笑った。".to_string()];
        let wrong = checker.check(&srcs, &["席拉吉诺笑了。".to_string()], 3);
        assert_eq!(wrong[0].error, ErrorType::TermMissing);
        assert!(wrong[0].details.contains("シラジノオト→白地野音"));

        let correct = checker.check(&srcs, &["白地野音笑了。".to_string()], 3);
        assert_eq!(correct[0].error, ErrorType::None);
    }

    #[test]
    fn term_check_uses_boundaries_and_longest_match() {
        let checker = ResponseChecker::new("ja", "zh", 0.80, 2).with_glossary_entries(vec![
            GlossaryEntry {
                src: "アイ".to_string(),
                dst: "爱".to_string(),
                info: String::new(),
            },
            GlossaryEntry {
                src: "星園".to_string(),
                dst: "星园".to_string(),
                info: String::new(),
            },
            GlossaryEntry {
                src: "星園まなか".to_string(),
                dst: "星园真中".to_string(),
                info: String::new(),
            },
        ]);

        let item = checker.check(
            &["アイテムを取った。".to_string()],
            &["拿到了道具。".to_string()],
            0,
        );
        assert_ne!(item[0].error, ErrorType::TermMissing);

        let full_name = checker.check(
            &["星園まなかが来た。".to_string()],
            &["星园真中来了。".to_string()],
            0,
        );
        assert_eq!(full_name[0].error, ErrorType::None);
    }

    #[test]
    fn quoted_puzzle_tokens_are_protected_but_prose_is_not() {
        let checker = ResponseChecker::new("ja", "zh", 0.80, 2);
        let puzzle = checker.check(
            &["３─２　十二支の３番目「とら」の２文字目＝「ら」".to_string()],
            &["3-2，取十二生肖第3位「とら」的第2个字符，即「ら」。".to_string()],
            0,
        );
        assert_eq!(puzzle[0].error, ErrorType::None);

        let prose = checker.check(
            &["今日はいい天気ですね".to_string()],
            &["今天はいい天気ですね".to_string()],
            0,
        );
        assert_eq!(prose[0].error, ErrorType::KanaUntranslatedProse);
        assert!(prose[0].details.contains("はいい"));
    }

    #[test]
    fn credit_name_kana_is_protected_when_copied_from_source() {
        let checker = ResponseChecker::new("ja", "zh", 0.80, 2);
        let result = checker.check(
            &["口絵・本文イラスト●チェリ子".to_string()],
            &["彩页、正文插图：チェリ子".to_string()],
            0,
        );
        assert_eq!(result[0].error, ErrorType::None);
    }

    #[test]
    fn numbers_and_placeholders_are_hard_checks() {
        let checker = ResponseChecker::new("ja", "zh", 0.80, 0);
        let missing_number = checker.check(
            &["第12章、3人、3人".to_string()],
            &["第十二章，共三人。".to_string()],
            5,
        );
        assert_eq!(missing_number[0].error, ErrorType::NumberMismatch);

        let width_only = checker.check(
            &["第１２章、３人".to_string()],
            &["第12章，共3人。".to_string()],
            5,
        );
        assert_eq!(width_only[0].error, ErrorType::None);

        let missing_placeholder = checker.check(
            &["{player}のHPは%dです".to_string()],
            &["玩家的生命值是%d。".to_string()],
            5,
        );
        assert_eq!(missing_placeholder[0].error, ErrorType::PlaceholderMismatch);

        let placeholders_kept = checker.check(
            &["{player}のHPは%dです".to_string()],
            &["{player}的生命值是%d。".to_string()],
            5,
        );
        assert_eq!(placeholders_kept[0].error, ErrorType::None);
    }

    #[test]
    fn offline_audit_reports_multiple_findings_for_one_unit() {
        let checker =
            ResponseChecker::new("ja", "zh", 0.80, 2).with_glossary_entries(vec![GlossaryEntry {
                src: "シラジノオト".to_string(),
                dst: "白地野音".to_string(),
                info: String::new(),
            }]);
        let findings = checker.audit_line(
            "シラジノオトのHPは{player}に12ある。",
            "席拉吉诺的HP是10。",
            true,
        );
        let errors = findings
            .iter()
            .map(|finding| finding.error)
            .collect::<HashSet<_>>();

        assert!(errors.contains(&ErrorType::TermMissing));
        assert!(errors.contains(&ErrorType::NumberMismatch));
        assert!(errors.contains(&ErrorType::PlaceholderMismatch));
        assert!(!errors.contains(&ErrorType::KanaUntranslatedProse));
    }
}
