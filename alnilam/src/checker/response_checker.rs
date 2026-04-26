use std::collections::HashSet;

use regex::Regex;

use super::types::{CheckResult, ErrorType};

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

fn has_problematic_kana_residue(text: &str) -> bool {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut i = 0usize;

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

        if !is_allowed_shape_kana(run, prev, next) {
            return true;
        }
        i = end;
    }

    false
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

fn re_chapter_number() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"第\d+[話话章節节回]").expect("chapter number regex"))
}

fn re_digits() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\d+").expect("digits regex"))
}

// ── ResponseChecker ──────────────────────────────────────────────────────

pub struct ResponseChecker {
    source_lang: String,
    target_lang: String,
    similarity_threshold: f64,
    retry_threshold: usize,
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
        }
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

        // Kana residue (ja -> other), with small shape-marker tolerance.
        if self.source_lang == "ja" && self.target_lang != "ja" && has_problematic_kana_residue(dst)
        {
            return CheckResult {
                error: ErrorType::KanaResidue,
                details: "译文中残留假名".to_string(),
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

        // Number consistency check
        let src_normalized = fullwidth_to_halfwidth(src);
        let dst_normalized = fullwidth_to_halfwidth(dst);
        let src_numbers: HashSet<String> = re_digits()
            .find_iter(&src_normalized)
            .map(|m| m.as_str().to_string())
            .collect();
        let dst_numbers: HashSet<String> = re_digits()
            .find_iter(&dst_normalized)
            .map(|m| m.as_str().to_string())
            .collect();

        if !src_numbers.is_empty() && src_numbers.len() <= 3 {
            if src_numbers.intersection(&dst_numbers).count() == 0 {
                if re_chapter_number().is_match(&src_normalized) {
                    return CheckResult {
                        error: ErrorType::LengthMismatch,
                        details: format!(
                            "章节号不匹配: src含{:?}, dst含{:?}",
                            src_numbers, dst_numbers
                        ),
                    };
                }
            }
        }

        // Similarity check
        if self.check_similarity(src, dst) {
            let should_flag = if self.source_lang == "ja" && self.target_lang == "zh" {
                has_problematic_kana_residue(dst)
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
        assert_eq!(results[0].error, ErrorType::KanaResidue);
    }

    #[test]
    fn flags_unchanged_cjk_text_before_retry_threshold() {
        let checker = ResponseChecker::new("ja", "zh", 0.80, 2);
        let srcs = vec!["第一章".to_string()];
        let dsts = vec!["第一章".to_string()];
        let results = checker.check(&srcs, &dsts, 0);
        assert_eq!(results[0].error, ErrorType::HighSimilarity);
    }
}
