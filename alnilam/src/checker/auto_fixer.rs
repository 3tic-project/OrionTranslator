use std::collections::HashSet;

// ── Text helpers (shared with response_checker) ──────────────────────────

fn is_hiragana(c: char) -> bool {
    ('\u{3040}'..='\u{309F}').contains(&c)
}

fn is_katakana(c: char) -> bool {
    ('\u{30A0}'..='\u{30FF}').contains(&c)
}

fn is_kana(c: char) -> bool {
    is_hiragana(c) || is_katakana(c)
}

/// Onomatopoeia kana that can appear isolated
const ONOMATOPOEIA_KANA: &[char] = &[
    'ッ', 'っ', 'ぁ', 'ぃ', 'ぅ', 'ぇ', 'ぉ', 'ゃ', 'ゅ', 'ょ', 'ゎ',
];

/// Punctuation mapping: Japanese punctuation -> alternatives that LLMs may produce
const PUNCTUATION_MAP: &[(&str, &[&str])] = &[
    ("「", &["\u{201C}", "\u{2018}"]),  // "" '
    ("」", &["\u{201D}", "\u{2019}"]),  // "" '
    ("『", &["\u{201C}", "\u{2018}"]),
    ("』", &["\u{201D}", "\u{2019}"]),
    ("（", &["("]),
    ("）", &[")"]),
    ("【", &["["]),
    ("】", &["]"]),
    ("？", &["?"]),
    ("！", &["!"]),
    ("：", &[":"]),
    ("\u{3000}", &[" "]),
];

// ── AutoFixer ────────────────────────────────────────────────────────────

pub struct AutoFixer {
    source_lang: String,
    #[allow(dead_code)]
    target_lang: String,
    fix_kana: bool,
    fix_punctuation: bool,
    fix_quotes: bool,
}

impl AutoFixer {
    pub fn new(source_lang: &str, target_lang: &str) -> Self {
        Self {
            source_lang: source_lang.to_lowercase(),
            target_lang: target_lang.to_lowercase(),
            fix_kana: true,
            fix_punctuation: true,
            fix_quotes: true,
        }
    }

    pub fn fix(&self, src: &str, dst: &str) -> String {
        if dst.is_empty() {
            return dst.to_string();
        }

        let mut result = dst.to_string();

        // 1. Fix isolated kana (Japanese source)
        if self.fix_kana && self.source_lang == "ja" {
            result = self.fix_isolated_kana(&result);
        }

        // 2. Fix punctuation
        if self.fix_punctuation {
            result = self.fix_punctuation_fn(src, &result);
        }

        // 3. Fix quotes
        if self.fix_quotes {
            result = self.fix_quotes_fn(src, &result);
        }

        result
    }

    fn fix_isolated_kana(&self, dst: &str) -> String {
        let chars: Vec<char> = dst.chars().collect();
        let length = chars.len();
        let onomatopoeia_set: HashSet<char> = ONOMATOPOEIA_KANA.iter().copied().collect();
        let mut result = Vec::with_capacity(length);

        for (i, &ch) in chars.iter().enumerate() {
            if onomatopoeia_set.contains(&ch) {
                let prev_is_kana = if i > 0 { is_kana(chars[i - 1]) } else { false };
                let next_is_kana = if i + 1 < length {
                    is_kana(chars[i + 1])
                } else {
                    false
                };

                if !prev_is_kana && !next_is_kana {
                    continue; // Remove isolated onomatopoeia kana
                }
            }
            result.push(ch);
        }

        result.into_iter().collect()
    }

    fn fix_punctuation_fn(&self, src: &str, dst: &str) -> String {
        let mut result = dst.to_string();

        for &(target, alternatives) in PUNCTUATION_MAP {
            let src_count = src.matches(target).count();
            let dst_count = result.matches(target).count();
            let alt_count: usize = alternatives.iter().map(|a| result.matches(a).count()).sum();

            if src_count > 0 && src_count > dst_count && src_count == dst_count + alt_count {
                for alt in alternatives {
                    result = result.replace(alt, target);
                }
            }
        }

        result
    }

    fn fix_quotes_fn(&self, src: &str, dst: &str) -> String {
        let mut result = dst.to_string();

        let open_quotes = ['"', '\'', '「', '『', '\u{201C}', '\u{2018}'];
        let close_quotes = ['"', '\'', '」', '』', '\u{201D}', '\u{2019}'];

        let quote_pairs: &[(&str, &str)] = &[
            ("「", "」"),
            ("『", "』"),
            ("\u{201C}", "\u{201D}"),
            ("\u{2018}", "\u{2019}"),
        ];

        for &(open_q, close_q) in quote_pairs {
            // Fix opening quote
            if src.starts_with(open_q) {
                if let Some(first_char) = result.chars().next() {
                    if open_quotes.contains(&first_char) {
                        let first_str: String = std::iter::once(first_char).collect();
                        if first_str != open_q {
                            result = format!("{}{}", open_q, &result[first_char.len_utf8()..]);
                        }
                    }
                }
            }

            // Fix closing quote
            if src.ends_with(close_q) {
                if let Some(last_char) = result.chars().last() {
                    if close_quotes.contains(&last_char) {
                        let last_str: String = std::iter::once(last_char).collect();
                        if last_str != close_q {
                            let byte_len = result.len() - last_char.len_utf8();
                            result = format!("{}{}", &result[..byte_len], close_q);
                        }
                    }
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fix_quotes() {
        let fixer = AutoFixer::new("ja", "zh");
        let fixed = fixer.fix("「こんにちは」", "\u{201C}你好\u{201D}");
        assert_eq!(fixed, "「你好」");
    }

    #[test]
    fn test_fix_isolated_kana() {
        let fixer = AutoFixer::new("ja", "zh");
        let fixed = fixer.fix("テスト", "测试っ结果");
        assert_eq!(fixed, "测试结果");
    }
}
