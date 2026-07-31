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

/// Punctuation mapping: Japanese punctuation -> alternatives that LLMs may produce.
/// 引号（「」『』）不在此表：嵌套时计数冲突会导致整段跳过，改由 `fix_quotes_fn` 专门处理。
const PUNCTUATION_MAP: &[(&str, &[&str])] = &[
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

    /// 将译文中的中文/西文引号统一为日文引号，覆盖段中嵌套与首尾。
    ///
    /// 策略：译文一律不使用中文引号（与原文是否含「」『』无关）。
    /// - 大引号「」 ← 中文双引号 “ ”、ASCII `"`、全角 ＂
    /// - 小引号『』 ← 中文单引号 ‘ ’、全角 ＇
    /// - ASCII `'`：仅在原文含『』时改写，避免误伤英文撇号
    fn fix_quotes_fn(&self, src: &str, dst: &str) -> String {
        let src_has_nijuu = src.contains('『') || src.contains('』');

        let mut out = String::with_capacity(dst.len());
        let mut ascii_double_open = true;
        let mut ascii_single_open = true;
        let mut fullwidth_double_open = true;
        let mut fullwidth_single_open = true;

        for ch in dst.chars() {
            let mapped = match ch {
                // 中文弯双引号 → 「」
                '\u{201C}' => Some('「'),
                '\u{201D}' => Some('」'),
                // 中文弯单引号 → 『』（嵌套层级）
                '\u{2018}' => Some('『'),
                '\u{2019}' => Some('』'),
                // ASCII 双引号：按开合交替映射
                '"' => {
                    let q = if ascii_double_open { '「' } else { '」' };
                    ascii_double_open = !ascii_double_open;
                    Some(q)
                }
                // ASCII 单引号：仅在原文有『』时改写
                '\'' if src_has_nijuu => {
                    let q = if ascii_single_open { '『' } else { '』' };
                    ascii_single_open = !ascii_single_open;
                    Some(q)
                }
                // 全角引号
                '＂' => {
                    let q = if fullwidth_double_open { '「' } else { '」' };
                    fullwidth_double_open = !fullwidth_double_open;
                    Some(q)
                }
                '＇' => {
                    let q = if fullwidth_single_open { '『' } else { '』' };
                    fullwidth_single_open = !fullwidth_single_open;
                    Some(q)
                }
                _ => None,
            };

            out.push(mapped.unwrap_or(ch));
        }

        out
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
    fn test_fix_nested_quotes_mid_sentence() {
        let fixer = AutoFixer::new("ja", "zh");
        let src = "「謎が解けたらトイレに入れるよ、おにーちゃん！『お漏らしの危機からの脱出』だね！」";
        let dst = "\u{201C}哥哥！谜题解开了就能进厕所哦！——\u{2018}从漏尿危机中逃脱\u{2019}！\u{201D}";
        let fixed = fixer.fix(src, dst);
        assert_eq!(
            fixed,
            "「哥哥！谜题解开了就能进厕所哦！——『从漏尿危机中逃脱』！」"
        );
    }

    #[test]
    fn test_fix_quotes_inside_narrative_paragraph() {
        let fixer = AutoFixer::new("ja", "zh");
        let src = "なんのことはない。ミリハが「謎が解けたら入れるよ！『脱出』だね！」などと紙を押し付けてきた。";
        let dst = "说来也没什么大不了。米莉哈把纸塞过来，说什么\u{201C}哥哥！解开就能进！——\u{2018}逃脱\u{2019}！\u{201D}之类的。";
        let fixed = fixer.fix(src, dst);
        assert!(fixed.contains("「哥哥！解开就能进！——『逃脱』！」"));
        assert!(!fixed.contains('\u{201C}'));
        assert!(!fixed.contains('\u{2018}'));
    }

    #[test]
    fn test_fix_ascii_double_quotes_when_src_has_kagi() {
        let fixer = AutoFixer::new("ja", "zh");
        let fixed = fixer.fix("「こんにちは」", "\"你好\"");
        assert_eq!(fixed, "「你好」");
    }

    #[test]
    fn test_always_rewrite_chinese_quotes_even_without_src_quotes() {
        let fixer = AutoFixer::new("ja", "zh");
        let dst = "他说\u{201C}你好\u{201D}，又道\u{2018}再见\u{2019}。";
        let fixed = fixer.fix("彼は挨拶した。", dst);
        assert_eq!(fixed, "他说「你好」，又道『再见』。");
    }

    #[test]
    fn test_fix_isolated_kana() {
        let fixer = AutoFixer::new("ja", "zh");
        let fixed = fixer.fix("テスト", "测试っ结果");
        assert_eq!(fixed, "测试结果");
    }
}
