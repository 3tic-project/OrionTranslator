use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GlossaryEntry {
    pub src: String,
    pub dst: String,
    pub info: String,
}

/// 从 JSON 文件加载术语表
pub fn load_glossary(path: &std::path::Path) -> Result<Vec<GlossaryEntry>> {
    let content = std::fs::read_to_string(path)?;
    let entries: Vec<GlossaryEntry> = serde_json::from_str(&content)?;
    Ok(entries)
}

/// Load the v1 glossary plus explicitly confirmed ruby aliases from its sibling
/// `*.ruby-candidates.json` sidecar. Pending/rejected/semantic candidates never
/// become prompt constraints.
pub fn load_glossary_with_confirmed_aliases(path: &std::path::Path) -> Result<Vec<GlossaryEntry>> {
    let mut entries =
        load_glossary(path).with_context(|| format!("加载术语表失败: {}", path.display()))?;
    let review_path = bellatrix::ruby_review_path(path);
    if !review_path.exists() {
        return Ok(entries);
    }

    let review = bellatrix::load_ruby_alias_review(&review_path)
        .with_context(|| format!("加载 Ruby 审核状态失败: {}", review_path.display()))?;
    let mut targets_by_surface: HashMap<String, HashSet<String>> = HashMap::new();
    for entry in &entries {
        let src = entry.src.trim();
        let dst = entry.dst.trim();
        if !src.is_empty() && !dst.is_empty() {
            targets_by_surface
                .entry(src.to_string())
                .or_default()
                .insert(dst.to_string());
        }
    }
    if let Some((surface, targets)) = targets_by_surface
        .iter()
        .find(|(_, targets)| targets.len() > 1)
    {
        anyhow::bail!("术语表存在同源多译冲突: {} -> {:?}", surface, targets);
    }

    for candidate in review
        .candidates
        .iter()
        .filter(|candidate| candidate.decision == bellatrix::CandidateDecision::Confirmed)
    {
        if !candidate.classification.can_be_translation_alias() {
            anyhow::bail!(
                "已确认 Ruby 候选 {} 的分类 {:?} 不能作为翻译 alias",
                candidate.candidate_id,
                candidate.classification
            );
        }
        let target = candidate
            .target
            .as_deref()
            .or(candidate.base_dst.as_deref())
            .map(str::trim)
            .filter(|target| !target.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!("已确认 Ruby 候选 {} 缺少目标译名", candidate.candidate_id)
            })?;

        for surface in &candidate.reading_variants {
            let surface = surface.trim();
            if surface.is_empty() {
                continue;
            }
            let known_targets = targets_by_surface.entry(surface.to_string()).or_default();
            if !known_targets.is_empty() && !known_targets.contains(target) {
                anyhow::bail!(
                    "Ruby alias 译名冲突: {} 已映射到 {:?}，候选 {} 要求 {}",
                    surface,
                    known_targets,
                    candidate.candidate_id,
                    target
                );
            }
            if known_targets.insert(target.to_string()) {
                entries.push(GlossaryEntry {
                    src: surface.to_string(),
                    dst: target.to_string(),
                    info: format!(
                        "ruby_alias:{}; classification={:?}; revision={}",
                        candidate.base, candidate.classification, candidate.revision
                    ),
                });
            }
        }
    }
    Ok(entries)
}

/// 将术语表格式化为通用模型 prompt 中的文本格式
/// 输出格式：
/// src -> dst   #info
///
/// 通用模型只注入已有目标译名的条目。
pub fn format_glossary(entries: &[GlossaryEntry]) -> String {
    entries
        .iter()
        .filter_map(|e| {
            let src = e.src.trim();
            let dst = e.dst.trim();
            if src.is_empty() || dst.is_empty() {
                return None;
            }
            Some(format!("{} -> {}   #{}", src, dst, e.info.trim()))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// 按当前待译文本过滤命中条目。
///
/// - `require_dst=true`：通用模型，仅保留有目标译名的条目
/// - `require_dst=false`：Orion 可用空 dst 作为人物/实体候选提示
pub fn filter_glossary_for_texts_ex(
    entries: &[GlossaryEntry],
    texts: &[String],
    require_dst: bool,
) -> Vec<GlossaryEntry> {
    if entries.is_empty() || texts.is_empty() {
        return Vec::new();
    }

    let haystack = texts.join("\n").to_lowercase();
    let mut candidates = Vec::new();
    for (entry_index, entry) in entries.iter().enumerate() {
        let src = entry.src.trim();
        let dst = entry.dst.trim();
        if src.is_empty() || (require_dst && dst.is_empty()) {
            continue;
        }
        let normalized_src = src.to_lowercase();
        for (start, end) in find_term_spans(&haystack, &normalized_src) {
            candidates.push((start, end, entry_index, normalized_src.clone()));
        }
    }

    // 同一位置优先最长术语；完全相同的 src 保留全部条目，以便后续暴露译名冲突。
    candidates.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| (right.1 - right.0).cmp(&(left.1 - left.0)))
            .then(left.2.cmp(&right.2))
    });

    let mut accepted: Vec<(usize, usize, String)> = Vec::new();
    let mut matched = vec![false; entries.len()];
    for (start, end, entry_index, normalized_src) in candidates {
        let overlapping = accepted.iter().find(|(accepted_start, accepted_end, _)| {
            start < *accepted_end && *accepted_start < end
        });
        match overlapping {
            None => {
                accepted.push((start, end, normalized_src));
                matched[entry_index] = true;
            }
            Some((accepted_start, accepted_end, accepted_src))
                if *accepted_start == start
                    && *accepted_end == end
                    && *accepted_src == normalized_src =>
            {
                matched[entry_index] = true;
            }
            Some(_) => {}
        }
    }

    entries
        .iter()
        .enumerate()
        .filter(|(index, e)| {
            let src = e.src.trim();
            let dst = e.dst.trim();
            matched[*index] && !src.is_empty() && (!require_dst || !dst.is_empty())
        })
        .map(|(_, entry)| entry)
        .cloned()
        .collect()
}

fn find_term_spans(haystack: &str, needle: &str) -> Vec<(usize, usize)> {
    if needle.is_empty() {
        return Vec::new();
    }
    let shape = term_shape(needle);
    haystack
        .match_indices(needle)
        .filter_map(|(start, _)| {
            let end = start + needle.len();
            let previous = haystack[..start].chars().next_back();
            let next = haystack[end..].chars().next();
            if has_valid_boundaries(shape, previous, next) {
                Some((start, end))
            } else {
                None
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
enum TermShape {
    Hiragana,
    Katakana,
    AsciiWord,
    Other,
}

fn term_shape(term: &str) -> TermShape {
    let chars: Vec<char> = term.chars().collect();
    if !chars.is_empty()
        && chars
            .iter()
            .all(|character| is_hiragana_term_char(*character))
    {
        TermShape::Hiragana
    } else if !chars.is_empty()
        && chars
            .iter()
            .all(|character| is_katakana_term_char(*character))
    {
        TermShape::Katakana
    } else if !chars.is_empty()
        && chars
            .iter()
            .all(|character| character.is_ascii_alphanumeric() || *character == '_')
    {
        TermShape::AsciiWord
    } else {
        TermShape::Other
    }
}

fn has_valid_boundaries(shape: TermShape, previous: Option<char>, next: Option<char>) -> bool {
    match shape {
        TermShape::Hiragana => {
            !previous.is_some_and(is_hiragana_term_char) && !next.is_some_and(is_hiragana_term_char)
        }
        TermShape::Katakana => {
            !previous.is_some_and(is_katakana_term_char) && !next.is_some_and(is_katakana_term_char)
        }
        TermShape::AsciiWord => {
            let is_word = |character: char| character.is_ascii_alphanumeric() || character == '_';
            !previous.is_some_and(is_word) && !next.is_some_and(is_word)
        }
        TermShape::Other => true,
    }
}

fn is_hiragana_term_char(character: char) -> bool {
    matches!(character as u32, 0x3040..=0x309f) || character == 'ー'
}

fn is_katakana_term_char(character: char) -> bool {
    matches!(character as u32, 0x30a0..=0x30ff | 0xff66..=0xff9f)
        || matches!(character, 'ー' | '・')
}

pub fn filter_glossary_for_texts(
    entries: &[GlossaryEntry],
    texts: &[String],
) -> Vec<GlossaryEntry> {
    filter_glossary_for_texts_ex(entries, texts, true)
}

pub fn format_matched_glossary(entries: &[GlossaryEntry], texts: &[String]) -> String {
    let matched = filter_glossary_for_texts(entries, texts);
    format_grouped_constraints(&matched)
        .map(|constraints| constraints.trim_end().to_string())
        .unwrap_or_default()
}

fn format_grouped_constraints(entries: &[GlossaryEntry]) -> Option<String> {
    let mut by_target: BTreeMap<&str, (Vec<&str>, Vec<&str>)> = BTreeMap::new();
    let mut entities = Vec::new();
    for entry in entries {
        let src = entry.src.trim();
        let dst = entry.dst.trim();
        if src.is_empty() {
            continue;
        }
        if dst.is_empty() {
            entities.push(src);
        } else {
            let (surfaces, notes) = by_target.entry(dst).or_default();
            surfaces.push(src);
            let note = entry.info.trim();
            if !note.is_empty() {
                notes.push(note);
            }
        }
    }
    if by_target.is_empty() && entities.is_empty() {
        return None;
    }

    let mut result = String::new();
    for (target, (mut surfaces, mut notes)) in by_target {
        surfaces.sort_unstable();
        surfaces.dedup();
        notes.sort_unstable();
        notes.dedup();
        result.push_str(&surfaces.join(" / "));
        result.push_str(" -> ");
        result.push_str(target);
        if !notes.is_empty() {
            result.push_str("   #");
            result.push_str(&notes.join("; "));
        }
        result.push('\n');
    }
    if !entities.is_empty() {
        entities.sort_unstable();
        entities.dedup();
        result.push_str("Entity candidates:\n");
        for entity in entities {
            result.push_str(entity);
            result.push('\n');
        }
    }
    Some(result)
}

/// 将术语表格式化为 Orion 模型 prompt。
///
/// 输出格式：
/// ```text
/// 术语表：
/// src→dst          # 已确认译名
/// …
/// 人物候选：
/// src              # 仅有源名（NER/Orion 跳过 LLM 译名时）
/// ```
///
/// 返回 None 表示没有任何可注入信息。
pub fn format_glossary_for_orion(entries: &[GlossaryEntry]) -> Option<String> {
    let mut pairs = Vec::new();
    let mut entities = Vec::new();
    for entry in entries {
        let src = entry.src.trim();
        let dst = entry.dst.trim();
        if src.is_empty() {
            continue;
        }
        if dst.is_empty() {
            entities.push(src);
        } else {
            pairs.push((src, dst));
        }
    }
    if pairs.is_empty() && entities.is_empty() {
        return None;
    }
    pairs.sort_unstable();
    pairs.dedup();
    entities.sort_unstable();
    entities.dedup();

    // Preserve Orion SFT's one-surface-per-line shape. Confirmed aliases share
    // a target but remain separate mappings rather than introducing new syntax.
    let mut result = String::from("术语表：\n");
    for (src, dst) in pairs {
        result.push_str(src);
        result.push('→');
        result.push_str(dst);
        result.push('\n');
    }
    if !entities.is_empty() {
        result.push_str("人物候选：\n");
        for entity in entities {
            result.push_str(entity);
            result.push('\n');
        }
    }
    Some(result)
}

pub fn format_matched_glossary_for_orion(
    entries: &[GlossaryEntry],
    texts: &[String],
) -> Option<String> {
    format_glossary_for_orion(&filter_glossary_for_texts_ex(entries, texts, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_glossary() {
        let entries = vec![
            GlossaryEntry {
                src: "由紀".to_string(),
                dst: "由纪".to_string(),
                info: "女性".to_string(),
            },
            GlossaryEntry {
                src: "セナ".to_string(),
                dst: "濑名".to_string(),
                info: "女性".to_string(),
            },
        ];
        let text = format_glossary(&entries);
        assert_eq!(text, "由紀 -> 由纪   #女性\nセナ -> 濑名   #女性");
    }

    #[test]
    fn test_format_glossary_empty() {
        let entries: Vec<GlossaryEntry> = vec![];
        let text = format_glossary(&entries);
        assert_eq!(text, "");
    }

    #[test]
    fn test_format_glossary_trims_and_skips_blank_dst() {
        let entries = vec![
            GlossaryEntry {
                src: "  由紀  ".to_string(),
                dst: "  由纪  ".to_string(),
                info: "  女性  ".to_string(),
            },
            GlossaryEntry {
                src: "テスト".to_string(),
                dst: "  ".to_string(),
                info: String::new(),
            },
        ];
        let text = format_glossary(&entries);
        assert_eq!(text, "由紀 -> 由纪   #女性");
    }

    #[test]
    fn test_filter_glossary_for_texts_matches_current_text_only() {
        let entries = vec![
            GlossaryEntry {
                src: "ネギ".to_string(),
                dst: "涅吉".to_string(),
                info: "男".to_string(),
            },
            GlossaryEntry {
                src: "茶々丸".to_string(),
                dst: "茶茶丸".to_string(),
                info: "女".to_string(),
            },
            GlossaryEntry {
                src: "なのは".to_string(),
                dst: "奈叶".to_string(),
                info: "女".to_string(),
            },
            GlossaryEntry {
                src: "空白".to_string(),
                dst: " ".to_string(),
                info: String::new(),
            },
        ];
        let texts = vec![
            "なぜネギとニンニク？".to_string(),
            "「茶々丸か？」".to_string(),
        ];

        let matched = filter_glossary_for_texts(&entries, &texts);
        let sources: Vec<&str> = matched.iter().map(|entry| entry.src.as_str()).collect();

        assert_eq!(sources, vec!["ネギ", "茶々丸"]);
    }

    #[test]
    fn short_katakana_requires_a_surface_boundary() {
        let entries = vec![GlossaryEntry {
            src: "アイ".to_string(),
            dst: "爱".to_string(),
            info: String::new(),
        }];
        let false_positive_texts = vec!["アイテムとアイディアを見て、アイツは帰った。".to_string()];
        assert!(filter_glossary_for_texts(&entries, &false_positive_texts).is_empty());

        let independent_mention = vec!["アイが示す答え。".to_string()];
        assert_eq!(
            filter_glossary_for_texts(&entries, &independent_mention).len(),
            1
        );
    }

    #[test]
    fn longest_overlapping_surface_wins_unless_short_form_also_appears_alone() {
        let entries = vec![
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
        ];

        let only_full_name = vec!["星園まなかが来た。".to_string()];
        let matched = filter_glossary_for_texts(&entries, &only_full_name);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].src, "星園まなか");

        let both = vec!["星園まなかが来た。星園も呼ばれた。".to_string()];
        let matched = filter_glossary_for_texts(&entries, &both);
        assert_eq!(matched.len(), 2);
    }

    #[test]
    fn ascii_terms_use_word_boundaries_and_remain_case_insensitive() {
        let entries = vec![GlossaryEntry {
            src: "saber".to_string(),
            dst: "Saber".to_string(),
            info: String::new(),
        }];
        assert!(filter_glossary_for_texts(&entries, &["lightsaber".to_string()]).is_empty());
        assert_eq!(
            filter_glossary_for_texts(&entries, &["SABER arrived".to_string()]).len(),
            1
        );
    }

    #[test]
    fn test_format_matched_glossary_is_case_insensitive() {
        let entries = vec![GlossaryEntry {
            src: "saber".to_string(),
            dst: "Saber".to_string(),
            info: String::new(),
        }];
        let texts = vec!["SABER が現れた。".to_string()];

        let text = format_matched_glossary(&entries, &texts);

        assert_eq!(text, "saber -> Saber");
    }

    #[test]
    fn test_format_glossary_for_orion() {
        let entries = vec![
            GlossaryEntry {
                src: "グレン".to_string(),
                dst: "格伦".to_string(),
                info: "男性".to_string(),
            },
            GlossaryEntry {
                src: "ネメア".to_string(),
                dst: "涅米亚".to_string(),
                info: "地名".to_string(),
            },
        ];
        let result = format_glossary_for_orion(&entries).unwrap();
        assert!(result.starts_with("术语表：\n"));
        assert!(result.contains("グレン→格伦\n"));
        assert!(result.contains("ネメア→涅米亚\n"));
    }

    #[test]
    fn test_format_glossary_for_orion_empty_dst_as_entity_hint() {
        // Orion 跳过 LLM 译名时 dst 为空，仍应作为人物候选注入
        let entries = vec![GlossaryEntry {
            src: "テスト".to_string(),
            dst: String::new(),
            info: String::new(),
        }];
        let result = format_glossary_for_orion(&entries).unwrap();
        assert!(result.contains("人物候选：\n"));
        assert!(result.contains("テスト\n"));
        assert!(!result.contains('→'));
    }

    #[test]
    fn test_format_glossary_for_orion_whitespace_dst_as_entity_hint() {
        let entries = vec![GlossaryEntry {
            src: "テスト".to_string(),
            dst: "  ".to_string(),
            info: String::new(),
        }];
        let result = format_glossary_for_orion(&entries).unwrap();
        assert!(result.contains("人物候选：\nテスト\n"));
    }

    #[test]
    fn test_format_glossary_for_orion_mixed_pairs_and_entities() {
        let entries = vec![
            GlossaryEntry {
                src: "ネギ".to_string(),
                dst: "涅吉".to_string(),
                info: String::new(),
            },
            GlossaryEntry {
                src: "茶々丸".to_string(),
                dst: String::new(),
                info: String::new(),
            },
        ];
        let result = format_glossary_for_orion(&entries).unwrap();
        assert!(result.contains("ネギ→涅吉\n"));
        assert!(result.contains("人物候选：\n茶々丸\n"));
    }

    #[test]
    fn test_filter_allows_empty_dst_for_orion_match() {
        let entries = vec![GlossaryEntry {
            src: "ネギ".to_string(),
            dst: String::new(),
            info: String::new(),
        }];
        let texts = vec!["ネギが走った。".to_string()];
        let matched = filter_glossary_for_texts_ex(&entries, &texts, false);
        assert_eq!(matched.len(), 1);
        let for_common = filter_glossary_for_texts(&entries, &texts);
        assert!(for_common.is_empty());
    }

    #[test]
    fn test_format_glossary_for_orion_sorted() {
        let entries = vec![
            GlossaryEntry {
                src: "ネメア".to_string(),
                dst: "涅米亚".to_string(),
                info: String::new(),
            },
            GlossaryEntry {
                src: "グレン".to_string(),
                dst: "格伦".to_string(),
                info: String::new(),
            },
        ];
        let result = format_glossary_for_orion(&entries).unwrap();
        let lines: Vec<&str> = result.lines().collect();
        // Should be sorted: グレン before ネメア
        assert_eq!(lines[0], "术语表：");
        assert_eq!(lines[1], "グレン→格伦");
        assert_eq!(lines[2], "ネメア→涅米亚");
    }

    #[test]
    fn test_format_matched_glossary_for_orion_filters_unmatched() {
        let entries = vec![
            GlossaryEntry {
                src: "ネギ".to_string(),
                dst: "涅吉".to_string(),
                info: String::new(),
            },
            GlossaryEntry {
                src: "なのは".to_string(),
                dst: "奈叶".to_string(),
                info: String::new(),
            },
        ];
        let texts = vec!["「ネギとニンニクは……」".to_string()];

        let result = format_matched_glossary_for_orion(&entries, &texts).unwrap();

        assert!(result.contains("ネギ→涅吉\n"));
        assert!(!result.contains("なのは"));
    }

    fn temp_glossary_path(label: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "alnilam-glossary-{label}-{}-{nonce}.json",
            std::process::id()
        ))
    }

    fn ruby(base: &str, reading: &str) -> betelgeuse::RubyAnnotation {
        betelgeuse::RubyAnnotation {
            base: base.to_string(),
            reading: reading.to_string(),
            source_path: "Text/ch1.xhtml".to_string(),
            context: format!("{base}が来た"),
            combined: false,
        }
    }

    #[test]
    fn confirmed_ruby_aliases_join_translation_constraints() {
        let glossary_path = temp_glossary_path("confirmed-alias");
        let review_path = bellatrix::ruby_review_path(&glossary_path);
        let base_entries = vec![GlossaryEntry {
            src: "白地野音".to_string(),
            dst: "白地野音".to_string(),
            info: "人物".to_string(),
        }];
        std::fs::write(
            &glossary_path,
            serde_json::to_string_pretty(&base_entries).unwrap(),
        )
        .unwrap();
        let translations = [bellatrix::llm::TranslationEntry {
            src: "白地野音".to_string(),
            dst: "白地野音".to_string(),
            info: "人物".to_string(),
        }];
        let document = bellatrix::refresh_ruby_alias_review(
            &review_path,
            &[ruby("白地野音", "しらじのおと")],
            &["シラジノオトが来た。".to_string()],
            &translations,
        )
        .unwrap();
        bellatrix::update_ruby_alias_decision(
            &review_path,
            &document.candidates[0].candidate_id,
            bellatrix::RubyAliasClassification::PhoneticReading,
            bellatrix::CandidateDecision::Confirmed,
            None,
            None,
        )
        .unwrap();

        let loaded = load_glossary_with_confirmed_aliases(&glossary_path).unwrap();
        assert!(loaded
            .iter()
            .any(|entry| entry.src == "しらじのおと" && entry.dst == "白地野音"));
        assert!(loaded
            .iter()
            .any(|entry| entry.src == "シラジノオト" && entry.dst == "白地野音"));
        let prompt =
            format_matched_glossary_for_orion(&loaded, &["シラジノオトが来た。".to_string()])
                .unwrap();
        assert!(prompt.contains("シラジノオト→白地野音"));

        let _ = std::fs::remove_file(glossary_path);
        let _ = std::fs::remove_file(review_path);
    }

    #[test]
    fn pending_ruby_candidate_does_not_join_glossary() {
        let glossary_path = temp_glossary_path("pending-alias");
        let review_path = bellatrix::ruby_review_path(&glossary_path);
        let base_entries = vec![GlossaryEntry {
            src: "桃谷".to_string(),
            dst: "桃谷".to_string(),
            info: String::new(),
        }];
        std::fs::write(
            &glossary_path,
            serde_json::to_string_pretty(&base_entries).unwrap(),
        )
        .unwrap();
        let translations = [bellatrix::llm::TranslationEntry {
            src: "桃谷".to_string(),
            dst: "桃谷".to_string(),
            info: String::new(),
        }];
        bellatrix::refresh_ruby_alias_review(
            &review_path,
            &[ruby("桃谷", "ライバル")],
            &["ライバルが来た。".to_string()],
            &translations,
        )
        .unwrap();

        let loaded = load_glossary_with_confirmed_aliases(&glossary_path).unwrap();
        assert_eq!(loaded, base_entries);

        let _ = std::fs::remove_file(glossary_path);
        let _ = std::fs::remove_file(review_path);
    }
}
