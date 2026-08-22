//! Character / entity aggregation in the style of OrionTranslator `bellatrix::detector`.
//!
//! - Filter by label (default PER) and confidence
//! - Group by surface form
//! - Keep source line + surrounding context for each mention
//! - Drop names below `min_count`

use crate::ner::NerResult;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

const DEFAULT_PERSON_TYPES: &[&str] = &["PER", "PERSON"];
const PUNCTUATION_CHARS: &str = "。！？、，．「」『』（）【】〈〉《》・～…\"'“”‘’　 \t\r\n";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mention {
    /// 1-based line number in the original document.
    pub line: usize,
    pub line_text: String,
    /// Char offsets within the line (half-open).
    pub start: usize,
    pub end: usize,
    pub above: Vec<String>,
    pub follow: Vec<String>,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterInfo {
    pub name: String,
    pub label: String,
    pub count: usize,
    pub max_score: f32,
    pub mean_score: f32,
    pub first_line: usize,
    /// Up to `max_source_lines` sample line numbers.
    pub lines: Vec<usize>,
    pub mentions: Vec<Mention>,
}

#[derive(Debug, Clone)]
pub struct AggregateConfig {
    pub min_score: f32,
    pub min_count: usize,
    /// Keep only these BIOES types (e.g. PER). Empty = all.
    pub labels: Vec<String>,
    pub context_size: usize,
    pub max_mentions_per_name: usize,
    pub max_source_lines: usize,
    /// Drop pure punctuation / whitespace surfaces.
    pub drop_punct_only: bool,
    /// Drop single-char names unless count is high.
    pub min_count_single_char: usize,
}

impl Default for AggregateConfig {
    fn default() -> Self {
        Self {
            min_score: 0.90,
            min_count: 2,
            labels: DEFAULT_PERSON_TYPES
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            context_size: 5,
            max_mentions_per_name: 50,
            max_source_lines: 20,
            drop_punct_only: true,
            min_count_single_char: 5,
        }
    }
}

fn should_skip_line(line: &str) -> bool {
    let cleaned: String = line.replace('\u{3000}', "").trim().to_string();
    if cleaned.chars().count() < 2 {
        return true;
    }
    cleaned.chars().all(|c| PUNCTUATION_CHARS.contains(c))
}

fn is_punct_only(s: &str) -> bool {
    s.chars()
        .all(|c| PUNCTUATION_CHARS.contains(c) || c.is_ascii_punctuation())
}

fn label_ok(label: &str, allowed: &[String]) -> bool {
    if allowed.is_empty() {
        return true;
    }
    allowed
        .iter()
        .any(|a| label.eq_ignore_ascii_case(a) || label.contains(a.as_str()))
}

fn context_lines(lines: &[String], line_index: usize, ctx: usize) -> (Vec<String>, Vec<String>) {
    let total = lines.len();
    let above_start = line_index.saturating_sub(ctx);
    let mut above: Vec<String> = lines[above_start..line_index]
        .iter()
        .map(|l| l.trim().to_string())
        .collect();
    while above.len() < ctx {
        above.insert(0, String::new());
    }

    let follow_end = (line_index + ctx + 1).min(total);
    let mut follow: Vec<String> = if line_index + 1 < total {
        lines[line_index + 1..follow_end]
            .iter()
            .map(|l| l.trim().to_string())
            .collect()
    } else {
        Vec::new()
    };
    while follow.len() < ctx {
        follow.push(String::new());
    }
    (above, follow)
}

/// Aggregate entities from per-line NER results into character dossiers.
///
/// `line_results[i]` must correspond to `all_lines[i]` (same index, including empties
/// if you keep them). Prefer passing only non-empty lines with matching indices via
/// `line_indices` (original 0-based indices into the full document).
pub fn aggregate_characters(
    all_lines: &[String],
    line_indices: &[usize],
    line_results: &[NerResult],
    cfg: &AggregateConfig,
) -> Vec<CharacterInfo> {
    assert_eq!(line_indices.len(), line_results.len());

    let mut buckets: HashMap<(String, String), Vec<Mention>> = HashMap::new();

    for (res, &orig_idx) in line_results.iter().zip(line_indices.iter()) {
        if should_skip_line(&res.text) {
            continue;
        }
        let (above, follow) = context_lines(all_lines, orig_idx, cfg.context_size);
        for ent in &res.entities {
            if !label_ok(&ent.label, &cfg.labels) {
                continue;
            }
            if ent.score < cfg.min_score {
                continue;
            }
            if cfg.drop_punct_only && is_punct_only(&ent.text) {
                continue;
            }
            if ent.text.trim().is_empty() {
                continue;
            }

            let mention = Mention {
                line: orig_idx + 1,
                line_text: res.text.clone(),
                start: ent.start,
                end: ent.end,
                above: above.clone(),
                follow: follow.clone(),
                score: ent.score,
            };
            buckets
                .entry((ent.label.clone(), ent.text.clone()))
                .or_default()
                .push(mention);
        }
    }

    let mut out = Vec::new();
    for ((label, name), mut mentions) in buckets {
        let count = mentions.len();
        let single = name.chars().count() == 1;
        let need = if single {
            cfg.min_count_single_char.max(cfg.min_count)
        } else {
            cfg.min_count
        };
        if count < need {
            continue;
        }

        mentions.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.line.cmp(&b.line))
        });

        let max_score = mentions.iter().map(|m| m.score).fold(0.0f32, f32::max);
        let mean_score = mentions.iter().map(|m| m.score).sum::<f32>() / count as f32;
        let first_line = mentions.iter().map(|m| m.line).min().unwrap_or(0);

        let mut lines_sample = Vec::new();
        for m in &mentions {
            if !lines_sample.contains(&m.line) {
                lines_sample.push(m.line);
            }
            if lines_sample.len() >= cfg.max_source_lines {
                break;
            }
        }

        if mentions.len() > cfg.max_mentions_per_name {
            mentions.truncate(cfg.max_mentions_per_name);
        }

        out.push(CharacterInfo {
            name,
            label,
            count,
            max_score,
            mean_score,
            first_line,
            lines: lines_sample,
            mentions,
        });
    }

    out.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| {
                b.mean_score
                    .partial_cmp(&a.mean_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.name.cmp(&b.name))
    });
    out
}

/// Threshold sweep helpers for choosing `min_score`.
#[derive(Debug, Clone, Serialize)]
pub struct ThresholdRow {
    pub min_score: f32,
    pub mentions: usize,
    pub unique_names: usize,
    pub single_char_unique: usize,
    pub top10: Vec<(String, usize, f32)>,
}

/// Collect all PER (or configured) mentions without min_count for analysis.
pub fn collect_raw_mentions(
    _all_lines: &[String],
    line_indices: &[usize],
    line_results: &[NerResult],
    labels: &[String],
    drop_punct: bool,
) -> Vec<(String, String, f32, usize, usize, usize)> {
    // (label, text, score, line_1based, start, end)
    let mut out = Vec::new();
    for (res, &orig_idx) in line_results.iter().zip(line_indices.iter()) {
        if should_skip_line(&res.text) {
            continue;
        }
        for ent in &res.entities {
            if !label_ok(&ent.label, labels) {
                continue;
            }
            if drop_punct && is_punct_only(&ent.text) {
                continue;
            }
            out.push((
                ent.label.clone(),
                ent.text.clone(),
                ent.score,
                orig_idx + 1,
                ent.start,
                ent.end,
            ));
        }
    }
    out
}

pub fn sweep_thresholds(
    raw: &[(String, String, f32, usize, usize, usize)],
    thresholds: &[f32],
    min_count: usize,
) -> Vec<ThresholdRow> {
    thresholds
        .iter()
        .map(|&thr| {
            let mut counts: BTreeMap<String, (usize, f32, f32)> = BTreeMap::new();
            // name -> (count, max, sum)
            let mut mentions = 0usize;
            for (_lab, text, score, _, _, _) in raw {
                if *score < thr {
                    continue;
                }
                mentions += 1;
                let e = counts.entry(text.clone()).or_insert((0, 0.0, 0.0));
                e.0 += 1;
                e.1 = e.1.max(*score);
                e.2 += *score;
            }
            let kept: Vec<_> = counts
                .into_iter()
                .filter(|(_, (c, _, _))| *c >= min_count)
                .collect();
            let single_char_unique = kept.iter().filter(|(n, _)| n.chars().count() == 1).count();
            let mut top: Vec<_> = kept
                .iter()
                .map(|(n, (c, mx, _))| (n.clone(), *c, *mx))
                .collect();
            top.sort_by(|a, b| b.1.cmp(&a.1));
            top.truncate(10);
            ThresholdRow {
                min_score: thr,
                mentions,
                unique_names: kept.len(),
                single_char_unique,
                top10: top,
            }
        })
        .collect()
}

/// Markdown report for characters.
pub fn characters_to_markdown(
    chars: &[CharacterInfo],
    title: &str,
    cfg: &AggregateConfig,
) -> String {
    let mut md = String::new();
    md.push_str(&format!("# {title}\n\n"));
    md.push_str(&format!(
        "- min_score: `{}`, min_count: `{}`, labels: {:?}\n",
        cfg.min_score, cfg.min_count, cfg.labels
    ));
    md.push_str(&format!("- characters: **{}**\n\n", chars.len()));
    md.push_str("| # | 名称 | 次数 | mean | max | 首行 | 样例行 |\n");
    md.push_str("|---:|---|---:|---:|---:|---:|---|\n");
    for (i, c) in chars.iter().enumerate() {
        let lines: String = c
            .lines
            .iter()
            .take(8)
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        md.push_str(&format!(
            "| {} | {} | {} | {:.3} | {:.3} | {} | {} |\n",
            i + 1,
            c.name,
            c.count,
            c.mean_score,
            c.max_score,
            c.first_line,
            lines
        ));
    }
    md.push_str("\n## 来源摘录 (Top 15)\n");
    for c in chars.iter().take(15) {
        md.push_str(&format!("\n### {} x{}\n", c.name, c.count));
        for m in c.mentions.iter().take(3) {
            md.push_str(&format!(
                "- L{} [{},{}) score={:.3}: {}\n",
                m.line, m.start, m.end, m.score, m.line_text
            ));
        }
    }
    md
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ner::{NerEntity, NerResult};

    fn ent(text: &str, start: usize, score: f32) -> NerEntity {
        NerEntity {
            text: text.into(),
            label: "PER".into(),
            start,
            end: start + text.chars().count(),
            score,
        }
    }

    #[test]
    fn aggregates_and_filters() {
        let lines: Vec<String> = vec![
            "健太が来た。".into(),
            "健太は走った。".into(),
            "美咲だけ。".into(),
            "太郎です。".into(),
        ];
        let results = vec![
            NerResult {
                text: lines[0].clone(),
                entities: vec![ent("健太", 0, 0.95)],
                labels: vec![],
            },
            NerResult {
                text: lines[1].clone(),
                entities: vec![ent("健太", 0, 0.99)],
                labels: vec![],
            },
            NerResult {
                text: lines[2].clone(),
                entities: vec![ent("美咲", 0, 0.5)], // below min_score
                labels: vec![],
            },
            NerResult {
                text: lines[3].clone(),
                entities: vec![ent("太郎", 0, 0.99)], // count=1 < min_count
                labels: vec![],
            },
        ];
        let idx = vec![0, 1, 2, 3];
        let mut cfg = AggregateConfig::default();
        cfg.min_score = 0.9;
        cfg.min_count = 2;
        cfg.min_count_single_char = 5;
        let chars = aggregate_characters(&lines, &idx, &results, &cfg);
        assert_eq!(
            chars.len(),
            1,
            "got: {:?}",
            chars.iter().map(|c| &c.name).collect::<Vec<_>>()
        );
        assert_eq!(chars[0].name, "健太");
        assert_eq!(chars[0].count, 2);
        // mentions sorted by score desc → line 2 (0.99) first
        assert_eq!(chars[0].mentions[0].line, 2);
        assert_eq!(chars[0].first_line, 1);
    }

    #[test]
    fn threshold_sweep_monotonic() {
        let raw = vec![
            ("PER".into(), "A".into(), 0.95f32, 1, 0, 1),
            ("PER".into(), "A".into(), 0.80, 2, 0, 1),
            ("PER".into(), "B".into(), 0.99, 3, 0, 1),
        ];
        let rows = sweep_thresholds(&raw, &[0.7, 0.9, 0.99], 1);
        assert!(rows[0].unique_names >= rows[1].unique_names);
        assert!(rows[1].unique_names >= rows[2].unique_names);
    }
}
