use crate::ner::NerEntity;
use crate::{emit, GlossaryProgressCallback, GlossaryProgressEvent, LoadedNerPipeline};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const PERSON_TYPES: &[&str] = &["PER", "PERSON"];
const MIN_SCORE: f32 = 0.9;
const CONTEXT_SIZE: usize = 5;
const PUNCTUATION_CHARS: &str = "。！？、，．「」『』（）【】〈〉《》・～…";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mention {
    pub line: usize,
    pub line_text: String,
    pub above: Vec<String>,
    pub follow: Vec<String>,
    pub confidence: f32,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterInfo {
    pub name: String,
    pub count: usize,
    pub content: Vec<Mention>,
}

fn should_skip_line(line: &str) -> bool {
    let cleaned: String = line.replace(['\u{3000}', '\n'], "").trim().to_string();

    if cleaned.chars().count() < 2 {
        return true;
    }

    if cleaned.chars().all(|c| PUNCTUATION_CHARS.contains(c)) {
        return true;
    }

    false
}

fn get_context_lines(lines: &[String], line_index: usize) -> (Vec<String>, Vec<String>) {
    let total = lines.len();

    let above_start = line_index.saturating_sub(CONTEXT_SIZE);
    let mut above: Vec<String> = lines[above_start..line_index]
        .iter()
        .map(|l| l.trim().to_string())
        .collect();
    while above.len() < CONTEXT_SIZE {
        above.insert(0, String::new());
    }

    let follow_end = (line_index + CONTEXT_SIZE + 1).min(total);
    let mut follow: Vec<String> = lines[line_index + 1..follow_end]
        .iter()
        .map(|l| l.trim().to_string())
        .collect();
    while follow.len() < CONTEXT_SIZE {
        follow.push(String::new());
    }

    (above, follow)
}

fn process_batch_entities(
    entities_per_text: &[Vec<NerEntity>],
    batch_texts: &[String],
    batch_indices: &[usize],
    all_lines: &[String],
) -> HashMap<String, Vec<Mention>> {
    let mut mentions: HashMap<String, Vec<Mention>> = HashMap::new();

    for (text_idx, (entities, &line_idx)) in entities_per_text
        .iter()
        .zip(batch_indices.iter())
        .enumerate()
    {
        let line_text = &batch_texts[text_idx];

        for entity in entities {
            let is_person = PERSON_TYPES.iter().any(|t| entity.label.contains(t));
            if !is_person || entity.score < MIN_SCORE {
                continue;
            }

            let (above, follow) = get_context_lines(all_lines, line_idx);

            let mention = Mention {
                line: line_idx + 1,
                line_text: line_text.clone(),
                above,
                follow,
                confidence: entity.score,
                score: entity.score,
            };

            mentions
                .entry(entity.text.clone())
                .or_default()
                .push(mention);
        }
    }

    mentions
}

/// Detect characters using an embedded NER pipeline (no HTTP).
pub(crate) async fn detect_characters_embedded(
    lines: &[String],
    pipeline: &LoadedNerPipeline,
    min_count: usize,
    progress: GlossaryProgressCallback,
) -> Result<HashMap<String, CharacterInfo>> {
    // Filter valid lines
    let mut valid_lines: Vec<String> = Vec::new();
    let mut line_indices: Vec<usize> = Vec::new();

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim().to_string();
        if !should_skip_line(&trimmed) {
            valid_lines.push(trimmed);
            line_indices.push(idx);
        }
    }

    emit(
        &progress,
        GlossaryProgressEvent::Log {
            message: format!(
                "有效文本行数: {} (跳过了 {} 行)",
                valid_lines.len(),
                lines.len() - valid_lines.len()
            ),
        },
    );

    if valid_lines.is_empty() {
        return Ok(HashMap::new());
    }

    emit(
        &progress,
        GlossaryProgressEvent::Log {
            message: "开始 NER 长度打包与后端优化推理".to_string(),
        },
    );

    let report_progress = |completed, total| {
        emit(
            &progress,
            GlossaryProgressEvent::NerProgress { completed, total },
        );
    };
    let results = pipeline.predict_document(&valid_lines, Some(&report_progress))?;
    let entities_per_text: Vec<Vec<NerEntity>> =
        results.into_iter().map(|result| result.entities).collect();
    let character_mentions =
        process_batch_entities(&entities_per_text, &valid_lines, &line_indices, lines);

    // Filter by min_count
    let mut characters: HashMap<String, CharacterInfo> = HashMap::new();
    for (name, mentions) in character_mentions {
        if mentions.len() >= min_count {
            let count = mentions.len();
            characters.insert(
                name.clone(),
                CharacterInfo {
                    name,
                    count,
                    content: mentions,
                },
            );
        }
    }

    emit(
        &progress,
        GlossaryProgressEvent::Log {
            message: format!("识别到 {} 个人物 (出现≥{}次)", characters.len(), min_count),
        },
    );

    Ok(characters)
}
