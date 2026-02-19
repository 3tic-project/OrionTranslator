use crate::ner::{NerPipeline, NerEntity};
use crate::{GlossaryProgressCallback, GlossaryProgressEvent, emit};
use anyhow::Result;
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

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
    let cleaned: String = line
        .replace('\u{3000}', "")
        .replace('\n', "")
        .trim()
        .to_string();

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

    for (text_idx, (entities, &line_idx)) in
        entities_per_text.iter().zip(batch_indices.iter()).enumerate()
    {
        let line_text = &batch_texts[text_idx];

        for entity in entities {
            let is_person = PERSON_TYPES
                .iter()
                .any(|t| entity.label.contains(t));
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
pub async fn detect_characters_embedded<B: Backend + 'static>(
    lines: &[String],
    pipeline: Arc<Mutex<NerPipeline<B>>>,
    batch_size: usize,
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

    emit(&progress, GlossaryProgressEvent::Log {
        message: format!(
            "有效文本行数: {} (跳过了 {} 行)",
            valid_lines.len(),
            lines.len() - valid_lines.len()
        ),
    });

    if valid_lines.is_empty() {
        return Ok(HashMap::new());
    }

    // Create batches
    let mut batches: Vec<(Vec<String>, Vec<usize>)> = Vec::new();
    for i in (0..valid_lines.len()).step_by(batch_size) {
        let end = (i + batch_size).min(valid_lines.len());
        let batch_texts = valid_lines[i..end].to_vec();
        let batch_indices_slice = line_indices[i..end].to_vec();
        batches.push((batch_texts, batch_indices_slice));
    }

    let total_batches = batches.len();
    emit(&progress, GlossaryProgressEvent::Log {
        message: format!(
            "开始NER批量处理 ({} 个批次，每批 {} 行)",
            total_batches, batch_size
        ),
    });

    let mut character_mentions: HashMap<String, Vec<Mention>> = HashMap::new();

    for (batch_idx, (batch_texts, batch_idx_list)) in batches.into_iter().enumerate() {
        // Run inference under lock
        let texts_ref: Vec<&str> = batch_texts.iter().map(|s| s.as_str()).collect();
        let results = {
            let pipe = pipeline.lock().await;
            pipe.predict_batch(&texts_ref)?
        };

        // Convert NerResult → entities per text
        let entities_per_text: Vec<Vec<NerEntity>> = results
            .into_iter()
            .map(|r| r.entities)
            .collect();

        let batch_mentions = process_batch_entities(
            &entities_per_text,
            &batch_texts,
            &batch_idx_list,
            lines,
        );

        for (name, mentions) in batch_mentions {
            character_mentions
                .entry(name)
                .or_default()
                .extend(mentions);
        }

        emit(&progress, GlossaryProgressEvent::NerProgress {
            completed: batch_idx + 1,
            total: total_batches,
        });
    }

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

    emit(&progress, GlossaryProgressEvent::Log {
        message: format!("识别到 {} 个人物 (出现≥{}次)", characters.len(), min_count),
    });

    Ok(characters)
}
