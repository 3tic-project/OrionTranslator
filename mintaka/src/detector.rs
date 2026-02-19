use crate::ner_client::{EntityOutput, NerClient};
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use log::warn;
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

/// Check if a line should be skipped for NER processing
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

/// Get context lines around a given line index
fn get_context_lines(lines: &[String], line_index: usize) -> (Vec<String>, Vec<String>) {
    let total = lines.len();

    // Above context
    let above_start = line_index.saturating_sub(CONTEXT_SIZE);
    let mut above: Vec<String> = lines[above_start..line_index]
        .iter()
        .map(|l| l.trim().to_string())
        .collect();
    while above.len() < CONTEXT_SIZE {
        above.insert(0, String::new());
    }

    // Below context
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

/// Process a single batch: call NER API and extract person mentions
fn process_batch_results(
    batch_entities: &[Vec<EntityOutput>],
    batch_texts: &[String],
    batch_indices: &[usize],
    all_lines: &[String],
) -> HashMap<String, Vec<Mention>> {
    let mut mentions: HashMap<String, Vec<Mention>> = HashMap::new();

    for (text_idx, (entities, &line_idx)) in
        batch_entities.iter().zip(batch_indices.iter()).enumerate()
    {
        let line_text = &batch_texts[text_idx];

        for entity in entities {
            let is_person = PERSON_TYPES
                .iter()
                .any(|t| entity.entity_type.contains(t));
            if !is_person || entity.score < MIN_SCORE {
                continue;
            }

            let (above, follow) = get_context_lines(all_lines, line_idx);

            let mention = Mention {
                line: line_idx + 1, // 1-based
                line_text: line_text.clone(),
                above,
                follow,
                confidence: entity.score,
                score: entity.score,
            };

            mentions
                .entry(entity.word.clone())
                .or_default()
                .push(mention);
        }
    }

    mentions
}

/// Main character detection: process all lines with concurrent batch requests
pub async fn detect_characters(
    lines: &[String],
    client: &NerClient,
    batch_size: usize,
    max_concurrent: usize,
    min_count: usize,
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

    println!(
        "📊 有效文本行数: {} (跳过了 {} 行)",
        valid_lines.len(),
        lines.len() - valid_lines.len()
    );

    if valid_lines.is_empty() {
        return Ok(HashMap::new());
    }

    // Create batches
    let mut batches: Vec<(Vec<String>, Vec<usize>)> = Vec::new();
    for i in (0..valid_lines.len()).step_by(batch_size) {
        let end = (i + batch_size).min(valid_lines.len());
        let batch_texts = valid_lines[i..end].to_vec();
        let batch_indices = line_indices[i..end].to_vec();
        batches.push((batch_texts, batch_indices));
    }

    let total_batches = batches.len();
    println!(
        "🔄 开始并发批量处理 ({} 个批次，每批 {} 行，{} 并发)",
        total_batches, batch_size, max_concurrent
    );

    // Progress bar
    let pb = ProgressBar::new(total_batches as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  "),
    );

    // Process batches concurrently with semaphore-based concurrency control
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent));
    let lines_arc = std::sync::Arc::new(lines.to_vec());
    let client_arc = std::sync::Arc::new(client.clone());
    let pb_arc = std::sync::Arc::new(pb);

    let mut handles = Vec::new();

    for (batch_texts, batch_indices) in batches {
        let sem = semaphore.clone();
        let lines_ref = lines_arc.clone();
        let client_ref = client_arc.clone();
        let pb_ref = pb_arc.clone();

        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();

            let result = client_ref
                .predict_batch(batch_texts.clone(), vec!["PER".to_string()])
                .await;

            pb_ref.inc(1);

            match result {
                Ok(entities) => {
                    Ok(process_batch_results(&entities, &batch_texts, &batch_indices, &lines_ref))
                }
                Err(e) => {
                    warn!("批次处理失败: {}", e);
                    Err(e)
                }
            }
        });

        handles.push(handle);
    }

    // Collect results
    let mut character_mentions: HashMap<String, Vec<Mention>> = HashMap::new();
    let mut completed = 0usize;

    for handle in handles {
        match handle.await {
            Ok(Ok(batch_mentions)) => {
                for (name, mentions) in batch_mentions {
                    character_mentions
                        .entry(name)
                        .or_default()
                        .extend(mentions);
                }
                completed += 1;
            }
            Ok(Err(_)) | Err(_) => {
                // Already warned above
            }
        }
    }

    pb_arc.finish_with_message("完成");

    println!(
        "🚀 处理完成: 成功处理 {}/{} 个批次",
        completed, total_batches
    );

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

    println!("🎯 识别到 {} 个人物 (出现≥{}次)", characters.len(), min_count);
    Ok(characters)
}
