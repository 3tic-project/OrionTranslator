//! NER pipeline: tokenize → ModernBERT → BIOES decode → spans.

use crate::config::ModernBertNerConfig;
use crate::model::{ForwardCache, ModernBertForTokenClassification, NerBatch};
use crate::pack::{self, PackedBatch};
use crate::tokenizer::{CharNerTokenizer, EncodedChars};
use anyhow::Result;
use burn::prelude::ToElement;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NerEntity {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NerResult {
    pub text: String,
    pub entities: Vec<NerEntity>,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct BatchProfile {
    pub tokenize_ms: f32,
    pub tensor_ms: f32,
    pub forward_ms: f32,
    pub post_ms: f32,
    pub total_ms: f32,
    pub batch_size: usize,
    pub max_seq: usize,
    pub chars: usize,
}

#[derive(Debug, Clone, Default)]
pub struct ProfileAccum {
    pub batches: usize,
    pub tokenize_ms: f32,
    pub tensor_ms: f32,
    pub forward_ms: f32,
    pub post_ms: f32,
    pub total_ms: f32,
    pub chars: usize,
}

impl ProfileAccum {
    pub fn add(&mut self, p: &BatchProfile) {
        self.batches += 1;
        self.tokenize_ms += p.tokenize_ms;
        self.tensor_ms += p.tensor_ms;
        self.forward_ms += p.forward_ms;
        self.post_ms += p.post_ms;
        self.total_ms += p.total_ms;
        self.chars += p.chars;
    }

    pub fn report(&self) -> String {
        format!(
            "profile batches={} tokenize={:.1}ms ({:.0}%) tensor={:.1}ms ({:.0}%) forward={:.1}ms ({:.0}%) post={:.1}ms ({:.0}%) total={:.1}ms chars={} ({:.0} c/s)",
            self.batches,
            self.tokenize_ms,
            pct(self.tokenize_ms, self.total_ms),
            self.tensor_ms,
            pct(self.tensor_ms, self.total_ms),
            self.forward_ms,
            pct(self.forward_ms, self.total_ms),
            self.post_ms,
            pct(self.post_ms, self.total_ms),
            self.total_ms,
            self.chars,
            if self.total_ms > 0.0 {
                self.chars as f32 / (self.total_ms / 1000.0)
            } else {
                0.0
            }
        )
    }
}

fn pct(part: f32, total: f32) -> f32 {
    if total <= 0.0 {
        0.0
    } else {
        100.0 * part / total
    }
}

/// Runtime knobs for packing / decoding.
#[derive(Debug, Clone)]
pub struct InferOptions {
    pub max_sentences: usize,
    /// Approx token budget per pack: max_seq * batch_items.
    pub max_tokens: usize,
    pub sort_by_length: bool,
    /// If true, skip probability scores (argmax only, score=1.0).
    pub skip_scores: bool,
}

impl Default for InferOptions {
    fn default() -> Self {
        Self {
            max_sentences: 16,
            max_tokens: 2048,
            sort_by_length: true,
            skip_scores: false,
        }
    }
}

pub struct NerPipeline<B: Backend> {
    model: ModernBertForTokenClassification<B>,
    /// Shared: cloning a `Tokenizer` deep-copies the 100k-entry vocab.
    tokenizer: Arc<CharNerTokenizer>,
    cfg: ModernBertNerConfig,
    #[allow(dead_code)]
    id2label: HashMap<usize, String>,
    label_table: Vec<String>,
    device: B::Device,
    /// Per-instance; clones get a fresh one so workers never share it.
    cache: RefCell<ForwardCache<B>>,
    pub options: InferOptions,
}

impl<B: Backend> Clone for NerPipeline<B>
where
    B::Device: Clone,
    ModernBertForTokenClassification<B>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            cfg: self.cfg.clone(),
            id2label: self.id2label.clone(),
            label_table: self.label_table.clone(),
            device: self.device.clone(),
            cache: RefCell::new(ForwardCache::default()),
            options: self.options.clone(),
        }
    }
}

impl<B: Backend> NerPipeline<B> {
    pub fn new(
        model: ModernBertForTokenClassification<B>,
        tokenizer: CharNerTokenizer,
        cfg: ModernBertNerConfig,
        device: B::Device,
    ) -> Self {
        let id2label: HashMap<usize, String> = cfg
            .id2label
            .iter()
            .map(|(k, v)| (k.parse::<usize>().unwrap_or(0), v.clone()))
            .collect();
        let label_table = label_table(&cfg);
        Self {
            model,
            tokenizer: Arc::new(tokenizer),
            cfg,
            id2label,
            label_table,
            device,
            cache: RefCell::new(ForwardCache::default()),
            options: InferOptions::default(),
        }
    }

    pub fn config(&self) -> &ModernBertNerConfig {
        &self.cfg
    }

    pub fn with_options(mut self, options: InferOptions) -> Self {
        self.options = options;
        self
    }

    pub fn predict(&self, text: &str) -> Result<NerResult> {
        let mut batch = self.predict_batch(&[text])?;
        Ok(batch.pop().unwrap())
    }

    /// Full document: length-pack → micro-batches → restore original order.
    pub fn predict_document(
        &self,
        texts: &[String],
    ) -> Result<(Vec<NerResult>, ProfileAccum, Vec<PackedBatch>)> {
        let packs = pack::pack_texts(
            texts,
            self.options.max_sentences,
            self.options.max_tokens,
            self.options.sort_by_length,
        );
        let mut accum = ProfileAccum::default();
        let mut pack_results = Vec::with_capacity(packs.len());
        for pack in &packs {
            let refs: Vec<&str> = pack.texts.iter().map(|s| s.as_str()).collect();
            let (results, prof) = self.predict_batch_profiled(&refs, true)?;
            accum.add(&prof);
            pack_results.push(results);
        }
        let ordered = pack::unsort_results(texts.len(), &packs, pack_results);
        Ok((ordered, accum, packs))
    }

    pub fn predict_batch(&self, texts: &[&str]) -> Result<Vec<NerResult>> {
        Ok(self.predict_batch_profiled(texts, false)?.0)
    }

    pub fn predict_batch_profiled(
        &self,
        texts: &[&str],
        _collect_profile: bool,
    ) -> Result<(Vec<NerResult>, BatchProfile)> {
        let t_all = Instant::now();
        if texts.is_empty() {
            return Ok((vec![], BatchProfile::default()));
        }

        let t0 = Instant::now();
        let encoded: Vec<EncodedChars> = self.tokenizer.encode_batch(texts)?;
        let tokenize_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let chars: usize = encoded.iter().map(|e| e.chars.len()).sum();

        let batch_size = encoded.len();
        let max_len = encoded.iter().map(|e| e.input_ids.len()).max().unwrap_or(0);
        let pad_id = self.tokenizer.pad_token_id as i64;
        let num_labels = self.label_table.len().max(self.cfg.num_labels());

        let t1 = Instant::now();
        let mut ids = Vec::with_capacity(batch_size * max_len);
        let mut mask = Vec::with_capacity(batch_size * max_len);
        for enc in &encoded {
            let n = enc.input_ids.len();
            ids.extend(enc.input_ids.iter().map(|&x| x as i64));
            ids.resize(ids.len() + (max_len - n), pad_id);
            mask.extend(enc.attention_mask.iter().map(|&x| x as f32));
            mask.resize(mask.len() + (max_len - n), 0.0);
        }

        let input_ids = Tensor::<B, 1, Int>::from_ints(
            burn::tensor::TensorData::new(ids, [batch_size * max_len]),
            &self.device,
        )
        .reshape([batch_size, max_len]);
        let attention_mask = Tensor::<B, 1>::from_floats(
            burn::tensor::TensorData::new(mask, [batch_size * max_len]),
            &self.device,
        )
        .reshape([batch_size, max_len]);

        let batch = NerBatch {
            input_ids,
            attention_mask,
            has_padding: encoded.iter().any(|e| e.input_ids.len() != max_len),
        };
        let tensor_ms = t1.elapsed().as_secs_f32() * 1000.0;

        let t2 = Instant::now();
        let logits = self
            .model
            .forward(&batch, &self.cfg, &mut self.cache.borrow_mut()); // [B, S, C]
                                                                       // One host transfer of raw logits; decode on CPU (cheap for C≈17).
        let logits_host = tensor_to_f32_3d::<B>(logits, batch_size, max_len, num_labels);
        let forward_ms = t2.elapsed().as_secs_f32() * 1000.0;

        let t3 = Instant::now();
        let mut results = Vec::with_capacity(batch_size);
        for (i, (text, enc)) in texts.iter().zip(encoded.iter()).enumerate() {
            let seq_len = enc.input_ids.len();
            let (pred_ids, conf) = decode_logits_row(
                &logits_host[i],
                seq_len,
                num_labels,
                self.options.skip_scores,
            );
            let (labels, scores) = align_to_chars(enc, &pred_ids, &conf, &self.label_table);
            let entities = bioes_spans(&enc.chars, &labels, &scores);
            results.push(NerResult {
                text: (*text).to_string(),
                entities,
                labels,
            });
        }
        let post_ms = t3.elapsed().as_secs_f32() * 1000.0;
        let total_ms = t_all.elapsed().as_secs_f32() * 1000.0;

        let profile = BatchProfile {
            tokenize_ms,
            tensor_ms,
            forward_ms,
            post_ms,
            total_ms,
            batch_size,
            max_seq: max_len,
            chars,
        };
        Ok((results, profile))
    }
}

/// Dense `id -> label` table built from the config's `id2label` map.
pub(crate) fn label_table(cfg: &ModernBertNerConfig) -> Vec<String> {
    let id2label: HashMap<usize, String> = cfg
        .id2label
        .iter()
        .map(|(k, v)| (k.parse::<usize>().unwrap_or(0), v.clone()))
        .collect();
    let max_id = id2label.keys().copied().max().unwrap_or(0);
    let mut table = vec!["O".to_string(); max_id + 1];
    for (id, lab) in &id2label {
        if *id < table.len() {
            table[*id] = lab.clone();
        }
    }
    table
}

/// Stable host-side argmax + optional max-softmax score.
pub(crate) fn decode_logits_row(
    row: &[f32], // length max_len * num_labels, row-major [S, C]
    seq_len: usize,
    num_labels: usize,
    skip_scores: bool,
) -> (Vec<usize>, Vec<f32>) {
    let mut preds = Vec::with_capacity(seq_len);
    let mut scores = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let base = t * num_labels;
        let logits = &row[base..base + num_labels];
        let mut best_i = 0usize;
        let mut best_v = logits[0];
        for (i, &v) in logits.iter().enumerate().skip(1) {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        preds.push(best_i);
        if skip_scores {
            scores.push(1.0);
        } else {
            // softmax probability of argmax class
            let mut sum = 0.0f32;
            for &v in logits {
                sum += (v - best_v).exp();
            }
            scores.push(1.0 / sum.max(1e-12));
        }
    }
    (preds, scores)
}

pub(crate) fn align_to_chars(
    enc: &EncodedChars,
    pred_ids: &[usize],
    conf: &[f32],
    label_table: &[String],
) -> (Vec<String>, Vec<f32>) {
    let n = enc.chars.len();
    let mut labels = vec!["O".to_string(); n];
    let mut scores = vec![0.0f32; n];
    let mut prev: Option<usize> = None;
    for (tid, wid_opt) in enc.word_ids.iter().enumerate() {
        let Some(wid) = *wid_opt else {
            prev = None;
            continue;
        };
        if Some(wid) != prev && wid < n {
            let pid = pred_ids.get(tid).copied().unwrap_or(0);
            labels[wid] = label_table
                .get(pid)
                .cloned()
                .unwrap_or_else(|| "O".to_string());
            scores[wid] = conf.get(tid).copied().unwrap_or(0.0);
        }
        prev = Some(wid);
    }
    (labels, scores)
}

/// Lenient BIOES span decode (matches models/scripts/infer_book.py).
pub fn bioes_spans(chars: &[char], tags: &[String], scores: &[f32]) -> Vec<NerEntity> {
    let mut entities = Vec::new();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        let t = tags.get(i).map(String::as_str).unwrap_or("O");
        if t.is_empty() || t == "O" {
            i += 1;
            continue;
        }
        if let Some(et) = t.strip_prefix("S-") {
            let sc = scores.get(i).copied().unwrap_or(0.0);
            entities.push(NerEntity {
                text: chars[i].to_string(),
                label: et.to_string(),
                start: i,
                end: i + 1,
                score: sc,
            });
            i += 1;
            continue;
        }
        if let Some(et) = t.strip_prefix("B-") {
            let start = i;
            let mut j = i + 1;
            while j < n {
                let tj = tags.get(j).map(String::as_str).unwrap_or("O");
                if tj.len() > 2 && &tj[..2] == "I-" && &tj[2..] == et {
                    j += 1;
                    continue;
                }
                if tj.len() > 2 && &tj[..2] == "E-" && &tj[2..] == et {
                    j += 1;
                    break;
                }
                break;
            }
            let sc: f32 = scores[start..j].iter().sum::<f32>() / (j - start).max(1) as f32;
            entities.push(NerEntity {
                text: chars[start..j].iter().collect(),
                label: et.to_string(),
                start,
                end: j,
                score: sc,
            });
            i = j;
            continue;
        }
        if let Some(et) = t.strip_prefix("I-").or_else(|| t.strip_prefix("E-")) {
            let sc = scores.get(i).copied().unwrap_or(0.0);
            entities.push(NerEntity {
                text: chars[i].to_string(),
                label: et.to_string(),
                start: i,
                end: i + 1,
                score: sc,
            });
        }
        i += 1;
    }
    entities
}

fn tensor_to_f32_3d<B: Backend>(
    tensor: Tensor<B, 3>,
    batch: usize,
    seq: usize,
    classes: usize,
) -> Vec<Vec<f32>> {
    // returns batch rows, each flat [seq * classes]
    let data = tensor.into_data();
    let slice = data.as_slice::<B::FloatElem>().unwrap();
    let row = seq * classes;
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let start = b * row;
        out.push(
            slice[start..start + row]
                .iter()
                .map(|x| x.to_f32())
                .collect(),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bioes_basic() {
        let chars: Vec<char> = "艾莉是人".chars().collect();
        let tags = vec!["B-PER".into(), "E-PER".into(), "O".into(), "S-PER".into()];
        let scores = vec![0.9, 0.8, 0.1, 0.7];
        let ents = bioes_spans(&chars, &tags, &scores);
        assert_eq!(ents.len(), 2);
        assert_eq!(ents[0].text, "艾莉");
        assert_eq!(ents[0].label, "PER");
        assert_eq!(ents[1].text, "人");
    }

    #[test]
    fn decode_logits_argmax_and_score() {
        // 2 positions, 3 classes
        let row = vec![
            0.0, 5.0, 1.0, // pred 1
            2.0, 0.0, 0.0, // pred 0
        ];
        let (p, s) = decode_logits_row(&row, 2, 3, false);
        assert_eq!(p, vec![1, 0]);
        assert!(s[0] > 0.9);
        assert!(s[1] > 0.7);
    }

    #[test]
    fn decode_skip_scores() {
        let row = vec![1.0, 2.0];
        let (p, s) = decode_logits_row(&row, 1, 2, true);
        assert_eq!(p, vec![1]);
        assert_eq!(s, vec![1.0]);
    }
}
