//! NER pipeline on top of the dedicated CPU engine.

use anyhow::Result;
use std::cell::RefCell;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::config::ModernBertNerConfig;
use crate::cpu::{CpuModel, Scratch};
use crate::ner::{
    align_to_chars, bioes_spans, decode_logits_row, label_table, BatchProfile, InferOptions,
    NerResult, ProfileAccum,
};
use crate::pack::{self, PackedBatch};
use crate::tokenizer::{CharNerTokenizer, EncodedChars};

pub struct CpuNerPipeline {
    model: Arc<CpuModel>,
    tokenizer: Arc<CharNerTokenizer>,
    label_table: Arc<Vec<String>>,
    /// Per-instance reusable buffers; clones get their own, so workers never share.
    scratch: RefCell<Scratch>,
    pub options: InferOptions,
}

impl Clone for CpuNerPipeline {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            label_table: self.label_table.clone(),
            scratch: RefCell::new(Scratch::default()),
            options: self.options.clone(),
        }
    }
}

impl CpuNerPipeline {
    pub fn load(model_dir: impl AsRef<Path>, max_length: usize) -> Result<Self> {
        let dir = model_dir.as_ref();
        let model = CpuModel::load(dir, max_length)?;
        let tokenizer = CharNerTokenizer::from_model_dir(dir, max_length)?;
        let label_table = label_table(model.config());
        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            label_table: Arc::new(label_table),
            scratch: RefCell::new(Scratch::default()),
            options: InferOptions::default(),
        })
    }

    pub fn config(&self) -> &ModernBertNerConfig {
        self.model.config()
    }

    pub fn with_options(mut self, options: InferOptions) -> Self {
        self.options = options;
        self
    }

    pub fn predict(&self, text: &str) -> Result<NerResult> {
        let mut batch = self.predict_batch(&[text])?;
        Ok(batch.pop().unwrap())
    }

    pub fn predict_batch(&self, texts: &[&str]) -> Result<Vec<NerResult>> {
        Ok(self.predict_batch_profiled(texts)?.0)
    }

    pub fn predict_batch_profiled(&self, texts: &[&str]) -> Result<(Vec<NerResult>, BatchProfile)> {
        let t_all = Instant::now();
        if texts.is_empty() {
            return Ok((vec![], BatchProfile::default()));
        }

        let t0 = Instant::now();
        let encoded: Vec<EncodedChars> = self.tokenizer.encode_batch(texts)?;
        let tokenize_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let chars: usize = encoded.iter().map(|e| e.chars.len()).sum();

        // Fully packed: no padding tokens are ever fed to the model.
        let t1 = Instant::now();
        let total: usize = encoded.iter().map(|e| e.input_ids.len()).sum();
        let mut ids = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(encoded.len() + 1);
        offsets.push(0usize);
        for enc in &encoded {
            ids.extend_from_slice(&enc.input_ids);
            offsets.push(ids.len());
        }
        let max_seq = (0..encoded.len())
            .map(|i| offsets[i + 1] - offsets[i])
            .max()
            .unwrap_or(0);
        let tensor_ms = t1.elapsed().as_secs_f32() * 1000.0;

        let num_labels = self.label_table.len();
        let t2 = Instant::now();
        let mut scratch = self.scratch.borrow_mut();
        let logits = self.model.forward(&ids, &offsets, &mut scratch)?;
        let forward_ms = t2.elapsed().as_secs_f32() * 1000.0;

        let t3 = Instant::now();
        let mut results = Vec::with_capacity(encoded.len());
        for (i, (text, enc)) in texts.iter().zip(encoded.iter()).enumerate() {
            let seq_len = offsets[i + 1] - offsets[i];
            let row = &logits[offsets[i] * num_labels..offsets[i + 1] * num_labels];
            let (pred_ids, conf) =
                decode_logits_row(row, seq_len, num_labels, self.options.skip_scores);
            let (labels, scores) = align_to_chars(enc, &pred_ids, &conf, &self.label_table);
            let entities = bioes_spans(&enc.chars, &labels, &scores);
            results.push(NerResult {
                text: (*text).to_string(),
                entities,
                labels,
            });
        }
        let post_ms = t3.elapsed().as_secs_f32() * 1000.0;

        Ok((
            results,
            BatchProfile {
                tokenize_ms,
                tensor_ms,
                forward_ms,
                post_ms,
                total_ms: t_all.elapsed().as_secs_f32() * 1000.0,
                batch_size: encoded.len(),
                max_seq,
                chars,
            },
        ))
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
            let (results, prof) = self.predict_batch_profiled(&refs)?;
            accum.add(&prof);
            pack_results.push(results);
        }
        let ordered = pack::unsort_results(texts.len(), &packs, pack_results);
        Ok((ordered, accum, packs))
    }
}
