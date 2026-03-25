use crate::model::{BertForTokenClassification, BertInferenceBatch};
use crate::tokenizer::{EncodedInput, JapaneseBertTokenizer};
use anyhow::Result;
use burn::prelude::ToElement;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};
use log::debug;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
}

pub struct NerPipeline<B: Backend> {
    model: BertForTokenClassification<B>,
    tokenizer: JapaneseBertTokenizer,
    id2label: HashMap<usize, String>,
    device: B::Device,
    pad_token_id: usize,
}

impl<B: Backend> NerPipeline<B> {
    pub fn new(
        model: BertForTokenClassification<B>,
        tokenizer: JapaneseBertTokenizer,
        id2label: HashMap<String, String>,
        pad_token_id: usize,
        device: B::Device,
    ) -> Self {
        let id2label: HashMap<usize, String> = id2label
            .iter()
            .map(|(k, v)| (k.parse::<usize>().unwrap_or(0), v.clone()))
            .collect();

        Self {
            model,
            tokenizer,
            id2label,
            device,
            pad_token_id,
        }
    }

    pub fn predict(&self, text: &str) -> Result<NerResult> {
        let start_total = std::time::Instant::now();

        let start_token = std::time::Instant::now();
        let encoded = self.tokenizer.encode(text)?;
        let token_dur = start_token.elapsed();

        let start_tensor = std::time::Instant::now();
        let seq_len = encoded.input_ids.len();

        // Build input tensors: [1, seq_len]
        let input_ids_data: Vec<i64> = encoded.input_ids.iter().map(|&x| x as i64).collect();
        let input_ids = Tensor::<B, 1, Int>::from_ints(
            burn::tensor::TensorData::new(input_ids_data, [seq_len]),
            &self.device,
        )
        .unsqueeze_dim(0);

        // Padding mask: false = not padded, true = padded
        let mask_data: Vec<bool> = encoded
            .input_ids
            .iter()
            .map(|&id| id == self.pad_token_id as u32)
            .collect();
        let mask_pad = Tensor::<B, 1, Bool>::from_bool(
            burn::tensor::TensorData::new(mask_data, [seq_len]),
            &self.device,
        )
        .unsqueeze_dim(0);

        let batch = BertInferenceBatch {
            tokens: input_ids,
            mask_pad,
        };
        let tensor_dur = start_tensor.elapsed();

        let start_infer = std::time::Instant::now();
        let logits = self.model.forward(&batch);
        let infer_dur = start_infer.elapsed();

        let start_post = std::time::Instant::now();
        // Softmax + argmax
        let probabilities = softmax(logits.clone(), 2);
        let predictions = logits.argmax(2); // [1, seq_len, 1]
        let max_probs = probabilities.max_dim(2); // [1, seq_len, 1]

        // Extract to CPU: [1, seq_len, 1] -> [seq_len]
        let predictions_vec = tensor_to_vec_usize::<B>(predictions.reshape([seq_len]));
        let max_probs_vec = tensor_to_vec_f32::<B>(max_probs.reshape([seq_len]));

        let entities = self.extract_entities(&encoded, &predictions_vec, &max_probs_vec, text);
        let post_dur = start_post.elapsed();

        let total_dur = start_total.elapsed();
        debug!(
            "Perf | Tokenize: {:?} | TensorInit: {:?} | Inference: {:?} | PostProc: {:?} | Total: {:?}",
            token_dur, tensor_dur, infer_dur, post_dur, total_dur
        );

        Ok(NerResult {
            text: text.to_string(),
            entities,
        })
    }

    pub fn predict_batch(&self, texts: &[&str]) -> Result<Vec<NerResult>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        if texts.len() == 1 {
            return Ok(vec![self.predict(texts[0])?]);
        }

        let start_total = std::time::Instant::now();
        let batch_size = texts.len();

        // 1. Tokenize all texts
        let start_token = std::time::Instant::now();
        let encoded_list: Vec<EncodedInput> = texts
            .iter()
            .map(|t| self.tokenizer.encode(t))
            .collect::<Result<Vec<_>>>()?;
        let token_dur = start_token.elapsed();

        // 2. Find max length and pad
        let max_len = encoded_list
            .iter()
            .map(|e| e.input_ids.len())
            .max()
            .unwrap_or(0);

        let start_tensor = std::time::Instant::now();
        let mut batch_input_ids: Vec<i64> = Vec::with_capacity(batch_size * max_len);
        let mut batch_mask: Vec<bool> = Vec::with_capacity(batch_size * max_len);

        for encoded in &encoded_list {
            let seq_len = encoded.input_ids.len();
            let pad_len = max_len - seq_len;

            batch_input_ids.extend(encoded.input_ids.iter().map(|&x| x as i64));
            batch_input_ids.extend(std::iter::repeat(self.pad_token_id as i64).take(pad_len));

            batch_mask.extend(std::iter::repeat(false).take(seq_len));
            batch_mask.extend(std::iter::repeat(true).take(pad_len));
        }

        let input_ids = Tensor::<B, 2, Int>::from_ints(
            burn::tensor::TensorData::new(batch_input_ids, [batch_size, max_len]),
            &self.device,
        );
        let mask_pad = Tensor::<B, 2, Bool>::from_bool(
            burn::tensor::TensorData::new(batch_mask, [batch_size, max_len]),
            &self.device,
        );

        let batch = BertInferenceBatch {
            tokens: input_ids,
            mask_pad,
        };
        let tensor_dur = start_tensor.elapsed();

        // 3. Batch forward
        let start_infer = std::time::Instant::now();
        let logits = self.model.forward(&batch);
        let forward_dur = start_infer.elapsed();

        // 4. Post-processing
        let start_post = std::time::Instant::now();
        let probabilities = softmax(logits.clone(), 2);
        let predictions = logits.argmax(2);
        let max_probs = probabilities.max_dim(2);

        // 5. Split results per text
        let mut results = Vec::with_capacity(batch_size);
        for (i, (text, encoded)) in texts.iter().zip(encoded_list.iter()).enumerate() {
            let seq_len = encoded.input_ids.len();

            // [batch, seq_len, 1] -> slice [1, seq_len, 1] -> reshape [seq_len]
            let pred_slice: Tensor<B, 1, Int> = predictions
                .clone()
                .slice([i..i + 1, 0..seq_len])
                .reshape([seq_len]);
            let prob_slice: Tensor<B, 1> = max_probs
                .clone()
                .slice([i..i + 1, 0..seq_len])
                .reshape([seq_len]);

            let predictions_vec = tensor_to_vec_usize::<B>(pred_slice);
            let max_probs_vec = tensor_to_vec_f32::<B>(prob_slice);

            let entities = self.extract_entities(encoded, &predictions_vec, &max_probs_vec, text);

            results.push(NerResult {
                text: text.to_string(),
                entities,
            });
        }
        let post_dur = start_post.elapsed();

        let total_dur = start_total.elapsed();
        debug!(
            "[Batch] size={} | Tokenize: {:?} | TensorInit: {:?} | Forward: {:?} | PostProc: {:?} | Total: {:?} | Per-item: {:?}",
            batch_size, token_dur, tensor_dur, forward_dur, post_dur, total_dur,
            total_dur / batch_size as u32
        );

        Ok(results)
    }

    fn extract_entities(
        &self,
        encoded: &EncodedInput,
        predictions: &[usize],
        max_probs: &[f32],
        _original_text: &str,
    ) -> Vec<NerEntity> {
        let mut entities = Vec::new();
        // (text, label, start, end, score_sum, token_count)
        let mut current_entity: Option<(String, String, usize, usize, f32, usize)> = None;

        for (i, (pred, char_opt)) in predictions
            .iter()
            .zip(encoded.original_chars.iter())
            .enumerate()
        {
            let label = self
                .id2label
                .get(pred)
                .cloned()
                .unwrap_or_else(|| "O".to_string());

            let score = max_probs.get(i).copied().unwrap_or(0.0);

            if let Some((ch, orig_pos)) = char_opt {
                if label.starts_with("B-") {
                    if let Some((text, entity_label, start, end, score_sum, count)) =
                        current_entity.take()
                    {
                        entities.push(NerEntity {
                            text,
                            label: entity_label,
                            start,
                            end,
                            score: score_sum / count as f32,
                        });
                    }
                    let entity_type = label[2..].to_string();
                    current_entity = Some((
                        ch.to_string(),
                        entity_type,
                        *orig_pos,
                        *orig_pos + 1,
                        score,
                        1,
                    ));
                } else if label.starts_with("I-") {
                    if let Some((
                        ref mut text,
                        ref entity_label,
                        _,
                        ref mut end,
                        ref mut score_sum,
                        ref mut count,
                    )) = current_entity
                    {
                        let expected_type = &label[2..];
                        if entity_label == expected_type {
                            text.push(*ch);
                            *end = *orig_pos + 1;
                            *score_sum += score;
                            *count += 1;
                        } else {
                            let (text, entity_label, start, end, score_sum, count) =
                                current_entity.take().unwrap();
                            entities.push(NerEntity {
                                text,
                                label: entity_label,
                                start,
                                end,
                                score: score_sum / count as f32,
                            });
                        }
                    }
                } else {
                    if let Some((text, entity_label, start, end, score_sum, count)) =
                        current_entity.take()
                    {
                        entities.push(NerEntity {
                            text,
                            label: entity_label,
                            start,
                            end,
                            score: score_sum / count as f32,
                        });
                    }
                }
            }
        }

        if let Some((text, entity_label, start, end, score_sum, count)) = current_entity {
            entities.push(NerEntity {
                text,
                label: entity_label,
                start,
                end,
                score: score_sum / count as f32,
            });
        }

        entities
    }
}

fn tensor_to_vec_usize<B: Backend>(tensor: Tensor<B, 1, Int>) -> Vec<usize> {
    let data = tensor.into_data();
    let slice = data.as_slice::<B::IntElem>().unwrap();
    slice.iter().map(|x| x.to_usize()).collect()
}

fn tensor_to_vec_f32<B: Backend>(tensor: Tensor<B, 1>) -> Vec<f32> {
    let data = tensor.into_data();
    let slice = data.as_slice::<B::FloatElem>().unwrap();
    slice.iter().map(|x| x.to_f32()).collect()
}
