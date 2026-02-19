use burn::config::Config;
use burn::module::Module;
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::Initializer::KaimingUniform;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};
use serde::Deserialize;
use std::collections::HashMap;

use crate::embedding::{BertEmbeddings, BertEmbeddingsConfig};

/// NER-specific config parsed from HuggingFace config.json
#[derive(Debug, Clone, Deserialize)]
pub struct NerConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub hidden_dropout_prob: f64,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    #[serde(default)]
    pub model_type: String,
    pub id2label: HashMap<String, String>,
}

fn default_layer_norm_eps() -> f64 {
    1e-12
}

impl NerConfig {
    pub fn num_labels(&self) -> usize {
        self.id2label.len()
    }
}

/// BERT model config compatible with burn
#[derive(Config, Debug)]
pub struct BertModelConfig {
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub layer_norm_eps: f64,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub hidden_dropout_prob: f64,
    pub model_type: String,
    pub pad_token_id: usize,
}

impl From<&NerConfig> for BertModelConfig {
    fn from(c: &NerConfig) -> Self {
        Self {
            num_attention_heads: c.num_attention_heads,
            num_hidden_layers: c.num_hidden_layers,
            layer_norm_eps: c.layer_norm_eps,
            hidden_size: c.hidden_size,
            intermediate_size: c.intermediate_size,
            vocab_size: c.vocab_size,
            max_position_embeddings: c.max_position_embeddings,
            type_vocab_size: c.type_vocab_size,
            hidden_dropout_prob: c.hidden_dropout_prob,
            model_type: c.model_type.clone(),
            pad_token_id: c.pad_token_id,
        }
    }
}

impl BertModelConfig {
    pub fn init_bert<B: Backend>(&self, device: &B::Device) -> BertModel<B> {
        let embeddings = self.get_embeddings_config().init(device);
        let encoder = self.get_encoder_config().init(device);

        BertModel {
            embeddings,
            encoder,
        }
    }

    pub fn init_for_token_classification<B: Backend>(
        &self,
        num_labels: usize,
        device: &B::Device,
    ) -> BertForTokenClassification<B> {
        let bert = self.init_bert(device);
        let classifier = LinearConfig::new(self.hidden_size, num_labels).init(device);

        BertForTokenClassification {
            bert,
            classifier,
            num_labels,
        }
    }

    fn get_embeddings_config(&self) -> BertEmbeddingsConfig {
        BertEmbeddingsConfig {
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            type_vocab_size: self.type_vocab_size,
            hidden_size: self.hidden_size,
            hidden_dropout_prob: self.hidden_dropout_prob,
            layer_norm_eps: self.layer_norm_eps,
            pad_token_idx: self.pad_token_id,
        }
    }

    fn get_encoder_config(&self) -> TransformerEncoderConfig {
        TransformerEncoderConfig {
            n_heads: self.num_attention_heads,
            n_layers: self.num_hidden_layers,
            d_model: self.hidden_size,
            d_ff: self.intermediate_size,
            dropout: self.hidden_dropout_prob,
            norm_first: false,
            quiet_softmax: false,
            initializer: KaimingUniform {
                gain: 1.0 / libm::sqrt(3.0),
                fan_out_only: false,
            },
        }
    }
}

/// Base BERT model (embeddings + encoder)
#[derive(Module, Debug)]
pub struct BertModel<B: Backend> {
    pub embeddings: BertEmbeddings<B>,
    pub encoder: TransformerEncoder<B>,
}

/// Input batch for inference
#[derive(Debug, Clone)]
pub struct BertInferenceBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub mask_pad: Tensor<B, 2, Bool>,
}

impl<B: Backend> BertModel<B> {
    pub fn forward(&self, input: &BertInferenceBatch<B>) -> Tensor<B, 3> {
        let embedding = self.embeddings.forward(input);
        let device = &self.embeddings.devices()[0];
        let mask_pad = input.mask_pad.clone().to_device(device);
        let encoder_input = TransformerEncoderInput::new(embedding).mask_pad(mask_pad);
        self.encoder.forward(encoder_input)
    }
}

/// BERT for Token Classification (NER)
#[derive(Module, Debug)]
pub struct BertForTokenClassification<B: Backend> {
    pub bert: BertModel<B>,
    pub classifier: Linear<B>,
    #[allow(dead_code)]
    pub num_labels: usize,
}

impl<B: Backend> BertForTokenClassification<B> {
    /// Forward pass: returns logits of shape [batch_size, seq_len, num_labels]
    pub fn forward(&self, input: &BertInferenceBatch<B>) -> Tensor<B, 3> {
        let hidden_states = self.bert.forward(input);
        self.classifier.forward(hidden_states)
    }
}
