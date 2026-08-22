//! HuggingFace ModernBERT config (token classification).

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct ModernBertNerConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default)]
    pub norm_bias: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default)]
    pub classifier_bias: bool,
    #[serde(default = "default_global_attn")]
    pub global_attn_every_n_layers: usize,
    #[serde(default = "default_local_attention")]
    pub local_attention: usize,
    #[serde(default = "default_global_rope")]
    pub global_rope_theta: f64,
    #[serde(default = "default_local_rope")]
    pub local_rope_theta: f64,
    pub pad_token_id: usize,
    #[serde(default = "default_cls")]
    pub cls_token_id: usize,
    #[serde(default = "default_sep")]
    pub sep_token_id: usize,
    #[serde(default)]
    pub model_type: String,
    pub id2label: HashMap<String, String>,
    pub label2id: HashMap<String, usize>,
    #[serde(default = "default_hidden_act")]
    pub hidden_activation: String,
    #[serde(default = "default_classifier_act")]
    pub classifier_activation: String,
}

fn default_norm_eps() -> f64 {
    1e-5
}
fn default_layer_norm_eps() -> f64 {
    1e-5
}
fn default_global_attn() -> usize {
    3
}
fn default_local_attention() -> usize {
    128
}
fn default_global_rope() -> f64 {
    160_000.0
}
fn default_local_rope() -> f64 {
    10_000.0
}
fn default_cls() -> usize {
    6
}
fn default_sep() -> usize {
    4
}
fn default_hidden_act() -> String {
    "gelu".into()
}
fn default_classifier_act() -> String {
    "gelu".into()
}

impl ModernBertNerConfig {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let cfg: Self = serde_json::from_str(&text)?;
        Ok(cfg)
    }

    pub fn num_labels(&self) -> usize {
        self.id2label.len()
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn is_global_layer(&self, layer_id: usize) -> bool {
        layer_id % self.global_attn_every_n_layers.max(1) == 0
    }

    pub fn rope_theta_for_layer(&self, layer_id: usize) -> f64 {
        if self.is_global_layer(layer_id) {
            self.global_rope_theta
        } else {
            self.local_rope_theta
        }
    }

    /// Half-window radius for sliding attention (tokens on each side).
    pub fn local_window_radius(&self) -> usize {
        self.local_attention / 2
    }

    pub fn eps(&self) -> f64 {
        // Prefer norm_eps when present; fall back to layer_norm_eps.
        if self.norm_eps > 0.0 {
            self.norm_eps
        } else {
            self.layer_norm_eps
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ja_finetune_config() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../alnilam/ner_model/config.json"
        );
        let cfg = ModernBertNerConfig::load(path).expect("config");
        assert_eq!(cfg.hidden_size, 256);
        assert_eq!(cfg.num_hidden_layers, 10);
        assert_eq!(cfg.num_attention_heads, 4);
        assert_eq!(cfg.num_labels(), 17);
        assert!(cfg.is_global_layer(0));
        assert!(!cfg.is_global_layer(1));
        assert!(cfg.is_global_layer(3));
        assert_eq!(cfg.head_dim(), 64);
    }
}
