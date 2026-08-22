//! ModernBERT encoder + token-classification head (Burn).

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Float, Int, Tensor, TensorData};
use std::collections::HashMap;

use crate::config::ModernBertNerConfig;
use crate::ops::{attention_forward, gelu, linear2, pad_bias_from_mask, window_bias_host};
use crate::rope::{apply_rope, cos_sin_tables};

/// Inference batch.
#[derive(Debug, Clone)]
pub struct NerBatch<B: Backend> {
    /// Token ids `[batch, seq]`
    pub input_ids: Tensor<B, 2, Int>,
    /// Attention keep mask `[batch, seq]` with 1.0 = real token, 0.0 = pad
    pub attention_mask: Tensor<B, 2>,
    /// When false the pad bias is skipped entirely.
    pub has_padding: bool,
}

/// Device tensors that depend only on sequence length, reused across forward passes.
///
/// Without this the RoPE tables and the sliding-window mask are rebuilt on the host and
/// re-uploaded every call; on GPU that upload plus the materialised bias dominates.
pub struct ForwardCache<B: Backend> {
    rope: HashMap<(usize, u64), (Tensor<B, 4>, Tensor<B, 4>)>,
    window: HashMap<(usize, usize), Tensor<B, 4>>,
}

impl<B: Backend> Default for ForwardCache<B> {
    fn default() -> Self {
        Self {
            rope: HashMap::new(),
            window: HashMap::new(),
        }
    }
}

impl<B: Backend> ForwardCache<B> {
    fn rope(
        &mut self,
        seq: usize,
        head_dim: usize,
        theta: f64,
        device: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let (cos, sin) = self
            .rope
            .entry((seq, theta.to_bits()))
            .or_insert_with(|| cos_sin_tables::<B>(seq, head_dim, theta, device));
        (cos.clone(), sin.clone())
    }

    fn window(&mut self, seq: usize, radius: usize, device: &B::Device) -> Tensor<B, 4> {
        self.window
            .entry((seq, radius))
            .or_insert_with(|| {
                let host = window_bias_host(seq, radius);
                Tensor::<B, 1>::from_floats(TensorData::new(host, [seq * seq]), device)
                    .reshape([1, 1, seq, seq])
            })
            .clone()
    }
}

#[derive(Module, Debug)]
pub struct ModernBertEmbeddings<B: Backend> {
    pub tok_embeddings: Embedding<B>,
    pub norm: LayerNorm<B>,
}

impl<B: Backend> ModernBertEmbeddings<B> {
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = self.tok_embeddings.forward(input_ids);
        self.norm.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct ModernBertMlp<B: Backend> {
    pub wi: Linear<B>,
    pub wo: Linear<B>,
}

impl<B: Backend> ModernBertMlp<B> {
    /// `x` is `[tokens, hidden]` (batch and seq already flattened).
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = linear2(&self.wi, x);
        let [n, d2] = h.dims();
        let half = d2 / 2;
        let input = h.clone().slice([0..n, 0..half]);
        let gate = h.slice([0..n, half..d2]);
        linear2(&self.wo, gelu(input).mul(gate))
    }
}

#[derive(Module, Debug)]
pub struct ModernBertAttentionWeights<B: Backend> {
    pub wqkv: Linear<B>,
    pub wo: Linear<B>,
}

/// One encoder layer. `layer_id` is metadata for RoPE / window routing.
#[derive(Module, Debug)]
pub struct ModernBertLayer<B: Backend> {
    /// Present for layers > 0; layer 0 skips pre-attention norm.
    pub attn_norm: Option<LayerNorm<B>>,
    pub attn: ModernBertAttentionWeights<B>,
    pub mlp_norm: LayerNorm<B>,
    pub mlp: ModernBertMlp<B>,
    pub layer_id: usize,
}

#[derive(Module, Debug)]
pub struct ModernBertModel<B: Backend> {
    pub embeddings: ModernBertEmbeddings<B>,
    pub layers: Vec<ModernBertLayer<B>>,
    pub final_norm: LayerNorm<B>,
}

#[derive(Module, Debug)]
pub struct ModernBertPredictionHead<B: Backend> {
    pub dense: Linear<B>,
    pub norm: LayerNorm<B>,
}

impl<B: Backend> ModernBertPredictionHead<B> {
    /// `x` is `[tokens, hidden]`.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = linear2(&self.dense, x);
        let x = gelu(x);
        self.norm.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct ModernBertForTokenClassification<B: Backend> {
    pub model: ModernBertModel<B>,
    pub head: ModernBertPredictionHead<B>,
    pub classifier: Linear<B>,
}

/// Init empty structure on device (random weights); real weights come from safetensors.
pub fn init_model<B: Backend>(
    cfg: &ModernBertNerConfig,
    device: &B::Device,
) -> ModernBertForTokenClassification<B> {
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let eps = cfg.eps();

    let tok_embeddings = EmbeddingConfig::new(cfg.vocab_size, h).init(device);
    let emb_norm = LayerNormConfig::new(h)
        .with_epsilon(eps)
        .with_bias(cfg.norm_bias)
        .init(device);
    let embeddings = ModernBertEmbeddings {
        tok_embeddings,
        norm: emb_norm,
    };

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for layer_id in 0..cfg.num_hidden_layers {
        let attn_norm = if layer_id == 0 {
            None
        } else {
            Some(
                LayerNormConfig::new(h)
                    .with_epsilon(eps)
                    .with_bias(cfg.norm_bias)
                    .init(device),
            )
        };
        let wqkv = LinearConfig::new(h, 3 * h)
            .with_bias(cfg.attention_bias)
            .init(device);
        let wo = LinearConfig::new(h, h)
            .with_bias(cfg.attention_bias)
            .init(device);
        let mlp_norm = LayerNormConfig::new(h)
            .with_epsilon(eps)
            .with_bias(cfg.norm_bias)
            .init(device);
        let wi = LinearConfig::new(h, inter * 2)
            .with_bias(cfg.mlp_bias)
            .init(device);
        let wo_mlp = LinearConfig::new(inter, h)
            .with_bias(cfg.mlp_bias)
            .init(device);
        layers.push(ModernBertLayer {
            attn_norm,
            attn: ModernBertAttentionWeights { wqkv, wo },
            mlp_norm,
            mlp: ModernBertMlp { wi, wo: wo_mlp },
            layer_id,
        });
    }

    let final_norm = LayerNormConfig::new(h)
        .with_epsilon(eps)
        .with_bias(cfg.norm_bias)
        .init(device);
    let model = ModernBertModel {
        embeddings,
        layers,
        final_norm,
    };

    let head_dense = LinearConfig::new(h, h)
        .with_bias(cfg.classifier_bias)
        .init(device);
    let head_norm = LayerNormConfig::new(h)
        .with_epsilon(eps)
        .with_bias(cfg.norm_bias)
        .init(device);
    let head = ModernBertPredictionHead {
        dense: head_dense,
        norm: head_norm,
    };
    let classifier = LinearConfig::new(h, cfg.num_labels())
        .with_bias(true)
        .init(device);

    ModernBertForTokenClassification {
        model,
        head,
        classifier,
    }
}

impl<B: Backend> ModernBertForTokenClassification<B> {
    /// Forward: logits `[batch, seq, num_labels]`.
    pub fn forward(
        &self,
        batch: &NerBatch<B>,
        cfg: &ModernBertNerConfig,
        cache: &mut ForwardCache<B>,
    ) -> Tensor<B, 3, Float> {
        let device = &batch.input_ids.device();
        let [b, seq] = batch.input_ids.dims();
        let n_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim();
        let hidden = cfg.hidden_size;
        let radius = cfg.local_window_radius();
        let tokens = b * seq;

        // Encoder state stays rank-2 `[tokens, hidden]`: every projection is then a
        // single GEMM instead of one small GEMM per batch item.
        let mut hidden_states = self
            .model
            .embeddings
            .forward(batch.input_ids.clone())
            .reshape([tokens, hidden]);

        // The sliding mask is a no-op only when every pair is inside the window,
        // i.e. when the largest distance `seq - 1` does not exceed the radius.
        let window_active = seq > radius + 1;
        let pad_bias = batch
            .has_padding
            .then(|| pad_bias_from_mask::<B>(&batch.attention_mask));
        let window_bias = window_active.then(|| cache.window(seq, radius, device));

        // RoPE: two thetas (global / local layers).
        let (cos_g, sin_g) = cache.rope(seq, head_dim, cfg.global_rope_theta, device);
        let (cos_l, sin_l) = cache.rope(seq, head_dim, cfg.local_rope_theta, device);

        for layer in &self.model.layers {
            let residual = hidden_states.clone();
            let x = if let Some(norm) = &layer.attn_norm {
                norm.forward(hidden_states)
            } else {
                hidden_states
            };

            let qkv = linear2(&layer.attn.wqkv, x);
            // Reshape once to [B, S, 3, H, D], then pick Q/K/V via narrow on dim 2.
            let qkv = qkv.reshape([b, seq, 3, n_heads, head_dim]);
            let q = qkv
                .clone()
                .slice([0..b, 0..seq, 0..1, 0..n_heads, 0..head_dim])
                .reshape([b, seq, n_heads, head_dim])
                .swap_dims(1, 2);
            let k = qkv
                .clone()
                .slice([0..b, 0..seq, 1..2, 0..n_heads, 0..head_dim])
                .reshape([b, seq, n_heads, head_dim])
                .swap_dims(1, 2);
            let v = qkv
                .slice([0..b, 0..seq, 2..3, 0..n_heads, 0..head_dim])
                .reshape([b, seq, n_heads, head_dim])
                .swap_dims(1, 2);

            let is_global = cfg.is_global_layer(layer.layer_id);
            let (cos, sin) = if is_global {
                (cos_g.clone(), sin_g.clone())
            } else {
                (cos_l.clone(), sin_l.clone())
            };
            let q = apply_rope(q, cos.clone(), sin.clone());
            let k = apply_rope(k, cos, sin);

            let attn_window = if is_global {
                None
            } else {
                window_bias.as_ref()
            };
            let ctx = attention_forward(q, k, v, pad_bias.as_ref(), attn_window);
            let ctx = ctx.swap_dims(1, 2).reshape([tokens, hidden]);
            let attn_out = linear2(&layer.attn.wo, ctx);
            hidden_states = residual + attn_out;

            let residual = hidden_states.clone();
            let mlp_in = layer.mlp_norm.forward(hidden_states);
            let mlp_out = layer.mlp.forward(mlp_in);
            hidden_states = residual + mlp_out;
        }

        let hidden_states = self.model.final_norm.forward(hidden_states);
        let hidden_states = self.head.forward(hidden_states);
        linear2(&self.classifier, hidden_states).reshape([b, seq, cfg.num_labels()])
    }
}
