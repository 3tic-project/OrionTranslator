//! Load ModernBERT token-classification weights from safetensors into Burn modules.

use crate::config::ModernBertNerConfig;
use crate::model::{init_model, ModernBertForTokenClassification};
use burn::module::{ConstantRecord, Module, Param};
use burn::nn::{EmbeddingRecord, LayerNormRecord, LinearRecord};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use candle_core::{safetensors, Device as CandleDevice, Tensor as CandleTensor};
use std::collections::HashMap;
use std::path::Path;

fn load_1d<B: Backend>(t: &CandleTensor, device: &B::Device) -> Tensor<B, 1> {
    let dims = t.dims();
    let data = t.to_vec1::<f32>().expect("f32 1d");
    let shape: [usize; 1] = dims.try_into().expect("1d");
    Tensor::from_floats(TensorData::new(data, Shape::new(shape)), device)
}

fn load_2d<B: Backend>(t: &CandleTensor, device: &B::Device) -> Tensor<B, 2> {
    let dims = t.dims();
    let data = t
        .to_vec2::<f32>()
        .expect("f32 2d")
        .into_iter()
        .flatten()
        .collect::<Vec<f32>>();
    let shape: [usize; 2] = dims.try_into().expect("2d");
    Tensor::from_floats(TensorData::new(data, Shape::new(shape)), device)
}

/// PyTorch Linear weight is `[out, in]`; Burn Row layout is `[in, out]`.
fn pt_linear_weight<B: Backend>(weight_pt: &CandleTensor, device: &B::Device) -> Tensor<B, 2> {
    load_2d::<B>(weight_pt, device).transpose()
}

fn linear_record_no_bias<B: Backend>(
    weight_pt: &CandleTensor,
    device: &B::Device,
) -> LinearRecord<B> {
    LinearRecord {
        weight: Param::from_tensor(pt_linear_weight::<B>(weight_pt, device)),
        bias: None,
    }
}

fn linear_record_bias<B: Backend>(
    weight_pt: &CandleTensor,
    bias_pt: &CandleTensor,
    device: &B::Device,
) -> LinearRecord<B> {
    LinearRecord {
        weight: Param::from_tensor(pt_linear_weight::<B>(weight_pt, device)),
        bias: Some(Param::from_tensor(load_1d::<B>(bias_pt, device))),
    }
}

fn layer_norm_record_weight_only<B: Backend>(
    weight: &CandleTensor,
    device: &B::Device,
) -> LayerNormRecord<B> {
    LayerNormRecord {
        gamma: Param::from_tensor(load_1d::<B>(weight, device)),
        beta: None,
        epsilon: ConstantRecord::new(),
    }
}

fn embedding_record<B: Backend>(weight: &CandleTensor, device: &B::Device) -> EmbeddingRecord<B> {
    EmbeddingRecord {
        weight: Param::from_tensor(load_2d::<B>(weight, device)),
    }
}

fn require<'a>(
    map: &'a HashMap<String, CandleTensor>,
    key: &str,
) -> anyhow::Result<&'a CandleTensor> {
    map.get(key)
        .ok_or_else(|| anyhow::anyhow!("missing weight tensor: {key}"))
}

/// Load a full NER model from a HuggingFace directory (`model.safetensors` + `config.json`).
pub fn load_from_dir<B: Backend>(
    model_dir: impl AsRef<Path>,
    device: &B::Device,
) -> anyhow::Result<(ModernBertForTokenClassification<B>, ModernBertNerConfig)> {
    let model_dir = model_dir.as_ref();
    let config = ModernBertNerConfig::load(model_dir.join("config.json"))?;
    let weights_path = model_dir.join("model.safetensors");
    let model = load_from_safetensors::<B>(&weights_path, &config, device)?;
    Ok((model, config))
}

pub fn load_from_safetensors<B: Backend>(
    path: &Path,
    cfg: &ModernBertNerConfig,
    device: &B::Device,
) -> anyhow::Result<ModernBertForTokenClassification<B>> {
    let weights = safetensors::load(path, &CandleDevice::Cpu)
        .map_err(|e| anyhow::anyhow!("safetensors load failed: {e:?}"))?;

    let mut model = init_model::<B>(cfg, device);

    // Embeddings
    model.model.embeddings.tok_embeddings = model
        .model
        .embeddings
        .tok_embeddings
        .clone()
        .load_record(embedding_record(
            require(&weights, "model.embeddings.tok_embeddings.weight")?,
            device,
        ));
    model.model.embeddings.norm =
        model
            .model
            .embeddings
            .norm
            .clone()
            .load_record(layer_norm_record_weight_only(
                require(&weights, "model.embeddings.norm.weight")?,
                device,
            ));

    // Layers
    for layer_id in 0..cfg.num_hidden_layers {
        let prefix = format!("model.layers.{layer_id}");
        let layer = &mut model.model.layers[layer_id];

        if let Some(norm) = layer.attn_norm.take() {
            let w = require(&weights, &format!("{prefix}.attn_norm.weight"))?;
            layer.attn_norm = Some(norm.load_record(layer_norm_record_weight_only(w, device)));
        }

        layer.attn.wqkv = layer.attn.wqkv.clone().load_record(linear_record_no_bias(
            require(&weights, &format!("{prefix}.attn.Wqkv.weight"))?,
            device,
        ));
        layer.attn.wo = layer.attn.wo.clone().load_record(linear_record_no_bias(
            require(&weights, &format!("{prefix}.attn.Wo.weight"))?,
            device,
        ));
        layer.mlp_norm = layer
            .mlp_norm
            .clone()
            .load_record(layer_norm_record_weight_only(
                require(&weights, &format!("{prefix}.mlp_norm.weight"))?,
                device,
            ));
        layer.mlp.wi = layer.mlp.wi.clone().load_record(linear_record_no_bias(
            require(&weights, &format!("{prefix}.mlp.Wi.weight"))?,
            device,
        ));
        layer.mlp.wo = layer.mlp.wo.clone().load_record(linear_record_no_bias(
            require(&weights, &format!("{prefix}.mlp.Wo.weight"))?,
            device,
        ));
    }

    model.model.final_norm =
        model
            .model
            .final_norm
            .clone()
            .load_record(layer_norm_record_weight_only(
                require(&weights, "model.final_norm.weight")?,
                device,
            ));

    model.head.dense = model.head.dense.clone().load_record(linear_record_no_bias(
        require(&weights, "head.dense.weight")?,
        device,
    ));
    model.head.norm = model
        .head
        .norm
        .clone()
        .load_record(layer_norm_record_weight_only(
            require(&weights, "head.norm.weight")?,
            device,
        ));
    model.classifier = model.classifier.clone().load_record(linear_record_bias(
        require(&weights, "classifier.weight")?,
        require(&weights, "classifier.bias")?,
        device,
    ));

    Ok(model)
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use std::path::PathBuf;

    type B = NdArray;

    fn model_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../alnilam/ner_model")
    }

    #[test]
    fn loads_real_checkpoint() {
        let dir = model_dir();
        if !dir.join("model.safetensors").exists() {
            eprintln!("skip: model not present at {}", dir.display());
            return;
        }
        let device = Default::default();
        let (model, cfg) = load_from_dir::<B>(&dir, &device).expect("load");
        assert_eq!(cfg.hidden_size, 256);
        assert_eq!(model.model.layers.len(), 10);

        use crate::model::{ForwardCache, NerBatch};
        use burn::tensor::{Int, Tensor};
        let ids = Tensor::<B, 1, Int>::from_ints([6i64, 100, 200, 4], &device).unsqueeze_dim(0);
        let mask = Tensor::<B, 2>::ones([1, 4], &device);
        let batch = NerBatch {
            input_ids: ids,
            attention_mask: mask,
            has_padding: false,
        };
        let logits = model.forward(&batch, &cfg, &mut ForwardCache::default());
        assert_eq!(logits.dims(), [1, 4, 17]);
    }
}
