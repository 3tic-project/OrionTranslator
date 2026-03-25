use crate::embedding::BertEmbeddingsRecord;
use crate::model::{BertForTokenClassificationRecord, BertModelConfig, BertModelRecord, NerConfig};
use burn::module::{ConstantRecord, Param};
use burn::nn::attention::MultiHeadAttentionRecord;
use burn::nn::transformer::{
    PositionWiseFeedForwardRecord, TransformerEncoderLayerRecord, TransformerEncoderRecord,
};
use burn::nn::{EmbeddingRecord, LayerNormRecord, LinearRecord};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use candle_core::{safetensors, Device, Tensor as CandleTensor};
use std::collections::HashMap;
use std::path::Path;

fn load_1d_tensor_from_candle<B: Backend>(
    tensor: &CandleTensor,
    device: &B::Device,
) -> Tensor<B, 1> {
    let dims = tensor.dims();
    let data = tensor.to_vec1::<f32>().unwrap();
    let array: [usize; 1] = dims.try_into().expect("Unexpected size");
    let data = TensorData::new(data, Shape::new(array));
    Tensor::<B, 1>::from_floats(data, &device.clone())
}

fn load_2d_tensor_from_candle<B: Backend>(
    tensor: &CandleTensor,
    device: &B::Device,
) -> Tensor<B, 2> {
    let dims = tensor.dims();
    let data = tensor
        .to_vec2::<f32>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect::<Vec<f32>>();
    let array: [usize; 2] = dims.try_into().expect("Unexpected size");
    let data = TensorData::new(data, Shape::new(array));
    Tensor::<B, 2>::from_floats(data, &device.clone())
}

fn load_layer_norm_safetensor<B: Backend>(
    bias: &CandleTensor,
    weight: &CandleTensor,
    device: &B::Device,
) -> LayerNormRecord<B> {
    let beta = load_1d_tensor_from_candle::<B>(bias, device);
    let gamma = load_1d_tensor_from_candle::<B>(weight, device);

    LayerNormRecord {
        beta: Some(Param::from_tensor(beta)),
        gamma: Param::from_tensor(gamma),
        epsilon: ConstantRecord::new(),
    }
}

fn load_linear_safetensor<B: Backend>(
    bias: &CandleTensor,
    weight: &CandleTensor,
    device: &B::Device,
) -> LinearRecord<B> {
    let bias = load_1d_tensor_from_candle::<B>(bias, device);
    let weight = load_2d_tensor_from_candle::<B>(weight, device);
    let weight = weight.transpose();

    LinearRecord {
        weight: Param::from_tensor(weight),
        bias: Some(Param::from_tensor(bias)),
    }
}

fn load_intermediate_layer_safetensor<B: Backend>(
    linear_inner_weight: &CandleTensor,
    linear_inner_bias: &CandleTensor,
    linear_outer_weight: &CandleTensor,
    linear_outer_bias: &CandleTensor,
    device: &B::Device,
) -> PositionWiseFeedForwardRecord<B> {
    let linear_inner = load_linear_safetensor::<B>(linear_inner_bias, linear_inner_weight, device);
    let linear_outer = load_linear_safetensor::<B>(linear_outer_bias, linear_outer_weight, device);

    PositionWiseFeedForwardRecord {
        linear_inner,
        linear_outer,
        dropout: ConstantRecord::new(),
        gelu: ConstantRecord::new(),
    }
}

fn load_attention_layer_safetensor<B: Backend>(
    attention_tensors: HashMap<String, CandleTensor>,
    device: &B::Device,
) -> MultiHeadAttentionRecord<B> {
    let query = load_linear_safetensor::<B>(
        &attention_tensors["attention.self.query.bias"],
        &attention_tensors["attention.self.query.weight"],
        device,
    );
    let key = load_linear_safetensor::<B>(
        &attention_tensors["attention.self.key.bias"],
        &attention_tensors["attention.self.key.weight"],
        device,
    );
    let value = load_linear_safetensor::<B>(
        &attention_tensors["attention.self.value.bias"],
        &attention_tensors["attention.self.value.weight"],
        device,
    );
    let output = load_linear_safetensor::<B>(
        &attention_tensors["attention.output.dense.bias"],
        &attention_tensors["attention.output.dense.weight"],
        device,
    );

    MultiHeadAttentionRecord {
        query,
        key,
        value,
        output,
        d_model: ConstantRecord::new(),
        dropout: ConstantRecord::new(),
        activation: ConstantRecord::new(),
        n_heads: ConstantRecord::new(),
        d_k: ConstantRecord::new(),
        min_float: ConstantRecord::new(),
        quiet_softmax: ConstantRecord::new(),
    }
}

fn load_encoder_from_safetensors<B: Backend>(
    encoder_tensors: HashMap<String, CandleTensor>,
    device: &B::Device,
) -> TransformerEncoderRecord<B> {
    let mut layers: HashMap<usize, HashMap<String, CandleTensor>> = HashMap::new();

    for (key, value) in encoder_tensors.iter() {
        let layer_number = key.split('.').collect::<Vec<&str>>()[2]
            .parse::<usize>()
            .unwrap();
        layers
            .entry(layer_number)
            .or_default()
            .insert(key.to_string(), value.clone());
    }

    let mut layers = layers
        .into_iter()
        .collect::<Vec<(usize, HashMap<String, CandleTensor>)>>();
    layers.sort_by(|a, b| a.0.cmp(&b.0));

    let mut bert_encoder_layers: Vec<TransformerEncoderLayerRecord<B>> = Vec::new();
    for (key, value) in layers.iter() {
        let layer_key = format!("encoder.layer.{}", key);
        let attention_tensors = value.clone();
        let attention_tensors = attention_tensors
            .iter()
            .map(|(k, v)| (k.replace(&format!("{}.", layer_key), ""), v.clone()))
            .collect::<HashMap<String, CandleTensor>>();

        let attention_layer =
            load_attention_layer_safetensor::<B>(attention_tensors.clone(), device);

        let (bias_suffix, weight_suffix) =
            if attention_tensors.contains_key("attention.output.LayerNorm.bias") {
                ("bias", "weight")
            } else {
                ("beta", "gamma")
            };

        let norm_1 = load_layer_norm_safetensor(
            &attention_tensors[&format!("attention.output.LayerNorm.{}", bias_suffix)],
            &attention_tensors[&format!("attention.output.LayerNorm.{}", weight_suffix)],
            device,
        );

        let pwff = load_intermediate_layer_safetensor::<B>(
            &value[&format!("{}.intermediate.dense.weight", layer_key)],
            &value[&format!("{}.intermediate.dense.bias", layer_key)],
            &value[&format!("{}.output.dense.weight", layer_key)],
            &value[&format!("{}.output.dense.bias", layer_key)],
            device,
        );

        let norm_2 = load_layer_norm_safetensor::<B>(
            &value[&format!("{}.output.LayerNorm.{}", layer_key, bias_suffix)],
            &value[&format!("{}.output.LayerNorm.{}", layer_key, weight_suffix)],
            device,
        );

        let layer_record = TransformerEncoderLayerRecord {
            mha: attention_layer,
            pwff,
            norm_1,
            norm_2,
            dropout: ConstantRecord::new(),
            norm_first: ConstantRecord::new(),
        };

        bert_encoder_layers.push(layer_record);
    }

    TransformerEncoderRecord {
        layers: bert_encoder_layers,
        d_model: ConstantRecord::new(),
        d_ff: ConstantRecord::new(),
        n_heads: ConstantRecord::new(),
        n_layers: ConstantRecord::new(),
        dropout: ConstantRecord::new(),
        norm_first: ConstantRecord::new(),
        quiet_softmax: ConstantRecord::new(),
    }
}

fn load_embeddings_from_safetensors<B: Backend>(
    embedding_tensors: HashMap<String, CandleTensor>,
    device: &B::Device,
) -> BertEmbeddingsRecord<B> {
    let word_embeddings = load_embedding_safetensor(
        &embedding_tensors["embeddings.word_embeddings.weight"],
        device,
    );
    let position_embeddings = load_embedding_safetensor(
        &embedding_tensors["embeddings.position_embeddings.weight"],
        device,
    );
    let token_type_embeddings = load_embedding_safetensor(
        &embedding_tensors["embeddings.token_type_embeddings.weight"],
        device,
    );

    let (bias_key, weight_key) = if embedding_tensors.contains_key("embeddings.LayerNorm.bias") {
        ("embeddings.LayerNorm.bias", "embeddings.LayerNorm.weight")
    } else {
        ("embeddings.LayerNorm.beta", "embeddings.LayerNorm.gamma")
    };

    let layer_norm = load_layer_norm_safetensor::<B>(
        &embedding_tensors[bias_key],
        &embedding_tensors[weight_key],
        device,
    );

    BertEmbeddingsRecord {
        word_embeddings,
        position_embeddings,
        token_type_embeddings,
        layer_norm,
        dropout: ConstantRecord::new(),
        max_position_embeddings: ConstantRecord::new(),
        pad_token_idx: ConstantRecord::new(),
    }
}

fn load_embedding_safetensor<B: Backend>(
    weight: &CandleTensor,
    device: &B::Device,
) -> EmbeddingRecord<B> {
    let weight = load_2d_tensor_from_candle(weight, device);
    EmbeddingRecord {
        weight: Param::from_tensor(weight),
    }
}

/// Load BertForTokenClassification from a safetensors file + NerConfig
pub fn load_ner_model_from_safetensors<B: Backend>(
    model_path: &Path,
    config: &NerConfig,
    device: &B::Device,
) -> anyhow::Result<BertForTokenClassificationRecord<B>> {
    let weights = safetensors::load::<&Path>(model_path, &Device::Cpu)
        .map_err(|e| anyhow::anyhow!("Failed to load safetensors: {:?}", e))?;

    let model_prefix = if config.model_type.is_empty() {
        "".to_string()
    } else {
        format!("{}.", config.model_type)
    };

    let mut encoder_layers: HashMap<String, CandleTensor> = HashMap::new();
    let mut embeddings_layers: HashMap<String, CandleTensor> = HashMap::new();
    let mut classifier_layers: HashMap<String, CandleTensor> = HashMap::new();

    for (key, value) in weights.iter() {
        let key_clean = if !model_prefix.is_empty() {
            key.replace(&model_prefix, "")
        } else {
            key.clone()
        };

        if key_clean.starts_with("encoder.layer.") {
            encoder_layers.insert(key_clean, value.clone());
        } else if key_clean.starts_with("embeddings.") {
            embeddings_layers.insert(key_clean, value.clone());
        } else if key.starts_with("classifier.") {
            classifier_layers.insert(key.clone(), value.clone());
        }
    }

    let embeddings_record = load_embeddings_from_safetensors::<B>(embeddings_layers, device);
    let encoder_record = load_encoder_from_safetensors::<B>(encoder_layers, device);

    let bert_record = BertModelRecord {
        embeddings: embeddings_record,
        encoder: encoder_record,
    };

    // Load classifier head
    let classifier_record = load_linear_safetensor::<B>(
        &classifier_layers["classifier.bias"],
        &classifier_layers["classifier.weight"],
        device,
    );

    Ok(BertForTokenClassificationRecord {
        bert: bert_record,
        classifier: classifier_record,
        num_labels: ConstantRecord::new(),
    })
}

/// Load NerConfig from config.json
pub fn load_ner_config(config_path: &Path) -> anyhow::Result<NerConfig> {
    let config_str = std::fs::read_to_string(config_path)?;
    let config: NerConfig = serde_json::from_str(&config_str)?;
    Ok(config)
}

/// Build a BertModelConfig from NerConfig
pub fn build_model_config(ner_config: &NerConfig) -> BertModelConfig {
    BertModelConfig::from(ner_config)
}
