use crate::model::BertInferenceBatch;
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Float, Int, Tensor};

#[derive(Config, Debug)]
pub struct BertEmbeddingsConfig {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub hidden_size: usize,
    pub hidden_dropout_prob: f64,
    pub layer_norm_eps: f64,
    pub pad_token_idx: usize,
}

#[derive(Module, Debug)]
pub struct BertEmbeddings<B: Backend> {
    pub pad_token_idx: usize,
    word_embeddings: Embedding<B>,
    position_embeddings: Embedding<B>,
    token_type_embeddings: Embedding<B>,
    layer_norm: LayerNorm<B>,
    dropout: Dropout,
    max_position_embeddings: usize,
}

impl BertEmbeddingsConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BertEmbeddings<B> {
        let word_embeddings = EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device);
        let position_embeddings =
            EmbeddingConfig::new(self.max_position_embeddings, self.hidden_size).init(device);
        let token_type_embeddings =
            EmbeddingConfig::new(self.type_vocab_size, self.hidden_size).init(device);
        let layer_norm = LayerNormConfig::new(self.hidden_size)
            .with_epsilon(self.layer_norm_eps)
            .init(device);
        let dropout = DropoutConfig::new(self.hidden_dropout_prob).init();

        BertEmbeddings {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
            max_position_embeddings: self.max_position_embeddings,
            pad_token_idx: self.pad_token_idx,
        }
    }
}

impl<B: Backend> BertEmbeddings<B> {
    pub fn forward(&self, item: &BertInferenceBatch<B>) -> Tensor<B, 3, Float> {
        let input_shape = item.tokens.shape();
        let input_ids = item.tokens.clone();

        let inputs_embeds = self.word_embeddings.forward(input_ids);
        let mut embeddings = inputs_embeds;

        let device = &self.position_embeddings.devices()[0];

        let token_type_ids = Tensor::<B, 2, Int>::zeros(input_shape.clone(), device);
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids);
        embeddings = embeddings + token_type_embeddings;

        let seq_length = input_shape.dims[1];

        // For BERT (max_position_embeddings == 512), use simple 0..seq_length positions
        // For RoBERTa (max_position_embeddings != 512), use pad_token_idx+1 based positions
        let position_ids_tensor: Tensor<B, 2, Int> = if self.max_position_embeddings == 512 {
            Tensor::arange(0..seq_length as i64, device)
                .reshape([1, seq_length])
                .expand(input_shape)
        } else {
            let position_ids = Tensor::arange(
                (self.pad_token_idx as i64) + 1
                    ..(seq_length as i64) + (self.pad_token_idx as i64) + 1,
                device,
            )
            .reshape([1, seq_length])
            .expand(input_shape);
            position_ids.mask_fill(item.mask_pad.clone(), self.pad_token_idx as i32)
        };

        let position_embeddings = self.position_embeddings.forward(position_ids_tensor);
        embeddings = embeddings + position_embeddings;

        let embeddings = self.layer_norm.forward(embeddings);
        self.dropout.forward(embeddings)
    }
}
