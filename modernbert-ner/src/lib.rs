//! ModernBERT-JA token-classification NER inference.
//!
//! Backends:
//! - **cpu** — dedicated f32 engine (`cpu` module), fastest
//! - **ndarray** (CPU) — Burn reference, feature `ndarray`
//! - **wgpu** (GPU) — feature `wgpu`
//!
//! Model layout (HuggingFace directory):
//! ```text
//! model_dir/
//!   config.json
//!   model.safetensors
//!   tokenizer.json
//! ```

// The fusion-wrapped wgpu backend nests deeply enough to exhaust the default limit.
#![recursion_limit = "512"]

pub mod aggregate;
pub mod config;
pub mod cpu;
pub mod cpu_pipeline;
pub mod loader;
pub mod model;
pub mod ner;
pub mod ops;
pub mod pack;
pub mod rope;
pub mod tokenizer;

use anyhow::Result;
use std::path::Path;

pub use aggregate::{
    aggregate_characters, characters_to_markdown, collect_raw_mentions, sweep_thresholds,
    AggregateConfig, CharacterInfo, Mention, ThresholdRow,
};
pub use config::ModernBertNerConfig;
pub use cpu::CpuModel;
pub use cpu_pipeline::CpuNerPipeline;
pub use ner::{BatchProfile, InferOptions, NerEntity, NerPipeline, NerResult, ProfileAccum};
pub use pack::{estimate_pad_waste, pack_texts, PackedBatch};
pub use tokenizer::CharNerTokenizer;

/// Load NER pipeline on the given Burn device.
pub fn load_pipeline<B: burn::tensor::backend::Backend>(
    model_dir: impl AsRef<Path>,
    device: B::Device,
    max_length: usize,
) -> Result<NerPipeline<B>> {
    let model_dir = model_dir.as_ref();
    let (model, cfg) = loader::load_from_dir::<B>(model_dir, &device)?;
    let tokenizer = CharNerTokenizer::from_model_dir(model_dir, max_length)?;
    Ok(NerPipeline::new(model, tokenizer, cfg, device))
}

/// Convenience: load the dedicated CPU engine (fastest path for batch inference).
pub fn load_pipeline_cpu(model_dir: impl AsRef<Path>, max_length: usize) -> Result<CpuNerPipeline> {
    CpuNerPipeline::load(model_dir, max_length)
}

/// Load the Burn `ndarray` reference pipeline. Slower than [`load_pipeline_cpu`];
/// kept as the cross-check for the hand-written CPU kernels.
#[cfg(feature = "ndarray")]
pub fn load_pipeline_burn_cpu(
    model_dir: impl AsRef<Path>,
    max_length: usize,
) -> Result<NerPipeline<burn::backend::NdArray>> {
    let device = Default::default();
    load_pipeline::<burn::backend::NdArray>(model_dir, device, max_length)
}

/// Convenience: load with Wgpu backend.
#[cfg(feature = "wgpu")]
pub fn load_pipeline_wgpu(
    model_dir: impl AsRef<Path>,
    max_length: usize,
) -> Result<NerPipeline<burn::backend::Wgpu>> {
    use burn::backend::wgpu::WgpuDevice;
    let device = WgpuDevice::default();
    load_pipeline::<burn::backend::Wgpu>(model_dir, device, max_length)
}
