#![recursion_limit = "256"]

mod api;
mod batcher;
mod embedding;
mod loader;
mod model;
mod ner;
mod tokenizer;

use crate::api::{create_router, AppState};
use crate::batcher::{BatcherConfig, DynamicBatcher};
use crate::loader::{build_model_config, load_ner_config, load_ner_model_from_safetensors};
use crate::ner::NerPipeline;
use crate::tokenizer::JapaneseBertTokenizer;
use anyhow::Result;
use burn::module::Module;
use burn::tensor::backend::Backend;
use env_logger::Env;
use log::{info, warn};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

fn get_model_dir() -> String {
    std::env::var("MODEL_DIR").unwrap_or_else(|_| "./model".to_string())
}

fn launch<B: Backend + 'static>(device: B::Device) -> Result<()> {
    info!("Initializing BERT NER service (burn backend)...");

    let model_dir = get_model_dir();
    let model_path = format!("{}/model.safetensors", model_dir);
    let config_path = format!("{}/config.json", model_dir);
    let vocab_path = format!("{}/vocab.txt", model_dir);

    // Load NER config
    info!("Loading config from: {}", config_path);
    let ner_config = load_ner_config(Path::new(&config_path))?;
    info!(
        "Model config: hidden_size={}, layers={}, labels={:?}",
        ner_config.hidden_size, ner_config.num_hidden_layers, ner_config.id2label
    );

    // Build burn model config and initialize model structure
    let model_config = build_model_config(&ner_config);
    let num_labels = ner_config.num_labels();
    info!(
        "Initializing BertForTokenClassification with {} labels",
        num_labels
    );

    // Load weights from safetensors
    info!("Loading weights from: {}", model_path);
    let record =
        load_ner_model_from_safetensors::<B>(Path::new(&model_path), &ner_config, &device)?;

    // Initialize model and load record
    let model = model_config
        .init_for_token_classification::<B>(num_labels, &device)
        .load_record(record);
    info!("Model loaded successfully");

    // Load tokenizer
    let default_dict_path = format!("{}/system.dic.zst", model_dir);
    let dict_path: Option<String> = std::env::var("MECAB_DICT_PATH").ok().or_else(|| {
        if std::path::Path::new(&default_dict_path).exists() {
            Some(default_dict_path.clone())
        } else {
            None
        }
    });

    info!("Loading tokenizer from: {}", vocab_path);
    if let Some(ref dp) = dict_path {
        info!("Using MeCab dictionary: {}", dp);
    } else {
        warn!("No MeCab dictionary found. Using character-only tokenization.");
    }
    let tokenizer_obj = JapaneseBertTokenizer::new(&vocab_path, dict_path.as_deref(), 512)?;
    info!("Tokenizer loaded successfully");

    // Build NER pipeline
    let pipeline = NerPipeline::new(
        model,
        tokenizer_obj,
        ner_config.id2label.clone(),
        ner_config.pad_token_id,
        device,
    );

    // Build app state
    let use_dynamic_batch = std::env::var("DYNAMIC_BATCH")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false);

    let state = if use_dynamic_batch {
        let config = BatcherConfig {
            max_batch_size: std::env::var("BATCH_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8),
            max_wait_ms: std::env::var("BATCH_WAIT_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10),
        };
        info!(
            "Dynamic batching ENABLED: max_batch_size={}, max_wait_ms={}",
            config.max_batch_size, config.max_wait_ms
        );
        let batcher = DynamicBatcher::new(Arc::new(Mutex::new(pipeline)), config);
        Arc::new(AppState {
            pipeline: None,
            batcher: Some(Arc::new(batcher)),
        })
    } else {
        info!("Dynamic batching DISABLED (set DYNAMIC_BATCH=1 to enable)");
        Arc::new(AppState {
            pipeline: Some(Arc::new(Mutex::new(pipeline))),
            batcher: None,
        })
    };

    let app = create_router(state);

    let addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:3000".to_string());
    info!("Starting server on {}", addr);

    // Build and run tokio runtime
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;
        Ok::<(), anyhow::Error>(())
    })?;

    Ok(())
}

fn main() -> Result<()> {
    env_logger::init_from_env(Env::default().default_filter_or("info"));

    #[cfg(feature = "wgpu")]
    {
        use burn::backend::wgpu::{Wgpu, WgpuDevice};
        info!("Using WGPU backend");
        let device = WgpuDevice::default();
        launch::<Wgpu>(device)?;
    }

    #[cfg(all(feature = "ndarray", not(feature = "wgpu"), not(feature = "cuda")))]
    {
        use burn::backend::ndarray::{NdArray, NdArrayDevice};
        info!("Using NdArray backend (CPU)");
        launch::<NdArray>(NdArrayDevice::Cpu)?;
    }

    #[cfg(all(feature = "cuda", not(feature = "wgpu")))]
    {
        use burn::backend::{cuda::CudaDevice, Cuda};
        info!("Using CUDA backend");
        launch::<Cuda>(CudaDevice::default())?;
    }

    Ok(())
}
