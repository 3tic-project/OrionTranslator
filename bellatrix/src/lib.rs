pub mod detector;
pub mod embedding;
pub mod llm;
pub mod loader;
pub mod model;
pub mod ner;
pub mod tokenizer;

use anyhow::Result;
use burn::tensor::backend::Backend;
use log::info;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::loader::{build_model_config, load_ner_config, load_ner_model_from_safetensors};
use crate::ner::NerPipeline;
use crate::tokenizer::JapaneseBertTokenizer;
use burn::module::Module;

/// Progress callback for glossary generation
pub type GlossaryProgressCallback = Option<Arc<dyn Fn(GlossaryProgressEvent) + Send + Sync>>;

/// Progress events during glossary generation
#[derive(Debug, Clone)]
pub enum GlossaryProgressEvent {
    /// Stage started
    StageStarted { stage: String, detail: String },
    /// NER batch progress
    NerProgress { completed: usize, total: usize },
    /// LLM translation progress
    LlmProgress { completed: usize, total: usize },
    /// Log message
    Log { message: String },
    /// Completed successfully
    Completed {
        output_path: String,
        entry_count: usize,
    },
    /// Error occurred
    Error { message: String },
}

fn emit(cb: &GlossaryProgressCallback, event: GlossaryProgressEvent) {
    if let Some(f) = cb {
        f(event);
    }
}

/// Load the NER pipeline from a model directory.
/// Returns an `Arc<Mutex<NerPipeline>>` that can be shared across tasks.
pub fn load_ner_pipeline<B: Backend + 'static>(
    model_dir: &str,
    device: B::Device,
) -> Result<Arc<Mutex<NerPipeline<B>>>> {
    let model_path = format!("{}/model.safetensors", model_dir);
    let config_path = format!("{}/config.json", model_dir);
    let vocab_path = format!("{}/vocab.txt", model_dir);

    // Verify all required files exist, with clear error messages
    let model_dir_abs =
        std::fs::canonicalize(model_dir).unwrap_or_else(|_| std::path::PathBuf::from(model_dir));
    for (name, path) in [
        ("config.json", &config_path),
        ("model.safetensors", &model_path),
        ("vocab.txt", &vocab_path),
    ] {
        if !Path::new(path).exists() {
            anyhow::bail!(
                "NER模型文件不存在: {}\n  查找路径: {}\n  模型目录: {} ({})",
                name,
                path,
                model_dir,
                model_dir_abs.display()
            );
        }
    }

    info!("Loading NER config from: {}", config_path);
    let ner_config = load_ner_config(Path::new(&config_path))
        .map_err(|e| anyhow::anyhow!("加载NER配置失败 ({}): {}", config_path, e))?;
    info!(
        "Model config: hidden_size={}, layers={}, labels={:?}",
        ner_config.hidden_size, ner_config.num_hidden_layers, ner_config.id2label
    );

    let model_config = build_model_config(&ner_config);
    let num_labels = ner_config.num_labels();

    info!("Loading weights from: {}", model_path);
    let record = load_ner_model_from_safetensors::<B>(Path::new(&model_path), &ner_config, &device)
        .map_err(|e| anyhow::anyhow!("加载NER模型权重失败 ({}): {}", model_path, e))?;

    let model = model_config
        .init_for_token_classification::<B>(num_labels, &device)
        .load_record(record);
    info!("NER model loaded successfully");

    let default_dict_path = format!("{}/system.dic.zst", model_dir);
    let dict_path: Option<String> = std::env::var("MECAB_DICT_PATH").ok().or_else(|| {
        if Path::new(&default_dict_path).exists() {
            Some(default_dict_path.clone())
        } else {
            None
        }
    });

    info!("Loading tokenizer from: {}", vocab_path);
    let tokenizer_obj = JapaneseBertTokenizer::new(&vocab_path, dict_path.as_deref(), 512)
        .map_err(|e| {
            anyhow::anyhow!(
                "加载分词器失败 (vocab: {}, dict: {:?}): {}",
                vocab_path,
                dict_path,
                e
            )
        })?;
    info!("NER tokenizer loaded successfully");

    let pipeline = NerPipeline::new(
        model,
        tokenizer_obj,
        ner_config.id2label.clone(),
        ner_config.pad_token_id,
        device,
    );

    Ok(Arc::new(Mutex::new(pipeline)))
}

/// Configuration for glossary generation
#[derive(Debug, Clone)]
pub struct GlossaryConfig {
    /// Pre-extracted text lines from the input file
    pub lines: Vec<String>,
    /// NER model directory
    pub model_dir: String,
    /// NER batch size
    pub ner_batch_size: usize,
    /// Minimum character occurrence count
    pub min_count: usize,
    /// LLM API URL
    pub llm_url: String,
    /// LLM API key
    pub llm_api_key: String,
    /// LLM model name
    pub llm_model: String,
    /// LLM concurrent workers
    pub llm_workers: usize,
    /// Output glossary path
    pub output_path: std::path::PathBuf,
    /// Skip LLM translation (for Orion models, only run NER)
    pub skip_llm_translation: bool,
}

impl GlossaryConfig {
    /// 校验会触发 panic/永久等待的参数（`step_by(0)`、0-permit semaphore）。
    pub fn validate(&self) -> Result<()> {
        if self.ner_batch_size == 0 {
            anyhow::bail!("ner_batch_size 必须 >= 1（当前为 0，会导致 step_by panic）");
        }
        if !self.skip_llm_translation && self.llm_workers == 0 {
            anyhow::bail!("llm_workers 必须 >= 1（当前为 0，会导致信号量永久等待）");
        }
        if self.min_count == 0 {
            anyhow::bail!("min_count 必须 >= 1");
        }
        Ok(())
    }
}

/// Run the full glossary generation pipeline:
/// 1. Load NER model (wgpu)
/// 2. Run NER to detect characters from provided text lines
/// 3. Use LLM to translate names and generate glossary
/// 4. Save glossary JSON
///
/// Callers must extract text lines from EPUB/TXT before calling this.
/// Returns the path to the saved glossary file.
#[cfg(feature = "wgpu")]
pub async fn generate_glossary(
    config: GlossaryConfig,
    progress: GlossaryProgressCallback,
) -> Result<std::path::PathBuf> {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    config.validate()?;

    emit(
        &progress,
        GlossaryProgressEvent::StageStarted {
            stage: "初始化".to_string(),
            detail: "加载NER模型 (WGPU)...".to_string(),
        },
    );

    let device = WgpuDevice::default();
    let pipeline = load_ner_pipeline::<Wgpu>(&config.model_dir, device)?;

    emit(
        &progress,
        GlossaryProgressEvent::Log {
            message: "NER模型加载完成".to_string(),
        },
    );

    let lines = &config.lines;
    if lines.is_empty() {
        anyhow::bail!("输入文本内容为空");
    }

    emit(
        &progress,
        GlossaryProgressEvent::Log {
            message: format!("共 {} 行文本", lines.len()),
        },
    );

    // Run NER detection
    emit(
        &progress,
        GlossaryProgressEvent::StageStarted {
            stage: "实体识别".to_string(),
            detail: format!("处理 {} 行文本...", lines.len()),
        },
    );

    // `pipeline` (Arc<Mutex<NerPipeline>>) is moved into detect_characters_embedded.
    // When that function returns, the Arc's reference count drops to zero and
    // the model weights / GPU buffers are freed before LLM translation starts.
    let characters = detector::detect_characters_embedded(
        lines,
        pipeline,
        config.ner_batch_size,
        config.min_count,
        progress.clone(),
    )
    .await?;

    // NER pipeline Arc was consumed above — GPU memory is released here.
    emit(
        &progress,
        GlossaryProgressEvent::Log {
            message: "NER模型已卸载，GPU内存已释放".to_string(),
        },
    );

    if characters.is_empty() {
        anyhow::bail!("未识别到出现≥{}次的人物", config.min_count);
    }

    emit(
        &progress,
        GlossaryProgressEvent::Log {
            message: format!("识别到 {} 个人物", characters.len()),
        },
    );

    // Generate translations (or raw entries for Orion models)
    let translations = if config.skip_llm_translation {
        emit(
            &progress,
            GlossaryProgressEvent::Log {
                message: "Orion模型模式：跳过LLM译名，生成人物候选术语表（dst 为空，翻译时作实体提示）"
                    .to_string(),
            },
        );
        // Create raw entries with empty dst/info for Orion models
        characters
            .into_iter()
            .map(|(name, _info)| llm::TranslationEntry {
                src: name,
                dst: String::new(),
                info: String::new(),
            })
            .collect()
    } else {
        // LLM translation for generic models
        emit(
            &progress,
            GlossaryProgressEvent::StageStarted {
                stage: "术语翻译".to_string(),
                detail: format!("使用LLM翻译 {} 个人物...", characters.len()),
            },
        );

        let llm_client =
            llm::LlmClient::new(&config.llm_url, &config.llm_api_key, &config.llm_model);
        let translations = llm_client
            .translate_all(&characters, config.llm_workers, progress.clone())
            .await;

        if translations.is_empty() {
            anyhow::bail!("术语翻译结果为空");
        }
        translations
    };

    // Save glossary
    let output_path = &config.output_path;

    let json = serde_json::to_string_pretty(&translations)?;
    std::fs::write(output_path, &json)?;

    emit(
        &progress,
        GlossaryProgressEvent::Completed {
            output_path: output_path.display().to_string(),
            entry_count: translations.len(),
        },
    );

    Ok(output_path.clone())
}

/// Check if a model name is a "generic" (non-Orion) model
pub fn is_generic_model(model_name: &str) -> bool {
    !model_name.to_lowercase().contains("orion")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn sample_config() -> GlossaryConfig {
        GlossaryConfig {
            lines: vec!["テスト".into()],
            model_dir: ".".into(),
            ner_batch_size: 8,
            min_count: 1,
            llm_url: "http://127.0.0.1/v1".into(),
            llm_api_key: String::new(),
            llm_model: "deepseek-v4-flash".into(),
            llm_workers: 4,
            output_path: PathBuf::from("/tmp/glossary.json"),
            skip_llm_translation: false,
        }
    }

    #[test]
    fn validate_rejects_zero_batch_and_workers() {
        let mut cfg = sample_config();
        cfg.ner_batch_size = 0;
        assert!(cfg.validate().is_err());

        let mut cfg = sample_config();
        cfg.llm_workers = 0;
        assert!(cfg.validate().is_err());

        let mut cfg = sample_config();
        cfg.llm_workers = 0;
        cfg.skip_llm_translation = true;
        // Orion 跳过 LLM 时 workers 可为 0
        assert!(cfg.validate().is_ok());

        assert!(sample_config().validate().is_ok());
    }

    #[test]
    fn is_generic_model_detects_orion_substring() {
        assert!(!is_generic_model("Orion-Qwen3-1.7B-SFT"));
        assert!(is_generic_model("deepseek-v4-flash"));
    }
}
