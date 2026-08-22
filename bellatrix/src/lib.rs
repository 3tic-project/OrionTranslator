#![recursion_limit = "512"]

pub mod detector;
pub mod llm;

/// Compatibility re-exports for callers that used `bellatrix::ner` types.
pub mod ner {
    pub use modernbert_ner::{NerEntity, NerResult};
}

use anyhow::Result;
use log::info;
use modernbert_ner::{CpuNerPipeline, InferOptions, NerResult};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;
use std::time::SystemTime;

const NER_MAX_LENGTH: usize = 256;
const AUTO_BENCH_LINES: usize = 128;
static AUTO_BACKEND_CACHE: OnceLock<Mutex<HashMap<String, NerBackend>>> = OnceLock::new();

/// Runtime backend used by the embedded ModernBERT NER pipeline.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NerBackend {
    Cpu,
    Gpu,
    #[default]
    Auto,
}

impl NerBackend {
    pub const ALL: [Self; 3] = [Self::Cpu, Self::Gpu, Self::Auto];

    pub fn index(self) -> usize {
        match self {
            Self::Cpu => 0,
            Self::Gpu => 1,
            Self::Auto => 2,
        }
    }

    pub fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Cpu,
            1 => Self::Gpu,
            _ => Self::Auto,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Gpu => "GPU",
            Self::Auto => "Auto",
        }
    }
}

pub(crate) enum LoadedNerPipeline {
    Cpu(CpuNerPipeline),
    #[cfg(feature = "wgpu")]
    Gpu(modernbert_ner::NerPipeline<burn::backend::Wgpu>),
}

impl LoadedNerPipeline {
    /// Run the same length-aware packing and scheduling path used by the
    /// standalone modernbert-ner CLI. CPU packs are dynamically distributed
    /// across workers; GPU packs remain sequential on one device queue.
    pub(crate) fn predict_document(
        &self,
        texts: &[String],
        progress: Option<&(dyn Fn(usize, usize) + Sync)>,
    ) -> Result<Vec<NerResult>> {
        match self {
            Self::Cpu(pipeline) => predict_document_cpu(pipeline, texts, progress),
            #[cfg(feature = "wgpu")]
            Self::Gpu(pipeline) => catch_unwind(AssertUnwindSafe(|| {
                let options = &pipeline.options;
                let packs = modernbert_ner::pack_texts(
                    texts,
                    options.max_sentences,
                    options.max_tokens,
                    options.sort_by_length,
                );
                let total = packs.len();
                let mut pack_results = Vec::with_capacity(total);
                for (index, pack) in packs.iter().enumerate() {
                    let refs: Vec<&str> = pack.texts.iter().map(String::as_str).collect();
                    pack_results.push(pipeline.predict_batch(&refs)?);
                    if let Some(report) = progress {
                        report(index + 1, total);
                    }
                }
                Ok::<_, anyhow::Error>(modernbert_ner::pack::unsort_results(
                    texts.len(),
                    &packs,
                    pack_results,
                ))
            }))
            .map_err(|payload| {
                let detail = payload
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("WGPU 推理发生未知 panic");
                anyhow::anyhow!("GPU NER 推理失败: {detail}")
            })?,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Cpu(_) => "CPU",
            #[cfg(feature = "wgpu")]
            Self::Gpu(_) => "GPU (WGPU)",
        }
    }

    fn scheduling_summary(&self) -> String {
        let (options, workers) = match self {
            Self::Cpu(pipeline) => (&pipeline.options, cpu_worker_limit()),
            #[cfg(feature = "wgpu")]
            Self::Gpu(pipeline) => (&pipeline.options, 1),
        };
        format!(
            "NER 调度：{}，batch 上限 {}，token 预算 {}，长度排序 {}，worker {}",
            self.label(),
            options.max_sentences,
            options.max_tokens,
            if options.sort_by_length {
                "开启"
            } else {
                "关闭"
            },
            workers,
        )
    }
}

fn cpu_worker_limit() -> usize {
    std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(4)
        .max(1)
}

fn predict_document_cpu(
    pipeline: &CpuNerPipeline,
    texts: &[String],
    progress: Option<&(dyn Fn(usize, usize) + Sync)>,
) -> Result<Vec<NerResult>> {
    let options = &pipeline.options;
    let packs = modernbert_ner::pack_texts(
        texts,
        options.max_sentences,
        options.max_tokens,
        options.sort_by_length,
    );
    if packs.is_empty() {
        return Ok(Vec::new());
    }

    let jobs = cpu_worker_limit().min(packs.len());
    let workers: Vec<CpuNerPipeline> = (0..jobs).map(|_| pipeline.clone()).collect();

    // Longest-processing-time order avoids leaving one expensive pack as the
    // tail of a worker. Results are restored to document order below.
    let mut order: Vec<usize> = (0..packs.len()).collect();
    order.sort_by_key(|&index| {
        std::cmp::Reverse(
            packs[index]
                .texts
                .iter()
                .map(|text| text.chars().count())
                .sum::<usize>(),
        )
    });

    let cursor = AtomicUsize::new(0);
    let completed = Mutex::new(0usize);
    let total = packs.len();
    let partials: Vec<Vec<(usize, Vec<NerResult>)>> = workers
        .into_par_iter()
        .map(|worker| -> Result<Vec<(usize, Vec<NerResult>)>> {
            let mut local = Vec::new();
            loop {
                let slot = cursor.fetch_add(1, Ordering::Relaxed);
                if slot >= total {
                    break;
                }
                let pack_index = order[slot];
                let pack = &packs[pack_index];
                let refs: Vec<&str> = pack.texts.iter().map(String::as_str).collect();
                local.push((pack_index, worker.predict_batch(&refs)?));
                if let Some(report) = progress {
                    // Serialize the counter and callback so concurrent workers
                    // cannot publish progress in decreasing order.
                    let mut done = completed.lock().expect("NER progress mutex poisoned");
                    *done += 1;
                    report(*done, total);
                }
            }
            Ok(local)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut by_pack: Vec<Option<Vec<NerResult>>> = (0..packs.len()).map(|_| None).collect();
    for partial in partials {
        for (pack_index, results) in partial {
            by_pack[pack_index] = Some(results);
        }
    }
    let pack_results = by_pack
        .into_iter()
        .map(|results| results.expect("missing NER pack result"))
        .collect();
    Ok(modernbert_ner::pack::unsort_results(
        texts.len(),
        &packs,
        pack_results,
    ))
}

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

fn infer_options(batch_size: usize, gpu: bool) -> InferOptions {
    let default_batch = if gpu { 128 } else { 24 };
    InferOptions {
        max_sentences: if batch_size == 0 {
            default_batch
        } else {
            batch_size
        },
        max_tokens: if gpu { 32_768 } else { 1_536 },
        sort_by_length: true,
        skip_scores: false,
    }
}

fn validate_model_dir(model_dir: &Path) -> Result<()> {
    let model_dir_abs =
        std::fs::canonicalize(model_dir).unwrap_or_else(|_| model_dir.to_path_buf());
    for name in ["config.json", "model.safetensors", "tokenizer.json"] {
        let path = model_dir.join(name);
        if !path.is_file() {
            anyhow::bail!(
                "ModernBERT NER模型文件不存在: {}\n  查找路径: {}\n  模型目录: {}",
                name,
                path.display(),
                model_dir_abs.display()
            );
        }
    }
    Ok(())
}

fn load_cpu_pipeline(model_dir: &Path, batch_size: usize) -> Result<LoadedNerPipeline> {
    let pipeline = modernbert_ner::load_pipeline_cpu(model_dir, NER_MAX_LENGTH)?
        .with_options(infer_options(batch_size, false));
    Ok(LoadedNerPipeline::Cpu(pipeline))
}

#[cfg(feature = "wgpu")]
fn load_gpu_pipeline(model_dir: &Path, batch_size: usize) -> Result<LoadedNerPipeline> {
    let loaded = catch_unwind(AssertUnwindSafe(|| {
        modernbert_ner::load_pipeline_wgpu(model_dir, NER_MAX_LENGTH)
            .map(|pipeline| pipeline.with_options(infer_options(batch_size, true)))
    }))
    .map_err(|payload| {
        let detail = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("WGPU 初始化发生未知 panic");
        anyhow::anyhow!("GPU 后端初始化失败: {detail}")
    })??;

    Ok(LoadedNerPipeline::Gpu(loaded))
}

#[cfg(not(feature = "wgpu"))]
fn load_gpu_pipeline(_model_dir: &Path, _batch_size: usize) -> Result<LoadedNerPipeline> {
    anyhow::bail!("当前构建未启用 WGPU，无法使用 GPU NER 后端")
}

fn benchmark_sample(lines: &[String]) -> Vec<String> {
    let valid: Vec<&String> = lines
        .iter()
        .filter(|line| line.trim().chars().count() >= 2)
        .collect();
    if valid.len() <= AUTO_BENCH_LINES {
        return valid.into_iter().cloned().collect();
    }

    let stride = valid.len() as f64 / AUTO_BENCH_LINES as f64;
    (0..AUTO_BENCH_LINES)
        .map(|index| valid[((index as f64 * stride) as usize).min(valid.len() - 1)].clone())
        .collect()
}

fn auto_backend_cache_key(model_dir: &Path, batch_size: usize) -> String {
    let canonical = fs::canonicalize(model_dir).unwrap_or_else(|_| model_dir.to_path_buf());
    let weights = model_dir.join("model.safetensors");
    let metadata = fs::metadata(weights).ok();
    let size = metadata.as_ref().map(|value| value.len()).unwrap_or(0);
    let modified = metadata
        .and_then(|value| value.modified().ok())
        .and_then(|value| value.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|value| value.as_secs())
        .unwrap_or(0);
    format!("{}:{size}:{modified}:{batch_size}", canonical.display())
}

fn cached_auto_backend(key: &str) -> Option<NerBackend> {
    AUTO_BACKEND_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .ok()
        .and_then(|cache| cache.get(key).copied())
}

fn remember_auto_backend(key: String, backend: NerBackend) {
    if let Ok(mut cache) = AUTO_BACKEND_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
    {
        cache.insert(key, backend);
    }
}

fn run_benchmark(pipeline: &LoadedNerPipeline, sample: &[String]) -> Result<f64> {
    if sample.is_empty() {
        anyhow::bail!("没有可用于 NER 后端基准测试的有效文本");
    }

    // Warm exactly the shapes that will be timed. In particular, Burn/WGPU
    // autotunes per shape, so a smaller warmup batch would charge compilation
    // work only to the GPU's timed pass.
    pipeline.predict_document(sample, None)?;

    let chars: usize = sample.iter().map(|line| line.chars().count()).sum();
    let started = Instant::now();
    pipeline.predict_document(sample, None)?;
    Ok(chars as f64 / started.elapsed().as_secs_f64().max(1e-9))
}

fn load_ner_pipeline(
    model_dir: &Path,
    backend: NerBackend,
    lines: &[String],
    batch_size: usize,
    progress: &GlossaryProgressCallback,
) -> Result<LoadedNerPipeline> {
    validate_model_dir(model_dir)?;

    match backend {
        NerBackend::Cpu => {
            emit(
                progress,
                GlossaryProgressEvent::StageStarted {
                    stage: "初始化".to_string(),
                    detail: "加载 ModernBERT NER（CPU）...".to_string(),
                },
            );
            load_cpu_pipeline(model_dir, batch_size)
        }
        NerBackend::Gpu => {
            emit(
                progress,
                GlossaryProgressEvent::StageStarted {
                    stage: "初始化".to_string(),
                    detail: "加载 ModernBERT NER（GPU/WGPU）...".to_string(),
                },
            );
            load_gpu_pipeline(model_dir, batch_size)
        }
        NerBackend::Auto => {
            let cache_key = auto_backend_cache_key(model_dir, batch_size);
            if let Some(cached) = cached_auto_backend(&cache_key) {
                let loaded = match cached {
                    NerBackend::Cpu => load_cpu_pipeline(model_dir, batch_size),
                    NerBackend::Gpu => load_gpu_pipeline(model_dir, batch_size),
                    NerBackend::Auto => unreachable!("Auto is never cached as a concrete backend"),
                };
                match loaded {
                    Ok(pipeline) => {
                        emit(
                            progress,
                            GlossaryProgressEvent::Log {
                                message: format!(
                                    "Auto 复用本次应用会话的 {} 基准结果（模型或batch变化时自动重测）",
                                    cached.label()
                                ),
                            },
                        );
                        return Ok(pipeline);
                    }
                    Err(error) => emit(
                        progress,
                        GlossaryProgressEvent::Log {
                            message: format!("Auto 缓存后端加载失败，重新基准：{error:#}"),
                        },
                    ),
                }
            }

            let sample = benchmark_sample(lines);
            let sample_chars: usize = sample.iter().map(|line| line.chars().count()).sum();
            emit(
                progress,
                GlossaryProgressEvent::StageStarted {
                    stage: "后端基准".to_string(),
                    detail: format!(
                        "抽样 {} 行 / {} 字，预热后比较 CPU 与 GPU...",
                        sample.len(),
                        sample_chars
                    ),
                },
            );

            let cpu = load_cpu_pipeline(model_dir, batch_size)?;
            let cpu_rate = run_benchmark(&cpu, &sample)?;
            emit(
                progress,
                GlossaryProgressEvent::Log {
                    message: format!("Auto 基准：CPU {:.0} 字/秒", cpu_rate),
                },
            );

            let gpu_attempt = catch_unwind(AssertUnwindSafe(|| {
                let gpu = load_gpu_pipeline(model_dir, batch_size)?;
                let rate = run_benchmark(&gpu, &sample)?;
                Ok::<_, anyhow::Error>((gpu, rate))
            }));

            match gpu_attempt {
                Ok(Ok((gpu, gpu_rate))) => {
                    emit(
                        progress,
                        GlossaryProgressEvent::Log {
                            message: format!("Auto 基准：GPU {:.0} 字/秒", gpu_rate),
                        },
                    );
                    if gpu_rate > cpu_rate {
                        emit(
                            progress,
                            GlossaryProgressEvent::Log {
                                message: "Auto 已选择 GPU（当前文档抽样更快）".to_string(),
                            },
                        );
                        remember_auto_backend(cache_key, NerBackend::Gpu);
                        Ok(gpu)
                    } else {
                        emit(
                            progress,
                            GlossaryProgressEvent::Log {
                                message: "Auto 已选择 CPU（当前文档抽样更快）".to_string(),
                            },
                        );
                        remember_auto_backend(cache_key, NerBackend::Cpu);
                        Ok(cpu)
                    }
                }
                Ok(Err(error)) => {
                    emit(
                        progress,
                        GlossaryProgressEvent::Log {
                            message: format!("Auto：GPU 不可用，回退 CPU：{error:#}"),
                        },
                    );
                    remember_auto_backend(cache_key, NerBackend::Cpu);
                    Ok(cpu)
                }
                Err(_) => {
                    emit(
                        progress,
                        GlossaryProgressEvent::Log {
                            message: "Auto：GPU 后端发生异常，已安全回退 CPU".to_string(),
                        },
                    );
                    remember_auto_backend(cache_key, NerBackend::Cpu);
                    Ok(cpu)
                }
            }
        }
    }
}

/// Configuration for glossary generation
#[derive(Debug, Clone)]
pub struct GlossaryConfig {
    /// Pre-extracted text lines from the input file
    pub lines: Vec<String>,
    /// EPUB ruby 证据；TXT 或旧调用方可传空数组。
    pub ruby_annotations: Vec<betelgeuse::RubyAnnotation>,
    /// NER model directory
    pub model_dir: String,
    /// NER batch size. 0 selects backend-tuned defaults (CPU 24 / GPU 128).
    pub ner_batch_size: usize,
    /// CPU, GPU, or a short benchmark before full inference.
    pub ner_backend: NerBackend,
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

pub const RUBY_REVIEW_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RubyAliasClassification {
    #[default]
    Unclassified,
    PhoneticReading,
    OrthographicAlias,
    NicknameCue,
    SemanticRuby,
    OrdinaryReading,
    WordplayToken,
}

impl RubyAliasClassification {
    pub fn can_be_translation_alias(self) -> bool {
        matches!(
            self,
            Self::PhoneticReading | Self::OrthographicAlias | Self::NicknameCue
        )
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "unclassified" => Ok(Self::Unclassified),
            "phonetic_reading" => Ok(Self::PhoneticReading),
            "orthographic_alias" => Ok(Self::OrthographicAlias),
            "nickname_cue" => Ok(Self::NicknameCue),
            "semantic_ruby" => Ok(Self::SemanticRuby),
            "ordinary_reading" => Ok(Self::OrdinaryReading),
            "wordplay_token" => Ok(Self::WordplayToken),
            other => anyhow::bail!("未知 ruby 分类: {other}"),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CandidateDecision {
    #[default]
    Pending,
    Confirmed,
    Rejected,
}

impl CandidateDecision {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "pending" => Ok(Self::Pending),
            "confirmed" => Ok(Self::Confirmed),
            "rejected" => Ok(Self::Rejected),
            other => anyhow::bail!("未知审核决定: {other}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RubySurfaceEvidence {
    pub surface: String,
    pub mention_count: usize,
    pub exact_reading: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RubyAliasReviewEntry {
    /// Stable across re-scans of the same base/reading pair.
    #[serde(default)]
    pub candidate_id: String,
    pub base: String,
    pub reading: String,
    pub reading_variants: Vec<String>,
    pub surface_evidence: Vec<RubySurfaceEvidence>,
    pub source_paths: Vec<String>,
    pub contexts: Vec<String>,
    pub independent_mentions: usize,
    pub base_dst: Option<String>,
    pub existing_reading_dst: Option<String>,
    /// Machine/user classification. Unclassified candidates never become constraints.
    #[serde(default)]
    pub classification: RubyAliasClassification,
    /// Conservative machine suggestion; never activates an alias by itself.
    #[serde(default)]
    pub suggested_classification: RubyAliasClassification,
    #[serde(default)]
    pub suggestion_confidence: u8,
    #[serde(default)]
    pub suggestion_reason: String,
    /// Explicit review decision, preserved when this sidecar is regenerated.
    #[serde(default)]
    pub decision: CandidateDecision,
    /// Target used by confirmed aliases. Falls back to base_dst only after validation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision_note: Option<String>,
    #[serde(default)]
    pub revision: u64,
    pub status: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RubyAliasReviewDocument {
    pub schema_version: u32,
    pub candidates: Vec<RubyAliasReviewEntry>,
}

impl RubyAliasReviewDocument {
    pub fn new(candidates: Vec<RubyAliasReviewEntry>) -> Self {
        Self {
            schema_version: RUBY_REVIEW_SCHEMA_VERSION,
            candidates,
        }
    }
}

impl GlossaryConfig {
    /// 校验会触发永久等待或无意义结果的参数。
    pub fn validate(&self) -> Result<()> {
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
/// 1. Load the ModernBERT NER model on the requested backend
/// 2. Run NER to detect characters from provided text lines
/// 3. Use LLM to translate names and generate glossary
/// 4. Save glossary JSON
///
/// Callers must extract text lines from EPUB/TXT before calling this.
/// Returns the path to the saved glossary file.
pub async fn generate_glossary(
    config: GlossaryConfig,
    progress: GlossaryProgressCallback,
) -> Result<std::path::PathBuf> {
    config.validate()?;

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

    let pipeline = load_ner_pipeline(
        Path::new(&config.model_dir),
        config.ner_backend,
        lines,
        config.ner_batch_size,
        &progress,
    )?;

    info!("ModernBERT NER loaded on {}", pipeline.label());
    emit(
        &progress,
        GlossaryProgressEvent::Log {
            message: format!("ModernBERT NER 模型加载完成，使用 {}", pipeline.label()),
        },
    );
    emit(
        &progress,
        GlossaryProgressEvent::Log {
            message: pipeline.scheduling_summary(),
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

    let characters =
        detector::detect_characters_embedded(lines, &pipeline, config.min_count, progress.clone())
            .await?;

    drop(pipeline);
    emit(
        &progress,
        GlossaryProgressEvent::Log {
            message: "NER模型已卸载，推理内存已释放".to_string(),
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
    let translation_report = if config.skip_llm_translation {
        emit(
            &progress,
            GlossaryProgressEvent::Log {
                message:
                    "Orion模型模式：跳过LLM译名，生成人物候选术语表（dst 为空，翻译时作实体提示）"
                        .to_string(),
            },
        );
        // Create raw entries with empty dst/info for Orion models
        llm::GlossaryTranslationReport {
            entries: characters
                .into_keys()
                .map(|name| llm::TranslationEntry {
                    src: name,
                    dst: String::new(),
                    info: String::new(),
                })
                .collect(),
            ..llm::GlossaryTranslationReport::default()
        }
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
        llm_client
            .translate_all_detailed(&characters, config.llm_workers, progress.clone())
            .await
    };
    let llm::GlossaryTranslationReport {
        entries: translations,
        issues: generation_issues,
        review_changes,
        review_batches,
        review_failures,
    } = translation_report;

    // Ruby 证据只生成审核清单，不在缺乏分类/确认时自动提升为强制术语。
    // 这避免把 `桃谷<rt>ライバル</rt>` 一类语义 ruby 错绑成人名读音。
    // Save glossary
    let output_path = &config.output_path;

    let json = serde_json::to_string_pretty(&translations)?;
    atomic_write(output_path, json.as_bytes())?;

    if !config.skip_llm_translation {
        let report_path = glossary_generation_report_path(output_path);
        let unresolved = generation_issues
            .iter()
            .filter(|issue| issue.kind == llm::GlossaryIssueKind::Unresolved)
            .count();
        let rejected = generation_issues
            .iter()
            .filter(|issue| issue.kind == llm::GlossaryIssueKind::Rejected)
            .count();
        let review_change_count = review_changes.len();
        let review_rejected = review_changes
            .iter()
            .filter(|change| change.after_dst.is_none())
            .count();
        let report = serde_json::json!({
            "schema_version": 2,
            "resolved_entries": translations.len(),
            "unresolved_clusters": unresolved,
            "rejected_clusters": rejected,
            "final_review_batches": review_batches,
            "final_review_failures": review_failures,
            "final_review_rejected": review_rejected,
            "final_review_changes": review_changes,
            "issues": generation_issues,
        });
        atomic_write(
            &report_path,
            serde_json::to_string_pretty(&report)?.as_bytes(),
        )?;
        emit(
            &progress,
            GlossaryProgressEvent::Log {
                message: format!(
                    "术语生成报告: {}（resolved={}，unresolved={}，rejected={}，review_changes={}）",
                    report_path.display(),
                    translations.len(),
                    unresolved,
                    rejected,
                    review_change_count
                ),
            },
        );
        if translations.is_empty() {
            anyhow::bail!("术语翻译结果为空；诊断已保存到 {}", report_path.display());
        }
    }

    if !config.ruby_annotations.is_empty() {
        let review_path = ruby_review_path(output_path);
        let ruby_review = refresh_ruby_alias_review(
            &review_path,
            &config.ruby_annotations,
            &config.lines,
            &translations,
        )?;
        let actionable = ruby_review
            .candidates
            .iter()
            .filter(|entry| is_actionable_review_status(&entry.status))
            .count();
        emit(
            &progress,
            GlossaryProgressEvent::Log {
                message: format!(
                    "Ruby别名审核清单: {}（{} 条需审核；未自动提升为强制术语）",
                    review_path.display(),
                    actionable
                ),
            },
        );
    }

    emit(
        &progress,
        GlossaryProgressEvent::Completed {
            output_path: output_path.display().to_string(),
            entry_count: translations.len(),
        },
    );

    Ok(output_path.clone())
}

pub fn ruby_review_path(glossary_path: &Path) -> PathBuf {
    let stem = glossary_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("glossary");
    glossary_path.with_file_name(format!("{stem}.ruby-candidates.json"))
}

pub fn glossary_generation_report_path(glossary_path: &Path) -> PathBuf {
    let stem = glossary_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("glossary");
    glossary_path.with_file_name(format!("{stem}.generation-report.json"))
}

pub fn save_ruby_alias_review(path: &Path, review: &[RubyAliasReviewEntry]) -> Result<()> {
    save_ruby_alias_review_document(path, &RubyAliasReviewDocument::new(review.to_vec()))
}

pub fn save_ruby_alias_review_document(
    path: &Path,
    review: &RubyAliasReviewDocument,
) -> Result<()> {
    if review.schema_version != RUBY_REVIEW_SCHEMA_VERSION {
        anyhow::bail!(
            "不支持的 ruby 审核 schema_version: {}（当前支持 {}）",
            review.schema_version,
            RUBY_REVIEW_SCHEMA_VERSION
        );
    }
    let json = serde_json::to_string_pretty(review)?;
    atomic_write(path, json.as_bytes())
}

pub fn load_ruby_alias_review(path: &Path) -> Result<RubyAliasReviewDocument> {
    let json = fs::read_to_string(path)?;
    let value: serde_json::Value = serde_json::from_str(&json)?;
    let mut document = if value.is_array() {
        RubyAliasReviewDocument::new(serde_json::from_value(value)?)
    } else {
        serde_json::from_value(value)?
    };
    if document.schema_version != RUBY_REVIEW_SCHEMA_VERSION {
        anyhow::bail!(
            "不支持的 ruby 审核 schema_version: {}（当前支持 {}）",
            document.schema_version,
            RUBY_REVIEW_SCHEMA_VERSION
        );
    }
    normalize_candidate_ids(&mut document.candidates);
    Ok(document)
}

pub fn refresh_ruby_alias_review(
    path: &Path,
    annotations: &[betelgeuse::RubyAnnotation],
    lines: &[String],
    translations: &[llm::TranslationEntry],
) -> Result<RubyAliasReviewDocument> {
    let previous = if path.exists() {
        Some(load_ruby_alias_review(path)?)
    } else {
        None
    };
    let mut candidates = build_ruby_alias_review(annotations, lines, translations);
    if let Some(previous) = previous.as_ref() {
        preserve_review_decisions(&mut candidates, &previous.candidates);
    }
    let document = RubyAliasReviewDocument::new(candidates);
    save_ruby_alias_review_document(path, &document)?;
    Ok(document)
}

pub fn update_ruby_alias_decision(
    path: &Path,
    candidate_id: &str,
    classification: RubyAliasClassification,
    decision: CandidateDecision,
    target: Option<String>,
    note: Option<String>,
) -> Result<RubyAliasReviewEntry> {
    let mut document = load_ruby_alias_review(path)?;
    let candidate = document
        .candidates
        .iter_mut()
        .find(|candidate| candidate.candidate_id == candidate_id)
        .ok_or_else(|| anyhow::anyhow!("找不到 ruby 候选: {candidate_id}"))?;

    let target = target
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| candidate.base_dst.clone());
    if decision == CandidateDecision::Confirmed {
        if !classification.can_be_translation_alias() {
            anyhow::bail!(
                "分类 {:?} 不能确认为翻译 alias；只允许 phonetic_reading、orthographic_alias、nickname_cue",
                classification
            );
        }
        if target.as_deref().is_none_or(str::is_empty) {
            anyhow::bail!("确认 ruby alias 必须提供非空 target，且候选没有可回退的 base_dst");
        }
    }

    candidate.classification = classification;
    candidate.decision = decision;
    candidate.target = if decision == CandidateDecision::Confirmed {
        target
    } else {
        None
    };
    candidate.decision_note = note
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    candidate.revision = candidate.revision.saturating_add(1);
    let updated = candidate.clone();
    save_ruby_alias_review_document(path, &document)?;
    Ok(updated)
}

fn preserve_review_decisions(
    current: &mut [RubyAliasReviewEntry],
    previous: &[RubyAliasReviewEntry],
) {
    let decisions: HashMap<&str, &RubyAliasReviewEntry> = previous
        .iter()
        .filter(|entry| !entry.candidate_id.is_empty())
        .map(|entry| (entry.candidate_id.as_str(), entry))
        .collect();
    for candidate in current {
        let Some(previous) = decisions.get(candidate.candidate_id.as_str()) else {
            continue;
        };
        candidate.classification = previous.classification;
        candidate.decision = previous.decision;
        candidate.target = previous.target.clone();
        candidate.decision_note = previous.decision_note.clone();
        candidate.revision = previous.revision;
    }
}

fn normalize_candidate_ids(candidates: &mut [RubyAliasReviewEntry]) {
    for candidate in candidates {
        if candidate.candidate_id.is_empty() {
            candidate.candidate_id = ruby_candidate_id(&candidate.base, &candidate.reading);
        }
    }
}

pub fn ruby_candidate_id(base: &str, reading: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"orion-ruby-candidate-v2\0");
    hasher.update(base.trim().as_bytes());
    hasher.update(b"\0");
    hasher.update(reading.trim().as_bytes());
    format!("ruby-v2:{:x}", hasher.finalize())
}

pub fn build_ruby_alias_review(
    annotations: &[betelgeuse::RubyAnnotation],
    lines: &[String],
    translations: &[llm::TranslationEntry],
) -> Vec<RubyAliasReviewEntry> {
    let mut grouped: HashMap<(String, String), (HashSet<String>, HashSet<String>)> = HashMap::new();
    for annotation in annotations {
        let base = annotation.base.trim();
        let reading = annotation.reading.trim();
        if base.is_empty() || reading.is_empty() {
            continue;
        }
        let (paths, contexts) = grouped
            .entry((base.to_string(), reading.to_string()))
            .or_default();
        if !annotation.source_path.trim().is_empty() {
            paths.insert(annotation.source_path.clone());
        }
        if !annotation.context.trim().is_empty() && contexts.len() < 5 {
            contexts.insert(annotation.context.chars().take(240).collect());
        }
    }

    let mut review = Vec::new();
    for ((base, reading), (paths, contexts)) in grouped {
        let variants = kana_variants(&reading);
        let surface_evidence: Vec<RubySurfaceEvidence> = variants
            .iter()
            .map(|surface| RubySurfaceEvidence {
                surface: surface.clone(),
                mention_count: count_surface_mentions(lines, surface),
                exact_reading: surface == &reading,
            })
            .collect();
        let independent_mentions = surface_evidence
            .iter()
            .map(|evidence| evidence.mention_count)
            .sum();
        let base_entry = find_translation_entry(translations, std::slice::from_ref(&base));
        let reading_entry = variants
            .iter()
            .find_map(|variant| {
                translations
                    .iter()
                    .find(|entry| entry.src.trim() == variant)
            })
            .or_else(|| find_translation_entry(translations, &variants));
        let base_dst = base_entry
            .map(|entry| entry.dst.trim().to_string())
            .filter(|value| !value.is_empty());
        let existing_reading_dst = reading_entry
            .map(|entry| entry.dst.trim().to_string())
            .filter(|value| !value.is_empty());
        let (suggested_classification, suggestion_confidence, suggestion_reason) =
            suggest_ruby_classification(
                &base,
                &reading,
                base_entry.is_some(),
                independent_mentions,
            );

        let (status, reason) = if !is_kana_surface(&reading) || reading.chars().count() < 2 {
            (
                "invalid_or_high_ambiguity",
                "读音不是至少两个字符的纯假名表面，禁止自动提升",
            )
        } else if base_entry.is_none() {
            ("base_not_confirmed", "ruby base 未被 NER/LLM 确认为实体")
        } else if base_dst.is_none() {
            ("base_unresolved", "ruby base 尚无已确认中文译名")
        } else if independent_mentions == 0 {
            (
                "no_independent_mention",
                "读音及其平片假名变体未在 rt 外独立出现",
            )
        } else if let Some(reading_dst) = existing_reading_dst.as_deref() {
            if base_dst.as_deref() == Some(reading_dst) {
                ("confirmed_existing", "读音已有相同目标译名，无需新增")
            } else {
                ("conflict", "ruby base 与读音现有译名冲突，必须人工消歧")
            }
        } else if reading.chars().count() <= 2 {
            (
                "high_ambiguity_review",
                "1–2 个假名可能是普通词或昵称；即使复现也必须结合实体上下文审核",
            )
        } else {
            (
                "review_required",
                "具备实体 base 与 rt 外复现证据；需区分发音、昵称和语义 ruby",
            )
        };

        let mut source_paths: Vec<String> = paths.into_iter().collect();
        source_paths.sort();
        let mut contexts: Vec<String> = contexts.into_iter().collect();
        contexts.sort();
        review.push(RubyAliasReviewEntry {
            candidate_id: ruby_candidate_id(&base, &reading),
            base,
            reading,
            reading_variants: variants,
            surface_evidence,
            source_paths,
            contexts,
            independent_mentions,
            base_dst,
            existing_reading_dst,
            classification: RubyAliasClassification::Unclassified,
            suggested_classification,
            suggestion_confidence,
            suggestion_reason,
            decision: CandidateDecision::Pending,
            target: None,
            decision_note: None,
            revision: 0,
            status: status.to_string(),
            reason: reason.to_string(),
        });
    }

    review.sort_by(|left, right| {
        review_status_rank(&left.status)
            .cmp(&review_status_rank(&right.status))
            .then(right.independent_mentions.cmp(&left.independent_mentions))
            .then(left.base.cmp(&right.base))
            .then(left.reading.cmp(&right.reading))
    });
    review
}

fn suggest_ruby_classification(
    base: &str,
    reading: &str,
    base_confirmed: bool,
    independent_mentions: usize,
) -> (RubyAliasClassification, u8, String) {
    let reading_len = reading.chars().count();
    if !base.chars().any(is_cjk_ideograph) || !is_kana_surface(reading) {
        return (
            RubyAliasClassification::Unclassified,
            0,
            "base/reading 形态不足，无法提出安全分类建议".to_string(),
        );
    }
    if reading_len <= 2 {
        return (
            RubyAliasClassification::Unclassified,
            25,
            "短假名歧义过高，必须结合实体上下文人工分类".to_string(),
        );
    }
    if !base_confirmed {
        return (
            RubyAliasClassification::Unclassified,
            15,
            "ruby base 尚未被术语生成确认为实体，不提出 alias 分类".to_string(),
        );
    }
    if independent_mentions == 0 {
        return (
            RubyAliasClassification::Unclassified,
            30,
            "读音未在 rt 外独立复现，暂不提出 alias 分类".to_string(),
        );
    }
    if reading.chars().all(is_hiragana_char) {
        return (
            RubyAliasClassification::PhoneticReading,
            85,
            "汉字 base 配纯平假名，符合常见发音 ruby；仍需人工确认实体身份".to_string(),
        );
    }
    if reading.chars().all(is_katakana_char) {
        return (
            RubyAliasClassification::Unclassified,
            40,
            "片假名 ruby 可能是发音、昵称或语义标注，禁止仅凭字形自动分类".to_string(),
        );
    }
    (
        RubyAliasClassification::Unclassified,
        15,
        "混合假名结构可能包含特殊表记或文字游戏".to_string(),
    )
}

fn is_cjk_ideograph(character: char) -> bool {
    matches!(character as u32, 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff)
}

fn review_status_rank(status: &str) -> usize {
    match status {
        "conflict" => 0,
        "review_required" => 1,
        "high_ambiguity_review" => 2,
        "confirmed_existing" => 3,
        "base_unresolved" => 4,
        "no_independent_mention" => 5,
        "base_not_confirmed" => 6,
        _ => 7,
    }
}

pub fn is_actionable_review_status(status: &str) -> bool {
    matches!(
        status,
        "conflict" | "review_required" | "high_ambiguity_review"
    )
}

fn find_translation_entry<'a>(
    translations: &'a [llm::TranslationEntry],
    surfaces: &[String],
) -> Option<&'a llm::TranslationEntry> {
    translations
        .iter()
        .find(|entry| surfaces.iter().any(|surface| entry.src.trim() == surface))
        .or_else(|| {
            translations.iter().find(|entry| {
                let canonical = llm::canonical_key(&entry.src);
                surfaces.contains(&canonical)
            })
        })
}

fn kana_variants(reading: &str) -> Vec<String> {
    let mut variants = vec![reading.to_string()];
    let hiragana: String = reading
        .chars()
        .map(|character| {
            let code = character as u32;
            if (0x30a1..=0x30f6).contains(&code) {
                char::from_u32(code - 0x60).unwrap_or(character)
            } else {
                character
            }
        })
        .collect();
    let katakana: String = reading
        .chars()
        .map(|character| {
            let code = character as u32;
            if (0x3041..=0x3096).contains(&code) {
                char::from_u32(code + 0x60).unwrap_or(character)
            } else {
                character
            }
        })
        .collect();
    for variant in [hiragana, katakana] {
        if !variants.contains(&variant) {
            variants.push(variant);
        }
    }
    variants
}

fn is_kana_surface(surface: &str) -> bool {
    !surface.is_empty()
        && surface.chars().all(|character| {
            matches!(character as u32, 0x3040..=0x309f | 0x30a0..=0x30ff | 0xff66..=0xff9f)
                || matches!(character, 'ー' | '・')
        })
}

fn count_surface_mentions(lines: &[String], surface: &str) -> usize {
    let katakana = surface.chars().all(is_katakana_char);
    let hiragana = surface.chars().all(is_hiragana_char);
    lines
        .iter()
        .map(|line| {
            line.match_indices(surface)
                .filter(|(start, _)| {
                    let end = *start + surface.len();
                    let previous = line[..*start].chars().next_back();
                    let next = line[end..].chars().next();
                    if katakana {
                        !previous.is_some_and(is_katakana_char)
                            && !next.is_some_and(is_katakana_char)
                    } else if hiragana {
                        !previous.is_some_and(is_hiragana_char)
                            && !next.is_some_and(is_hiragana_char)
                    } else {
                        true
                    }
                })
                .count()
        })
        .sum()
}

fn is_hiragana_char(character: char) -> bool {
    matches!(character as u32, 0x3040..=0x309f) || character == 'ー'
}

fn is_katakana_char(character: char) -> bool {
    matches!(character as u32, 0x30a0..=0x30ff | 0xff66..=0xff9f)
        || matches!(character, 'ー' | '・')
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("glossary.json");
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut last_error = None;
    for sequence in 0..100u32 {
        let temp_path = parent.join(format!(
            ".{file_name}.tmp.{}.{}",
            std::process::id(),
            sequence
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
        {
            Ok(mut file) => {
                let result = (|| -> Result<()> {
                    file.write_all(contents)?;
                    file.flush()?;
                    file.sync_all()?;
                    drop(file);
                    fs::rename(&temp_path, path)?;
                    Ok(())
                })();
                if result.is_err() {
                    let _ = fs::remove_file(&temp_path);
                }
                return result;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                last_error = Some(error)
            }
            Err(error) => return Err(error.into()),
        }
    }
    Err(last_error
        .map(anyhow::Error::from)
        .unwrap_or_else(|| anyhow::anyhow!("无法创建术语表临时文件")))
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
            ruby_annotations: Vec::new(),
            model_dir: ".".into(),
            ner_batch_size: 8,
            ner_backend: NerBackend::Auto,
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
    fn validate_accepts_auto_batch_and_rejects_zero_workers() {
        let mut cfg = sample_config();
        cfg.ner_batch_size = 0;
        assert!(cfg.validate().is_ok());

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

    #[test]
    fn ner_backend_order_matches_gui_and_roundtrips() {
        assert_eq!(
            NerBackend::ALL.map(NerBackend::label),
            ["CPU", "GPU", "Auto"]
        );
        for (index, backend) in NerBackend::ALL.into_iter().enumerate() {
            assert_eq!(NerBackend::from_index(index), backend);
            let json = serde_json::to_string(&backend).unwrap();
            assert_eq!(serde_json::from_str::<NerBackend>(&json).unwrap(), backend);
        }
    }

    #[test]
    fn backend_tuned_batch_defaults_match_modernbert_cli() {
        let cpu = infer_options(0, false);
        assert_eq!((cpu.max_sentences, cpu.max_tokens), (24, 1_536));

        let gpu = infer_options(0, true);
        assert_eq!((gpu.max_sentences, gpu.max_tokens), (128, 32_768));

        assert_eq!(infer_options(48, false).max_sentences, 48);
        assert_eq!(infer_options(48, true).max_sentences, 48);
    }

    #[test]
    fn auto_benchmark_sample_is_short_and_spans_document() {
        let lines: Vec<String> = (0..1000)
            .map(|index| format!("第{index}行の文章"))
            .collect();
        let sample = benchmark_sample(&lines);
        assert_eq!(sample.len(), AUTO_BENCH_LINES);
        assert_eq!(sample.first(), lines.first());
        assert!(sample.last().unwrap().contains("992"));
    }

    #[test]
    #[ignore = "loads the real checkpoint and initializes the system GPU"]
    fn auto_backend_benchmarks_real_cpu_and_gpu() {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../alnilam/ner_model");
        if !model_dir.join("model.safetensors").is_file() {
            return;
        }

        let seeds = [
            "橘真児は東京で美咲と会った。",
            "艾莉同学は教室へ向かった。",
            "健太と志保子が廊下で話している。",
            "沙由美ちゃんが図書館から帰ってきた。",
        ];
        let lines: Vec<String> = (0..AUTO_BENCH_LINES)
            .map(|index| format!("{} 第{}章", seeds[index % seeds.len()], index + 1))
            .collect();
        let events = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let captured = events.clone();
        let progress: GlossaryProgressCallback = Some(Arc::new(move |event| {
            if let GlossaryProgressEvent::Log { message } = event {
                captured.lock().unwrap().push(message);
            }
        }));

        let pipeline = load_ner_pipeline(&model_dir, NerBackend::Auto, &lines, 0, &progress)
            .expect("Auto backend should select a working pipeline");
        let results = pipeline
            .predict_document(&lines, None)
            .expect("selected backend inference");
        assert_eq!(results.len(), lines.len());
        drop(pipeline);

        let cached = load_ner_pipeline(&model_dir, NerBackend::Auto, &lines, 0, &progress)
            .expect("cached Auto backend should reload without benchmarking");
        drop(cached);

        let events = events.lock().unwrap();
        assert!(events.iter().any(|message| message.contains("CPU")));
        assert!(events
            .iter()
            .any(|message| message.contains("GPU") || message.contains("回退 CPU")));
        assert!(events
            .iter()
            .any(|message| message.contains("复用本次应用会话")));
        for event in events.iter() {
            eprintln!("{event}");
        }
    }

    fn ruby(base: &str, reading: &str) -> betelgeuse::RubyAnnotation {
        betelgeuse::RubyAnnotation {
            base: base.to_string(),
            reading: reading.to_string(),
            source_path: "Text/ch1.xhtml".to_string(),
            context: format!("{base}が現れた"),
            combined: false,
        }
    }

    fn translation(src: &str, dst: &str) -> llm::TranslationEntry {
        llm::TranslationEntry {
            src: src.to_string(),
            dst: dst.to_string(),
            info: String::new(),
        }
    }

    #[test]
    fn ruby_review_finds_hiragana_or_katakana_independent_mentions() {
        let review = build_ruby_alias_review(
            &[ruby("白地野音", "しらじのおと")],
            &["後でシラジノオトが笑った。".to_string()],
            &[translation("白地野音", "白地野音")],
        );

        assert_eq!(review.len(), 1);
        assert_eq!(review[0].status, "review_required");
        assert_eq!(review[0].independent_mentions, 1);
        assert!(review[0]
            .reading_variants
            .contains(&"シラジノオト".to_string()));
        assert_eq!(
            review[0].suggested_classification,
            RubyAliasClassification::PhoneticReading
        );
        assert_eq!(review[0].suggestion_confidence, 85);
    }

    #[test]
    fn ruby_review_does_not_suggest_alias_for_unconfirmed_base() {
        let review = build_ruby_alias_review(
            &[ruby("白地野音", "しらじのおと")],
            &["後でしらじのおとが笑った。".to_string()],
            &[],
        );

        assert_eq!(review[0].status, "base_not_confirmed");
        assert_eq!(
            review[0].suggested_classification,
            RubyAliasClassification::Unclassified
        );
        assert_eq!(review[0].suggestion_confidence, 15);
        assert!(review[0]
            .suggestion_reason
            .contains("尚未被术语生成确认为实体"));
    }

    #[test]
    fn ruby_review_does_not_suggest_alias_without_independent_recurrence() {
        let review = build_ruby_alias_review(
            &[ruby("白地野音", "しらじのおと")],
            &["白地野音が笑った。".to_string()],
            &[translation("白地野音", "白地野音")],
        );

        assert_eq!(review[0].status, "no_independent_mention");
        assert_eq!(
            review[0].suggested_classification,
            RubyAliasClassification::Unclassified
        );
        assert_eq!(review[0].suggestion_confidence, 30);
        assert!(review[0].suggestion_reason.contains("rt 外独立复现"));
    }

    #[test]
    fn ruby_review_does_not_count_short_katakana_inside_longer_words() {
        let review = build_ruby_alias_review(
            &[ruby("愛", "アイ")],
            &["アイテム、アイディア、アイツ。".to_string()],
            &[translation("愛", "爱")],
        );

        assert_eq!(review[0].status, "no_independent_mention");
        assert_eq!(review[0].independent_mentions, 0);
    }

    #[test]
    fn semantic_ruby_remains_review_only_even_when_it_reappears() {
        let review = build_ruby_alias_review(
            &[ruby("桃谷", "ライバル")],
            &["ライバルが立ちはだかった。".to_string()],
            &[translation("桃谷", "桃谷")],
        );

        assert_eq!(review[0].status, "review_required");
        assert!(review[0].reason.contains("语义 ruby"));
        assert_eq!(
            review[0].suggested_classification,
            RubyAliasClassification::Unclassified
        );
        assert!(review[0].suggestion_reason.contains("片假名 ruby"));
    }

    #[test]
    fn ruby_review_exposes_existing_translation_conflict() {
        let review = build_ruby_alias_review(
            &[ruby("白地野音", "シラジノ")],
            &["シラジノが来た。".to_string()],
            &[
                translation("白地野音", "白地野音"),
                translation("シラジノ", "席拉吉诺"),
            ],
        );

        assert_eq!(review[0].status, "conflict");
        assert_eq!(review[0].base_dst.as_deref(), Some("白地野音"));
        assert_eq!(review[0].existing_reading_dst.as_deref(), Some("席拉吉诺"));
    }

    #[test]
    fn ruby_review_matches_kana_variant_with_honorific_suffix() {
        let review = build_ruby_alias_review(
            &[ruby("最乃", "もの")],
            &["モノくんが来た。".to_string()],
            &[
                translation("最乃", "入座最乃"),
                translation("モノくん", "物君"),
            ],
        );

        assert_eq!(review[0].status, "conflict");
        assert!(review[0]
            .surface_evidence
            .iter()
            .any(|evidence| evidence.surface == "モノ" && evidence.mention_count == 1));
    }

    #[test]
    fn ruby_review_sidecar_keeps_glossary_stem() {
        assert_eq!(
            ruby_review_path(Path::new("book_glossary.json")),
            PathBuf::from("book_glossary.ruby-candidates.json")
        );
        assert_eq!(
            glossary_generation_report_path(Path::new("book_glossary.json")),
            PathBuf::from("book_glossary.generation-report.json")
        );
    }

    fn temp_review_path(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "bellatrix-ruby-review-{label}-{}-{nonce}.json",
            std::process::id()
        ))
    }

    #[test]
    fn ruby_review_v2_roundtrip_and_refresh_preserve_decision() {
        let path = temp_review_path("preserve");
        let annotations = [ruby("白地野音", "しらじのおと")];
        let lines = ["シラジノオトが来た。".to_string()];
        let translations = [translation("白地野音", "白地野音")];

        let first = refresh_ruby_alias_review(&path, &annotations, &lines, &translations).unwrap();
        assert_eq!(first.schema_version, RUBY_REVIEW_SCHEMA_VERSION);
        let id = first.candidates[0].candidate_id.clone();
        update_ruby_alias_decision(
            &path,
            &id,
            RubyAliasClassification::PhoneticReading,
            CandidateDecision::Confirmed,
            Some("白地野音".to_string()),
            Some("正文复现已核对".to_string()),
        )
        .unwrap();

        let refreshed =
            refresh_ruby_alias_review(&path, &annotations, &lines, &translations).unwrap();
        let candidate = &refreshed.candidates[0];
        assert_eq!(candidate.candidate_id, id);
        assert_eq!(candidate.decision, CandidateDecision::Confirmed);
        assert_eq!(
            candidate.classification,
            RubyAliasClassification::PhoneticReading
        );
        assert_eq!(candidate.target.as_deref(), Some("白地野音"));
        assert_eq!(candidate.revision, 1);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn semantic_ruby_cannot_be_confirmed_as_translation_alias() {
        let path = temp_review_path("semantic");
        let review = build_ruby_alias_review(
            &[ruby("桃谷", "ライバル")],
            &["ライバルが来た。".to_string()],
            &[translation("桃谷", "桃谷")],
        );
        save_ruby_alias_review(&path, &review).unwrap();
        let id = review[0].candidate_id.clone();

        let error = update_ruby_alias_decision(
            &path,
            &id,
            RubyAliasClassification::SemanticRuby,
            CandidateDecision::Confirmed,
            Some("桃谷".to_string()),
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("不能确认为翻译 alias"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn legacy_array_sidecar_is_loaded_as_v2() {
        let path = temp_review_path("legacy");
        let review = build_ruby_alias_review(
            &[ruby("奈子", "なこ")],
            &["ナコちゃん".to_string()],
            &[translation("奈子", "奈子")],
        );
        atomic_write(
            &path,
            serde_json::to_string_pretty(&review).unwrap().as_bytes(),
        )
        .unwrap();

        let loaded = load_ruby_alias_review(&path).unwrap();
        assert_eq!(loaded.schema_version, RUBY_REVIEW_SCHEMA_VERSION);
        assert_eq!(loaded.candidates.len(), 1);
        assert!(!loaded.candidates[0].candidate_id.is_empty());

        let _ = fs::remove_file(path);
    }
}
