use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use serde_json;
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use crate::checker::{AutoFixer, ErrorRecord, ErrorType, ResponseChecker};
use crate::config::{
    PipelineConfig, DEFAULT_CONTEXT_WINDOW_MIN, DEFAULT_CONTEXT_WINDOW_MULTIPLIER,
};
use crate::context::{
    precompute_context, select_context, select_context_precomputed, ContextDetector,
    PrecomputedContext,
};
use crate::epub::{EpubHandler, TranslationBlock};
use crate::llm::{BatchTranslationResponse, LlmClient};
use crate::txt;

// ============================================================================
// Progress Reporting
// ============================================================================

/// Progress event sent from the pipeline to the GUI or CLI
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    /// Pipeline stage started
    StageStarted { stage: String, detail: String },
    /// Batch progress update
    BatchProgress {
        completed: usize,
        total: usize,
        total_lines: usize,
        translated: usize,
        failed: usize,
        corrected: usize,
        quality_failed: usize,
        api_failed: usize,
    },
    /// Log message
    Log { message: String },
    /// Pipeline completed
    Completed {
        total: usize,
        translated: usize,
        fixed: usize,
        failed: usize,
        output_path: String,
        error_report_path: Option<String>,
    },
    /// Pipeline completed but some lines still failed
    CompletedWithErrors {
        total: usize,
        translated: usize,
        fixed: usize,
        failed: usize,
        output_path: String,
        error_report_path: Option<String>,
    },
    /// Pipeline error
    Error { message: String },
}

/// A callback type for progress reporting. None = no callback (CLI mode)
pub type ProgressCallback = Option<Arc<dyn Fn(ProgressEvent) + Send + Sync>>;

/// A cancellation flag shared between GUI and pipeline. true = cancel requested.
pub type CancelFlag = Option<Arc<AtomicBool>>;

/// Helper to send progress events
#[allow(dead_code)]
fn emit_progress(cb: &ProgressCallback, event: ProgressEvent) {
    if let Some(ref callback) = cb {
        callback(event);
    }
}

fn is_cancelled(cancel_flag: &CancelFlag) -> bool {
    cancel_flag
        .as_ref()
        .is_some_and(|flag| flag.load(Ordering::Relaxed))
}

fn batch_progress_event(
    completed: usize,
    total: usize,
    total_lines: usize,
    translated: usize,
    corrected: usize,
    quality_failed: usize,
    api_failed: usize,
) -> ProgressEvent {
    ProgressEvent::BatchProgress {
        completed,
        total,
        total_lines,
        translated,
        failed: quality_failed + api_failed,
        corrected,
        quality_failed,
        api_failed,
    }
}

fn completion_event(
    total: usize,
    translated: usize,
    fixed: usize,
    failed: usize,
    output_path: String,
    error_report_path: Option<String>,
) -> ProgressEvent {
    if failed > 0 {
        ProgressEvent::CompletedWithErrors {
            total,
            translated,
            fixed,
            failed,
            output_path,
            error_report_path,
        }
    } else {
        ProgressEvent::Completed {
            total,
            translated,
            fixed,
            failed,
            output_path,
            error_report_path,
        }
    }
}

#[derive(Debug, Default)]
struct BatchOutcome {
    fixed: HashMap<usize, String>,
    quality_failed: Vec<usize>,
    api_failed: Vec<usize>,
    failure_causes: HashMap<usize, FailureCause>,
    corrected_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FailureCause {
    error_type: String,
    details: String,
    stage: String,
    retry_count: usize,
}

impl FailureCause {
    fn new(
        error_type: impl Into<String>,
        details: impl Into<String>,
        stage: impl Into<String>,
        retry_count: usize,
    ) -> Self {
        Self {
            error_type: error_type.into(),
            details: details.into(),
            stage: stage.into(),
            retry_count,
        }
    }
}

#[derive(Debug, Clone)]
struct BlockReportMetadata {
    file_name: Option<String>,
    block_index: usize,
    title: Option<String>,
    page_type: String,
}

fn push_unique(target: &mut Vec<usize>, idx: usize) {
    if !target.contains(&idx) {
        target.push(idx);
    }
}

fn extend_unique(target: &mut Vec<usize>, indices: impl IntoIterator<Item = usize>) {
    for idx in indices {
        push_unique(target, idx);
    }
}

fn extend_failure_causes(
    target: &mut HashMap<usize, FailureCause>,
    source: HashMap<usize, FailureCause>,
) {
    for (idx, cause) in source {
        target.insert(idx, cause);
    }
}

fn failure_cause_entry(cause: &FailureCause) -> serde_json::Value {
    serde_json::json!({
        "error_type": &cause.error_type,
        "details": &cause.details,
        "stage": &cause.stage,
        "retry_count": cause.retry_count,
    })
}

fn parser_diagnostic_summary(response: &BatchTranslationResponse) -> String {
    let mut details = Vec::new();
    if !response.diagnostics.missing_indices.is_empty() {
        details.push(format!(
            "missing={:?}",
            response.diagnostics.missing_indices
        ));
    }
    if !response.diagnostics.duplicate_indices.is_empty() {
        details.push(format!(
            "duplicate={:?}",
            response.diagnostics.duplicate_indices
        ));
    }
    if !response.diagnostics.out_of_range_indices.is_empty() {
        details.push(format!(
            "out_of_range={:?}",
            response.diagnostics.out_of_range_indices
        ));
    }
    if !response.diagnostics.malformed_lines.is_empty() {
        details.push(format!(
            "malformed_lines={}",
            response.diagnostics.malformed_lines.len()
        ));
    }

    if details.is_empty() {
        "LLM 未返回该行 JSONL 译文".to_string()
    } else {
        details.join(", ")
    }
}

// ============================================================================
// Work Directory Management
// ============================================================================

/// Create a work directory for storing intermediate translation data.
/// The directory sits next to the output file and is named `{stem}_work/`.
///
/// Example:
///   output = `/data/book.ja-zh[deepseek-v4-flash].epub`
///   work   = `/data/book.ja-zh[deepseek-v4-flash]_work/`
fn create_work_dir(output_path: &Path) -> Result<PathBuf> {
    let stem = output_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let parent = output_path.parent().unwrap_or(Path::new("."));
    let work_dir = parent.join(format!("{}_work", stem));
    std::fs::create_dir_all(&work_dir)?;
    Ok(work_dir)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ResumeManifest {
    version: u32,
    source_kind: String,
    input_path: String,
    input_sha256: String,
    config_sha256: String,
    model: String,
    llm_url: String,
    mode: String,
    batch_size: usize,
    context_lines: usize,
    temperature: String,
    top_p: String,
    top_k: u32,
    glossary_sha256: Option<String>,
    rules_sha256: Option<String>,
    extractor_version: String,
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn sha256_file(path: &Path) -> Result<String> {
    Ok(sha256_bytes(&std::fs::read(path)?))
}

fn optional_file_hash(path: Option<&PathBuf>) -> Result<Option<String>> {
    match path {
        Some(path) if path.exists() => sha256_file(path).map(Some),
        _ => Ok(None),
    }
}

fn rules_hash(config: &PipelineConfig) -> Result<Option<String>> {
    if let Some(path) = config.rules_path.as_ref() {
        return optional_file_hash(Some(path));
    }
    Ok(EMBEDDED_RULES.map(|rules| sha256_bytes(rules.as_bytes())))
}

fn build_resume_manifest(
    source_kind: &str,
    input_path: &Path,
    config: &PipelineConfig,
) -> Result<ResumeManifest> {
    let input_sha256 = sha256_file(input_path)?;
    let glossary_sha256 = optional_file_hash(config.glossary_path.as_ref())?;
    let rules_sha256 = rules_hash(config)?;
    let config_value = serde_json::json!({
        "source_kind": source_kind,
        "input_sha256": &input_sha256,
        "model": &config.model,
        "llm_url": &config.llm_url,
        "mode": config.mode.to_string(),
        "batch_size": config.batch_size,
        "context_lines": config.context_lines,
        "max_retry": config.max_retry,
        "translation_gap": config.translation_gap.as_deref(),
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "glossary_sha256": &glossary_sha256,
        "rules_sha256": &rules_sha256,
        "extractor_version": "2026-04-25-positioned-epub-v2",
    });
    let config_sha256 = sha256_bytes(serde_json::to_string(&config_value)?.as_bytes());

    Ok(ResumeManifest {
        version: 1,
        source_kind: source_kind.to_string(),
        input_path: input_path.display().to_string(),
        input_sha256,
        config_sha256,
        model: config.model.clone(),
        llm_url: config.llm_url.clone(),
        mode: config.mode.to_string(),
        batch_size: config.batch_size,
        context_lines: config.context_lines,
        temperature: config.temperature.to_string(),
        top_p: config.top_p.to_string(),
        top_k: config.top_k,
        glossary_sha256,
        rules_sha256,
        extractor_version: "2026-04-25-positioned-epub-v2".to_string(),
    })
}

fn load_resume_manifest(work_dir: &Path) -> Result<Option<ResumeManifest>> {
    let path = work_dir.join("resume_manifest.json");
    if !path.exists() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(path)?;
    Ok(Some(serde_json::from_str(&content)?))
}

fn save_resume_manifest(work_dir: &Path, manifest: &ResumeManifest) -> Result<()> {
    let path = work_dir.join("resume_manifest.json");
    std::fs::write(path, serde_json::to_string_pretty(manifest)?)?;
    Ok(())
}

fn can_resume_from_manifest(work_dir: &Path, current: &ResumeManifest) -> Result<bool> {
    match load_resume_manifest(work_dir)? {
        Some(existing) if existing == *current => Ok(true),
        Some(existing) => {
            info!(
                "恢复缓存指纹不匹配，跳过复用。旧 config={}, 新 config={}",
                existing.config_sha256, current.config_sha256
            );
            Ok(false)
        }
        None => {
            info!("未找到恢复指纹，跳过旧缓存复用");
            Ok(false)
        }
    }
}

/// Save a stage snapshot (EPUB) into the work directory.
/// `stage_name` examples: `step2_first_pass`, `step3_after_retry`
fn save_epub_stage(
    work_dir: &Path,
    stage_name: &str,
    data: &[TranslationBlock],
    translations: &HashMap<usize, String>,
    failed_indices: &[usize],
    failure_causes: &HashMap<usize, FailureCause>,
) -> Result<()> {
    // Build the snapshot: enrich data with current translations + failure markers
    let snapshot: Vec<serde_json::Value> = data
        .iter()
        .enumerate()
        .map(|(i, block)| {
            let mut entry = serde_json::json!({
                "index": i,
                "global_index": i,
                "file_name": &block.file_name,
                "block_index": block.index,
                "title": &block.title,
                "page_type": &block.page_type,
                "src_text": &block.src_text,
            });
            if let Some(dst) = translations.get(&i) {
                entry["dst_text"] = serde_json::json!(dst);
                entry["status"] = serde_json::json!("translated");
            } else if failed_indices.contains(&i) {
                entry["status"] = serde_json::json!("failed");
                if let Some(cause) = failure_causes.get(&i) {
                    entry["failure"] = failure_cause_entry(cause);
                }
            } else {
                entry["status"] = serde_json::json!("pending");
            }
            entry
        })
        .collect();

    let summary = serde_json::json!({
        "stage": stage_name,
        "total": data.len(),
        "translated": translations.len(),
        "failed": failed_indices.len(),
        "pending": data.len() - translations.len() - failed_indices.len(),
        "failed_indices": failed_indices,
        "data": snapshot,
    });

    let path = work_dir.join(format!("{}.json", stage_name));
    std::fs::write(&path, serde_json::to_string_pretty(&summary)?)?;
    info!("阶段快照已保存: {}", path.display());
    Ok(())
}

/// Save a stage snapshot (TXT) into the work directory.
fn save_txt_stage(
    work_dir: &Path,
    stage_name: &str,
    data: &[txt::TxtBlock],
    translations: &HashMap<usize, String>,
    failed_indices: &[usize],
    failure_causes: &HashMap<usize, FailureCause>,
) -> Result<()> {
    let snapshot: Vec<serde_json::Value> = data
        .iter()
        .enumerate()
        .map(|(i, block)| {
            let mut entry = serde_json::json!({
                "index": i,
                "global_index": i,
                "block_index": block.index,
                "page_type": "txt",
                "src_text": &block.src_text,
            });
            if let Some(dst) = translations.get(&i) {
                entry["dst_text"] = serde_json::json!(dst);
                entry["status"] = serde_json::json!("translated");
            } else if failed_indices.contains(&i) {
                entry["status"] = serde_json::json!("failed");
                if let Some(cause) = failure_causes.get(&i) {
                    entry["failure"] = failure_cause_entry(cause);
                }
            } else {
                entry["status"] = serde_json::json!("pending");
            }
            entry
        })
        .collect();

    let summary = serde_json::json!({
        "stage": stage_name,
        "total": data.len(),
        "translated": translations.len(),
        "failed": failed_indices.len(),
        "pending": data.len() - translations.len() - failed_indices.len(),
        "failed_indices": failed_indices,
        "data": snapshot,
    });

    let path = work_dir.join(format!("{}.json", stage_name));
    std::fs::write(&path, serde_json::to_string_pretty(&summary)?)?;
    info!("阶段快照已保存: {}", path.display());
    Ok(())
}

// ============================================================================
// Embedded Rules (compile-time)
// ============================================================================

/// When built with `--features embed-rules`, the rules JSON is baked into the binary.
#[cfg(feature = "embed-rules")]
const EMBEDDED_RULES: Option<&str> = Some(include_str!(env!("ORION_RULES_PATH")));

#[cfg(not(feature = "embed-rules"))]
const EMBEDDED_RULES: Option<&str> = None;

/// Load the context detector: CLI path > embedded > filesystem fallback
fn load_context_detector(config: &PipelineConfig) -> Option<ContextDetector> {
    // 1. Explicit CLI path
    if let Some(ref path) = config.rules_path {
        match ContextDetector::from_file(path) {
            Ok(d) => {
                info!("已加载上下文检测器（文件: {}）", path.display());
                return Some(d);
            }
            Err(e) => {
                warn!("加载规则文件失败: {}", e);
            }
        }
    }

    // 2. Embedded rules (always present when built with default features)
    if let Some(rules_json) = EMBEDDED_RULES {
        match serde_json::from_str::<serde_json::Value>(rules_json) {
            Ok(v) => match ContextDetector::from_config(&v) {
                Ok(d) => {
                    info!("已加载上下文检测器（内置规则）");
                    return Some(d);
                }
                Err(e) => {
                    warn!("内置规则构建ContextDetector失败: {}", e);
                }
            },
            Err(e) => {
                warn!("内置规则JSON解析失败: {}", e);
            }
        }
    }

    // This path should not be reached in normal builds (embed-rules is the default feature).
    // If reached, it means the binary was built with --no-default-features and no --rules-path
    // was specified at runtime.
    warn!(
        "未能加载上下文规则，回退到简单上下文（翻译质量可能下降）。\
           建议：使用默认编译（embed-rules 特性），或通过 --rules-path 指定规则文件。"
    );
    None
}

// ============================================================================
// Glossary Loading
// ============================================================================

fn load_glossary_text(config: &PipelineConfig) -> String {
    use crate::llm::glossary;
    match &config.glossary_path {
        Some(path) if path.exists() => match glossary::load_glossary(path) {
            Ok(entries) => {
                info!("已加载术语表: {} 条", entries.len());
                glossary::format_glossary(&entries)
            }
            Err(e) => {
                warn!("加载术语表失败: {}", e);
                String::new()
            }
        },
        _ => String::new(),
    }
}

/// 加载术语表并格式化为 Orion 模型专用格式（与 SFT 训练一致）
fn load_orion_glossary_text(config: &PipelineConfig) -> Option<String> {
    use crate::llm::glossary;
    match &config.glossary_path {
        Some(path) if path.exists() => match glossary::load_glossary(path) {
            Ok(entries) => glossary::format_glossary_for_orion(&entries),
            Err(_) => None,
        },
        _ => None,
    }
}

// ============================================================================
// Error Report Generation
// ============================================================================

fn epub_report_metadata(data: &[TranslationBlock]) -> Vec<BlockReportMetadata> {
    data.iter()
        .map(|block| BlockReportMetadata {
            file_name: Some(block.file_name.clone()),
            block_index: block.index,
            title: Some(block.title.clone()),
            page_type: block.page_type.clone(),
        })
        .collect()
}

fn txt_report_metadata(data: &[txt::TxtBlock]) -> Vec<BlockReportMetadata> {
    data.iter()
        .map(|block| BlockReportMetadata {
            file_name: None,
            block_index: block.index,
            title: None,
            page_type: "txt".to_string(),
        })
        .collect()
}

fn error_report_entry(
    record: &ErrorRecord,
    metadata: Option<&BlockReportMetadata>,
) -> serde_json::Value {
    let mut entry = serde_json::json!({
        "index": record.index,
        "global_index": record.index,
        "src_text": &record.src_text,
        "dst_text": &record.dst_text,
        "error_type": &record.error_type,
        "fix_details": &record.fix_details,
        "retry_count": record.retry_count,
    });

    if let Some(metadata) = metadata {
        entry["block_index"] = serde_json::json!(metadata.block_index);
        entry["page_type"] = serde_json::json!(&metadata.page_type);
        if let Some(file_name) = &metadata.file_name {
            entry["file_name"] = serde_json::json!(file_name);
        }
        if let Some(title) = &metadata.title {
            entry["title"] = serde_json::json!(title);
        }
    }

    if let Some(stage) = &record.stage {
        entry["stage"] = serde_json::json!(stage);
    }
    if let Some(details) = &record.details {
        entry["details"] = serde_json::json!(details);
    }
    if let Some(last_error_type) = &record.last_error_type {
        entry["last_error_type"] = serde_json::json!(last_error_type);
    }
    if let Some(last_error_details) = &record.last_error_details {
        entry["last_error_details"] = serde_json::json!(last_error_details);
    }

    entry
}

fn generate_error_report(
    error_records: &[ErrorRecord],
    work_dir: &Path,
    metadata: Option<&[BlockReportMetadata]>,
) -> Result<HashMap<String, serde_json::Value>> {
    let mut total_records: u64 = 0;
    let mut normalization_count: u64 = 0;
    let mut fixed_quality_count: u64 = 0;
    let mut failed_count: u64 = 0;
    let mut normalization_list = Vec::new();
    let mut fixed_quality_list = Vec::new();
    let mut failed_list = Vec::new();

    for record in error_records {
        total_records += 1;
        let entry = error_report_entry(record, metadata.and_then(|items| items.get(record.index)));

        if record.fixed && record.error_type == "AUTO_FIX" {
            normalization_count += 1;
            normalization_list.push(entry);
        } else if record.fixed {
            fixed_quality_count += 1;
            fixed_quality_list.push(entry);
        } else {
            failed_count += 1;
            failed_list.push(entry);
        }
    }

    let report_path = work_dir.join("error_report.json");
    let summary = serde_json::json!({
        "total_records": total_records,
        "total_errors": fixed_quality_count + failed_count,
        "normalizations": normalization_count,
        "fixed": fixed_quality_count,
        "fixed_quality_errors": fixed_quality_count,
        "failed": failed_count,
        "report_path": report_path.display().to_string(),
    });

    let report = serde_json::json!({
        "summary": summary,
        "normalizations": normalization_list,
        "fixed": fixed_quality_list.clone(),
        "fixed_quality_errors": fixed_quality_list,
        "failed": failed_list,
    });

    std::fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;

    let result: HashMap<String, serde_json::Value> =
        serde_json::from_value(summary).unwrap_or_default();
    Ok(result)
}

// ============================================================================
// Check & Fix Translations
// ============================================================================

fn check_and_fix_translations(
    checker: &ResponseChecker,
    fixer: &AutoFixer,
    srcs: &[String],
    dsts: &[String],
    original_indices: &[usize],
    error_records: &Arc<Mutex<Vec<ErrorRecord>>>,
    retry_count: usize,
) -> (
    HashMap<usize, String>,
    Vec<usize>,
    usize,
    HashMap<usize, FailureCause>,
) {
    let check_results = checker.check(srcs, dsts, retry_count);
    let mut fixed_results = HashMap::new();
    let mut failed_indices = Vec::new();
    let mut corrected_count = 0usize;
    let mut failure_causes = HashMap::new();

    for (i, result) in check_results.iter().enumerate() {
        let orig_idx = original_indices[i];
        let src = &srcs[i];
        let dst = &dsts[i];

        if result.error == ErrorType::None {
            let fixed_dst = fixer.fix(src, dst);
            fixed_results.insert(orig_idx, fixed_dst.clone());

            if fixed_dst != *dst {
                corrected_count += 1;
                info!(
                    "[#{}] 自动修复: '{:.30}...' -> '{:.30}...'",
                    orig_idx, dst, fixed_dst
                );
                let mut records = error_records.lock().unwrap_or_else(|e| e.into_inner());
                records.push(ErrorRecord {
                    index: orig_idx,
                    src_text: src.clone(),
                    dst_text: dst.clone(),
                    error_type: "AUTO_FIX".to_string(),
                    fixed: true,
                    fix_details: format!("Auto-fixed: '{}' -> '{}'", dst, fixed_dst),
                    retry_count,
                    stage: Some("normalization".to_string()),
                    details: Some("标点/引号自动正规化".to_string()),
                    last_error_type: None,
                    last_error_details: None,
                });
            }
        } else {
            let fixed_dst = fixer.fix(src, dst);
            let recheck = checker.check(&[src.clone()], &[fixed_dst.clone()], retry_count);

            if !recheck.is_empty() && recheck[0].error == ErrorType::None {
                fixed_results.insert(orig_idx, fixed_dst.clone());
                corrected_count += 1;
                info!(
                    "[#{}] {} -> 已修复: '{:.20}...' -> '{:.20}...': {}",
                    orig_idx, result.error, dst, fixed_dst, result.details
                );
                let mut records = error_records.lock().unwrap_or_else(|e| e.into_inner());
                records.push(ErrorRecord {
                    index: orig_idx,
                    src_text: src.clone(),
                    dst_text: dst.clone(),
                    error_type: result.error.to_string(),
                    fixed: true,
                    fix_details: format!("Auto-fixed: '{}' -> '{}'", dst, fixed_dst),
                    retry_count,
                    stage: Some("quality_check".to_string()),
                    details: Some(result.details.clone()),
                    last_error_type: None,
                    last_error_details: None,
                });
            } else {
                failed_indices.push(orig_idx);
                failure_causes.insert(
                    orig_idx,
                    FailureCause::new(
                        result.error.to_string(),
                        result.details.clone(),
                        "quality_check",
                        retry_count,
                    ),
                );
                info!(
                    "[#{}] {} 需重试: src='{:.30}...' dst='{:.30}...': {}",
                    orig_idx, result.error, src, dst, result.details
                );
            }
        }
    }

    (
        fixed_results,
        failed_indices,
        corrected_count,
        failure_causes,
    )
}

fn process_batch_response(
    response: BatchTranslationResponse,
    start_idx: usize,
    end_idx: usize,
    src_text_at: impl Fn(usize) -> String,
    checker: &ResponseChecker,
    fixer: &AutoFixer,
    error_records: &Arc<Mutex<Vec<ErrorRecord>>>,
) -> BatchOutcome {
    let mut srcs = Vec::new();
    let mut dsts = Vec::new();
    let mut indices = Vec::new();
    let mut api_failed = Vec::new();
    let mut failure_causes = HashMap::new();
    let diagnostic_summary = parser_diagnostic_summary(&response);

    for idx in start_idx..end_idx {
        let jsonl_idx = idx - start_idx + 1;
        if let Some(translated) = response.translations.get(&jsonl_idx) {
            srcs.push(src_text_at(idx));
            dsts.push(translated.clone());
            indices.push(idx);
        } else {
            push_unique(&mut api_failed, idx);
            failure_causes.insert(
                idx,
                FailureCause::new(
                    ErrorType::JsonStructureError.to_string(),
                    format!(
                        "LLM 响应缺少 JSONL 序号 {}；{}",
                        jsonl_idx, diagnostic_summary
                    ),
                    "first_pass_parser",
                    0,
                ),
            );
        }
    }

    if response.diagnostics.has_issues() {
        warn!(
            "批次 {}..{} 响应结构异常: {}",
            start_idx, end_idx, diagnostic_summary
        );
    }

    if srcs.is_empty() {
        return BatchOutcome {
            api_failed,
            failure_causes,
            ..BatchOutcome::default()
        };
    }

    let (fixed, quality_failed, corrected_count, quality_failure_causes) =
        check_and_fix_translations(checker, fixer, &srcs, &dsts, &indices, error_records, 0);
    failure_causes.extend(quality_failure_causes);

    BatchOutcome {
        fixed,
        quality_failed,
        api_failed,
        failure_causes,
        corrected_count,
    }
}

// ============================================================================
// Context Building
// ============================================================================

#[allow(dead_code)]
fn build_context_for_batch(
    detector: Option<&ContextDetector>,
    all_src_lines: &[String],
    start_idx: usize,
    end_idx: usize,
    context_lines: usize,
    data_len: usize,
) -> Vec<String> {
    if let Some(det) = detector {
        if start_idx > 0 {
            let window =
                (context_lines * DEFAULT_CONTEXT_WINDOW_MULTIPLIER).max(DEFAULT_CONTEXT_WINDOW_MIN);
            match select_context(
                det,
                all_src_lines,
                start_idx + 1,
                end_idx,
                window,
                context_lines,
            ) {
                Ok(sel) => {
                    return sel.selected.iter().map(|s| s.text.clone()).collect();
                }
                Err(e) => {
                    tracing::debug!("Context detector fallback: {}", e);
                }
            }
        }
    }

    // Fallback: simple previous lines
    let context_start = start_idx.saturating_sub(context_lines);
    (context_start..start_idx)
        .filter(|&i| i < data_len)
        .map(|i| all_src_lines[i].clone())
        .collect()
}

/// 使用预计算上下文数据构建批次上下文（高效版本）
fn build_context_for_batch_precomputed(
    precomputed: Option<&PrecomputedContext>,
    all_src_lines: &[String],
    start_idx: usize,
    end_idx: usize,
    context_lines: usize,
    data_len: usize,
) -> Vec<String> {
    if let Some(pc) = precomputed {
        if start_idx > 0 {
            let window =
                (context_lines * DEFAULT_CONTEXT_WINDOW_MULTIPLIER).max(DEFAULT_CONTEXT_WINDOW_MIN);
            match select_context_precomputed(
                all_src_lines,
                pc,
                start_idx + 1,
                end_idx,
                window,
                context_lines,
            ) {
                Ok(sel) => {
                    return sel
                        .selected
                        .iter()
                        .map(|s| {
                            // precomputed 版本返回的 text 为空，需要从原始数据获取
                            if s.text.is_empty() {
                                all_src_lines[s.line_number - 1].clone()
                            } else {
                                s.text.clone()
                            }
                        })
                        .collect();
                }
                Err(e) => {
                    tracing::debug!("Context precomputed fallback: {}", e);
                }
            }
        }
    }

    // Fallback: simple previous lines
    let context_start = start_idx.saturating_sub(context_lines);
    (context_start..start_idx)
        .filter(|&i| i < data_len)
        .map(|i| all_src_lines[i].clone())
        .collect()
}

// ============================================================================
// Retry Failed Translations
// ============================================================================

async fn retry_failed_with_context(
    llm: &LlmClient,
    checker: &ResponseChecker,
    fixer: &AutoFixer,
    detector: Option<&ContextDetector>,
    all_src_lines: &[String],
    data: &[(String, Option<String>)], // (src_text, dst_text)
    failed_indices: &[usize],
    context_lines: usize,
    error_records: &Arc<Mutex<Vec<ErrorRecord>>>,
    failure_causes: &Arc<Mutex<HashMap<usize, FailureCause>>>,
    max_retry: usize,
    current_retry: usize,
    workers: usize,
    cancel_flag: &CancelFlag,
) -> Result<HashMap<usize, String>> {
    if is_cancelled(cancel_flag) {
        info!("重试被用户取消");
        return Ok(HashMap::new());
    }

    if current_retry > max_retry {
        for &idx in failed_indices {
            let last_cause = failure_causes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(&idx)
                .cloned();
            let mut records = error_records.lock().unwrap_or_else(|e| e.into_inner());
            records.push(ErrorRecord {
                index: idx,
                src_text: data[idx].0.clone(),
                dst_text: data[idx].1.clone().unwrap_or_default(),
                error_type: "MAX_RETRY_EXCEEDED".to_string(),
                fixed: false,
                fix_details: format!("Failed after {} retries", max_retry),
                retry_count: current_retry,
                stage: Some("retry".to_string()),
                details: Some(format!("Failed after {} retries", max_retry)),
                last_error_type: last_cause.as_ref().map(|cause| cause.error_type.clone()),
                last_error_details: last_cause.as_ref().map(|cause| cause.details.clone()),
            });
        }
        return Ok(HashMap::new());
    }

    let mut results = HashMap::new();
    let mut still_failed = Vec::new();

    // Pre-build context for each failed index
    let mut retry_items: Vec<(usize, Vec<String>)> = Vec::new();
    for &idx in failed_indices {
        let context_before = if let Some(det) = detector {
            if idx > 0 {
                let window = (context_lines * DEFAULT_CONTEXT_WINDOW_MULTIPLIER)
                    .max(DEFAULT_CONTEXT_WINDOW_MIN);
                match select_context(det, all_src_lines, idx + 1, idx + 1, window, context_lines) {
                    Ok(sel) => sel
                        .selected
                        .iter()
                        .map(|s| {
                            let line_idx = s.line_number - 1;
                            data.get(line_idx)
                                .and_then(|(_, dst)| dst.clone())
                                .unwrap_or_else(|| s.text.clone())
                        })
                        .collect::<Vec<_>>(),
                    Err(_) => {
                        let start = idx.saturating_sub(context_lines);
                        (start..idx)
                            .map(|i| data[i].1.clone().unwrap_or_else(|| data[i].0.clone()))
                            .collect()
                    }
                }
            } else {
                vec![]
            }
        } else {
            let start = idx.saturating_sub(context_lines);
            (start..idx)
                .map(|i| data[i].1.clone().unwrap_or_else(|| data[i].0.clone()))
                .collect()
        };

        let context_after_end = (idx + context_lines + 1).min(data.len());
        let context_after: Vec<String> = ((idx + 1)..context_after_end)
            .map(|i| data[i].0.clone())
            .collect();

        let full_context: Vec<String> = context_before.into_iter().chain(context_after).collect();

        retry_items.push((idx, full_context));
    }

    // Concurrent retry using semaphore (same concurrency as main translation)
    let semaphore = Arc::new(tokio::sync::Semaphore::new(workers.max(1)));
    let mut join_set = tokio::task::JoinSet::new();

    for (idx, full_context) in retry_items {
        let sem = semaphore.clone();
        let src_text = data[idx].0.clone();
        let batch_id = format!("retry_{}_{}", idx, current_retry);
        // We need to reference the LlmClient, checker, fixer etc.
        // Since they are borrowed, we use pointers via unsafe or restructure.
        // Instead, we'll build the prompt and call inline.
        // But LlmClient is not Send-safe for sharing across tasks easily.
        // Actually, translate_single takes &self, so we need Arc.
        // Let's use a simple approach: create a lightweight future per item.
        let llm_url = llm.llm_url().to_string();
        let model = llm.model().to_string();
        let temperature = llm.temperature();
        let top_p = llm.top_p();
        let top_k = llm.top_k();
        let glossary_text = llm.glossary_text().to_string();
        let orion_glossary_text = llm.orion_glossary_text().map(|s| s.to_string());
        let api_key = llm.api_key().cloned();
        let err_rec = error_records.clone();
        let task_cancel = cancel_flag.clone();

        join_set.spawn(async move {
            // Check cancel before waiting for semaphore
            if task_cancel
                .as_ref()
                .is_some_and(|f| f.load(Ordering::Relaxed))
            {
                return (idx, src_text, Ok(None), err_rec);
            }
            let result = async {
                let _permit = sem.acquire().await.map_err(|e| anyhow::anyhow!("{}", e))?;
                // Check cancel again after acquiring permit
                if task_cancel
                    .as_ref()
                    .is_some_and(|f| f.load(Ordering::Relaxed))
                {
                    return Ok(None);
                }
                let retry_llm = LlmClient::with_params(
                    &llm_url,
                    &model,
                    1,
                    temperature,
                    top_p,
                    top_k,
                    glossary_text,
                    orion_glossary_text,
                    api_key,
                )?;
                retry_llm
                    .translate_single(&src_text, &full_context, &batch_id)
                    .await
            }
            .await;
            (idx, src_text, result, err_rec)
        });
    }

    while let Some(join_result) = join_set.join_next().await {
        // Check cancel flag before processing each result
        if is_cancelled(cancel_flag) {
            info!("重试过程中收到取消请求，中止剩余任务");
            join_set.abort_all();
            // Drain remaining tasks to avoid leak
            while join_set.join_next().await.is_some() {}
            return Ok(results);
        }

        match join_result {
            Ok((idx, src_text, Ok(Some(translated)), _err_rec)) if !translated.is_empty() => {
                let (fixed_results, _retry_failed, _corrected, retry_failure_causes) =
                    check_and_fix_translations(
                        checker,
                        fixer,
                        &[src_text.clone()],
                        &[translated],
                        &[idx],
                        error_records,
                        current_retry,
                    );
                if !retry_failure_causes.is_empty() {
                    let mut causes = failure_causes.lock().unwrap_or_else(|e| e.into_inner());
                    causes.extend(retry_failure_causes);
                }

                if let Some(fixed) = fixed_results.get(&idx) {
                    results.insert(idx, fixed.clone());
                    info!(
                        "[#{}] 重试{}成功: '{:.30}...'",
                        idx, current_retry, src_text
                    );
                } else {
                    still_failed.push(idx);
                }
            }
            Ok((idx, _, Ok(_), _)) => {
                let mut causes = failure_causes.lock().unwrap_or_else(|e| e.into_inner());
                causes.insert(
                    idx,
                    FailureCause::new(
                        ErrorType::EmptyTranslation.to_string(),
                        "重试返回空译文或无可解析译文",
                        "retry_parser",
                        current_retry,
                    ),
                );
                still_failed.push(idx);
            }
            Ok((idx, _, Err(e), _)) => {
                warn!("重试任务错误: {}", e);
                let mut causes = failure_causes.lock().unwrap_or_else(|e| e.into_inner());
                causes.insert(
                    idx,
                    FailureCause::new("API_ERROR", e.to_string(), "retry_api", current_retry),
                );
                still_failed.push(idx);
            }
            Err(e) => {
                warn!("重试任务 join 错误: {}", e);
            }
        }
    }

    if !still_failed.is_empty() && current_retry < max_retry && !is_cancelled(cancel_flag) {
        let more = Box::pin(retry_failed_with_context(
            llm,
            checker,
            fixer,
            detector,
            all_src_lines,
            data,
            &still_failed,
            context_lines,
            error_records,
            failure_causes,
            max_retry,
            current_retry + 1,
            workers,
            cancel_flag,
        ))
        .await?;
        results.extend(more);
    } else if !still_failed.is_empty() {
        for &idx in &still_failed {
            info!(
                "[#{}] 重试{}次后仍失败: '{:.30}...'",
                idx, current_retry, data[idx].0
            );
            let last_cause = failure_causes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(&idx)
                .cloned();
            let mut records = error_records.lock().unwrap_or_else(|e| e.into_inner());
            records.push(ErrorRecord {
                index: idx,
                src_text: data[idx].0.clone(),
                dst_text: data[idx].1.clone().unwrap_or_default(),
                error_type: "RETRY_FAILED".to_string(),
                fixed: false,
                fix_details: format!("Failed after {} retries", current_retry),
                retry_count: current_retry,
                stage: Some("retry".to_string()),
                details: Some(format!("Failed after {} retries", current_retry)),
                last_error_type: last_cause.as_ref().map(|cause| cause.error_type.clone()),
                last_error_details: last_cause.as_ref().map(|cause| cause.details.clone()),
            });
        }
    }

    Ok(results)
}

// ============================================================================
// EPUB Translation Pipeline
// ============================================================================

pub async fn translate_epub(
    input_epub: &Path,
    output_epub: &Path,
    config: &PipelineConfig,
    progress_cb: ProgressCallback,
    cancel_flag: CancelFlag,
) -> Result<bool> {
    config.validate()?;

    // Early cancel check before any work
    if is_cancelled(&cancel_flag) {
        return Ok(false);
    }

    let checker = ResponseChecker::new("ja", "zh", 0.80, config.max_retry);
    let fixer = AutoFixer::new("ja", "zh");
    let error_records: Arc<Mutex<Vec<ErrorRecord>>> = Arc::new(Mutex::new(Vec::new()));

    // Initialize context detector
    let detector: Arc<Option<ContextDetector>> = Arc::new(load_context_detector(config));

    println!("{}", "=".repeat(60));
    println!("EPUB 日译中翻译流程");
    println!("{}", "=".repeat(60));
    println!("输入: {}", input_epub.display());
    println!("输出: {}", output_epub.display());
    println!("模式: {}", config.mode);
    println!("LLM: {} / {}", config.llm_url, config.model);
    println!(
        "批次大小: {}, 上下文行数: {}",
        config.batch_size, config.context_lines
    );
    println!("并行数: {}, 最大重试: {}", config.workers, config.max_retry);
    if let Some(ref gap) = config.translation_gap {
        println!("译文间距: {}", gap);
    }
    println!("{}", "=".repeat(60));
    emit_progress(
        &progress_cb,
        ProgressEvent::Log {
            message: format!(
                "EPUB 翻译: {} → {}",
                input_epub.display(),
                output_epub.display()
            ),
        },
    );

    // ========================================================================
    // Step 1: Load and extract EPUB
    // ========================================================================
    println!("\n[Step 1/5] 加载并解析 EPUB...");
    emit_progress(
        &progress_cb,
        ProgressEvent::StageStarted {
            stage: "Step 1/5".into(),
            detail: "加载并解析 EPUB...".into(),
        },
    );
    let mut handler = EpubHandler::new(&input_epub.to_string_lossy());
    handler.load()?;

    let data = handler.extract_translation_data();
    let all_src_lines: Vec<String> = data.iter().map(|b| b.src_text.clone()).collect();
    println!("提取到 {} 个可翻译文本块", data.len());
    emit_progress(
        &progress_cb,
        ProgressEvent::Log {
            message: format!("提取到 {} 个可翻译文本块", data.len()),
        },
    );

    // Precompute context for all lines (efficient: O(n) total instead of O(n^2))
    let precomputed: Arc<Option<PrecomputedContext>> = Arc::new(
        detector
            .as_ref()
            .as_ref()
            .map(|det| precompute_context(det, &all_src_lines)),
    );

    // Create work directory for intermediate data
    let work_dir = create_work_dir(output_epub)?;
    println!("工作目录: {}", work_dir.display());
    let json_path = work_dir
        .join("translation_data.json")
        .to_string_lossy()
        .to_string();
    let resume_manifest = build_resume_manifest("epub", input_epub, config)?;
    let resume_allowed = can_resume_from_manifest(&work_dir, &resume_manifest)?;

    // Check for existing translation data (resume support)
    let mut translations: HashMap<usize, String> = HashMap::new();
    if resume_allowed && std::path::Path::new(&json_path).exists() {
        if let Ok(existing_data) = EpubHandler::load_translation_data(&json_path) {
            let mut resumed = 0usize;
            for (i, block) in existing_data.iter().enumerate() {
                if let Some(ref dst) = block.dst_text {
                    if !dst.is_empty() && i < data.len() && block.src_text == data[i].src_text {
                        translations.insert(i, dst.clone());
                        resumed += 1;
                    }
                }
            }
            if resumed > 0 {
                println!("从已有翻译数据恢复了 {} 个译文", resumed);
            }
        }
    }

    EpubHandler::save_translation_data(&data, &json_path)?;
    save_resume_manifest(&work_dir, &resume_manifest)?;

    // ========================================================================
    // Step 2: Translate with LLM
    // ========================================================================
    println!("\n[Step 2/5] 调用 LLM 进行翻译...");
    emit_progress(
        &progress_cb,
        ProgressEvent::StageStarted {
            stage: "Step 2/5".into(),
            detail: "调用 LLM 进行翻译...".into(),
        },
    );

    let glossary_text = load_glossary_text(config);
    let orion_glossary_text = load_orion_glossary_text(config);
    let llm = LlmClient::with_params(
        &config.llm_url,
        &config.model,
        3,
        config.temperature,
        Some(config.top_p),
        Some(config.top_k),
        glossary_text.clone(),
        orion_glossary_text.clone(),
        config.api_key.clone(),
    )?;
    let total_batches = (data.len() + config.batch_size - 1) / config.batch_size;
    let batch_indices: Vec<usize> = (0..data.len()).step_by(config.batch_size).collect();

    let pb = ProgressBar::new(total_batches as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap_or(ProgressStyle::default_bar()),
    );
    pb.set_message("翻译中");

    let mut translated_count = 0usize;
    let mut fixed_count = 0usize;
    let mut failed_count = 0usize;
    let mut all_failed_indices: Vec<usize> = Vec::new();
    let mut quality_failed_indices: Vec<usize> = Vec::new();
    let mut api_failed_indices: Vec<usize> = Vec::new();
    let mut failure_causes: HashMap<usize, FailureCause> = HashMap::new();
    let mut corrected_count = 0usize;

    // Process batches (concurrent with semaphore for workers > 1)
    if config.workers <= 1 {
        for (batch_num, &start_idx) in batch_indices.iter().enumerate() {
            if is_cancelled(&cancel_flag) {
                emit_progress(
                    &progress_cb,
                    ProgressEvent::Error {
                        message: "用户已取消翻译".into(),
                    },
                );
                return Ok(false);
            }

            let end_idx = (start_idx + config.batch_size).min(data.len());

            // Skip batch if all blocks already translated (resume support)
            let untranslated: Vec<usize> = (start_idx..end_idx)
                .filter(|i| !translations.contains_key(i))
                .collect();
            if untranslated.is_empty() {
                translated_count += end_idx - start_idx;
                pb.inc(1);
                continue;
            }

            let texts: Vec<String> = data[start_idx..end_idx]
                .iter()
                .map(|b| b.src_text.clone())
                .collect();

            let context = build_context_for_batch_precomputed(
                precomputed.as_ref().as_ref(),
                &all_src_lines,
                start_idx,
                end_idx,
                config.context_lines,
                data.len(),
            );

            match llm
                .translate_batch_detailed(&texts, &context, &batch_num.to_string())
                .await
            {
                Ok(response) => {
                    let outcome = process_batch_response(
                        response,
                        start_idx,
                        end_idx,
                        |idx| data[idx].src_text.clone(),
                        &checker,
                        &fixer,
                        &error_records,
                    );
                    for (idx, translated) in outcome.fixed {
                        translations.insert(idx, translated);
                        translated_count += 1;
                    }
                    corrected_count += outcome.corrected_count;
                    extend_failure_causes(&mut failure_causes, outcome.failure_causes);
                    extend_unique(&mut quality_failed_indices, outcome.quality_failed.clone());
                    extend_unique(&mut api_failed_indices, outcome.api_failed.clone());
                    extend_unique(&mut all_failed_indices, outcome.quality_failed);
                    extend_unique(&mut all_failed_indices, outcome.api_failed);
                }
                Err(e) => {
                    warn!("批次 {} API 错误: {}", batch_num, e);
                    let failed_range: Vec<usize> = (start_idx..end_idx).collect();
                    for idx in &failed_range {
                        failure_causes.insert(
                            *idx,
                            FailureCause::new("API_ERROR", e.to_string(), "first_pass_api", 0),
                        );
                    }
                    extend_unique(&mut api_failed_indices, failed_range.clone());
                    extend_unique(&mut all_failed_indices, failed_range);
                    emit_progress(
                        &progress_cb,
                        ProgressEvent::Log {
                            message: format!("批次 {} API 错误，将进入重试: {}", batch_num + 1, e),
                        },
                    );
                }
            }

            pb.inc(1);
            emit_progress(
                &progress_cb,
                batch_progress_event(
                    batch_num + 1,
                    total_batches,
                    data.len(),
                    translated_count,
                    corrected_count,
                    quality_failed_indices.len(),
                    api_failed_indices.len(),
                ),
            );
        }
    } else {
        // Concurrent processing
        let semaphore = Arc::new(tokio::sync::Semaphore::new(config.workers));
        let llm = Arc::new(llm);
        let detector_arc = detector.clone();
        let precomputed_arc = precomputed.clone();
        let all_src_lines = Arc::new(all_src_lines.clone());
        let data_arc = Arc::new(data.clone());
        let checker = Arc::new(checker);
        let fixer = Arc::new(fixer);
        let error_records_clone = error_records.clone();

        let mut join_set = tokio::task::JoinSet::new();
        let mut completed_batches = 0usize;

        for (batch_num, &start_idx) in batch_indices.iter().enumerate() {
            let end_idx = (start_idx + config.batch_size).min(data.len());

            // Skip batch if all blocks already translated (resume support)
            let all_done = (start_idx..end_idx).all(|i| translations.contains_key(&i));
            if all_done {
                translated_count += end_idx - start_idx;
                pb.inc(1);
                completed_batches += 1;
                emit_progress(
                    &progress_cb,
                    batch_progress_event(
                        completed_batches,
                        total_batches,
                        data.len(),
                        translated_count,
                        corrected_count,
                        quality_failed_indices.len(),
                        api_failed_indices.len(),
                    ),
                );
                continue;
            }

            let sem = semaphore.clone();
            let llm = llm.clone();
            let _det = detector_arc.clone();
            let pc = precomputed_arc.clone();
            let src_lines = all_src_lines.clone();
            let data_c = data_arc.clone();
            let chk = checker.clone();
            let fix = fixer.clone();
            let err_rec = error_records_clone.clone();
            let batch_size = config.batch_size;
            let context_lines = config.context_lines;

            join_set.spawn(async move {
                let end_idx = (start_idx + batch_size).min(data_c.len());
                let result = async {
                    let _permit = sem.acquire().await.map_err(|e| anyhow::anyhow!("{}", e))?;

                    let texts: Vec<String> = data_c[start_idx..end_idx]
                        .iter()
                        .map(|b| b.src_text.clone())
                        .collect();

                    let context = build_context_for_batch_precomputed(
                        pc.as_ref().as_ref(),
                        &src_lines,
                        start_idx,
                        end_idx,
                        context_lines,
                        data_c.len(),
                    );

                    let response = llm
                        .translate_batch_detailed(&texts, &context, &batch_num.to_string())
                        .await?;

                    Ok::<_, anyhow::Error>(process_batch_response(
                        response,
                        start_idx,
                        end_idx,
                        |idx| data_c[idx].src_text.clone(),
                        &chk,
                        &fix,
                        &err_rec,
                    ))
                }
                .await;

                (start_idx, end_idx, result)
            });
        }

        while !join_set.is_empty() {
            if is_cancelled(&cancel_flag) {
                join_set.abort_all();
                emit_progress(
                    &progress_cb,
                    ProgressEvent::Error {
                        message: "用户已取消翻译".into(),
                    },
                );
                return Ok(false);
            }

            let maybe_joined =
                tokio::time::timeout(Duration::from_millis(120), join_set.join_next())
                    .await
                    .ok()
                    .flatten();

            let Some(join_result) = maybe_joined else {
                continue;
            };

            completed_batches += 1;
            pb.inc(1);

            match join_result {
                Ok((_start_idx, _end_idx, Ok(outcome))) => {
                    for (idx, translated) in outcome.fixed {
                        translations.insert(idx, translated);
                        translated_count += 1;
                    }
                    corrected_count += outcome.corrected_count;
                    extend_failure_causes(&mut failure_causes, outcome.failure_causes);
                    extend_unique(&mut quality_failed_indices, outcome.quality_failed.clone());
                    extend_unique(&mut api_failed_indices, outcome.api_failed.clone());
                    extend_unique(&mut all_failed_indices, outcome.quality_failed);
                    extend_unique(&mut all_failed_indices, outcome.api_failed);
                    emit_progress(
                        &progress_cb,
                        batch_progress_event(
                            completed_batches,
                            total_batches,
                            data.len(),
                            translated_count,
                            corrected_count,
                            quality_failed_indices.len(),
                            api_failed_indices.len(),
                        ),
                    );
                }
                Ok((start_idx, end_idx, Err(e))) => {
                    warn!("批次处理错误: {}", e);
                    let failed_range: Vec<usize> = (start_idx..end_idx).collect();
                    for idx in &failed_range {
                        failure_causes.insert(
                            *idx,
                            FailureCause::new("API_ERROR", e.to_string(), "first_pass_api", 0),
                        );
                    }
                    extend_unique(&mut api_failed_indices, failed_range.clone());
                    extend_unique(&mut all_failed_indices, failed_range);
                    emit_progress(
                        &progress_cb,
                        ProgressEvent::Log {
                            message: format!("批次错误: {}", e),
                        },
                    );
                    emit_progress(
                        &progress_cb,
                        batch_progress_event(
                            completed_batches,
                            total_batches,
                            data.len(),
                            translated_count,
                            corrected_count,
                            quality_failed_indices.len(),
                            api_failed_indices.len(),
                        ),
                    );
                }
                Err(e) => {
                    warn!("Task join error: {}", e);
                    emit_progress(
                        &progress_cb,
                        batch_progress_event(
                            completed_batches,
                            total_batches,
                            data.len(),
                            translated_count,
                            corrected_count,
                            quality_failed_indices.len(),
                            api_failed_indices.len(),
                        ),
                    );
                }
            }
        }
    }

    pb.finish_with_message("翻译完成");

    // Save stage snapshot after first pass
    save_epub_stage(
        &work_dir,
        "step2_first_pass",
        &data,
        &translations,
        &all_failed_indices,
        &failure_causes,
    )?;

    // Also save intermediate translation_data.json for crash recovery
    {
        let mut mid_data: Vec<TranslationBlock> = data.clone();
        for (idx, translated) in &translations {
            if *idx < mid_data.len() {
                mid_data[*idx].dst_text = Some(translated.clone());
            }
        }
        EpubHandler::save_translation_data(&mid_data, &json_path)?;
    }

    // ========================================================================
    // Step 3: Retry failed translations
    // ========================================================================
    if !all_failed_indices.is_empty() {
        if is_cancelled(&cancel_flag) {
            emit_progress(
                &progress_cb,
                ProgressEvent::Error {
                    message: "用户已取消翻译".into(),
                },
            );
            return Ok(false);
        }

        println!(
            "\n[Step 3/5] 重试 {} 个失败的翻译...",
            all_failed_indices.len()
        );
        emit_progress(
            &progress_cb,
            ProgressEvent::StageStarted {
                stage: "Step 3/5".into(),
                detail: format!("重试 {} 个失败的翻译...", all_failed_indices.len()),
            },
        );

        // Build data pairs for retry
        let data_pairs: Vec<(String, Option<String>)> = data
            .iter()
            .enumerate()
            .map(|(i, b)| (b.src_text.clone(), translations.get(&i).cloned()))
            .collect();

        let llm_retry = LlmClient::with_params(
            &config.llm_url,
            &config.model,
            1,
            config.temperature,
            Some(config.top_p),
            Some(config.top_k),
            glossary_text.clone(),
            orion_glossary_text.clone(),
            config.api_key.clone(),
        )?;
        let checker_retry = ResponseChecker::new("ja", "zh", 0.80, config.max_retry);
        let fixer_retry = AutoFixer::new("ja", "zh");
        let failure_causes_shared = Arc::new(Mutex::new(failure_causes.clone()));

        let retry_results = retry_failed_with_context(
            &llm_retry,
            &checker_retry,
            &fixer_retry,
            detector.as_ref().as_ref(),
            &all_src_lines,
            &data_pairs,
            &all_failed_indices,
            config.context_lines,
            &error_records,
            &failure_causes_shared,
            config.max_retry,
            1,
            config.workers,
            &cancel_flag,
        )
        .await?;
        failure_causes = failure_causes_shared
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();

        for (idx, translated) in &retry_results {
            translations.insert(*idx, translated.clone());
            translated_count += 1;
            fixed_count += 1;
        }

        for &idx in &all_failed_indices {
            if !retry_results.contains_key(&idx) {
                failed_count += 1;
            }
        }

        if is_cancelled(&cancel_flag) {
            println!("重试被用户取消");
        } else {
            println!(
                "重试完成: {} 个修复, {} 个仍失败",
                retry_results.len(),
                all_failed_indices.len() - retry_results.len()
            );
        }
    } else {
        println!("\n[Step 3/5] 无需重试");
    }

    // Save stage snapshot after retry
    {
        let remaining_failed: Vec<usize> = all_failed_indices
            .iter()
            .filter(|idx| !translations.contains_key(idx))
            .copied()
            .collect();
        save_epub_stage(
            &work_dir,
            "step3_after_retry",
            &data,
            &translations,
            &remaining_failed,
            &failure_causes,
        )?;
    }

    // Save translation data with results
    let mut final_data: Vec<TranslationBlock> = data.clone();
    for (idx, translated) in &translations {
        if *idx < final_data.len() {
            final_data[*idx].dst_text = Some(translated.clone());
        }
    }
    EpubHandler::save_translation_data(&final_data, &json_path)?;
    println!("翻译数据已保存: {}", json_path);

    // ========================================================================
    // Step 4: Inject translations back to EPUB
    // ========================================================================
    println!("\n[Step 4/5] 将译文注入 EPUB (模式: {})...", config.mode);
    if is_cancelled(&cancel_flag) {
        emit_progress(
            &progress_cb,
            ProgressEvent::Error {
                message: "用户已取消翻译".into(),
            },
        );
        return Ok(false);
    }

    emit_progress(
        &progress_cb,
        ProgressEvent::StageStarted {
            stage: "Step 4/5".into(),
            detail: "将译文注入 EPUB...".into(),
        },
    );

    let mut handler2 = EpubHandler::new(&input_epub.to_string_lossy());
    handler2.load()?;
    handler2.inject_translations(&final_data, config.mode, config.translation_gap.as_deref())?;

    // ========================================================================
    // Step 5: Apply fixes and save
    // ========================================================================
    println!("\n[Step 5/5] 应用格式修复并保存...");
    emit_progress(
        &progress_cb,
        ProgressEvent::StageStarted {
            stage: "Step 5/5".into(),
            detail: "应用格式修复并保存...".into(),
        },
    );
    handler2.save(&output_epub.to_string_lossy(), config.apply_fixes)?;
    println!("输出 EPUB: {}", output_epub.display());

    // Generate error report
    let error_summary = {
        let records = error_records.lock().unwrap_or_else(|e| e.into_inner());
        if !records.is_empty() {
            println!("\n生成错误报告...");
            let metadata = epub_report_metadata(&data);
            let summary = generate_error_report(&records, &work_dir, Some(&metadata))?;
            summary
        } else {
            HashMap::new()
        }
    };

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("翻译完成");
    println!("{}", "=".repeat(60));
    println!("总文本块: {}", data.len());
    println!("已翻译: {}", translated_count);
    println!("自动纠正: {}", corrected_count);
    println!("重试修复: {}", fixed_count);
    println!("失败: {}", failed_count);
    println!("输出 EPUB: {}", output_epub.display());
    println!("工作目录: {}", work_dir.display());

    let total_errors = error_summary
        .get("total_errors")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    if total_errors > 0 {
        println!("\n--- 错误统计 ---");
        println!("总问题数: {}", total_errors);
        println!(
            "  已修复: {}",
            error_summary
                .get("fixed")
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
        );
        println!(
            "  未修复: {}",
            error_summary
                .get("failed")
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
        );
    }

    let error_report_path = error_summary
        .get("report_path")
        .and_then(|v| v.as_str())
        .map(|path| path.to_string());
    emit_progress(
        &progress_cb,
        completion_event(
            data.len(),
            translated_count,
            fixed_count + corrected_count,
            failed_count,
            output_epub.display().to_string(),
            error_report_path,
        ),
    );

    Ok(true)
}

// ============================================================================
// Export helpers (re-export from existing translation data, no LLM call)
// ============================================================================

/// Export an EPUB using pre-existing translation data (no LLM call).
/// This loads the original EPUB, injects translations with the given mode,
/// and saves to `output_epub`.
pub fn export_epub_from_data(
    input_epub: &Path,
    translation_data: &[TranslationBlock],
    output_epub: &Path,
    mode: crate::config::TranslationMode,
    translation_gap: Option<&str>,
    apply_fixes: bool,
    progress_cb: &ProgressCallback,
) -> Result<()> {
    emit_progress(
        progress_cb,
        ProgressEvent::Log {
            message: format!("导出 {} 模式 → {}", mode, output_epub.display()),
        },
    );
    let mut handler = EpubHandler::new(&input_epub.to_string_lossy());
    handler.load()?;
    handler.inject_translations(translation_data, mode, translation_gap)?;
    handler.save(&output_epub.to_string_lossy(), apply_fixes)?;
    emit_progress(
        progress_cb,
        ProgressEvent::Log {
            message: format!("导出完成: {}", output_epub.display()),
        },
    );
    Ok(())
}

/// Export a TXT using pre-existing translation data (no LLM call).
pub fn export_txt_from_data(
    translation_data: &[txt::TxtBlock],
    output_txt: &Path,
    mode: crate::config::TranslationMode,
    progress_cb: &ProgressCallback,
) -> Result<()> {
    emit_progress(
        progress_cb,
        ProgressEvent::Log {
            message: format!("导出 {} 模式 → {}", mode, output_txt.display()),
        },
    );
    txt::write_txt_output(translation_data, output_txt, mode)?;
    emit_progress(
        progress_cb,
        ProgressEvent::Log {
            message: format!("导出完成: {}", output_txt.display()),
        },
    );
    Ok(())
}

/// Load TXT translation data from a sidecar JSON file.
pub fn load_txt_translation_data(json_path: &str) -> Result<Vec<txt::TxtBlock>> {
    let content = std::fs::read_to_string(json_path)?;
    let data: Vec<txt::TxtBlock> = serde_json::from_str(&content)?;
    Ok(data)
}

// ============================================================================
// TXT Translation Pipeline
// ============================================================================

pub async fn translate_txt(
    input_txt: &Path,
    output_txt: &Path,
    config: &PipelineConfig,
    progress_cb: ProgressCallback,
    cancel_flag: CancelFlag,
) -> Result<bool> {
    config.validate()?;

    // Early cancel check before any work
    if is_cancelled(&cancel_flag) {
        return Ok(false);
    }

    let checker = ResponseChecker::new("ja", "zh", 0.80, config.max_retry);
    let fixer = AutoFixer::new("ja", "zh");
    let error_records: Arc<Mutex<Vec<ErrorRecord>>> = Arc::new(Mutex::new(Vec::new()));

    let detector = load_context_detector(config);

    println!("{}", "=".repeat(60));
    println!("TXT 日译中翻译流程");
    println!("{}", "=".repeat(60));
    println!("输入: {}", input_txt.display());
    println!("输出: {}", output_txt.display());
    println!("模式: {}", config.mode);
    println!("LLM: {} / {}", config.llm_url, config.model);
    println!(
        "批次大小: {}, 上下文行数: {}",
        config.batch_size, config.context_lines
    );
    println!("并行数: {}, 最大重试: {}", config.workers, config.max_retry);
    println!("{}", "=".repeat(60));
    emit_progress(
        &progress_cb,
        ProgressEvent::Log {
            message: format!(
                "TXT 翻译: {} → {}",
                input_txt.display(),
                output_txt.display()
            ),
        },
    );

    // Step 1: Load TXT
    println!("\n[Step 1/5] 加载并解析 TXT...");
    emit_progress(
        &progress_cb,
        ProgressEvent::StageStarted {
            stage: "Step 1/5".into(),
            detail: "加载并解析 TXT...".into(),
        },
    );
    let mut data = txt::read_txt_data(input_txt)?;
    println!("提取到 {} 个可翻译文本行", data.len());
    emit_progress(
        &progress_cb,
        ProgressEvent::Log {
            message: format!("提取到 {} 个可翻译文本行", data.len()),
        },
    );

    // Create work directory for intermediate data
    let work_dir = create_work_dir(output_txt)?;
    println!("工作目录: {}", work_dir.display());
    let txt_json_path = work_dir
        .join("translation_data.json")
        .to_string_lossy()
        .to_string();
    let resume_manifest = build_resume_manifest("txt", input_txt, config)?;
    let resume_allowed = can_resume_from_manifest(&work_dir, &resume_manifest)?;

    // Check for existing translation data (resume support, same as EPUB)
    let mut translations: HashMap<usize, String> = HashMap::new();
    if resume_allowed && std::path::Path::new(&txt_json_path).exists() {
        if let Ok(existing_data) = load_txt_translation_data(&txt_json_path) {
            let mut resumed = 0usize;
            for (i, block) in existing_data.iter().enumerate() {
                if let Some(ref dst) = block.dst_text {
                    if !dst.is_empty() && i < data.len() && block.src_text == data[i].src_text {
                        translations.insert(i, dst.clone());
                        resumed += 1;
                    }
                }
            }
            if resumed > 0 {
                println!("从已有翻译数据恢复了 {} 个译文", resumed);
                emit_progress(
                    &progress_cb,
                    ProgressEvent::Log {
                        message: format!("从已有翻译数据恢复了 {} 个译文", resumed),
                    },
                );
            }
        }
    }

    // Save initial data to disk (crash recovery)
    let json_str_init = serde_json::to_string_pretty(&data)?;
    std::fs::write(&txt_json_path, &json_str_init)?;
    save_resume_manifest(&work_dir, &resume_manifest)?;

    // Step 2: Translate
    println!("\n[Step 2/5] 调用 LLM 进行翻译...");
    emit_progress(
        &progress_cb,
        ProgressEvent::StageStarted {
            stage: "Step 2/5".into(),
            detail: "调用 LLM 进行翻译...".into(),
        },
    );
    let glossary_text = load_glossary_text(config);
    let orion_glossary_text = load_orion_glossary_text(config);
    let llm = LlmClient::with_params(
        &config.llm_url,
        &config.model,
        3,
        config.temperature,
        Some(config.top_p),
        Some(config.top_k),
        glossary_text.clone(),
        orion_glossary_text.clone(),
        config.api_key.clone(),
    )?;
    let total_batches = (data.len() + config.batch_size - 1) / config.batch_size;
    let batch_indices: Vec<usize> = (0..data.len()).step_by(config.batch_size).collect();

    let pb = ProgressBar::new(total_batches as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} 翻译中")
            .unwrap_or(ProgressStyle::default_bar()),
    );

    let all_src_lines: Arc<Vec<String>> =
        Arc::new(data.iter().map(|b| b.src_text.clone()).collect());
    let detector: Arc<Option<ContextDetector>> = Arc::new(detector);

    // Precompute context for all lines
    let precomputed: Arc<Option<PrecomputedContext>> = Arc::new(
        detector
            .as_ref()
            .as_ref()
            .map(|det| precompute_context(det, &all_src_lines)),
    );

    let mut translated_count = 0usize;
    let mut fixed_count = 0usize;
    let mut failed_count = 0usize;
    let mut all_failed_indices: Vec<usize> = Vec::new();
    let mut quality_failed_indices: Vec<usize> = Vec::new();
    let mut api_failed_indices: Vec<usize> = Vec::new();
    let mut failure_causes: HashMap<usize, FailureCause> = HashMap::new();
    let mut corrected_count = 0usize;

    if config.workers <= 1 {
        // Sequential processing
        for (batch_num, &start_idx) in batch_indices.iter().enumerate() {
            if is_cancelled(&cancel_flag) {
                emit_progress(
                    &progress_cb,
                    ProgressEvent::Error {
                        message: "用户已取消翻译".into(),
                    },
                );
                return Ok(false);
            }

            let end_idx = (start_idx + config.batch_size).min(data.len());

            // Skip batch if all lines already translated (resume support)
            let untranslated: Vec<usize> = (start_idx..end_idx)
                .filter(|i| !translations.contains_key(i))
                .collect();
            if untranslated.is_empty() {
                translated_count += end_idx - start_idx;
                pb.inc(1);
                continue;
            }

            let texts: Vec<String> = data[start_idx..end_idx]
                .iter()
                .map(|b| b.src_text.clone())
                .collect();

            let context = build_context_for_batch_precomputed(
                precomputed.as_ref().as_ref(),
                &all_src_lines,
                start_idx,
                end_idx,
                config.context_lines,
                data.len(),
            );

            match llm
                .translate_batch_detailed(&texts, &context, &batch_num.to_string())
                .await
            {
                Ok(response) => {
                    let outcome = process_batch_response(
                        response,
                        start_idx,
                        end_idx,
                        |idx| data[idx].src_text.clone(),
                        &checker,
                        &fixer,
                        &error_records,
                    );
                    for (idx, translated) in outcome.fixed {
                        translations.insert(idx, translated);
                        translated_count += 1;
                    }
                    corrected_count += outcome.corrected_count;
                    extend_failure_causes(&mut failure_causes, outcome.failure_causes);
                    extend_unique(&mut quality_failed_indices, outcome.quality_failed.clone());
                    extend_unique(&mut api_failed_indices, outcome.api_failed.clone());
                    extend_unique(&mut all_failed_indices, outcome.quality_failed);
                    extend_unique(&mut all_failed_indices, outcome.api_failed);
                }
                Err(e) => {
                    warn!("TXT 批次 {} API 错误: {}", batch_num, e);
                    let failed_range: Vec<usize> = (start_idx..end_idx).collect();
                    for idx in &failed_range {
                        failure_causes.insert(
                            *idx,
                            FailureCause::new("API_ERROR", e.to_string(), "first_pass_api", 0),
                        );
                    }
                    extend_unique(&mut api_failed_indices, failed_range.clone());
                    extend_unique(&mut all_failed_indices, failed_range);
                    emit_progress(
                        &progress_cb,
                        ProgressEvent::Log {
                            message: format!("批次 {} API 错误，将进入重试: {}", batch_num + 1, e),
                        },
                    );
                }
            }

            pb.inc(1);
            emit_progress(
                &progress_cb,
                batch_progress_event(
                    batch_num + 1,
                    total_batches,
                    data.len(),
                    translated_count,
                    corrected_count,
                    quality_failed_indices.len(),
                    api_failed_indices.len(),
                ),
            );
        }
    } else {
        // Concurrent processing with JoinSet
        let semaphore = Arc::new(tokio::sync::Semaphore::new(config.workers));
        let llm = Arc::new(llm);
        let data_arc = Arc::new(data.clone());
        let checker = Arc::new(checker);
        let fixer = Arc::new(fixer);
        let error_records_clone = error_records.clone();

        let mut join_set = tokio::task::JoinSet::new();
        let mut completed_batches = 0usize;

        for (batch_num, &start_idx) in batch_indices.iter().enumerate() {
            let end_idx = (start_idx + config.batch_size).min(data.len());

            // Skip batch if all lines already translated (resume support)
            let all_done = (start_idx..end_idx).all(|i| translations.contains_key(&i));
            if all_done {
                translated_count += end_idx - start_idx;
                pb.inc(1);
                completed_batches += 1;
                emit_progress(
                    &progress_cb,
                    batch_progress_event(
                        completed_batches,
                        total_batches,
                        data.len(),
                        translated_count,
                        corrected_count,
                        quality_failed_indices.len(),
                        api_failed_indices.len(),
                    ),
                );
                continue;
            }

            let sem = semaphore.clone();
            let llm = llm.clone();
            let pc = precomputed.clone();
            let src_lines = all_src_lines.clone();
            let data_c = data_arc.clone();
            let chk = checker.clone();
            let fix = fixer.clone();
            let err_rec = error_records_clone.clone();
            let batch_size = config.batch_size;
            let context_lines = config.context_lines;

            join_set.spawn(async move {
                let end_idx = (start_idx + batch_size).min(data_c.len());
                let result = async {
                    let _permit = sem.acquire().await.map_err(|e| anyhow::anyhow!("{}", e))?;

                    let texts: Vec<String> = data_c[start_idx..end_idx]
                        .iter()
                        .map(|b| b.src_text.clone())
                        .collect();

                    let context = build_context_for_batch_precomputed(
                        pc.as_ref().as_ref(),
                        &src_lines,
                        start_idx,
                        end_idx,
                        context_lines,
                        data_c.len(),
                    );

                    let response = llm
                        .translate_batch_detailed(&texts, &context, &batch_num.to_string())
                        .await?;

                    Ok::<_, anyhow::Error>(process_batch_response(
                        response,
                        start_idx,
                        end_idx,
                        |idx| data_c[idx].src_text.clone(),
                        &chk,
                        &fix,
                        &err_rec,
                    ))
                }
                .await;

                (start_idx, end_idx, result)
            });
        }

        while !join_set.is_empty() {
            if is_cancelled(&cancel_flag) {
                join_set.abort_all();
                emit_progress(
                    &progress_cb,
                    ProgressEvent::Error {
                        message: "用户已取消翻译".into(),
                    },
                );
                return Ok(false);
            }

            let maybe_joined =
                tokio::time::timeout(Duration::from_millis(120), join_set.join_next())
                    .await
                    .ok()
                    .flatten();

            let Some(join_result) = maybe_joined else {
                continue;
            };

            completed_batches += 1;
            pb.inc(1);

            match join_result {
                Ok((_start_idx, _end_idx, Ok(outcome))) => {
                    for (idx, translated) in outcome.fixed {
                        translations.insert(idx, translated);
                        translated_count += 1;
                    }
                    corrected_count += outcome.corrected_count;
                    extend_failure_causes(&mut failure_causes, outcome.failure_causes);
                    extend_unique(&mut quality_failed_indices, outcome.quality_failed.clone());
                    extend_unique(&mut api_failed_indices, outcome.api_failed.clone());
                    extend_unique(&mut all_failed_indices, outcome.quality_failed);
                    extend_unique(&mut all_failed_indices, outcome.api_failed);
                    emit_progress(
                        &progress_cb,
                        batch_progress_event(
                            completed_batches,
                            total_batches,
                            data.len(),
                            translated_count,
                            corrected_count,
                            quality_failed_indices.len(),
                            api_failed_indices.len(),
                        ),
                    );
                }
                Ok((start_idx, end_idx, Err(e))) => {
                    warn!("TXT 批次处理错误: {}", e);
                    let failed_range: Vec<usize> = (start_idx..end_idx).collect();
                    for idx in &failed_range {
                        failure_causes.insert(
                            *idx,
                            FailureCause::new("API_ERROR", e.to_string(), "first_pass_api", 0),
                        );
                    }
                    extend_unique(&mut api_failed_indices, failed_range.clone());
                    extend_unique(&mut all_failed_indices, failed_range);
                    emit_progress(
                        &progress_cb,
                        ProgressEvent::Log {
                            message: format!("批次错误: {}", e),
                        },
                    );
                    emit_progress(
                        &progress_cb,
                        batch_progress_event(
                            completed_batches,
                            total_batches,
                            data.len(),
                            translated_count,
                            corrected_count,
                            quality_failed_indices.len(),
                            api_failed_indices.len(),
                        ),
                    );
                }
                Err(e) => {
                    warn!("Task join error: {}", e);
                    emit_progress(
                        &progress_cb,
                        batch_progress_event(
                            completed_batches,
                            total_batches,
                            data.len(),
                            translated_count,
                            corrected_count,
                            quality_failed_indices.len(),
                            api_failed_indices.len(),
                        ),
                    );
                }
            }
        }
    }

    pb.finish_with_message("翻译完成");

    // Save stage snapshot after first pass
    save_txt_stage(
        &work_dir,
        "step2_first_pass",
        &data,
        &translations,
        &all_failed_indices,
        &failure_causes,
    )?;

    // Also save intermediate translation_data.json for crash recovery
    {
        let mut mid_data = data.clone();
        for (idx, translated) in &translations {
            if *idx < mid_data.len() {
                mid_data[*idx].dst_text = Some(translated.clone());
            }
        }
        let json_str_mid = serde_json::to_string_pretty(&mid_data)?;
        std::fs::write(&txt_json_path, &json_str_mid)?;
    }

    // Step 3: Retry
    if !all_failed_indices.is_empty() {
        if is_cancelled(&cancel_flag) {
            emit_progress(
                &progress_cb,
                ProgressEvent::Error {
                    message: "用户已取消翻译".into(),
                },
            );
            return Ok(false);
        }

        println!(
            "\n[Step 3/5] 重试 {} 个失败的翻译...",
            all_failed_indices.len()
        );
        emit_progress(
            &progress_cb,
            ProgressEvent::StageStarted {
                stage: "Step 3/5".into(),
                detail: format!("重试 {} 个失败的翻译...", all_failed_indices.len()),
            },
        );

        let data_pairs: Vec<(String, Option<String>)> = data
            .iter()
            .enumerate()
            .map(|(i, b)| (b.src_text.clone(), translations.get(&i).cloned()))
            .collect();

        let llm_retry = LlmClient::with_params(
            &config.llm_url,
            &config.model,
            1,
            config.temperature,
            Some(config.top_p),
            Some(config.top_k),
            glossary_text.clone(),
            orion_glossary_text.clone(),
            config.api_key.clone(),
        )?;
        let checker_retry = ResponseChecker::new("ja", "zh", 0.80, config.max_retry);
        let fixer_retry = AutoFixer::new("ja", "zh");
        let failure_causes_shared = Arc::new(Mutex::new(failure_causes.clone()));

        let retry_results = retry_failed_with_context(
            &llm_retry,
            &checker_retry,
            &fixer_retry,
            detector.as_ref().as_ref(),
            &all_src_lines,
            &data_pairs,
            &all_failed_indices,
            config.context_lines,
            &error_records,
            &failure_causes_shared,
            config.max_retry,
            1,
            config.workers,
            &cancel_flag,
        )
        .await?;
        failure_causes = failure_causes_shared
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();

        for (idx, translated) in &retry_results {
            translations.insert(*idx, translated.clone());
            translated_count += 1;
            fixed_count += 1;
        }

        for &idx in &all_failed_indices {
            if !retry_results.contains_key(&idx) {
                failed_count += 1;
            }
        }

        if is_cancelled(&cancel_flag) {
            println!("重试被用户取消");
        } else {
            println!(
                "重试完成: {} 个修复, {} 个仍失败",
                retry_results.len(),
                all_failed_indices.len() - retry_results.len()
            );
        }
    } else {
        println!("\n[Step 3/5] 无需重试");
    }

    // Save stage snapshot after retry
    {
        let remaining_failed: Vec<usize> = all_failed_indices
            .iter()
            .filter(|idx| !translations.contains_key(idx))
            .copied()
            .collect();
        save_txt_stage(
            &work_dir,
            "step3_after_retry",
            &data,
            &translations,
            &remaining_failed,
            &failure_causes,
        )?;
    }

    // Save translation data with results
    for (idx, translated) in &translations {
        if *idx < data.len() {
            data[*idx].dst_text = Some(translated.clone());
        }
    }
    let json_str = serde_json::to_string_pretty(&data)?;
    std::fs::write(&txt_json_path, &json_str)?;
    println!("翻译数据已保存: {}", txt_json_path);

    // Step 4: Write output
    println!("\n[Step 4/5] 写入输出 TXT (模式: {})...", config.mode);
    if is_cancelled(&cancel_flag) {
        emit_progress(
            &progress_cb,
            ProgressEvent::Error {
                message: "用户已取消翻译".into(),
            },
        );
        return Ok(false);
    }

    emit_progress(
        &progress_cb,
        ProgressEvent::StageStarted {
            stage: "Step 4/5".into(),
            detail: "写入输出 TXT...".into(),
        },
    );

    txt::write_txt_output(&data, output_txt, config.mode)?;
    println!("输出 TXT: {}", output_txt.display());

    // Step 5: Error report (same as EPUB)
    println!("\n[Step 5/5] 生成错误报告...");
    emit_progress(
        &progress_cb,
        ProgressEvent::StageStarted {
            stage: "Step 5/5".into(),
            detail: "生成错误报告...".into(),
        },
    );
    let error_summary = {
        let records = error_records.lock().unwrap_or_else(|e| e.into_inner());
        if !records.is_empty() {
            let metadata = txt_report_metadata(&data);
            let summary = generate_error_report(&records, &work_dir, Some(&metadata))?;
            summary
        } else {
            HashMap::new()
        }
    };

    let error_report_path = error_summary
        .get("report_path")
        .and_then(|v| v.as_str())
        .map(|path| path.to_string());
    emit_progress(
        &progress_cb,
        completion_event(
            data.len(),
            translated_count,
            fixed_count + corrected_count,
            failed_count,
            output_txt.display().to_string(),
            error_report_path,
        ),
    );

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("翻译完成");
    println!("{}", "=".repeat(60));
    println!("总文本行: {}", data.len());
    println!("已翻译: {}", translated_count);
    println!("自动纠正: {}", corrected_count);
    println!("重试修复: {}", fixed_count);
    println!("失败: {}", failed_count);
    println!("输出 TXT: {}", output_txt.display());
    println!("工作目录: {}", work_dir.display());

    let total_errors = error_summary
        .get("total_errors")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    if total_errors > 0 {
        println!("\n--- 错误统计 ---");
        println!("总问题数: {}", total_errors);
        println!(
            "  已修复: {}",
            error_summary
                .get("fixed")
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
        );
        println!(
            "  未修复: {}",
            error_summary
                .get("failed")
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
        );
    }

    Ok(true)
}
