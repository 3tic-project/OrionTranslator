use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use gpui::*;

use alnilam::pipeline::{self, ProgressCallback, ProgressEvent};
use alnilam::{
    config::{
        self, PipelineConfig, TranslationMode, DEFAULT_BATCH_SIZE, DEFAULT_CONTEXT_LINES,
        DEFAULT_LLM_URL, DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P,
    },
    llm::LlmClient,
};

use crate::app::OrionApp;
use crate::types::{GlossaryGenStatus, ModelPreset, TranslationStatus};

// ============================================================================
// Event Handlers
// ============================================================================

impl OrionApp {
    fn compact_single_line(text: &str, max_chars: usize) -> String {
        let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let chars: Vec<char> = compact.chars().collect();
        if chars.len() <= max_chars {
            compact
        } else {
            format!("{}…", chars[..max_chars].iter().collect::<String>())
        }
    }

    fn collect_cache_targets(path: &Path) -> Vec<PathBuf> {
        let mut targets = Vec::new();
        let parent = path.parent().unwrap_or(Path::new("."));
        let stem = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let legacy_suffixes = [
            "_translation_data.json",
            "_error_report.json",
            "_glossary.json",
            "_characters.json",
            "_output.json",
        ];

        if let Ok(entries) = std::fs::read_dir(parent) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                let file_name = entry.file_name().to_string_lossy().to_string();

                if entry_path.is_file()
                    && file_name.starts_with(&stem)
                    && legacy_suffixes
                        .iter()
                        .any(|suffix| file_name.ends_with(suffix))
                {
                    targets.push(entry_path.clone());
                }

                if entry_path.is_dir() {
                    let exact_work = format!("{}_work", stem);
                    let prefixed_work = format!("{}.", stem);
                    if file_name == exact_work
                        || (file_name.starts_with(&prefixed_work) && file_name.ends_with("_work"))
                    {
                        targets.push(entry_path);
                    }
                }
            }
        }

        targets
    }

    pub fn on_preset_changed(
        &mut self,
        selected_ix: &usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let preset = ModelPreset::from_index(*selected_ix);
        self.model_preset = preset;

        self.llm_url_input.update(cx, |state, cx| {
            state.set_value(preset.llm_url(), window, cx)
        });
        self.model_input.update(cx, |state, cx| {
            state.set_value(preset.model_name(), window, cx)
        });
        self.batch_size_input.update(cx, |state, cx| {
            state.set_value(preset.batch_size().to_string(), window, cx)
        });
        self.workers_input.update(cx, |state, cx| {
            state.set_value(preset.workers().to_string(), window, cx)
        });
        self.context_lines_input.update(cx, |state, cx| {
            state.set_value(preset.context_lines().to_string(), window, cx)
        });

        // Reset model test status since model changed
        self.model_test_ok = None;
        self.model_test_message = "未测试".into();

        self.add_log(&format!(
            "已切换预设: {} (URL: {}, 模型: {}, 批次: {}, 上下文: {})",
            preset.label(),
            preset.llm_url(),
            preset.model_name(),
            preset.batch_size(),
            preset.context_lines(),
        ));
        cx.notify();
    }

    pub fn pick_input_file(
        &mut self,
        _: &ClickEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let path_future = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("选择 EPUB 或 TXT 文件".into()),
        });

        let entity = cx.entity();
        cx.spawn(async move |_, cx| {
            if let Ok(Ok(Some(paths))) = path_future.await {
                if let Some(p) = paths.into_iter().next() {
                    _ = entity.update(cx, |this, cx| {
                        let file_name = p
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| p.display().to_string());
                        this.add_log(&format!("已选择文件: {}", file_name));

                        let (bilingual, mono) = Self::compute_output_paths(&p);
                        this.output_bilingual_path = Some(bilingual);
                        this.output_mono_path = Some(mono);
                        this.input_path = Some(p.clone());

                        // Auto-detect glossary file next to input
                        let stem = p.with_extension("").to_string_lossy().to_string();
                        let glossary_candidate =
                            std::path::PathBuf::from(format!("{}_glossary.json", stem));
                        if glossary_candidate.exists() {
                            let gname = glossary_candidate
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_default();
                            this.add_log(&format!("自动检测到术语表: {}", gname));
                            this.glossary_path = Some(glossary_candidate);
                        } else {
                            this.glossary_path = None;
                        }

                        cx.notify();
                    });
                }
            }
        })
        .detach();
    }

    fn parse_config_number<T>(value: &str, label: &str) -> Result<T>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        let value = value.trim();
        if value.is_empty() {
            return Err(anyhow!("{} 不能为空", label));
        }
        value
            .parse::<T>()
            .map_err(|e| anyhow!("{} 必须是有效数字，当前为 {:?}: {}", label, value, e))
    }

    pub fn build_config(&self, cx: &App, mode: TranslationMode) -> Result<PipelineConfig> {
        let llm_url = self.llm_url_input.read(cx).value().trim().to_string();
        let model = self.model_input.read(cx).value().trim().to_string();
        let api_key_raw = self.api_key_input.read(cx).value().to_string();
        let api_key = if api_key_raw.trim().is_empty() {
            None
        } else {
            Some(api_key_raw.trim().to_string())
        };
        let batch_size = Self::parse_config_number::<usize>(
            &self.batch_size_input.read(cx).value(),
            "批次大小",
        )?;
        let workers =
            Self::parse_config_number::<usize>(&self.workers_input.read(cx).value(), "并行数")?;
        let context_lines = Self::parse_config_number::<usize>(
            &self.context_lines_input.read(cx).value(),
            "上下文行数",
        )?;
        let max_retry = Self::parse_config_number::<usize>(
            &self.max_retry_input.read(cx).value(),
            "最大重试次数",
        )?;
        let temperature = Self::parse_config_number::<f64>(
            &self.temperature_input.read(cx).value(),
            "temperature",
        )?;
        let top_p = Self::parse_config_number::<f64>(&self.top_p_input.read(cx).value(), "top_p")?;
        let top_k = Self::parse_config_number::<u32>(&self.top_k_input.read(cx).value(), "top_k")?;
        Ok(PipelineConfig {
            llm_url: if llm_url.is_empty() {
                DEFAULT_LLM_URL.to_string()
            } else {
                llm_url
            },
            model: if model.is_empty() {
                DEFAULT_MODEL.to_string()
            } else {
                model
            },
            batch_size,
            context_lines,
            workers,
            max_retry,
            mode,
            apply_fixes: true,
            rules_path: None,
            translation_gap: Some(config::DEFAULT_TRANSLATION_GAP.to_string()),
            temperature,
            top_p,
            top_k,
            glossary_path: self.glossary_path.clone(),
            api_key,
        })
    }

    fn sanitize_model_name(model: &str) -> String {
        model
            .chars()
            .map(|c| match c {
                '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
                _ => c,
            })
            .collect()
    }

    fn add_model_suffix(path: &std::path::Path, model: &str) -> PathBuf {
        let tag = Self::sanitize_model_name(model);
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let parent = path.parent().unwrap_or(std::path::Path::new("."));
        if ext.is_empty() {
            parent.join(format!("{}[{}]", stem, tag))
        } else {
            parent.join(format!("{}[{}].{}", stem, tag, ext))
        }
    }

    pub fn start_translation(
        &mut self,
        _: &ClickEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let input_path = match &self.input_path {
            Some(p) => p.clone(),
            None => {
                self.add_log("错误: 请先选择输入文件");
                cx.notify();
                return;
            }
        };

        if !self.output_bilingual && !self.output_mono {
            self.add_log("错误: 请至少选择一种输出（双语或单语）");
            cx.notify();
            return;
        }

        let config_bilingual = match self.build_config(cx, TranslationMode::Bilingual) {
            Ok(config) => config,
            Err(e) => {
                self.add_log(&format!("配置错误: {}", e));
                cx.notify();
                return;
            }
        };
        let config_mono = match self.build_config(cx, TranslationMode::Replace) {
            Ok(config) => config,
            Err(e) => {
                self.add_log(&format!("配置错误: {}", e));
                cx.notify();
                return;
            }
        };

        if let Err(e) = config_bilingual.validate() {
            self.add_log(&format!("配置错误: {}", e));
            cx.notify();
            return;
        }
        if let Err(e) = config_mono.validate() {
            self.add_log(&format!("配置错误: {}", e));
            cx.notify();
            return;
        }
        for warning in config_bilingual.api_security_warnings() {
            self.add_log(&format!("安全提示: {}", warning));
        }

        let model_name = config_bilingual.model.clone();
        let output_bilingual_path = if self.output_bilingual {
            self.output_bilingual_path
                .as_ref()
                .map(|p| Self::add_model_suffix(p, &model_name))
        } else {
            None
        };
        let output_mono_path = if self.output_mono {
            self.output_mono_path
                .as_ref()
                .map(|p| Self::add_model_suffix(p, &model_name))
        } else {
            None
        };

        if self.output_bilingual && output_bilingual_path.is_none() {
            self.add_log("错误: 双语输出路径未设置");
            cx.notify();
            return;
        }
        if self.output_mono && output_mono_path.is_none() {
            self.add_log("错误: 单语输出路径未设置");
            cx.notify();
            return;
        }
        let llm_url = config_bilingual.llm_url.clone();
        let model = config_bilingual.model.clone();

        self.status = TranslationStatus::Running;
        self.run_id = self.run_id.wrapping_add(1);
        let current_run_id = self.run_id;
        self.progress = 0.0;
        self.progress_detail = "准备开始...".into();
        self.started_at = Some(Instant::now());
        self.finished_at = None;
        self.translated_lines = 0;
        self.remaining_lines = 0;
        self.total_lines = 0;
        self.speed_lines_per_sec = 0.0;
        self.eta = None;
        self.last_output_paths.clear();
        self.add_log("开始翻译...");
        self.add_log(&format!("输入: {}", input_path.display()));
        self.add_log(&format!("模型: {} @ {}", model, llm_url));
        if self.output_bilingual {
            if let Some(path) = &output_bilingual_path {
                self.add_log(&format!("双语输出: {}", path.display()));
            }
        }
        if self.output_mono {
            if let Some(path) = &output_mono_path {
                self.add_log(&format!("单语输出: {}", path.display()));
            }
        }

        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.cancel_flag = Some(cancel_flag.clone());
        let completed_with_errors = Arc::new(AtomicBool::new(false));
        cx.notify();

        let entity = cx.entity();

        // Create a channel for progress events from the pipeline
        let (progress_tx, progress_rx) = smol::channel::unbounded::<ProgressEvent>();

        // Build the progress callback (runs in the background tokio thread)
        let callback_completed_with_errors = completed_with_errors.clone();
        let progress_cb: ProgressCallback = Some(Arc::new(move |event| {
            if matches!(event, ProgressEvent::CompletedWithErrors { .. }) {
                callback_completed_with_errors.store(true, Ordering::Relaxed);
            }
            let _ = progress_tx.send_blocking(event);
        }));

        // Spawn a listener task that reads progress events and updates the UI
        let listener_entity = entity.clone();
        let listener_task = cx.spawn(async move |_, cx| {
            while let Ok(event) = progress_rx.recv().await {
                let _ = listener_entity.update(cx, |this, cx| {
                    if this.run_id != current_run_id {
                        return;
                    }
                    match &event {
                        ProgressEvent::StageStarted { stage, detail } => {
                            this.add_log(&format!("[{}] {}", stage, detail));
                        }
                        ProgressEvent::BatchProgress {
                            completed,
                            total,
                            total_lines,
                            translated,
                            failed,
                            corrected,
                            quality_failed,
                            api_failed,
                        } => {
                            if *total > 0 {
                                this.progress = (*completed as f32 / *total as f32) * 100.0;
                            }
                            this.total_lines = *total_lines;
                            this.translated_lines = *translated;
                            this.remaining_lines = total_lines.saturating_sub(*translated);
                            if let Some(started) = this.started_at {
                                let elapsed = started.elapsed();
                                let elapsed_secs = elapsed.as_secs_f32().max(1.0);
                                this.speed_lines_per_sec = *translated as f32 / elapsed_secs;
                                if this.speed_lines_per_sec > 0.0 {
                                    let eta_secs =
                                        (this.remaining_lines as f32 / this.speed_lines_per_sec)
                                            .max(0.0)
                                            .round() as u64;
                                    this.eta = Some(Duration::from_secs(eta_secs));
                                } else {
                                    this.eta = None;
                                }
                            }
                            this.progress_detail = format!(
                                "进度 {}/{} | 已译 {} | 修正 {} | 质量失败 {} | API失败 {} | 总失败 {}",
                                completed,
                                total,
                                translated,
                                corrected,
                                quality_failed,
                                api_failed,
                                failed
                            )
                            .into();
                        }
                        ProgressEvent::Log { message } => {
                            this.add_log(message);
                        }
                        ProgressEvent::Completed {
                            total,
                            translated,
                            fixed,
                            failed,
                            output_path,
                            error_report_path,
                        } => {
                            this.add_log(&format!(
                                "完成: 总计 {} | 已译 {} | 修复 {} | 失败 {} | 输出: {}",
                                total, translated, fixed, failed, output_path
                            ));
                            if let Some(path) = error_report_path {
                                this.add_log(&format!("错误报告: {}", path));
                            }
                            this.progress_detail = "当前任务完成".into();
                        }
                        ProgressEvent::CompletedWithErrors {
                            total,
                            translated,
                            fixed,
                            failed,
                            output_path,
                            error_report_path,
                        } => {
                            this.status = TranslationStatus::CompletedWithErrors;
                            this.add_log(&format!(
                                "完成但有错误: 总计 {} | 已译 {} | 修复 {} | 失败 {} | 输出: {}",
                                total, translated, fixed, failed, output_path
                            ));
                            if let Some(path) = error_report_path {
                                this.add_log(&format!("错误报告: {}", path));
                            }
                            this.progress_detail =
                                format!("当前任务完成，但仍有 {} 项失败", failed).into();
                        }
                        ProgressEvent::Error { message } => {
                            this.add_log(&format!("错误: {}", message));
                            if message.contains("取消") {
                                this.progress_detail = "已取消".into();
                            }
                        }
                    }
                    cx.notify();
                });
            }
        });
        listener_task.detach();

        // Run translation in a background thread with its own tokio runtime
        let translation_task = cx.spawn(async move |_, cx| {
            let result = smol::unblock({
                let input = input_path.clone();
                let output_bilingual = output_bilingual_path.clone();
                let output_mono = output_mono_path.clone();
                let run_bilingual = output_bilingual_path.is_some();
                let run_mono = output_mono_path.is_some();
                let cfg_bilingual = config_bilingual.clone();
                let cfg_mono = config_mono.clone();
                let cb = progress_cb.clone();
                let cancel = Some(cancel_flag.clone());
                let completed_with_errors = completed_with_errors.clone();
                move || {
                    let rt =
                        tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
                    rt.block_on(async {
                        let ext = input
                            .extension()
                            .and_then(|e| e.to_str())
                            .unwrap_or("")
                            .to_lowercase();
                        let is_txt = ext == "txt";

                        // Determine primary and secondary outputs.
                        // Primary runs the full pipeline (translate + export).
                        // Secondary re-uses the same translation data (export only).
                        let (primary_output, primary_cfg, secondary): (
                            PathBuf,
                            PipelineConfig,
                            Option<(PathBuf, TranslationMode)>,
                        ) = if run_bilingual && run_mono {
                            // Primary = bilingual (full pipeline), secondary = mono (export only)
                            (
                                output_bilingual.clone().unwrap(),
                                cfg_bilingual.clone(),
                                Some((output_mono.clone().unwrap(), TranslationMode::Replace)),
                            )
                        } else if run_bilingual {
                            (
                                output_bilingual.clone().unwrap(),
                                cfg_bilingual.clone(),
                                None,
                            )
                        } else {
                            (output_mono.clone().unwrap(), cfg_mono.clone(), None)
                        };

                        // Step A: Run full pipeline for the primary output
                        let ok = if is_txt {
                            pipeline::translate_txt(
                                &input,
                                &primary_output,
                                &primary_cfg,
                                cb.clone(),
                                cancel.clone(),
                            )
                            .await?
                        } else {
                            pipeline::translate_epub(
                                &input,
                                &primary_output,
                                &primary_cfg,
                                cb.clone(),
                                cancel.clone(),
                            )
                            .await?
                        };
                        if !ok {
                            return Ok::<(Vec<PathBuf>, bool), anyhow::Error>((Vec::new(), false));
                        }
                        let mut outputs = vec![primary_output.clone()];

                        // Step B: If there is a secondary output, export using the
                        // translation data that the primary run already saved to disk.
                        if let Some((secondary_output, secondary_mode)) = secondary {
                            // Translation data is in the work dir: {stem}_work/translation_data.json
                            let primary_stem = primary_output
                                .file_stem()
                                .unwrap_or_default()
                                .to_string_lossy();
                            let primary_parent =
                                primary_output.parent().unwrap_or(std::path::Path::new("."));
                            let work_dir = primary_parent.join(format!("{}_work", primary_stem));
                            let json_path = work_dir
                                .join("translation_data.json")
                                .to_string_lossy()
                                .to_string();

                            if is_txt {
                                let data = pipeline::load_txt_translation_data(&json_path)?;
                                pipeline::export_txt_from_data(
                                    &data,
                                    &secondary_output,
                                    secondary_mode,
                                    &cb,
                                )?;
                                outputs.push(secondary_output);
                            } else {
                                let data =
                                    alnilam::epub::EpubHandler::load_translation_data(&json_path)?;
                                pipeline::export_epub_from_data(
                                    &input,
                                    &data,
                                    &secondary_output,
                                    secondary_mode,
                                    primary_cfg.translation_gap.as_deref(),
                                    primary_cfg.apply_fixes,
                                    &cb,
                                )?;
                                outputs.push(secondary_output);
                            }
                        }

                        Ok::<(Vec<PathBuf>, bool), anyhow::Error>((
                            outputs,
                            completed_with_errors.load(Ordering::Relaxed),
                        ))
                    })
                }
            })
            .await;

            _ = entity.update(cx, |this, cx| {
                if this.run_id != current_run_id {
                    return;
                }
                match result {
                    Ok((outputs, had_errors)) if !outputs.is_empty() => {
                        this.status = if had_errors {
                            TranslationStatus::CompletedWithErrors
                        } else {
                            TranslationStatus::Completed
                        };
                        this.progress = 100.0;
                        this.remaining_lines = 0;
                        this.eta = Some(Duration::from_secs(0));
                        this.finished_at = Some(Instant::now());
                        this.progress_detail = if had_errors {
                            "全部任务完成，但存在失败项".into()
                        } else {
                            "全部任务完成".into()
                        };
                        this.last_output_paths = outputs.clone();
                        for out in outputs {
                            if had_errors {
                                this.add_log(&format!(
                                    "翻译完成（有失败项）! 输出: {}",
                                    out.display()
                                ));
                            } else {
                                this.add_log(&format!("翻译完成! 输出: {}", out.display()));
                            }
                        }
                    }
                    Ok(_) => {
                        this.status = TranslationStatus::Cancelled;
                        this.eta = None;
                        this.finished_at = Some(Instant::now());
                        this.progress_detail = "已取消".into();
                        this.add_log("翻译已取消");
                    }
                    Err(e) => {
                        this.status = TranslationStatus::Failed;
                        this.eta = None;
                        this.finished_at = Some(Instant::now());
                        this.progress_detail = "任务失败".into();
                        this.add_log(&format!("翻译错误: {}", e));
                    }
                }
                this.cancel_flag = None;
                cx.notify();
            });
        });

        self._translation_task = Some(translation_task);
    }

    pub fn cancel_translation(
        &mut self,
        _: &ClickEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(flag) = &self.cancel_flag {
            flag.store(true, Ordering::Relaxed);
            self.add_log("已请求中断翻译，正在安全停止...");
            self.progress_detail = "正在取消...".into();
            cx.notify();
        }
    }

    pub fn test_model(&mut self, _: &ClickEvent, _window: &mut Window, cx: &mut Context<Self>) {
        let llm_url = self.llm_url_input.read(cx).value().to_string();
        let model = self.model_input.read(cx).value().to_string();
        let api_key_raw = self.api_key_input.read(cx).value().to_string();
        let url = if llm_url.is_empty() {
            DEFAULT_LLM_URL.to_string()
        } else {
            llm_url
        };
        let mdl = if model.is_empty() {
            DEFAULT_MODEL.to_string()
        } else {
            model
        };
        let api_key = if api_key_raw.trim().is_empty() {
            None
        } else {
            Some(api_key_raw.trim().to_string())
        };

        let test_config = PipelineConfig {
            llm_url: url.clone(),
            model: mdl.clone(),
            batch_size: DEFAULT_BATCH_SIZE,
            context_lines: DEFAULT_CONTEXT_LINES,
            workers: 1,
            max_retry: 1,
            mode: TranslationMode::Bilingual,
            apply_fixes: true,
            rules_path: None,
            translation_gap: Some(config::DEFAULT_TRANSLATION_GAP.to_string()),
            temperature: DEFAULT_TEMPERATURE,
            top_p: DEFAULT_TOP_P,
            top_k: DEFAULT_TOP_K,
            glossary_path: None,
            api_key: api_key.clone(),
        };
        if let Err(e) = test_config.validate() {
            self.model_test_ok = Some(false);
            self.model_test_message = format!("✗ 配置错误: {}", e).into();
            self.add_log(&format!("模型测试配置错误: {}", e));
            cx.notify();
            return;
        }
        for warning in test_config.api_security_warnings() {
            self.add_log(&format!("安全提示: {}", warning));
        }

        self.model_test_running = true;
        self.model_test_ok = None;
        self.model_test_message = "测试中...".into();
        self.add_log(&format!("开始测试模型: {}", mdl));
        cx.notify();

        let entity = cx.entity();
        cx.spawn(async move |_, cx| {
            let result = smol::unblock(move || {
                let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
                rt.block_on(async {
                    let llm = LlmClient::with_params(
                        &url,
                        &mdl,
                        1,
                        0.8,
                        None,
                        None,
                        String::new(),
                        None,
                        api_key,
                    )?;

                    let result = if llm.is_orion_model() {
                        tokio::time::timeout(Duration::from_secs(30), async {
                            match llm
                                .translate_single("今日はいい天気ですね。", &[], "model-test")
                                .await?
                            {
                                Some(translated) if !translated.trim().is_empty() => {
                                    Ok(translated.trim().to_string())
                                }
                                _ => Err(anyhow::anyhow!("模型未返回有效结果")),
                            }
                        })
                        .await
                    } else {
                        tokio::time::timeout(Duration::from_secs(30), llm.test_translation()).await
                    };

                    match result {
                        Ok(v) => v,
                        Err(_) => Err(anyhow::anyhow!("模型测试超时（30s）")),
                    }
                })
            })
            .await;

            _ = entity.update(cx, |this, cx| {
                this.model_test_running = false;
                match result {
                    Ok(translated) => {
                        this.model_test_ok = Some(true);
                        this.model_test_message =
                            format!("✓ {}", Self::compact_single_line(&translated, 72)).into();
                        this.add_log(&format!(
                            "模型测试成功: \"今日はいい天気ですね。\" → \"{}\"",
                            translated
                        ));
                    }
                    Err(e) => {
                        this.model_test_ok = Some(false);
                        let error_text = Self::compact_single_line(&e.to_string(), 72);
                        this.model_test_message = format!("✗ {}", error_text).into();
                        this.add_log(&format!("模型测试失败: {}", e));
                    }
                }
                cx.notify();
            });
        })
        .detach();
    }

    pub fn pick_glossary_file(
        &mut self,
        _: &ClickEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let path_future = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("选择术语表 JSON 文件".into()),
        });

        let entity = cx.entity();
        cx.spawn(async move |_, cx| {
            if let Ok(Ok(Some(paths))) = path_future.await {
                if let Some(p) = paths.into_iter().next() {
                    _ = entity.update(cx, |this, cx| {
                        let file_name = p
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| p.display().to_string());
                        this.add_log(&format!("已选择术语表: {}", file_name));
                        this.glossary_path = Some(p);
                        cx.notify();
                    });
                }
            }
        })
        .detach();
    }

    pub fn clear_glossary(&mut self, _: &ClickEvent, _window: &mut Window, cx: &mut Context<Self>) {
        self.glossary_path = None;
        self.add_log("已清除术语表");
        cx.notify();
    }

    pub fn clear_cache(&mut self, _: &ClickEvent, _window: &mut Window, cx: &mut Context<Self>) {
        let mut removed = 0usize;
        let mut seen_targets: HashSet<PathBuf> = HashSet::new();

        let mut scan_paths: Vec<PathBuf> = Vec::new();
        if let Some(p) = self.input_path.clone() {
            scan_paths.push(p);
        }
        if let Some(p) = self.output_bilingual_path.clone() {
            scan_paths.push(p);
        }
        if let Some(p) = self.output_mono_path.clone() {
            scan_paths.push(p);
        }
        scan_paths.extend(self.last_output_paths.clone());

        for path in scan_paths {
            for target in Self::collect_cache_targets(&path) {
                if !seen_targets.insert(target.clone()) {
                    continue;
                }
                let removed_ok = if target.is_dir() {
                    std::fs::remove_dir_all(&target).is_ok()
                } else {
                    std::fs::remove_file(&target).is_ok()
                };
                if removed_ok {
                    removed += 1;
                }
            }
        }

        // Also clear glossary selection since the file may have been deleted
        self.glossary_path = None;

        self.add_log(&format!("缓存清理完成，共删除 {} 个文件", removed));
        cx.notify();
    }

    pub fn reveal_output(&mut self, _: &ClickEvent, _window: &mut Window, cx: &mut Context<Self>) {
        if let Some(path) = self.last_output_paths.first() {
            cx.reveal_path(path);
            self.add_log(&format!("已在文件管理器中定位: {}", path.display()));
        } else if let Some(path) = &self.output_bilingual_path {
            cx.reveal_path(path);
            self.add_log(&format!("已在文件管理器中定位: {}", path.display()));
        } else if let Some(path) = &self.output_mono_path {
            cx.reveal_path(path);
            self.add_log(&format!("已在文件管理器中定位: {}", path.display()));
        } else {
            self.add_log("暂无可定位的输出文件");
        }
        cx.notify();
    }

    pub fn copy_logs(&mut self, _: &ClickEvent, _window: &mut Window, cx: &mut Context<Self>) {
        cx.write_to_clipboard(ClipboardItem::new_string(self.joined_log_text()));
        self.add_log("日志已复制到剪贴板");
        cx.notify();
    }

    pub fn toggle_log_follow(
        &mut self,
        _: &ClickEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.auto_scroll_log = !self.auto_scroll_log;
        self.add_log(&format!(
            "日志跟随: {}",
            if self.auto_scroll_log {
                "开启"
            } else {
                "关闭"
            }
        ));
        cx.notify();
    }

    pub fn generate_glossary(
        &mut self,
        _: &ClickEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // 1. Check input file
        let input_path = match &self.input_path {
            Some(p) => p.clone(),
            None => {
                self.add_log("错误: 请先选择输入文件");
                cx.notify();
                return;
            }
        };

        // 2. Read model name and determine if we skip LLM translation
        let model = self.model_input.read(cx).value().to_string();
        let model_name = if model.is_empty() {
            alnilam::config::DEFAULT_MODEL.to_string()
        } else {
            model
        };

        let is_orion = !bellatrix::is_generic_model(&model_name);
        if is_orion {
            self.add_log("检测到Orion模型：将只执行NER识别，不执行LLM翻译");
            self.add_log("生成的术语表将不包含译名和说明（dst和info为空）");
        }

        // 3. Read LLM config
        let llm_url_raw = self.llm_url_input.read(cx).value().to_string();
        let llm_url = if llm_url_raw.is_empty() {
            alnilam::config::DEFAULT_LLM_URL.to_string()
        } else {
            llm_url_raw
        };
        let api_key_raw = self.api_key_input.read(cx).value().to_string();
        let api_key = if api_key_raw.trim().is_empty() {
            if !is_orion {
                self.add_log("错误: 术语表生成需要 API 密钥（通用模型）");
                cx.notify();
                return;
            }
            String::new()
        } else {
            api_key_raw
        };

        // Update state
        self.glossary_gen_status = GlossaryGenStatus::Running;
        self.glossary_gen_progress = "准备中...".into();
        self.add_log("开始生成术语表...");
        self.add_log(&format!("输入文件: {}", input_path.display()));
        self.add_log(&format!("LLM模型: {} @ {}", model_name, llm_url));
        cx.notify();

        // 4. Test connectivity first, then run glossary generation
        let entity = cx.entity();

        // Create a channel for progress events
        let (progress_tx, progress_rx) =
            smol::channel::unbounded::<bellatrix::GlossaryProgressEvent>();

        // Build progress callback
        let progress_cb: bellatrix::GlossaryProgressCallback = Some(Arc::new(move |event| {
            let _ = progress_tx.send_blocking(event);
        }));

        // Spawn listener to update UI from progress events
        let listener_entity = entity.clone();
        let listener_task = cx.spawn(async move |_, cx| {
            while let Ok(event) = progress_rx.recv().await {
                let _ = listener_entity.update(cx, |this, cx| {
                    match &event {
                        bellatrix::GlossaryProgressEvent::StageStarted { stage, detail } => {
                            this.glossary_gen_progress = format!("[{}] {}", stage, detail).into();
                            this.add_log(&format!("[术语表] [{}] {}", stage, detail));
                        }
                        bellatrix::GlossaryProgressEvent::NerProgress { completed, total } => {
                            this.glossary_gen_progress =
                                format!("NER识别 {}/{}", completed, total).into();
                        }
                        bellatrix::GlossaryProgressEvent::LlmProgress { completed, total } => {
                            this.glossary_gen_progress =
                                format!("LLM翻译 {}/{}", completed, total).into();
                        }
                        bellatrix::GlossaryProgressEvent::Log { message } => {
                            this.add_log(&format!("[术语表] {}", message));
                        }
                        bellatrix::GlossaryProgressEvent::Completed {
                            output_path,
                            entry_count,
                        } => {
                            this.add_log(&format!(
                                "[术语表] 完成: {} ({} 条)",
                                output_path, entry_count
                            ));
                        }
                        bellatrix::GlossaryProgressEvent::Error { message } => {
                            this.add_log(&format!("[术语表] 错误: {}", message));
                        }
                    }
                    cx.notify();
                });
            }
        });
        listener_task.detach();

        // Determine NER model directory (relative to executable or workspace)
        let model_dir = match Self::find_ner_model_dir() {
            Ok(dir) => {
                self.add_log(&format!("找到 NER 模型目录: {}", dir));
                dir
            }
            Err(err_msg) => {
                self.glossary_gen_status = GlossaryGenStatus::Failed;
                self.glossary_gen_progress = "".into();
                for line in err_msg.lines() {
                    self.add_log(line);
                }
                cx.notify();
                return;
            }
        };

        // Build glossary config
        let output_path = {
            let base_name = input_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let parent = input_path.parent().unwrap_or(std::path::Path::new("."));
            parent.join(format!("{}_glossary.json", base_name))
        };

        let glossary_config = bellatrix::GlossaryConfig {
            lines: Vec::new(), // will be populated in background thread
            model_dir,
            ner_batch_size: 16,
            min_count: 2,
            llm_url: llm_url.clone(),
            llm_api_key: api_key,
            llm_model: model_name.clone(),
            llm_workers: 4,
            output_path,
            skip_llm_translation: is_orion,
        };

        // Test connectivity + run generation in background
        let test_url = llm_url.clone();
        let test_model = model_name.clone();
        let test_api_key_for_test = glossary_config.llm_api_key.clone();
        let skip_llm = is_orion;

        let glossary_task = cx.spawn(async move |_, cx| {
            let result = smol::unblock({
                let mut config = glossary_config;
                let cb = progress_cb;
                let test_url = test_url;
                let test_model = test_model;
                let test_key = test_api_key_for_test;
                let file_path = input_path.clone();
                move || {
                    // Extract text lines from the input file (blocking I/O)
                    let ext = file_path
                        .extension()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_lowercase();
                    let lines = match ext.as_str() {
                        "epub" => betelgeuse::extract_epub_lines(&file_path),
                        "txt" => betelgeuse::extract_txt_lines(&file_path),
                        _ => Err(anyhow::anyhow!(
                            "不支持的文件格式: .{} (仅支持 .epub / .txt)",
                            ext
                        )),
                    }?;
                    config.lines = lines;

                    let rt =
                        tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
                    rt.block_on(async {
                        // Test LLM connectivity (only for generic models)
                        if !skip_llm {
                            let llm = alnilam::llm::LlmClient::with_params(
                                &test_url,
                                &test_model,
                                1,
                                0.8,
                                None,
                                None,
                                String::new(),
                                None,
                                Some(test_key),
                            )?;
                            let test_result = tokio::time::timeout(
                                Duration::from_secs(30),
                                llm.test_translation(),
                            )
                            .await;
                            match test_result {
                                Ok(Ok(_)) => {} // connectivity OK
                                Ok(Err(e)) => {
                                    anyhow::bail!("LLM连通性测试失败: {}", e);
                                }
                                Err(_) => {
                                    anyhow::bail!("LLM连通性测试超时 (30s)");
                                }
                            }
                        }

                        // Run the glossary generation pipeline
                        let output_path = bellatrix::generate_glossary(config, cb).await?;
                        Ok::<std::path::PathBuf, anyhow::Error>(output_path)
                    })
                }
            })
            .await;

            _ = entity.update(cx, |this, cx| {
                match result {
                    Ok(output_path) => {
                        this.glossary_gen_status = GlossaryGenStatus::Completed;
                        this.glossary_gen_progress = "".into();
                        this.glossary_path = Some(output_path.clone());
                        this.add_log(&format!("术语表生成完成: {}", output_path.display()));
                    }
                    Err(e) => {
                        this.glossary_gen_status = GlossaryGenStatus::Failed;
                        this.glossary_gen_progress = "".into();
                        // Show full error chain for debugging
                        for (i, cause) in e.chain().enumerate() {
                            if i == 0 {
                                this.add_log(&format!("术语表生成失败: {}", cause));
                            } else {
                                this.add_log(&format!("  原因: {}", cause));
                            }
                        }
                    }
                }
                cx.notify();
            });
        });

        self._glossary_gen_task = Some(glossary_task);
    }

    /// Find the NER model directory.
    /// Searches common locations relative to the executable or workspace.
    fn find_ner_model_dir() -> Result<String, String> {
        let cwd = std::env::current_dir().unwrap_or_default();
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|e| e.parent().map(|p| p.to_path_buf()));

        let mut candidates: Vec<String> = vec![
            // Relative to CWD — local project copy (preferred)
            "ner_model".to_string(),
            "../ner_model".to_string(),
            // Workspace-relative paths (alnilam keeps ner_model)
            "alnilam/ner_model".to_string(),
            "../alnilam/ner_model".to_string(),
            // Legacy paths
            "rigel/model".to_string(),
            "../rigel/model".to_string(),
        ];

        // Relative to executable
        if let Some(ref exe) = exe_dir {
            candidates.push(exe.join("ner_model").to_string_lossy().to_string());
            candidates.push(exe.join("alnilam/ner_model").to_string_lossy().to_string());
            if let Some(parent) = exe.parent() {
                candidates.push(parent.join("ner_model").to_string_lossy().to_string());
                candidates.push(
                    parent
                        .join("alnilam/ner_model")
                        .to_string_lossy()
                        .to_string(),
                );
            }
            // macOS .app bundle: Contents/MacOS/<exe> → Contents/Resources/ner_model
            // exe_dir is already Contents/MacOS/, so check exe itself
            #[cfg(target_os = "macos")]
            if exe.ends_with("MacOS") {
                if let Some(contents_dir) = exe.parent() {
                    let resources = contents_dir.join("Resources").join("ner_model");
                    candidates.push(resources.to_string_lossy().to_string());
                }
            }
        }

        let required_files = ["model.safetensors", "config.json", "vocab.txt"];

        for candidate in &candidates {
            if candidate.is_empty() {
                continue;
            }
            let path = std::path::Path::new(candidate);
            if required_files.iter().all(|f| path.join(f).exists()) {
                return Ok(candidate.clone());
            }
        }

        // Build detailed error message
        let mut msg = format!(
            "找不到 NER 模型文件。\n  CWD: {}\n  搜索的路径:",
            cwd.display()
        );
        for candidate in &candidates {
            if candidate.is_empty() {
                continue;
            }
            let path = std::path::Path::new(candidate);
            let abs = std::fs::canonicalize(path)
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| format!("<不存在: {}>", path.display()));
            let missing: Vec<&str> = required_files
                .iter()
                .filter(|f| !path.join(f).exists())
                .copied()
                .collect();
            msg.push_str(&format!(
                "\n  - {} ({}) [缺少: {}]",
                candidate,
                abs,
                if missing.is_empty() {
                    "OK".to_string()
                } else {
                    missing.join(", ")
                }
            ));
        }
        msg.push_str("\n  请将 NER 模型文件 (model.safetensors, config.json, vocab.txt, system.dic.zst) 放入 ner_model/ 目录");
        Err(msg)
    }
}
