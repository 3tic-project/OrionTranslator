use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use gpui::*;
use gpui_component::input::InputState;

use crate::types::{GlossaryGenStatus, ModelPreset, TranslationStatus};

// ============================================================================
// Main Application State
// ============================================================================

pub struct OrionApp {
    // File selection
    pub input_path: Option<PathBuf>,
    pub glossary_path: Option<PathBuf>,
    pub output_bilingual_path: Option<PathBuf>,
    pub output_mono_path: Option<PathBuf>,
    pub last_output_paths: Vec<PathBuf>,

    // Config inputs
    pub llm_url_input: Entity<InputState>,
    pub model_input: Entity<InputState>,
    pub api_key_input: Entity<InputState>,
    pub batch_size_input: Entity<InputState>,
    pub workers_input: Entity<InputState>,
    pub context_lines_input: Entity<InputState>,
    pub max_retry_input: Entity<InputState>,
    pub temperature_input: Entity<InputState>,
    pub top_p_input: Entity<InputState>,
    pub top_k_input: Entity<InputState>,

    // Config state
    pub output_bilingual: bool,
    pub output_mono: bool,
    pub model_preset: ModelPreset,

    // Translation state
    pub status: TranslationStatus,
    pub progress: f32,
    pub progress_detail: SharedString,
    pub started_at: Option<Instant>,
    pub finished_at: Option<Instant>,
    pub translated_lines: usize,
    pub remaining_lines: usize,
    pub total_lines: usize,
    pub speed_lines_per_sec: f32,
    pub eta: Option<Duration>,
    pub log_messages: Vec<SharedString>,
    pub auto_scroll_log: bool,
    pub scroll_handle: ScrollHandle,
    pub model_test_message: SharedString,
    pub model_test_ok: Option<bool>,
    pub model_test_running: bool,
    pub run_id: u64,

    // Background task
    pub _translation_task: Option<Task<()>>,
    pub cancel_flag: Option<Arc<AtomicBool>>,

    // Glossary generation state
    pub glossary_gen_status: GlossaryGenStatus,
    pub glossary_gen_progress: SharedString,
    pub _glossary_gen_task: Option<Task<()>>,
}

impl OrionApp {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        use alnilam::config::*;

        let llm_url_input = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("LLM API BASE_URL")
                .default_value(DEFAULT_LLM_URL)
        });
        let model_input = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("模型名称")
                .default_value(DEFAULT_MODEL)
        });
        let batch_size_input = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("批次大小")
                .default_value(&DEFAULT_BATCH_SIZE.to_string())
        });
        let workers_input = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("并行数")
                .default_value(&DEFAULT_WORKERS.to_string())
        });
        let context_lines_input = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("上下文行数")
                .default_value(&DEFAULT_CONTEXT_LINES.to_string())
        });
        let max_retry_input = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("最大重试")
                .default_value(&DEFAULT_MAX_RETRY.to_string())
        });
        let temperature_input = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("温度")
                .default_value(&DEFAULT_TEMPERATURE.to_string())
        });
        let top_p_input = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("Top-P")
                .default_value(&DEFAULT_TOP_P.to_string())
        });
        let top_k_input = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("Top-K")
                .default_value(&DEFAULT_TOP_K.to_string())
        });
        let api_key_input = cx.new(|cx| {
            InputState::new(window, cx)
                .placeholder("API 密钥（可选）")
                .masked(true)
        });

        Self {
            input_path: None,
            glossary_path: None,
            output_bilingual_path: None,
            output_mono_path: None,
            last_output_paths: Vec::new(),
            llm_url_input,
            model_input,
            api_key_input,
            batch_size_input,
            workers_input,
            context_lines_input,
            max_retry_input,
            temperature_input,
            top_p_input,
            top_k_input,
            output_bilingual: true,
            output_mono: false,
            model_preset: ModelPreset::DeepSeek,
            status: TranslationStatus::Idle,
            progress: 0.0,
            progress_detail: "等待开始".into(),
            started_at: None,
            finished_at: None,
            translated_lines: 0,
            remaining_lines: 0,
            total_lines: 0,
            speed_lines_per_sec: 0.0,
            eta: None,
            log_messages: vec!["Orion 翻译器就绪".into()],
            auto_scroll_log: true,
            scroll_handle: ScrollHandle::new(),
            model_test_message: "未测试".into(),
            model_test_ok: None,
            model_test_running: false,
            run_id: 0,
            _translation_task: None,
            cancel_flag: None,
            glossary_gen_status: GlossaryGenStatus::Idle,
            glossary_gen_progress: "".into(),
            _glossary_gen_task: None,
        }
    }
}
