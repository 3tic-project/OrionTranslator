use std::time::Instant;

use gpui::prelude::FluentBuilder as _;
use gpui::*;
use gpui_component::{
    ActiveTheme, Disableable as _, Icon, IconName, Sizable as _,
    button::{Button, ButtonVariants as _},
    checkbox::Checkbox,
    h_flex,
    input::Input,
    progress::Progress,
    radio::{Radio, RadioGroup},
    spinner::Spinner,
    v_flex,
};

use crate::app::OrionApp;
use crate::types::{GlossaryGenStatus, ModelPreset, TranslationStatus};

// ============================================================================
// UI Rendering
// ============================================================================

impl OrionApp {
    pub fn render_file_section(&self, cx: &Context<Self>) -> impl IntoElement {
        let input_path_display: SharedString = match &self.input_path {
            Some(p) => p.to_string_lossy().to_string().into(),
            None => "未选择文件".into(),
        };

        let output_dir_display: SharedString = match self
            .last_output_paths
            .first()
            .or(self.output_bilingual_path.as_ref())
            .or(self.output_mono_path.as_ref())
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        {
            Some(p) => p.to_string_lossy().to_string().into(),
            None => "自动".into(),
        };

        let glossary_display: SharedString = match &self.glossary_path {
            Some(p) => p
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| p.display().to_string())
                .into(),
            None => "无".into(),
        };

        let is_running = self.status == TranslationStatus::Running;

        let is_glossary_generating = self.glossary_gen_status == GlossaryGenStatus::Running;

        v_flex()
            .id("file-drop-zone")
            .gap_2()
            .drag_over::<ExternalPaths>(|style, _, _, _| {
                style
                    .border_color(gpui::blue())
                    .bg(gpui::blue().opacity(0.08))
            })
            .on_drop(cx.listener(|this, paths: &ExternalPaths, _window, cx| {
                if this.status == TranslationStatus::Running {
                    return;
                }
                for p in paths.paths() {
                    let ext = p
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("")
                        .to_lowercase();
                    if ext == "epub" || ext == "txt" {
                        let file_name = p
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| p.display().to_string());
                        this.add_log(&format!("已拖入文件: {}", file_name));
                        let (bilingual, mono) = Self::compute_output_paths(&p.clone());
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
                    } else if ext == "json" {
                        let file_name = p
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| p.display().to_string());
                        this.add_log(&format!("已拖入术语表: {}", file_name));
                        this.glossary_path = Some(p.clone());
                    } else {
                        this.add_log("拖入的文件不是 EPUB/TXT/JSON 格式");
                    }
                }
                cx.notify();
            }))
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(cx.theme().foreground)
                    .child("文件"),
            )
            .child(
                h_flex()
                    .gap_4()
                    .items_center()
                    // Left: input file
                    .child(
                        h_flex()
                            .flex_1()
                            .min_w(px(0.))
                            .gap_2()
                            .items_center()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child("输入"),
                            )
                            .child(
                                Button::new("pick-file")
                                    .small()
                                    .outline()
                                    .label("选择文件")
                                    .disabled(is_running)
                                    .on_click(cx.listener(Self::pick_input_file)),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .truncate()
                                    .flex_1()
                                    .child(input_path_display),
                            ),
                    )
                    // Right: output directory
                    .child(
                        h_flex()
                            .flex_1()
                            .min_w(px(0.))
                            .gap_2()
                            .items_center()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child("输出"),
                            )
                            .child(
                                Button::new("reveal-output")
                                    .small()
                                    .outline()
                                    .label("定位输出")
                                    .on_click(cx.listener(Self::reveal_output)),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .truncate()
                                    .flex_1()
                                    .child(output_dir_display),
                            )
                            .child(
                                Button::new("clear-cache")
                                    .small()
                                    .outline()
                                    .label("清缓存")
                                    .disabled(is_running)
                                    .on_click(cx.listener(Self::clear_cache)),
                            ),
                    ),
            )
            // Glossary row
            .child(
                h_flex().gap_4().items_center().child(
                    h_flex()
                        .flex_1()
                        .min_w(px(0.))
                        .gap_2()
                        .items_center()
                        .child(
                            div()
                                .text_xs()
                                .text_color(cx.theme().muted_foreground)
                                .child("术语表"),
                        )
                        .child(
                            Button::new("pick-glossary")
                                .small()
                                .outline()
                                .label("选择文件")
                                .disabled(is_running || is_glossary_generating)
                                .on_click(cx.listener(Self::pick_glossary_file)),
                        )
                        .child(
                            Button::new("gen-glossary")
                                .small()
                                .outline()
                                .label(if is_glossary_generating {
                                    "生成中..."
                                } else {
                                    "生成术语表"
                                })
                                .disabled(
                                    is_running
                                        || is_glossary_generating
                                        || self.input_path.is_none(),
                                )
                                .on_click(cx.listener(Self::generate_glossary)),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(if is_glossary_generating {
                                    cx.theme().yellow
                                } else {
                                    cx.theme().muted_foreground
                                })
                                .truncate()
                                .flex_1()
                                .child(if is_glossary_generating {
                                    self.glossary_gen_progress.clone()
                                } else {
                                    glossary_display
                                }),
                        )
                        .child(
                            Button::new("clear-glossary")
                                .small()
                                .outline()
                                .label("取消选择")
                                .disabled(
                                    is_running
                                        || is_glossary_generating
                                        || self.glossary_path.is_none(),
                                )
                                .on_click(cx.listener(Self::clear_glossary)),
                        ),
                ),
            )
    }

    pub fn render_config_section(&self, cx: &Context<Self>) -> impl IntoElement {
        let is_running = self.status == TranslationStatus::Running;
        let model_status_color = if self.model_test_running {
            cx.theme().yellow
        } else {
            match self.model_test_ok {
                Some(true) => cx.theme().green,
                Some(false) => cx.theme().red,
                None => cx.theme().muted_foreground,
            }
        };

        v_flex()
            .gap_2()
            .child(
                h_flex()
                    .gap_4()
                    .items_center()
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(cx.theme().foreground)
                            .child("配置"),
                    )
                    .child(
                        h_flex()
                            .gap_2()
                            .items_center()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child("预设"),
                            )
                            .child(
                                RadioGroup::horizontal("model-preset")
                                    .children(
                                        ModelPreset::ALL.iter().map(|p| {
                                            Radio::new(p.label()).label(p.label()).small()
                                        }),
                                    )
                                    .text_xs()
                                    .gap_1p5()
                                    .selected_index(Some(self.model_preset.index()))
                                    .disabled(is_running)
                                    .on_click(cx.listener(Self::on_preset_changed)),
                            ),
                    ),
            )
            // LLM URL + Model (with inline test button) + API Key
            .child(
                h_flex()
                    .gap_3()
                    .items_end()
                    .child(
                        v_flex()
                            .w(px(270.))
                            .gap_0p5()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child("LLM BASE_URL"),
                            )
                            .child(Input::new(&self.llm_url_input).small()),
                    )
                    .child(
                        v_flex()
                            .w(px(270.))
                            .gap_0p5()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child("模型"),
                            )
                            .child(Input::new(&self.model_input).small()),
                    )
                    .child(
                        v_flex()
                            .flex_1()
                            .min_w(px(120.))
                            .gap_0p5()
                            .child(
                                h_flex()
                                    .gap_2()
                                    .items_center()
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(cx.theme().muted_foreground)
                                            .child("API 密钥"),
                                    )
                                    .child(
                                        Button::new("test-model")
                                            .xsmall()
                                            .outline()
                                            .label("测试")
                                            .disabled(is_running || self.model_test_running)
                                            .on_click(cx.listener(Self::test_model)),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(model_status_color)
                                            .flex_1()
                                            .min_w(px(0.))
                                            .truncate()
                                            .child(self.model_test_message.clone()),
                                    ),
                            )
                            .child(Input::new(&self.api_key_input).small().mask_toggle()),
                    ),
            )
            // Batch size, Workers, Context lines, Max retry + Temperature, Top_P, Top_K
            .child(
                h_flex()
                    .gap_2()
                    .child(
                        v_flex()
                            .w(px(72.))
                            .gap_0p5()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child("批次大小"),
                            )
                            .child(Input::new(&self.batch_size_input).small()),
                    )
                    .child(
                        v_flex()
                            .w(px(72.))
                            .gap_0p5()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child("并行数"),
                            )
                            .child(Input::new(&self.workers_input).small()),
                    )
                    .child(
                        v_flex()
                            .w(px(72.))
                            .gap_0p5()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child("上下文"),
                            )
                            .child(Input::new(&self.context_lines_input).small()),
                    )
                    .child(
                        v_flex()
                            .w(px(72.))
                            .gap_0p5()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child("重试"),
                            )
                            .child(Input::new(&self.max_retry_input).small()),
                    )
                    .child(
                        v_flex()
                            .w(px(80.))
                            .gap_0p5()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child("温度"),
                            )
                            .child(Input::new(&self.temperature_input).small()),
                    )
                    .child(
                        v_flex()
                            .w(px(80.))
                            .gap_0p5()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child("Top P"),
                            )
                            .child(Input::new(&self.top_p_input).small()),
                    )
                    .child(
                        v_flex()
                            .w(px(80.))
                            .gap_0p5()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child("Top K"),
                            )
                            .child(Input::new(&self.top_k_input).small()),
                    )
                    .child(
                        v_flex()
                            .ml_4()
                            .gap_0p5()
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(cx.theme().muted_foreground)
                                    .child("输出模式"),
                            )
                            .child(
                                h_flex()
                                    .gap_3()
                                    .items_center()
                                    .child(
                                        Checkbox::new("output-bilingual-toggle")
                                            .small()
                                            .label("对照")
                                            .checked(self.output_bilingual)
                                            .disabled(is_running)
                                            .on_click(cx.listener(
                                                |this, checked: &bool, _window, cx| {
                                                    this.output_bilingual = *checked;
                                                    if let Some(input) = &this.input_path {
                                                        let (bilingual, mono) =
                                                            Self::compute_output_paths(input);
                                                        this.output_bilingual_path =
                                                            Some(bilingual);
                                                        this.output_mono_path = Some(mono);
                                                    }
                                                    this.add_log(&format!(
                                                        "中日对照输出: {}",
                                                        if this.output_bilingual {
                                                            "开启"
                                                        } else {
                                                            "关闭"
                                                        }
                                                    ));
                                                    cx.notify();
                                                },
                                            )),
                                    )
                                    .child(
                                        Checkbox::new("output-mono-toggle")
                                            .small()
                                            .label("中文")
                                            .checked(self.output_mono)
                                            .disabled(is_running)
                                            .on_click(cx.listener(
                                                |this, checked: &bool, _window, cx| {
                                                    this.output_mono = *checked;
                                                    if let Some(input) = &this.input_path {
                                                        let (bilingual, mono) =
                                                            Self::compute_output_paths(input);
                                                        this.output_bilingual_path =
                                                            Some(bilingual);
                                                        this.output_mono_path = Some(mono);
                                                    }
                                                    this.add_log(&format!(
                                                        "仅中文输出: {}",
                                                        if this.output_mono {
                                                            "开启"
                                                        } else {
                                                            "关闭"
                                                        }
                                                    ));
                                                    cx.notify();
                                                },
                                            )),
                                    ),
                            ),
                    ),
            )
    }

    pub fn render_stat_card(
        &self,
        title: &str,
        value: String,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        v_flex()
            .gap_0p5()
            .p_2()
            .rounded(cx.theme().radius)
            .border_1()
            .border_color(cx.theme().border)
            .bg(cx.theme().secondary)
            .child(
                div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child(title.to_string()),
            )
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(cx.theme().foreground)
                    .child(value),
            )
    }

    pub fn render_dashboard_section(&self, cx: &Context<Self>) -> impl IntoElement {
        let elapsed = self
            .started_at
            .map(|start| {
                let end = self.finished_at.unwrap_or_else(Instant::now);
                Self::format_duration_hms(end.duration_since(start))
            })
            .unwrap_or_else(|| "00:00".to_string());
        let eta = self
            .eta
            .map(Self::format_duration_hms)
            .unwrap_or_else(|| "--:--".to_string());
        let speed = if self.speed_lines_per_sec > 0.0 {
            format!("{:.2} 行/秒", self.speed_lines_per_sec)
        } else {
            "0.00 行/秒".to_string()
        };

        div()
            .w_full()
            .grid()
            .grid_cols(3)
            .gap_2()
            .child(self.render_stat_card("已用时", elapsed, cx))
            .child(self.render_stat_card("预计剩余", eta, cx))
            .child(self.render_stat_card("翻译速度", speed, cx))
            .child(self.render_stat_card("已翻译行数", self.translated_lines.to_string(), cx))
            .child(self.render_stat_card("剩余行数", self.remaining_lines.to_string(), cx))
            .child(self.render_stat_card("进度百分比", format!("{:.1}%", self.progress), cx))
    }

    pub fn render_heatmap_progress(&self, cx: &Context<Self>) -> impl IntoElement {
        let total_cells = 160usize;
        let filled =
            (self.progress.clamp(0.0, 100.0) / 100.0 * total_cells as f32).round() as usize;
        let active = filled.min(total_cells);

        v_flex()
            .h_full()
            .justify_center()
            .items_center()
            .p_2()
            .rounded(cx.theme().radius)
            .border_1()
            .border_color(cx.theme().border)
            .bg(cx.theme().secondary)
            .child(
                div()
                    .grid()
                    .grid_cols(20)
                    .gap_0p5()
                    .children((0..total_cells).map(|i| {
                        let color = if i < active {
                            cx.theme().green.opacity(0.85)
                        } else {
                            cx.theme().secondary_foreground.opacity(0.12)
                        };
                        div().size(px(12.)).bg(color)
                    })),
            )
    }

    pub fn render_progress_section(&self, cx: &Context<Self>) -> impl IntoElement {
        let status_text: SharedString = match self.status {
            TranslationStatus::Idle => "就绪".into(),
            TranslationStatus::Running => "翻译中...".to_string().into(),
            TranslationStatus::Completed => "完成".to_string().into(),
            TranslationStatus::Cancelled => "已取消".to_string().into(),
            TranslationStatus::Failed => "失败".into(),
        };

        let status_color = match self.status {
            TranslationStatus::Idle => cx.theme().muted_foreground,
            TranslationStatus::Running => cx.theme().blue,
            TranslationStatus::Completed => cx.theme().green,
            TranslationStatus::Cancelled => cx.theme().yellow,
            TranslationStatus::Failed => cx.theme().red,
        };

        let can_start = self.input_path.is_some() && self.status != TranslationStatus::Running;

        v_flex()
            .gap_2()
            .child(
                h_flex()
                    .justify_between()
                    .items_center()
                    .child(
                        h_flex()
                            .gap_2()
                            .items_center()
                            .when(self.status == TranslationStatus::Running, |this| {
                                this.child(Spinner::new().small())
                            })
                            .child(
                                div()
                                    .text_sm()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(status_color)
                                    .child(status_text),
                            ),
                    )
                    .child(
                        h_flex()
                            .gap_2()
                            .child(
                                Button::new("start-btn")
                                    .when(self.status == TranslationStatus::Running, |btn| {
                                        btn.danger().label("翻译中...").disabled(true)
                                    })
                                    .when(self.status != TranslationStatus::Running, |btn| {
                                        btn.primary()
                                            .label("开始翻译")
                                            .icon(Icon::new(IconName::ArrowRight))
                                            .disabled(!can_start)
                                    })
                                    .on_click(cx.listener(Self::start_translation)),
                            )
                            .child(
                                Button::new("cancel-btn")
                                    .danger()
                                    .label("中断")
                                    .disabled(self.status != TranslationStatus::Running)
                                    .on_click(cx.listener(Self::cancel_translation)),
                            ),
                    ),
            )
            .child(
                h_flex()
                    .gap_4()
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(0.))
                            .child(self.render_dashboard_section(cx)),
                    )
                    .child(self.render_heatmap_progress(cx)),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child(self.progress_detail.clone()),
            )
            .child(Progress::new().h_2().value(self.progress))
    }

    pub fn render_log_section(&self, cx: &Context<Self>) -> impl IntoElement {
        v_flex()
            .h_full()
            .min_h(px(0.))
            .gap_1()
            .child(
                h_flex()
                    .justify_between()
                    .items_center()
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(cx.theme().foreground)
                            .child("日志"),
                    )
                    .child(
                        h_flex()
                            .gap_2()
                            .child(
                                Button::new("log-follow")
                                    .small()
                                    .outline()
                                    .label(if self.auto_scroll_log {
                                        "跟随: 开"
                                    } else {
                                        "跟随: 关"
                                    })
                                    .on_click(cx.listener(Self::toggle_log_follow)),
                            )
                            .child(
                                Button::new("copy-logs")
                                    .small()
                                    .outline()
                                    .label("复制日志")
                                    .on_click(cx.listener(Self::copy_logs)),
                            ),
                    ),
            )
            .child(
                div()
                    .id("log-area")
                    .flex_1()
                    .min_h(px(0.))
                    .w_full()
                    .p_2()
                    .rounded(cx.theme().radius)
                    .border_1()
                    .border_color(cx.theme().border)
                    .bg(cx.theme().secondary)
                    .overflow_y_scroll()
                    .track_scroll(&self.scroll_handle)
                    .children(self.log_messages.iter().map(|msg| {
                        div()
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .py_0p5()
                            .child(msg.clone())
                    })),
            )
    }
}
