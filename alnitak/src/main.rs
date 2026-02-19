mod app;
mod handlers;
mod types;
mod ui;
mod utils;

use gpui::*;
use gpui_component::{
    h_flex, v_flex, ActiveTheme, Root, Theme, ThemeMode, TitleBar,
};
use gpui_component_assets::Assets;

use app::OrionApp;
use types::Quit;

// ============================================================================
// Render Implementation
// ============================================================================

impl Render for OrionApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .size_full()
            .child(
                TitleBar::new().child(
                    h_flex()
                        .items_center()
                        .gap_2()
                        .child(
                            div()
                                .text_sm()
                                .font_weight(FontWeight::SEMIBOLD)
                                .child("Orion 翻译器"),
                        ),
                ),
            )
            .child(
                v_flex()
                    .id("main-content")
                    .flex_1()
                    .min_h(px(0.))
                    .p_4()
                    .gap_4()
                    .child(self.render_file_section(cx))
                    .child(div().h(px(1.)).w_full().bg(cx.theme().border))
                    .child(self.render_config_section(cx))
                    .child(div().h(px(1.)).w_full().bg(cx.theme().border))
                    .child(self.render_progress_section(cx))
                    .child(div().h(px(1.)).w_full().bg(cx.theme().border))
                    .child(
                        v_flex()
                            .flex_1()
                            .min_h(px(120.))
                            .child(self.render_log_section(cx)),
                    ),
            )
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let app = Application::new().with_assets(Assets);

    app.run(move |cx| {
        gpui_component::init(cx);

        cx.bind_keys([
            #[cfg(target_os = "macos")]
            KeyBinding::new("cmd-q", Quit, None),
            #[cfg(not(target_os = "macos"))]
            KeyBinding::new("alt-f4", Quit, None),
        ]);

        cx.on_action(|_: &Quit, cx: &mut App| {
            // Quit action is handled via window close confirmation or direct quit
            // when no windows are open.
            if cx.windows().is_empty() {
                cx.quit();
            }
            // If windows exist, the on_window_should_close handler will show
            // the confirmation dialog.
        });

        let window_options = WindowOptions {
            titlebar: Some(TitleBar::title_bar_options()),
            window_bounds: Some(WindowBounds::centered(size(px(980.), px(760.)), cx)),
            is_movable: true,
            is_resizable: false,
            ..Default::default()
        };

        cx.spawn(async move |cx| {
            cx.open_window(window_options, |window, cx| {
                cx.activate(true);
                window.activate_window();
                window.set_window_title("Orion 翻译器");

                Theme::change(ThemeMode::Dark, Some(window), cx);

                // Register close confirmation dialog
                window.on_window_should_close(cx, |window, cx| {
                    let answer = window.prompt(
                        PromptLevel::Warning,
                        "确认退出",
                        Some("确定要关闭 Orion 翻译器吗？"),
                        &[PromptButton::ok("确认退出"), PromptButton::cancel("取消")],
                        cx,
                    );
                    cx.spawn(async move |cx| {
                        if answer.await == Ok(0) {
                            let _ = cx.update(|cx| cx.quit());
                        }
                    })
                    .detach();
                    // Return false to prevent immediate close; we quit asynchronously above
                    false
                });

                let view = cx.new(|cx| OrionApp::new(window, cx));
                cx.new(|cx| Root::new(view, window, cx))
            })?;

            Ok::<_, anyhow::Error>(())
        })
        .detach();
    });
}
