use alnilam::{config, pipeline};

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

/// Orion EPUB/TXT 日译中一站式翻译工具
#[derive(Parser, Debug)]
#[command(name = "alnilam", about = "EPUB/TXT 日译中一站式翻译工具")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// 输入 EPUB/TXT 文件路径
    input: Option<PathBuf>,

    /// 输出文件路径 (默认: <input>.ja-zh.epub/txt)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// 翻译模式: bilingual(双语) 或 replace(替换)
    #[arg(short, long, default_value = "bilingual")]
    mode: String,

    /// 每批翻译的行数
    #[arg(short, long, default_value_t = config::DEFAULT_BATCH_SIZE)]
    batch_size: usize,

    /// 上下文行数
    #[arg(short, long, default_value_t = config::DEFAULT_CONTEXT_LINES)]
    context_lines: usize,

    /// 并行任务数
    #[arg(short, long, default_value_t = config::DEFAULT_WORKERS)]
    workers: usize,

    /// LLM API BASE_URL（如 https://api.deepseek.com/v1）
    #[arg(long, default_value = config::DEFAULT_LLM_URL)]
    llm_url: String,

    /// 模型名称
    #[arg(long, default_value = config::DEFAULT_MODEL)]
    model: String,

    /// 最大重试次数
    #[arg(long, default_value_t = config::DEFAULT_MAX_RETRY)]
    max_retry: usize,

    /// 不应用格式修复
    #[arg(long)]
    no_fix: bool,

    /// 调试模式
    #[arg(short, long)]
    debug: bool,

    /// 上下文规则文件路径
    #[arg(long)]
    rules_path: Option<PathBuf>,

    /// 双语模式下译文段落的底部间距 (如 "1rem", "0.5em", "8px")
    /// 设为 "0" 或使用 --no-gap 禁用
    #[arg(long, default_value = config::DEFAULT_TRANSLATION_GAP)]
    gap: String,

    /// 禁用译文段落间距
    #[arg(long)]
    no_gap: bool,

    /// 术语表 JSON 文件路径（通用模型使用）
    #[arg(long)]
    glossary_path: Option<PathBuf>,

    /// API 密钥（用于需要鉴权的 LLM 服务）
    #[arg(long)]
    api_key: Option<String>,
}

#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// 使用已有 translation_data.json 无模型重导出 EPUB/TXT
    Export {
        /// 原始 EPUB/TXT 文件路径
        input: PathBuf,

        /// 已有翻译数据 JSON
        #[arg(long)]
        translation_data: PathBuf,

        /// 输出 EPUB/TXT 文件路径（必须与输入不同）
        #[arg(short, long)]
        output: PathBuf,

        /// 翻译模式: bilingual(双语) 或 replace(替换)
        #[arg(short, long, default_value = "bilingual")]
        mode: String,

        /// 双语译文段落底部间距；0 表示禁用
        #[arg(long, default_value = config::DEFAULT_TRANSLATION_GAP)]
        gap: String,

        /// 显式启用竖排转横排、RTL 转 LTR、SVG 简化等有损格式修复
        #[arg(long)]
        apply_fixes: bool,
    },

    /// 离线审查已有术语表与 EPUB ruby/后文假名覆盖，不调用模型
    GlossaryAudit {
        /// 原始 EPUB 文件
        file: PathBuf,

        /// 已有 v1 术语表 JSON
        #[arg(long)]
        glossary_path: PathBuf,

        /// 审核清单输出路径（默认: *.ruby-candidates.json）
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// 更新 ruby 候选的分类和确认/拒绝决定
    GlossaryReview {
        /// glossary-audit 生成的 *.ruby-candidates.json
        review_file: PathBuf,

        /// 候选的稳定 candidate_id
        #[arg(long)]
        candidate_id: String,

        /// unclassified|phonetic_reading|orthographic_alias|nickname_cue|semantic_ruby|ordinary_reading|wordplay_token
        #[arg(long)]
        classification: String,

        /// pending|confirmed|rejected
        #[arg(long)]
        decision: String,

        /// confirmed 时使用的中文译名；省略则使用候选 base_dst
        #[arg(long)]
        target: Option<String>,

        /// 人工审核备注
        #[arg(long)]
        note: Option<String>,
    },

    /// 自动生成术语表（NER实体识别 + LLM翻译）
    Glossary {
        /// 输入 EPUB/TXT 文件路径
        file: PathBuf,

        /// NER 模型目录（包含 model.safetensors, config.json, vocab.txt 等）
        #[arg(long, default_value = "./ner_model")]
        model_dir: String,

        /// NER 批处理大小
        #[arg(long, default_value_t = 16)]
        ner_batch_size: usize,

        /// 最小出现次数
        #[arg(long, default_value_t = 2)]
        min_count: usize,

        /// LLM API BASE_URL（如 https://api.deepseek.com/v1）
        #[arg(long, default_value = config::DEFAULT_LLM_URL)]
        llm_url: String,

        /// LLM API Key
        #[arg(long, env = "LLM_API_KEY")]
        llm_key: Option<String>,

        /// LLM 模型名
        #[arg(long, default_value = "deepseek-v4-flash")]
        llm_model: String,

        /// LLM 翻译并发数
        #[arg(long, default_value_t = 4)]
        llm_workers: usize,

        /// 输出路径
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// 调试模式
        #[arg(short, long)]
        debug: bool,
    },
}

/// 将模型名称中的不安全文件名字符替换为下划线
fn sanitize_model_name(model: &str) -> String {
    model
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            _ => c,
        })
        .collect()
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Route to subcommand if present
    if let Some(command) = cli.command {
        return run_subcommand(command).await;
    }

    // Legacy flat mode - require input
    let input = match cli.input {
        Some(p) => p,
        None => {
            eprintln!("错误: 请提供输入文件路径，或使用子命令 (如 glossary)");
            eprintln!("用法: alnilam <INPUT> 或 alnilam glossary <FILE>");
            std::process::exit(1);
        }
    };

    // Initialize tracing
    let filter = if let Ok(env_filter) = EnvFilter::try_from_default_env() {
        env_filter
    } else if cli.debug {
        EnvFilter::new("alnilam=debug")
    } else {
        EnvFilter::new("alnilam=info")
    };
    tracing_subscriber::fmt().with_env_filter(filter).init();

    // Validate input
    if !input.exists() {
        anyhow::bail!("输入文件不存在: {}", input.display());
    }

    let ext = input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let is_txt = ext == "txt";

    // Determine output path (with [model_name] suffix)
    let model_tag = sanitize_model_name(&cli.model);
    let output = cli.output.unwrap_or_else(|| {
        let stem = input.file_stem().unwrap_or_default().to_string_lossy();
        let new_ext = if is_txt { "txt" } else { "epub" };
        input
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join(format!("{}.ja-zh[{}].{}", stem, model_tag, new_ext))
    });

    // Determine rules path (CLI > embedded > filesystem fallback)
    let rules_path = cli.rules_path.or_else(|| {
        let candidates = [
            PathBuf::from("vendor/dynamic_context_detector_v2/rules/ja2zh_context_rules.json"),
            PathBuf::from("../vendor/dynamic_context_detector_v2/rules/ja2zh_context_rules.json"),
            // Legacy paths
            PathBuf::from("dynamic_context_detector_v2/rules/ja2zh_context_rules.json"),
            PathBuf::from("../dynamic_context_detector_v2/rules/ja2zh_context_rules.json"),
        ];
        candidates.into_iter().find(|p| p.exists())
    });

    // Determine translation gap
    let translation_gap = if cli.no_gap || cli.gap == "0" {
        None
    } else {
        Some(cli.gap)
    };

    let pipeline_config = config::PipelineConfig {
        llm_url: cli.llm_url,
        model: cli.model,
        batch_size: cli.batch_size,
        context_lines: cli.context_lines,
        workers: cli.workers,
        max_retry: cli.max_retry,
        mode: match cli.mode.as_str() {
            "replace" => config::TranslationMode::Replace,
            _ => config::TranslationMode::Bilingual,
        },
        apply_fixes: !cli.no_fix,
        rules_path,
        translation_gap,
        temperature: config::DEFAULT_TEMPERATURE,
        top_p: config::DEFAULT_TOP_P,
        top_k: config::DEFAULT_TOP_K,
        glossary_path: cli.glossary_path,
        api_key: cli
            .api_key
            .map(|key| key.trim().to_string())
            .filter(|key| !key.is_empty()),
    };
    pipeline_config.validate()?;
    for warning in pipeline_config.api_security_warnings() {
        eprintln!("安全提示: {}", warning);
    }

    let success = if is_txt {
        pipeline::translate_txt(&input, &output, &pipeline_config, None, None).await?
    } else {
        pipeline::translate_epub(&input, &output, &pipeline_config, None, None).await?
    };

    std::process::exit(if success { 0 } else { 1 });
}

async fn run_subcommand(cmd: Commands) -> Result<()> {
    match cmd {
        Commands::Export {
            input,
            translation_data,
            output,
            mode,
            gap,
            apply_fixes,
        } => {
            if !input.exists() {
                anyhow::bail!("输入文件不存在: {}", input.display());
            }
            if !translation_data.exists() {
                anyhow::bail!("翻译数据不存在: {}", translation_data.display());
            }
            pipeline::validate_distinct_input_output(&input, &output)?;

            let mode = match mode.as_str() {
                "bilingual" => config::TranslationMode::Bilingual,
                "replace" => config::TranslationMode::Replace,
                other => anyhow::bail!("不支持的翻译模式: {}", other),
            };
            let gap = if gap == "0" { None } else { Some(gap) };
            if let Some(value) = gap.as_deref() {
                config::validate_css_length(value)?;
            }

            let extension = input
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            match extension.as_str() {
                "epub" => {
                    let data = alnilam::epub::EpubHandler::load_translation_data(
                        &translation_data.to_string_lossy(),
                    )?;
                    pipeline::export_epub_from_data(
                        &input,
                        &data,
                        &output,
                        mode,
                        gap.as_deref(),
                        apply_fixes,
                        &None,
                    )?;
                }
                "txt" => {
                    let data =
                        pipeline::load_txt_translation_data(&translation_data.to_string_lossy())?;
                    pipeline::export_txt_from_data(&data, &output, mode, &None)?;
                }
                other => anyhow::bail!("不支持的输入格式: .{}", other),
            }
            println!("导出完成: {}", output.display());
        }
        Commands::Glossary {
            file,
            model_dir,
            ner_batch_size,
            min_count,
            llm_url,
            llm_key,
            llm_model,
            llm_workers,
            output,
            debug,
        } => {
            let filter = if debug {
                EnvFilter::new("bellatrix=debug,alnilam=debug")
            } else {
                EnvFilter::new("bellatrix=info,alnilam=info")
            };
            tracing_subscriber::fmt().with_env_filter(filter).init();

            if !file.exists() {
                anyhow::bail!("输入文件不存在: {}", file.display());
            }

            let api_key = llm_key.unwrap_or_else(|| {
                eprintln!("错误: 需要 LLM API Key，使用 --llm-key 或设置 LLM_API_KEY 环境变量");
                std::process::exit(1);
            });

            // Check that model is generic (non-Orion)
            if !bellatrix::is_generic_model(&llm_model) {
                eprintln!(
                    "警告: 术语表生成需要通用模型（如 deepseek-v4-flash），当前模型 \"{}\" 是专用模型",
                    llm_model
                );
                eprintln!("       专用模型不支持 NER 术语翻译任务，请切换到通用模型");
                std::process::exit(1);
            }

            // Build progress callback that prints to console
            use std::sync::Arc;
            let progress: bellatrix::GlossaryProgressCallback =
                Some(Arc::new(|event| match event {
                    bellatrix::GlossaryProgressEvent::StageStarted { stage, detail } => {
                        println!("📋 [{}] {}", stage, detail);
                    }
                    bellatrix::GlossaryProgressEvent::NerProgress { completed, total } => {
                        print!("\r🔍 NER进度: {}/{}  ", completed, total);
                        if completed == total {
                            println!();
                        }
                    }
                    bellatrix::GlossaryProgressEvent::LlmProgress { completed, total } => {
                        print!("\r🌐 LLM翻译进度: {}/{}  ", completed, total);
                        if completed == total {
                            println!();
                        }
                    }
                    bellatrix::GlossaryProgressEvent::Log { message } => {
                        println!("  {}", message);
                    }
                    bellatrix::GlossaryProgressEvent::Completed {
                        output_path,
                        entry_count,
                    } => {
                        println!("✅ 术语表已保存: {} ({} 条)", output_path, entry_count);
                    }
                    bellatrix::GlossaryProgressEvent::Error { message } => {
                        eprintln!("❌ {}", message);
                    }
                }));

            let (lines, ruby_annotations) = extract_text_data(&file)?;
            let config = bellatrix::GlossaryConfig {
                lines,
                ruby_annotations,
                model_dir,
                ner_batch_size,
                min_count,
                llm_url,
                llm_api_key: api_key,
                llm_model,
                llm_workers,
                output_path: output.unwrap_or_else(|| {
                    let base_name = file
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    let parent = file.parent().unwrap_or(std::path::Path::new("."));
                    parent.join(format!("{}_glossary.json", base_name))
                }),
                skip_llm_translation: false,
            };

            let result = bellatrix::generate_glossary(config, progress).await?;
            println!("📁 术语表输出: {}", result.display());
        }
        Commands::GlossaryAudit {
            file,
            glossary_path,
            output,
        } => {
            if !file.exists() {
                anyhow::bail!("输入文件不存在: {}", file.display());
            }
            if !glossary_path.exists() {
                anyhow::bail!("术语表不存在: {}", glossary_path.display());
            }
            let (lines, ruby_annotations) = extract_text_data(&file)?;
            if ruby_annotations.is_empty() {
                anyhow::bail!("EPUB 中未抽取到 ruby 标注");
            }
            let glossary = alnilam::llm::glossary::load_glossary(&glossary_path)?;
            let translations: Vec<bellatrix::llm::TranslationEntry> = glossary
                .into_iter()
                .map(|entry| bellatrix::llm::TranslationEntry {
                    src: entry.src,
                    dst: entry.dst,
                    info: entry.info,
                })
                .collect();
            let output = output.unwrap_or_else(|| bellatrix::ruby_review_path(&glossary_path));
            pipeline::validate_distinct_input_output(&glossary_path, &output)?;
            let review = bellatrix::refresh_ruby_alias_review(
                &output,
                &ruby_annotations,
                &lines,
                &translations,
            )?;

            let actionable = review
                .candidates
                .iter()
                .filter(|entry| bellatrix::is_actionable_review_status(&entry.status))
                .count();
            println!(
                "Ruby审核完成: {}（共 {} 条，{} 条需审核）",
                output.display(),
                review.candidates.len(),
                actionable
            );
        }
        Commands::GlossaryReview {
            review_file,
            candidate_id,
            classification,
            decision,
            target,
            note,
        } => {
            if !review_file.exists() {
                anyhow::bail!("Ruby审核文件不存在: {}", review_file.display());
            }
            let classification = bellatrix::RubyAliasClassification::parse(&classification)?;
            let decision = bellatrix::CandidateDecision::parse(&decision)?;
            let updated = bellatrix::update_ruby_alias_decision(
                &review_file,
                &candidate_id,
                classification,
                decision,
                target,
                note,
            )?;
            println!(
                "Ruby候选已更新: {} / {} -> {:?} {:?}（revision {}）",
                updated.base,
                updated.reading,
                updated.classification,
                updated.decision,
                updated.revision
            );
        }
    }

    Ok(())
}

/// Extract text lines from an EPUB or TXT file
fn extract_text_data(
    path: &std::path::Path,
) -> Result<(Vec<String>, Vec<betelgeuse::RubyAnnotation>)> {
    let ext = path
        .extension()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();

    match ext.as_str() {
        "epub" => {
            let extraction = betelgeuse::extract_epub_text(path)?;
            Ok((extraction.lines, extraction.ruby_annotations))
        }
        "txt" => {
            let lines = betelgeuse::extract_txt_lines(path)?;
            Ok((lines, Vec::new()))
        }
        _ => {
            anyhow::bail!("不支持的文件格式: .{} (仅支持 .epub / .txt)", ext);
        }
    }
}
