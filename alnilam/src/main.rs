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
        #[arg(long, default_value = "deepseek-chat")]
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
        api_key: cli.api_key,
    };

    let success = if is_txt {
        pipeline::translate_txt(&input, &output, &pipeline_config, None, None).await?
    } else {
        pipeline::translate_epub(&input, &output, &pipeline_config, None, None).await?
    };

    std::process::exit(if success { 0 } else { 1 });
}

async fn run_subcommand(cmd: Commands) -> Result<()> {
    match cmd {
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
                    "警告: 术语表生成需要通用模型（如 deepseek-chat），当前模型 \"{}\" 是专用模型",
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

            let config = bellatrix::GlossaryConfig {
                lines: extract_text_lines(&file)?,
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
    }

    Ok(())
}

/// Extract text lines from an EPUB or TXT file
fn extract_text_lines(path: &std::path::Path) -> Result<Vec<String>> {
    let ext = path
        .extension()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();

    match ext.as_str() {
        "epub" => {
            let lines = betelgeuse::extract_epub_lines(path)?;
            Ok(lines)
        }
        "txt" => {
            let lines = betelgeuse::extract_txt_lines(path)?;
            Ok(lines)
        }
        _ => {
            anyhow::bail!("不支持的文件格式: .{} (仅支持 .epub / .txt)", ext);
        }
    }
}
