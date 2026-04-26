mod analysis;
mod detector;
mod llm;
mod ner_client;

use anyhow::Result;
use clap::Parser;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(name = "mintaka")]
#[command(about = "日语轻小说人物识别器 - Rust高性能版")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// 从 .txt/.epub 识别人名并可选生成术语表
    Run {
        /// 输入文件路径 (支持 .txt, .epub)
        file: PathBuf,

        /// 最小出现次数（低于此数的人物将被过滤）
        #[arg(short, long, default_value_t = 2)]
        min_count: usize,

        /// NER API 批处理大小
        #[arg(short, long, default_value_t = 16)]
        batch_size: usize,

        /// 并发请求数
        #[arg(short, long, default_value_t = 8)]
        workers: usize,

        /// NER 后端 API 地址
        #[arg(long, default_value = "http://127.0.0.1:3000")]
        api_url: String,

        /// LLM API BASE_URL（如 https://api.deepseek.com/v1）
        #[arg(long, default_value = "https://api.deepseek.com/v1")]
        llm_url: String,

        /// LLM API Key (也可通过 LLM_API_KEY 环境变量设置)
        #[arg(long, env = "LLM_API_KEY")]
        llm_key: Option<String>,

        /// LLM 模型名
        #[arg(long, default_value = "deepseek-v4-flash")]
        llm_model: String,

        /// LLM 翻译并发数
        #[arg(long, default_value_t = 4)]
        llm_workers: usize,

        /// 跳过 LLM 翻译步骤（仅输出 NER 识别结果）
        #[arg(long)]
        skip_llm: bool,
    },

    /// 对已有 *_output.json 做离线质量分析
    Analyze {
        /// 术语表 JSON 文件路径（mintaka 输出的 *_output.json）
        output: PathBuf,
    },

    /// 对已有 *_characters.json 做离线质量分析
    AnalyzeChars {
        /// NER 聚合结果 JSON 文件路径（mintaka 输出的 *_characters.json）
        characters: PathBuf,
    },

    /// 批量分析目录内的 *_output.json 或 *_glossary.json
    AnalyzeDir {
        /// 目录路径
        dir: PathBuf,
    },

    /// 批量分析目录内的 *_characters.json
    AnalyzeCharsDir {
        /// 目录路径
        dir: PathBuf,
    },

    /// 一键生成可用 glossary（会输出 *_glossary.json）
    Glossary {
        /// 输入文件路径 (支持 .txt, .epub)
        file: PathBuf,

        /// 最小出现次数（低于此数的人物将被过滤）
        #[arg(short, long, default_value_t = 2)]
        min_count: usize,

        /// NER API 批处理大小
        #[arg(short, long, default_value_t = 16)]
        batch_size: usize,

        /// 并发请求数
        #[arg(short, long, default_value_t = 8)]
        workers: usize,

        /// NER 后端 API 地址
        #[arg(long, default_value = "http://127.0.0.1:3000")]
        api_url: String,

        /// LLM API BASE_URL（如 https://api.deepseek.com/v1）
        #[arg(long, default_value = "https://api.deepseek.com/v1")]
        llm_url: String,

        /// LLM API Key (也可通过 LLM_API_KEY 环境变量设置)
        #[arg(long, env = "LLM_API_KEY")]
        llm_key: Option<String>,

        /// LLM 模型名
        #[arg(long, default_value = "deepseek-v4-flash")]
        llm_model: String,

        /// LLM 翻译并发数
        #[arg(long, default_value_t = 4)]
        llm_workers: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    let cli = Cli::parse();

    match cli.command {
        Command::Analyze { output } => {
            if !output.exists() {
                eprintln!("❌ 文件不存在: {}", output.display());
                std::process::exit(1);
            }
            analysis::analyze_output_json(&output)?;
        }
        Command::AnalyzeChars { characters } => {
            if !characters.exists() {
                eprintln!("❌ 文件不存在: {}", characters.display());
                std::process::exit(1);
            }
            analysis::analyze_characters_json(&characters)?;
        }
        Command::AnalyzeDir { dir } => {
            if !dir.exists() {
                eprintln!("❌ 目录不存在: {}", dir.display());
                std::process::exit(1);
            }
            analysis::analyze_output_dir(&dir)?;
        }
        Command::AnalyzeCharsDir { dir } => {
            if !dir.exists() {
                eprintln!("❌ 目录不存在: {}", dir.display());
                std::process::exit(1);
            }
            analysis::analyze_characters_dir(&dir)?;
        }
        Command::Glossary {
            file,
            min_count,
            batch_size,
            workers,
            api_url,
            llm_url,
            llm_key,
            llm_model,
            llm_workers,
        } => {
            if !file.exists() {
                eprintln!("❌ 文件不存在: {}", file.display());
                std::process::exit(1);
            }
            let api_key = llm_key.unwrap_or_else(|| {
                eprintln!("❌ 需要 LLM API Key: 使用 --llm-key 或设置 LLM_API_KEY 环境变量");
                std::process::exit(1);
            });

            let lines = read_input_file(&file)?;
            if lines.is_empty() {
                eprintln!("❌ 文件内容为空");
                std::process::exit(1);
            }
            let total_chars: usize = lines.iter().map(|l| l.len()).sum();
            println!("📊 总文本行数: {} (共 {} 字符)", lines.len(), total_chars);

            println!("🔍 连接到NER API服务器: {} ...", api_url);
            let ner_client = ner_client::NerClient::new(&api_url);
            match ner_client.health_check().await {
                Ok(true) => println!("✅ API服务器连接成功"),
                _ => {
                    eprintln!("❌ API服务器连接失败，请确保 rigel 正在运行");
                    std::process::exit(1);
                }
            }

            let characters =
                detector::detect_characters(&lines, &ner_client, batch_size, workers, min_count)
                    .await?;
            if characters.is_empty() {
                println!("❌ 未找到出现次数≥{}的人物", min_count);
                return Ok(());
            }

            let base_name = file
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let parent = file.parent().unwrap_or(Path::new("."));

            let llm_client = llm::LlmClient::new(&llm_url, &api_key, &llm_model);
            println!(
                "\n🌐 开始LLM翻译 ({} 个人物, {} 并发)...",
                characters.len(),
                llm_workers
            );
            let translations = llm_client.translate_all(&characters, llm_workers).await;

            let glossary_file = parent.join(format!("{}_glossary.json", base_name));
            let output_json = serde_json::to_string_pretty(&translations)?;
            std::fs::write(&glossary_file, &output_json)?;
            println!(
                "💾 glossary 已保存到: {} ({} 条)",
                glossary_file.display(),
                translations.len()
            );

            analysis::analyze_output_json(&glossary_file)?;
        }
        Command::Run {
            file,
            min_count,
            batch_size,
            workers,
            api_url,
            llm_url,
            llm_key,
            llm_model,
            llm_workers,
            skip_llm,
        } => {
            if !file.exists() {
                eprintln!("❌ 文件不存在: {}", file.display());
                std::process::exit(1);
            }

            let lines = read_input_file(&file)?;
            if lines.is_empty() {
                eprintln!("❌ 文件内容为空");
                std::process::exit(1);
            }

            let total_chars: usize = lines.iter().map(|l| l.len()).sum();
            println!("📊 总文本行数: {} (共 {} 字符)", lines.len(), total_chars);

            println!("🔍 连接到NER API服务器: {} ...", api_url);
            let ner_client = ner_client::NerClient::new(&api_url);
            match ner_client.health_check().await {
                Ok(true) => println!("✅ API服务器连接成功"),
                _ => {
                    eprintln!("❌ API服务器连接失败，请确保 rigel 正在运行");
                    std::process::exit(1);
                }
            }

            let characters =
                detector::detect_characters(&lines, &ner_client, batch_size, workers, min_count)
                    .await?;

            if characters.is_empty() {
                println!("❌ 未找到出现次数≥{}的人物", min_count);
                return Ok(());
            }

            print_character_summary(&characters);

            let base_name = file
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let parent = file.parent().unwrap_or(Path::new("."));

            let characters_file = parent.join(format!("{}_characters.json", base_name));
            let json = serde_json::to_string_pretty(&characters)?;
            std::fs::write(&characters_file, &json)?;
            println!("💾 NER结果已保存到: {}", characters_file.display());

            if !skip_llm {
                let api_key = llm_key.unwrap_or_else(|| {
                    eprintln!("❌ 需要 LLM API Key: 使用 --llm-key 或设置 LLM_API_KEY 环境变量");
                    eprintln!("   或使用 --skip-llm 跳过翻译步骤");
                    std::process::exit(1);
                });

                println!(
                    "\n🌐 开始LLM翻译 ({} 个人物, {} 并发)...",
                    characters.len(),
                    llm_workers
                );

                let llm_client = llm::LlmClient::new(&llm_url, &api_key, &llm_model);
                let translations = llm_client.translate_all(&characters, llm_workers).await;

                let output_file = parent.join(format!("{}_output.json", base_name));
                let output_json = serde_json::to_string_pretty(&translations)?;
                std::fs::write(&output_file, &output_json)?;
                println!(
                    "💾 翻译结果已保存到: {} ({} 条)",
                    output_file.display(),
                    translations.len()
                );
            }
        }
    }

    Ok(())
}

fn read_input_file(path: &Path) -> Result<Vec<String>> {
    let ext = path
        .extension()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();

    match ext.as_str() {
        "epub" => {
            println!("📚 解析EPUB文件...");
            let lines = betelgeuse::extract_epub_lines(path)?;
            println!("📖 从EPUB提取了 {} 行文本", lines.len());
            Ok(lines)
        }
        "txt" => {
            println!("📄 解析TXT文件...");
            let lines = betelgeuse::extract_txt_lines(path)?;
            println!("📖 从TXT提取了 {} 行文本", lines.len());
            Ok(lines)
        }
        _ => {
            anyhow::bail!("不支持的文件格式: .{}\n   支持格式: .txt, .epub", ext);
        }
    }
}

fn print_character_summary(characters: &HashMap<String, detector::CharacterInfo>) {
    println!("\n{}", "=".repeat(50));
    println!("🎭 识别的人物:");
    println!("{}", "=".repeat(50));

    let mut sorted: Vec<_> = characters.values().collect();
    sorted.sort_by(|a, b| b.count.cmp(&a.count));

    for info in &sorted {
        println!("\n👤 {}", info.name);
        println!("   出现次数: {}", info.count);
        if let Some(first) = info.content.first() {
            println!("   首次出现: 第{}行", first.line);
        }
        for (i, mention) in info.content.iter().take(3).enumerate() {
            let preview: String = mention.line_text.chars().take(50).collect();
            println!("   [{}] 第{}行: {}...", i + 1, mention.line, preview);
        }
        if info.content.len() > 3 {
            println!("   ... 还有{}处提及", info.content.len() - 3);
        }
    }
}
