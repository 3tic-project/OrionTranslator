//! ModernBERT NER CLI.
//!
//! Benchmarks the available backends on a sample of the document, runs the whole
//! document on whichever was fastest, then writes a report plus the aggregated
//! entities. Designed so that dropping a `.txt` onto the executable just works.

// `burn::backend::Wgpu` is a deeply nested `Fusion<CubeBackend<..>>`; the default
// limit is not enough to prove `Sync` for it.
#![recursion_limit = "512"]

mod progress;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use modernbert_ner::{
    aggregate_characters, characters_to_markdown, collect_raw_mentions, estimate_pad_waste,
    pack_texts, sweep_thresholds, AggregateConfig, BatchProfile, CpuNerPipeline, InferOptions,
    NerResult, ProfileAccum,
};
#[cfg(any(feature = "wgpu", feature = "ndarray"))]
use modernbert_ner::{load_pipeline, NerPipeline};
use progress::Progress;
use rayon::prelude::*;
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// Burn's ndarray backend allocates a fresh buffer per tensor op; the system allocator
// serialises those across worker threads and dominated the profile (malloc/madvise >> sgemm).
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Marker the GPU probe child process prints so the parent can read its result.
const BENCH_MARKER: &str = "BENCH_CHARS_PER_SEC ";

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BackendKind {
    /// Time every available backend on a sample, then use the fastest.
    Auto,
    /// Hand-written CPU engine.
    Cpu,
    /// GPU via Burn wgpu.
    #[cfg(feature = "wgpu")]
    Wgpu,
    /// Burn `ndarray` reference implementation, for cross-checking `cpu`.
    #[cfg(feature = "ndarray")]
    BurnCpu,
}

impl BackendKind {
    fn label(self) -> &'static str {
        match self {
            BackendKind::Auto => "auto",
            BackendKind::Cpu => "cpu",
            #[cfg(feature = "wgpu")]
            BackendKind::Wgpu => "wgpu",
            #[cfg(feature = "ndarray")]
            BackendKind::BurnCpu => "burn-cpu",
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "modernbert-ner",
    about = "ModernBERT-JA character NER with automatic backend benchmarking",
    long_about = "Drop a .txt file onto this executable, or pass it as the first argument.\n\
                  Each backend is timed on a sample of the document; the fastest one then\n\
                  processes the whole file. Results are written next to the input."
)]
struct Cli {
    /// Input .txt, one line per sentence or paragraph.
    #[arg(value_name = "INPUT.txt")]
    input: Option<PathBuf>,

    /// Model directory (config.json + model.safetensors + tokenizer.json).
    /// Auto-discovered next to the executable or under ./models when omitted.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Analyse this string instead of a file.
    #[arg(long)]
    text: Option<String>,

    /// Report directory. Defaults to `<input>_ner` beside the input file.
    #[arg(long)]
    out_dir: Option<PathBuf>,

    /// Additionally write the per-line JSONL to this exact path.
    #[arg(long)]
    output: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
    backend: BackendKind,

    /// Lines sampled for the backend benchmark.
    #[arg(long, default_value_t = 400)]
    bench_lines: usize,

    /// Do not wait for a keypress before exiting.
    #[arg(long, default_value_t = false)]
    no_wait: bool,

    /// Internal: time one backend on the input and print the rate.
    #[arg(long, value_enum, hide = true)]
    bench_only: Option<BackendKind>,

    #[arg(long, default_value_t = 256)]
    max_length: usize,

    /// Sentences per micro-batch. Default: 24 on CPU, 128 on GPU.
    #[arg(long)]
    batch_size: Option<usize>,

    /// Token budget per micro-batch. Default: 1536 on CPU, 32768 on GPU.
    #[arg(long)]
    max_tokens: Option<usize>,

    #[arg(long, default_value_t = false)]
    no_sort: bool,

    #[arg(long, default_value_t = false)]
    skip_scores: bool,

    #[arg(long, default_value_t = 0)]
    jobs: usize,

    /// Confidence floor for aggregation.
    #[arg(long, default_value_t = 0.9)]
    min_score: f32,

    /// Minimum mention count for a name to be kept.
    #[arg(long, default_value_t = 2)]
    min_count: usize,

    /// Entity types to aggregate, comma-separated.
    #[arg(long, default_value = "PER")]
    labels: String,

    /// Write threshold sweep JSON into out-dir.
    #[arg(long, default_value_t = true)]
    threshold_report: bool,

    #[arg(long, default_value_t = false)]
    profile: bool,
}

/// Directories to search when `--model` is omitted, so drag-and-drop needs no flags.
fn discover_model() -> Result<PathBuf> {
    let looks_like_model = |p: &Path| {
        p.join("config.json").is_file()
            && p.join("model.safetensors").is_file()
            && p.join("tokenizer.json").is_file()
    };

    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(env) = std::env::var("MODERNBERT_NER_MODEL") {
        roots.push(PathBuf::from(env));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            roots.push(dir.join("model"));
            roots.push(dir.join("models"));
            roots.push(dir.to_path_buf());
        }
    }
    roots.push(PathBuf::from("models"));
    roots.push(PathBuf::from("."));

    for root in &roots {
        if looks_like_model(root) {
            return Ok(root.clone());
        }
        // One level down, e.g. models/modernbert_ja_30m_combined_ja.
        if let Ok(entries) = std::fs::read_dir(root) {
            let mut candidates: Vec<PathBuf> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| looks_like_model(p))
                .collect();
            candidates.sort();
            if let Some(found) = candidates.into_iter().next() {
                return Ok(found);
            }
        }
    }

    bail!(
        "could not find a model directory. Pass --model <dir>, set MODERNBERT_NER_MODEL, \
         or place the model beside the executable (it needs config.json, model.safetensors \
         and tokenizer.json)."
    )
}

fn main() {
    let code = match real_main() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!();
            eprintln!("ERROR: {e:#}");
            1
        }
    };
    std::process::exit(code);
}

fn real_main() -> Result<()> {
    // The GPU probe is noisy; keep its adapter chatter out of the normal run.
    let default_level = if std::env::args().any(|a| a == "--bench-only") {
        "warn"
    } else {
        "info"
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_level))
        .format_timestamp(None)
        .init();

    let cli = Cli::parse();
    // Drag-and-drop gives exactly one argument, so that is when a pause is useful.
    let wait_on_exit = !cli.no_wait && cli.bench_only.is_none() && std::env::args().count() <= 2;

    let result = run(&cli);

    if wait_on_exit && std::io::stdin().is_terminal() {
        eprintln!();
        eprint!("Press Enter to close this window...");
        let _ = std::io::stderr().flush();
        let mut buf = [0u8; 1];
        let _ = std::io::stdin().read(&mut buf);
    }
    result
}

/// Timing of one backend over the benchmark sample.
struct BenchResult {
    backend: BackendKind,
    chars_per_sec: f64,
    note: String,
}

fn rule(title: &str) {
    eprintln!();
    eprintln!(
        "== {title} {}",
        "=".repeat(60usize.saturating_sub(title.len()))
    );
}

fn run(cli: &Cli) -> Result<()> {
    let model_dir = match &cli.model {
        Some(p) => p.clone(),
        None => discover_model()?,
    };
    let lines = load_lines(cli)?;
    if lines.is_empty() {
        bail!("input has no non-empty lines");
    }

    // Internal probe mode: time one backend and report the rate to the parent.
    if let Some(backend) = cli.bench_only {
        let rate = time_backend(backend, cli, &model_dir, &lines)?;
        println!("{BENCH_MARKER}{rate}");
        return Ok(());
    }
    let total_chars: usize = lines.iter().map(|l| l.chars().count()).sum();
    let mut len: Vec<usize> = lines.iter().map(|l| l.chars().count()).collect();
    len.sort_unstable();

    rule("input");
    eprintln!("model      {}", model_dir.display());
    match &cli.input {
        Some(p) => eprintln!("input      {}", p.display()),
        None => eprintln!("input      <--text>"),
    }
    eprintln!(
        "lines      {} ({} chars, min {} / median {} / p95 {} / max {} chars per line)",
        lines.len(),
        total_chars,
        len[0],
        len[len.len() / 2],
        len[len.len() * 95 / 100],
        len[len.len() - 1]
    );
    eprintln!(
        "threads    {} logical cores, max_length {}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        cli.max_length
    );

    let (backend, benches) = select_backend(cli, &model_dir, &lines)?;

    rule("inference");
    eprintln!("backend    {}", backend.label());
    let load_started = Instant::now();
    let pipeline = load_backend(backend, cli, &model_dir)?;
    eprintln!(
        "model      loaded in {:.2}s",
        load_started.elapsed().as_secs_f32()
    );

    let out_dir = resolve_out_dir(cli);
    let jobs = pipeline.jobs(cli);
    let options = pipeline.options();
    let started = Instant::now();
    let all = pipeline.run(&lines, cli, false)?;
    let elapsed = started.elapsed().as_secs_f32();

    write_outputs(
        cli, &options, &lines, &all, elapsed, jobs, backend, &benches, &out_dir, &model_dir,
    )
}

/// Evenly spaced sample so the benchmark sees the document's real length mix.
fn bench_sample(lines: &[String], want: usize) -> Vec<String> {
    if lines.len() <= want {
        return lines.to_vec();
    }
    let stride = lines.len() as f64 / want as f64;
    (0..want)
        .map(|i| lines[((i as f64 * stride) as usize).min(lines.len() - 1)].clone())
        .collect()
}

fn select_backend(
    cli: &Cli,
    model_dir: &Path,
    lines: &[String],
) -> Result<(BackendKind, Vec<BenchResult>)> {
    if cli.backend != BackendKind::Auto {
        return Ok((cli.backend, Vec::new()));
    }

    let sample = bench_sample(lines, cli.bench_lines.max(1));
    let sample_chars: usize = sample.iter().map(|l| l.chars().count()).sum();
    rule("backend benchmark");
    eprintln!(
        "sample     {} lines / {} chars (evenly spaced across the document)",
        sample.len(),
        sample_chars
    );

    let mut results = Vec::new();

    eprint!("cpu        timing... ");
    let _ = std::io::stderr().flush();
    let cpu_rate = time_backend(BackendKind::Cpu, cli, model_dir, &sample)?;
    eprintln!("{cpu_rate:.0} chars/s");
    results.push(BenchResult {
        backend: BackendKind::Cpu,
        chars_per_sec: cpu_rate,
        note: String::new(),
    });

    #[cfg(feature = "wgpu")]
    {
        eprint!("wgpu       timing... ");
        let _ = std::io::stderr().flush();
        match probe_gpu(cli, model_dir, &sample) {
            Ok(rate) => {
                eprintln!("{rate:.0} chars/s");
                results.push(BenchResult {
                    backend: BackendKind::Wgpu,
                    chars_per_sec: rate,
                    note: String::new(),
                });
            }
            Err(e) => {
                eprintln!("unavailable ({e})");
                results.push(BenchResult {
                    backend: BackendKind::Wgpu,
                    chars_per_sec: 0.0,
                    note: format!("unavailable: {e}"),
                });
            }
        }
    }

    let best = results
        .iter()
        .max_by(|a, b| a.chars_per_sec.total_cmp(&b.chars_per_sec))
        .map(|r| r.backend)
        .unwrap_or(BackendKind::Cpu);
    eprintln!("selected   {} (fastest on the sample)", best.label());
    Ok((best, results))
}

/// Runs the GPU benchmark in a child process: a missing or broken adapter aborts
/// inside wgpu, which would otherwise take the whole run down with it.
#[cfg(feature = "wgpu")]
fn probe_gpu(cli: &Cli, model_dir: &Path, sample: &[String]) -> Result<f64> {
    let tmp = std::env::temp_dir().join(format!("modernbert-ner-probe-{}.txt", std::process::id()));
    std::fs::write(&tmp, sample.join("\n"))?;

    let exe = std::env::current_exe().context("locating own executable")?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg(&tmp)
        .arg("--bench-only")
        .arg("wgpu")
        .arg("--model")
        .arg(model_dir)
        .arg("--max-length")
        .arg(cli.max_length.to_string())
        .arg("--no-wait");
    if let Some(b) = cli.batch_size {
        cmd.arg("--batch-size").arg(b.to_string());
    }
    if let Some(t) = cli.max_tokens {
        cmd.arg("--max-tokens").arg(t.to_string());
    }
    let out = cmd.output().context("spawning GPU probe")?;
    let _ = std::fs::remove_file(&tmp);

    if !out.status.success() {
        bail!("probe exited with {}", out.status);
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|l| l.strip_prefix(BENCH_MARKER))
        .and_then(|v| v.trim().parse::<f64>().ok())
        .ok_or_else(|| anyhow::anyhow!("probe produced no timing"))
}

/// A loaded backend. Keeping the pipeline alive lets the benchmark time inference
/// only, instead of also timing the one-off model load.
enum Loaded {
    Cpu(Box<CpuNerPipeline>),
    #[cfg(feature = "wgpu")]
    Wgpu(Box<NerPipeline<burn::backend::Wgpu>>),
    #[cfg(feature = "ndarray")]
    BurnCpu(Box<NerPipeline<burn::backend::NdArray>>),
}

impl Loaded {
    fn options(&self) -> InferOptions {
        match self {
            Loaded::Cpu(p) => p.options.clone(),
            #[cfg(feature = "wgpu")]
            Loaded::Wgpu(p) => p.options.clone(),
            #[cfg(feature = "ndarray")]
            Loaded::BurnCpu(p) => p.options.clone(),
        }
    }

    fn jobs(&self, cli: &Cli) -> usize {
        match self {
            #[cfg(feature = "wgpu")]
            Loaded::Wgpu(_) => 1,
            _ => resolve_jobs(cli, false),
        }
    }

    fn run(&self, lines: &[String], cli: &Cli, quiet: bool) -> Result<Vec<(usize, NerResult)>> {
        let jobs = self.jobs(cli);
        match self {
            Loaded::Cpu(p) => run_workers(p.as_ref(), lines, cli, jobs, quiet),
            #[cfg(feature = "wgpu")]
            Loaded::Wgpu(p) => run_workers(p.as_ref(), lines, cli, jobs, quiet),
            #[cfg(feature = "ndarray")]
            Loaded::BurnCpu(p) => run_workers(p.as_ref(), lines, cli, jobs, quiet),
        }
    }
}

fn load_backend(backend: BackendKind, cli: &Cli, model_dir: &Path) -> Result<Loaded> {
    match backend {
        BackendKind::Auto | BackendKind::Cpu => Ok(Loaded::Cpu(Box::new(
            CpuNerPipeline::load(model_dir, cli.max_length)?.with_options(make_options(cli, false)),
        ))),
        #[cfg(feature = "wgpu")]
        BackendKind::Wgpu => {
            use burn::backend::wgpu::{Wgpu, WgpuDevice};
            Ok(Loaded::Wgpu(Box::new(
                load_pipeline::<Wgpu>(model_dir, WgpuDevice::default(), cli.max_length)?
                    .with_options(make_options(cli, true)),
            )))
        }
        #[cfg(feature = "ndarray")]
        BackendKind::BurnCpu => {
            use burn::backend::NdArray;
            Ok(Loaded::BurnCpu(Box::new(
                load_pipeline::<NdArray>(model_dir, Default::default(), cli.max_length)?
                    .with_options(make_options(cli, false)),
            )))
        }
    }
}

/// One warm-up pass (allocator, autotune, shader cache) then a timed pass over the
/// whole sample. Model loading happens before the clock starts.
fn time_backend(
    backend: BackendKind,
    cli: &Cli,
    model_dir: &Path,
    sample: &[String],
) -> Result<f64> {
    let pipeline = load_backend(backend, cli, model_dir)?;
    let warmup = &sample[..sample.len().div_ceil(4)];
    pipeline.run(warmup, cli, true)?;

    let chars: usize = sample.iter().map(|l| l.chars().count()).sum();
    let t = Instant::now();
    pipeline.run(sample, cli, true)?;
    Ok(chars as f64 / t.elapsed().as_secs_f64().max(1e-9))
}

fn resolve_out_dir(cli: &Cli) -> PathBuf {
    if let Some(dir) = &cli.out_dir {
        return dir.clone();
    }
    match &cli.input {
        Some(input) => {
            let stem = input
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "ner".into());
            input
                .parent()
                .unwrap_or(Path::new("."))
                .join(format!("{stem}_ner"))
        }
        None => PathBuf::from("ner_report"),
    }
}

fn resolve_jobs(cli: &Cli, is_gpu: bool) -> usize {
    if is_gpu {
        return 1;
    }
    if cli.jobs > 0 {
        cli.jobs
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(1)
    }
}

fn load_lines(cli: &Cli) -> Result<Vec<String>> {
    if let Some(text) = &cli.text {
        Ok(vec![text.clone()])
    } else {
        let raw = std::fs::read_to_string(cli.input.as_ref().unwrap())?;
        Ok(raw
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|s| s.to_string())
            .collect())
    }
}

fn make_options(cli: &Cli, is_gpu: bool) -> InferOptions {
    // The GPU only reaches peak throughput with large micro-batches; the CPU engine
    // prefers smaller ones that keep scratch buffers cache-resident.
    let (batch, tokens) = if is_gpu { (128, 32768) } else { (24, 1536) };
    InferOptions {
        max_sentences: cli.batch_size.unwrap_or(batch).max(1),
        max_tokens: cli.max_tokens.unwrap_or(tokens).max(1),
        sort_by_length: !cli.no_sort,
        skip_scores: cli.skip_scores,
    }
}

/// One micro-batch of sentences through whichever backend the CLI selected.
trait PackPredictor: Clone + Send {
    fn options(&self) -> &InferOptions;
    fn predict_pack(&self, texts: &[&str]) -> Result<(Vec<NerResult>, BatchProfile)>;
}

impl PackPredictor for CpuNerPipeline {
    fn options(&self) -> &InferOptions {
        &self.options
    }
    fn predict_pack(&self, texts: &[&str]) -> Result<(Vec<NerResult>, BatchProfile)> {
        self.predict_batch_profiled(texts)
    }
}

#[cfg(feature = "wgpu")]
impl PackPredictor for NerPipeline<burn::backend::Wgpu> {
    fn options(&self) -> &InferOptions {
        &self.options
    }
    fn predict_pack(&self, texts: &[&str]) -> Result<(Vec<NerResult>, BatchProfile)> {
        self.predict_batch_profiled(texts, true)
    }
}

#[cfg(feature = "ndarray")]
impl PackPredictor for NerPipeline<burn::backend::NdArray> {
    fn options(&self) -> &InferOptions {
        &self.options
    }
    fn predict_pack(&self, texts: &[&str]) -> Result<(Vec<NerResult>, BatchProfile)> {
        self.predict_batch_profiled(texts, true)
    }
}

fn run_workers<P: PackPredictor>(
    base: &P,
    lines: &[String],
    cli: &Cli,
    jobs: usize,
    quiet: bool,
) -> Result<Vec<(usize, NerResult)>> {
    let options = base.options().clone();

    let packs = pack_texts(
        lines,
        options.max_sentences,
        options.max_tokens,
        options.sort_by_length,
    );
    if !quiet {
        eprintln!(
            "packing    {} lines -> {} micro-batches (pad waste {:.1}%, batch {}, max_tokens {}, jobs {})",
            lines.len(),
            packs.len(),
            estimate_pad_waste(&packs) * 100.0,
            options.max_sentences,
            options.max_tokens,
            jobs
        );
    }

    // Each worker gets its own handle: weights are refcounted, only scratch is per-worker.
    let workers: Vec<P> = (0..jobs).map(|_| base.clone()).collect();

    // Longest-processing-time order + a shared cursor: workers pull the next pack when
    // free, so a slow tail pack cannot stall a statically assigned worker.
    let mut order: Vec<usize> = (0..packs.len()).collect();
    order
        .sort_by_key(|&i| std::cmp::Reverse(packs[i].texts.iter().map(|t| t.len()).sum::<usize>()));

    let total_chars: u64 = lines.iter().map(|l| l.chars().count() as u64).sum();
    let bar = (!quiet).then(|| std::sync::Mutex::new(Progress::new("progress  ", total_chars)));

    let profile_accum = std::sync::Mutex::new(ProfileAccum::default());
    let cursor = AtomicUsize::new(0);
    let total = packs.len();
    let want_profile = cli.profile;
    let packs_ref = &packs;
    let order_ref = &order;
    let bar_ref = bar.as_ref();

    let partials: Vec<Vec<(usize, Vec<NerResult>)>> = workers
        .into_par_iter()
        .map(|pipeline| -> Result<Vec<(usize, Vec<NerResult>)>> {
            let mut local = Vec::new();
            loop {
                let slot = cursor.fetch_add(1, Ordering::Relaxed);
                if slot >= total {
                    break;
                }
                let pack_id = order_ref[slot];
                let pack = &packs_ref[pack_id];
                let refs: Vec<&str> = pack.texts.iter().map(|s| s.as_str()).collect();
                let (results, prof) = pipeline.predict_pack(&refs)?;
                if want_profile {
                    profile_accum.lock().unwrap().add(&prof);
                }
                if let Some(bar) = bar_ref {
                    let done: u64 = pack.texts.iter().map(|t| t.chars().count() as u64).sum();
                    bar.lock().unwrap().add(done);
                }
                local.push((pack_id, results));
            }
            Ok(local)
        })
        .collect::<Result<Vec<_>>>()?;

    if let Some(bar) = bar {
        bar.into_inner().unwrap().finish();
    }

    let mut by_pack: Vec<Option<Vec<NerResult>>> = (0..packs.len()).map(|_| None).collect();
    for part in partials {
        for (pack_id, results) in part {
            by_pack[pack_id] = Some(results);
        }
    }
    let pack_results: Vec<Vec<NerResult>> = by_pack
        .into_iter()
        .map(|x| x.expect("missing pack result"))
        .collect();
    let ordered = modernbert_ner::pack::unsort_results(lines.len(), &packs, pack_results);

    if want_profile && !quiet {
        eprintln!("{}", profile_accum.lock().unwrap().report());
    }

    Ok(ordered
        .into_iter()
        .enumerate()
        .map(|(i, r)| (i + 1, r))
        .collect())
}

fn parse_labels(s: &str) -> Vec<String> {
    s.split(',')
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn write_outputs(
    cli: &Cli,
    opts: &InferOptions,
    lines: &[String],
    all: &[(usize, NerResult)],
    elapsed: f32,
    jobs: usize,
    backend: BackendKind,
    benches: &[BenchResult],
    dir: &Path,
    model_dir: &Path,
) -> Result<()> {
    let chars: usize = all.iter().map(|(_, r)| r.text.chars().count()).sum();
    let ents: usize = all.iter().map(|(_, r)| r.entities.len()).sum();
    let rate = if elapsed > 0.0 {
        chars as f32 / elapsed
    } else {
        0.0
    };

    eprintln!(
        "completed  {} lines / {} chars in {:.2}s ({:.0} chars/s), {} raw entities",
        all.len(),
        chars,
        elapsed,
        rate,
        ents
    );

    let line_indices: Vec<usize> = (0..lines.len()).collect();
    let line_results: Vec<NerResult> = all.iter().map(|(_, r)| r.clone()).collect();

    let labels = parse_labels(&cli.labels);
    let agg_cfg = AggregateConfig {
        min_score: cli.min_score,
        min_count: cli.min_count,
        labels: labels.clone(),
        ..AggregateConfig::default()
    };

    let characters = aggregate_characters(lines, &line_indices, &line_results, &agg_cfg);
    eprintln!(
        "aggregated {} names (min_score {}, min_count {}, labels {:?})",
        characters.len(),
        cli.min_score,
        cli.min_count,
        labels
    );

    rule("output");
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating report directory {}", dir.display()))?;
    let mut written: Vec<PathBuf> = Vec::new();

    if let Some(out) = &cli.output {
        write_jsonl(out, all)?;
        written.push(out.clone());
    }

    let jsonl = dir.join("ner_lines.jsonl");
    write_jsonl(&jsonl, all)?;
    written.push(jsonl);

    let mut mentions_jsonl = String::new();
    for (line_no, r) in all {
        for e in &r.entities {
            if !labels.is_empty() && !labels.iter().any(|l| e.label.contains(l.as_str())) {
                continue;
            }
            let row = serde_json::json!({
                "line_no": line_no,
                "start": e.start,
                "end": e.end,
                "type": e.label,
                "text": e.text,
                "score": e.score,
                "line_text": r.text,
            });
            mentions_jsonl.push_str(&serde_json::to_string(&row)?);
            mentions_jsonl.push('\n');
        }
    }
    std::fs::write(dir.join("mentions.jsonl"), mentions_jsonl)?;
    written.push(dir.join("mentions.jsonl"));

    std::fs::write(
        dir.join("characters.json"),
        serde_json::to_string_pretty(&characters)? + "\n",
    )?;
    written.push(dir.join("characters.json"));

    let title = cli
        .input
        .as_ref()
        .and_then(|p| p.file_stem())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "NER".into());
    std::fs::write(
        dir.join("characters.md"),
        characters_to_markdown(&characters, &format!("{title} characters"), &agg_cfg),
    )?;
    written.push(dir.join("characters.md"));

    if cli.threshold_report {
        let raw = collect_raw_mentions(lines, &line_indices, &line_results, &labels, true);
        let mut thr: Vec<f32> = (50..=99).map(|x| x as f32 / 100.0).collect();
        for x in [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99] {
            if !thr.iter().any(|t| (*t - x).abs() < 1e-6) {
                thr.push(x);
            }
        }
        thr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let rows = sweep_thresholds(&raw, &thr, cli.min_count);
        std::fs::write(
            dir.join("threshold_sweep.json"),
            serde_json::to_string_pretty(&rows)? + "\n",
        )?;
        written.push(dir.join("threshold_sweep.json"));

        let mut tmd = String::from(
            "# Threshold sweep\n\n| min_score | mentions | unique | single_char |\n|---:|---:|---:|---:|\n",
        );
        for r in &rows {
            if (r.min_score * 100.0).round() as i32 % 5 == 0
                || (r.min_score - 0.9).abs() < 1e-6
                || (r.min_score - 0.95).abs() < 1e-6
            {
                tmd.push_str(&format!(
                    "| {:.2} | {} | {} | {} |\n",
                    r.min_score, r.mentions, r.unique_names, r.single_char_unique
                ));
            }
        }
        std::fs::write(dir.join("threshold_sweep.md"), tmd)?;
        written.push(dir.join("threshold_sweep.md"));
    }

    let bench_json: Vec<serde_json::Value> = benches
        .iter()
        .map(|b| {
            serde_json::json!({
                "backend": b.backend.label(),
                "chars_per_sec": b.chars_per_sec,
                "note": b.note,
            })
        })
        .collect();
    let summary = serde_json::json!({
        "model": model_dir.display().to_string(),
        "input": cli.input.as_ref().map(|p| p.display().to_string()),
        "backend": backend.label(),
        "backend_benchmark": bench_json,
        "lines": all.len(),
        "chars": chars,
        "raw_entities": ents,
        "seconds": elapsed,
        "chars_per_sec": rate,
        "jobs": jobs,
        "batch_size": opts.max_sentences,
        "max_tokens": opts.max_tokens,
        "min_score": cli.min_score,
        "min_count": cli.min_count,
        "labels": labels,
        "characters": characters.len(),
        "top_characters": characters.iter().take(20).map(|c| {
            serde_json::json!({
                "name": c.name,
                "count": c.count,
                "mean_score": c.mean_score,
                "max_score": c.max_score,
            })
        }).collect::<Vec<_>>(),
    });
    std::fs::write(
        dir.join("summary.json"),
        serde_json::to_string_pretty(&summary)? + "\n",
    )?;
    written.push(dir.join("summary.json"));

    let mut report = String::new();
    report.push_str(&format!("# {title} - NER report\n\n"));
    report.push_str("## Run\n\n");
    report.push_str(&format!("- model: `{}`\n", model_dir.display()));
    if let Some(p) = &cli.input {
        report.push_str(&format!("- input: `{}`\n", p.display()));
    }
    report.push_str(&format!("- backend: `{}`\n", backend.label()));
    report.push_str(&format!(
        "- lines: {} ({} chars)\n- elapsed: {elapsed:.2} s ({rate:.0} chars/s)\n",
        all.len(),
        chars
    ));
    report.push_str(&format!(
        "- batching: batch {} / max_tokens {} / jobs {}\n",
        opts.max_sentences, opts.max_tokens, jobs
    ));
    if !benches.is_empty() {
        report.push_str("\n## Backend benchmark\n\n| backend | chars/s | note |\n|---|---:|---|\n");
        for b in benches {
            report.push_str(&format!(
                "| {} | {:.0} | {} |\n",
                b.backend.label(),
                b.chars_per_sec,
                if b.note.is_empty() { "-" } else { &b.note }
            ));
        }
    }
    report.push_str(&format!(
        "\n## Entities\n\n- raw mentions: {ents}\n- aggregated names: {} (min_score {}, min_count {})\n\n",
        characters.len(),
        cli.min_score,
        cli.min_count
    ));
    report.push_str("| name | count | mean score | max score |\n|---|---:|---:|---:|\n");
    for c in characters.iter().take(30) {
        report.push_str(&format!(
            "| {} | {} | {:.3} | {:.3} |\n",
            c.name, c.count, c.mean_score, c.max_score
        ));
    }
    std::fs::write(dir.join("report.md"), report)?;
    written.push(dir.join("report.md"));

    eprintln!("directory  {}", dir.display());
    for p in &written {
        let size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        eprintln!(
            "  {:<24} {:>10} bytes",
            p.file_name().unwrap_or_default().to_string_lossy(),
            size
        );
    }

    rule("top names");
    if characters.is_empty() {
        eprintln!("(none above the thresholds)");
    }
    for c in characters.iter().take(20) {
        eprintln!("  {:<16} x{:<5} mean {:.3}", c.name, c.count, c.mean_score);
    }
    Ok(())
}

fn write_jsonl(path: &Path, all: &[(usize, NerResult)]) -> Result<()> {
    let mut f = String::new();
    for (line_no, r) in all {
        let row = serde_json::json!({
            "line_no": line_no,
            "text": r.text,
            "entities": r.entities,
            "labels": r.labels,
        });
        f.push_str(&serde_json::to_string(&row)?);
        f.push('\n');
    }
    std::fs::write(path, f)?;
    Ok(())
}
