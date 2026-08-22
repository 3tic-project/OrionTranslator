# modernbert-ner

Rust inference for fine-tuned **ModernBERT-JA** token classification NER.

Backends:

| `--backend` | Engine | Use |
|---|---|---|
| `cpu` *(default)* | hand-written f32 kernels over `gemm` | fastest; production batch inference |
| `burn-cpu` | Burn `ndarray` | **reference only, ~3x slower.** Exists so `cpu` can be cross-checked without a GPU |
| `wgpu` | Burn `wgpu` (fusion + autotune) | GPU |

Architecture matches HuggingFace `ModernBertForTokenClassification` (RoPE, GeGLU MLP, local/global alternating attention, prediction head + linear classifier). Character alignment matches training (`is_split_into_words` on Unicode chars + BIOES).

## Model directory

```text
alnilam/ner_model/
  config.json
  model.safetensors
  tokenizer.json
```

## Build

```bash
# Default: CPU engine + Burn reference + GPU (wgpu)
cargo build -p modernbert-ner --release

# CPU only (smaller binary, no GPU stack)
cargo build -p modernbert-ner --release --no-default-features --features ndarray
```

## Usage

最简用法：**把 .txt 拖到可执行文件图标上**。程序自动完成全部流程，跑完后停在
`Press Enter to close this window...`，等你看完报告再退出。

等价命令行：

```bash
modernbert-ner book.txt
```

流程：

1. 定位模型（`--model` > 环境变量 `MODERNBERT_NER_MODEL` > 可执行文件旁的 `model/`、`models/` > `./models`）
2. 从全文**等间隔抽样**若干行（默认 400），逐个后端计时；每个后端先跑一次预热，
   且**只对推理计时，不含模型加载**
3. 用最快的后端跑完全量，带进度条（百分比 / 已处理字数 / 速率 / 已用时 / ETA）
4. 报告与聚合结果写入 `<输入文件名>_ner/`

输出目录内容：

| 文件 | 内容 |
|---|---|
| `report.md` | 运行参数、后端跑分对比、实体统计、Top 30 名称 |
| `summary.json` | 同上的机器可读版本 |
| `ner_lines.jsonl` | 每行的标签与实体 |
| `mentions.jsonl` | 展平后的每个 mention（含所在行原文） |
| `characters.json` / `characters.md` | 聚合后的人名及出处摘录 |
| `threshold_sweep.json` / `.md` | min_score 阈值扫描，用于调参 |

典型输出：

```text
== backend benchmark ===========================================
sample     400 lines / 13419 chars (evenly spaced across the document)
cpu        timing... 15813 chars/s
wgpu       timing... 8183 chars/s
selected   cpu (fastest on the sample)

== inference ===================================================
backend    cpu
model      loaded in 0.27s
packing    3487 lines -> 156 micro-batches (pad waste 1.2%, batch 24, max_tokens 1536, jobs 12)
progress   [================================] 100.0%  128036/128036 chars  17767 chars/s  elapsed 00:00:07  eta 00:00:00
completed  3487 lines / 128036 chars in 7.21s (17761 chars/s), 2166 raw entities
```

### Flags

| Flag | 含义 |
|---|---|
| `--backend auto\|cpu\|wgpu\|burn-cpu` | 默认 `auto`（跑分选最快）；显式指定则跳过跑分 |
| `--bench-lines N` | 跑分抽样行数（默认 400） |
| `--out-dir DIR` | 报告目录（默认 `<输入>_ner`） |
| `--model DIR` | 模型目录，省略时自动探测 |
| `--text "..."` | 分析一段字符串而非文件 |
| `--min-score` / `--min-count` / `--labels` | 聚合阈值与实体类型（默认 0.9 / 2 / PER） |
| `--jobs N` | 并行 worker 数（CPU 默认=逻辑核；GPU 恒为 1） |
| `--batch-size N` / `--max-tokens N` | 每 pack 句数 / token 预算（默认 CPU 24/1536，GPU 128/32768） |
| `--no-sort` / `--skip-scores` | 关闭长度排序 / 只要标签不要置信度 |
| `--no-wait` | 跑完直接退出，不等待按键（脚本中使用） |
| `--profile` | 输出 tokenize/tensor/forward/post 分阶段耗时 |

> GPU 跑分在**子进程**内进行：缺失或异常的适配器会在 wgpu 内部 abort，放进子进程才不会
> 把整次运行带崩；探测失败时自动退回 CPU 并在报告里记录原因。
>
> GPU 需要远大于 CPU 的 micro-batch 才能吃满，默认值已按后端区分。显式传 `--batch-size`
> 会同时覆盖两个后端。

## Performance

基准：整本日文小说 3487 行 / 128k 字，i5-12400（6C/12T）+ Radeon RX 580，`--release`，各后端默认 batching。

| 后端 | 优化前 | 优化后 | 峰值内存 |
|---|---:|---:|---:|
| `cpu`（默认，自研引擎） | 4.1k chars/s | **16.5–20.5k** | 0.73 GB |
| `burn-cpu`（参考实现） | 4.1k | 5.1–5.6k | — |
| `wgpu`（RX 580） | ~6.0k | **~11k**（9.7–12.8k，波动 ±15%） | — |

CPU 约 **4.0×**，GPU 约 **1.8×**；三条路径输出互相一致
（`cpu vs burn-cpu` 与 `cpu vs wgpu` 均为 3487 行 0 处标签/实体差异，置信度最大偏差 <2e-5）。

主要来源：

1. **专用 CPU 引擎**（`src/cpu.rs`）。Burn 的 `ndarray` 后端每次 `reshape` 都整份复制数组
   （一次 matmul 复制 3 份，包括权重），GELU 走 `libm` 的标量 f64 `erf`。改为直接在 `f32`
   缓冲区上用 `gemm` 的 strided 视图计算，投影与注意力全程零复制、零转置。
2. **完全无 padding**。batch 以 packed 形式（`ids` + `offsets`）送入模型，注意力按每条序列
   自身长度计算；padding 浪费由 ~1–7% 降为 0，也不再需要构造 attention bias 张量。
3. **缓冲区复用**。每个 worker 持有一份 scratch，稳态推理不再分配内存。
4. **可内联的 `exp` / `erf`**，使 GELU 与 softmax 循环能自动向量化（此前每个元素一次动态链接调用）。
5. **`mimalloc`** 作为全局分配器；优化前 profile 中 `malloc`/`madvise` 的样本数是 `sgemm` 的十倍以上。
6. **动态最长优先调度**，消除尾部负载不均。

此外 Burn 路径本身也修好了三处 feature/实现问题：`burn` 以 `default-features = false` 引入时，
`burn-ndarray` 的 `simd` 和 `burn-wgpu` 的 `fusion` / `autotune` 都被静默丢弃（后者是 GPU 提速的主因）；
以及 `Linear` 在 rank-3 输入上会退化成「每个 batch 一次小 GEMM」（现统一压平成 `[tokens, features]`）。

优化后 CPU profile 中约 78% 的时间落在 gemm 的 FMA 微内核上，已接近该精度下的计算上限。

### GPU 侧的实测与边界

`wgpu` 做过的改动与效果：

| 改动 | 效果 |
|---|---|
| 启用 `fusion` + `autotune` | ~6.0k → ~11–12k，**唯一有量级效果的一项** |
| batching 默认值按后端区分（128/32768） | 之前用 CPU 的 24/1536 只能跑 ~6k |
| RoPE / 滑动窗口张量做设备端缓存 | 墙钟无可测变化，但去掉了每次 forward 的主机端克隆、上传与一个 33.5MB 的 bias 张量 |
| 继续加大 pack（512 / 1024 / 2048） | **变慢**（8.3k / 7.4k / 0.6k），128 已是最优点 |

没有继续深挖的原因：等长输入实验显示吞吐在 seq 16→250 之间基本持平（11–17k 字/s），
即 attention 的 `S²` 并非瓶颈；换算下来 GPU 只跑到 RX 580 峰值的约 6%，
剩余损耗在 cubecl 的 wgpu matmul 内核对 `K=256` 这种小规约维度、以及老 GCN 架构的适配上，
无法在本 crate 内解决。**这台机器上 GPU 仍慢于 CPU 引擎，默认走 `cpu` 是对的。**

> **GPU 首次运行较慢**：autotune 需要为每种 matmul 形状做基准测试，结果缓存在 `target/autotune`，
> 会被 `cargo clean` 清除。改变 `--batch-size` 会引入新形状并触发一次重新调优（实测可慢 5×）。
>
> `burn/metal`（把内核编译成 MSL 而非 WGSL）在测试用的 AMD/Metal 驱动上会 shader 验证失败
> （`SC compilation failure`），因此没有提供该 feature。

### 为什么保留 `burn-cpu`

它没有任何性能优势（比 `cpu` 慢约 3 倍），但它是 `ops.rs` / `rope.rs` / `loader.rs` 单测的测试后端，
也是 `tests/cpu_parity.rs` 里**唯一不依赖 GPU 就能校验手写 CPU kernel 的 oracle**。运行期零成本，故保留。

随 `cpu` 引擎改用 `gemm`，Accelerate/BLAS（原 `cpu-blas` feature）已无人使用，实测移除后
`cpu` 与 `burn-cpu` 速度均无变化，故已删除该依赖。

### Bug fix

局部滑动窗口原先的生效条件是 `seq > 2*radius + 1`，导致长度落在 `(radius+1, 2*radius+1]`
（本模型即 65–129 字）的句子**完全跳过** sliding-window mask，与 HuggingFace 不一致，
且结果会随 batch 组成变化。现修正为 `seq > radius + 1`；整本书 3487 行中有 28 行标签因此改变。

## Library

```rust
use modernbert_ner::load_pipeline_cpu;

let pipeline = load_pipeline_cpu("alnilam/ner_model", 256)?;
let result = pipeline.predict("艾莉同学、おはよう")?;
for e in result.entities {
    println!("{} {} {}-{}", e.label, e.text, e.start, e.end);
}
```

`load_pipeline_burn_cpu` / `load_pipeline_wgpu` 提供 Burn 后端的等价 API。

## Tests

```bash
cargo test -p modernbert-ner
```

`tests/cpu_parity.rs` 用真实 checkpoint 对拍 CPU 引擎与 Burn 参考实现，覆盖滑动窗口阈值
两侧的长度（63/64/65/66/100/129/130/200），并校验 batch 不变性。其余集成测试在
`alnilam/ner_model` 存在时加载真实权重，缺失时跳过。

## Benchmarking

```bash
crates/modernbert-ner/bench.sh <tag> <jobs>                    # 单次跑分 + 输出哈希
crates/modernbert-ner/measure.sh <jobs> <batch> <max_tokens> N # 重复取最好值
crates/modernbert-ner/sweep.sh                                 # batch × jobs 扫描
```
