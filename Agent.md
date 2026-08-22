# OrionTranslator Agent Notes

本文档面向维护者和编码代理，适用于整个仓库。面向普通用户的 README 应只保留完成安装、配置、术语表生成和翻译所需的步骤；架构、实现约束、测试、CLI 维护命令及发布细节放在本文档或对应 crate 的 README 中。

## Workspace 与模块边界

| Crate | 职责 |
|-------|------|
| `betelgeuse` | EPUB/TXT 文本提取与结构解析 |
| `modernbert-ner` | ModernBERT-JA CPU/WGPU NER 推理实现与基准程序 |
| `bellatrix` | 内嵌 NER、术语候选聚合、LLM 术语生成与审核 |
| `alnilam` | 翻译核心库及 CLI；批处理、质量检查、恢复、EPUB/TXT 回写 |
| `alnitak` | GPUI 桌面应用；产品版本号来源 |
| `rigel` | 可独立部署的 HTTP NER 服务 |
| `mintaka` | 调用独立 NER 服务的 CLI 客户端 |

主要数据流：

```text
EPUB/TXT
  -> betelgeuse 提取文本
  -> bellatrix/modernbert-ner 识别人名
  -> LLM 聚合、翻译并审核术语
  -> alnilam 分批翻译、检查与重试
  -> EPUB/TXT 输出、恢复数据和错误报告
```

## 常用构建与运行命令

项目使用 Rust stable。默认 release profile 开启 `opt-level=3`、LTO 和单 codegen unit。

```bash
# 获取并校验默认 NER 模型
./scripts/fetch_ner_model.sh

# GUI
cargo run --release -p alnitak

# 翻译 CLI
cargo build --release -p alnilam

# 整个 workspace
cargo build --release
```

主要 feature：

- `alnilam/embed-rules`：默认开启，将上下文规则编译进二进制。
- `modernbert-ner`、`bellatrix`：默认使用 WGPU；`--no-default-features` 是 CPU-only ndarray 路径。
- `rigel/wgpu` 为默认 GPU 路径，另有 `ndarray` 和 `cuda`。

## 默认 NER 模型契约

- 仓库：`3tic/Orion-NER-30M-v1`
- 固定 revision：`6fe4a2a0563d9fe102e25c6c0f7b22677c383801`
- 配置与各文件 SHA-256：`scripts/ner_model.conf`
- 本地默认目录：`alnilam/ner_model`
- 必需文件：`model.safetensors`、`config.json`、`tokenizer.json`、模型 `README.md`

不要手工替换固定模型或绕过哈希校验。修改模型版本时，应同时更新 revision、所有哈希、加载测试和打包验证。模型权重受 `.gitignore` 排除，不应提交到 Git。

发布脚本会复用已验证的本地模型，否则通过 `hf` 或 `curl` 下载。验证现有目录：

```bash
./scripts/fetch_ner_model.sh --verify-only alnilam/ner_model
```

## NER 后端约束

GUI 和 `bellatrix::NerBackend` 的顺序均为 `CPU / GPU / Auto`，该顺序参与持久化映射，修改时必须同步迁移配置。

- CPU 使用 packed/ndarray 路径。
- GPU 使用 WGPU，并允许初始化失败时安全回退 CPU。
- Auto 只抽取文档中的短样本，分别测量可用后端，选择实际吞吐更高的一方。
- Auto 仅缓存具体的 CPU/GPU 结果；模型、batch 或会话变化时重新测试，不把 `Auto` 本身作为缓存结果。
- 调整 Auto 样本或预热次数时，应运行真实 checkpoint 测试并确认短文档与长文档都不会退化。

## 术语表与 LLM 聚合

术语生成不是单次无条件合并：先进行实体 cluster 预处理，再分批请求，最后执行跨 cluster 一致性审核。允许使用多个 LLM 请求换取完整性，但必须保留稳定键、来源实体和显式协议校验。

生成文件：

- `*_glossary.json`：可用于翻译 prompt 的术语表。
- `*_glossary.generation-report.json`：记录 `resolved`、`unresolved`、`rejected`，禁止静默丢弃空响应或失败 cluster。
- `*_glossary.ruby-candidates.json`：schema v2 的 ruby/假名 alias 证据与人工决定。

Ruby 候选的机器建议不能直接变成强制翻译约束。只有人工确认且分类为 `phonetic_reading`、`orthographic_alias` 或 `nickname_cue`，并具有目标译名，才可加入 prompt 与 `TERM_MISSING` 硬检查。语义 ruby、普通注音、双关、待审核和拒绝项不得启用；同一 surface 对应多个译名时应在翻译前失败。

旧 v1 扁平术语缺少 render policy，只作为 prompt 指引。只有在 `info` 中显式设置 `enforcement=hard` 的手工条目才能进入在线硬 QA。

## 翻译、恢复和质量约束

- 每个翻译单元使用稳定 `unit_id` 与 `source_sha256`。
- 恢复按身份映射，不依赖 JSON 数组位置；重复、未知或哈希不一致的新版快照整份拒绝。
- 模型请求继续使用 `1..N` 数字 JSONL key，同时维护顺序敏感的 `batch_revision` 和 position-to-UnitId 契约。
- 响应只有在 revision 与完整 UnitId 顺序都吻合时才能提交，避免并发响应写入错误批次。
- Provider 错误必须区分永久错误和瞬时错误。401/403/422 等永久错误立即停止；408/425/429/5xx、瞬时传输错误按有限预算重试，并尊重可解析的 `Retry-After`。
- 硬质量门包括显式 hard 术语、空译、数值、占位符、未译正文和 JSON 映射泄漏；同一单元允许报告多个问题。
- 数值检查需接受全/半角、阿拉伯与中日汉字数字、常见范围及 12/24 小时制的等值本地化。

## EPUB 输出约束

- 未修改章节尽量保留原 ZIP/XHTML 字节，避免无关的 HTML5 归一化。
- 离线重导出禁止输入与输出为同一路径，并在写出前完整校验 file、block、source、UnitId/hash。
- `replace` 遇到 ruby、链接、媒体或跨文本节点内容时，不得清空原内联树。当前策略是以 `display:none !important` 保存源树，并用带 `zh-CN` 标记的独立节点显示译文。
- 双语输出仍是跨阅读器兼容性更稳妥的默认路径；涉及回写结构的改动应执行真实阅读器冒烟测试。
- 竖排/RTL/SVG 修复可能有损，离线重导出默认不得自动开启。

## CLI 维护与审核

```bash
# 翻译
alnilam novel.epub \
  --llm-url "https://api.deepseek.com/v1" \
  --model "deepseek-v4-flash" \
  --api-key "sk-xxx" \
  --glossary-path novel_glossary.json

# 生成术语表
alnilam glossary novel.epub \
  --llm-key "sk-xxx" \
  --llm-model "deepseek-v4-flash"

# 从既有翻译数据安全重导出
alnilam export novel.epub \
  --translation-data novel_work/translation_data.json \
  --output novel.reexport.epub \
  --mode bilingual

# 生成 ruby/假名候选审核文件
alnilam glossary-audit novel.epub \
  --glossary-path novel_glossary.json

# 确认一个低风险 alias
alnilam glossary-review novel_glossary.ruby-candidates.json \
  --candidate-id ruby-v1:... \
  --classification phonetic_reading \
  --decision confirmed \
  --target 白地野音

# 完整离线质量诊断
alnilam quality-audit novel_work/translation_data.json \
  --glossary-path novel_glossary.json

# 只复现在线硬门
alnilam quality-audit novel_work/translation_data.json \
  --glossary-path novel_glossary.json \
  --hard-only --hard-terms-only
```

独立 NER 服务开发：

```bash
cargo run --release -p rigel
LLM_API_KEY="sk-xxx" cargo run --release -p mintaka -- glossary novel.epub
```

## 修改后的最小验证

根据改动范围选取测试；涉及公共类型、Cargo feature 或跨 crate 调用时运行 workspace check。

```bash
cargo fmt -p modernbert-ner -p bellatrix -p alnitak -- --check
cargo test -p modernbert-ner --no-default-features --lib
cargo test -p bellatrix --no-default-features --lib
cargo test -p alnilam --lib
cargo check -p alnitak
cargo check --workspace
git diff --check
```

说明：`alnilam` 仍有不属于当前改动的历史 rustfmt 差异，因此 CI 当前只对 `modernbert-ner`、`bellatrix`、`alnitak` 执行格式门禁。不要为了无关任务批量格式化整个 crate。

真实模型 CPU 合约测试需要先下载默认模型：

```bash
./scripts/fetch_ner_model.sh
cargo test -p modernbert-ner --no-default-features --test e2e_cpu
```

## CI、打包与版本发布

- `.github/workflows/ci.yml`：main push/PR 上执行 CPU 核心测试、macOS GUI check 和固定模型合约测试。
- `.github/workflows/package.yml`：手动运行或推送 `v*` tag 时构建 macOS DMG 与 Windows ZIP。
- `scripts/build_macos.sh`：生成包含默认模型的 `.app`/DMG。
- `scripts/build_windows.sh`：生成包含模型的 `Full.zip` 和仅程序的 `Update.zip`。
- Windows Full 包必须包含模型、配置、tokenizer 与 provenance；Update 包只适合覆盖已经安装完整模型的目录。

产品版本只以 `alnitak/Cargo.toml` 的 package version 为来源，修改后由 Cargo 同步 `Cargo.lock`。发布前至少运行格式检查、CPU 测试、workspace check、模型验证与对应平台打包检查，然后提交独立 release commit，创建注释 `vX.Y.Z` tag 并推送。

## 凭据与生成物

- `LLM_endpoint.json`、`.env*`、模型权重、测试书籍、`dist/` 和翻译产物不得提交。
- GUI 凭据在 Unix/macOS 上以 `0600` 原子保存，但内容只是本地 XOR 混淆，不是系统密钥库加密。
- 不要在测试日志、错误信息、fixture 或文档中写入真实 API key。
