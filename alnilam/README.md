# Alnilam

EPUB/TXT 日译中翻译管线核心，同时作为 CLI 工具和库使用。

## 功能

- **EPUB 翻译**：解析 → 分批 LLM 翻译 → 质量检查 → 自动修复 → 重试 → EPUB 回写
- **TXT 翻译**：逐行分批翻译 + JSONL 数据保存
- **双输出**：日中双语对照 + 纯中文替换
- **断点续翻**：自动保存工作区快照；单元携带稳定 `unit_id + source_sha256`，恢复按身份映射且对重复/篡改快照整份拒收；只重发未完成单元
- **批次契约**：保留模型兼容的数字 JSONL key，同时用顺序敏感 `batch_revision + position→UnitId` 约束本地响应提交，跨批对象不匹配时整批拒收
- **上下文感知**：规则化场景检测（对话/叙述/标题/场景切换）+ 智能上下文选择
- **质量保障**：区分未译假名正文与受保护的谜题/署名 token；人工确认术语、数值、占位符和 JSON 映射泄漏为重试后也不会关闭的硬检查；允许 `12↔十二`、`17時↔下午五点` 等等值本地化；通过项仅做安全引号/标点正规化，失败项才激进修复并复检
- **Provider 重试**：按 HTTP/transport/protocol 强类型分类；401/403/422 等永久错误立即停止，408/425/429/5xx 与瞬时传输错误在明确预算内重试，支持秒数型 `Retry-After`
- **术语表生成/审核**：内嵌 NER（via bellatrix）；双通道抽取 ruby base/reading，显式报告 resolved/unresolved/rejected 结果，并生成可持久确认/拒绝的别名审核清单
- **术语匹配**：带假名/ASCII 边界和 leftmost-longest 重叠解析，避免 `アイ` 命中 `アイテム`
- **EPUB 保真**：未改章节保留 ZIP 原始 XHTML；目录跨节点时在原链接内追加中文节点；Replace 遇到复杂内联树时隐藏并保留源节点、另显示中文节点，避免清空 ruby/链接内容
- **事务输出**：EPUB/TXT/快照原子写入，拒绝输入输出同路径；EPUB 重导出对 file/block/source/UnitId 身份做全量硬校验，拒绝静默缺段
- **格式修复**：纵书→横书、RTL→LTR、SVG 图片简化（可选，默认仍受 CLI `--no-fix` 控制）

## 结构

```
alnilam/
├── build.rs                    # embed-rules 特性：编译时嵌入上下文规则 JSON
├── common_prompt.txt           # 通用模型翻译提示词模板
├── rules/
│   └── ja2zh_context_rules.json  # 上下文检测规则（770 条）
└── src/
    ├── main.rs                 # CLI 入口（translate / export / glossary / glossary-audit / glossary-review / quality-audit）
    ├── lib.rs                  # 库导出
    ├── config.rs               # PipelineConfig 配置 + 默认值
    ├── pipeline.rs             # EPUB/TXT 翻译编排（并发批处理 + 重试）
    ├── txt.rs                  # TXT 读写
    ├── llm/
    │   ├── client.rs           # LLM 客户端（重试 + 限流 + Orion/通用双模式）
    │   ├── prompt.rs           # 提示词构建（JSONL / 模板）
    │   ├── parser.rs           # JSONL 响应解析
    │   └── glossary.rs         # 术语表加载与格式化
    ├── checker/
    │   ├── response_checker.rs # 翻译质量检测
    │   ├── auto_fixer.rs       # 自动修复（标点/引号/假名）
    │   └── types.rs            # ErrorType、CheckResult、ErrorRecord
    ├── context/
    │   ├── detector.rs         # 规则化上下文检测（Trie + 正则）
    │   ├── selector.rs         # 上下文选择策略（特征评分 + 需求评估）
    │   ├── trie.rs             # Trie 匹配器
    │   └── types.rs            # LineType 枚举
    └── epub/
        ├── handler.rs          # EPUB 加载/解析/注入/回写
        └── format_fixer.rs     # CSS/OPF/SVG 格式修复
```

## CLI 使用

```bash
# EPUB 翻译（Orion 模型）
alnilam novel.epub

# 通用模型 + 术语表
alnilam novel.epub \
  --llm-url "https://api.deepseek.com" \
  --model "deepseek-v4-flash" \
  --api-key "sk-xxx" \
  --glossary-path glossary.json

# TXT 翻译
alnilam novel.txt -m replace -w 4

# 术语表生成（内嵌 NER）
alnilam glossary novel.epub --llm-key "sk-xxx"

# 使用已有翻译数据离线重导出（默认不启用有损格式修复）
alnilam export novel.epub \
  --translation-data novel_work/translation_data.json \
  --output novel.reexport.epub

# 离线审查 ruby 读音、平片假名复现与术语冲突
alnilam glossary-audit novel.epub \
  --glossary-path novel_glossary.json

# 人工确认低风险发音 alias（candidate_id 来自审核清单）
alnilam glossary-review novel_glossary.ruby-candidates.json \
  --candidate-id ruby-v1:... \
  --classification phonetic_reading \
  --decision confirmed \
  --target 白地野音

# 离线扫描历史翻译；默认把全部 v1 术语缺失作为诊断
alnilam quality-audit novel_work/translation_data.json \
  --glossary-path novel_glossary.json

# 复现在线硬门：只检查显式 hard 术语和硬质量项
alnilam quality-audit novel_work/translation_data.json \
  --glossary-path novel_glossary.json \
  --hard-only \
  --hard-terms-only
```

术语生成同时写出 `*_glossary.generation-report.json`，其中区分成功解析、模型/协议未解决和明确拒绝的实体 cluster；不会再把空 `choices` 或单个 cluster 失败静默丢掉。

`glossary-audit` 输出 schema v2 `*.ruby-candidates.json`。候选包含稳定 ID、证据、保守分类建议、人工分类、`pending|confirmed|rejected` 决定和 revision。机器建议本身不会启用 alias；只有人工确认为 `phonetic_reading`、`orthographic_alias` 或 `nickname_cue` 且具有目标译名的候选，才会把平/片假名变体加入 prompt 和 `TERM_MISSING` 硬检查。语义 ruby、普通注音、双关 token、待审核和已拒绝候选始终不会进入翻译约束；同一 surface 多译会在翻译前硬失败。

旧 v1 扁平术语缺少姓/名/昵称 render policy，因此继续作为 prompt 指引，但默认不进入在线硬 QA，避免把 `真里→笔名真里` 一类短称强制扩成全名。需要强制的手工 v1 条目可在 `info` 中加入 `enforcement=hard`。`quality-audit` 默认报告所有 v1 术语诊断；`--hard-terms-only` 只复现在线显式硬约束。报告会保存所有问题单元及同一单元的多重发现，不会让术语错误遮住数字或占位符错误。

`replace` 对纯文本块直接替换；对 ruby、链接、媒体或跨文本节点块，保留原 inner tree 到 `orion-source-hidden`（内联 `display:none !important`），并以 `orion-replace-translation` 显示中文。这是单语视觉输出的止损策略，不代表已完成所有阅读器的可访问性/显示兼容验证；发布前仍应执行阅读器冒烟测试。

## 编译特性

| Feature | 说明 |
|---------|------|
| `embed-rules`（默认） | 将上下文规则 JSON 编译进二进制 |

## 作为库使用

```rust
use alnilam::{config::PipelineConfig, pipeline};

let config = PipelineConfig { /* ... */ };
pipeline::translate_epub(&config, progress_cb, cancel_flag).await?;
```

## 依赖

- **bellatrix**：NER 推理 + 术语表生成
- **betelgeuse**：EPUB/TXT 文本提取
