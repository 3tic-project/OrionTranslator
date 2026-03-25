# Alnilam

EPUB/TXT 日译中翻译管线核心，同时作为 CLI 工具和库使用。

## 功能

- **EPUB 翻译**：解析 → 分批 LLM 翻译 → 质量检查 → 自动修复 → 重试 → EPUB 回写
- **TXT 翻译**：逐行分批翻译 + JSONL 数据保存
- **双输出**：日中双语对照 + 纯中文替换
- **断点续翻**：自动保存 `_translation_data.json`，中断后恢复进度
- **上下文感知**：规则化场景检测（对话/叙述/标题/场景切换）+ 智能上下文选择
- **质量保障**：假名残留、韩文残留、长度异常、相似度检测、自动修复
- **术语表生成**：内嵌 NER 模型（via bellatrix），CLI `glossary` 子命令
- **格式修复**：纵书→横书、RTL→LTR、SVG 图片简化

## 结构

```
alnilam/
├── build.rs                    # embed-rules 特性：编译时嵌入上下文规则 JSON
├── common_prompt.txt           # 通用模型翻译提示词模板
├── rules/
│   └── ja2zh_context_rules.json  # 上下文检测规则（770 条）
└── src/
    ├── main.rs                 # CLI 入口（translate / glossary 子命令）
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
  --llm-url "https://api.deepseek.com/v1" \
  --model "deepseek-chat" \
  --api-key "sk-xxx" \
  --glossary-path glossary.json

# TXT 翻译
alnilam novel.txt -m replace -w 4

# 术语表生成（内嵌 NER）
alnilam glossary novel.epub --llm-key "sk-xxx"
```

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
