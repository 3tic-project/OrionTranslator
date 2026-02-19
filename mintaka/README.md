# Mintaka

命令行 NER 工具。通过 HTTP 调用 rigel 进行人物识别，再使用 LLM 生成术语表。提供离线质量分析功能。

## 功能

- **人物识别**：批量 NER + 高频人名聚合（HTTP 模式，调用 rigel）
- **术语表生成**：NER 识别 → LLM 人名翻译 → JSON 输出
- **离线分析**：对已有的 `_output.json` / `_characters.json` 进行质量分析
- **批量分析**：遍历目录下所有术语表文件进行批量审计

## 结构

```
mintaka/src/
├── main.rs        # clap CLI 入口（run / glossary / analyze 等子命令）
├── ner_client.rs  # HTTP 客户端（调用 rigel 的 /ner 端点）
├── detector.rs    # 人物检测（HTTP 批量 NER + 实体聚合）
├── llm.rs         # LLM 人名翻译（与 bellatrix/llm.rs 同源）
└── analysis.rs    # 离线质量分析（假名残留/重复/聚类冲突检测）
```

## 子命令

```bash
# 人物识别 + 术语表生成
mintaka run novel.epub --llm-key "sk-xxx"

# 仅人物识别（跳过 LLM 翻译）
mintaka run novel.epub --skip-llm

# 术语表质量分析
mintaka analyze novel_output.json
mintaka analyze-chars novel_characters.json

# 批量分析目录
mintaka analyze-dir ./results/
mintaka analyze-chars-dir ./results/
```

## 前置依赖

- 需要 **rigel** 运行中（默认 `http://127.0.0.1:3000`）
- LLM 翻译需要 OpenAI-compatible API
