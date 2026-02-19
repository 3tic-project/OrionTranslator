# Bellatrix

嵌入式 NER 模型推理 + 术语表生成库。使用 Burn 框架在进程内直接运行 BERT Token Classification 模型，无需启动 HTTP 服务。

## 功能

- **嵌入式推理**：BERT NER 模型直接在进程内运行（wgpu / ndarray 后端）
- **人物检测**：从日语文本中识别人名实体，按出现频次聚合
- **术语表生成**：NER 识别 → 人名聚类 → LLM 翻译 → 术语表 JSON，一键完成
- **进度回调**：`GlossaryProgressCallback` 支持实时进度上报

## 结构

```
bellatrix/src/
├── lib.rs          # 公共 API：generate_glossary(), GlossaryConfig, load_ner_pipeline()
├── model.rs        # BERT 模型定义（Burn Module）
├── embedding.rs    # BERT 嵌入层
├── loader.rs       # SafeTensors → Burn 权重加载（candle-core）
├── tokenizer.rs    # 日语 BERT 字符级分词器（vibrato/MeCab）
├── ner.rs          # NER 推理管线（分词 → 推理 → BIO 解码）
├── detector.rs     # 人物检测（批量 NER + 实体聚合 + 上下文收集）
└── llm.rs          # LLM 人名翻译（聚类 + 性别推断 + 后缀映射）
```

## 公共 API

```rust
use bellatrix::{GlossaryConfig, generate_glossary, GlossaryProgressEvent};

let config = GlossaryConfig {
    lines: text_lines,
    model_dir: "ner_model".to_string(),
    llm_url: "https://api.deepseek.com".to_string(),
    llm_model: "deepseek-chat".to_string(),
    // ...
};

let output_path = generate_glossary(config, progress_callback).await?;
```

## 模型文件要求

```
ner_model/
├── model.safetensors   # BERT 权重
├── config.json         # HuggingFace 配置
├── vocab.txt           # 词表
└── system.dic.zst      # MeCab 词典（可选，提升分词精度）
```

## 编译特性

| Feature | 说明 |
|---------|------|
| `wgpu`（默认） | GPU 推理 |
| `ndarray` | CPU 推理 |

## 被依赖

- **alnilam**（翻译管线 CLI 的 `glossary` 子命令）
- **alnitak**（GUI 的术语表生成功能）
