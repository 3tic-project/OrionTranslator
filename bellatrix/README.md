# Bellatrix

嵌入式 NER + 术语表生成库。底层使用 `modernbert-ner` 在进程内运行 ModernBERT-JA Token Classification 模型，无需启动 HTTP 服务。

## 功能

- **嵌入式推理**：专用 CPU 引擎或 WGPU 后端
- **自动选后端**：从当前文档均匀抽样最多 400 行，分别完整预热、测速并使用更快的后端；GPU 不可用时回退 CPU
- **优化调度**：按长度打包；CPU 默认 24 行 / 1536 token 并使用全部逻辑核动态调度，GPU 默认 128 行 / 32768 token
- **人物检测**：从日语文本中识别人名实体，按出现频次聚合
- **术语表生成**：NER 识别 → 边界清洗 → 分簇翻译 → 关联分组总审校 → 确定性协议校验
- **进度回调**：`GlossaryProgressCallback` 支持实时进度上报

LLM 总审校每批最多 24 个关联条目，并保证同一人物的短名、全名、昵称和带敬称形式不被拆到不同请求。缺项、重复 `src`、非法译名或请求失败时，该批保留初步结果并记录到 `*.generation-report.json`；报告 schema v2 同时保存所有改名/拒绝决定。

## 结构

```
bellatrix/src/
├── lib.rs          # 公共 API、CPU/GPU/Auto 选择和 Auto 基准
├── detector.rs     # 人物检测（批量 NER + 实体聚合 + 上下文收集）
└── llm.rs          # LLM 人名翻译（聚类 + 性别推断 + 后缀映射）
```

## 公共 API

```rust
use bellatrix::{GlossaryConfig, NerBackend, generate_glossary};

let config = GlossaryConfig {
    lines: text_lines,
    ruby_annotations, // TXT 可传 Vec::new()
    model_dir: "ner_model".to_string(),
    ner_batch_size: 0, // 使用后端优化默认值
    ner_backend: NerBackend::Auto,
    llm_url: "https://api.deepseek.com".to_string(),
    llm_model: "deepseek-v4-flash".to_string(),
    // ...
};

let output_path = generate_glossary(config, progress_callback).await?;
```

EPUB 有 ruby 证据时会额外生成 `*.ruby-candidates.json`。候选按 `confirmed_existing`、`review_required`、`high_ambiguity_review`、`conflict` 等状态输出；未审核候选不会自动变成 `src -> dst` 强制术语。

## 模型文件要求

```
ner_model/
├── model.safetensors   # ModernBERT-JA 权重
├── config.json         # HuggingFace 配置
└── tokenizer.json      # HuggingFace tokenizer
```

## 编译特性

| Feature | 说明 |
|---------|------|
| `wgpu`（默认） | 启用 GPU 推理；专用 CPU 引擎始终可用 |
| `ndarray` | 启用 Burn CPU 参考后端（生产流程不使用） |

## 被依赖

- **alnilam**（翻译管线 CLI 的 `glossary` 子命令）
- **alnitak**（GUI 的术语表生成功能）
