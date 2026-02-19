# Rigel

独立的 HTTP NER 推理服务。基于 Burn 框架运行 BERT Token Classification 模型，通过 axum 提供 REST API。

## 功能

- **GPU 推理**：支持 wgpu（Metal/Vulkan/DX12）、ndarray（CPU）、CUDA 三种后端
- **动态批处理**：自动收集短时间内的请求合并推理，减少 GPU 调度开销
- **REST API**：`GET /health` 健康检查，`POST /ner` 实体识别
- **自动合并**：相邻同类实体自动合并，支持按类型过滤

## 结构

```
rigel/src/
├── main.rs       # axum 服务入口，按 feature 切换后端
├── api.rs        # REST 路由 + CORS + 请求/响应类型
├── batcher.rs    # 动态批处理（channel + 超时收集）
├── model.rs      # BERT 模型定义（Burn Module）
├── embedding.rs  # BERT 嵌入层（word + position + token_type）
├── loader.rs     # SafeTensors → Burn 权重加载
├── tokenizer.rs  # 日语 BERT 字符级分词器（vibrato/MeCab）
└── ner.rs        # NER 推理管线（分词 → 推理 → BIO 解码）
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_DIR` | `./model` | 模型文件目录 |
| `BIND_ADDR` | `0.0.0.0:3000` | 服务监听地址 |
| `MECAB_DICT_PATH` | `{MODEL_DIR}/system.dic.zst` | MeCab 词典路径 |

## 模型文件要求

```
model/
├── model.safetensors   # BERT 权重
├── config.json         # HuggingFace 配置
├── vocab.txt           # 词表
└── system.dic.zst      # MeCab 词典（可选）
```

## API

```bash
# 健康检查
curl http://127.0.0.1:3000/health

# NER 推理
curl -X POST http://127.0.0.1:3000/ner \
  -H "Content-Type: application/json" \
  -d '{"texts": ["田中太郎は東京に住んでいる。"], "combine_entities": true}'
```

## 编译特性

| Feature | 说明 |
|---------|------|
| `wgpu`（默认） | GPU 推理（Metal/Vulkan/DX12） |
| `ndarray` | CPU 推理 |
| `cuda` | CUDA GPU 推理 |

## 与 mintaka 配合

mintaka 通过 HTTP 调用 rigel 的 `/ner` 端点进行实体识别。启动 rigel 后即可使用 mintaka 的所有功能。
