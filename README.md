# OrionTranslator

<p align="center">
  <img src="assets/logo.png" alt="OrionTranslator" width="180">
</p>

<p align="center">日译中轻小说翻译工具，集成 NER 人物识别、LLM 翻译、术语表生成、EPUB/TXT 格式处理与桌面 GUI，全部使用Rust代码编写。</p>

<p align="center">
  <video src="assets/demo.mp4" width="860" controls muted></video>
</p>

## 快速开始

### GUI（推荐）

**1. 构建并启动**

构建主程序：
```bash
git clone https://github.com/3tic-project/OrionTranslator
cd OrionTranslator
cargo build --release -p alnitak
./target/release/alnitak
```
从 https://huggingface.co/3tic/Orion-NER-110M-v1 下载 NER 模型文件，在主程序同目录下创建 `ner_model` 文件夹并放入模型文件。

最后的结构应如下所示：

```
├── alnitak                 # GUI 可执行文件
├── ner_model/              # NER 模型文件夹
│   ├── model.safetensors
│   ├── config.json
│   ├── vocab.txt
│   ├── system.dic.zst 
│   └── ……     
```

然后启动 GUI 程序 （`./alnitak`）。

**2. 配置 LLM**

在界面顶部填写以下信息，内置了三个预设，默认为deepseek-chat：

| 字段 | 示例值 | 说明 |
|------|--------|------|
| API URL | `https://api.deepseek.com` | OpenAI-compatible 接口地址 |
| 模型名称 | `deepseek-chat` | 填入 `orion` 系列名称可启用专用格式 |
| API Key | `sk-xxx` | 对应服务的密钥 |


对于Orion-HYMT、Orion_Qwen等专用模型，请参考 https://huggingface.co/3tic/Orion-HYMT1.5-1.8B-SFT-v2601-GGUF 获取本地部署的方式。

**3. 选择文件**

点击 **选择文件** 按钮或直接将 `.epub` / `.txt` 文件拖入窗口。

**4. 加载术语表（可选）**

若已有 `*_glossary.json`，可在「术语表路径」填入文件路径以提升人名一致性。
若无术语表，可先点击 **生成术语表** 自动识别并翻译人名，完成后路径会自动填入。（需要GPU支持vulkan/metal）

**5. 开始翻译**

点击 **开始翻译**，进度热力图实时反映各段落翻译状态。翻译完成后点击 **打开输出** 直接定位到产物文件。

**输出文件**（与输入文件同目录）：

```
novel.ja-zh[deepseek-chat].epub   # 日中双语对照
novel.zh[deepseek-chat].epub      # 纯中文替换
novel_translation_data.json       # 翻译数据（支持断点续翻）
novel_error_report.json           # 错误报告
```

> **断点续翻**：中途中止后，下次翻译同一文件时若检测到 `*_translation_data.json`，已完成的段落会自动跳过。

### 关于本地模型

本项目针对轻小说翻译场景开发了精简的专用模型格式，详见 https://huggingface.co/3tic ，现在已经发布了测试模型 https://huggingface.co/3tic/Orion-HYMT1.5-7B-SFT-v2601 和 https://huggingface.co/3tic/Orion-Qwen3-1.7B-SFT-v2601 ，目前是概念验证阶段，欢迎测试反馈，不建议直接使用，后续将训练更完善的版本。

我们也针对上述模型开发了优化的推理引擎 https://github.com/3tic-project/HunyuanMT-Crane ，支持 CPU/Metal/CUDA 多种后端，在单卡4090上最高吞吐量可达2500 tokens/s，一分钟即可翻译完一本书，正在研发测试中，也欢迎参与测试和反馈。

## 架构

```
┌───────────────────────────────────────────────────────────────────┐
│                          用户界面层                                │
│  ┌─────────────────────┐     ┌───────────────────────────────┐   │
│  │  alnitak (GUI)      │     │  alnilam (CLI)                │   │
│  │  GPUI 桌面应用       │     │  翻译 / 术语表 子命令            │   │
│  └──────────┬──────────┘     └───────────────┬───────────────┘   │
│             │                                │                    │
│             ▼                                ▼                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                    alnilam (核心库)                           │ │
│  │  翻译管线 · LLM Client · Prompt · Parser · Checker            │ │
│  │  上下文检测 · EPUB 读写 · 格式修复 · TXT 处理                    │ │
│  └──────────┬──────────────────────────────────┬───────────────┘ │
│             │                                  │                  │
│             ▼                                  ▼                  │
│  ┌──────────────────────┐         ┌────────────────────────────┐ │
│  │  bellatrix           │         │  betelgeuse                │ │
│  │  嵌入式 NER 推理       │         │  EPUB/TXT 文本提取          │ │
│  │    术语表生成          │         │  (Spine 排序 / Ruby 去除)   │ │
│  └──────────────────────┘         └────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘

  独立部署（可选）
  ┌──────────────────┐
  │  rigel           │
  │  HTTP NER 服务    │
  └────────┬─────────┘
           │ HTTP API
  ┌────────┴─────────┐
  │  mintaka         │
  │  CLI NER 工具     │
  └──────────────────┘
```

### 数据流

```
输入 EPUB/TXT
      │
      ├──→ [betelgeuse] 文本提取
      │         │
      │         ▼
      │    [bellatrix] NER 人物识别 → LLM 人名翻译 → 术语表 JSON
      │                                                │
      ▼                                                ▼
 [alnilam 翻译管线]
      │  1. 加载术语表 + 上下文规则
      │  2. 分批调用 LLM（Orion / 通用模型）
      │  3. 质量检查 + 自动修复 + 重试
      │  4. EPUB 注入译文 / TXT 写出
      ▼
输出 翻译后 EPUB/TXT + 翻译数据 JSON + 错误报告 JSON
```

## 项目结构

```
OrionTranslator/
├── Cargo.toml              # Workspace 根配置
├── common_prompt.txt        # 通用模型翻译提示词模板
│
├── betelgeuse/              # EPUB/TXT 文本提取库
│   └── src/{lib,epub,txt}.rs
│
├── bellatrix/               # 嵌入式 NER 推理 + 术语表生成库
│   └── src/{lib,model,embedding,loader,tokenizer,ner,detector,llm}.rs
│
├── rigel/                   # HTTP NER 推理服务（独立部署）
│   └── src/{main,api,batcher,model,embedding,loader,tokenizer,ner}.rs
│
├── mintaka/                 # CLI NER 工具（调用 rigel）
│   └── src/{main,ner_client,detector,llm,analysis}.rs
│
├── alnilam/                 # 翻译管线核心（CLI + 库）
│   ├── build.rs             # embed-rules 构建脚本
│   ├── common_prompt.txt    # 通用模型提示词模板
│   ├── rules/               # 上下文检测规则
│   └── src/
│       ├── {main,lib,config,pipeline,txt}.rs
│       ├── llm/{client,prompt,parser,glossary}.rs
│       ├── checker/{response_checker,auto_fixer,types}.rs
│       ├── context/{detector,selector,trie,types}.rs
│       └── epub/{handler,format_fixer}.rs
│
├── alnitak/                 # 桌面 GUI（GPUI）
    ├── src/{main,app,types,ui,handlers,utils}.rs
    ├── DEV.md               # 开发文档
    └── CODING_STANDARDS.md  # 代码规范


```

## 模块说明

| 模块 | Crate 名 | 类型 | 说明 |
|------|----------|------|------|
| [betelgeuse](betelgeuse/) | `betelgeuse` | lib | EPUB/TXT 文本提取 |
| [bellatrix](bellatrix/) | `bellatrix` | lib | 嵌入式 NER 推理 + 术语表生成 |
| [rigel](rigel/) | `rigel` | bin | HTTP NER 服务（独立部署） |
| [mintaka](mintaka/) | `mintaka` | bin | CLI NER 工具 |
| [alnilam](alnilam/) | `alnilam` | lib+bin | 翻译管线核心 + CLI |
| [alnitak](alnitak/) | `alnitak` | bin | 桌面 GUI |

## 快速开始

### 环境要求

- **Rust** stable 1.85+（alnitak 需要 edition 2024）
- **GPU**：支持 wgpu 的显卡（Metal/Vulkan/DX12）
- **NER 模型文件**（术语表功能）：`model.safetensors`、`config.json`、`vocab.txt`、`system.dic.zst`
- **LLM API**：任何 OpenAI-compatible Chat Completions 接口

### 构建

```bash
git clone <repo-url>
cd OrionTranslator

# 构建所有模块
cargo build --release

# 仅构建 CLI
cargo build --release -p alnilam

# 仅构建 GUI
cargo build --release -p alnitak

# 仅构建 NER 后端
cargo build --release -p rigel
```

构建产物位于 `target/release/`：`alnilam`（CLI）、`alnitak`（GUI）、`rigel`、`mintaka`。

### 编译特性

| Crate | Feature | 说明 |
|-------|---------|------|
| alnilam | `embed-rules`（默认） | 将上下文规则 JSON 编译进二进制 |
| rigel | `wgpu`（默认） | GPU 推理 |
| rigel | `ndarray` | CPU 推理 |
| rigel | `cuda` | CUDA 推理 |
| bellatrix | `wgpu`（默认） | GPU 推理 |
| bellatrix | `ndarray` | CPU 推理 |

## 使用方式

### 1. CLI 翻译

```bash
# Orion 本地模型
alnilam novel.epub

# 通用模型 + 术语表
alnilam novel.epub \
  --llm-url "https://api.deepseek.com" \
  --model "deepseek-chat" \
  --api-key "sk-xxx" \
  --glossary-path glossary.json

# TXT 翻译（纯替换模式）
alnilam novel.txt -m replace -w 4
```

### 2. 术语表生成

```bash
alnilam glossary novel.epub \
  --llm-key "sk-xxx" \
  --llm-model "deepseek-chat"
```

### 3. 桌面 GUI

```bash
cargo run --release -p alnitak
```

### 4. 独立 NER 服务 + CLI 工具

```bash
# 终端 1：启动 NER 后端
cargo run --release -p rigel

# 终端 2：使用 CLI 工具
export LLM_API_KEY="sk-xxx"
cargo run --release -p mintaka -- glossary novel.epub
```

## 模型支持

### Orion 专用模型

模型名称包含 `orion`（大小写不敏感）时自动识别，使用精简 JSONL 格式直接交互。

### 通用大语言模型

DeepSeek、Qwen、GPT 等模型使用翻译提示词模板，支持术语表注入、上下文规则检测。

## 输出文件

| 文件 | 说明 |
|------|------|
| `*.ja-zh[model].epub` | 日中双语对照 EPUB |
| `*.zh[model].epub` | 纯中文替换 EPUB |
| `*_translation_data.json` | 翻译数据（断点续翻用） |
| `*_error_report.json` | 翻译错误报告 |
| `*_glossary.json` | 术语表 |
| `*_characters.json` | NER 人物候选列表 |
| `*_output.json` | NER + LLM 术语条目 |

## 许可证

私有项目。
