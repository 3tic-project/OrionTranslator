# OrionTranslator

<p align="center">
  <img src="assets/logo.png" alt="OrionTranslator" width="180">
</p>

<p align="center">面向日文轻小说的 EPUB/TXT 中文翻译工具，提供桌面 GUI、人物术语表识别和断点续翻。</p>

<p align="center" width="100%">
  <video src="https://github.com/user-attachments/assets/c0e3d34c-a724-4c28-9aa4-98a389bbe824" width="80%" controls></video>
</p>

## 使用步骤

### 1. 安装并启动

发布包已经包含默认 NER 模型。macOS 安装 DMG 后启动 OrionTranslator；Windows 首次安装使用 `Full.zip`，解压后运行 `alnitak.exe`。

也可以从源码启动（需要 Rust stable）：

```bash
git clone https://github.com/3tic-project/OrionTranslator
cd OrionTranslator
./scripts/fetch_ner_model.sh
cargo run --release -p alnitak
```

### 2. 配置翻译模型

在界面顶部选择 DeepSeek、火山引擎或 Orion 预设，然后填写：

| 字段 | 填写内容 |
|------|----------|
| API URL | OpenAI-compatible API 的 Base URL |
| 模型名称 | 服务商提供的模型 ID |
| API Key | 对应服务的密钥 |

使用 Orion 本地模型时，可参考 [Orion-HYMT1.5-1.8B-SFT-v2601-GGUF](https://huggingface.co/3tic/Orion-HYMT1.5-1.8B-SFT-v2601-GGUF) 的部署说明。

### 3. 选择文件与术语表

点击“选择文件”，或把 `.epub` / `.txt` 文件拖入窗口。

已有 `*_glossary.json` 时可直接选择；没有时，在术语表识别区域选择推理方式并点击“生成术语表”：

| 模式 | 说明 |
|------|------|
| CPU | 兼容性最好，无需独立显卡 |
| GPU | 使用 Metal、Vulkan 或 DX12 加速；不可用时回退 CPU |
| Auto | 首次用当前文档做一次短基准，自动选择更快的后端；同一会话会复用结果 |

默认人物识别模型为 [3tic/Orion-NER-30M-v1](https://huggingface.co/3tic/Orion-NER-30M-v1)。

### 4. 开始翻译

点击“开始翻译”。完成后可点击“打开输出”定位结果文件。

常见输出包括：

| 文件 | 说明 |
|------|------|
| `*.ja-zh[model].epub` | 日中双语对照版 |
| `*.zh[model].epub` | 纯中文替换版 |
| `*_translation_data.json` | 翻译进度与断点数据 |
| `*_error_report.json` | 翻译错误报告 |
| `*_glossary.json` | 人物术语表 |
| `*_glossary.generation-report.json` | 术语生成报告 |
| `*_glossary.ruby-candidates.json` | Ruby 读音候选审核数据 |

再次翻译同一文件时，程序会读取有效的翻译数据并跳过已经完成的内容。

## 必要说明

- 调用在线 LLM 会把待翻译文本发送给所选服务商，并可能产生 API 费用。
- API 配置保存在本机。密钥目前是本地混淆存储，不应当作系统密钥库级别的加密。
- 首次使用 Windows `Update.zip` 前，必须已经有完整版中的 `ner_model` 目录。

## License

Apache License 2.0
