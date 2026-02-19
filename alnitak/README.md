# Alnitak

基于 GPUI（Zed 编辑器 UI 框架）的桌面翻译应用。

## 功能

- **文件拖拽**：支持 EPUB/TXT/JSON 文件直接拖入
- **模型预设**：DeepSeek / Orion-HYMT / Orion-Qwen 一键切换
- **参数调节**：批大小、并发数、上下文行数、温度、Top-P、Top-K
- **实时进度**：热力图可视化翻译进度 + 统计仪表盘
- **术语表生成**：集成 NER 模型，一键生成人名术语表
- **断点续翻**：自动检测已有翻译数据，支持中断恢复
- **缓存管理**：一键清理中间文件
- **双输出**：日中双语对照 + 纯中文替换

## 结构

```
alnitak/src/
├── main.rs       # GPUI 应用入口 + Render 实现
├── app.rs        # OrionApp 状态结构 + 初始化
├── types.rs      # 状态枚举（翻译/术语表）+ 模型预设
├── ui.rs         # 界面布局（文件/配置/进度/日志 四区域）
├── handlers.rs   # 事件处理（翻译/取消/测试/术语表/缓存）
└── utils.rs      # 工具函数（日志/时间格式/路径计算）
```

## 运行

```bash
cd alnitak
cargo run --release
```

## 技术栈

- **UI 框架**：GPUI（GPU 渲染，Zed 编辑器同款）
- **异步桥接**：smol channel + cx.spawn（GPUI 主线程）↔ tokio（后台翻译）
- **翻译引擎**：alnilam（翻译管线库）
- **NER 模型**：bellatrix（嵌入式推理）
- **文本提取**：betelgeuse（EPUB/TXT）

## 开发文档

- [DEV.md](DEV.md) — 架构设计、异步模式、调试技巧
- [CODING_STANDARDS.md](CODING_STANDARDS.md) — 代码规范
