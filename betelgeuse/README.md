# Betelgeuse

EPUB/TXT 文本提取库，为 OrionTranslator 生态提供统一的文本读取接口。

## 功能

- **EPUB 解析**：按 Spine 顺序提取 EPUB 中所有块级元素（`<p>`, `<h1>`–`<h6>`, `<li>`, `<div>` 等）的文本
- **Ruby 去除**：自动剥离 `<rt>` 注音标签，仅保留主体文字
- **TXT 读取**：读取纯文本文件并返回非空行列表
- **XHTML 兼容**：自动修复 XHTML 自闭合标签以适配 html5ever 解析器

## 结构

```
betelgeuse/src/
├── lib.rs      # 公共 API：extract_epub_lines, extract_txt_lines
├── epub.rs     # EPUB 解析（ZIP → OPF → Spine → HTML → 文本行）
└── txt.rs      # TXT 文件按行读取
```

## 核心流程（EPUB）

1. **ZIP 读取** → 将 EPUB 文件全部条目读入内存
2. **OPF 定位** → 从 `META-INF/container.xml` 获取 OPF 路径
3. **Manifest + Spine 解析** → 正则提取 `<item>` 和 `<itemref>`，按 Spine 排序
4. **XHTML 预处理** → 修复自闭合非空标签、规范化 void 标签
5. **文本提取** → 遍历块级元素叶子节点，跳过 `<rt>` 注音，收集纯文本

## 公共 API

```rust
use betelgeuse::{extract_epub_lines, extract_txt_lines};

let lines: Vec<String> = extract_epub_lines(Path::new("novel.epub"))?;
let lines: Vec<String> = extract_txt_lines(Path::new("novel.txt"))?;
```

## 被依赖

- **alnilam**（翻译管线 CLI）
- **alnitak**（翻译 GUI）
- **mintaka**（NER CLI 工具）
