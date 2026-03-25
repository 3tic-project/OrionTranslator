/// Escape special characters for JSON string value
pub fn escape_json_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            _ => result.push(c),
        }
    }
    result
}

/// Build JSONL input format for batch translation
pub fn build_input_jsonl(texts: &[String]) -> String {
    let mut lines = Vec::with_capacity(texts.len());
    for (i, text) in texts.iter().enumerate() {
        let escaped = escape_json_string(text);
        lines.push(format!("{{\"{}\":\"{}\"}}", i + 1, escaped));
    }
    lines.join("\n")
}

/// 根据上文/术语表的有无选择对应的指令文本
///
/// 四种变体与 SFT 训练数据中的 prompt 格式一一对应：
///   - plain:            无上文 + 无术语表
///   - glossary:         无上文 + 有术语表
///   - context:          有上文 + 无术语表
///   - glossary_context: 有上文 + 有术语表
fn pick_instruction(has_context: bool, has_glossary: bool) -> &'static str {
    match (has_context, has_glossary) {
        (false, false) => "将以下文本翻译为简体中文，使用JSONLINE格式输出翻译结果，只需输出翻译结果，不要额外解释：\n",
        (false, true)  => "参考术语表中的译法，将以下文本翻译为简体中文，使用JSONLINE格式输出翻译结果，只需输出翻译结果：\n",
        (true,  false) => "参考上文信息，将以下文本翻译为简体中文，使用JSONLINE格式输出翻译结果，只需输出翻译结果：\n",
        (true,  true)  => "参考上文和术语表，将以下文本翻译为简体中文，使用JSONLINE格式输出翻译结果，只需输出翻译结果：\n",
    }
}

/// Build Orion model prompt with context and glossary for batch translation.
///
/// Prompt structure (matches SFT training format):
///   Layer 1 (optional): 上文
///   Layer 2 (optional): 术语表
///   Layer 3:            指令（根据上文/术语表的有无选择变体）
///   Layer 4:            待翻译 JSONL
pub fn build_prompt_with_context(
    texts: &[String],
    context: &[String],
    glossary: Option<&str>,
) -> String {
    let input_jsonl = build_input_jsonl(texts);
    let has_context = !context.is_empty();
    let has_glossary = glossary.is_some();

    let mut content = String::with_capacity(2048);

    // Layer 1: 上文（可选）
    if has_context {
        content.push_str(&context.join("\n"));
        content.push_str("\n\n");
    }

    // Layer 2: 术语表（可选）
    if let Some(g) = glossary {
        content.push_str(g);
        content.push('\n');
    }

    // Layer 3: 指令
    content.push_str(pick_instruction(has_context, has_glossary));

    // Layer 4: 待翻译 JSONL
    content.push_str(&input_jsonl);
    content.push('\n');

    content
}

/// Build Orion model prompt for single-line translation (retry path).
pub fn build_single_prompt_with_context(
    text: &str,
    context: &[String],
    glossary: Option<&str>,
) -> String {
    let escaped = escape_json_string(text);
    let input_jsonl = format!("{{\"1\":\"{}\"}}", escaped);
    let has_context = !context.is_empty();
    let has_glossary = glossary.is_some();

    let mut content = String::with_capacity(1024);

    // Layer 1: 上文
    if has_context {
        content.push_str(&context.join("\n"));
        content.push_str("\n\n");
    }

    // Layer 2: 术语表
    if let Some(g) = glossary {
        content.push_str(g);
        content.push('\n');
    }

    // Layer 3: 指令
    content.push_str(pick_instruction(has_context, has_glossary));

    // Layer 4: 待翻译 JSONL
    content.push_str(&input_jsonl);
    content.push('\n');

    content
}

/// 通用模型的 prompt 模板（编译期嵌入）
const COMMON_PROMPT_TEMPLATE: &str = include_str!("../../common_prompt.txt");

/// 为通用模型构建批量翻译 prompt
/// - texts: 待翻译文本数组
/// - context: 上下文行（由 detector 构建，和 Orion 一致）
/// - glossary_text: 已格式化的术语表文本（可为空字符串）
pub fn build_common_prompt_with_context(
    texts: &[String],
    context: &[String],
    glossary_text: &str,
) -> String {
    let input_jsonl = build_input_jsonl(texts);
    let context_text = if context.is_empty() {
        String::new()
    } else {
        context.join("\n")
    };

    COMMON_PROMPT_TEMPLATE
        .replace("{{context}}", &context_text)
        .replace("{{glossary}}", glossary_text)
        .replace("{\"1\": \"\"}\n{\"2\": \"\"}", &input_jsonl)
}

/// 为通用模型构建单行翻译 prompt（重试时使用）
pub fn build_common_single_prompt_with_context(
    text: &str,
    context: &[String],
    glossary_text: &str,
) -> String {
    let escaped = escape_json_string(text);
    let input_jsonl = format!("{{\"1\":\"{}\"}}", escaped);
    let context_text = if context.is_empty() {
        String::new()
    } else {
        context.join("\n")
    };

    COMMON_PROMPT_TEMPLATE
        .replace("{{context}}", &context_text)
        .replace("{{glossary}}", glossary_text)
        .replace("{\"1\": \"\"}\n{\"2\": \"\"}", &input_jsonl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_json() {
        assert_eq!(escape_json_string("hello"), "hello");
        assert_eq!(escape_json_string("he\"llo"), "he\\\"llo");
        assert_eq!(escape_json_string("line\nbreak"), "line\\nbreak");
    }

    #[test]
    fn test_build_jsonl() {
        let texts = vec!["こんにちは".to_string(), "ありがとう".to_string()];
        let jsonl = build_input_jsonl(&texts);
        assert!(jsonl.contains("{\"1\":\"こんにちは\"}"));
        assert!(jsonl.contains("{\"2\":\"ありがとう\"}"));
    }

    #[test]
    fn test_orion_prompt_plain() {
        let texts = vec!["テスト".to_string()];
        let context: Vec<String> = vec![];
        let prompt = build_prompt_with_context(&texts, &context, None);
        assert!(prompt.contains("将以下文本翻译为简体中文"));
        assert!(prompt.contains("不要额外解释"));
        assert!(prompt.contains("{\"1\":\"テスト\"}"));
        assert!(!prompt.contains("术语表"));
        assert!(!prompt.contains("参考上文"));
    }

    #[test]
    fn test_orion_prompt_with_glossary() {
        let texts = vec!["テスト".to_string()];
        let context: Vec<String> = vec![];
        let glossary = "术语表：\nグレン→格伦\n";
        let prompt = build_prompt_with_context(&texts, &context, Some(glossary));
        assert!(prompt.contains("术语表：\nグレン→格伦"));
        assert!(prompt.contains("参考术语表中的译法"));
        assert!(!prompt.contains("参考上文"));
    }

    #[test]
    fn test_orion_prompt_with_context() {
        let texts = vec!["テスト".to_string()];
        let context = vec!["前の文".to_string()];
        let prompt = build_prompt_with_context(&texts, &context, None);
        assert!(prompt.starts_with("前の文\n"));
        assert!(prompt.contains("参考上文信息"));
        assert!(!prompt.contains("术语表"));
    }

    #[test]
    fn test_orion_prompt_with_glossary_and_context() {
        let texts = vec!["テスト".to_string()];
        let context = vec!["前の文".to_string()];
        let glossary = "术语表：\nグレン→格伦\n";
        let prompt = build_prompt_with_context(&texts, &context, Some(glossary));
        assert!(prompt.starts_with("前の文\n"));
        assert!(prompt.contains("术语表：\nグレン→格伦"));
        assert!(prompt.contains("参考上文和术语表"));
    }
}
