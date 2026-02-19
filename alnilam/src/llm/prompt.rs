use crate::config::TARGET_LANG;

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

/// Build prompt with context for batch (multi-line) translation
pub fn build_prompt_with_context(texts: &[String], context: &[String]) -> String {
    let input_jsonl = build_input_jsonl(texts);

    if !context.is_empty() {
        let context_text = context.join("\n");
        format!(
            "{}\n参考上面的信息，把下面的文本翻译成{}，使用JSONLINE格式输出翻译结果，注意不需要翻译上文，也不要额外解释：\n{}\n",
            context_text, TARGET_LANG, input_jsonl
        )
    } else {
        format!(
            "将以下文本翻译为{}，使用JSONLINE格式输出翻译结果，注意只需要输出翻译后的结果，不要额外解释：\n\n{}",
            TARGET_LANG, input_jsonl
        )
    }
}

/// Build prompt for single-line translation (matches SFT training format)
pub fn build_single_prompt_with_context(text: &str, context: &[String]) -> String {
    if !context.is_empty() {
        let context_text = context.join("\n");
        format!(
            "{}\n参考上面的信息，把下面的文本翻译成{}，注意不需要翻译上文，也不要额外解释：\n{}\n",
            context_text, TARGET_LANG, text
        )
    } else {
        format!(
            "将以下文本翻译为{}，注意只需要输出翻译后的结果，不要额外解释：\n\n{}",
            TARGET_LANG, text
        )
    }
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
        .replace(
            "{\"1\": \"\"}\n{\"2\": \"\"}",
            &input_jsonl,
        )
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
        .replace(
            "{\"1\": \"\"}\n{\"2\": \"\"}",
            &input_jsonl,
        )
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
}
