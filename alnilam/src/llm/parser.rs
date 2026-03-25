use std::collections::HashMap;

use regex::Regex;

/// Parse JSONL response from LLM into {index -> translated_text}
pub fn parse_jsonl_response(response: &str, _expected_count: usize) -> HashMap<usize, String> {
    let mut results = HashMap::new();
    let response = strip_leading_thinking_content(response);

    for line in response.trim().lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Try standard JSON parse first
        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(map) = obj.as_object() {
                for (key, value) in map {
                    if let Ok(idx) = key.parse::<usize>() {
                        let text = match value {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        results.insert(idx, text);
                    }
                }
            }
            continue;
        }

        // Fallback: regex-based parsing for malformed JSON
        let re = re_jsonl_fallback();
        for caps in re.captures_iter(line) {
            if let (Some(idx_match), Some(val_match)) = (caps.get(1), caps.get(2)) {
                if let Ok(idx) = idx_match.as_str().parse::<usize>() {
                    let mut value = val_match.as_str().to_string();
                    value = value.replace("\\n", "\n");
                    value = value.replace("\\r", "\r");
                    value = value.replace("\\t", "\t");
                    value = value.replace("\\\"", "\"");
                    value = value.replace("\\\\", "\\");
                    results.insert(idx, value);
                }
            }
        }
    }

    results
}

fn strip_leading_thinking_content(response: &str) -> &str {
    let mut remaining = response.trim_start();

    loop {
        if let Some(rest) = remaining.strip_prefix("<think>") {
            if let Some(end) = rest.find("</think>") {
                remaining = rest[end + "</think>".len()..].trim_start();
                continue;
            }
        }

        if let Some(rest) = remaining.strip_prefix("<thinking>") {
            if let Some(end) = rest.find("</thinking>") {
                remaining = rest[end + "</thinking>".len()..].trim_start();
                continue;
            }
        }

        break;
    }

    remaining
}

fn re_jsonl_fallback() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"\{["\']?(\d+)["\']?\s*:\s*["\'](.+?)["\']\}"#).expect("jsonl fallback regex")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_normal_jsonl() {
        let response = r#"{"1":"你好"}
{"2":"谢谢"}"#;
        let results = parse_jsonl_response(response, 2);
        assert_eq!(results.get(&1).map(|s| s.as_str()), Some("你好"));
        assert_eq!(results.get(&2).map(|s| s.as_str()), Some("谢谢"));
    }

    #[test]
    fn test_parse_malformed_jsonl() {
        let response = r#"{"1":"你好"}
{2:"谢谢"}"#;
        let results = parse_jsonl_response(response, 2);
        assert_eq!(results.get(&1).map(|s| s.as_str()), Some("你好"));
    }

    #[test]
    fn test_parse_jsonl_with_thinking_block() {
        let response = r#"<think>
先分析一下
</think>
{"1":"你好"}"#;
        let results = parse_jsonl_response(response, 1);
        assert_eq!(results.get(&1).map(|s| s.as_str()), Some("你好"));
    }
}
