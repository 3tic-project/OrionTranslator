use std::collections::HashMap;

use regex::Regex;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParseDiagnostics {
    pub missing_indices: Vec<usize>,
    pub duplicate_indices: Vec<usize>,
    pub out_of_range_indices: Vec<usize>,
    pub malformed_lines: Vec<String>,
}

impl ParseDiagnostics {
    pub fn has_issues(&self) -> bool {
        !self.missing_indices.is_empty()
            || !self.duplicate_indices.is_empty()
            || !self.out_of_range_indices.is_empty()
            || !self.malformed_lines.is_empty()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParsedJsonlResponse {
    pub translations: HashMap<usize, String>,
    pub diagnostics: ParseDiagnostics,
}

/// Parse JSONL response from LLM into {index -> translated_text}
pub fn parse_jsonl_response(response: &str, _expected_count: usize) -> HashMap<usize, String> {
    parse_jsonl_response_detailed(response, _expected_count).translations
}

/// Parse JSONL response from LLM and report structural issues.
pub fn parse_jsonl_response_detailed(response: &str, expected_count: usize) -> ParsedJsonlResponse {
    let mut results = HashMap::new();
    let mut diagnostics = ParseDiagnostics::default();
    let response = strip_leading_thinking_content(response);

    for line in response.trim().lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let mut parsed_any = false;

        // Try standard JSON parse first
        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(map) = obj.as_object() {
                for (key, value) in map {
                    if let Ok(idx) = key.parse::<usize>() {
                        parsed_any = true;
                        insert_translation(
                            &mut results,
                            &mut diagnostics,
                            expected_count,
                            idx,
                            match value {
                                serde_json::Value::String(s) => s.clone(),
                                other => other.to_string(),
                            },
                        );
                    }
                }
            }
            if !parsed_any {
                diagnostics.malformed_lines.push(line.to_string());
            }
            continue;
        }

        // Fallback: regex-based parsing for malformed JSON
        let re = re_jsonl_fallback();
        for caps in re.captures_iter(line) {
            if let (Some(idx_match), Some(val_match)) = (caps.get(1), caps.get(2)) {
                if let Ok(idx) = idx_match.as_str().parse::<usize>() {
                    parsed_any = true;
                    let mut value = val_match.as_str().to_string();
                    value = value.replace("\\n", "\n");
                    value = value.replace("\\r", "\r");
                    value = value.replace("\\t", "\t");
                    value = value.replace("\\\"", "\"");
                    value = value.replace("\\\\", "\\");
                    insert_translation(&mut results, &mut diagnostics, expected_count, idx, value);
                }
            }
        }
        if !parsed_any {
            diagnostics.malformed_lines.push(line.to_string());
        }
    }

    for idx in 1..=expected_count {
        if !results.contains_key(&idx) {
            diagnostics.missing_indices.push(idx);
        }
    }

    ParsedJsonlResponse {
        translations: results,
        diagnostics,
    }
}

fn insert_translation(
    results: &mut HashMap<usize, String>,
    diagnostics: &mut ParseDiagnostics,
    expected_count: usize,
    idx: usize,
    value: String,
) {
    if idx == 0 || idx > expected_count {
        diagnostics.out_of_range_indices.push(idx);
        return;
    }
    if results.insert(idx, value).is_some() {
        diagnostics.duplicate_indices.push(idx);
    }
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

    #[test]
    fn reports_parse_diagnostics() {
        let response = r#"{"1":"你好"}
{"1":"重复"}
{"3":"越界"}
not json"#;
        let parsed = parse_jsonl_response_detailed(response, 2);

        assert_eq!(
            parsed.translations.get(&1).map(|s| s.as_str()),
            Some("重复")
        );
        assert_eq!(parsed.diagnostics.missing_indices, vec![2]);
        assert_eq!(parsed.diagnostics.duplicate_indices, vec![1]);
        assert_eq!(parsed.diagnostics.out_of_range_indices, vec![3]);
        assert_eq!(parsed.diagnostics.malformed_lines, vec!["not json"]);
    }
}
