use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct GlossaryEntry {
    pub src: String,
    pub dst: String,
    pub info: String,
}

/// 从 JSON 文件加载术语表
pub fn load_glossary(path: &std::path::Path) -> Result<Vec<GlossaryEntry>> {
    let content = std::fs::read_to_string(path)?;
    let entries: Vec<GlossaryEntry> = serde_json::from_str(&content)?;
    Ok(entries)
}

/// 将术语表格式化为通用模型 prompt 中的文本格式
/// 输出格式：
/// src -> dst   #info
/// src -> dst   #info
///
/// 注意：过滤掉 dst 为空的条目（Orion模型生成的原始术语表）
pub fn format_glossary(entries: &[GlossaryEntry]) -> String {
    entries
        .iter()
        .filter(|e| !e.dst.is_empty())
        .map(|e| format!("{} -> {}   #{}", e.src, e.dst, e.info))
        .collect::<Vec<_>>()
        .join("\n")
}

/// 将术语表格式化为 Orion 模型 prompt 中的格式（与 SFT 训练数据一致）
/// 输出格式：
/// 术语表：
/// src→dst
/// src→dst
///
/// 返回 None 表示没有有效术语可用
pub fn format_glossary_for_orion(entries: &[GlossaryEntry]) -> Option<String> {
    let mut pairs: Vec<(&str, &str)> = entries
        .iter()
        .filter(|e| !e.dst.is_empty())
        .map(|e| (e.src.as_str(), e.dst.as_str()))
        .collect();

    if pairs.is_empty() {
        return None;
    }

    pairs.sort();

    let mut result = String::from("术语表：\n");
    for (src, dst) in &pairs {
        result.push_str(src);
        result.push('→');
        result.push_str(dst);
        result.push('\n');
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_glossary() {
        let entries = vec![
            GlossaryEntry {
                src: "由紀".to_string(),
                dst: "由纪".to_string(),
                info: "女性".to_string(),
            },
            GlossaryEntry {
                src: "セナ".to_string(),
                dst: "濑名".to_string(),
                info: "女性".to_string(),
            },
        ];
        let text = format_glossary(&entries);
        assert_eq!(text, "由紀 -> 由纪   #女性\nセナ -> 濑名   #女性");
    }

    #[test]
    fn test_format_glossary_empty() {
        let entries: Vec<GlossaryEntry> = vec![];
        let text = format_glossary(&entries);
        assert_eq!(text, "");
    }

    #[test]
    fn test_format_glossary_for_orion() {
        let entries = vec![
            GlossaryEntry {
                src: "グレン".to_string(),
                dst: "格伦".to_string(),
                info: "男性".to_string(),
            },
            GlossaryEntry {
                src: "ネメア".to_string(),
                dst: "涅米亚".to_string(),
                info: "地名".to_string(),
            },
        ];
        let result = format_glossary_for_orion(&entries).unwrap();
        assert!(result.starts_with("术语表：\n"));
        assert!(result.contains("グレン→格伦\n"));
        assert!(result.contains("ネメア→涅米亚\n"));
    }

    #[test]
    fn test_format_glossary_for_orion_empty_dst() {
        // Entries with empty dst should be filtered out
        let entries = vec![
            GlossaryEntry {
                src: "テスト".to_string(),
                dst: String::new(),
                info: String::new(),
            },
        ];
        assert!(format_glossary_for_orion(&entries).is_none());
    }

    #[test]
    fn test_format_glossary_for_orion_sorted() {
        let entries = vec![
            GlossaryEntry {
                src: "ネメア".to_string(),
                dst: "涅米亚".to_string(),
                info: String::new(),
            },
            GlossaryEntry {
                src: "グレン".to_string(),
                dst: "格伦".to_string(),
                info: String::new(),
            },
        ];
        let result = format_glossary_for_orion(&entries).unwrap();
        let lines: Vec<&str> = result.lines().collect();
        // Should be sorted: グレン before ネメア
        assert_eq!(lines[0], "术语表：");
        assert_eq!(lines[1], "グレン→格伦");
        assert_eq!(lines[2], "ネメア→涅米亚");
    }
}
