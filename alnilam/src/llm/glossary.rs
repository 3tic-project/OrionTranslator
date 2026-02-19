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

/// 将术语表格式化为 prompt 中的文本格式
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
}
