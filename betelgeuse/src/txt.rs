use anyhow::{Context, Result};
use std::path::Path;

/// Read a TXT file and return non-empty trimmed lines
pub fn extract_txt_lines(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read TXT: {}", path.display()))?;

    let lines: Vec<String> = text
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    Ok(lines)
}
