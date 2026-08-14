use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::TranslationMode;
use crate::io_utils::atomic_write_with;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxtBlock {
    /// Stable identity for recovery/protocol mapping. Empty only in legacy JSON.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub unit_id: String,
    /// Versioned hash of the exact extracted source text. Empty only in legacy JSON.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source_sha256: String,
    pub index: usize,
    pub src_text: String,
    pub dst_text: Option<String>,
}

/// Read a TXT file and create translation data (each non-empty line is a unit)
pub fn read_txt_data(input_path: &Path) -> Result<Vec<TxtBlock>> {
    let text = std::fs::read_to_string(input_path)
        .with_context(|| format!("Failed to read TXT: {}", input_path.display()))?;

    let mut data = Vec::new();
    let mut source_occurrences: HashMap<String, usize> = HashMap::new();
    for line in text.lines() {
        let stripped = line.trim();
        if !stripped.is_empty() {
            let occurrence = source_occurrences.entry(stripped.to_string()).or_default();
            data.push(TxtBlock {
                unit_id: crate::unit_identity::unit_id("txt", "document", stripped, *occurrence),
                source_sha256: crate::unit_identity::source_sha256(stripped),
                index: data.len(),
                src_text: stripped.to_string(),
                dst_text: None,
            });
            *occurrence += 1;
        }
    }

    Ok(data)
}

/// Write translation results to TXT file
pub fn write_txt_output(
    data: &[TxtBlock],
    output_path: &Path,
    mode: TranslationMode,
) -> Result<()> {
    atomic_write_with(output_path, |file| {
        for item in data {
            let dst = item.dst_text.as_deref().unwrap_or(&item.src_text);
            match mode {
                TranslationMode::Bilingual => {
                    writeln!(file, "{}", item.src_text)?;
                    writeln!(file, "{}", dst)?;
                    writeln!(file)?;
                }
                TranslationMode::Replace => {
                    writeln!(file, "{}", dst)?;
                }
            }
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn read_txt_assigns_stable_identity_and_distinguishes_duplicates() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "orion_txt_identity_{}_{}.txt",
            std::process::id(),
            nonce
        ));
        std::fs::write(&path, "同じ行\n別の行\n同じ行\n").unwrap();

        let data = read_txt_data(&path).unwrap();

        assert_eq!(data.len(), 3);
        assert_ne!(data[0].unit_id, data[2].unit_id);
        assert!(data
            .iter()
            .all(|block| crate::unit_identity::source_hash_matches(
                &block.src_text,
                &block.source_sha256
            )));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn legacy_txt_json_remains_export_compatible() {
        let block: TxtBlock =
            serde_json::from_str(r#"{"index":0,"src_text":"原文","dst_text":"译文"}"#).unwrap();

        assert!(block.unit_id.is_empty());
        assert!(block.source_sha256.is_empty());
        assert_eq!(block.dst_text.as_deref(), Some("译文"));
    }
}
