use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::TranslationMode;
use crate::io_utils::atomic_write_with;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxtBlock {
    pub index: usize,
    pub src_text: String,
    pub dst_text: Option<String>,
}

/// Read a TXT file and create translation data (each non-empty line is a unit)
pub fn read_txt_data(input_path: &Path) -> Result<Vec<TxtBlock>> {
    let text = std::fs::read_to_string(input_path)
        .with_context(|| format!("Failed to read TXT: {}", input_path.display()))?;

    let mut data = Vec::new();
    for line in text.lines() {
        let stripped = line.trim();
        if !stripped.is_empty() {
            data.push(TxtBlock {
                index: data.len(),
                src_text: stripped.to_string(),
                dst_text: None,
            });
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
