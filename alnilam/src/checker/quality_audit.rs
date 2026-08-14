use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

use super::ResponseChecker;
use crate::epub::TranslationBlock;
use crate::io_utils::atomic_write;
use crate::llm::glossary;
use crate::txt::TxtBlock;

pub const QUALITY_AUDIT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct QualityAuditFinding {
    pub error_type: String,
    pub details: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct QualityAuditIssue {
    pub record_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_name: Option<String>,
    pub block_index: usize,
    pub source_text: String,
    pub translated_text: String,
    pub findings: Vec<QualityAuditFinding>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct QualityAuditSummary {
    pub total_units: usize,
    pub translated_units: usize,
    pub untranslated_units: usize,
    pub issue_units: usize,
    pub total_findings: usize,
    pub findings_by_type: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct QualityAuditReport {
    pub schema_version: u32,
    pub translation_data: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub glossary_path: Option<String>,
    pub source_language: String,
    pub target_language: String,
    pub include_soft_checks: bool,
    pub term_check_mode: String,
    pub glossary_entry_count: usize,
    pub limitations: Vec<String>,
    pub summary: QualityAuditSummary,
    pub issues: Vec<QualityAuditIssue>,
}

#[derive(Debug, Clone)]
struct AuditUnit {
    unit_id: Option<String>,
    file_name: Option<String>,
    block_index: usize,
    source_text: String,
    translated_text: String,
}

pub fn default_quality_audit_path(translation_data: &Path) -> PathBuf {
    let stem = translation_data
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("translation_data");
    translation_data.with_file_name(format!("{stem}.quality-report.json"))
}

pub fn audit_translation_data(
    translation_data: &Path,
    glossary_path: Option<&Path>,
    source_language: &str,
    target_language: &str,
    include_soft_checks: bool,
    hard_terms_only: bool,
) -> Result<QualityAuditReport> {
    let units = load_audit_units(translation_data)?;
    let loaded_glossary_entries = match glossary_path {
        Some(path) => glossary::load_glossary_with_confirmed_aliases(path)?,
        None => Vec::new(),
    };
    let glossary_entries = if hard_terms_only {
        glossary::hard_quality_constraints(&loaded_glossary_entries)
    } else {
        loaded_glossary_entries
    };
    let glossary_entry_count = glossary_entries.len();
    let checker = ResponseChecker::new(source_language, target_language, 0.80, usize::MAX)
        .with_glossary_entries(glossary_entries);

    Ok(build_report(
        translation_data,
        glossary_path,
        source_language,
        target_language,
        include_soft_checks,
        hard_terms_only,
        glossary_entry_count,
        units,
        &checker,
    ))
}

pub fn save_quality_audit_report(path: &Path, report: &QualityAuditReport) -> Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    atomic_write(path, json.as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn build_report(
    translation_data: &Path,
    glossary_path: Option<&Path>,
    source_language: &str,
    target_language: &str,
    include_soft_checks: bool,
    hard_terms_only: bool,
    glossary_entry_count: usize,
    units: Vec<AuditUnit>,
    checker: &ResponseChecker,
) -> QualityAuditReport {
    let mut summary = QualityAuditSummary {
        total_units: units.len(),
        ..QualityAuditSummary::default()
    };
    let mut issues = Vec::new();

    for (record_index, unit) in units.into_iter().enumerate() {
        if unit.translated_text.trim().is_empty() {
            summary.untranslated_units += 1;
        } else {
            summary.translated_units += 1;
        }

        let findings = checker.audit_line(
            &unit.source_text,
            &unit.translated_text,
            include_soft_checks,
        );
        if findings.is_empty() {
            continue;
        }

        let findings = findings
            .into_iter()
            .map(|finding| {
                let error_type = finding.error.to_string();
                *summary
                    .findings_by_type
                    .entry(error_type.clone())
                    .or_default() += 1;
                QualityAuditFinding {
                    error_type,
                    details: finding.details,
                }
            })
            .collect::<Vec<_>>();
        summary.total_findings += findings.len();
        summary.issue_units += 1;
        issues.push(QualityAuditIssue {
            record_index,
            unit_id: unit.unit_id,
            file_name: unit.file_name,
            block_index: unit.block_index,
            source_text: unit.source_text,
            translated_text: unit.translated_text,
            findings,
        });
    }

    let mut limitations = vec![
        "离线规则只能发现结构化异常和可疑项，不能证明译文语义正确。".to_string(),
        "软质量项是启发式信号，需要抽样或人工复核。".to_string(),
    ];
    if glossary_path.is_some() && !hard_terms_only {
        limitations.push(
            "TERM_MISSING 对 v1 扁平术语表使用单一目标字符串；姓/名/昵称的允许译法需待 Entity render policy 完成后再作为发布 Gate。"
                .to_string(),
        );
    }

    QualityAuditReport {
        schema_version: QUALITY_AUDIT_SCHEMA_VERSION,
        translation_data: translation_data.display().to_string(),
        glossary_path: glossary_path.map(|path| path.display().to_string()),
        source_language: source_language.to_string(),
        target_language: target_language.to_string(),
        include_soft_checks,
        term_check_mode: if glossary_path.is_none() {
            "disabled"
        } else if hard_terms_only {
            "explicit_hard_constraints"
        } else {
            "diagnostic_all_glossary_entries"
        }
        .to_string(),
        glossary_entry_count,
        limitations,
        summary,
        issues,
    }
}

fn load_audit_units(path: &Path) -> Result<Vec<AuditUnit>> {
    let json = std::fs::read_to_string(path)
        .with_context(|| format!("读取翻译数据失败: {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&json)
        .with_context(|| format!("解析翻译数据失败: {}", path.display()))?;
    let records = value
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("翻译数据必须是 JSON 数组: {}", path.display()))?;
    let is_epub = records
        .first()
        .and_then(serde_json::Value::as_object)
        .is_some_and(|record| record.contains_key("file_id") || record.contains_key("file_name"));

    if is_epub {
        let blocks: Vec<TranslationBlock> = serde_json::from_value(value)
            .with_context(|| format!("解析 EPUB 翻译数据失败: {}", path.display()))?;
        Ok(blocks
            .into_iter()
            .map(|block| AuditUnit {
                unit_id: nonempty(block.unit_id),
                file_name: nonempty(block.file_name),
                block_index: block.index,
                source_text: block.src_text,
                translated_text: block.dst_text.unwrap_or_default(),
            })
            .collect())
    } else {
        let blocks: Vec<TxtBlock> = serde_json::from_value(value)
            .with_context(|| format!("解析 TXT 翻译数据失败: {}", path.display()))?;
        Ok(blocks
            .into_iter()
            .map(|block| AuditUnit {
                unit_id: nonempty(block.unit_id),
                file_name: None,
                block_index: block.index,
                source_text: block.src_text,
                translated_text: block.dst_text.unwrap_or_default(),
            })
            .collect())
    }
}

fn nonempty(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_counts_all_findings_without_hiding_same_unit_errors() {
        let checker = ResponseChecker::new("ja", "zh", 0.80, usize::MAX);
        let units = vec![
            AuditUnit {
                unit_id: Some("u1".to_string()),
                file_name: Some("Text/ch1.xhtml".to_string()),
                block_index: 3,
                source_text: "第12章、{player}の番です。".to_string(),
                translated_text: "轮到玩家了。".to_string(),
            },
            AuditUnit {
                unit_id: Some("u2".to_string()),
                file_name: Some("Text/ch1.xhtml".to_string()),
                block_index: 4,
                source_text: "こんにちは".to_string(),
                translated_text: "你好".to_string(),
            },
        ];

        let report = build_report(
            Path::new("translation_data.json"),
            None,
            "ja",
            "zh",
            true,
            false,
            0,
            units,
            &checker,
        );

        assert_eq!(report.summary.total_units, 2);
        assert_eq!(report.summary.issue_units, 1);
        assert_eq!(report.summary.total_findings, 2);
        assert_eq!(report.summary.findings_by_type["NUMBER_MISMATCH"], 1);
        assert_eq!(report.summary.findings_by_type["PLACEHOLDER_MISMATCH"], 1);
        assert_eq!(report.issues[0].block_index, 3);
    }

    #[test]
    fn default_report_path_keeps_translation_data_stem() {
        assert_eq!(
            default_quality_audit_path(Path::new("book_work/translation_data.json")),
            PathBuf::from("book_work/translation_data.quality-report.json")
        );
    }
}
