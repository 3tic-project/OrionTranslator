use serde::{Deserialize, Serialize};

/// Error type for translation quality issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorType {
    None,
    FailParse,
    FailLineCount,
    EmptyTranslation,
    Degradation,
    KanaResidue,
    KanaUntranslatedProse,
    HangeulResidue,
    HighSimilarity,
    CodeMismatch,
    JsonStructureError,
    LengthMismatch,
    TermMissing,
    NumberMismatch,
    PlaceholderMismatch,
}

impl std::fmt::Display for ErrorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorType::None => write!(f, "NONE"),
            ErrorType::FailParse => write!(f, "FAIL_PARSE"),
            ErrorType::FailLineCount => write!(f, "FAIL_LINE_COUNT"),
            ErrorType::EmptyTranslation => write!(f, "EMPTY_TRANSLATION"),
            ErrorType::Degradation => write!(f, "DEGRADATION"),
            ErrorType::KanaResidue => write!(f, "KANA_RESIDUE"),
            ErrorType::KanaUntranslatedProse => write!(f, "KANA_UNTRANSLATED_PROSE"),
            ErrorType::HangeulResidue => write!(f, "HANGEUL_RESIDUE"),
            ErrorType::HighSimilarity => write!(f, "HIGH_SIMILARITY"),
            ErrorType::CodeMismatch => write!(f, "CODE_MISMATCH"),
            ErrorType::JsonStructureError => write!(f, "JSON_STRUCTURE_ERROR"),
            ErrorType::LengthMismatch => write!(f, "LENGTH_MISMATCH"),
            ErrorType::TermMissing => write!(f, "TERM_MISSING"),
            ErrorType::NumberMismatch => write!(f, "NUMBER_MISMATCH"),
            ErrorType::PlaceholderMismatch => write!(f, "PLACEHOLDER_MISMATCH"),
        }
    }
}

/// Single check result
#[derive(Debug, Clone)]
pub struct CheckResult {
    pub error: ErrorType,
    pub details: String,
}

/// Error record for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRecord {
    pub index: usize,
    pub src_text: String,
    pub dst_text: String,
    pub error_type: String,
    pub fixed: bool,
    pub fix_details: String,
    pub retry_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error_details: Option<String>,
}
