mod client;
pub mod glossary;
mod parser;
mod prompt;

pub use client::{BatchTranslationResponse, LlmClient};
pub use parser::ParseDiagnostics;
