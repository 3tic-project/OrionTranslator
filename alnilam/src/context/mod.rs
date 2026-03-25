mod detector;
mod selector;
mod trie;
mod types;

pub use detector::ContextDetector;
pub use selector::{
    precompute_context, select_context, select_context_precomputed, PrecomputedContext,
    SelectionResult,
};
