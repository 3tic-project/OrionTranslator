use sha2::{Digest, Sha256};

const SOURCE_ID_VERSION: &str = "orion-source-v1";
const UNIT_ID_VERSION: &str = "orion-unit-v1";

fn sha256_hex(parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

pub(crate) fn source_sha256(source: &str) -> String {
    format!("source-v1:{}", sha256_hex(&[SOURCE_ID_VERSION, source]))
}

/// 生成不依赖全局数组位置的稳定单元身份。
///
/// `scope` 是 EPUB 内的规范文档路径，TXT 使用固定逻辑作用域；`occurrence`
/// 只用于区分同一作用域内完全相同的重复原文。因此插入不相干段落不会改变已有 ID。
pub(crate) fn unit_id(kind: &str, scope: &str, source: &str, occurrence: usize) -> String {
    let source_hash = source_sha256(source);
    format!(
        "unit-v1:{}",
        sha256_hex(&[
            UNIT_ID_VERSION,
            kind,
            scope,
            &source_hash,
            &occurrence.to_string(),
        ])
    )
}

pub(crate) fn source_hash_matches(source: &str, expected: &str) -> bool {
    !expected.is_empty() && source_sha256(source) == expected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_id_is_stable_and_scope_sensitive() {
        let first = unit_id("epub", "Text/ch1.xhtml", "同じ原文", 0);
        assert_eq!(first, unit_id("epub", "Text/ch1.xhtml", "同じ原文", 0));
        assert_ne!(first, unit_id("epub", "Text/ch2.xhtml", "同じ原文", 0));
        assert_ne!(first, unit_id("epub", "Text/ch1.xhtml", "同じ原文", 1));
        assert_ne!(first, unit_id("epub", "Text/ch1.xhtml", "変更後", 0));
    }

    #[test]
    fn source_hash_rejects_changed_text() {
        let hash = source_sha256("原文");
        assert!(source_hash_matches("原文", &hash));
        assert!(!source_hash_matches("原文。", &hash));
        assert!(!source_hash_matches("原文", ""));
    }
}
