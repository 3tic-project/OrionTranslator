use std::collections::HashMap;

/// Trie node for longest-match keyword matching
#[derive(Debug, Default)]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    /// (category_id, weight)
    outputs: Vec<(String, f64)>,
}

/// A keyword match found in text
#[derive(Debug, Clone)]
pub struct KeywordMatch {
    pub category_id: String,
    pub text: String,
    pub weight: f64,
    pub start: usize,
    pub end: usize,
    pub kind: MatchKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchKind {
    Keyword,
    Regex,
    Heuristic,
}

/// Aho–Corasick style Trie that emits only the LONGEST match at each start
/// position, preventing substring double-counting (e.g. 彼 inside 彼女).
#[derive(Debug, Default)]
pub struct TrieMatcher {
    root: TrieNode,
}

impl TrieMatcher {
    pub fn new() -> Self {
        Self {
            root: TrieNode::default(),
        }
    }

    pub fn add(&mut self, keyword: &str, category_id: &str, weight: f64) {
        let mut node = &mut self.root;
        for ch in keyword.chars() {
            node = node.children.entry(ch).or_default();
        }
        node.outputs.push((category_id.to_string(), weight));
    }

    pub fn find(&self, text: &str) -> Vec<KeywordMatch> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut matches = Vec::new();

        // Build byte-offset map for char indices
        let mut char_byte_offsets: Vec<usize> = Vec::with_capacity(n + 1);
        let mut offset = 0;
        for ch in &chars {
            char_byte_offsets.push(offset);
            offset += ch.len_utf8();
        }
        char_byte_offsets.push(offset);

        for i in 0..n {
            let mut node = &self.root;
            let mut j = i;
            // Track ALL outputs along the path, grouped by char end position
            let mut all_outputs: Vec<(usize, Vec<(String, f64)>)> = Vec::new();

            while j < n {
                let ch = chars[j];
                let next = match node.children.get(&ch) {
                    Some(n) => n,
                    None => break,
                };
                node = next;
                j += 1;
                if !node.outputs.is_empty() {
                    all_outputs.push((j, node.outputs.clone()));
                }
            }

            if all_outputs.is_empty() {
                continue;
            }

            // Only emit the longest match(es)
            let (longest_j, ref longest_outputs) = all_outputs[all_outputs.len() - 1];
            let matched_text: String = chars[i..longest_j].iter().collect();
            for (cat_id, weight) in longest_outputs {
                matches.push(KeywordMatch {
                    category_id: cat_id.clone(),
                    text: matched_text.clone(),
                    weight: *weight,
                    start: char_byte_offsets[i],
                    end: char_byte_offsets[longest_j],
                    kind: MatchKind::Keyword,
                });
            }

            // Also emit shorter matches that belong to DIFFERENT categories
            let longest_cat_ids: std::collections::HashSet<&str> =
                longest_outputs.iter().map(|(id, _)| id.as_str()).collect();

            for (end_j, outputs) in &all_outputs[..all_outputs.len() - 1] {
                for (cat_id, weight) in outputs {
                    if !longest_cat_ids.contains(cat_id.as_str()) {
                        let shorter_text: String = chars[i..*end_j].iter().collect();
                        matches.push(KeywordMatch {
                            category_id: cat_id.clone(),
                            text: shorter_text,
                            weight: *weight,
                            start: char_byte_offsets[i],
                            end: char_byte_offsets[*end_j],
                            kind: MatchKind::Keyword,
                        });
                    }
                }
            }
        }

        matches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trie_basic() {
        let mut trie = TrieMatcher::new();
        trie.add("彼", "pronoun3", 20.0);
        trie.add("彼女", "pronoun3", 20.0);

        let matches = trie.find("彼女は美しい");
        // Should only match 彼女 (longest), not 彼
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].text, "彼女");
    }

    #[test]
    fn test_trie_multiple_categories() {
        let mut trie = TrieMatcher::new();
        trie.add("その", "deixis", 14.0);
        trie.add("その人", "pronoun3", 20.0);

        let matches = trie.find("その人は");
        // Should match "その人" for pronoun3, and "その" for deixis (different category)
        assert_eq!(matches.len(), 2);
    }
}
