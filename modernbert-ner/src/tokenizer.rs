//! Character-level NER encoding using HuggingFace `tokenizer.json` (Unigram).
//!
//! Matches Python training: each Unicode char is a "word" (`is_split_into_words=True`);
//! the label is taken from the first subword of each character.

use anyhow::{anyhow, Result};
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct EncodedChars {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    /// For each model token: `Some(char_index)` if it is the first subword of that char.
    pub word_ids: Vec<Option<usize>>,
    pub chars: Vec<char>,
}

#[derive(Clone)]
pub struct CharNerTokenizer {
    inner: Tokenizer,
    pub max_length: usize,
    pub pad_token_id: u32,
    pub cls_token_id: u32,
    pub sep_token_id: u32,
}

impl CharNerTokenizer {
    pub fn from_model_dir(model_dir: impl AsRef<Path>, max_length: usize) -> Result<Self> {
        let path = model_dir.as_ref().join("tokenizer.json");
        let inner =
            Tokenizer::from_file(&path).map_err(|e| anyhow!("load tokenizer.json failed: {e}"))?;

        // modernbert-ja TemplateProcessing uses <s> / </s> (not <cls>/<sep>).
        let cls_token_id = token_id(&inner, "<s>")
            .or_else(|| token_id(&inner, "<cls>"))
            .unwrap_or(1);
        let sep_token_id = token_id(&inner, "</s>")
            .or_else(|| token_id(&inner, "<sep>"))
            .unwrap_or(2);
        let pad_token_id = token_id(&inner, "<pad>").unwrap_or(3);

        Ok(Self {
            inner,
            max_length,
            pad_token_id,
            cls_token_id,
            sep_token_id,
        })
    }

    /// Encode one text as character "words" (aligned for token classification).
    pub fn encode_chars(&self, text: &str) -> Result<EncodedChars> {
        let chars: Vec<char> = text.chars().collect();
        if chars.is_empty() {
            return Ok(EncodedChars {
                input_ids: vec![self.cls_token_id, self.sep_token_id],
                attention_mask: vec![1, 1],
                word_ids: vec![None, None],
                chars,
            });
        }

        let pieces: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        // is_split_into_words=true → InputSequence::PreTokenizedOwned
        let encoding = self
            .inner
            .encode(pieces, true)
            .map_err(|e| anyhow!("tokenize failed: {e}"))?;

        let mut input_ids = encoding.get_ids().to_vec();
        let mut attention_mask = encoding.get_attention_mask().to_vec();
        let mut word_ids: Vec<Option<usize>> = encoding
            .get_word_ids()
            .iter()
            .map(|w| w.map(|x| x as usize))
            .collect();

        // Truncate to max_length, keep room is already handled by truncate if configured;
        // enforce hard cap.
        if input_ids.len() > self.max_length {
            input_ids.truncate(self.max_length);
            attention_mask.truncate(self.max_length);
            word_ids.truncate(self.max_length);
            // ensure last is SEP if possible
            if let Some(last) = input_ids.last_mut() {
                *last = self.sep_token_id;
            }
            if let Some(last) = word_ids.last_mut() {
                *last = None;
            }
        }

        Ok(EncodedChars {
            input_ids,
            attention_mask,
            word_ids,
            chars,
        })
    }

    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<EncodedChars>> {
        texts.iter().map(|t| self.encode_chars(t)).collect()
    }
}

fn token_id(tok: &Tokenizer, piece: &str) -> Option<u32> {
    tok.token_to_id(piece)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../alnilam/ner_model")
    }

    #[test]
    fn encodes_japanese_chars_with_specials() {
        let dir = model_dir();
        if !dir.join("tokenizer.json").exists() {
            return;
        }
        let tok = CharNerTokenizer::from_model_dir(&dir, 256).unwrap();
        let enc = tok.encode_chars("艾莉").unwrap();
        assert!(enc.input_ids.len() >= 2);
        assert_eq!(enc.input_ids[0], tok.cls_token_id);
        assert_eq!(*enc.input_ids.last().unwrap(), tok.sep_token_id);
        // every char should have at least one first-subword mapping
        let firsts: Vec<_> = enc.word_ids.iter().flatten().copied().collect();
        assert!(firsts.contains(&0));
        assert!(firsts.contains(&1) || enc.chars.len() == 1);
    }
}
