use anyhow::{anyhow, Result};
use log::info;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use vibrato::{Dictionary, Tokenizer};

#[allow(dead_code)]
pub struct JapaneseBertTokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    mecab_tokenizer: Option<Tokenizer>,
    cls_token_id: u32,
    sep_token_id: u32,
    pad_token_id: u32,
    unk_token_id: u32,
    max_length: usize,
}

impl JapaneseBertTokenizer {
    pub fn new<P: AsRef<Path>>(
        vocab_path: P,
        dict_path: Option<&str>,
        max_length: usize,
    ) -> Result<Self> {
        let vocab = Self::load_vocab(&vocab_path)?;
        let id_to_token: HashMap<u32, String> =
            vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

        let cls_token_id = *vocab
            .get("[CLS]")
            .ok_or_else(|| anyhow!("[CLS] not in vocab"))?;
        let sep_token_id = *vocab
            .get("[SEP]")
            .ok_or_else(|| anyhow!("[SEP] not in vocab"))?;
        let pad_token_id = *vocab
            .get("[PAD]")
            .ok_or_else(|| anyhow!("[PAD] not in vocab"))?;
        let unk_token_id = *vocab
            .get("[UNK]")
            .ok_or_else(|| anyhow!("[UNK] not in vocab"))?;

        let mecab_tokenizer = if let Some(path) = dict_path {
            Some(Self::load_vibrato_dict(path)?)
        } else {
            info!("No MeCab dictionary provided, using character-only tokenization");
            None
        };

        Ok(Self {
            vocab,
            id_to_token,
            mecab_tokenizer,
            cls_token_id,
            sep_token_id,
            pad_token_id,
            unk_token_id,
            max_length,
        })
    }

    fn load_vibrato_dict(path: &str) -> Result<Tokenizer> {
        let file = File::open(path)?;
        let dict = if path.ends_with(".zst") {
            let decoder = zstd::Decoder::new(file)?;
            Dictionary::read(decoder)?
        } else {
            Dictionary::read(file)?
        };
        Ok(Tokenizer::new(dict))
    }

    fn load_vocab<P: AsRef<Path>>(path: P) -> Result<HashMap<String, u32>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut vocab = HashMap::new();

        for (idx, line) in reader.lines().enumerate() {
            let token = line?;
            vocab.insert(token, idx as u32);
        }

        Ok(vocab)
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        if let Some(tokenizer) = &self.mecab_tokenizer {
            let mut worker = tokenizer.new_worker();
            worker.reset_sentence(text);
            worker.tokenize();

            let mut chars = Vec::new();
            for i in 0..worker.num_tokens() {
                let token = worker.token(i);
                let surface = token.surface();
                for ch in surface.chars() {
                    chars.push(ch.to_string());
                }
            }
            chars
        } else {
            text.chars().map(|c| c.to_string()).collect()
        }
    }

    pub fn encode(&self, text: &str) -> Result<EncodedInput> {
        let tokens = self.tokenize(text);
        let mut input_ids = vec![self.cls_token_id];
        let mut original_chars: Vec<Option<(char, usize)>> = vec![None];

        for (char_idx, token) in tokens.iter().enumerate() {
            let token_id = self.vocab.get(token).copied();
            if let Some(id) = token_id {
                input_ids.push(id);
                original_chars.push(Some((token.chars().next().unwrap(), char_idx)));
            }
        }

        input_ids.push(self.sep_token_id);
        original_chars.push(None);

        let seq_len = input_ids.len();
        let attention_mask: Vec<u32> = vec![1; seq_len];
        let token_type_ids: Vec<u32> = vec![0; seq_len];

        if seq_len > self.max_length {
            Ok(EncodedInput {
                input_ids: input_ids[..self.max_length].to_vec(),
                attention_mask: attention_mask[..self.max_length].to_vec(),
                token_type_ids: token_type_ids[..self.max_length].to_vec(),
                original_chars: original_chars[..self.max_length].to_vec(),
            })
        } else {
            Ok(EncodedInput {
                input_ids,
                attention_mask,
                token_type_ids,
                original_chars,
            })
        }
    }

    #[allow(dead_code)]
    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    #[allow(dead_code)]
    pub fn decode_token(&self, token_id: u32) -> Option<&String> {
        self.id_to_token.get(&token_id)
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct EncodedInput {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub original_chars: Vec<Option<(char, usize)>>,
}

#[allow(dead_code)]
impl EncodedInput {
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }
}
