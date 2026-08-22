//! End-to-end CPU inference against the fine-tuned JA ModernBERT NER checkpoint.

use modernbert_ner::{load_pipeline_cpu, pack_texts, InferOptions};
use std::path::PathBuf;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../alnilam/ner_model")
}

#[test]
fn predict_japanese_sentence_returns_structured_entities() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skip e2e: model missing at {}", dir.display());
        return;
    }

    let pipeline = load_pipeline_cpu(&dir, 256).expect("load pipeline");
    assert_eq!(pipeline.config().num_labels(), 17);
    assert_eq!(pipeline.config().hidden_size, 256);

    let text = "橘真児は東京で美咲と会った。";
    let result = pipeline.predict(text).expect("predict");
    assert_eq!(result.text, text);
    assert_eq!(result.labels.len(), text.chars().count());
    for lab in &result.labels {
        assert!(lab == "O" || lab.contains('-'), "unexpected label {lab}");
    }
    for e in &result.entities {
        assert!(e.score.is_finite());
        assert!(e.start < e.end);
        let surface: String = text.chars().skip(e.start).take(e.end - e.start).collect();
        assert_eq!(surface, e.text);
    }

    let batch = pipeline
        .predict_batch(&[text, "今日は雨だ。"])
        .expect("batch");
    assert_eq!(batch.len(), 2);
    assert_eq!(batch[0].labels, result.labels);
}

#[test]
fn packing_preserves_labels_vs_unsorted_batch() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        return;
    }
    let pipeline = load_pipeline_cpu(&dir, 256)
        .unwrap()
        .with_options(InferOptions {
            max_sentences: 8,
            max_tokens: 512,
            sort_by_length: true,
            skip_scores: false,
        });

    let texts = vec![
        "短い。".to_string(),
        "橘真児は東京で美咲と会った。とても長い文をここに書いてパディング差を作る。".to_string(),
        "今日は雨だ。".to_string(),
        "健太と志保子。".to_string(),
        "x".to_string(),
        "沙由美ちゃんが来た。".to_string(),
    ];

    // Document path (packed)
    let (doc_results, _, packs) = pipeline.predict_document(&texts).unwrap();
    assert!(!packs.is_empty());
    assert_eq!(doc_results.len(), texts.len());

    // Naive per-line
    for (i, t) in texts.iter().enumerate() {
        let single = pipeline.predict(t).unwrap();
        assert_eq!(
            doc_results[i].labels, single.labels,
            "labels mismatch at line {i}"
        );
        let a: Vec<_> = doc_results[i]
            .entities
            .iter()
            .map(|e| (e.start, e.end, e.label.clone(), e.text.clone()))
            .collect();
        let b: Vec<_> = single
            .entities
            .iter()
            .map(|e| (e.start, e.end, e.label.clone(), e.text.clone()))
            .collect();
        assert_eq!(a, b, "entities mismatch at line {i}");
    }
}

#[test]
fn pack_texts_covers_all_indices() {
    let texts: Vec<String> = (0..50).map(|i| "あ".repeat(1 + (i % 20))).collect();
    let packs = pack_texts(&texts, 8, 256, true);
    let mut seen = vec![false; texts.len()];
    for p in packs {
        for i in p.orig_indices {
            assert!(!seen[i]);
            seen[i] = true;
        }
    }
    assert!(seen.iter().all(|&x| x));
}
