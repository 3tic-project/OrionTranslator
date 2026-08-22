//! The hand-written CPU engine must agree with the Burn reference implementation.

#![cfg(feature = "ndarray")]

use modernbert_ner::{load_pipeline_burn_cpu, load_pipeline_cpu, InferOptions};
use std::path::PathBuf;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../alnilam/ner_model")
}

/// Includes lengths on both sides of the sliding-window threshold (radius = 64).
fn sample_texts() -> Vec<String> {
    let mut texts = vec![
        "橘真児は東京で美咲と会った。".to_string(),
        "今日は雨だ。".to_string(),
        "x".to_string(),
        "艾莉同学、おはよう".to_string(),
    ];
    for len in [20usize, 63, 64, 65, 66, 100, 129, 130, 200] {
        let mut s = String::new();
        while s.chars().count() < len {
            s.push_str("橘真児と美咲が廊下で会話をした。");
        }
        texts.push(s.chars().take(len).collect());
    }
    texts
}

#[test]
fn cpu_engine_matches_burn_reference() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skip parity: model missing at {}", dir.display());
        return;
    }

    let fast = load_pipeline_cpu(&dir, 256).expect("cpu engine");
    let reference = load_pipeline_burn_cpu(&dir, 256).expect("burn reference");

    for text in sample_texts() {
        let a = fast.predict(&text).expect("cpu predict");
        let b = reference.predict(&text).expect("burn predict");
        assert_eq!(
            a.labels,
            b.labels,
            "label mismatch for len={} text={}",
            text.chars().count(),
            &text.chars().take(24).collect::<String>()
        );
        assert_eq!(a.entities.len(), b.entities.len());
        for (x, y) in a.entities.iter().zip(b.entities.iter()) {
            assert_eq!(
                (x.start, x.end, &x.label, &x.text),
                (y.start, y.end, &y.label, &y.text)
            );
            assert!(
                (x.score - y.score).abs() < 5e-3,
                "score drift {} vs {}",
                x.score,
                y.score
            );
        }
    }
}

#[test]
fn cpu_engine_is_batch_invariant() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        return;
    }
    let pipeline = load_pipeline_cpu(&dir, 256)
        .unwrap()
        .with_options(InferOptions {
            max_sentences: 4,
            max_tokens: 512,
            sort_by_length: true,
            skip_scores: false,
        });

    let texts = sample_texts();
    let (batched, _, packs) = pipeline.predict_document(&texts).unwrap();
    assert!(!packs.is_empty());
    for (i, t) in texts.iter().enumerate() {
        let single = pipeline.predict(t).unwrap();
        assert_eq!(
            batched[i].labels, single.labels,
            "line {i} differs when batched"
        );
    }
}
