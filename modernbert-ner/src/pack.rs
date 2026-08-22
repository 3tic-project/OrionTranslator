//! Length-aware batch packing to cut padding waste.
//!
//! Book lines vary a lot (mean ~37, max 200+). Fixed-order batching pads every
//! item to the longest in the micro-batch (~60% pad). Sorting by length drops
//! pad waste to ~1–2%.

/// One packed micro-batch: original indices + texts (already sorted by length).
#[derive(Debug, Clone)]
pub struct PackedBatch {
    /// Indices into the original input list (for restoring order).
    pub orig_indices: Vec<usize>,
    pub texts: Vec<String>,
    /// Approx max char length in this pack (for diagnostics).
    pub max_chars: usize,
}

/// Pack texts into micro-batches.
///
/// * `max_sentences` — hard cap on items per pack (GPU/CPU micro-batch size)
/// * `max_tokens` — soft budget: `max_len_in_pack * num_items` (chars+2 approx)
/// * `sort` — sort by char length before packing (strongly recommended)
pub fn pack_texts(
    texts: &[String],
    max_sentences: usize,
    max_tokens: usize,
    sort: bool,
) -> Vec<PackedBatch> {
    let max_sentences = max_sentences.max(1);
    let max_tokens = max_tokens.max(1);

    let mut order: Vec<usize> = (0..texts.len()).collect();
    if sort {
        order.sort_by_key(|&i| texts[i].chars().count());
    }

    let mut packs = Vec::new();
    let mut cur_idx = Vec::new();
    let mut cur_texts = Vec::new();
    let mut cur_max = 0usize;

    for &i in &order {
        let chars = texts[i].chars().count();
        // +2 for <s>/</s> specials (approx token count for packing budget)
        let approx_tok = chars.saturating_add(2);
        let new_max = cur_max.max(approx_tok);
        let new_count = cur_idx.len() + 1;
        let would_tokens = new_max.saturating_mul(new_count);

        let overflow =
            !cur_idx.is_empty() && (new_count > max_sentences || would_tokens > max_tokens);

        if overflow {
            packs.push(PackedBatch {
                orig_indices: std::mem::take(&mut cur_idx),
                texts: std::mem::take(&mut cur_texts),
                max_chars: cur_max.saturating_sub(2),
            });
            cur_max = 0;
        }

        cur_max = if cur_idx.is_empty() {
            approx_tok
        } else {
            cur_max.max(approx_tok)
        };
        cur_idx.push(i);
        cur_texts.push(texts[i].clone());

        // Single oversize line: flush immediately if alone and still over budget
        if cur_idx.len() == 1 && cur_max > max_tokens && max_sentences == 1 {
            // keep as-is; cannot split further here
        }
    }

    if !cur_idx.is_empty() {
        packs.push(PackedBatch {
            orig_indices: cur_idx,
            texts: cur_texts,
            max_chars: cur_max.saturating_sub(2),
        });
    }
    packs
}

/// Restore results to original order given packs and per-pack results.
pub fn unsort_results<T>(
    n_original: usize,
    packs: &[PackedBatch],
    pack_results: Vec<Vec<T>>,
) -> Vec<T> {
    assert_eq!(packs.len(), pack_results.len());
    let mut slots: Vec<Option<T>> = (0..n_original).map(|_| None).collect();
    for (pack, results) in packs.iter().zip(pack_results) {
        assert_eq!(pack.orig_indices.len(), results.len());
        for (orig, item) in pack.orig_indices.iter().zip(results) {
            slots[*orig] = Some(item);
        }
    }
    slots
        .into_iter()
        .map(|x| x.expect("missing packed result"))
        .collect()
}

/// Estimate pad waste ratio for diagnostics (char-level, +2 specials).
pub fn estimate_pad_waste(packs: &[PackedBatch]) -> f32 {
    let mut waste = 0usize;
    let mut total = 0usize;
    for p in packs {
        let lens: Vec<usize> = p.texts.iter().map(|t| t.chars().count() + 2).collect();
        if lens.is_empty() {
            continue;
        }
        let m = *lens.iter().max().unwrap();
        for l in lens {
            waste += m - l;
            total += m;
        }
    }
    if total == 0 {
        0.0
    } else {
        waste as f32 / total as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_reduces_pad_waste() {
        let texts: Vec<String> = (0..64)
            .map(|i| "あ".repeat(if i % 2 == 0 { 5 } else { 80 }))
            .collect();
        let unsorted = pack_texts(&texts, 8, 10_000, false);
        let sorted = pack_texts(&texts, 8, 10_000, true);
        let w_u = estimate_pad_waste(&unsorted);
        let w_s = estimate_pad_waste(&sorted);
        assert!(
            w_s < w_u * 0.5,
            "sorted waste {w_s} should be much less than unsorted {w_u}"
        );
    }

    #[test]
    fn unsort_restores_order() {
        let texts = vec!["ccc".into(), "a".into(), "bb".into()];
        let packs = pack_texts(&texts, 2, 1000, true);
        // fabricate results = original index as string
        let pack_results: Vec<Vec<String>> = packs
            .iter()
            .map(|p| p.orig_indices.iter().map(|i| format!("r{i}")).collect())
            .collect();
        let restored = unsort_results(3, &packs, pack_results);
        assert_eq!(
            restored,
            vec!["r0".to_string(), "r1".to_string(), "r2".to_string()]
        );
    }

    #[test]
    fn respects_max_sentences() {
        let texts: Vec<String> = (0..20).map(|i| format!("x{i}")).collect();
        let packs = pack_texts(&texts, 4, 100_000, true);
        assert!(packs.iter().all(|p| p.texts.len() <= 4));
        assert_eq!(packs.iter().map(|p| p.texts.len()).sum::<usize>(), 20);
    }

    #[test]
    fn token_budget_splits() {
        // many short lines: budget forces smaller packs when max_sentences is large
        let texts: Vec<String> = (0..32).map(|_| "あ".repeat(30)).collect();
        let packs = pack_texts(&texts, 32, 128, true);
        assert!(packs.len() > 1);
        for p in &packs {
            let max_t = p.texts.iter().map(|t| t.chars().count() + 2).max().unwrap();
            assert!(max_t * p.texts.len() <= 128 + max_t); // allow one oversize
        }
    }
}
