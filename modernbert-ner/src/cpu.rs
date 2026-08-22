//! Dedicated CPU inference engine for ModernBERT token classification.
//!
//! Written against raw `f32` buffers rather than the Burn `ndarray` backend, because
//! that backend copies every array on `reshape` (three full copies per matmul) and
//! evaluates GELU through `libm`'s scalar f64 `erf`. Here the batch is stored fully
//! packed (no padding), every projection is a single `gemm` call over strided views,
//! and all intermediate buffers are reused across layers and batches.

use anyhow::{anyhow, bail, Result};
use candle_core::{safetensors, Device as CandleDevice, Tensor as CandleTensor};
use gemm::{gemm, Parallelism};
use std::collections::HashMap;
use std::path::Path;

use crate::config::ModernBertNerConfig;

/// `dst[m, n] = x[m, k] · wᵀ`, where `w` is a PyTorch-style `[n, k]` row-major weight.
///
/// With `accumulate` the product is added to `dst` instead of overwriting it, which
/// lets residual branches write straight into the hidden state.
#[inline]
fn matmul_nt(
    dst: &mut [f32],
    x: &[f32],
    w: &[f32],
    m: usize,
    k: usize,
    n: usize,
    accumulate: bool,
) {
    debug_assert_eq!(dst.len(), m * n);
    debug_assert_eq!(x.len(), m * k);
    debug_assert_eq!(w.len(), n * k);
    if m == 0 {
        return;
    }
    // SAFETY: shapes are checked above and all three buffers are distinct row-major
    // allocations owned by the caller.
    unsafe {
        gemm(
            m,
            n,
            k,
            dst.as_mut_ptr(),
            1,
            n as isize,
            accumulate,
            x.as_ptr(),
            1,
            k as isize,
            w.as_ptr(),
            k as isize,
            1,
            if accumulate { 1.0 } else { 0.0 },
            1.0,
            false,
            false,
            false,
            Parallelism::None,
        );
    }
}

/// Branch-free `e^x`, accurate to ~1e-7 relative.
///
/// `f32::exp` resolves to a `libm` call that cannot be inlined, so the GELU and
/// softmax loops around it stay scalar and pay a PLT stub per element. This version
/// inlines and lets LLVM auto-vectorise those loops.
#[inline(always)]
fn exp(x: f32) -> f32 {
    // ln(2) split in two parts to keep the range reduction accurate in f32.
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;

    // Clamp so the 2^k reconstruction below stays inside the f32 exponent range.
    let x = x.clamp(-87.3, 88.0);
    let k = (x * std::f32::consts::LOG2_E).round_ties_even();
    let r = x - k * LN2_HI - k * LN2_LO;
    let poly = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0))))));
    f32::from_bits(((k as i32 + 127) as u32) << 23) * poly
}

/// Error of the Abramowitz–Stegun 7.1.26 rational form is <= 1.5e-7, i.e. below f32
/// resolution, so this stays numerically equivalent to the exact `erf` GELU.
#[inline(always)]
fn erf(x: f32) -> f32 {
    const A1: f32 = 0.254_829_6;
    const A2: f32 = -0.284_496_72;
    const A3: f32 = 1.421_413_8;
    const A4: f32 = -1.453_152_1;
    const A5: f32 = 1.061_405_4;
    const P: f32 = 0.327_591_1;

    let sign = if x < 0.0 { -1.0f32 } else { 1.0f32 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    sign * (1.0 - poly * exp(-ax * ax))
}

#[inline(always)]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

fn layer_norm_rows(x: &mut [f32], rows: usize, dim: usize, gamma: &[f32], eps: f32) {
    let inv_dim = 1.0 / dim as f32;
    for r in 0..rows {
        let row = &mut x[r * dim..(r + 1) * dim];
        let mut sum = 0.0f32;
        for &v in row.iter() {
            sum += v;
        }
        let mean = sum * inv_dim;
        let mut var = 0.0f32;
        for &v in row.iter() {
            let d = v - mean;
            var += d * d;
        }
        let inv_std = 1.0 / (var * inv_dim + eps).sqrt();
        for (v, g) in row.iter_mut().zip(gamma.iter()) {
            *v = (*v - mean) * inv_std * *g;
        }
    }
}

/// Softmax over `row[lo..hi]`, zeroing everything outside the window.
#[inline]
fn softmax_window(row: &mut [f32], lo: usize, hi: usize) {
    for v in row[..lo].iter_mut() {
        *v = 0.0;
    }
    for v in row[hi..].iter_mut() {
        *v = 0.0;
    }
    let active = &mut row[lo..hi];
    let mut max = f32::NEG_INFINITY;
    for &v in active.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in active.iter_mut() {
        let e = exp(*v - max);
        *v = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for v in active.iter_mut() {
        *v *= inv;
    }
}

struct Layer {
    /// `None` for layer 0, which skips the pre-attention norm.
    attn_norm: Option<Vec<f32>>,
    wqkv: Vec<f32>,
    wo: Vec<f32>,
    mlp_norm: Vec<f32>,
    wi: Vec<f32>,
    wo_mlp: Vec<f32>,
}

/// Reusable per-worker buffers. Sized on first use and grown only when a larger
/// pack arrives, so steady-state inference performs no allocation.
#[derive(Default)]
pub struct Scratch {
    hidden: Vec<f32>,
    normed: Vec<f32>,
    qkv: Vec<f32>,
    ctx: Vec<f32>,
    mlp: Vec<f32>,
    act: Vec<f32>,
    scores: Vec<f32>,
    logits: Vec<f32>,
}

fn resize(buf: &mut Vec<f32>, len: usize) {
    if buf.len() < len {
        buf.resize(len, 0.0);
    }
}

pub struct CpuModel {
    cfg: ModernBertNerConfig,
    embed: Vec<f32>,
    embed_norm: Vec<f32>,
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
    head_dense: Vec<f32>,
    head_norm: Vec<f32>,
    classifier_w: Vec<f32>,
    classifier_b: Vec<f32>,
    /// `[max_positions, head_dim/2]` tables; the second half of each head mirrors these.
    rope_global: (Vec<f32>, Vec<f32>),
    rope_local: (Vec<f32>, Vec<f32>),
    max_positions: usize,
}

fn rope_tables(max_positions: usize, head_dim: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; max_positions * half];
    let mut sin = vec![0.0f32; max_positions * half];
    for i in 0..half {
        let inv = (1.0 / theta.powf((2 * i) as f64 / head_dim as f64)) as f32;
        for t in 0..max_positions {
            let angle = t as f32 * inv;
            cos[t * half + i] = angle.cos();
            sin[t * half + i] = angle.sin();
        }
    }
    (cos, sin)
}

fn take_2d(
    weights: &HashMap<String, CandleTensor>,
    key: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    let t = weights
        .get(key)
        .ok_or_else(|| anyhow!("missing weight tensor: {key}"))?;
    let dims = t.dims();
    if dims != [rows, cols] {
        bail!("weight {key} has shape {dims:?}, expected [{rows}, {cols}]");
    }
    let v = t
        .flatten_all()
        .and_then(|f| f.to_vec1::<f32>())
        .map_err(|e| anyhow!("read {key}: {e}"))?;
    Ok(v)
}

fn take_1d(weights: &HashMap<String, CandleTensor>, key: &str, len: usize) -> Result<Vec<f32>> {
    let t = weights
        .get(key)
        .ok_or_else(|| anyhow!("missing weight tensor: {key}"))?;
    let dims = t.dims();
    if dims != [len] {
        bail!("weight {key} has shape {dims:?}, expected [{len}]");
    }
    t.to_vec1::<f32>().map_err(|e| anyhow!("read {key}: {e}"))
}

impl CpuModel {
    pub fn load(model_dir: impl AsRef<Path>, max_positions: usize) -> Result<Self> {
        let dir = model_dir.as_ref();
        let cfg = ModernBertNerConfig::load(dir.join("config.json"))?;
        let weights = safetensors::load(dir.join("model.safetensors"), &CandleDevice::Cpu)
            .map_err(|e| anyhow!("safetensors load failed: {e:?}"))?;
        Self::from_weights(cfg, &weights, max_positions)
    }

    fn from_weights(
        cfg: ModernBertNerConfig,
        w: &HashMap<String, CandleTensor>,
        max_positions: usize,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let labels = cfg.num_labels();
        let max_positions = max_positions.max(1);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                attn_norm: if i == 0 {
                    None
                } else {
                    Some(take_1d(w, &format!("{p}.attn_norm.weight"), h)?)
                },
                wqkv: take_2d(w, &format!("{p}.attn.Wqkv.weight"), 3 * h, h)?,
                wo: take_2d(w, &format!("{p}.attn.Wo.weight"), h, h)?,
                mlp_norm: take_1d(w, &format!("{p}.mlp_norm.weight"), h)?,
                wi: take_2d(w, &format!("{p}.mlp.Wi.weight"), 2 * inter, h)?,
                wo_mlp: take_2d(w, &format!("{p}.mlp.Wo.weight"), h, inter)?,
            });
        }

        let head_dim = cfg.head_dim();
        Ok(Self {
            embed: take_2d(
                w,
                "model.embeddings.tok_embeddings.weight",
                cfg.vocab_size,
                h,
            )?,
            embed_norm: take_1d(w, "model.embeddings.norm.weight", h)?,
            layers,
            final_norm: take_1d(w, "model.final_norm.weight", h)?,
            head_dense: take_2d(w, "head.dense.weight", h, h)?,
            head_norm: take_1d(w, "head.norm.weight", h)?,
            classifier_w: take_2d(w, "classifier.weight", labels, h)?,
            classifier_b: take_1d(w, "classifier.bias", labels)?,
            rope_global: rope_tables(max_positions, head_dim, cfg.global_rope_theta),
            rope_local: rope_tables(max_positions, head_dim, cfg.local_rope_theta),
            max_positions,
            cfg,
        })
    }

    pub fn config(&self) -> &ModernBertNerConfig {
        &self.cfg
    }

    /// Forward a fully packed batch.
    ///
    /// `ids` holds every sequence back to back; `offsets` has `batch + 1` entries with
    /// `offsets[0] == 0` and `offsets[batch] == ids.len()`. Returns `[tokens, num_labels]`
    /// logits in the same packed order.
    pub fn forward<'s>(
        &self,
        ids: &[u32],
        offsets: &[usize],
        s: &'s mut Scratch,
    ) -> Result<&'s [f32]> {
        let h = self.cfg.hidden_size;
        let inter = self.cfg.intermediate_size;
        let labels = self.cfg.num_labels();
        let n_heads = self.cfg.num_attention_heads;
        let head_dim = self.cfg.head_dim();
        let radius = self.cfg.local_window_radius();
        let eps = self.cfg.eps() as f32;
        let n = ids.len();
        let batch = offsets.len().saturating_sub(1);

        if offsets.first() != Some(&0) || offsets.last() != Some(&n) {
            bail!("offsets must start at 0 and end at ids.len()");
        }
        let max_len = (0..batch)
            .map(|b| offsets[b + 1] - offsets[b])
            .max()
            .unwrap_or(0);
        if max_len > self.max_positions {
            bail!(
                "sequence length {max_len} exceeds RoPE table size {}",
                self.max_positions
            );
        }
        resize(&mut s.logits, n * labels);
        if n == 0 {
            return Ok(&s.logits[..0]);
        }

        resize(&mut s.hidden, n * h);
        resize(&mut s.normed, n * h);
        resize(&mut s.qkv, n * 3 * h);
        resize(&mut s.ctx, n * h);
        resize(&mut s.mlp, n * 2 * inter);
        resize(&mut s.act, n * inter);
        resize(&mut s.scores, max_len * max_len);

        let hidden = &mut s.hidden[..n * h];
        for (t, &id) in ids.iter().enumerate() {
            let src = (id as usize) * h;
            if src + h > self.embed.len() {
                bail!("token id {id} out of vocabulary");
            }
            hidden[t * h..(t + 1) * h].copy_from_slice(&self.embed[src..src + h]);
        }
        layer_norm_rows(hidden, n, h, &self.embed_norm, eps);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let half = head_dim / 2;
        let qkv_row = 3 * h;

        for (layer_id, layer) in self.layers.iter().enumerate() {
            let normed = &mut s.normed[..n * h];
            normed.copy_from_slice(hidden);
            if let Some(gamma) = &layer.attn_norm {
                layer_norm_rows(normed, n, h, gamma, eps);
            }

            let qkv = &mut s.qkv[..n * qkv_row];
            matmul_nt(qkv, normed, &layer.wqkv, n, h, 3 * h, false);

            let (cos, sin) = if self.cfg.is_global_layer(layer_id) {
                (&self.rope_global.0, &self.rope_global.1)
            } else {
                (&self.rope_local.0, &self.rope_local.1)
            };

            // RoPE in place over the Q and K thirds of each row.
            for b in 0..batch {
                let (start, len) = (offsets[b], offsets[b + 1] - offsets[b]);
                for t in 0..len {
                    let row = (start + t) * qkv_row;
                    let tc = &cos[t * half..t * half + half];
                    let ts = &sin[t * half..t * half + half];
                    for part in 0..2 {
                        let base = row + part * h;
                        for head in 0..n_heads {
                            let o = base + head * head_dim;
                            let (lo, hi) = qkv[o..o + head_dim].split_at_mut(half);
                            for i in 0..half {
                                let (c, sn) = (tc[i], ts[i]);
                                let (x1, x2) = (lo[i], hi[i]);
                                lo[i] = x1 * c - x2 * sn;
                                hi[i] = x2 * c + x1 * sn;
                            }
                        }
                    }
                }
            }

            let window = if self.cfg.is_global_layer(layer_id) {
                None
            } else {
                Some(radius)
            };
            let ctx = &mut s.ctx[..n * h];
            let scores = &mut s.scores[..max_len * max_len];
            for b in 0..batch {
                let (start, len) = (offsets[b], offsets[b + 1] - offsets[b]);
                if len == 0 {
                    continue;
                }
                for head in 0..n_heads {
                    let q = start * qkv_row + head * head_dim;
                    let k = q + h;
                    let v = q + 2 * h;
                    // SAFETY: `q`/`k`/`v` index inside `qkv`, `scores` holds len*len entries,
                    // and the strided views stay within the packed sequence for item `b`.
                    unsafe {
                        gemm(
                            len,
                            len,
                            head_dim,
                            scores.as_mut_ptr(),
                            1,
                            len as isize,
                            false,
                            qkv.as_ptr().add(q),
                            1,
                            qkv_row as isize,
                            qkv.as_ptr().add(k),
                            qkv_row as isize,
                            1,
                            0.0,
                            scale,
                            false,
                            false,
                            false,
                            Parallelism::None,
                        );
                    }
                    for i in 0..len {
                        let (lo, hi) = match window {
                            Some(r) => (i.saturating_sub(r), (i + r + 1).min(len)),
                            None => (0, len),
                        };
                        softmax_window(&mut scores[i * len..(i + 1) * len], lo, hi);
                    }
                    // SAFETY: same reasoning; `ctx` is a distinct buffer of n*h floats.
                    unsafe {
                        gemm(
                            len,
                            head_dim,
                            len,
                            ctx.as_mut_ptr().add(start * h + head * head_dim),
                            1,
                            h as isize,
                            false,
                            scores.as_ptr(),
                            1,
                            len as isize,
                            qkv.as_ptr().add(v),
                            1,
                            qkv_row as isize,
                            0.0,
                            1.0,
                            false,
                            false,
                            false,
                            Parallelism::None,
                        );
                    }
                }
            }

            matmul_nt(hidden, ctx, &layer.wo, n, h, h, true);

            let normed = &mut s.normed[..n * h];
            normed.copy_from_slice(hidden);
            layer_norm_rows(normed, n, h, &layer.mlp_norm, eps);

            let mlp = &mut s.mlp[..n * 2 * inter];
            matmul_nt(mlp, normed, &layer.wi, n, h, 2 * inter, false);
            let act = &mut s.act[..n * inter];
            for t in 0..n {
                let row = &mlp[t * 2 * inter..(t + 1) * 2 * inter];
                let (input, gate) = row.split_at(inter);
                let out = &mut act[t * inter..(t + 1) * inter];
                for ((o, &i), &g) in out.iter_mut().zip(input.iter()).zip(gate.iter()) {
                    *o = gelu(i) * g;
                }
            }
            matmul_nt(hidden, act, &layer.wo_mlp, n, inter, h, true);
        }

        layer_norm_rows(hidden, n, h, &self.final_norm, eps);

        let head = &mut s.normed[..n * h];
        matmul_nt(head, hidden, &self.head_dense, n, h, h, false);
        for v in head.iter_mut() {
            *v = gelu(*v);
        }
        layer_norm_rows(head, n, h, &self.head_norm, eps);

        let logits = &mut s.logits[..n * labels];
        matmul_nt(logits, head, &self.classifier_w, n, h, labels, false);
        for t in 0..n {
            for (v, b) in logits[t * labels..(t + 1) * labels]
                .iter_mut()
                .zip(self.classifier_b.iter())
            {
                *v += *b;
            }
        }
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_matches_std() {
        for i in -870..=880 {
            let x = i as f32 / 10.0;
            let (a, b) = (exp(x), x.exp());
            assert!((a - b).abs() <= 1e-6 * b, "exp({x}) = {a} vs {b}");
        }
        assert_eq!(exp(-200.0), exp(-87.3));
    }

    #[test]
    fn erf_matches_reference() {
        // The series reference loses catastrophic precision past |x| ~ 4, so compare
        // against it inside that range and check saturation outside.
        for i in -40..=40 {
            let x = i as f64 / 10.0;
            let reference = series_erf(x) as f32;
            assert!(
                (erf(x as f32) - reference).abs() < 2e-6,
                "erf({x}) = {} vs {reference}",
                erf(x as f32)
            );
        }
        assert!((erf(6.0) - 1.0).abs() < 1e-6);
        assert!((erf(-6.0) + 1.0).abs() < 1e-6);
    }

    /// Maclaurin series reference so the test does not depend on the approximation.
    fn series_erf(x: f64) -> f64 {
        let mut term = x;
        let mut sum = x;
        for k in 1..200 {
            term *= -x * x / k as f64;
            sum += term / (2 * k + 1) as f64;
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    }

    #[test]
    fn gelu_known_values() {
        assert!((gelu(0.0)).abs() < 1e-7);
        assert!((gelu(1.0) - 0.841_345).abs() < 1e-4);
        assert!((gelu(-1.0) + 0.158_655).abs() < 1e-4);
    }

    #[test]
    fn layer_norm_normalizes() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma = vec![1.0f32; 4];
        layer_norm_rows(&mut x, 1, 4, &gamma, 1e-5);
        let mean: f32 = x.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
        let var: f32 = x.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn matmul_nt_matches_naive() {
        let (m, k, n) = (3, 4, 5);
        let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.37).sin()).collect();
        let w: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.11).cos()).collect();
        let mut dst = vec![0.0f32; m * n];
        matmul_nt(&mut dst, &x, &w, m, k, n, false);
        for r in 0..m {
            for c in 0..n {
                let expect: f32 = (0..k).map(|i| x[r * k + i] * w[c * k + i]).sum();
                assert!((dst[r * n + c] - expect).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn softmax_window_masks_outside() {
        let mut row = vec![1.0f32, 2.0, 3.0, 4.0];
        softmax_window(&mut row, 1, 3);
        assert_eq!(row[0], 0.0);
        assert_eq!(row[3], 0.0);
        assert!((row[1] + row[2] - 1.0).abs() < 1e-6);
        assert!(row[2] > row[1]);
    }
}
