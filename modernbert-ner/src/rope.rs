//! Rotary position embeddings (RoPE) for ModernBERT.

use burn::tensor::backend::Backend;
use burn::tensor::{Float, Tensor};
use std::collections::HashMap;
use std::sync::Mutex;

/// Precomputed inv_freq: `1 / theta^(2i / dim)` for i in 0..dim/2.
pub fn inv_freq(head_dim: usize, theta: f64) -> Vec<f32> {
    assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");
    (0..head_dim / 2)
        .map(|i| {
            let exponent = (2 * i) as f64 / head_dim as f64;
            (1.0 / theta.powf(exponent)) as f32
        })
        .collect()
}

/// Host cos/sin tables, length `seq_len * head_dim` each.
pub fn cos_sin_host(seq_len: usize, head_dim: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let inv = inv_freq(head_dim, theta);
    let half = head_dim / 2;
    let mut cos_data = vec![0.0f32; seq_len * head_dim];
    let mut sin_data = vec![0.0f32; seq_len * head_dim];
    for t in 0..seq_len {
        let base = t * head_dim;
        for i in 0..half {
            let angle = (t as f32) * inv[i];
            let c = angle.cos();
            let s = angle.sin();
            cos_data[base + i] = c;
            cos_data[base + half + i] = c;
            sin_data[base + i] = s;
            sin_data[base + half + i] = s;
        }
    }
    (cos_data, sin_data)
}

/// Build cos/sin tables of shape `[1, 1, seq_len, head_dim]`.
pub fn cos_sin_tables<B: Backend>(
    seq_len: usize,
    head_dim: usize,
    theta: f64,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let (cos_data, sin_data) = cos_sin_host(seq_len, head_dim, theta);
    let cos = Tensor::<B, 1>::from_floats(
        burn::tensor::TensorData::new(cos_data, [seq_len * head_dim]),
        device,
    )
    .reshape([1, 1, seq_len, head_dim]);
    let sin = Tensor::<B, 1>::from_floats(
        burn::tensor::TensorData::new(sin_data, [seq_len * head_dim]),
        device,
    )
    .reshape([1, 1, seq_len, head_dim]);
    (cos, sin)
}

/// Process-wide host cache for cos/sin (seq, head_dim, theta_bits).
pub fn cached_cos_sin_host(seq_len: usize, head_dim: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    // Quantize theta to bits for HashMap key
    let key = (seq_len, head_dim, theta.to_bits());
    static CACHE: Mutex<Option<HashMap<(usize, usize, u64), (Vec<f32>, Vec<f32>)>>> =
        Mutex::new(None);
    let mut guard = CACHE.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    map.entry(key)
        .or_insert_with(|| cos_sin_host(seq_len, head_dim, theta))
        .clone()
}

pub fn cos_sin_tables_cached<B: Backend>(
    seq_len: usize,
    head_dim: usize,
    theta: f64,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let (cos_data, sin_data) = cached_cos_sin_host(seq_len, head_dim, theta);
    let cos = Tensor::<B, 1>::from_floats(
        burn::tensor::TensorData::new(cos_data, [seq_len * head_dim]),
        device,
    )
    .reshape([1, 1, seq_len, head_dim]);
    let sin = Tensor::<B, 1>::from_floats(
        burn::tensor::TensorData::new(sin_data, [seq_len * head_dim]),
        device,
    )
    .reshape([1, 1, seq_len, head_dim]);
    (cos, sin)
}

/// Rotate half of the last dim: `[-x2, x1]`.
pub fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, h, s, d] = x.dims();
    let half = d / 2;
    let x1 = x.clone().slice([0..b, 0..h, 0..s, 0..half]);
    let x2 = x.slice([0..b, 0..h, 0..s, half..d]);
    Tensor::cat(vec![x2.mul_scalar(-1.0), x1], 3)
}

/// Apply RoPE to query/key: `(x * cos) + (rotate_half(x) * sin)`.
pub fn apply_rope<B: Backend>(
    x: Tensor<B, 4>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
) -> Tensor<B, 4, Float> {
    let rotated = rotate_half(x.clone());
    x.mul(cos) + rotated.mul(sin)
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn inv_freq_length_and_monotonic() {
        let f = inv_freq(64, 10_000.0);
        assert_eq!(f.len(), 32);
        assert!(f[0] > f[31]);
        assert!((f[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rope_preserves_shape() {
        let device = Default::default();
        let (cos, sin) = cos_sin_tables::<B>(8, 64, 10_000.0, &device);
        let x = Tensor::<B, 4>::zeros([1, 4, 8, 64], &device);
        let y = apply_rope(x, cos, sin);
        assert_eq!(y.dims(), [1, 4, 8, 64]);
    }

    #[test]
    fn cache_is_stable() {
        let a = cached_cos_sin_host(16, 64, 10_000.0);
        let b = cached_cos_sin_host(16, 64, 10_000.0);
        assert_eq!(a.0, b.0);
    }
}
