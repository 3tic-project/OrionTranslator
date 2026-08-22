//! Small numeric helpers (GELU, attention masks, flattened linear).

use burn::nn::Linear;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Float, Tensor};

/// Exact GELU via burn.
#[inline]
pub fn gelu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    burn::tensor::activation::gelu(x)
}

/// Rank-2 linear: one large GEMM.
///
/// `burn::nn::Linear::forward` on a rank-3 input broadcasts the weight and lands in
/// the backend's batched matmul, which issues one small GEMM per batch item. Keeping
/// activations as `[batch*seq, features]` collapses that into a single GEMM call.
#[inline]
pub fn linear2<B: Backend>(lin: &Linear<B>, x: Tensor<B, 2>) -> Tensor<B, 2> {
    let y = x.matmul(lin.weight.val());
    match &lin.bias {
        Some(b) => y + b.val().unsqueeze::<2>(),
        None => y,
    }
}

/// Host-side window bias template for a given seq length and radius.
/// Returns flat `seq*seq` values (0 or -1e4).
pub fn window_bias_host(seq: usize, radius: usize) -> Vec<f32> {
    let mut dist = vec![0.0f32; seq * seq];
    for i in 0..seq {
        let row = i * seq;
        let j0 = i.saturating_sub(radius);
        let j1 = (i + radius + 1).min(seq);
        // block left of window
        for j in 0..j0 {
            dist[row + j] = -1.0e4;
        }
        // block right of window
        for j in j1..seq {
            dist[row + j] = -1.0e4;
        }
    }
    dist
}

/// Additive pad mask `[batch, 1, 1, seq]`; broadcasts over heads and query positions.
pub fn pad_bias_from_mask<B: Backend>(attention_mask: &Tensor<B, 2>) -> Tensor<B, 4> {
    let [batch, seq] = attention_mask.dims();
    let key_keep = attention_mask.clone().reshape([batch, 1, 1, seq]);
    (key_keep.ones_like() - key_keep).mul_scalar(-1.0e4)
}

/// Scaled dot-product attention. q,k,v: `[B, H, S, D]`.
///
/// Biases stay separate and broadcast into the score tensor: combining them first
/// would materialise a `[B, 1, S, S]` intermediate for no benefit.
pub fn attention_forward<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    pad_bias: Option<&Tensor<B, 4>>,
    window_bias: Option<&Tensor<B, 4>>,
) -> Tensor<B, 4, Float> {
    let head_dim = q.dims()[3] as f32;
    // Scaling q (S x D) is cheaper than scaling the S x S scores.
    let q = q.mul_scalar(1.0 / head_dim.sqrt());
    let mut scores = q.matmul(k.swap_dims(2, 3));
    if let Some(bias) = pad_bias {
        scores = scores + bias.clone();
    }
    if let Some(bias) = window_bias {
        scores = scores + bias.clone();
    }
    softmax(scores, 3).matmul(v)
}

#[allow(dead_code)]
pub fn keep_mask_from_pad_bool<B: Backend>(pad: Tensor<B, 2, Bool>) -> Tensor<B, 2, Float> {
    pad.bool_not().float()
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray;

    #[test]
    fn gelu_zero() {
        let device = Default::default();
        let x = Tensor::<B, 1>::from_floats(TensorData::new(vec![0.0f32], [1]), &device);
        let y = gelu(x).into_data();
        let v: &[f32] = y.as_slice().unwrap();
        assert!(v[0].abs() < 1e-6);
    }

    #[test]
    fn pad_bias_blocks_padding_only() {
        let device = Default::default();
        let mask = Tensor::<B, 2>::from_floats([[1.0f32, 1.0, 0.0]], &device);
        let bias = pad_bias_from_mask::<B>(&mask);
        assert_eq!(bias.dims(), [1, 1, 1, 3]);
        let v: Vec<f32> = bias.into_data().to_vec().unwrap();
        assert!(v[0].abs() < 1.0);
        assert!(v[1].abs() < 1.0);
        assert!(v[2] < -1.0e3);
    }

    #[test]
    fn window_host_matches_template() {
        let t = window_bias_host(8, 2);
        assert_eq!(t.len(), 64);
        assert!(t[0 * 8 + 3] < -1.0e3);
        assert!(t[0 * 8 + 2].abs() < 1.0);
    }
}
