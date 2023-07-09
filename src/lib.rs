use ndarray::{array, ArrayView1};
use std::io;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NdarrayErrors {
    #[error("vector size mismatched: x.len()={x_len}, y.len()={y_len}")]
    Mismatched { x_len: usize, y_len: usize },

    #[error("unknown NdarrayErrors error")]
    Unknown,
}

/// ### Def.
///     get a l2 norm for 1d vector
/// ### Examples
///     >>> let x = array![1., 1., 1., 1.];
///     >>> l2_norm(x.view());
fn l2_norm(x: ArrayView1<f64>) -> f64 {
    x.dot(&x).sqrt()
}

/// ### Def.
///     get a cosine similarity between two 1d vectors.
/// ### Examples
///     >>> let x = array![1., 2., 3., 4.]
///     >>> let y = array![5., 6., 7., 8.]
///     >>> let cos_sim = cosine_smilarity(x.view(), y.view())
pub fn cosine_similarity(x: ArrayView1<f64>, y: ArrayView1<f64>) -> Result<f64, NdarrayErrors> {
    if x.len() != y.len() {
        return Err(NdarrayErrors::Mismatched {
            x_len: (x.len()),
            y_len: (y.len()),
        });
    }

    let norm_x = l2_norm(x);
    let norm_y = l2_norm(y);

    if norm_x == 0. || norm_y == 0. {
        // divide by zero is completely well-defined on f64
        // but cosine similarity between two vectors that one of them is zero-vector remains UNDEFINED.
        // one can raises undefined error, we simply return zero.
        return Ok(0.);
    }

    let cosine = x.dot(&y) / (norm_x * norm_y);
    return Ok(cosine);
}

#[cfg(test)]
mod test_lib {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let source = array![1., 1., 1., 1.];

        // zero-case:
        let target = array![0., 0., 0., 0.];
        let cos_sim = cosine_similarity(source.view(), target.view()).unwrap();
        assert_eq!(0.0, cos_sim);

        // exact-similar case
        let cos_sim = cosine_similarity(source.view(), source.view()).unwrap();
        assert_eq!(1.0, cos_sim);

        // non-similar case
        let target = array![-1., -1., -1., -1.];
        let cos_sim = cosine_similarity(source.view(), target.view()).unwrap();
        assert_eq!(-1.0, cos_sim);

        // size mismatch error
        let target = array![-1., -1., -1.,];
        let result = cosine_similarity(source.view(), target.view());
        assert!(matches!(
            result,
            Err(NdarrayErrors::Mismatched { x_len, y_len })
        ));
    }
}
