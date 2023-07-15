use ndarray::parallel::prelude::*;
use ndarray::{array, Array1, ArrayView1, ArrayView2, Axis};

use std::sync::{Arc, Mutex};
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
///     >>> let x = array![1., 2., 3., 4.];
///     >>> let y = array![5., 6., 7., 8.];
///     >>> let cos_sim = cosine_smilarity(x.view(), y.view());
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
        // but cosine similarity between two vectors that one of them is zero-vector remain UNDEFINED.
        // one can raises undefined error, we simply return zero.
        return Ok(0.);
    }

    let cosine = x.dot(&y) / (norm_x * norm_y);
    return Ok(cosine);
}

/// ### Def.
///     get a cosine similarity between 1d vector and 2d array.
/// ### Examples
///     >>> let x = array![1., 2., 3., 4.,];
///     >>> let y = array![[1., 2., 3., 4.,], [1., 2., 3., 4.,], [1., 2., 3., 4.,], [1., 2., 3., 4.,]];
///     >>> cosine_similarity_bulk(x.view(), y.view());
/// ### Execution time:
///     100,000 vectors with 1024 dim, < 2.0 sec.

pub fn cosine_similarity_bulk(x: ArrayView1<f64>, y: ArrayView2<f64>) -> Array1<f64> {
    let mut res: Array1<f64> = Array1::zeros(y.shape()[0]);

    for (idx, row) in y.axis_iter(Axis(0)).enumerate() {
        let cos_row = cosine_similarity(x, row).unwrap();
        res[idx] = cos_row;
    }

    return res;
}

/// ### Def.
///     parrallel version of buld cosine similarity
/// ### Examples
///     >>> let x = array![1., 2., 3., 4.,];
///     >>> let y = array![[1., 2., 3., 4.,], [1., 2., 3., 4.,], [1., 2., 3., 4.,], [1., 2., 3., 4.,]];
///     >>> cosine_similarity_bulk_parrallel(x.view(), y.view());
/// ### Execution time:
///     100,000 vectors with 1024 dim, < 0.3 sec(=300ms).
pub fn cosine_similarity_bulk_parrallel(x: ArrayView1<f64>, y: ArrayView2<f64>) -> Array1<f64> {
    let res: Arc<Mutex<Array1<f64>>> = Arc::new(Mutex::new(Array1::zeros(y.shape()[0])));

    y.axis_iter(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(idx, row)| {
            let cos_row = cosine_similarity(x.view(), row).unwrap();
            let mut res = res.lock().unwrap();
            res[idx] = cos_row; //if res was defined as Array1<f64> type, this line could NOT be compiled.
        });

    return Arc::try_unwrap(res).unwrap().into_inner().unwrap();
}

#[cfg(test)]
mod test_lib {
    use super::*;

    use ndarray::Array;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use std::array;
    use std::time::{Duration, Instant};

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

    #[test]
    fn test_cosine_similarity_bulk() {
        // check logic
        let source = array![1., 1., 1., 1.];
        let target = array![[0., 0., 0., 0.], [1., 1., 1., 1.], [-1., -1., -1., -1.],];
        let result = cosine_similarity_bulk(source.view(), target.view());
        assert_eq!(array![0., 1., -1.], result);

        // check execution time.
        let n = 100000;
        let dim = 1024;
        let source = Array::random(dim, Uniform::new(0., 10.));
        let target = Array::random((n, dim), Uniform::new(0., 10.));

        let start = Instant::now();
        cosine_similarity_bulk(source.view(), target.view());
        let duration = start.elapsed();
        assert!(duration.as_secs_f64() < 2.0); // single thread, (10만 개, 1024)에서 2초 넘으면 fail.
        println!("Execution time for single thread: {:.4?}", duration);
    }

    #[test]
    fn test_cosine_similarity_bulk_parralle() {
        // check logic
        let source = array![1., 1., 1., 1.];
        let target = array![[0., 0., 0., 0.], [1., 1., 1., 1.], [-1., -1., -1., -1.],];
        let result = cosine_similarity_bulk_parrallel(source.view(), target.view());
        assert_eq!(array![0., 1., -1.], result);

        // check execution time.
        let n = 100000;
        let dim = 1024;
        let source = Array::random(dim, Uniform::new(0., 10.));
        let target = Array::random((n, dim), Uniform::new(0., 10.));

        let start = Instant::now();
        cosine_similarity_bulk_parrallel(source.view(), target.view());
        let duration = start.elapsed();
        assert!(duration.as_secs_f64() < 0.3); // 10만 개, 1024 dim -> 16 threads 에서 300 ms 넘으면 fail.
        println!("Execution time for multi thread: {:.4?}", duration);
    }
}
