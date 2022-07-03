extern crate std;

use bitarray::{BitArray, Hamming};
use hgg::Hgg;
use ndarray::{array, Array2, ArrayView1};
use rand::{seq::SliceRandom, Rng, SeedableRng};
use serde::Serialize;
use space::{Knn, KnnInsert, KnnPoints, Metric};
use std::{io::Read, time::Instant};

struct Euclidean;

impl Metric<ArrayView1<'_, f64>> for Euclidean {
    type Unit = u64;

    fn distance(&self, a: &ArrayView1<'_, f64>, b: &ArrayView1<'_, f64>) -> Self::Unit {
        let delta = a - b;
        let distance = delta.dot(&delta).sqrt();
        debug_assert!(!distance.is_nan());
        distance.to_bits()
    }
}

fn main() {
    let a = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [3.0, 2.0]];

    let mut hgg = Hgg::new(Euclidean);

    for row in a.rows() {
        println!("{:?}", row.as_slice().unwrap());
        hgg.insert(row, ());
    }

    for row in a.rows() {
        println!(
            "Closest 3 rows (closest to furthest) to {:?}:",
            row.as_slice().unwrap()
        );
        for close_row in hgg.knn_points(&row, 3) {
            println!("{:?}", close_row.1.as_slice().unwrap());
        }
    }
}
