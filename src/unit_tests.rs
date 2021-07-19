extern crate std;

use crate::Hgg;
use alloc::vec::Vec;
use bitarray::{BitArray, Hamming};
use rand::{Rng, SeedableRng};
use space::{Knn, KnnInsert};
use std::eprintln;

#[test]
fn random_insertion_stats() {
    let mut hgg: Hgg<Hamming, BitArray<32>, ()> = Hgg::default().insert_knn(100);

    // Use a PRNG with good statistical properties for generating 64-bit numbers.
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0);

    // Generate random keys.
    let keys: Vec<BitArray<32>> = (&mut rng)
        .sample_iter::<[u8; 32], _>(rand::distributions::Standard)
        .map(BitArray::new)
        .take(1 << 12)
        .collect();

    // Insert keys into HGG.
    for (ix, &key) in keys.iter().enumerate() {
        if ix % 1000 == 0 {
            eprintln!("Inserting {}", ix);
        }
        hgg.insert(key, ());
    }

    eprintln!("Histogram: {:?}", hgg.histogram_neighbors());

    for (ix, key) in keys.iter().enumerate() {
        if ix % 100 == 0 {
            eprintln!("Searching {}", ix);
        }
        // Search each key.
        let distance = hgg.knn(key, 5)[0].distance;
        // Make sure that the best result is this key.
        assert_eq!(distance, 0);
    }
}
