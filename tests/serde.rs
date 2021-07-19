#![cfg(feature = "serde")]

#[macro_use]
extern crate std;

use bitarray::{BitArray, Hamming};
use hgg::{Hgg, HggLite};
use rand::{Rng, SeedableRng};
use space::{Knn, KnnInsert};

#[test]
fn serde_json_bincode_round_trip_lite() {
    let mut hgg: HggLite<Hamming, BitArray<32>, ()> = HggLite::default().insert_knn(100);

    // Use a PRNG with good statistical properties for generating 64-bit numbers.
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0);

    // Generate random keys.
    let keys: Vec<BitArray<32>> = (&mut rng)
        .sample_iter::<[u8; 32], _>(rand::distributions::Standard)
        .map(BitArray::new)
        .take(1 << 8)
        .collect();

    // Insert keys into HGG.
    for &key in &keys {
        hgg.insert(key, ());
    }

    // Find the 10 nearest neighbors to every node.
    let old_knns: Vec<Vec<_>> = keys.iter().map(|k| hgg.knn(k, 10)).collect();

    // Serialize and deserialize with both serde_json and bincode back-to-back.
    let hgg: HggLite<Hamming, BitArray<32>, ()> = serde_json::from_str(
        &serde_json::to_string(&hgg).expect("failed to serialize with serde_json"),
    )
    .expect("failed to deserialize with serde_json");
    let mut bdata = vec![];
    bincode::serialize_into(&mut bdata, &hgg).expect("failed to serialize with bincode");
    let hgg: HggLite<Hamming, BitArray<32>, ()> =
        bincode::deserialize_from(bdata.as_slice()).expect("failed to deserialize with bincode");

    // Find the 10 nearest neighbors to every node again.
    let new_knns: Vec<Vec<_>> = keys.iter().map(|k| hgg.knn(k, 10)).collect();

    // If they are exactly equal, everything worked as expected.
    assert_eq!(old_knns, new_knns);
}

#[test]
fn serde_json_bincode_round_trip_hgg() {
    let mut hgg: Hgg<Hamming, BitArray<32>, ()> = Hgg::default().insert_knn(100);

    // Use a PRNG with good statistical properties for generating 64-bit numbers.
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0);

    // Generate random keys.
    let keys: Vec<BitArray<32>> = (&mut rng)
        .sample_iter::<[u8; 32], _>(rand::distributions::Standard)
        .map(BitArray::new)
        .take(1 << 8)
        .collect();

    // Insert keys into HGG.
    for &key in &keys {
        hgg.insert(key, ());
    }

    // Find the 10 nearest neighbors to every node.
    let old_knns: Vec<Vec<_>> = keys.iter().map(|k| hgg.knn(k, 10)).collect();

    // Serialize and deserialize with both serde_json and bincode back-to-back.
    let hgg: Hgg<Hamming, BitArray<32>, ()> = serde_json::from_str(
        &serde_json::to_string(&hgg).expect("failed to serialize with serde_json"),
    )
    .expect("failed to deserialize with serde_json");
    let mut bdata = vec![];
    bincode::serialize_into(&mut bdata, &hgg).expect("failed to serialize with bincode");
    let hgg: Hgg<Hamming, BitArray<32>, ()> =
        bincode::deserialize_from(bdata.as_slice()).expect("failed to deserialize with bincode");

    // Find the 10 nearest neighbors to every node again.
    let new_knns: Vec<Vec<_>> = keys.iter().map(|k| hgg.knn(k, 10)).collect();

    // If they are exactly equal, everything worked as expected.
    assert_eq!(old_knns, new_knns);
}
