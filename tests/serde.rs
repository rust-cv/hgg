#![cfg(feature = "serde")]

#[macro_use]
extern crate std;

use hgg::Hgg;
use rand::{Rng, SeedableRng};
use space::{Bits256, Knn};

#[test]
fn serde_json_bincode_round_trip() {
    let mut hgg: Hgg<Bits256, ()> = Hgg::new().insert_knn(100);

    // Use a PRNG with good statistical properties for generating 64-bit numbers.
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0);

    // Generate random keys.
    let keys: Vec<Bits256> = (&mut rng)
        .sample_iter::<[u8; 32], _>(rand::distributions::Standard)
        .map(Bits256)
        .take(1 << 6)
        .collect();

    // Insert keys into HGG.
    for (ix, &key) in keys.iter().enumerate() {
        if ix % 1000 == 0 {
            eprintln!("Inserting {}", ix);
        }
        hgg.insert(key, ());
    }

    // Find the 10 nearest neighbors to every node.
    let old_knns: Vec<Vec<_>> = keys.iter().map(|k| hgg.knn(k, 10)).collect();

    // Serialize and deserialize with both serde_json and bincode back-to-back.
    let hgg: Hgg<Bits256, ()> = serde_json::from_str(
        &serde_json::to_string(&hgg).expect("failed to serialize with serde_json"),
    )
    .expect("failed to deserialize with serde_json");
    let mut bdata = vec![];
    bincode::serialize_into(&mut bdata, &hgg).expect("failed to serialize with bincode");
    let hgg: Hgg<Bits256, ()> =
        bincode::deserialize_from(bdata.as_slice()).expect("failed to deserialize with bincode");

    // Find the 10 nearest neighbors to every node again.
    let new_knns: Vec<Vec<_>> = keys.iter().map(|k| hgg.knn(k, 10)).collect();

    // If they are exactly equal, everything worked as expected.
    assert_eq!(old_knns, new_knns);
}
