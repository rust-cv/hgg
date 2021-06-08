use bitvec::vec::BitVec;
use hrc::{HrcCore, LayerIndex};
use rand::{Rng, SeedableRng};
use space::{Bits256, Hamming};

fn main() {
    let mut hrc: HrcCore<Hamming<Bits256>, ()> = HrcCore::new()
        .max_cluster_len(5)
        .new_layer_threshold_clusters(5);

    let mut candidates = [(LayerIndex::empty(), !0); 1024];
    let mut cluster_candidates = [(!0, !0); 1];
    let mut to_search = vec![];
    let mut searched = BitVec::new();

    // Use a PRNG with good statistical properties for generating 64-bit numbers.
    let rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0);

    // Generate random keys.
    let keys: Vec<Hamming<Bits256>> = rng
        .sample_iter::<[u8; 32], _>(rand::distributions::Standard)
        .map(Bits256)
        .map(Hamming)
        .take(1 << 12)
        .collect();

    // Insert keys into HRC.
    for (ix, &key) in keys.iter().enumerate() {
        if ix % 1000 == 0 {
            eprintln!("Inserting {}", ix);
            eprintln!("Stats: {:?}", hrc.stats());
        }
        hrc.insert(
            key,
            (),
            &mut candidates,
            &mut cluster_candidates,
            &mut to_search,
            &mut searched,
        );
    }

    for (ix, key) in keys.iter().enumerate() {
        if ix % 100 == 0 {
            eprintln!("Searching {}", ix);
        }
        // Search each key.
        hrc.search(key, &mut candidates, &mut to_search, &mut searched);
        // Make sure that the best result is this key.
        assert_eq!(hrc.get(candidates[0].0).unwrap().0, key);
    }
}
