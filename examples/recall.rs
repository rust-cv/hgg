extern crate std;

use bitvec::vec::BitVec;
use hrc::{HrcCore, LayerIndex};
use rand::{Rng, SeedableRng};
use serde::Serialize;
use space::{Bits256, Hamming, MetricPoint};
use std::time::Instant;

const HIGHEST_POWER_SEARCH_SPACE: u32 = 12;
const NUM_SEARCH_QUERRIES: usize = 1 << 10;
const HIGHEST_POWER_CANDIDATES: u32 = 14;

#[derive(Debug, Serialize)]
struct Record {
    recall: f64,
    search_size: usize,
    candidates_size: usize,
    num_queries: usize,
    layers: usize,
    queries_per_second: f64,
}

fn main() {
    let mut hrc: HrcCore<Hamming<Bits256>, ()> = HrcCore::new()
        .max_cluster_len(5)
        .new_layer_threshold_clusters(5);

    let mut candidates = [(LayerIndex::empty(), !0); 4096];
    let mut cluster_candidates = [(!0, !0); 1];
    let mut to_search = vec![];
    let mut searched = BitVec::new();

    // Use a PRNG with good statistical properties for generating 64-bit numbers.
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0);

    // Generate random keys.
    let keys: Vec<Hamming<Bits256>> = (&mut rng)
        .sample_iter::<[u8; 32], _>(rand::distributions::Standard)
        .map(Bits256)
        .map(Hamming)
        .take(1 << HIGHEST_POWER_SEARCH_SPACE)
        .collect();

    // Generate random search queries.
    let queries: Vec<Hamming<Bits256>> = rng
        .sample_iter::<[u8; 32], _>(rand::distributions::Standard)
        .map(Bits256)
        .map(Hamming)
        .take(NUM_SEARCH_QUERRIES)
        .collect();

    let stdout = std::io::stdout();
    let mut csv_out = csv::Writer::from_writer(stdout.lock());
    for pow in 0..=HIGHEST_POWER_SEARCH_SPACE {
        let range = if pow == 0 {
            // If the pow is 0, we can't compute the lower bound properly.
            0..1
        } else {
            // In all other cases, take the range from the previous to the current pow.
            1 << (pow - 1)..1 << pow
        };
        // Insert keys into HRC.
        for &key in &keys[range] {
            hrc.insert(
                key,
                (),
                &mut candidates,
                &mut cluster_candidates,
                &mut to_search,
                &mut searched,
            );
        }

        let correct_nearest: Vec<Hamming<Bits256>> = queries
            .iter()
            .map(|query| {
                keys[..1 << pow]
                    .iter()
                    .copied()
                    .min_by_key(|key| query.distance(key))
                    .unwrap()
            })
            .collect();

        for pow_candidates in 0..=HIGHEST_POWER_CANDIDATES {
            eprintln!(
                "doing size {} with candidates {}",
                1 << pow,
                1 << pow_candidates
            );
            let candidates_size = 1 << pow_candidates;
            let mut candidates = vec![(LayerIndex::empty(), !0); candidates_size];
            let start_time = Instant::now();
            let search_bests: Vec<LayerIndex> = queries
                .iter()
                .map(|query| {
                    hrc.search(query, &mut candidates[..], &mut to_search, &mut searched);
                    candidates[0].0
                })
                .collect();
            let end_time = Instant::now();
            let num_correct = search_bests
                .iter()
                .zip(correct_nearest.iter())
                .filter(|&(&searched_ix, &correct)| *hrc.get(searched_ix).unwrap().0 == correct)
                .count();
            let recall = num_correct as f64 / NUM_SEARCH_QUERRIES as f64;
            let queries_per_second = (end_time - start_time)
                .div_f64(NUM_SEARCH_QUERRIES as f64)
                .as_secs_f64()
                .recip();
            csv_out
                .serialize(Record {
                    recall,
                    candidates_size,
                    search_size: 1 << pow,
                    num_queries: NUM_SEARCH_QUERRIES,
                    layers: hrc.layers(),
                    queries_per_second,
                })
                .expect("failed to serialize record");
        }
    }
}
