extern crate std;

use bitvec::vec::BitVec;
use hrc::{HrcCore, LayerIndex};
use serde::Serialize;
use space::{Bits256, Hamming, MetricPoint};
use std::{io::Read, time::Instant};

const HIGHEST_POWER_SEARCH_SPACE: u32 = 17;
const NUM_SEARCH_QUERRIES: usize = 1 << 8;
const HIGHEST_POWER_CANDIDATES: u32 = 5;
const CREATION_CANDIDATES: usize = 1 << 8;

#[derive(Debug, Serialize)]
struct Record {
    recall: f64,
    search_size: usize,
    candidates_size: usize,
    num_queries: usize,
    layers: usize,
    seconds_per_query: f64,
    queries_per_second: f64,
}

fn retrieve_search_and_query() -> (Vec<Hamming<Bits256>>, Vec<Hamming<Bits256>>) {
    let descriptor_size_bytes = 61;
    let total_descriptors = 1 << HIGHEST_POWER_SEARCH_SPACE;
    let filepath = "akaze";
    let total_query_strings = NUM_SEARCH_QUERRIES;
    // Read in search space.
    eprintln!(
        "Reading {} search space descriptors of size {} bytes from file \"{}\"...",
        total_descriptors, descriptor_size_bytes, filepath
    );
    let mut file = std::fs::File::open(filepath).expect("unable to open file");
    let mut v = vec![0u8; total_descriptors * descriptor_size_bytes];
    file.read_exact(&mut v).expect(
        "unable to read enough search descriptors from the file; add more descriptors to file",
    );
    let search_space: Vec<Hamming<Bits256>> = v
        .chunks_exact(descriptor_size_bytes)
        .map(|b| {
            let mut arr = [0; 32];
            for (d, &s) in arr.iter_mut().zip(b) {
                *d = s;
            }
            Hamming(Bits256(arr))
        })
        .collect();
    eprintln!("Done.");

    // Read in query strings.
    eprintln!(
        "Reading {} query descriptors of size {} bytes from file \"{}\"...",
        total_query_strings, descriptor_size_bytes, filepath
    );
    let mut v = vec![0u8; total_query_strings * descriptor_size_bytes];
    file.read_exact(&mut v).expect(
        "unable to read enough search descriptors from the file; add more descriptors to file",
    );
    let query_strings: Vec<Hamming<Bits256>> = v
        .chunks_exact(descriptor_size_bytes)
        .map(|b| {
            let mut arr = [0; 32];
            for (d, &s) in arr.iter_mut().zip(b) {
                *d = s;
            }
            Hamming(Bits256(arr))
        })
        .collect();
    eprintln!("Done.");

    (search_space, query_strings)
}

fn main() {
    let mut hrc: HrcCore<Hamming<Bits256>, ()> = HrcCore::new()
        .max_cluster_len(5)
        .new_layer_threshold_clusters(5);

    let mut candidates = [(LayerIndex::empty(), !0); CREATION_CANDIDATES];
    let mut cluster_candidates = [(!0, !0); 1];
    let mut to_search = vec![];
    let mut searched = BitVec::new();

    // Generate random keys.
    let (keys, queries) = retrieve_search_and_query();

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
        eprintln!("Inserting keys into HRC");
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
            let seconds_per_query = (end_time - start_time)
                .div_f64(NUM_SEARCH_QUERRIES as f64)
                .as_secs_f64();
            let queries_per_second = seconds_per_query.recip();
            csv_out
                .serialize(Record {
                    recall,
                    candidates_size,
                    search_size: 1 << pow,
                    num_queries: NUM_SEARCH_QUERRIES,
                    layers: hrc.layers(),
                    seconds_per_query,
                    queries_per_second,
                })
                .expect("failed to serialize record");
        }
    }
}
