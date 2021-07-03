extern crate std;

use hrc::Hrc;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use serde::Serialize;
use space::{Bits256, Hamming, MetricPoint};
use std::{io::Read, time::Instant};

// Dataset sizes.
const HIGHEST_POWER_SEARCH_SPACE: u32 = 23;
const TRAIN_FEATURES: usize = 1 << 12;
const TEST_QUERRIES: usize = 1 << 15;

// We test with every k from 1..=HIGHEST_KNN to create the recall curve.
const HIGHEST_KNN: usize = 1 << 5;

// The higher the optimization, the better the recall curve, but the longer inserts take.
const INSERT_OPTIMIZATIONS: usize = 1 << 12;

#[derive(Debug, Serialize)]
struct Record {
    recall: f64,
    search_size: usize,
    knn: usize,
    num_queries: usize,
    seconds_per_query: f64,
    queries_per_second: f64,
}

struct Dataset {
    search: Vec<Hamming<Bits256>>,
    train: Vec<Hamming<Bits256>>,
    test: Vec<Hamming<Bits256>>,
}

fn retrieve_search_and_train(rng: &mut impl Rng) -> Dataset {
    let descriptor_size_bytes = 61;
    let search_descriptors = 1 << HIGHEST_POWER_SEARCH_SPACE;
    let test_descriptors = TEST_QUERRIES;
    let train_descriptors = TRAIN_FEATURES;
    let total_descriptors = search_descriptors + test_descriptors + train_descriptors;
    let filepath = "akaze";
    eprintln!(
        "Reading {} descriptors of size {} bytes from file \"{}\"...",
        total_descriptors, descriptor_size_bytes, filepath
    );
    let mut file = std::fs::File::open(filepath).expect("unable to open file");

    // Read the data.
    let mut v = vec![0u8; total_descriptors * descriptor_size_bytes];
    file.read_exact(&mut v)
        .expect("unable to read enough descriptors from the file; add more descriptors to file");

    eprintln!("Finished reading descriptors from file. Converting to keys.");

    // Convert the data into descriptors.
    let mut all: Vec<Hamming<Bits256>> = v
        .chunks_exact(descriptor_size_bytes)
        .map(|b| {
            let mut arr = [0; 32];
            for (d, &s) in arr.iter_mut().zip(b) {
                *d = s;
            }
            Hamming(Bits256(arr))
        })
        .collect();
    drop(v);

    eprintln!("Finished converting to keys. Shuffling dataset to avoid bias.");

    all.shuffle(rng);

    eprintln!(
        "Finished shuffling. Splitting dataset into {} search, {} train, and {} test descriptors",
        search_descriptors, train_descriptors, test_descriptors
    );

    let mut all = all.into_iter();

    Dataset {
        search: (&mut all).take(search_descriptors).collect(),
        train: (&mut all).take(train_descriptors).collect(),
        test: all.take(test_descriptors).collect(),
    }
}

fn main() {
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0);
    let mut hrc: Hrc<Hamming<Bits256>, (), u32> = Hrc::new().max_cluster_len(5);
    let Dataset {
        search,
        train,
        test,
    } = retrieve_search_and_train(&mut rng);

    let stdout = std::io::stdout();
    let mut csv_out = csv::Writer::from_writer(stdout.lock());
    for pow in 0..=HIGHEST_POWER_SEARCH_SPACE {
        let new_search_items = if pow == 0 {
            // If the pow is 0, we can't compute the lower bound properly.
            &search[0..1]
        } else {
            // In all other cases, take the range from the previous to the current pow.
            &search[1 << (pow - 1)..1 << pow]
        };
        // Insert keys into HRC.
        eprintln!("Inserting keys into HRC size {}", 1 << pow);
        let start_time = Instant::now();
        for &key in new_search_items {
            // Insert the key.
            hrc.insert(0, key, (), INSERT_OPTIMIZATIONS);
        }

        hrc.optimize_everything(0);
        hrc.train(train.iter());

        let end_time = Instant::now();
        eprintln!(
            "Finished inserting. Speed was {} inserts per second",
            new_search_items.len() as f64 / (end_time - start_time).as_secs_f64()
        );

        eprintln!("Histogram: {:?}", hrc.histogram());
        let average_neighbors = hrc.edges() as f64 * 2.0 / hrc.len() as f64;
        eprintln!("Average neighbors: {}", average_neighbors);

        eprintln!("Computing correct nearest neighbors for recall calculation");
        let correct_nn_distances: Vec<u32> = test
            .iter()
            .map(|query| {
                search[..1 << pow]
                    .iter()
                    .map(|key| query.distance(key) as u32)
                    .min()
                    .unwrap()
            })
            .collect();
        eprintln!("Finished computing the correct nearest neighbors");

        for knn in 1..=HIGHEST_KNN {
            eprintln!("doing size {} with knn {}", 1 << pow, knn);
            let start_time = Instant::now();
            let hrc_nn_distances: Vec<u32> = test
                .iter()
                .map(|query| hrc.search_knn_from(0, 0, query, knn).next().unwrap().1)
                // .map(|query| hrc.search_from(0, query).0)
                .collect();
            let end_time = Instant::now();
            let num_correct = correct_nn_distances
                .iter()
                .copied()
                .zip(hrc_nn_distances)
                .filter(|&(correct_distance, hrc_distance)| correct_distance == hrc_distance)
                .count();
            let recall = num_correct as f64 / test.len() as f64;
            let seconds_per_query = (end_time - start_time)
                .div_f64(test.len() as f64)
                .as_secs_f64();
            let queries_per_second = seconds_per_query.recip();
            csv_out
                .serialize(Record {
                    recall,
                    search_size: 1 << pow,
                    knn,
                    num_queries: test.len(),
                    seconds_per_query,
                    queries_per_second,
                })
                .expect("failed to serialize record");
            csv_out.flush().expect("failed to flush record");
            eprintln!("finished size {} with knn {}", 1 << pow, knn);
        }
    }
}
