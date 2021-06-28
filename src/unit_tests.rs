extern crate std;

// fn test_layer() -> HrcLayer<Hamming<u8>, &'static str> {
//     let mut first_cluster = HrcCluster::new(Hamming(0b1101));
//     first_cluster.insert(Hamming(0b1101), "a");
//     first_cluster.insert(Hamming(0b1001), "b");
//     first_cluster.insert(Hamming(0b1100), "c");
//     first_cluster.insert(Hamming(0b0101), "d");
//     first_cluster.neighbors.push(1);

//     let mut second_cluster = HrcCluster::new(Hamming(0b0010));
//     second_cluster.insert(Hamming(0b0010), "q");
//     second_cluster.insert(Hamming(0b0110), "r");
//     second_cluster.insert(Hamming(0b0000), "s");
//     second_cluster.insert(Hamming(0b0011), "t");
//     second_cluster.neighbors.push(0);
//     second_cluster.neighbors.push(2);

//     let mut third_cluster = HrcCluster::new(Hamming(0b11110010));
//     third_cluster.insert(Hamming(0b11110010), "w");
//     third_cluster.insert(Hamming(0b11110110), "x");
//     third_cluster.insert(Hamming(0b11110000), "y");
//     third_cluster.insert(Hamming(0b11110011), "z");
//     third_cluster.neighbors.push(1);

//     HrcLayer {
//         clusters: vec![first_cluster, second_cluster, third_cluster],
//     }
// }

// #[test]
// fn test_layer_search() {
//     let layer = test_layer();

//     let mut candidates = [(LayerIndex::empty(), !0); 5];
//     let mut to_search = vec![0];
//     let mut searched = BitVec::new();

//     let num_found = layer.search(
//         &Hamming(0b0101),
//         &mut candidates,
//         &mut to_search,
//         &mut searched,
//     );
//     assert_eq!(num_found, 5);
//     assert_eq!(
//         &candidates[..],
//         &[
//             (
//                 LayerIndex {
//                     cluster: 0,
//                     item: 3
//                 },
//                 0
//             ),
//             (
//                 LayerIndex {
//                     cluster: 0,
//                     item: 0
//                 },
//                 1
//             ),
//             (
//                 LayerIndex {
//                     cluster: 0,
//                     item: 1
//                 },
//                 2
//             ),
//             (
//                 LayerIndex {
//                     cluster: 0,
//                     item: 2
//                 },
//                 2
//             ),
//             (
//                 LayerIndex {
//                     cluster: 1,
//                     item: 1
//                 },
//                 2
//             )
//         ]
//     );
//     assert_eq!(
//         layer.get(candidates[0].0).unwrap(),
//         (&Hamming(0b0101), &"d")
//     );
// }

// #[test]
// fn test_cluster_split() {
//     let mut hrc: Hrc<Hamming<u8>, ()> = Hrc::new().max_cluster_len(3);

//     // Insert a series of keys.
//     for &key in &[0b0000_0000, 0b1111_1111, 0b0000_1111, 0b1111_0000] {
//         hrc.insert(Hamming(key), ());
//     }

//     assert_eq!(
//         hrc,
//         HrcCore {
//             layers: vec![],
//             values: HrcLayer {
//                 clusters: vec![
//                     HrcCluster {
//                         key: Hamming(0),
//                         neighbors: vec![1],
//                         keys: vec![Hamming(0), Hamming(15), Hamming(240)],
//                         values: vec![(), (), ()],
//                         distances: vec![0, 4, 4],
//                     },
//                     HrcCluster {
//                         key: Hamming(255),
//                         neighbors: vec![0],
//                         keys: vec![Hamming(255)],
//                         values: vec![()],
//                         distances: vec![0],
//                     },
//                 ],
//             },
//             len: 3,
//             max_cluster_len: 3,
//             new_layer_threshold_clusters: 2,
//         }
//     );

//     // Insert a series of keys.
//     for &key in &[
//         0b1100_0000,
//         0b0011_0000,
//         0b0011_1111,
//         0b1100_1111,
//         0b1110_1111,
//         0b1010_1111,
//         0b1010_1011,
//         0b1010_1010,
//     ] {
//         hrc.insert(
//             Hamming(key),
//             (),
//             &mut candidates,
//             &mut cluster_candidates,
//             &mut to_search,
//             &mut searched,
//         );
//     }

//     assert_eq!(
//         hrc,
//         HrcCore {
//             layers: vec![HrcLayer {
//                 clusters: vec![
//                     HrcCluster {
//                         key: Hamming(0),
//                         neighbors: vec![1],
//                         keys: vec![Hamming(0), Hamming(240)],
//                         values: vec![0, 2],
//                         distances: vec![0, 4]
//                     },
//                     HrcCluster {
//                         key: Hamming(255),
//                         neighbors: vec![0],
//                         keys: vec![Hamming(255), Hamming(207), Hamming(171)],
//                         values: vec![1, 3, 4],
//                         distances: vec![0, 2, 3]
//                     }
//                 ]
//             }],
//             values: HrcLayer {
//                 clusters: vec![
//                     HrcCluster {
//                         key: Hamming(0),
//                         neighbors: vec![1, 2, 3, 4],
//                         keys: vec![Hamming(0), Hamming(192)],
//                         values: vec![(), ()],
//                         distances: vec![0, 2]
//                     },
//                     HrcCluster {
//                         key: Hamming(255),
//                         neighbors: vec![0, 2, 3, 3, 4],
//                         keys: vec![Hamming(255), Hamming(239), Hamming(63)],
//                         values: vec![(), (), ()],
//                         distances: vec![0, 1, 2]
//                     },
//                     HrcCluster {
//                         key: Hamming(240),
//                         neighbors: vec![1, 0, 3],
//                         keys: vec![Hamming(240), Hamming(48)],
//                         values: vec![(), ()],
//                         distances: vec![0, 2]
//                     },
//                     HrcCluster {
//                         key: Hamming(207),
//                         neighbors: vec![1, 1, 2, 0, 4, 4],
//                         keys: vec![Hamming(207), Hamming(15)],
//                         values: vec![(), ()],
//                         distances: vec![0, 2]
//                     },
//                     HrcCluster {
//                         key: Hamming(171),
//                         neighbors: vec![3, 1, 3, 0],
//                         keys: vec![Hamming(171), Hamming(175), Hamming(170)],
//                         values: vec![(), (), ()],
//                         distances: vec![0, 1, 1]
//                     }
//                 ]
//             },
//             len: 11,
//             max_cluster_len: 3,
//             new_layer_threshold_clusters: 2
//         }
//     );

//     // Insert a series of keys.
//     for &key in &[0b1110_1110, 0b1110_1111, 0b1110_0111, 0b1110_0110] {
//         hrc.insert(
//             Hamming(key),
//             (),
//             &mut candidates,
//             &mut cluster_candidates,
//             &mut to_search,
//             &mut searched,
//         );
//     }

//     assert_eq!(
//         hrc,
//         HrcCore {
//             layers: vec![
//                 HrcLayer {
//                     clusters: vec![
//                         HrcCluster {
//                             key: Hamming(0),
//                             neighbors: vec![1, 2],
//                             keys: vec![Hamming(0), Hamming(240)],
//                             values: vec![0, 2],
//                             distances: vec![0, 4]
//                         },
//                         HrcCluster {
//                             key: Hamming(255),
//                             neighbors: vec![0, 2],
//                             keys: vec![Hamming(255), Hamming(207), Hamming(238)],
//                             values: vec![1, 3, 5],
//                             distances: vec![0, 2, 2]
//                         },
//                         HrcCluster {
//                             key: Hamming(171),
//                             neighbors: vec![1, 0],
//                             keys: vec![Hamming(171)],
//                             values: vec![4],
//                             distances: vec![0]
//                         }
//                     ]
//                 },
//                 HrcLayer {
//                     clusters: vec![HrcCluster {
//                         key: Hamming(0),
//                         neighbors: vec![],
//                         keys: vec![Hamming(0), Hamming(171), Hamming(255)],
//                         values: vec![0, 2, 1],
//                         distances: vec![0, 5, 8]
//                     }]
//                 }
//             ],
//             values: HrcLayer {
//                 clusters: vec![
//                     HrcCluster {
//                         key: Hamming(0),
//                         neighbors: vec![1, 2, 3, 4],
//                         keys: vec![Hamming(0), Hamming(192)],
//                         values: vec![(), ()],
//                         distances: vec![0, 2]
//                     },
//                     HrcCluster {
//                         key: Hamming(255),
//                         neighbors: vec![0, 2, 3, 3, 4, 5],
//                         keys: vec![Hamming(255), Hamming(239), Hamming(63)],
//                         values: vec![(), (), ()],
//                         distances: vec![0, 1, 2]
//                     },
//                     HrcCluster {
//                         key: Hamming(240),
//                         neighbors: vec![1, 0, 3],
//                         keys: vec![Hamming(240), Hamming(48)],
//                         values: vec![(), ()],
//                         distances: vec![0, 2]
//                     },
//                     HrcCluster {
//                         key: Hamming(207),
//                         neighbors: vec![1, 1, 2, 0, 4, 4, 5],
//                         keys: vec![Hamming(207), Hamming(239), Hamming(15)],
//                         values: vec![(), (), ()],
//                         distances: vec![0, 1, 2]
//                     },
//                     HrcCluster {
//                         key: Hamming(171),
//                         neighbors: vec![3, 1, 3, 0, 5, 5],
//                         keys: vec![Hamming(171), Hamming(175), Hamming(170)],
//                         values: vec![(), (), ()],
//                         distances: vec![0, 1, 1]
//                     },
//                     HrcCluster {
//                         key: Hamming(238),
//                         neighbors: vec![1, 3, 4, 4],
//                         keys: vec![Hamming(238), Hamming(230), Hamming(231)],
//                         values: vec![(), (), ()],
//                         distances: vec![0, 1, 2]
//                     }
//                 ]
//             },
//             len: 15,
//             max_cluster_len: 3,
//             new_layer_threshold_clusters: 2
//         }
//     );
// }

#[cfg(feature = "stats")]
mod stats {
    use super::std::eprintln;
    use crate::Hrc;
    use alloc::vec::Vec;
    use rand::{Rng, SeedableRng};
    use space::{Bits256, Hamming};

    #[test]
    fn random_insertion_stats() {
        const NUM_TRAINING_PAIRS: usize = 1 << 16;
        let mut hrc: Hrc<Hamming<Bits256>, ()> = Hrc::new().max_cluster_len(5);

        // Use a PRNG with good statistical properties for generating 64-bit numbers.
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0);

        // Generate random keys.
        let keys: Vec<Hamming<Bits256>> = (&mut rng)
            .sample_iter::<[u8; 32], _>(rand::distributions::Standard)
            .map(Bits256)
            .map(Hamming)
            .take(1 << 12)
            .collect();

        // Insert keys into HRC.
        for (ix, &key) in keys.iter().enumerate() {
            if ix % 1000 == 0 {
                eprintln!("Inserting {}", ix);
                // eprintln!("Stats: {:?}", hrc.stats());
            }
            hrc.insert(0, key, (), 32);
        }

        eprintln!("Trimming graph");
        hrc.freshen_all();

        eprintln!("Training with {} random node pairs", NUM_TRAINING_PAIRS);
        for (a, b) in (0..NUM_TRAINING_PAIRS)
            .map(|_| (rng.gen_range(0..keys.len()), rng.gen_range(0..keys.len())))
        {
            hrc.optimize_connection(0, a, b);
        }
        eprintln!("Histogram: {:?}", hrc.histogram());

        for (ix, key) in keys.iter().enumerate() {
            if ix % 100 == 0 {
                eprintln!("Searching {}", ix);
            }
            // Search each key.
            let (_, distance) = hrc.search(0, key).unwrap();
            // Make sure that the best result is this key.
            assert_eq!(distance, 0);
        }

        // for layer in &hrc.layers {
        //     // Make sure that all the u32 index values in this layer are unique across the whole layer (or there is a bug).
        //     assert_eq!(
        //         layer
        //             .clusters
        //             .iter()
        //             .flat_map(|cluster| cluster.values.iter().copied())
        //             .unique()
        //             .count(),
        //         layer
        //             .clusters
        //             .iter()
        //             .flat_map(|cluster| cluster.values.iter().copied())
        //             .count()
        //     );
        // }
    }
}
