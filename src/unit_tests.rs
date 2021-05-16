use crate::{HrcCluster, HrcLayer, LayerIndex};
use alloc::vec;
use bitvec::vec::BitVec;
use space::Hamming;

fn test_layer() -> HrcLayer<Hamming<u8>, &'static str> {
    let mut first_cluster = HrcCluster::new(Hamming(0b1101), "a");
    first_cluster.insert(Hamming(0b1001), "b");
    first_cluster.insert(Hamming(0b1100), "c");
    first_cluster.insert(Hamming(0b0101), "d");
    first_cluster.neighbors.push(1);

    let mut second_cluster = HrcCluster::new(Hamming(0b0010), "q");
    second_cluster.insert(Hamming(0b0110), "r");
    second_cluster.insert(Hamming(0b0000), "s");
    second_cluster.insert(Hamming(0b0011), "t");
    second_cluster.neighbors.push(0);
    second_cluster.neighbors.push(2);

    let mut third_cluster = HrcCluster::new(Hamming(0b11110010), "w");
    third_cluster.insert(Hamming(0b11110110), "x");
    third_cluster.insert(Hamming(0b11110000), "y");
    third_cluster.insert(Hamming(0b11110011), "z");
    third_cluster.neighbors.push(1);

    HrcLayer {
        clusters: vec![first_cluster, second_cluster, third_cluster],
    }
}

#[test]
fn test_layer_search() {
    let layer = test_layer();

    let mut candidates = [(LayerIndex::empty(), !0); 5];
    let mut to_search = vec![0];
    let mut searched = BitVec::new();

    let num_found = layer.search(
        &Hamming(0b0101),
        &mut candidates,
        &mut to_search,
        &mut searched,
    );
    assert_eq!(num_found, 5);
    assert_eq!(
        &candidates[..],
        &[
            (
                LayerIndex {
                    cluster: 0,
                    item: 3
                },
                0
            ),
            (
                LayerIndex {
                    cluster: 0,
                    item: 0
                },
                1
            ),
            (
                LayerIndex {
                    cluster: 0,
                    item: 1
                },
                2
            ),
            (
                LayerIndex {
                    cluster: 0,
                    item: 2
                },
                2
            ),
            (
                LayerIndex {
                    cluster: 1,
                    item: 1
                },
                2
            )
        ]
    );
    assert_eq!(*layer.get(candidates[0].0).unwrap(), "d");
}
