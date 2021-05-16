use crate::{HrcCluster, HrcLayer, LayerIndex};
use alloc::vec;
use bitvec::vec::BitVec;
use space::Hamming;

#[test]
fn test_layer_search() {
    let layer: HrcLayer<Hamming<u8>, &'static str> = HrcLayer {
        clusters: vec![
            HrcCluster {
                neighbors: vec![1],
                ..HrcCluster::new(Hamming(0b1101), "first")
            },
            HrcCluster {
                neighbors: vec![0],
                ..HrcCluster::new(Hamming(0b0011), "second")
            },
        ],
    };

    let mut candidates = [(LayerIndex::empty(), !0); 10];
    let mut to_search = vec![0];
    let mut searched = BitVec::new();

    let num_found = layer.search(
        &Hamming(0b0101),
        &mut candidates,
        &mut to_search,
        &mut searched,
    );
    assert_eq!(num_found, 2);
    assert_eq!(
        &candidates[..2],
        &[
            (
                LayerIndex {
                    cluster: 0,
                    item: 0
                },
                1
            ),
            (
                LayerIndex {
                    cluster: 1,
                    item: 0
                },
                2
            )
        ]
    );
}
