#![no_std]
extern crate alloc;

use alloc::vec::Vec;
use bitvec::vec::BitVec;
use space::MetricPoint;

struct HrcCore<K, V, const N: usize> {
    /// Each layer maps keys (representing the cluster center) to the index in the next layer (as a u32).
    layers: Vec<HrcLayer<K, u32, N>>,
    /// The bottom layer maps keys directly into values.
    values: HrcLayer<K, V, N>,
}

struct HrcLayer<K, V, const N: usize> {
    clusters: Vec<HrcCluster<K, V, N>>,
}

impl<K, V, const N: usize> HrcLayer<K, V, N> {
    /// Searches the layer, placing the best candidate items into the slice from best to worst.
    ///
    /// `to_search` must contain the starting clusters to begin search from.
    ///
    /// Any unoccupied spots will be filled with an empty LayerIndex, which contains only `!0`.
    ///
    /// Returns the number of candidates populated.
    fn search(
        &self,
        query: &K,
        candidates: &mut [(LayerIndex, u32)],
        to_search: &mut Vec<u32>,
        searched: &mut BitVec,
    ) -> usize {
        // Initialize the mutable inputs to the correct values.
        candidates.fill((LayerIndex::empty(), !0));
        searched.clear();
        searched.resize(self.clusters.len(), false);

        // Continuously attempt to search clusters.
        while let Some(cluster) = to_search.pop() {
            // TODO: Maybe use epochs to understand shortest path distance to reason about distance?
            self.clusters[cluster].neighbors.iter().copied()
        }

        candidates.partition_point(|n| !n.0.is_empty())
    }
}

/// Must contain at least one item. The first item is the cluster center.
struct HrcCluster<K, V, const N: usize> {
    neighbors: Vec<u32>,
    keys: Vec<K>,
    values: Vec<V>,
}

impl<K, V, const N: usize> HrcCluster<K, V, N>
where
    K: MetricPoint,
{
    // Returns (index, distance) to closest member (biased towards beginning of vector).
    fn closest_to(&self, key: K) -> (usize, u32) {
        self.keys
            .iter()
            .map(|ck| key.distance(ck))
            .enumerate()
            .reduce(|(aix, ad), (bix, bd)| if bd < ad { (bix, bd) } else { (aix, ad) })
            .unwrap()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
struct LayerIndex {
    cluster: u32,
    item: u32,
}

impl LayerIndex {
    fn empty() -> Self {
        LayerIndex {
            cluster: !0,
            item: !0,
        }
    }

    fn is_empty(&self) -> bool {
        self.cluster == !0
    }
}

fn add_candidate(candidates: &mut [(LayerIndex, u32)], c: (LayerIndex, u32)) {
    if c.1 < candidates.last().unwrap().1 {
        let pos = candidates.partition_point(|other| other.1 < c.1);
        candidates[pos..].rotate_right(1);
        candidates[pos] = c;
    }
}
