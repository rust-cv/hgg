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

impl<K, V, const N: usize> HrcLayer<K, V, N>
where
    K: MetricPoint,
{
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

        // Set all the initial clusters as searched, since they were already added to `to_search`.
        for &cluster_ix in to_search.iter() {
            searched.set(cluster_ix as usize, true);
        }

        // Continuously attempt to search clusters.
        while let Some(cluster_ix) = to_search.pop() {
            let mut must_beat = candidates.last().unwrap().1;
            let cluster = &self.clusters[cluster_ix as usize];
            let center_distance = cluster.center_distance(query);
            let keys = cluster.potential_closer_items(center_distance, must_beat);

            // If this cluster has nothing to offer us, we should just skip it and not add its neighbors.
            if keys.is_empty() {
                continue;
            }

            // Search for a key that beats what we currently have.
            for (ix, key) in keys.iter().enumerate() {
                let distance = query.distance(key);
                // If we beat the must_beat, which is the worst candidate we have.
                if distance < must_beat {
                    // Add the candidate to the candidates.
                    add_candidate(
                        candidates,
                        (
                            LayerIndex {
                                cluster: cluster_ix,
                                item: ix as u32,
                            },
                            distance,
                        ),
                    );

                    // Update the must_beat.
                    must_beat = candidates.last().unwrap().1;
                }
            }

            // Set neighbor clusters as searched and add them to the to_search pool only if they weren't already searched.
            for &cluster_ix in &cluster.neighbors {
                if *searched.get(cluster_ix as usize).unwrap() {
                    searched.set(cluster_ix as usize, true);
                    to_search.push(cluster_ix);
                }
            }
        }

        candidates.partition_point(|n| !n.0.is_empty())
    }
}

/// Must contain at least one item. The first item is the cluster center.
struct HrcCluster<K, V, const N: usize> {
    neighbors: Vec<u32>,
    keys: Vec<K>,
    values: Vec<V>,
    /// This keeps track of the distances of all key-value pairs, and this is always ordered from least to greatest.
    distances: Vec<u32>,
}

impl<K, V, const N: usize> HrcCluster<K, V, N>
where
    K: MetricPoint,
{
    /// Returns (index, distance) to closest member (biased towards beginning of vector).
    fn closest_to(&self, key: &K) -> (usize, u32) {
        self.keys
            .iter()
            .map(|ck| key.distance(ck))
            .enumerate()
            .reduce(|(aix, ad), (bix, bd)| if bd < ad { (bix, bd) } else { (aix, ad) })
            .unwrap()
    }

    /// Returns the slice of keys and values that could beat a given distance.
    fn potential_closer_items(&self, center_distance: u32, must_beat: u32) -> &[K] {
        let minimum_inclusive = (center_distance + 1).saturating_sub(must_beat);
        let maximum_exclusive = center_distance.saturating_add(must_beat);
        let begin = self.distances.partition_point(|&d| d < minimum_inclusive);
        let end = self.distances.partition_point(|&d| d < maximum_exclusive);
        &self.keys[begin..end]
    }

    /// Computes the distance to the center of the cluster.
    fn center_distance(&self, key: &K) -> u32 {
        self.keys[0].distance(key)
    }

    /// Inserts an item into the cluster.
    fn insert(&mut self, key: K, value: V) {
        // Compute the distance of the key to the center.
        let center_distance = self.center_distance(&key);
        // Find the location to insert it using the distances vec.
        // Insert it right after the point at which its equal, so things which are added later are added further towards the end.
        // One reason why this might be necessary is to avoid replacing the cluster center.
        let position = self.distances.partition_point(|&d| d <= center_distance);
        // Insert the point to the vectors.
        self.keys.insert(position, key);
        self.values.insert(position, value);
        self.distances.insert(position, center_distance);
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
        let pos = candidates.partition_point(|other| other.1 <= c.1);
        candidates[pos..].rotate_right(1);
        candidates[pos] = c;
    }
}
