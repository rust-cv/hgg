#![no_std]
extern crate alloc;

#[cfg(test)]
mod unit_tests;

use alloc::vec;
use alloc::vec::Vec;
use bitvec::vec::BitVec;
use itertools::Itertools;
use space::MetricPoint;

#[derive(Clone, Debug)]
pub struct Stats {
    pub layer_cluster_neigbors_histogram: Vec<Vec<(usize, usize)>>,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct HrcCore<K, V> {
    /// Each layer maps keys (representing the cluster center) to the index in the next layer (as a u32).
    layers: Vec<HrcLayer<K, u32>>,
    /// The bottom layer maps keys directly into values.
    values: HrcLayer<K, V>,
    /// The number of items stored in this HRC.
    len: usize,
    /// Clusters with more items than this are split apart.
    pub max_cluster_len: usize,
    /// Number of clusters before a new layer is created.
    pub new_layer_threshold_clusters: usize,
}

impl<K, V> HrcCore<K, V> {
    /// Creates a new [`HrcCore`]. It will be empty and begin with default settings.
    pub fn new() -> Self {
        Self {
            layers: Default::default(),
            values: Default::default(),
            len: 0,
            max_cluster_len: 1024,
            new_layer_threshold_clusters: 1024,
        }
    }

    /// Retrieves the value from a [`LayerIndex`].
    pub fn get(&self, ix: LayerIndex) -> Option<(&K, &V)> {
        self.values.get(ix)
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Gets a vector over the layers (`Vec<Vec<usize>>`).
    /// Each layer is a vector of clusters (`Vec<usize>`), where each `usize` corresponds
    /// to a particular cluster's number of neighbors.
    pub fn layer_cluster_neighbors(&self) -> Vec<Vec<usize>> {
        Some(
            self.values
                .clusters
                .iter()
                .map(|cluster| cluster.neighbors.len())
                .collect(),
        )
        .into_iter()
        .chain(self.layers.iter().map(|layer| {
            layer
                .clusters
                .iter()
                .map(|cluster| cluster.neighbors.len())
                .collect()
        }))
        .collect()
    }

    /// Gets a vector over the layers (`Vec<Vec<(usize, usize)>>`).
    /// Each layer is a histogram over cluster neighbor counts (`Vec<(usize, usize)>`).
    /// The histogram is sorted from the lowest to highest neighbor count.
    /// Each neighbor count has an associated number of times it occurs in the layer.
    /// The tuple is formatted as `(neighbor count, occurences)`.
    pub fn layer_cluster_neighbors_histogram(&self) -> Vec<Vec<(usize, usize)>> {
        let layer_cluster_neighbors = self.layer_cluster_neighbors();
        layer_cluster_neighbors
            .into_iter()
            .map(|layer| {
                let mut histogram = vec![];
                for count in layer {
                    match histogram.binary_search_by_key(&count, |&(count, _)| count) {
                        Ok(pos) => histogram[pos].1 += 1,
                        Err(pos) => histogram.insert(pos, (count, 1)),
                    }
                }
                histogram
            })
            .collect()
    }

    pub fn stats(&self) -> Stats {
        Stats {
            layer_cluster_neigbors_histogram: self.layer_cluster_neighbors_histogram(),
        }
    }
}

impl<K, V> HrcCore<K, V>
where
    K: MetricPoint + Clone,
{
    /// Searches down to the bottom layer, placing the best candidate items into the slice from best to worst.
    ///
    /// `to_search` and `searched` will get cleared and used during the search.
    ///
    /// Any unoccupied spots will be filled with an empty LayerIndex, which contains only `!0`.
    ///
    /// Returns the number of candidates populated
    pub fn search(
        &self,
        query: &K,
        candidates: &mut [(LayerIndex, u32)],
        to_search: &mut Vec<u32>,
        searched: &mut BitVec,
    ) -> usize {
        // The below assumes that there is at least one cluster on some level, so exit if there are none.
        if self.values.clusters.is_empty() {
            candidates.fill((LayerIndex::empty(), !0));
            return 0;
        }

        // This will leave us with the candidates to search in the values layer in to_search.
        self.search_to_layer(query, 0, candidates, to_search, searched);

        // Now the to_search should contain value layer cluster IDs, so use them to search the value layer.
        self.values.search(query, candidates, to_search, searched)
    }

    /// Searches down to the chosen layer, placing the best candidate items into the slice from best to worst.
    ///
    /// `to_search` and `searched` will get cleared and used during the search.
    ///
    /// Any unoccupied spots will be filled with an empty LayerIndex, which contains only `!0`.
    ///
    /// `to_search` will get populated with the candidate clusters from the previous layer.
    ///
    /// Returns the number of candidates populated
    pub fn search_to_layer(
        &self,
        query: &K,
        layer: usize,
        candidates: &mut [(LayerIndex, u32)],
        to_search: &mut Vec<u32>,
        searched: &mut BitVec,
    ) -> usize {
        // We need to initialize to_search to only pull in the first cluster from the highest layer.
        to_search.clear();
        to_search.push(0);

        let mut found = 0;

        // Go through each layer from the highest to the lowest.
        for layer in self.layers[layer..].iter().rev() {
            // Search this layer for the best candidates.
            found = layer.search(query, candidates, to_search, searched);
            // The values from this layer are cluster IDs from the next layer, use those to populate the to_search.
            to_search.extend(
                candidates[..found]
                    .iter()
                    .map(|&(ix, _)| *layer.get(ix).unwrap().1),
            );
        }

        found
    }

    /// Inserts an item into the HRC.
    #[allow(clippy::too_many_arguments)]
    pub fn insert(
        &mut self,
        key: K,
        value: V,
        candidates: &mut [(LayerIndex, u32)],
        cluster_candidates: &mut [(u32, u32)],
        to_search: &mut Vec<u32>,
        searched: &mut BitVec,
    ) {
        // The below assumes that there is at least one cluster on some level.
        // If there are no clusters, create a cluster with the item and exit.
        if self.values.clusters.is_empty() {
            let mut cluster = HrcCluster::new(key.clone());
            cluster.insert(key, value);
            self.values.clusters.push(cluster);
            return;
        }

        // We need to initialize to_search to only pull in the first cluster from the highest layer.
        to_search.clear();
        to_search.push(0);

        // Go through each layer from the highest to the lowest.
        for layer in self.layers.iter().rev() {
            // Search this layer for the best candidates.
            let found = layer.search(&key, candidates, to_search, searched);
            // The values from this layer are cluster IDs from the next layer, use those to populate the to_search.
            to_search.extend(
                candidates[..found]
                    .iter()
                    .map(|&(ix, _)| *layer.get(ix).unwrap().1),
            );
        }

        // When we search the value layer, we only need to search the clusters to find the best match.
        self.values
            .search_clusters(&key, cluster_candidates, to_search, searched);

        // Insert it into the best cluster.
        self.values.clusters[cluster_candidates[0].0 as usize].insert(key, value);

        // If the cluster has gotten to large we must split it.
        if self.values.clusters[cluster_candidates[0].0 as usize].len() > self.max_cluster_len {
            self.split_value_cluster(
                cluster_candidates[0].0 as usize,
                candidates,
                cluster_candidates,
                to_search,
                searched,
            );
        }

        self.len += 1;
    }

    /// Splits a value layer cluster.
    fn split_value_cluster(
        &mut self,
        cluster_ix: usize,
        candidates: &mut [(LayerIndex, u32)],
        cluster_candidates: &mut [(u32, u32)],
        to_search: &mut Vec<u32>,
        searched: &mut BitVec,
    ) {
        // Start by taking the furthest item from the cluster center out of the cluster.
        let (key, value) = self.values.clusters[cluster_ix].remove_furthest();

        // Now we perform a standard search on the cluster to find the nearest neighbors to this key.
        let found = self.search(&key, candidates, to_search, searched);

        // Sort the candidates such that greater indices appear sooner in the candidates.
        // This is necessary to ensure we don't disturb indicies from other items in the candidates.
        // This sort must be stable to preserve the order of clusters so the dedup below still works.
        candidates[..found].sort_by_key(|(LayerIndex { item, .. }, _)| core::cmp::Reverse(*item));

        // Create a new cluster and add the key and value to it.
        let new_cluster_ix = self.values.clusters.len() as u32;
        let mut new_cluster = HrcCluster::new(key.clone());
        new_cluster.insert(key.clone(), value);

        // Check each of the found items in reverse.
        // Going in reverse is necessary to make sure when we remove items we don't disturb indicies
        // from other items we need to retrieve next.
        for &(layer_ix, distance) in candidates[..found].iter() {
            // Get the cluster the item belongs to.
            let other_cluster = &mut self.values.clusters[layer_ix.cluster as usize];
            // Get the distance of the item to its own cluster.
            let distance_to_other_cluster = other_cluster.distances[layer_ix.item as usize];
            // If the distance is closer to this cluster.
            if distance < distance_to_other_cluster {
                // Remove this item from the other cluster.
                let (key, value) = other_cluster.remove(layer_ix.item as usize);
                // Insert it to the new cluster.
                new_cluster.insert(key, value);
            }
        }

        // Grab all of the found neighbor clusters deduped and put them into to_search.
        to_search.clear();
        to_search.extend(
            candidates[..found]
                .iter()
                .map(|(layer_ix, _)| layer_ix.cluster)
                .dedup(),
        );

        // Search clusters on the values to find the closest clusters.
        let found = self
            .values
            .search_clusters(&key, cluster_candidates, to_search, searched);

        // Iterate through the closest clusters.
        for &(other_cluster_ix, _) in &cluster_candidates[..found] {
            // Connect the two clusters together.
            // These should not contain each other.
            self.values.clusters[other_cluster_ix as usize]
                .neighbors
                .push(new_cluster_ix);
            new_cluster.neighbors.push(other_cluster_ix);
        }

        let new_cluster_len = new_cluster.len();

        // Add the cluster.
        self.values.clusters.push(new_cluster);

        // At this point, we just added a new cluster, and this cluster needs to be added to the layers above.
        if self.layers.is_empty() {
            if self.values.clusters.len() > self.new_layer_threshold_clusters {
                // If the layer is empty, but we have crossed the threshold to initialize the first layer,
                // then we should initialize it.
                self.initialize_first_layer(candidates, cluster_candidates, to_search, searched);
            }
        } else {
            // Add the cluster to the first layer.
            self.insert_cluster_to_layer(
                key,
                new_cluster_ix,
                0,
                candidates,
                cluster_candidates,
                to_search,
                searched,
            );
        }

        // We may also need to split this cluster additional times if it is too large.
        if new_cluster_len > self.max_cluster_len {
            self.split_value_cluster(
                new_cluster_ix as usize,
                candidates,
                cluster_candidates,
                to_search,
                searched,
            );
        }
    }

    fn initialize_first_layer(
        &mut self,
        candidates: &mut [(LayerIndex, u32)],
        cluster_candidates: &mut [(u32, u32)],
        to_search: &mut Vec<u32>,
        searched: &mut BitVec,
    ) {
        self.layers.push(HrcLayer { clusters: vec![] });
        for cluster_ix in 0..self.values.clusters.len() {
            self.insert_cluster_to_layer(
                self.values.clusters[cluster_ix].key.clone(),
                cluster_ix as u32,
                0,
                candidates,
                cluster_candidates,
                to_search,
                searched,
            );
        }
    }

    /// Add a cluster from a lower layer to a higher layer.
    #[allow(clippy::too_many_arguments)]
    fn insert_cluster_to_layer(
        &mut self,
        key: K,
        cluster_ix: u32,
        layer: usize,
        candidates: &mut [(LayerIndex, u32)],
        cluster_candidates: &mut [(u32, u32)],
        to_search: &mut Vec<u32>,
        searched: &mut BitVec,
    ) {
        // The below assumes that there is at least one cluster in the layer already.
        // If there are no clusters, create a cluster and exit.
        if self.layers[layer].clusters.is_empty() {
            let mut cluster = HrcCluster::new(key.clone());
            cluster.insert(key, cluster_ix);
            self.layers[layer].clusters.push(cluster);
            return;
        }

        // We need to initialize to_search to only pull in the first cluster from the highest layer.
        to_search.clear();
        to_search.push(0);

        // Go through each layer from the highest to just above the one we need.
        for layer in self.layers[layer + 1..].iter().rev() {
            // Search this layer for the best candidates.
            let found = layer.search(&key, candidates, to_search, searched);
            // The values from this layer are cluster IDs from the next layer, use those to populate the to_search.
            to_search.extend(
                candidates[..found]
                    .iter()
                    .map(|&(ix, _)| *layer.get(ix).unwrap().1),
            );
        }

        // When we search the layer, we only need to search the clusters to find the best match.
        self.layers[layer].search_clusters(&key, cluster_candidates, to_search, searched);

        self.layers[layer].clusters[cluster_candidates[0].0 as usize].insert(key, cluster_ix);

        if self.layers[layer].clusters[cluster_candidates[0].0 as usize].len()
            > self.max_cluster_len
        {
            self.split_layer_cluster(
                cluster_candidates[0].0 as usize,
                layer,
                candidates,
                cluster_candidates,
                to_search,
                searched,
            );
        }
    }

    fn split_layer_cluster(
        &mut self,
        cluster_ix: usize,
        layer: usize,
        candidates: &mut [(LayerIndex, u32)],
        value_candidates: &mut [(u32, u32)],
        to_search: &mut Vec<u32>,
        searched: &mut BitVec,
    ) {
        // Start by taking the furthest item from the cluster center out of the cluster.
        let (key, value) = self.layers[layer].clusters[cluster_ix].remove_furthest();

        // Now we perform a standard search on the cluster to find the nearest neighbors to this key.
        let found = self.search_to_layer(&key, layer, candidates, to_search, searched);

        // Sort the candidates such that greater indices appear sooner in the candidates.
        // This is necessary to ensure we don't disturb indicies from other items in the candidates.
        // This sort must be stable to preserve the order of clusters so the dedup below still works.
        candidates[..found].sort_by_key(|(LayerIndex { item, .. }, _)| core::cmp::Reverse(*item));

        // Create a new cluster and add the key and value to it.
        let new_cluster_ix = self.layers[layer].clusters.len() as u32;
        let mut new_cluster = HrcCluster::new(key.clone());
        new_cluster.insert(key.clone(), value);

        // Check each of the found items and insert it if it's relevant.
        for &(layer_ix, distance) in candidates[..found].iter() {
            // Get the cluster the item belongs to.
            let other_cluster = &mut self.layers[layer].clusters[layer_ix.cluster as usize];
            // Get the distance of the item to its own cluster.
            let distance_to_other_cluster = other_cluster.distances[layer_ix.item as usize];
            // If the distance is closer to this cluster.
            if distance < distance_to_other_cluster {
                // Remove this item from the other cluster.
                let (key, value) = other_cluster.remove(layer_ix.item as usize);
                // Insert it to the new cluster.
                new_cluster.insert(key, value);
            }
        }

        // Grab all of the found neighbor clusters deduped and put them into to_search.
        to_search.clear();
        to_search.extend(
            candidates[..found]
                .iter()
                .map(|(layer_ix, _)| layer_ix.cluster)
                .dedup(),
        );

        // Search clusters on the values to find the closest clusters.
        let found = self.layers[layer].search_clusters(&key, value_candidates, to_search, searched);

        // Iterate through the closest clusters.
        for &(other_cluster_ix, _) in &value_candidates[..found] {
            // Connect the two clusters together.
            // These should not contain each other.
            self.layers[layer].clusters[other_cluster_ix as usize]
                .neighbors
                .push(new_cluster_ix);
            new_cluster.neighbors.push(other_cluster_ix);
        }

        let new_cluster_len = new_cluster.len();

        // Add the cluster.
        self.layers[layer].clusters.push(new_cluster);

        // At this point, we just added a new cluster, and this cluster needs to be added to the layers above.
        if layer + 1 >= self.layers.len() {
            if self.layers[layer].clusters.len() > self.new_layer_threshold_clusters {
                // If the layer doesn't exist, but we have crossed the threshold to initialize the layer,
                // then we should initialize it.
                self.initialize_layer_above(
                    layer,
                    candidates,
                    value_candidates,
                    to_search,
                    searched,
                );
            }
        } else {
            // Add the cluster to the first layer.
            self.insert_cluster_to_layer(
                key,
                new_cluster_ix,
                layer + 1,
                candidates,
                value_candidates,
                to_search,
                searched,
            );
        }

        // We may also need to split this cluster additional times if it is too large.
        if new_cluster_len > self.max_cluster_len {
            self.split_layer_cluster(
                new_cluster_ix as usize,
                layer,
                candidates,
                value_candidates,
                to_search,
                searched,
            );
        }
    }

    fn initialize_layer_above(
        &mut self,
        layer: usize,
        candidates: &mut [(LayerIndex, u32)],
        value_candidates: &mut [(u32, u32)],
        to_search: &mut Vec<u32>,
        searched: &mut BitVec,
    ) {
        self.layers.push(HrcLayer { clusters: vec![] });
        for cluster_ix in 0..self.layers[layer].clusters.len() {
            self.insert_cluster_to_layer(
                self.layers[layer].clusters[cluster_ix].key.clone(),
                cluster_ix as u32,
                layer + 1,
                candidates,
                value_candidates,
                to_search,
                searched,
            );
        }
    }
}

impl<K, V> Default for HrcCore<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HrcLayer<K, V> {
    clusters: Vec<HrcCluster<K, V>>,
}

impl<K, V> HrcLayer<K, V> {
    /// Creates an empty [`HrcLayer`].
    fn new() -> Self {
        Self {
            clusters: Default::default(),
        }
    }

    /// Retrieves the value from a [`LayerIndex`].
    fn get(&self, ix: LayerIndex) -> Option<(&K, &V)> {
        self.clusters.get(ix.cluster as usize).and_then(|cluster| {
            cluster
                .keys
                .get(ix.item as usize)
                .zip(cluster.values.get(ix.item as usize))
        })
    }
}

impl<K, V> HrcLayer<K, V>
where
    K: MetricPoint,
{
    /// Searches the layer, placing the best candidate items into the slice from best to worst.
    ///
    /// `to_search` must contain the starting clusters to begin search from. `searched` will get cleared
    /// and used during the search.
    ///
    /// Any unoccupied spots will be filled with an empty LayerIndex, which contains only `!0`.
    ///
    /// Returns the number of candidates populated
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

        let mut must_beat = !0;

        // Continuously attempt to search clusters.
        while let Some(cluster_ix) = to_search.pop() {
            let cluster = &self.clusters[cluster_ix as usize];
            let center_distance = cluster.center_distance(query);
            let (keys, potential) = cluster.potential_closer_items(center_distance, must_beat);

            // If this cluster is totally out of range, we should just skip it and not add its neighbors.
            if !potential {
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
                if !*searched.get(cluster_ix as usize).unwrap() {
                    searched.set(cluster_ix as usize, true);
                    to_search.push(cluster_ix);
                }
            }
        }

        candidates.partition_point(|n| !n.0.is_empty())
    }

    /// Searches the layer, placing the best clusters into the slice from best to worst.
    ///
    /// `to_search` must contain the starting clusters to begin search from. `searched` and `candidates`
    /// will get cleared and used during the search.
    ///
    /// Any unoccupied cluster spots will be filled with `!0`.
    ///
    /// Returns the number of candidates populated
    fn search_clusters(
        &self,
        query: &K,
        candidates: &mut [(u32, u32)],
        to_search: &mut Vec<u32>,
        searched: &mut BitVec,
    ) -> usize {
        // Initialize the mutable inputs to the correct values.
        candidates.fill((!0, !0));
        searched.clear();
        searched.resize(self.clusters.len(), false);

        // Set all the initial clusters as searched, since they were already added to `to_search`.
        for &cluster_ix in to_search.iter() {
            searched.set(cluster_ix as usize, true);
        }

        let mut must_beat = !0;

        // Continuously attempt to search clusters.
        while let Some(cluster_ix) = to_search.pop() {
            let cluster = &self.clusters[cluster_ix as usize];
            let center_distance = cluster.center_distance(query);
            // Check if the cluster is a new candidate.
            if center_distance < must_beat {
                // Add the candidate and update the must_beat.
                add_candidate(candidates, (cluster_ix, center_distance));
                must_beat = candidates.last().unwrap().1;

                // Set neighbor clusters as searched and add them to the to_search pool only if they weren't already searched.
                for &cluster_ix in &cluster.neighbors {
                    if !*searched.get(cluster_ix as usize).unwrap() {
                        searched.set(cluster_ix as usize, true);
                        to_search.push(cluster_ix);
                    }
                }
            }
        }

        candidates.partition_point(|n| n.0 != !0)
    }
}

impl<K, V> Default for HrcLayer<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

/// Must contain at least one item. The first item is the cluster center.
#[derive(Debug, Clone, PartialEq)]
struct HrcCluster<K, V> {
    key: K,
    neighbors: Vec<u32>,
    keys: Vec<K>,
    values: Vec<V>,
    /// This keeps track of the distances of all key-value pairs, and this is always ordered from least to greatest.
    distances: Vec<u32>,
}

impl<K, V> HrcCluster<K, V>
where
    K: MetricPoint,
{
    /// Creates a new cluster with the given center.
    fn new(key: K) -> Self {
        Self {
            key,
            neighbors: vec![],
            keys: vec![],
            values: vec![],
            distances: vec![],
        }
    }

    /// Returns the slice of keys and values that could beat a given distance and a bool indicating if the
    /// cluster radius distance could have encompassed a relevant point.
    fn potential_closer_items(&self, center_distance: u32, must_beat: u32) -> (&[K], bool) {
        let minimum_inclusive = (center_distance + 1).saturating_sub(must_beat);
        let maximum_exclusive = center_distance.saturating_add(must_beat);
        let begin = self.distances.partition_point(|&d| d < minimum_inclusive);
        let end = self.distances.partition_point(|&d| d < maximum_exclusive);
        (
            &self.keys[begin..end],
            minimum_inclusive <= self.distances.last().copied().unwrap_or(0),
        )
    }

    /// Computes the distance to the center of the cluster.
    fn center_distance(&self, key: &K) -> u32 {
        self.key.distance(key)
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

    /// Remove and return the furthest item from the center of the cluster.
    fn remove_furthest(&mut self) -> (K, V) {
        self.distances.pop();
        (self.keys.pop().unwrap(), self.values.pop().unwrap())
    }

    /// Remove a specific item from this cluster.
    fn remove(&mut self, item: usize) -> (K, V) {
        self.distances.remove(item);
        (self.keys.remove(item), self.values.remove(item))
    }

    /// Gets the number of items in the cluster.
    fn len(&self) -> usize {
        self.keys.len()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct LayerIndex {
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

fn add_candidate<T>(candidates: &mut [(T, u32)], c: (T, u32)) {
    if c.1 < candidates.last().unwrap().1 {
        let pos = candidates.partition_point(|other| other.1 <= c.1);
        candidates[pos..].rotate_right(1);
        candidates[pos] = c;
    }
}
