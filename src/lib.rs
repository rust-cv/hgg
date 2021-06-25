#![no_std]
extern crate alloc;

#[cfg(test)]
mod unit_tests;

#[cfg(feature = "stats")]
mod stats;

use itertools::Itertools;
#[cfg(feature = "stats")]
pub use stats::*;

use alloc::vec;
use alloc::vec::Vec;
use space::MetricPoint;

#[derive(Debug, Clone, PartialEq)]
struct HrcZeroNode<K, V> {
    key: K,
    value: V,
    edges: Vec<(K, usize)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hrc<K, V> {
    /// The zero layer.
    zero: Vec<HrcZeroNode<K, V>>,
    /// Clusters with more items than this are split apart.
    max_cluster_len: usize,
}

impl<K, V> Hrc<K, V> {
    /// Creates a new [`Hrc`]. It will be empty and begin with default settings.
    pub fn new() -> Self {
        Self {
            zero: vec![],
            max_cluster_len: 1024,
        }
    }

    /// Sets the max number of items allowed in a cluster before it is split apart.
    pub fn max_cluster_len(self, max_cluster_len: usize) -> Self {
        Self {
            max_cluster_len,
            ..self
        }
    }

    /// Get the (key, value) pair of a node.
    pub fn get(&self, node: usize) -> Option<(&K, &V)> {
        self.zero.get(node).map(|node| (&node.key, &node.value))
    }

    /// Get the key of a node.
    pub fn get_key(&self, node: usize) -> Option<&K> {
        self.zero.get(node).map(|node| &node.key)
    }

    /// Get the value of a node.
    pub fn get_value(&self, node: usize) -> Option<&V> {
        self.zero.get(node).map(|node| &node.value)
    }

    pub fn is_empty(&self) -> bool {
        self.zero.is_empty()
    }

    pub fn len(&self) -> usize {
        self.zero.len()
    }

    pub fn neighbors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.zero[node].edges.iter().map(|&(_, node)| node)
    }

    pub fn neighbor_keys(&self, node: usize) -> impl Iterator<Item = (&K, usize)> + '_ {
        self.zero[node].edges.iter().map(|(key, node)| (key, *node))
    }

    fn remove_edge(&mut self, a: usize, b: usize) {
        self.zero[a].edges.retain(|&(_, node)| node != b);
        self.zero[b].edges.retain(|&(_, node)| node != a);
    }

    pub fn histogram(&self) -> Vec<(usize, usize)> {
        let mut histogram = vec![];
        for edges in self.zero.iter().map(|node| node.edges.len()) {
            match histogram.binary_search_by_key(&edges, |&(search_edges, _)| search_edges) {
                Ok(pos) => histogram[pos].1 += 1,
                Err(pos) => histogram.insert(pos, (edges, 1)),
            }
        }
        histogram
    }
}

impl<K, V> Hrc<K, V>
where
    K: Clone,
{
    fn add_edge(&mut self, a: usize, b: usize) {
        let a_key = self.zero[a].key.clone();
        let b_key = self.zero[b].key.clone();
        self.zero[a].edges.push((b_key, b));
        self.zero[b].edges.push((a_key, a));
    }

    fn add_edge_dedup(&mut self, a: usize, b: usize) {
        if !self.neighbors(a).any(|node| node == b) {
            self.add_edge(a, b);
        }
    }
}

impl<K, V> Hrc<K, V>
where
    K: MetricPoint + Clone,
{
    /// Searches for the nearest neighbor greedily.
    ///
    /// Returns `(node, distance)`.
    pub fn search(&self, query: &K) -> Option<(usize, u32)> {
        if self.is_empty() {
            None
        } else {
            Some(self.search_from(0, query))
        }
    }

    /// Finds the nearest neighbor to the query key starting from the `from` node using greedy search.
    ///
    /// Returns `(node, distance)`.
    pub fn search_from(&self, from: usize, query: &K) -> (usize, u32) {
        let mut best_node = from;
        let mut best_distance = query.distance(&self.zero[from].key);

        while let Some((neighbor_node, distance)) = self
            .neighbor_keys(best_node)
            .map(|(neighbor_key, neighbor_node)| (neighbor_node, query.distance(neighbor_key)))
            .min_by_key(|&(_, distance)| distance)
        {
            if distance < best_distance {
                best_node = neighbor_node;
                best_distance = distance;
            } else {
                break;
            }
        }
        (best_node, best_distance)
    }

    /// Finds the knn greedily from a starting node `from`.
    ///
    /// Returns (node, distance, searched) pairs. `searched` will always be true, so you can ignore it.
    pub fn search_knn_from(&self, from: usize, query: &K, num: usize) -> Vec<(usize, u32, bool)> {
        assert!(
            num > 0,
            "the number of nearest neighbors queried MUST be at least 1"
        );
        // Perform a greedy search first to save time.
        let (from, from_distance) = self.search_from(from, query);
        // Contains the index and the distance as a pair.
        let mut bests = vec![(from, from_distance, false)];

        loop {
            if let Some((previous_node, _, searched)) =
                bests.iter_mut().find(|&&mut (_, _, searched)| !searched)
            {
                // Set this as searched (we are searching it now).
                *searched = true;
                // Erase the reference to the search node (to avoid lifetime & borrowing issues).
                let previous_node = *previous_node;

                for (search_key, search_node) in self.neighbor_keys(previous_node) {
                    // Make sure that we don't have a copy of this node already or we will get duplicates.
                    if bests.iter().any(|&(node, _, _)| search_node == node) {
                        continue;
                    }
                    // Compute the distance from the query.
                    let distance = query.distance(search_key);
                    // If we dont have enough yet, add it.
                    if bests.len() < num {
                        bests.insert(
                            bests.partition_point(|&(_, best_distance, _)| {
                                best_distance <= distance
                            }),
                            (search_node, distance, false),
                        );
                    } else if distance < bests.last().unwrap().1 {
                        // Otherwise only add it if its better than the worst item we have.
                        bests.pop();
                        bests.insert(
                            bests.partition_point(|&(_, best_distance, _)| {
                                best_distance <= distance
                            }),
                            (search_node, distance, false),
                        );
                    }
                }
            } else {
                return bests;
            }
        }
    }

    /// Performs a greedy search starting from node `from`. Keeps track of where it came from, and returns the path
    /// that it traveled to reach the destination.
    pub fn search_from_path(&self, from: usize, query: &K) -> Vec<usize> {
        let mut best_node = from;
        let mut best_distance = query.distance(&self.zero[from].key);
        let mut path = vec![from];

        while let Some((neighbor_node, distance)) = self
            .neighbor_keys(best_node)
            .map(|(neighbor_key, neighbor_node)| (neighbor_node, query.distance(neighbor_key)))
            .min_by_key(|&(_, distance)| distance)
        {
            if distance < best_distance {
                best_node = neighbor_node;
                best_distance = distance;
                path.push(best_node);
            } else {
                break;
            }
        }

        path
    }

    /// Insert a (key, value) pair.
    ///
    /// `quality` is a value of at least `1` which describes the number of nearest neighbors
    /// used to ensure greedy search around the inserted item. This number needs to be higher based
    /// on the dimensionality of the data set, and specifically the dimensionality of the region that
    /// this point is inserted.
    pub fn insert(&mut self, key: K, value: V, quality: usize) -> usize {
        // Add the node (it will be added this way regardless).
        let new_node = self.zero.len();
        self.zero.push(HrcZeroNode {
            key,
            value,
            edges: vec![],
        });

        // Find knn.
        let knn = self.search_knn_from(0, &self.zero[new_node].key, quality);

        // Connect the nearest neighbor.
        self.add_edge(knn[0].0, new_node);

        // Optimize the graph to each of the nearest neighbors.
        for &(nn, _, _) in &knn[1..] {
            self.optimize_connection(nn, new_node, quality);
        }

        new_node
    }

    /// Removes a node from the graph and then reinserts it with the given quality.
    ///
    /// This is useful to prune unecessary connections in the graph.
    pub fn reinsert(&mut self, node: usize, quality: usize) {
        // This wont work if we only have 1 node.
        if self.len() == 1 {
            return;
        }

        // Disconnect the node from the graph, keeping track of its old neighbors.
        // We need to do this to avoid splitting the graph into disconnected graphs.
        let neighbors = self.disconnect(node);
        // Make sure each neighbor can connect.
        for neighbor in neighbors {
            let knn = self.search_knn_from(neighbor, &self.zero[node].key, quality);
            let has_node = knn.iter().any(|&(nn, _, _)| nn == node);
            if !has_node {
                // Must be dedup because it could be many (quality) colocated nodes.
                self.add_edge_dedup(node, knn[0].0);
            }
            self.optimize_connection(neighbor, node, quality);
        }
    }

    /// Internal function for disconnecting a node from the graph.
    fn disconnect(&mut self, node: usize) -> Vec<usize> {
        let neighbors = self.neighbors(node).collect_vec();
        for &neighbor in &neighbors {
            self.remove_edge(node, neighbor);
        }
        neighbors
    }

    /// Trains the HRC by making connections so that the nearest neighbors to the given key can be found.
    pub fn train(&mut self, key: &K, quality: usize) {
        if self.zero.len() >= 2 {
            // First, we want to find `quality` nearest neighbors to the key.
            let knn = self.search_knn_from(0, key, quality);
            // Make sure that there is a greedy search path from the root node to the nearest neighbor.
            // We set the termination distance at the nearest neighbor's distance.
            self.optimize_target_directed(0, knn[0].1, key, quality);
        }
    }

    /// Optimizes the connection between two nodes to ensure the optimal greedy search path is available in both directions.
    ///
    /// This works even if the two nodes exist in totally disconnected graphs.
    pub fn optimize_connection(&mut self, a: usize, b: usize, quality: usize) {
        match (
            self.optimize_connection_directed(a, b, quality),
            self.optimize_connection_directed(b, a, quality),
        ) {
            (Some(opti_a), Some(opti_b)) => self.add_edge(opti_a, opti_b),
            (None, None) => {}
            _ => {
                unreachable!(
                    "this case can only occur if there is a directed edge, which shouldn't exist"
                )
            }
        }
    }

    pub fn optimize_connection_directed(
        &mut self,
        from: usize,
        to: usize,
        quality: usize,
    ) -> Option<usize> {
        let key = self.zero[to].key.clone();
        let found = self.optimize_target_directed(from, 0, &key, quality);
        if found != to {
            if self.distance(found, to) == 0 {
                self.add_edge_dedup(found, to);
                None
            } else {
                Some(found)
            }
        } else {
            None
        }
    }

    /// Ensures that the optimal greedy path exists towards a specific key from a specific node.
    ///
    /// Will terminate when a distance equal to or better than `to_distance` is reached.
    ///
    /// Returns the termination node.
    pub fn optimize_target_directed(
        &mut self,
        from: usize,
        min_distance: u32,
        target: &K,
        quality: usize,
    ) -> usize {
        // Search towards the target greedily.
        let (mut from, mut from_distance) = self.search_from(from, target);
        // This loop will gradually break through local minima using the nearest neighbor possible repeatedly
        // until a greedy search path is established.
        'outer: loop {
            // Check if we matched or exceeded expectations.
            if from_distance <= min_distance {
                return from;
            }

            // In any other case, we have hit a local (but not global) minima.
            // Our goal is to find the nearest neighbor which can break through the local minima.
            // This process will be tried with exponentially more nearest neighbors until
            // we find the nearest neighbor that can break through the minima.
            // We start with a specific quality so that we are more likely to get the true nearest neighbors
            // than if we just started with 2.
            for quality in core::iter::successors(Some(quality), |&quality| {
                if quality >= self.len() {
                    None
                } else {
                    Some(quality.saturating_mul(2))
                }
            }) {
                // Start by finding the nearest neighbors to the local minima starting at itself.
                let knn = self.search_knn_from(from, &self.zero[from].key, quality);
                // Go through the nearest neighbors in order from best to worst.
                for &(nn, _, _) in &knn[1..] {
                    // Compute the distance to the target from the nn.
                    let nn_distance = self.zero[nn].key.distance(target);
                    // Check if this node is closer to the target than `from`.
                    if nn_distance < from_distance {
                        // In this case, a greedy search to this node would get closer to the target,
                        // so add an edge to this node.
                        self.add_edge(from, nn);
                        // Then we need to perform a greedy search towards the target from this node.
                        // This will become the new node for the next round of the loop.
                        let (new_from, new_from_distance) = self.search_from(nn, target);
                        from = new_from;
                        from_distance = new_from_distance;
                        // Continue the outer loop to iteratively move towards the target.
                        continue 'outer;
                    }
                }
            }
            // If we get to this point, we searched the entire graph and there was no path.
            return from;
        }
    }

    /// Globally optimizes the graph with the given quality level.
    pub fn optimize(&mut self, quality: usize) {
        for node in 0..self.len() {
            self.reinsert(node, quality);
        }
    }

    /// Computes the distance between two nodes.
    pub fn distance(&self, a: usize, b: usize) -> u32 {
        self.zero[a].key.distance(&self.zero[b].key)
    }
}

impl<K, V> Default for Hrc<K, V> {
    fn default() -> Self {
        Self::new()
    }
}
