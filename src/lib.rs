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
    edges: Vec<usize>,
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

    fn add_edge(&mut self, a: usize, b: usize) {
        self.zero[a].edges.push(b);
        self.zero[b].edges.push(a);
    }

    fn add_edge_dedup(&mut self, a: usize, b: usize) {
        if !self.zero[a].edges.contains(&b) {
            self.add_edge(a, b);
        }
    }

    fn remove_edge(&mut self, a: usize, b: usize) {
        self.zero[a].edges.retain(|&node| node != b);
        self.zero[b].edges.retain(|&node| node != a);
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
    K: MetricPoint + Clone,
{
    /// Finds the nearest neighbor to the query key starting from the `from` node using greedy search.
    pub fn search_from(&self, from: usize, query: &K) -> usize {
        let mut queue = self.zero[from].edges.clone();
        let mut best_node = from;
        let mut best_distance = query.distance(&self.zero[from].key);

        while let Some(search_node) = queue.pop() {
            let distance = query.distance(&self.zero[search_node].key);
            if distance < best_distance {
                best_node = search_node;
                best_distance = distance;
                queue.extend(self.zero[search_node].edges.iter().copied());
            }
        }

        best_node
    }

    /// Performs a search to the query key using greedy search.
    pub fn search(&self, query: &K) -> Option<usize> {
        if self.is_empty() {
            None
        } else {
            Some(self.search_from(0, query))
        }
    }

    /// Finds the knn greedily from a starting node `from`.
    ///
    /// Returns (node, distance) pairs.
    pub fn search_knn_from(&self, from: usize, query: &K, num: usize) -> Vec<(usize, u32)> {
        assert!(
            num > 0,
            "the number of nearest neighbors queried MUST be at least 1"
        );
        // Perform a greedy search first to save time.
        let from = self.search_from(from, query);
        let mut queue = self.zero[from].edges.clone();
        // Contains the index and the distance as a pair.
        let mut bests = vec![(from, query.distance(&self.zero[from].key))];

        while let Some(search_node) = queue.pop() {
            if bests.iter().any(|&(node, _)| search_node == node) {
                continue;
            }
            let distance = query.distance(&self.zero[search_node].key);
            // If we dont have enough yet, add it.
            if bests.len() < num {
                bests.insert(
                    bests.partition_point(|&(_, best_distance)| best_distance <= distance),
                    (search_node, distance),
                );
                queue.extend(self.zero[search_node].edges.iter().copied());
                continue;
            }
            // Otherwise only add it if its better than the worst item we have.
            if distance < bests.last().unwrap().1 {
                bests.pop();
                bests.insert(
                    bests.partition_point(|&(_, best_distance)| best_distance <= distance),
                    (search_node, distance),
                );
                queue.extend(self.zero[search_node].edges.iter().copied());
            }
        }

        bests
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
            key: key.clone(),
            value,
            edges: vec![],
        });

        // If this is the only node, just return it.
        if new_node == 0 {
            return new_node;
        }

        // Search for the nearest neighbors.
        let knn = self.search_knn_from(0, &key, quality);

        for &(nn, _) in &knn {
            self.optimize_connection(nn, new_node);
        }

        new_node
    }

    /// Trains the HRC by making connections so that the nearest neighbors to the given key can be found.
    pub fn train(&mut self, key: &K, quality: usize) {
        if self.zero.len() >= 2 {
            // First, we want to find `quality` nearest neighbors to the key.
            let knn = self.search_knn_from(0, key, quality);
            // Next, we want to ensure that if we encounter a situation in which we cannot search from
            // a nearer nearest neighbor to the nearest neighbor that we make that connection.
            for &(nn, _) in &knn[1..] {
                // Perform this search on the non-nearest neighbor.
                let found = self.search_from(nn, key);
                // If found is not the nearest neighbor, make sure that there is a connection.
                if found != knn[0].0 {
                    self.add_edge_dedup(knn[0].0, found);
                };
            }
        }
    }

    /// Optimizes the connection between two nodes to ensure a greedy search path is available in both directions.
    pub fn optimize_connection(&mut self, a: usize, b: usize) {
        // Search from a to b.
        let found = self.search_from(a, &self.zero[b].key);
        // If we didn't reach b, connect the found node to b.
        if found != b {
            self.add_edge_dedup(found, b);
        }
    }

    /// Removes a node from the graph and then reinserts it with the given quality.
    ///
    /// This is useful to prune unecessary connections in the graph.
    pub fn optimize_node(&mut self, node: usize, quality: usize) {
        let neighbors = self.zero[node].edges.iter().copied().collect_vec();

        for neighbor in neighbors {
            self.remove_edge(node, neighbor);
        }

        // Search for the nearest neighbors.
        let knn = self.search_knn_from(
            if node == 0 {
                if self.len() == 1 {
                    return;
                } else {
                    1
                }
            } else {
                0
            },
            &self.zero[node].key,
            quality,
        );

        for (nn, _) in knn {
            // Search for the new node from the nearest neighbor, connecting the found node with the new_node.
            let found = self.search_from(nn, &self.zero[node].key);
            if found != node {
                self.add_edge_dedup(found, node);
            }
        }
    }

    /// Globally optimizes the graph with the given quality level.
    pub fn optimize(&mut self, quality: usize) {
        for node in 0..self.len() {
            self.optimize_node(node, quality);
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
