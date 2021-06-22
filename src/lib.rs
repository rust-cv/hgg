#![no_std]
extern crate alloc;

#[cfg(test)]
mod unit_tests;

#[cfg(feature = "stats")]
mod stats;

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
}

impl<K, V> Hrc<K, V>
where
    K: MetricPoint + Clone,
{
    /// Finds the nearest neighbor to the query key starting from the `from` node using greedy search.
    pub fn search_from(&self, from: usize, query: &K) -> usize {
        let mut queue = vec![from];
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

    /// Insert a (key, value) pair.
    pub fn insert(&mut self, key: K, value: V) -> usize {
        if self.is_empty() {
            self.zero.push(HrcZeroNode {
                key,
                value,
                edges: vec![],
            });
            self.zero.len() - 1
        } else {
            unimplemented!()
        }
    }
}

impl<K, V> Default for Hrc<K, V> {
    fn default() -> Self {
        Self::new()
    }
}
