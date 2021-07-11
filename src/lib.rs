#![no_std]
extern crate alloc;

mod hvec;
#[cfg(feature = "serde")]
mod serde_impl;
#[cfg(test)]
mod unit_tests;

use ahash::RandomState;
use alloc::{vec, vec::Vec};
use core::{
    fmt::Debug,
    iter,
    marker::PhantomData,
    ops::{Deref, DerefMut},
};
use hashbrown::HashSet;
use header_vec::HeaderVec;
use hvec::{HVec, HggEdge, HggHeader};
use num_traits::Zero;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use space::{Knn, MetricPoint, Neighbor};

#[derive(Debug)]
struct StrategyRegular;
#[derive(Debug)]
struct StrategyLite;

/// An approximate nearest neighbor search collection that pairs keys to values.
///
/// Use this HGG when it can fit in your memory and if your key isnt isn't too large.
/// It tends to be faster. If you have a situation where you are running out of memory or
/// you just have a really large key, your distance function is incredibly expensive to the
/// point where random access time in memory is irrelevant, or your key doesn't
/// implement [`Clone`], then use [`HggLite`] in those cases.
///
/// If your distance function is very expensive, you may also want to look at [`Hgg::exclude_all_searched`].
///
/// Always remember to benchmark rather than guess when it comes to the above choices.
///
/// If you are looking for how to perform kNN searches, see `impl<K, V> Knn<K> for Hgg<K, V>` below.
#[derive(Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "K: Serialize, V: Serialize",
        deserialize = "K: Deserialize<'de> + Clone, V: Deserialize<'de>"
    ))
)]
pub struct Hgg<K, V> {
    hgg: HggCore<K, V, K, StrategyRegular>,
}

impl<K, V> Knn<K> for Hgg<K, V>
where
    K: MetricPoint + Clone,
{
    type KnnIter = Vec<Neighbor<K::Metric>>;

    fn knn(&self, query: &K, num: usize) -> Self::KnnIter {
        self.hgg
            .search_knn(query, num)
            .map(|(index, distance)| Neighbor { index, distance })
            .collect()
    }

    fn nn(&self, query: &K) -> Option<Neighbor<K::Metric>> {
        self.hgg
            .search(query)
            .map(|(index, distance)| Neighbor { index, distance })
    }
}

impl<K, V> Hgg<K, V>
where
    K: MetricPoint + Clone,
{
    /// Creates a new [`Hgg`]. It will be empty and begin with default settings.
    pub fn new() -> Self {
        Self {
            hgg: HggCore::new(),
        }
    }

    /// Default value: `1`
    ///
    /// Increase the parameter `freshens` to freshen stale nodes in the graph. The higher this value, the longer the
    /// insert will take. However, in the long run, freshening may improve insert performance. It is recommended
    /// to benchmark with your data both the insert and lookup performance against recall using this
    /// parameter to determine the right value for you. The default should be fine for most users.
    pub fn freshens(self, freshens: usize) -> Self {
        Self {
            hgg: self.hgg.freshens(freshens),
        }
    }

    /// Default value: `false`
    ///
    /// If this is true, when doing a kNN search, any key which has already had its distance computed will not be
    /// computed again. kNN search (and insertion) is faster when this is set to `false` for keys with cheap
    /// distance functions. If your distance function is expensive, benchmark Hgg with this parameter set to `true`.
    /// For some distance functions/key types this will be better, and for some it will be worse.
    /// Benchmark your data and observe the recall curve to find out.
    pub fn exclude_all_searched(self, exclude_all_searched: bool) -> Self {
        Self {
            hgg: self.hgg.exclude_all_searched(exclude_all_searched),
        }
    }

    /// Default value: `64`
    ///
    /// This controls the number of nearest neighbors used during insertion. Setting this higher will cause the graph
    /// to become more connected if your data has thick Voronoi boundaries. If this is true of your dataset (
    /// usually due to using hamming distance or high dimensionality), then you may want to intentionally
    /// set this lower to avoid consuming too much memory, which can decrease performance if slower
    /// memory (such as swap space) is used.
    ///
    /// For all datasets, this value correlates positively with insertion time (inversely with speed). If you want insertions to go faster,
    /// consider decreasing this value.
    pub fn insert_knn(self, insert_knn: usize) -> Self {
        Self {
            hgg: self.hgg.insert_knn(insert_knn),
        }
    }

    /// Insert a (key, value) pair.
    pub fn insert(&mut self, key: K, value: V) -> usize {
        self.hgg.insert(key, value)
    }

    /// Get the (key, value) pair of a node.
    pub fn get(&self, node: usize) -> Option<(&K, &V)> {
        self.hgg.get(node)
    }

    /// Get the key of a node.
    pub fn get_key(&self, node: usize) -> Option<&K> {
        self.hgg.get_key(node)
    }

    /// Get the value of a node.
    pub fn get_value(&self, node: usize) -> Option<&V> {
        self.hgg.get_value(node)
    }

    /// Checks if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.hgg.is_empty()
    }

    /// Returns the number of (key, value) pairs added to the graph.
    pub fn len(&self) -> usize {
        self.hgg.len()
    }

    /// Returns the number of edges in the graph on each layer.
    pub fn edges(&self) -> Vec<usize> {
        self.hgg.edges()
    }

    /// Returns the number of layers in the graph.
    pub fn layers(&self) -> usize {
        self.hgg.layers()
    }

    pub fn histogram_layer_nodes(&self) -> Vec<usize> {
        self.hgg.histogram_layer_nodes()
    }

    pub fn histogram_neighbors(&self) -> Vec<Vec<(usize, usize)>> {
        self.hgg.histogram_neighbors()
    }

    pub fn average_neighbors(&self) -> Vec<f64> {
        self.hgg.average_neighbors()
    }

    pub fn simple_representation(&self) -> Vec<Vec<Vec<usize>>> {
        self.hgg.simple_representation()
    }
}

impl<K, V> Default for Hgg<K, V>
where
    K: MetricPoint + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

/// An approximate nearest neighbor search collection that pairs keys to values.
///
/// Use this HGG when you are running out of memory or your keys are very large.
/// This HGG tends to be slower, but may be faster with very large keys, especially if you are running out of RAM.
/// If you have a situation where you are running out of memory or you just have a really large key,
/// your distance function is incredibly expensive to the point where random access time in memory
/// is irrelevant, or your key doesn't implement [`Clone`], then use [`HggLite`] in those cases.
/// Otherwise, it is recommended to use [`Hgg`].
///
/// If your distance function is very expensive, you may also want to look at [`HggLite::exclude_all_searched`].
///
/// Always remember to benchmark rather than guess when it comes to the above choices.
///
/// If you are looking for how to perform kNN searches, see `impl<K, V> Knn<K> for HggLite<K, V>` below.
#[derive(Debug)]
pub struct HggLite<K, V> {
    hgg: HggCore<K, V, (), StrategyLite>,
}

impl<K, V> Knn<K> for HggLite<K, V>
where
    K: MetricPoint,
{
    type KnnIter = Vec<Neighbor<K::Metric>>;

    fn knn(&self, query: &K, num: usize) -> Self::KnnIter {
        self.hgg
            .search_knn(query, num)
            .map(|(index, distance)| Neighbor { index, distance })
            .collect()
    }

    fn nn(&self, query: &K) -> Option<Neighbor<K::Metric>> {
        self.hgg
            .search(query)
            .map(|(index, distance)| Neighbor { index, distance })
    }
}

impl<K, V> HggLite<K, V>
where
    K: MetricPoint,
{
    /// Creates a new [`Hgg`]. It will be empty and begin with default settings.
    pub fn new() -> Self {
        Self {
            hgg: HggCore::new(),
        }
    }

    /// Default value: `1`
    ///
    /// Increase the parameter `freshens` to freshen stale nodes in the graph. The higher this value, the longer the
    /// insert will take. However, in the long run, freshening may improve insert performance. It is recommended
    /// to benchmark with your data both the insert and lookup performance against recall using this
    /// parameter to determine the right value for you. The default should be fine for most users.
    pub fn freshens(self, freshens: usize) -> Self {
        Self {
            hgg: self.hgg.freshens(freshens),
        }
    }

    /// Default value: `false`
    ///
    /// If this is true, when doing a kNN search, any key which has already had its distance computed will not be
    /// computed again. kNN search (and insertion) is faster when this is set to `false` for keys with cheap
    /// distance functions. If your distance function is expensive, benchmark Hgg with this parameter set to `true`.
    /// For some distance functions/key types this will be better, and for some it will be worse.
    /// Benchmark your data and observe the recall curve to find out.
    pub fn exclude_all_searched(self, exclude_all_searched: bool) -> Self {
        Self {
            hgg: self.hgg.exclude_all_searched(exclude_all_searched),
        }
    }

    /// Default value: `64`
    ///
    /// This controls the number of nearest neighbors used during insertion. Setting this higher will cause the graph
    /// to become more connected if your data has thick Voronoi boundaries. If this is true of your dataset (
    /// usually due to using hamming distance or high dimensionality), then you may want to intentionally
    /// set this lower to avoid consuming too much memory, which can decrease performance if slower
    /// memory (such as swap space) is used.
    ///
    /// For all datasets, this value correlates positively with insertion time (inversely with speed). If you want insertions to go faster,
    /// consider decreasing this value.
    pub fn insert_knn(self, insert_knn: usize) -> Self {
        Self {
            hgg: self.hgg.insert_knn(insert_knn),
        }
    }

    /// Insert a (key, value) pair.
    pub fn insert(&mut self, key: K, value: V) -> usize {
        self.hgg.insert(key, value)
    }

    /// Get the (key, value) pair of a node.
    pub fn get(&self, node: usize) -> Option<(&K, &V)> {
        self.hgg.get(node)
    }

    /// Get the key of a node.
    pub fn get_key(&self, node: usize) -> Option<&K> {
        self.hgg.get_key(node)
    }

    /// Get the value of a node.
    pub fn get_value(&self, node: usize) -> Option<&V> {
        self.hgg.get_value(node)
    }

    /// Checks if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.hgg.is_empty()
    }

    /// Returns the number of (key, value) pairs added to the graph.
    pub fn len(&self) -> usize {
        self.hgg.len()
    }

    /// Returns the number of edges in the graph on each layer.
    pub fn edges(&self) -> Vec<usize> {
        self.hgg.edges()
    }

    /// Returns the number of layers in the graph.
    pub fn layers(&self) -> usize {
        self.hgg.layers()
    }

    pub fn histogram_layer_nodes(&self) -> Vec<usize> {
        self.hgg.histogram_layer_nodes()
    }

    pub fn histogram_neighbors(&self) -> Vec<Vec<(usize, usize)>> {
        self.hgg.histogram_neighbors()
    }

    pub fn average_neighbors(&self) -> Vec<f64> {
        self.hgg.average_neighbors()
    }

    pub fn simple_representation(&self) -> Vec<Vec<Vec<usize>>> {
        self.hgg.simple_representation()
    }
}

impl<K, V> Default for HggLite<K, V>
where
    K: MetricPoint,
{
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct HggNode<K, V, HK> {
    key: K,
    value: V,
    /// Contains the edges of each layer of the graph on which this exists.
    layers: Vec<HeaderVec<HggHeader<HK>, HggEdge<HK>>>,
    /// Forms a linked list through the nodes that creates the freshening order.
    next: usize,
}

impl<K, V, HK> HggNode<K, V, HK> {
    fn layers(&self) -> usize {
        self.layers.len()
    }
}

#[derive(Debug)]
struct NodeStorage<K, V, HK, Strategy>(Vec<HggNode<K, V, HK>>, PhantomData<Strategy>);

impl<K, V, HK, Strategy> Deref for NodeStorage<K, V, HK, Strategy> {
    type Target = Vec<HggNode<K, V, HK>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<K, V, HK, Strategy> DerefMut for NodeStorage<K, V, HK, Strategy> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Collection for retrieving entries based on key proximity in a metric space.
#[derive(Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "K: Serialize, V: Serialize, NodeStorage<K, V, HK, Strategy>: Serialize",
        deserialize = "K: Deserialize<'de>, V: Deserialize<'de>, NodeStorage<K, V, HK, Strategy>: Deserialize<'de>"
    ))
)]
struct HggCore<K, V, HK, Strategy> {
    /// The nodes of the graph. These nodes internally contain their own edges which form
    /// subgraphs of decreasing size called "layers". The lowest layer contains every node,
    /// while the highest layer contains only one node.
    nodes: NodeStorage<K, V, HK, Strategy>,
    /// This node exists on the top layer, and is the root of all searches.
    root: usize,
    /// The node which has been cleaned up/inserted most recently.
    freshest: usize,
    /// The number of edges in the graph on each layer.
    edges: Vec<usize>,
    /// The number of nodes in the graph on each layer.
    node_counts: Vec<usize>,
    /// Number of freshens per insert.
    freshens: usize,
    /// Whether to exclude all keys for which the distance has been calculated in kNN search.
    exclude_all_searched: bool,
    /// Determines the number of nearest neighbors used for inserting.
    insert_knn: usize,
    _phantom: PhantomData<Strategy>,
}

impl<K, V, HK, Strategy> HggCore<K, V, HK, Strategy> {
    /// Creates a new [`Hgg`]. It will be empty and begin with default settings.
    pub fn new() -> Self {
        Self {
            nodes: NodeStorage(vec![], PhantomData),
            root: 0,
            freshest: 0,
            edges: vec![],
            node_counts: vec![],
            freshens: 1,
            exclude_all_searched: false,
            insert_knn: 64,
            _phantom: PhantomData,
        }
    }

    /// Default value: `1`
    ///
    /// Increase the parameter `freshens` to freshen stale nodes in the graph. The higher this value, the longer the
    /// insert will take. However, in the long run, freshening may improve insert performance. It is recommended
    /// to benchmark with your data both the insert and lookup performance against recall using this
    /// parameter to determine the right value for you. The default should be fine for most users.
    pub fn freshens(self, freshens: usize) -> Self {
        Self { freshens, ..self }
    }

    /// Default value: `false`
    ///
    /// If this is true, when doing a kNN search, any key which has already had its distance computed will not be
    /// computed again. kNN search (and insertion) is faster when this is set to `false` for keys with cheap
    /// distance functions. If your distance function is expensive, benchmark Hgg with this parameter set to `true`.
    /// For some distance functions/key types this will be better, and for some it will be worse.
    /// Benchmark your data and observe the recall curve to find out.
    pub fn exclude_all_searched(self, exclude_all_searched: bool) -> Self {
        Self {
            exclude_all_searched,
            ..self
        }
    }

    /// Default value: `64`
    ///
    /// This controls the number of nearest neighbors used during insertion. Setting this higher will cause the graph
    /// to become more connected if your data has thick Voronoi boundaries. If this is true of your dataset (
    /// usually due to using hamming distance or high dimensionality), then you may want to intentionally
    /// set this lower to avoid consuming too much memory, which can decrease performance if slower
    /// memory (such as swap space) is used.
    ///
    /// For all datasets, this value correlates positively with insertion time (inversely with speed). If you want insertions to go faster,
    /// consider decreasing this value.
    pub fn insert_knn(self, insert_knn: usize) -> Self {
        assert!(
            insert_knn > 0,
            "insert_knn cant be less than 1 or graph will become disconnected"
        );
        Self { insert_knn, ..self }
    }

    /// Get the (key, value) pair of a node.
    pub fn get(&self, node: usize) -> Option<(&K, &V)> {
        self.nodes.get(node).map(|node| (&node.key, &node.value))
    }

    /// Get the key of a node.
    pub fn get_key(&self, node: usize) -> Option<&K> {
        self.nodes.get(node).map(|node| &node.key)
    }

    /// Get the value of a node.
    pub fn get_value(&self, node: usize) -> Option<&V> {
        self.nodes.get(node).map(|node| &node.value)
    }

    /// Checks if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the number of (key, value) pairs added to the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of edges in the graph on each layer.
    pub fn edges(&self) -> Vec<usize> {
        self.edges.clone()
    }

    /// Returns the number of layers in the graph.
    pub fn layers(&self) -> usize {
        if self.is_empty() {
            0
        } else {
            self.nodes[self.root].layers()
        }
    }

    pub fn histogram_layer_nodes(&self) -> Vec<usize> {
        let mut layers = vec![0; self.layers()];
        for node in &*self.nodes {
            for layer in &mut layers[0..node.layers()] {
                *layer += 1;
            }
        }
        layers
    }

    pub fn histogram_neighbors(&self) -> Vec<Vec<(usize, usize)>> {
        let mut histograms = vec![];
        for layer in 0..self.layers() {
            let mut histogram = vec![];
            for edges in self
                .nodes
                .iter()
                .filter_map(|node| node.layers.get(layer).map(|layer_node| layer_node.len()))
            {
                match histogram.binary_search_by_key(&edges, |&(search_edges, _)| search_edges) {
                    Ok(pos) => histogram[pos].1 += 1,
                    Err(pos) => histogram.insert(pos, (edges, 1)),
                }
            }
            histograms.push(histogram);
        }
        histograms
    }

    pub fn average_neighbors(&self) -> Vec<f64> {
        self.edges()
            .into_iter()
            .zip(self.histogram_layer_nodes())
            .map(|(edges, nodes)| edges as f64 * 2.0 / nodes as f64)
            .collect()
    }

    pub fn simple_representation(&self) -> Vec<Vec<Vec<usize>>> {
        let mut layers = vec![vec![]; self.layers()];
        for node in &*self.nodes {
            for (layer, layer_node) in node.layers.iter().enumerate() {
                layers[layer].push(
                    layer_node
                        .as_slice()
                        .iter()
                        .map(|HggEdge { neighbor, .. }| neighbor.node)
                        .collect::<Vec<_>>(),
                );
            }
        }
        layers
    }

    fn layer_node_weak(&self, layer: usize, node: usize) -> HVec<HK> {
        unsafe { HVec(self.nodes[node].layers[layer].weak()) }
    }
}
impl<K, V, HK, Strategy> HggCore<K, V, HK, Strategy>
where
    K: MetricPoint,
    Self: HggInternal<K = K, V = V, HK = HK>,
{
    /// Insert a (key, value) pair.
    fn insert(&mut self, key: K, value: V) -> usize {
        // Add the node (it will be added this way regardless).
        let node = self.nodes.len();
        // Create the node.
        // The current freshest node's `next` is the stalest node, which will subsequently become
        // the freshest when freshened. If this is the only node, looking up the freshest node will fail.
        // Due to that, we set this node's next to itself if its the only node.
        let next = if node == 0 {
            0
        } else {
            self.nodes[self.freshest].next
        };
        self.nodes.push(HggNode {
            key,
            value,
            layers: vec![],
            next,
        });
        // The previous freshest node should now be freshened right before this node, as this node is now fresher.
        // Even if this is the only node, this will still work because this node still comes after itself in the freshening order.
        self.nodes[self.freshest].next = node;
        // This is now the freshest node.
        self.freshest = node;

        if node == 0 {
            // Push the new layer 0.
            self.add_node_layer(node);
            self.edges.push(0);
            self.node_counts.push(1);
            // Set the root.
            self.root = 0;
            return 0;
        }

        // Find nearest neighbor path via greedy search.
        let path = self.search_path(&self.nodes[node].key);

        for (layer, (found, distance)) in path.into_iter().enumerate() {
            // Add the new layer to this node.
            self.add_node_layer(node);
            self.node_counts[layer] += 1;

            // If we are on the last layer, we now have exactly two nodes on the last layer,
            // and it is time to create a new layer.
            if layer == self.layers() - 1 {
                // Add edge to nearest neighbor (the only other node in this layer, the old root).
                self.layer_add_edge(layer, found, node);
                // Set the root to this node.
                self.root = node;
                // Create the new layer (totally empty).
                self.add_node_layer(node);
                self.edges.push(0);
                self.node_counts.push(1);
                // No need to do the remaining checks.
                break;
            }

            self.optimize_layer_neighborhood(layer, node, found, distance, false);

            // Check if any surrounding nodes are on the next layer.
            if self.any_neighbors_above_layer(layer, node) {
                // If any of the neighbors are on the next layer up, we don't need to add this node to more layers.
                break;
            }
        }

        // Freshen the graph to clean up older nodes.
        self.freshen();

        node
    }

    /// Optimizes a number of stale nodes equal to `self.freshens`.
    ///
    /// You do not need to call this yourself, as it is called on insert.
    fn freshen(&mut self) {
        let freshens = self.freshens;
        for node in self.stales().take(freshens).collect::<Vec<_>>() {
            // Start by reducing as many connections as possible on the layers it exists.
            for layer in 0..self.nodes[node].layers() {
                self.optimize_layer_neighborhood(layer, node, node, K::Metric::zero(), true)
            }
            // Next we want to check, starting on this node's highest layer, if it should be added to the next layer.
            for layer in self.nodes[node].layers() - 1..self.layers() {
                // An edge case occurs if we are on the top layer.
                if layer == self.layers() - 1 {
                    // Check if this node is the root node.
                    if node != self.root {
                        // In this case, we just raised this node to this layer, and now we need to add a new layer.
                        // Set the root to this node.
                        self.root = node;
                        // Create the new layer (totally empty).
                        self.add_node_layer(node);
                        self.edges.push(0);
                        self.node_counts.push(1);
                    }
                    // In either case, we are now done, as the top layer now has one node,
                    // regardless of if it is this node or the other node.
                    break;
                }

                // Check if there are no neighbors on the next layer.
                // Check if any surrounding nodes are on the next layer.
                if self
                    .layer_node_weak(layer, node)
                    .as_slice()
                    .iter()
                    .any(|HggEdge { neighbor, .. }| self.nodes[neighbor.node].layers() > layer + 1)
                {
                    // If any of the neighbors are on the next layer up, we don't need to add this node to more layers.
                    break;
                }

                // Add the new layer to this node.
                self.add_node_layer(node);
                // Note that since we are adding it to the NEXT layer, this (and further uses of layer)
                // are layer + 1.
                self.node_counts[layer + 1] += 1;

                // Find the nearest neighbor on the next layer (by greedy search).
                let (nn, distance) = self
                    .search_to_layer(layer + 1, &self.nodes[node].key)
                    .unwrap();

                self.optimize_layer_neighborhood(layer + 1, node, nn, distance, false);
            }
        }
    }

    /// Searches for the nearest neighbor greedily from the top layer to the bottom.
    ///
    /// This is faster than calling [`Hgg::search_knn`] with `num` of `1`.
    ///
    /// Returns `(node, distance)`.
    fn search(&self, query: &K) -> Option<(usize, K::Metric)> {
        self.search_to_layer(0, query)
    }

    /// Produces the stalest nodes and marks them as now the freshest nodes when consumed.
    ///
    /// This iterator is infinite, and will iterate through every entry in a specific order before repeating.
    fn stales(&mut self) -> impl Iterator<Item = usize> + '_ {
        let mut node = self.freshest;
        core::iter::from_fn(move || {
            node = self.nodes[node].next;
            self.freshest = node;
            Some(node)
        })
    }

    /// Updates the `HeaderVecWeak` in neighbors of this node.
    fn update_weak(&mut self, mut node: HVec<HK>, previous: *const (), add_last: bool) {
        let old_len = if add_last { node.len() } else { node.len() - 1 };
        let weak = node.weak();
        for HggEdge { neighbor, .. } in &mut node[..old_len] {
            if let Some(edge) = neighbor
                .as_mut_slice()
                .iter_mut()
                .find(|edge| edge.neighbor.is(previous))
            {
                edge.neighbor = weak.weak();
            } else {
                unreachable!("fatal; we did not find the edge in the neighbor");
            }
        }
    }

    fn layer_add_edge(&mut self, layer: usize, a: usize, b: usize) {
        self.layer_add_edge_weak(
            layer,
            &mut self.layer_node_weak(layer, a),
            &mut self.layer_node_weak(layer, b),
        );
    }

    fn layer_add_edge_dedup_weak(
        &mut self,
        layer: usize,
        a: &mut HVec<HK>,
        b: &mut HVec<HK>,
    ) -> bool {
        if !a.contains(b) {
            self.layer_add_edge_weak(layer, a, b);
            true
        } else {
            false
        }
    }

    /// Finds the nearest neighbor to the query key starting from the `from` node using greedy search.
    ///
    /// Returns `(node, distance)`.
    fn search_layer_from_weak(
        &self,
        from: HVec<HK>,
        from_distance: K::Metric,
        query: &K,
    ) -> (HVec<HK>, K::Metric) {
        let mut best_weak = from;
        let mut best_distance = from_distance;

        while let Some((neighbor_weak, distance)) = self.best_neighbor_distance(&best_weak, query) {
            if distance < best_distance {
                best_weak = neighbor_weak.weak();
                best_distance = distance;
            } else {
                break;
            }
        }
        (best_weak, best_distance)
    }

    /// Searches for the nearest neighbor greedily from the top layer to the bottom.
    ///
    /// This is faster than calling [`Hgg::search_knn`] with `num` of `1`.
    ///
    /// Returns `(node, distance)`.
    fn search_to_layer(&self, final_layer: usize, query: &K) -> Option<(usize, K::Metric)> {
        if self.is_empty() {
            return None;
        }
        let mut node = self.root;
        let mut distance = self.nodes[node].key.distance(query);
        // This assumes that the top layer only contains one node (as it should).
        for layer in (final_layer..self.layers() - 1).rev() {
            let node_weak = self.layer_node_weak(layer, node);
            let (new_node, new_distance) = self.search_layer_from_weak(node_weak, distance, query);
            node = new_node.node;
            distance = new_distance;
        }
        Some((node, distance))
    }

    /// Searches for the nearest neighbor greedily from the top layer to the bottom.
    ///
    /// This is faster than calling [`Hgg::search_knn`] with `num` of `1`.
    ///
    /// Returns the greedy search result on every layer as `(node, distance)`.
    fn search_path(&self, query: &K) -> Vec<(usize, K::Metric)> {
        if self.is_empty() {
            return vec![];
        }
        let init_node = self.root;
        let init_distance = self.nodes[init_node].key.distance(query);
        let mut path: Vec<(usize, K::Metric)> = iter::repeat_with(|| (init_node, init_distance))
            .take(self.layers())
            .collect();
        // This assumes that the top layer only contains one node (as it should).
        for layer in (0..self.layers() - 1).rev() {
            let node = self.layer_node_weak(layer, path[layer + 1].0);
            let distance = path[layer + 1].1;
            let (node, distance) = self.search_layer_from_weak(node, distance, query);
            path[layer] = (node.node, distance);
        }
        path
    }

    /// Finds the knn greedily from a starting node `from`.
    ///
    /// Returns (node, distance, searched) pairs. `searched` will always be true, so you can ignore it.
    fn search_layer_knn_from_weak(
        &self,
        from: HVec<HK>,
        from_distance: K::Metric,
        query: &K,
        num: usize,
    ) -> Vec<(HVec<HK>, K::Metric, bool)> {
        if num == 0 {
            return vec![];
        }
        // Perform a greedy search first to save time.
        let (from, from_distance) = self.search_layer_from_weak(from, from_distance, query);
        // Contains the index and the distance as a pair.
        let mut bests = vec![(from.weak(), from_distance, false)];

        // This set is used to more quickly determine if a node is contained in the best set.
        let mut exclude = HashSet::with_capacity_and_hasher(
            num.saturating_mul(2),
            RandomState::with_seeds(0, 0, 0, 0),
        );
        exclude.insert(from);

        loop {
            if let Some((previous_node, _, searched)) =
                bests.iter_mut().find(|&&mut (_, _, searched)| !searched)
            {
                // Set this as searched (we are searching it now).
                *searched = true;
                // Erase the reference to the search node (to avoid lifetime & borrowing issues).
                let previous_node = previous_node.weak();
                for edge in previous_node.as_slice() {
                    // TODO: Try this as a BTreeSet.
                    // Make sure that we don't have a copy of this node already or we will get duplicates.
                    if exclude.contains(&edge.neighbor) {
                        continue;
                    }

                    // Compute the distance from the query.
                    let distance = query.distance(self.edge_get_key(edge));
                    // If we dont have enough yet, add it.
                    if bests.len() < num {
                        bests.insert(
                            bests.partition_point(|&(_, best_distance, _)| {
                                best_distance <= distance
                            }),
                            (edge.neighbor.weak(), distance, false),
                        );
                        exclude.insert(edge.neighbor.weak());
                    } else if distance < bests.last().unwrap().1 {
                        // Otherwise only add it if its better than the worst item we have.
                        // Remove the worst item we have now and exclude it if exclude_all_searched is set.
                        if self.exclude_all_searched {
                            let (old_node, _, _) = bests.pop().unwrap();
                            exclude.remove(&old_node);
                        } else {
                            bests.pop();
                        }
                        exclude.insert(edge.neighbor.weak());
                        bests.insert(
                            bests.partition_point(|&(_, best_distance, _)| {
                                best_distance <= distance
                            }),
                            (edge.neighbor.weak(), distance, false),
                        );
                    }
                }
            } else {
                return bests;
            }
        }
    }

    fn any_neighbors_above_layer(&self, layer: usize, node: usize) -> bool {
        self.layer_node_weak(layer, node)
            .as_slice()
            .iter()
            .any(|HggEdge { neighbor, .. }| self.nodes[neighbor.node].layers() > layer + 1)
    }

    fn layer_add_edge_weak(&mut self, layer: usize, a: &mut HVec<HK>, b: &mut HVec<HK>) {
        // Add the edge from a to b.
        let edge = self.make_edge_to_node(b);
        // Insert it onto the end.
        if let Some(previous) = a.push(edge) {
            // Update the strong reference first.
            unsafe {
                self.nodes[a.node].layers[layer].update(a.weak().0);
            }
            // Update the neighbors.
            self.update_weak(a.weak(), previous, false);
        }

        // Add the edge from b to a.
        let edge = self.make_edge_to_node(a);
        // Insert it onto the end.
        if let Some(previous) = b.push(edge) {
            // Update the strong reference first.
            unsafe {
                self.nodes[b.node].layers[layer].update(b.weak().0);
            }
            // Update the neighbors.
            self.update_weak(b.weak(), previous, true);
        }

        self.edges[layer] += 1;
    }

    fn best_neighbor_distance(&self, node: &HVec<HK>, query: &K) -> Option<(HVec<HK>, K::Metric)> {
        node.as_slice()
            .iter()
            .map(|edge| {
                (
                    edge.neighbor.weak(),
                    self.edge_get_key(edge).distance(query),
                )
            })
            .min_by_key(|(_, distance)| *distance)
    }

    /// Searches for the nearest neighbor greedily from the top layer to the bottom.
    ///
    /// This implementation starts with 1nn search until the bottom layer and then
    /// performs kNN search.
    ///
    /// Returns `(node, distance)`.
    fn search_knn(&self, query: &K, num: usize) -> impl Iterator<Item = (usize, K::Metric)> + '_ {
        let mapfn = |(weak, distance, _): (HVec<HK>, K::Metric, bool)| (weak.node, distance);
        if let Some((node, distance)) = self.search_to_layer(0, query) {
            self.search_layer_knn_from_weak(self.layer_node_weak(0, node), distance, query, num)
                .into_iter()
                .map(mapfn)
        } else {
            vec![].into_iter().map(mapfn)
        }
    }
}

impl<K, V, HK, Strategy> Default for HggCore<K, V, HK, Strategy> {
    fn default() -> Self {
        Self::new()
    }
}

trait HggInternal {
    type K: MetricPoint;
    type V;
    type HK;
    fn make_edge_to_node(&self, node: &HVec<Self::HK>) -> HggEdge<Self::HK>;
    fn edge_get_key<'a>(&'a self, edge: &'a HggEdge<Self::HK>) -> &'a Self::K;
    fn node_get_key<'a>(&'a self, node: &'a HVec<Self::HK>) -> &'a Self::K;
    fn add_node_layer(&mut self, node: usize);
    /// `layer` is the layer to optimize on.
    /// `node` is the node we are optimizing.
    /// `found` is the node we found that is closest to the target node `node`.
    /// `distance` is the distance of `found` from `node`.
    /// `reconnect` tells us if the node is already connected and needs to be disconnected before optimizing.
    fn optimize_layer_neighborhood(
        &mut self,
        layer: usize,
        node: usize,
        found: usize,
        distance: <Self::K as MetricPoint>::Metric,
        reconnect: bool,
    );
}

impl<K, V> HggInternal for HggCore<K, V, K, StrategyRegular>
where
    K: MetricPoint + Clone,
{
    type K = K;
    type V = V;
    type HK = K;

    fn make_edge_to_node(&self, node: &HVec<Self::HK>) -> HggEdge<Self::HK> {
        HggEdge {
            key: node.key.clone(),
            neighbor: node.weak(),
        }
    }

    fn edge_get_key<'a>(&'a self, edge: &'a HggEdge<Self::HK>) -> &'a K {
        &edge.key
    }

    fn node_get_key<'a>(&'a self, node: &'a HVec<Self::HK>) -> &'a K {
        &node.key
    }

    fn add_node_layer(&mut self, node: usize) {
        let key = self.nodes[node].key.clone();
        self.nodes[node]
            .layers
            .push(HeaderVec::new(HggHeader { key, node }));
    }

    fn optimize_layer_neighborhood(
        &mut self,
        layer: usize,
        node: usize,
        found: usize,
        distance: K::Metric,
        reconnect: bool,
    ) {
        // Get the node's weak ref.
        let mut node = self.layer_node_weak(layer, node);

        // Do a knn search on this layer, starting at the found node.
        let mut knn: Vec<(usize, K)> = self
            .search_layer_knn_from_weak(
                self.layer_node_weak(layer, found),
                distance,
                &node.key,
                self.insert_knn,
            )
            .into_iter()
            .skip(if reconnect { 1 } else { 0 })
            .map(|(neighbor, _, _)| (neighbor.node, neighbor.key.clone()))
            .collect();

        // If we are reconnecting the node, we need to disconnect its edges first.
        if reconnect {
            self.disconnect_layer(layer, &mut node, &mut knn);
        }

        // The initial neighbors only includes the edge we just added.
        let mut neighbors: Vec<K> = Vec::with_capacity(self.insert_knn);

        let mut knn_index = 0;
        'knn_next: while let Some((target_node, target_key)) = knn.get(knn_index).cloned() {
            // Get this node's distance.
            let to_beat = node.key.distance(&target_key);
            // Check if the node is colocated.
            if to_beat == Zero::zero() {
                // In this case, add an edge (with dedup) between them to make sure there is a path.
                self.layer_add_edge_dedup_weak(
                    layer,
                    &mut node,
                    &mut self.layer_node_weak(layer, target_node),
                );
                knn_index += 1;
                continue 'knn_next;
            }

            if neighbors
                .iter()
                .any(|key| key.distance(&target_key) < to_beat)
            {
                // If any are better, then no optimization needed.
                knn_index += 1;
                continue 'knn_next;
            }

            // Go through the nearest neighbors in order from best to worst.
            for (nn, nn_key) in knn.iter().cloned() {
                // Compute the distance to the target from the nn.
                let nn_distance = nn_key.distance(&target_key);
                // Add the node as a neighbor (closer or not).
                // This will update the weak ref if necessary.
                if self.layer_add_edge_dedup_weak(
                    layer,
                    &mut self.layer_node_weak(layer, nn),
                    &mut node,
                ) {
                    neighbors.push(nn_key);
                }
                // Check if this node is closer to the target than `from`.
                if nn_distance < to_beat {
                    // The greedy path now exists, so exit.
                    knn_index += 1;
                    continue 'knn_next;
                }
            }
            unreachable!(
                "we should always be able to connect to all the neighbors using themselves"
            );
        }
    }
}

impl<K, V> HggCore<K, V, K, StrategyRegular>
where
    K: MetricPoint + Clone,
{
    /// Internal function for disconnecting a node from the graph on the layer this HVec exists on.
    ///
    /// Returns nodes as usize because as nodes are re-added, it is possible that neighbors reallocate
    /// and break the weak pointers.
    ///
    /// Returns (node, distance) pairs.
    fn disconnect_layer(
        &mut self,
        layer: usize,
        node: &mut HVec<K>,
        neighbors: &mut Vec<(usize, K)>,
    ) {
        let node_key = node.key.clone();
        let ptr = node.ptr();
        self.edges[layer] -= node.len();
        for HggEdge { neighbor, key } in node.as_mut_slice() {
            if !neighbors.iter().any(|&(cn, _)| cn == neighbor.node) {
                let distance = node_key.distance(key);
                let pos = neighbors.partition_point(|(_, k)| node_key.distance(k) <= distance);
                neighbors.insert(pos, (neighbor.node, key.clone()));
            }
            neighbor.retain(|HggEdge { neighbor, .. }| !neighbor.is(ptr));
        }
        node.retain(|_| false);
    }
}

impl<K, V> HggInternal for HggCore<K, V, (), StrategyLite>
where
    K: MetricPoint,
{
    type K = K;
    type V = V;
    type HK = ();

    fn make_edge_to_node(&self, node: &HVec<Self::HK>) -> HggEdge<Self::HK> {
        HggEdge {
            key: (),
            neighbor: node.weak(),
        }
    }

    fn edge_get_key<'a>(&'a self, edge: &'a HggEdge<Self::HK>) -> &'a K {
        &self.nodes[edge.neighbor.node].key
    }

    fn node_get_key<'a>(&'a self, node: &'a HVec<Self::HK>) -> &'a K {
        &self.nodes[node.node].key
    }

    fn add_node_layer(&mut self, node: usize) {
        self.nodes[node]
            .layers
            .push(HeaderVec::new(HggHeader { key: (), node }));
    }

    fn optimize_layer_neighborhood(
        &mut self,
        layer: usize,
        node: usize,
        found: usize,
        distance: K::Metric,
        reconnect: bool,
    ) {
        // Get the node's weak ref.
        let mut node = self.layer_node_weak(layer, node);

        // Do a knn search on this layer, starting at the found node.
        let mut knn: Vec<usize> = self
            .search_layer_knn_from_weak(
                self.layer_node_weak(layer, found),
                distance,
                &self.nodes[node.node].key,
                self.insert_knn,
            )
            .into_iter()
            .skip(if reconnect { 1 } else { 0 })
            .map(|(neighbor, _, _)| neighbor.node)
            .collect();

        // If we are reconnecting the node, we need to disconnect its edges first.
        if reconnect {
            self.disconnect_layer(layer, &mut node, &mut knn);
        }

        // The initial neighbors only includes the edge we just added.
        let mut neighbors: Vec<usize> = Vec::with_capacity(self.insert_knn);

        let mut knn_index = 0;
        'knn_next: while let Some(target_node) = knn.get(knn_index).copied() {
            // Get this node's distance.
            let to_beat = self.nodes[node.node]
                .key
                .distance(&self.nodes[target_node].key);
            // Check if the node is colocated.
            if to_beat == Zero::zero() {
                // In this case, add an edge (with dedup) between them to make sure there is a path.
                self.layer_add_edge_dedup_weak(
                    layer,
                    &mut node,
                    &mut self.layer_node_weak(layer, target_node),
                );
                knn_index += 1;
                continue 'knn_next;
            }

            if neighbors.iter().any(|&neighbor| {
                self.nodes[neighbor]
                    .key
                    .distance(&self.nodes[target_node].key)
                    < to_beat
            }) {
                // If any are better, then no optimization needed.
                knn_index += 1;
                continue 'knn_next;
            }

            // Go through the nearest neighbors in order from best to worst.
            for nn in knn.iter().copied() {
                // Compute the distance to the target from the nn.
                let nn_distance = self.nodes[nn].key.distance(&self.nodes[target_node].key);
                // Add the node as a neighbor (closer or not).
                // This will update the weak ref if necessary.
                if self.layer_add_edge_dedup_weak(
                    layer,
                    &mut self.layer_node_weak(layer, nn),
                    &mut node,
                ) {
                    neighbors.push(nn);
                }
                // Check if this node is closer to the target than `from`.
                if nn_distance < to_beat {
                    // The greedy path now exists, so exit.
                    knn_index += 1;
                    continue 'knn_next;
                }
            }
            unreachable!(
                "we should always be able to connect to all the neighbors using themselves"
            );
        }
    }
}

impl<K, V> HggCore<K, V, (), StrategyLite>
where
    K: MetricPoint,
{
    /// Internal function for disconnecting a node from the graph on the layer this HVec exists on.
    ///
    /// Returns nodes as usize because as nodes are re-added, it is possible that neighbors reallocate
    /// and break the weak pointers.
    ///
    /// Returns (node, distance) pairs.
    fn disconnect_layer(&mut self, layer: usize, node: &mut HVec<()>, neighbors: &mut Vec<usize>) {
        let ptr = node.ptr();
        self.edges[layer] -= node.len();
        let node_index = node.node;
        for HggEdge { neighbor, .. } in node.as_mut_slice() {
            if !neighbors.iter().any(|&cn| cn == neighbor.node) {
                let distance = self.nodes[neighbor.node]
                    .key
                    .distance(&self.nodes[node_index].key);
                let pos = neighbors.partition_point(|&cnn| {
                    self.nodes[cnn].key.distance(&self.nodes[node_index].key) <= distance
                });
                neighbors.insert(pos, neighbor.node);
            }
            neighbor.retain(|HggEdge { neighbor, .. }| !neighbor.is(ptr));
        }
        node.retain(|_| false);
    }
}
