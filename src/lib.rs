#![no_std]
extern crate alloc;

#[cfg(feature = "stats")]
mod stats;
#[cfg(test)]
mod unit_tests;

#[cfg(feature = "stats")]
pub use stats::*;

use alloc::vec;
use alloc::vec::Vec;
use core::cmp;
use core::fmt::Debug;
use core::marker::PhantomData;
use core::ops::Deref;
use core::ops::DerefMut;
use header_vec::{HeaderVec, HeaderVecWeak};
use num_traits::AsPrimitive;
use space::MetricPoint;

#[derive(Debug)]
struct HrcEdge<K> {
    key: K,
    neighbor: HVec<K>,
}

#[derive(Debug)]
struct HrcHeader<K> {
    key: K,
    node: usize,
}

#[derive(Debug)]
struct HVec<K>(HeaderVecWeak<HrcHeader<K>, HrcEdge<K>>);

impl<K> HVec<K> {
    fn weak(&self) -> Self {
        unsafe { Self(self.0.weak()) }
    }

    fn neighbors_mut(&mut self) -> impl Iterator<Item = &mut Self> + '_ {
        self.as_mut_slice()
            .iter_mut()
            .map(|HrcEdge { neighbor, .. }| neighbor)
    }
}

impl<K> HVec<K>
where
    K: MetricPoint,
{
    fn neighbors_distance<'a, D>(&'a self, query: &'a K) -> impl Iterator<Item = (Self, D)> + 'a
    where
        D: Copy + Ord + 'static,
        u64: AsPrimitive<D>,
    {
        self.as_slice()
            .iter()
            .map(move |HrcEdge { key, neighbor }| (neighbor.weak(), query.distance(key).as_()))
    }
}

impl<K> Deref for HVec<K> {
    type Target = HeaderVecWeak<HrcHeader<K>, HrcEdge<K>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<K> DerefMut for HVec<K> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug)]
struct HrcNode<K, V> {
    key: K,
    value: V,
    /// Contains the edges of each layer of the graph on which this exists.
    layers: Vec<HeaderVec<HrcHeader<K>, HrcEdge<K>>>,
    /// Forms a linked list through the nodes that creates the freshening order.
    next: usize,
}

impl<K, V> HrcNode<K, V> {
    fn layers(&self) -> usize {
        self.layers.len()
    }
}

/// Collection for retrieving entries based on key proximity in a metric space.
///
/// Optional type parameter `D` can be set to a smaller unsigned integer (`u8`, `u16`, `u32`) ONLY
/// if you know that the distance metric cannot overflow this unsigned integer. If it does, then
/// you will have issues. `f32` metric sources can be safely used with `u32`, as only the lower
/// 32 bits of the `u64` is utilized in that case, but `f64` CANNOT be used with anything smaller than `u64`.
/// There is no advantage to using `u128` as the distance metric is produced as `u64`.
/// This parameter DOES affect the performance in benchmarks, though the amount may vary between machines.
/// Smaller integer types will yield better performance, but the difference will likely be less than 25%.
/// On one machine, u64 -> u32 yielded 10-20% performance, but u32 -> u16 yielded less than 1%.
#[derive(Debug)]
pub struct Hrc<K, V, D = u64> {
    /// The nodes of the graph. These nodes internally contain their own edges which form
    /// subgraphs of decreasing size called "layers". The lowest layer contains every node,
    /// while the highest layer contains only one node.
    nodes: Vec<HrcNode<K, V>>,
    /// The node which has been cleaned up/inserted most recently.
    freshest: usize,
    /// The number of edges in the graph.
    edges: usize,
    /// The highest number of edges of any node.
    most_edges: usize,
    /// Number of freshens per insert.
    freshens: usize,
    /// This allows a consistent number to be used for distance storage during usage.
    _phantom: PhantomData<D>,
}

impl<K, V, D> Hrc<K, V, D> {
    /// Creates a new [`Hrc`]. It will be empty and begin with default settings.
    pub fn new() -> Self {
        Self {
            nodes: vec![],
            freshest: 0,
            edges: 0,
            most_edges: 0,
            freshens: 2,
            _phantom: PhantomData,
        }
    }

    /// Default value: `2`
    ///
    /// Increase the parameter `freshens` to freshen stale nodes in the graph. Freshening nodes can improve your
    /// recall curve. The higher this value, the longer the insert will take. However, in the long run, freshening will
    /// improve insert performance. Counter-intuitively, if you freshen too infrequently, your graph will become
    /// too accurate/too connected. If the graph grows too accurate, it worsens the recall curve and the
    /// insertion time will grow at a rate that is closer to exponential. For all intents and purposes,
    /// increasing the hight of the recall curve (Y = queries per second, X = recall) is your primary goal,
    /// so set `freshens` at the value which does this for your dataset.
    ///
    /// A value of at least `2` is recommended for `freshens`.
    pub fn freshens(self, freshens: usize) -> Self {
        Self { freshens, ..self }
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

    /// Returns the number of edges in the graph.
    pub fn edges(&self) -> usize {
        self.edges
    }

    pub fn histogram(&self) -> Vec<Vec<(usize, usize)>> {
        let mut histograms = vec![];
        for layer in 0.. {
            let mut histogram = vec![];
            for edges in self.nodes.iter().filter_map(|node| {
                if node.layers() > layer {
                    Some(node.layers[layer].len())
                } else {
                    None
                }
            }) {
                match histogram.binary_search_by_key(&edges, |&(search_edges, _)| search_edges) {
                    Ok(pos) => histogram[pos].1 += 1,
                    Err(pos) => histogram.insert(pos, (edges, 1)),
                }
            }
            if histogram.is_empty() {
                break;
            } else {
                histograms.push(histogram);
            }
        }
        histograms
    }

    pub fn simple_representation(&self) -> Vec<Vec<usize>> {
        self.nodes
            .iter()
            .map(|node| {
                node.layers[0]
                    .as_slice()
                    .iter()
                    .map(|HrcEdge { neighbor, .. }| neighbor.node)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }
}

impl<K, V, D> Hrc<K, V, D>
where
    K: MetricPoint + Clone,
    D: Copy + Ord + 'static,
    u64: AsPrimitive<D>,
{
    /// This gets the "optimal" k for inserting.
    ///
    /// This can be used to see what HRC is using internally for its `k` value during kNN searches.
    /// This value is not necessarily optimal for searches by the user, and is specifically set to
    /// prevent an explosion of graph connectivity and prevent a collapse of graph connectivity.
    /// The user should use a `k` value that corresponds to the recall they desire for the number
    /// of actual neighbors they want.
    pub fn optimal_k(&self) -> usize {
        (self.edges + 1).saturating_mul(3) / self.len()
    }

    fn node_weak(&self, layer: usize, node: usize) -> HVec<K> {
        unsafe { HVec(self.nodes[node].layers[layer].weak()) }
    }

    /// Updates the `HeaderVecWeak` in neighbors of this node.
    fn update_weak(&mut self, mut node: HVec<K>, previous: *const ()) {
        let weak = node.weak();
        for neighbor in node.neighbors_mut() {
            for neighbor_neighbor in neighbor.neighbors_mut() {
                if neighbor_neighbor.is(previous) {
                    *neighbor_neighbor = weak.weak();
                }
            }
        }
    }

    fn add_edge_weak(&mut self, layer: usize, a: &mut HVec<K>, b: &mut HVec<K>) {
        let a_key = a.key.clone();
        let b_key = b.key.clone();

        // Add the edge from a to b.
        let edge = HrcEdge {
            key: b_key,
            neighbor: b.weak(),
        };
        if let Some(previous) = a.push(edge) {
            // Update the strong reference first.
            unsafe {
                self.nodes[a.node].layers[layer].update(a.weak().0);
            }
            // Update the neighbors.
            self.update_weak(a.weak(), previous);
        }

        // Add the edge from b to a.
        let edge = HrcEdge {
            key: a_key,
            neighbor: a.weak(),
        };
        if let Some(previous) = b.push(edge) {
            // Update the strong reference first.
            unsafe {
                self.nodes[b.node].layers[layer].update(b.weak().0);
            }
            // Update the neighbors.
            self.update_weak(b.weak(), previous);
        }

        self.edges += 1;
        self.most_edges = cmp::max(self.most_edges, cmp::max(a.len(), b.len()));
    }

    fn add_edge(&mut self, layer: usize, a: usize, b: usize) {
        self.add_edge_weak(
            layer,
            &mut self.node_weak(layer, a),
            &mut self.node_weak(layer, b),
        );
    }

    fn add_edge_dedup_weak(&mut self, layer: usize, a: &mut HVec<K>, b: &mut HVec<K>) -> bool {
        let b_ptr = b.ptr();
        if !a.neighbors_mut().any(|neighbor| neighbor.is(b_ptr)) {
            self.add_edge_weak(layer, a, b);
            true
        } else {
            false
        }
    }

    /// Searches for the nearest neighbor greedily.
    ///
    /// This is faster than calling [`Hrc::search_knn`] with `num` of `1`.
    ///
    /// Returns `(node, distance)`.
    pub fn search(&self, layer: usize, query: &K) -> Option<(usize, D)> {
        if self.is_empty() {
            None
        } else {
            Some(self.search_from(layer, 0, query))
        }
    }

    /// Finds the knn greedily.
    ///
    /// Returns (node, distance) pairs from closest to furthest.
    pub fn search_knn(
        &self,
        layer: usize,
        query: &K,
        num: usize,
    ) -> impl Iterator<Item = (usize, D)> + '_ {
        self.search_knn_from(layer, 0, query, num)
    }

    /// Finds the nearest neighbor to the query key starting from the `from` node using greedy search.
    ///
    /// Returns `(node, distance)`.
    pub fn search_from(&self, layer: usize, from: usize, query: &K) -> (usize, D) {
        // Get the weak node that corresponds to the given node on its particular layer.
        let (weak, distance) = self.search_from_weak(
            self.node_weak(layer, from),
            query.distance(&self.nodes[from].key).as_(),
            query,
        );
        // Get the index from the weak node.
        (weak.node, distance)
    }

    /// Finds the knn greedily from a starting node `from`.
    ///
    /// Returns (node, distance) pairs from closest to furthest.
    pub fn search_knn_from(
        &self,
        layer: usize,
        from: usize,
        query: &K,
        num: usize,
    ) -> impl Iterator<Item = (usize, D)> + '_ {
        self.search_knn_from_weak(
            self.node_weak(layer, from),
            query.distance(&self.nodes[from].key).as_(),
            query,
            num,
        )
        .into_iter()
        .map(|(weak, distance, _)| (weak.node, distance))
    }

    /// Finds the knn of `node` greedily.
    pub fn search_knn_of(
        &self,
        layer: usize,
        node: usize,
        num: usize,
    ) -> impl Iterator<Item = (usize, D)> + '_ {
        self.search_knn_from(layer, node, &self.nodes[node].key, num)
    }

    /// Insert a (key, value) pair.
    pub fn insert(&mut self, layer: usize, key: K, value: V) -> usize {
        // Add the node (it will be added this way regardless).
        let node = self.nodes.len();
        // Create the node.
        // The current freshest node's `next` is the stalest node, which will subsequently become
        // the freshest when freshened. If this is the only node, looking up the freshest node will fail.
        // Due to that, we set this node's next to itself if its the only node.
        let node_header_vec = HeaderVec::new(HrcHeader {
            key: key.clone(),
            node,
        });
        self.nodes.push(HrcNode {
            key: key.clone(),
            value,
            layers: vec![node_header_vec],
            next: if node == 0 {
                0
            } else {
                self.nodes[self.freshest].next
            },
        });
        // The previous freshest node should now be freshened right before this node, as this node is now fresher.
        // Even if this is the only node, this will still work because this node still comes after itself in the freshening order.
        self.nodes[self.freshest].next = node;
        // This is now the freshest node.
        self.freshest = node;

        if node == 0 {
            return 0;
        }

        // Find nearest neighbor via greedy search.
        let mut knn: Vec<_> = self
            .search_knn_from(layer, 0, &key, self.optimal_k())
            .map(|(nn, _)| (nn, self.nodes[nn].key.clone()))
            .collect();

        // Add edge to nearest neighbor.
        self.add_edge(layer, knn[0].0, node);

        // The initial neighbors only includes the edge we just added.
        let mut neighbors = vec![knn[0].1.clone()];

        // Optimize its edges using stalest nodes.
        let mut node_weak = self.node_weak(layer, node);
        self.optimize_neighborhood(layer, &mut node_weak, &mut knn, &mut neighbors);

        // Freshen the graph to clean up older nodes.
        self.freshen(layer);

        node
    }

    /// Produces the stalest nodes and marks them as now the freshest nodes when consumed.
    ///
    /// This iterator is infinite, and will iterate through every entry in a specific order before repeating.
    pub fn stales(&mut self) -> impl Iterator<Item = usize> + '_ {
        let mut node = self.freshest;
        core::iter::from_fn(move || {
            node = self.nodes[node].next;
            self.freshest = node;
            Some(node)
        })
    }

    /// Computes the distance between two nodes.
    pub fn distance(&self, a: usize, b: usize) -> D {
        self.nodes[a].key.distance(&self.nodes[b].key).as_()
    }

    /// Optimizes a number of stale nodes equal to `self.freshens`.
    ///
    /// You do not need to call this yourself, as it is called on insert.
    pub fn freshen(&mut self, layer: usize) {
        let freshens = self.freshens;
        for node in self.stales().take(freshens).collect::<Vec<_>>() {
            self.optimize_node(layer, node);
        }
    }

    /// Trims everything from a node that is no longer needed, and then only adds back what is needed.
    pub fn optimize_node(&mut self, layer: usize, node: usize) {
        let mut knn: Vec<_> = self
            .search_knn_of(layer, node, self.optimal_k())
            .skip(1)
            .map(|(nn, _)| (nn, self.nodes[nn].key.clone()))
            .collect();
        self.reinsert(layer, node);
        let mut node = self.node_weak(layer, node);
        let mut neighbors = node
            .as_slice()
            .iter()
            .map(|HrcEdge { key, .. }| key.clone())
            .collect();
        self.optimize_neighborhood(layer, &mut node, &mut knn, &mut neighbors);
    }

    /// Finds the nearest neighbor to the query key starting from the `from` node using greedy search.
    ///
    /// Returns `(node, distance)`.
    fn search_from_weak(&self, from: HVec<K>, from_distance: D, query: &K) -> (HVec<K>, D) {
        let mut best_weak = from;
        let mut best_distance = from_distance;

        while let Some((neighbor_weak, distance)) = best_weak
            .neighbors_distance(query)
            .min_by_key(|(_, distance)| *distance)
        {
            if distance < best_distance {
                best_weak = neighbor_weak.weak();
                best_distance = distance;
            } else {
                break;
            }
        }
        (best_weak, best_distance)
    }

    /// Finds the knn greedily from a starting node `from`.
    ///
    /// Returns (node, distance, searched) pairs. `searched` will always be true, so you can ignore it.
    fn search_knn_from_weak(
        &self,
        from: HVec<K>,
        from_distance: D,
        query: &K,
        num: usize,
    ) -> Vec<(HVec<K>, D, bool)> {
        if num == 0 {
            return vec![];
        }
        // Perform a greedy search first to save time.
        let (from, from_distance) = self.search_from_weak(from, from_distance, query);
        // Contains the index and the distance as a pair.
        let mut bests = vec![(from, from_distance, false)];

        loop {
            if let Some((previous_node, _, searched)) =
                bests.iter_mut().find(|&&mut (_, _, searched)| !searched)
            {
                // Set this as searched (we are searching it now).
                *searched = true;
                // Erase the reference to the search node (to avoid lifetime & borrowing issues).
                let previous_node = previous_node.weak();
                for HrcEdge { key, neighbor } in previous_node.as_slice() {
                    // Make sure that we don't have a copy of this node already or we will get duplicates.
                    if bests.iter().any(|(node, _, _)| neighbor.is(node.ptr())) {
                        continue;
                    }

                    // Compute the distance from the query.
                    let distance = query.distance(key).as_();
                    // If we dont have enough yet, add it.
                    if bests.len() < num {
                        bests.insert(
                            bests.partition_point(|&(_, best_distance, _)| {
                                best_distance <= distance
                            }),
                            (neighbor.weak(), distance, false),
                        );
                    } else if distance < bests.last().unwrap().1 {
                        // Otherwise only add it if its better than the worst item we have.
                        bests.pop();
                        bests.insert(
                            bests.partition_point(|&(_, best_distance, _)| {
                                best_distance <= distance
                            }),
                            (neighbor.weak(), distance, false),
                        );
                    }
                }
            } else {
                return bests;
            }
        }
    }

    /// Finds the knn of `node` greedily. The first item in the returned vec will always be this node.
    fn search_knn_of_weak(&self, node: HVec<K>, num: usize) -> Vec<(HVec<K>, D, bool)> {
        let key = &node.key;
        self.search_knn_from_weak(node.weak(), 0.as_(), key, num)
    }

    /// Optimizes a node by discovering local minima, and then breaking through all the local minima
    /// to the closest neighbor which is closer to the target.
    fn optimize_neighborhood(
        &mut self,
        layer: usize,
        node: &mut HVec<K>,
        knn: &mut Vec<(usize, K)>,
        neighbors: &mut Vec<K>,
    ) {
        let mut knn_index = 0;
        'knn_next: while let Some((target_node, target_key)) = knn.get(knn_index).cloned() {
            // Get this node's distance.
            let to_beat = node.key.distance(&target_key).as_();
            // Check if the node is colocated.
            if to_beat == 0.as_() {
                // In this case, add an edge (with dedup) between them to make sure there is a path.
                self.add_edge_dedup_weak(layer, node, &mut self.node_weak(layer, target_node));
                knn_index += 1;
                continue 'knn_next;
            }

            if neighbors
                .iter()
                .any(|key| target_key.distance(key).as_() < to_beat)
            {
                // If any are better, then no optimization needed.
                knn_index += 1;
                continue 'knn_next;
            }

            // Otherwise, lets find the closest neighbor to `node` that is closer to `target` than `node` is.
            loop {
                // Go through the nearest neighbors in order from best to worst.
                for (nn, nn_key) in knn.iter().cloned() {
                    // Compute the distance to the target from the nn.
                    let nn_distance = nn_key.distance(&target_key).as_();
                    // Add the node as a neighbor (closer or not).
                    // This will update the weak ref if necessary.
                    if self.add_edge_dedup_weak(layer, &mut self.node_weak(layer, nn), node) {
                        neighbors.push(nn_key);
                    }
                    // Check if this node is closer to the target than `from`.
                    if nn_distance < to_beat {
                        // The greedy path now exists, so exit.
                        knn_index += 1;
                        continue 'knn_next;
                    }
                }

                let before_len = knn.len();
                // If we searched the maximum number of nodes that we are willing to (or can, if its len - 1), then exit early without improvement.
                if before_len == self.len() - 1 {
                    panic!("fatal; we searched entire graph and did not find the node!");
                }
                // Double the knn space.
                *knn = self
                    .search_knn_of_weak(node.weak(), (knn.len() + 1).saturating_mul(2))
                    .into_iter()
                    .skip(1)
                    .map(|(nn, _, _)| (nn.node, nn.key.clone()))
                    .collect();
                // Reset the knn index, as we may have new found neighbors now.
                knn_index = 0;
                // If the knn didnt grow yet we didnt reach the maximum_knn yet, the graph is disconnected.
                if before_len == knn.len() {
                    panic!(
                        "fatal; graph is disconnected: {:?}, knn: {:?}",
                        self.simple_representation(),
                        knn.iter().map(|&(nn, _)| nn).collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    /// Removes a node from the graph and then reinserts it with the minimum number of edges on a particular layer.
    ///
    /// It is recommended to use [`Hrc::freshen`] instead of this method.
    fn reinsert(&mut self, layer: usize, node: usize) {
        // This wont work if we only have 1 node.
        if self.len() == 1 {
            return;
        }

        let mut node = self.node_weak(layer, node);
        let node_key = node.key.clone();

        // Disconnect the node.
        let mut neighbors = self.disconnect(&mut node);

        // Sort the neighbors to minimize the number we insert.
        neighbors.sort_unstable_by_key(|&(_, distance)| distance);

        // Make sure each neighbor can connect greedily to prevent disconnected graphs.
        for (neighbor, distance) in neighbors {
            let (mut nn, _) =
                self.search_from_weak(self.node_weak(layer, neighbor), distance, &node_key);
            if !nn.is(node.ptr()) {
                self.add_edge_dedup_weak(layer, &mut nn, &mut node);
            }
        }
    }

    /// Internal function for disconnecting a node from the graph.
    ///
    /// Returns nodes as usize because as nodes are re-added, it is possible that neighbors reallocate
    /// and break the weak pointers.
    ///
    /// Returns (node, distance) pairs.
    fn disconnect(&mut self, node: &mut HVec<K>) -> Vec<(usize, D)> {
        let node_key = node.key.clone();
        let mut neighbors = vec![];
        let ptr = node.ptr();
        self.edges -= node.len();
        for (mut neighbor, distance) in node.neighbors_distance(&node_key) {
            neighbor.retain(|HrcEdge { neighbor, .. }| !neighbor.is(ptr));
            neighbors.push((neighbor.node, distance));
        }
        node.retain(|_| false);
        neighbors
    }
}

impl<K, V, D> Default for Hrc<K, V, D> {
    fn default() -> Self {
        Self::new()
    }
}
