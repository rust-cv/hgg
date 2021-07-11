extern crate std;

use core::marker::PhantomData;

use crate::{HVec, HggEdge, HggHeader, HggNode, NodeStorage, StrategyLite, StrategyRegular};
use alloc::{fmt, format, vec, vec::Vec};
use header_vec::HeaderVec;
use serde::{
    de::{self, SeqAccess, Unexpected, Visitor},
    ser::SerializeSeq,
    Deserialize, Deserializer, Serialize, Serializer,
};

#[derive(Debug, Serialize)]
struct HggNodeSerialize<'a, K, V> {
    key: &'a K,
    value: &'a V,
    /// Contains the edges of each layer of the graph on which this exists.
    layers: Vec<Vec<usize>>,
    /// Forms a linked list through the nodes that creates the freshening order.
    next: usize,
}

impl<K, V> Serialize for NodeStorage<K, V, K, StrategyRegular>
where
    K: Serialize,
    V: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_seq(self.iter().map(|e| {
            HggNodeSerialize {
                key: &e.key,
                value: &e.value,
                layers: e
                    .layers
                    .iter()
                    .map(|edges| {
                        edges
                            .as_slice()
                            .iter()
                            .map(|edge| edge.neighbor.node)
                            .collect()
                    })
                    .collect(),
                next: e.next,
            }
        }))
    }
}

#[derive(Debug, Deserialize)]
struct HggNodeDeserialize<K, V> {
    key: K,
    value: V,
    /// Contains the edges of each layer of the graph on which this exists.
    layers: Vec<Vec<usize>>,
    /// Forms a linked list through the nodes that creates the freshening order.
    next: usize,
}

// impl<'de, K, V> Deserialize<'de> for NodeStorage<K, V, K, StrategyRegular>
// where
//     K: Deserialize<'de> + Clone,
//     V: Deserialize<'de>,
// {
//     fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
//     where
//         D: Deserializer<'de>,
//     {
//         std::eprintln!("up here");
//         // Deserialize the intermediate nodes.
//         let nodesd = Vec::<HggNodeDeserialize<K, V>>::deserialize(deserializer)?;

//         std::eprintln!("over here {}", nodesd.len());

//         // Create all the HggNode, but do not populate the edges yet.
//         // Save the layers.
//         let mut nodes = vec![];
//         let mut nodes_layers = vec![];
//         for (
//             node,
//             HggNodeDeserialize {
//                 key,
//                 value,
//                 layers,
//                 next,
//             },
//         ) in nodesd.into_iter().enumerate()
//         {
//             let empty_layers = (0..layers.len())
//                 .map(|_| {
//                     HeaderVec::with_capacity(
//                         layers.len(),
//                         HggHeader {
//                             key: key.clone(),
//                             node,
//                         },
//                     )
//                 })
//                 .collect();
//             nodes.push(HggNode {
//                 key,
//                 value,
//                 layers: empty_layers,
//                 next,
//             });
//             nodes_layers.push(layers);
//         }

//         let node_data: Vec<(K, Vec<HVec<K>>)> = nodes
//             .iter()
//             .map(|node| {
//                 (
//                     node.key.clone(),
//                     node.layers
//                         .iter()
//                         .map(|layer| HVec(unsafe { layer.weak() }))
//                         .collect(),
//                 )
//             })
//             .collect();

//         // Populate the edges on the nodes. If an index is out of bounds, we need to report an error.
//         for (node, layers) in nodes.iter_mut().zip(nodes_layers) {
//             for (layer, neighbors) in layers.into_iter().enumerate() {
//                 for neighbor in neighbors {
//                     // Get the key and weak refs for the target node.
//                     let (key, neighbor_layers) = node_data.get(neighbor).ok_or_else(|| {
//                         // If the index is pointing to an out-of-bounds neighbor, handle that.
//                         de::Error::invalid_value(
//                             Unexpected::Unsigned(neighbor as u64),
//                             &format!("one of {} valid nodes", node_data.len()).as_str(),
//                         )
//                     })?;
//                     // Get the weak ref for the target node.
//                     let neighbor = neighbor_layers
//                         .get(layer)
//                         .ok_or_else(|| {
//                             // If it didn't have the expected layer, there is an error.
//                             de::Error::invalid_length(
//                                 layer,
//                                 &format!("one of target node's {} layers", neighbor_layers.len())
//                                     .as_str(),
//                             )
//                         })?
//                         .weak();
//                     // Add the edge.
//                     node.layers[layer].push(HggEdge {
//                         key: key.clone(),
//                         neighbor,
//                     });
//                 }
//             }
//         }

//         Ok(NodeStorage(nodes, PhantomData))
//     }
// }

impl<'de, K, V> Deserialize<'de> for NodeStorage<K, V, K, StrategyRegular>
where
    K: Deserialize<'de> + Clone,
    V: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct VecVisitor<T> {
            marker: PhantomData<T>,
        }

        impl<'de, T> Visitor<'de> for VecVisitor<T>
        where
            T: Deserialize<'de>,
        {
            type Value = Vec<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a sequence")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut values = Vec::with_capacity(seq.size_hint().unwrap_or(0));

                while let Some(value) = seq.next_element()? {
                    values.push(value);
                }

                Ok(values)
            }
        }

        let visitor = VecVisitor {
            marker: PhantomData,
        };

        let nodesd: Vec<HggNodeDeserialize<K, V>> = deserializer.deserialize_seq(visitor)?;

        std::eprintln!("over here {}", nodesd.len());

        // Create all the HggNode, but do not populate the edges yet.
        // Save the layers.
        let mut nodes = vec![];
        let mut nodes_layers = vec![];
        for (
            node,
            HggNodeDeserialize {
                key,
                value,
                layers,
                next,
            },
        ) in nodesd.into_iter().enumerate()
        {
            let empty_layers = (0..layers.len())
                .map(|_| {
                    HeaderVec::with_capacity(
                        layers.len(),
                        HggHeader {
                            key: key.clone(),
                            node,
                        },
                    )
                })
                .collect();
            nodes.push(HggNode {
                key,
                value,
                layers: empty_layers,
                next,
            });
            nodes_layers.push(layers);
        }

        let node_data: Vec<(K, Vec<HVec<K>>)> = nodes
            .iter()
            .map(|node| {
                (
                    node.key.clone(),
                    node.layers
                        .iter()
                        .map(|layer| HVec(unsafe { layer.weak() }))
                        .collect(),
                )
            })
            .collect();

        // Populate the edges on the nodes. If an index is out of bounds, we need to report an error.
        for (node, layers) in nodes.iter_mut().zip(nodes_layers) {
            for (layer, neighbors) in layers.into_iter().enumerate() {
                for neighbor in neighbors {
                    // Get the key and weak refs for the target node.
                    let (key, neighbor_layers) = node_data.get(neighbor).ok_or_else(|| {
                        // If the index is pointing to an out-of-bounds neighbor, handle that.
                        de::Error::invalid_value(
                            Unexpected::Unsigned(neighbor as u64),
                            &format!("one of {} valid nodes", node_data.len()).as_str(),
                        )
                    })?;
                    // Get the weak ref for the target node.
                    let neighbor = neighbor_layers
                        .get(layer)
                        .ok_or_else(|| {
                            // If it didn't have the expected layer, there is an error.
                            de::Error::invalid_length(
                                layer,
                                &format!("one of target node's {} layers", neighbor_layers.len())
                                    .as_str(),
                            )
                        })?
                        .weak();
                    // Add the edge.
                    node.layers[layer].push(HggEdge {
                        key: key.clone(),
                        neighbor,
                    });
                }
            }
        }

        Ok(NodeStorage(nodes, PhantomData))
    }
}
