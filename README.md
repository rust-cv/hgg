# hgg

[![Discord][dci]][dcl] [![Crates.io][ci]][cl] ![MIT/Apache][li] [![docs.rs][di]][dl] ![LoC][lo] ![Tests][btl] ![Lints][bll] ![no_std][bnl]

[ci]: https://img.shields.io/crates/v/hgg.svg
[cl]: https://crates.io/crates/hgg/

[li]: https://img.shields.io/crates/l/specs.svg?maxAge=2592000

[di]: https://docs.rs/hgg/badge.svg
[dl]: https://docs.rs/hgg/

[lo]: https://tokei.rs/b1/github/rust-cv/hgg?category=code

[dci]: https://img.shields.io/discord/550706294311485440.svg?logo=discord&colorB=7289DA
[dcl]: https://discord.gg/d32jaam

[btl]: https://github.com/rust-cv/hgg/workflows/tests/badge.svg
[bll]: https://github.com/rust-cv/hgg/workflows/lints/badge.svg
[bnl]: https://github.com/rust-cv/hgg/workflows/no-std/badge.svg

Hierarchical Greedy Graph

## How does it work

Firstly, this data structure was designed by Geordon Worley and does not have a corresponding paper. Anyone that would like to write a paper on this data structure can feel free to reach out to Geordon Worley. If looking to use the ideas mentioned here in a paper, please cite this repository and the Rust CV project.

This data structure is based around the concept of a hierarchiy of nearest neighbor search graphs, first pioneered in the [the HNSW paper](https://arxiv.org/pdf/1603.09320.pdf). However, most of the algorithms used in that paper for inserting things and searching the graph were abandoned in favor of more data-dependent alternatives. This data structure was designed to maximize data-dependence.

Points in a metric space can have a lower dimensionality than the space itself. For instance, the points making up the surface of a table exist in a 3d space, but they are 2d themselves. Similarly, if you draw points on the coastline of a map, it is said to have a fractal dimension of between 1 and 2, as it is not quite a line, nor a plane. When you have 2d or 3d data, there are known exact nearest neighbor search structures which utilize the dimensionality of the space. Similarly, there are dimensionality reduction techniques which can allow you to use a lower dimensionality data structure to store your data. Rather than trying to derive this dimensionality for the entire dataset, HGG allows the data to naturally form structures at different scales to promote high-recall greedy searches based on its own local dimensionality, meaning that different areas and different scales of the metric space can have their dimensionality accounted for. It does this using the function `optimize_layer_neighborhood`. This function is passed a neighborhood, and it iterates through each node in the neighborhood, starting from the nearest neighbors. It checks to make sure that the current node is not a local minima on the greedy search path to this neighbor. If it is, the algorithm adds an edge from this node to its nearest neighbors until it satisfies this condition for all neighbors. Given that the number of nearest neighbors passed in is sufficiently large, this will ensure that, in all known directions towards nodes in the neighborhood, we have a greedy search path in any direction, and connects the graph to the minimum number of nearest neighbors necessary to satisfy this greedy criteria.

Setting the `insert_knn` parameter allows the scope of these connections to be adjusted. If `insert_knn` is sufficiently large, and the dataset contains no concave outer regions where points could potentially exist, then the search should be exact given a large enough `insert_knn`. Concave edges in the search set reduce the effectiveness of this algorithm, so it is important that the input points cover the surface boundaries of potential search candidates, otherwise there is no way to know how large the volume is the search graph represents in the underlying metric space, and the algorithm can always find a path around the concave region. For very high-dimensional data, exact search is highly undesirable. If you are working in hamming space, or dimensionalities that are discrete with a small number of fixed positions, one reason is specified in the paper "Thick Boundaries in Binary Space and Their Influence on Nearest-Neighbor Search" by T. Trzciński, V. Lepetit, and P. Fua, where the equidistant boundaries between points are so large that the number of neighbors would become explosive, even moreso than high dimensionality in other kinds of spaces. Secondly, hamming or otherwise, the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) is a commonly understood phenomena that makes it prohibitive to perform exact search in higher dimensionality. Simply, the number of edges needed to perform exact search is far too large to make it efficient for specific datasets with think Voronoi boundaries, which occur due to the above phenomena. For that reason, you can use the `insert_knn` parameter to control how connected your graph is. The default parameter is set to `64`, which limits the number of edges that can be added per insertion (or freshen operation) to this number, but you may want to adjust this based on your dataset. If it is set too high and your dataset is large, it can degrade performance if slower memory (such as swap space) is used since more memory is consumed. `insert_knn` is also inversely correlated with insertion speed, so reduce it if you need insertion to occur faster, and it can be made up for by taking longer on lookup using a higher number of nearest neighbors in the greedy search to achieve the same recall at a lower speed.

To search the graph, only greedy search is used. A pool of the best nodes seen so far is retained, and if a node is better than the worst node seen, it is inserted into the pool. It then iterates through the best pool to find nodes which it has not searched the neighbors of yet. It always starts with the best node possible, and eliminates nodes from being searched which fall out of the pool. In the 1 nn case, the pool is not needed, and it simply takes the best neighbor of the current node repeatedly until it reaches a minima. The keys of all the neighbors are also stored inside of each node at one memory address that contains everything about that node, which makes it possible to quickly traverse between greedy searches using only one random lookup in memory per node and subsequent greedy search of its neighbors.

To make the graph into a "hierarchy", a simple method is used. When a node is inserted, a 1nn greedy search is performed from the top layer to the bottom layer. It retains the nn found on each layer. It then performs a knn search on each layer it is inserted to to further refine and find the best nn, and then uses that knn search to call `optimize_layer_neighborhood`. When a node is inserted to a layer, it then checks to see if at least one neighbor is on the next layer up (or above). If no neighbor is on the next layer up, it then iterates to the next layer and then performs the same knn search and optimization. Unlike HNSW, which uses an a priori method to choose the layer without considering data-dependence, this algorithm creates an implicit multi-dimensional skiplist with dimensionality that is roughly proportional to the underlying data in a local region and also spaces out the skiplist more accurately. For instance, if the data forms a perfect line in space, at most every other (1/2) and at least every third (1/3) of nodes would exist on the next layer up for all layers. This is because when a node is inserted, it will either be inserted between two nodes which are on the same layer, or it will be inserted between a node which is on the same layer and one which is above it. Assume that `_` represents a node on the lower/same layer and `*` represents a node on the layer above or higher. In the former case, it will be raised to the next layer, creating a `_-*-_` pattern, and in the latter case, it will not raise to the next layer, and form a `_-_-*` pattern. If it forms a `_-_-*` pattern, the pattern could expand to look like `_-_-_-*`, so it is possible that not every node will have a physical neighbor on the next layer, but this is not a significant issue, as the node further to the left is still connected to the node to the right, and will be a graph neighbor until the graph is freshened.

The last important thing is that as nodes are added to the graph, the graph is freshened. A linked list is formed in the graph which stores the order that this freshening occurs in. When a node is added, it is assumed to be the freshest node, and the next item in the list from this node is the stalest node. This forms a bizzare pattern in the nodes, since on each insert, we also freshen nodes, moving the freshest node in the linked list forward by some amount. The reason this freshening is done is to make sure that we don't have any unecessary edges in the graph. As we add nodes to the graph, we will add nodes inbetween each other, and it is important that all the edges are trimmed from these nodes so that the graph doesn't become too connected, which is wasteful. All we want is to make sure each node doesn't create a local minima on any greedy search through the graph, so extra edges simply slow down that process. We also want to make sure that if we freshen a node that it is added to the correct layer. Since we remove edges when freshening, it is possible that this node now no longer has a neighbor on the next layer up. If this is the case, we want to make sure that the node is moved up to the next layer so that all nodes have a graph neighbor on the next level.

As opposed to HNSW, no PRNG is used anywhere in HGG, and it is fully deterministic. Hashing using [`ahash`](https://github.com/tkaitchuck/aHash/) is used on pointers during greedy knn search to check for set inclusion using a `HashSet` from the [`hashbrown`](https://github.com/rust-lang/hashbrown) crate, but it is not keyed as the attacker likely has no control over the pointers of allocated data when attempting to perform a DOS attack.

## Recall Curve

To generate the `akaze` file used in the `recall_akaze` example, you first need a large number of unique images. It is recommended that these images come from a vSLAM dataset (TUM, KITTI, etc) if you want to test the performance in that particular setting.

For example, to get a chunk of the KITTI dataset, you can go to http://www.cvlibs.net/datasets/kitti/raw_data.php and download `2011_09_29_drive_0071` using the `[link unsynced+unrectified data]`. [Here](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_drive_0071/2011_09_29_drive_0071_extract.zip) is a link, though it may break. Extract the zip into a directory.

You will need to first install `docopt`, `progress`, and `opencv-contrib-python` (recommended) or `opencv-python` (use if you are worried about licensing issues). Next, you need to extract the features. `generate.py` is provided in this repository to do this. Here is an example:

```bash
echo ~/2011_09_29/2011_09_29_drive_0071_extract/image_*/data/* | xargs python generate.py akaze > akaze
```


It will provide a progress bar for each batch passed to it, but due to argument limit limitations it cant give you progress for the whole dataset. This will generate a file called `akaze` which contains many `61` byte AKAZE binary descriptors. You can do this with any dataset of images, but you may need to adjust the constant `HIGHEST_POWER_SEARCH_SPACE` in the `recall_akaze` example to a lower value if you do not have enough features in your file, which it will let you know when you run it.

To run the benchmark, it is recommended to first source `runrecall.sh`:

```bash
source runrecall.sh
```

This only needs to be done once per bash session, and will add the runrecall command to your bash session. Now run the following:

```bash
runrecall recall
```

This will begin running the program. You will see the following start to appear into `recall.csv`:

```csv
recall,search_size,knn,num_queries,seconds_per_query,queries_per_second,strategy
1.0,1,1,32768,1.58e-7,6329113.924050633,zero
1.0,1,1,32768,2.01e-7,4975124.378109452,regular
1.0,1,1,32768,7.8e-8,12820512.820512822,wide
1.0,1,2,32768,1.61e-7,6211180.124223603,zero
```

This file is populated as the benchmark runs. You can terminate the benchmark at any time with Ctrl+C, and the results computed so far will remain in this CSV file. It can be imported into a spreadsheet processor to generate graphs.

## Credits

I (Geordon Worley) designed and created this data structure through emperical testing, random ideas, and a focus on data-dependence. However, I did not create this data structure out of thin air. It builds upon the work of [the HNSW paper](https://arxiv.org/pdf/1603.09320.pdf). Although the resultant data-structure is very different from HNSW, the concept of a hierarchy of nearest neighbor graphs is retained. This is why the name of the crate is Hierarchical Greedy Graphs, as homage to Hierarchical Navigable Small Worlds.
