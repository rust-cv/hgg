use alloc::vec::Vec;

#[derive(Clone, Debug, PartialEq)]
pub struct Stats {
    pub len: usize,
    pub layer_cluster_neigbors_histogram: Vec<Vec<(usize, usize)>>,
}

impl Stats {
    pub fn simple(&self) -> SimpleStats {
        SimpleStats {
            len: self.len,
            layer_average_neighbors: self.layer_average_neighbors(),
        }
    }

    pub fn layer_average_neighbors(&self) -> Vec<f64> {
        self.layer_cluster_neigbors_histogram
            .iter()
            .map(|clusters| {
                clusters.iter().map(|&(_, n)| n as f64).sum::<f64>() / clusters.len() as f64
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct SimpleStats {
    pub len: usize,
    pub layer_average_neighbors: Vec<f64>,
}
