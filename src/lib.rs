//! High-performance, parallel k-means clustering for Rust applications.
//!
//! The crate ships a production-ready implementation that combines
//! deterministic initialisation, parallel centroid updates, structured
//! error handling, and a CLI entry-point for end-to-end experimentation.

use csv::ReaderBuilder;
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, Axis};
use ndarray_rand::rand_distr::{Distribution, Normal, Uniform};
use ndarray_rand::RandomExt;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Field, Row};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

/// Dense data representation used across the crate (rows = samples, columns = features).
pub type DataMatrix = Array2<f64>;

/// Error type used by operations in this crate.
#[derive(Debug, Error)]
pub enum KMeansError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("invalid data: {0}")]
    InvalidData(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Csv(#[from] csv::Error),
    #[error(transparent)]
    ParseFloat(#[from] std::num::ParseFloatError),
    #[error(transparent)]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
}

/// Convenient alias for results produced by this crate.
pub type Result<T> = std::result::Result<T, KMeansError>;

/// Strategy used to seed initial centroids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InitStrategy {
    /// Choose centroids uniformly at random from the dataset.
    Random,
    /// K-Means++ initialisation as described by Arthur/Vassilvitskii.
    #[serde(alias = "kmeans++", alias = "k-means++")]
    KMeansPlusPlus,
}

impl Default for InitStrategy {
    fn default() -> Self {
        Self::KMeansPlusPlus
    }
}

impl fmt::Display for InitStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InitStrategy::Random => write!(f, "random"),
            InitStrategy::KMeansPlusPlus => write!(f, "kmeans++"),
        }
    }
}

impl std::str::FromStr for InitStrategy {
    type Err = KMeansError;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "random" => Ok(Self::Random),
            "kmeans++" | "k-means++" => Ok(Self::KMeansPlusPlus),
            other => Err(KMeansError::InvalidConfig(format!(
                "unsupported init strategy '{other}'"
            ))),
        }
    }
}

/// Configurable knobs for a k-means training run.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KMeansConfig {
    /// Number of centroids to optimise.
    pub k: usize,
    /// Maximum iterations before giving up on convergence.
    pub max_iter: usize,
    /// Stop once the largest centroid shift falls below this tolerance.
    pub tol: f64,
    /// Centroid initialisation strategy.
    pub init: InitStrategy,
    /// Number of restarts (best run selected by inertia).
    pub n_init: usize,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 8,
            max_iter: 300,
            tol: 1e-4,
            init: InitStrategy::default(),
            n_init: 1,
        }
    }
}

impl KMeansConfig {
    /// Validate configuration parameters for a specific dataset.
    pub fn validate(&self, points: &DataMatrix) -> Result<()> {
        if self.k == 0 {
            return Err(KMeansError::InvalidConfig(
                "k must be greater than zero".into(),
            ));
        }
        if points.nrows() < self.k {
            return Err(KMeansError::InvalidConfig(format!(
                "dataset has {} samples but k = {}; add more data or decrease k",
                points.nrows(),
                self.k
            )));
        }
        if self.max_iter == 0 {
            return Err(KMeansError::InvalidConfig(
                "max_iter must be greater than zero".into(),
            ));
        }
        if self.tol < 0.0 {
            return Err(KMeansError::InvalidConfig(
                "tol must be non-negative".into(),
            ));
        }
        if self.n_init == 0 {
            return Err(KMeansError::InvalidConfig(
                "n_init must be at least 1".into(),
            ));
        }
        Ok(())
    }
}

/// Trained k-means model containing final centroids and config metadata.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KMeans {
    /// Configuration used during training.
    pub config: KMeansConfig,
    /// Centroids as rows (`k` × `dim`).
    pub centroids: DataMatrix,
}

impl KMeans {
    /// Create a new model from centroids and configuration.
    pub fn new(config: KMeansConfig, centroids: DataMatrix) -> Self {
        Self { config, centroids }
    }

    /// Predict the cluster index for a single point.
    pub fn predict_point(&self, point: &ArrayView1<f64>) -> usize {
        debug_assert_eq!(point.len(), self.centroids.ncols());
        let mut best = 0usize;
        let mut best_distance = squared_distance(point, &self.centroids.row(0));
        for cid in 1..self.centroids.nrows() {
            let distance = squared_distance(point, &self.centroids.row(cid));
            if distance < best_distance {
                best_distance = distance;
                best = cid;
            }
        }
        best
    }

    /// Predict cluster assignments for the entire dataset.
    pub fn predict(&self, points: &DataMatrix) -> Vec<usize> {
        (0..points.nrows())
            .into_par_iter()
            .map(|i| self.predict_point(&points.row(i)))
            .collect()
    }

    /// Inertia metric (sum of squared distances to centroids).
    pub fn inertia(&self, points: &DataMatrix) -> f64 {
        let assignments = self.predict(points);
        self.inertia_from_assignments(points, &assignments)
    }

    /// Compute inertia given pre-computed assignments.
    pub fn inertia_from_assignments(&self, points: &DataMatrix, assignments: &[usize]) -> f64 {
        assignments
            .par_iter()
            .enumerate()
            .map(|(idx, &cid)| {
                let point = points.row(idx);
                squared_distance(&point, &self.centroids.row(cid))
            })
            .sum()
    }

    /// Persist the model as JSON.
    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Train the model in-place, returning diagnostics about the run.
    pub fn fit<R: Rng + RngCore>(&mut self, points: &DataMatrix, rng: &mut R) -> FitOutcome {
        let k = self.centroids.nrows();
        let dim = self.centroids.ncols();
        let n = points.nrows();
        debug_assert_eq!(
            points.ncols(),
            dim,
            "dimension mismatch between data and centroids"
        );

        let zero = (vec![vec![0.0f64; dim]; k], vec![0usize; k]);
        let mut iterations = 0usize;

        let (final_assignments, final_counts, converged) = loop {
            iterations += 1;

            let assignments: Vec<usize> = (0..n)
                .into_par_iter()
                .map(|i| self.predict_point(&points.row(i)))
                .collect();

            let (sums, counts) = (0..n)
                .into_par_iter()
                .fold(
                    || zero.clone(),
                    |mut acc, row_idx| {
                        let cid = assignments[row_idx];
                        let row = points.row(row_idx);
                        for d in 0..dim {
                            acc.0[cid][d] += row[d];
                        }
                        acc.1[cid] += 1;
                        acc
                    },
                )
                .reduce(
                    || zero.clone(),
                    |mut left, right| {
                        for cid in 0..k {
                            left.1[cid] += right.1[cid];
                            for d in 0..dim {
                                left.0[cid][d] += right.0[cid][d];
                            }
                        }
                        left
                    },
                );

            let mut max_shift = 0.0;
            for cid in 0..k {
                let count = counts[cid];
                if count == 0 {
                    // Avoid dead clusters by respawning centroid at a random sample.
                    let idx = rng.gen_range(0..n);
                    self.centroids.row_mut(cid).assign(&points.row(idx));
                    continue;
                }
                for d in 0..dim {
                    let new_val = sums[cid][d] / count as f64;
                    let old_val = self.centroids[(cid, d)];
                    let shift = (old_val - new_val).abs();
                    if shift > max_shift {
                        max_shift = shift;
                    }
                    self.centroids[(cid, d)] = new_val;
                }
            }

            if max_shift <= self.config.tol || iterations >= self.config.max_iter {
                let converged = max_shift <= self.config.tol;
                break (assignments, counts, converged);
            }
        };

        let inertia = self.inertia_from_assignments(points, &final_assignments);
        FitOutcome {
            assignments: final_assignments,
            inertia,
            iterations,
            converged,
            cluster_sizes: final_counts,
        }
    }
}

/// Result of a single k-means optimisation pass.
#[derive(Debug, Clone)]
pub struct FitOutcome {
    /// Final assignments for each row in the dataset.
    pub assignments: Vec<usize>,
    /// Sum of squared distances to each centroid.
    pub inertia: f64,
    /// Number of completed iterations.
    pub iterations: usize,
    /// Whether the run satisfied the convergence tolerance.
    pub converged: bool,
    /// Number of samples assigned to each centroid.
    pub cluster_sizes: Vec<usize>,
}

/// Run k-means with multiple random restarts, tracking the best candidate.
pub fn kmeans_train_with_restarts(
    points: &DataMatrix,
    config: &KMeansConfig,
    rng: &mut ChaCha8Rng,
) -> Result<KMeansRun> {
    config.validate(points)?;

    let mut best_model: Option<KMeans> = None;
    let mut best_outcome: Option<FitOutcome> = None;

    for restart_idx in 0..config.n_init {
        let mut restart_rng = ChaCha8Rng::seed_from_u64(rng.next_u64());
        let centroids = match config.init {
            InitStrategy::KMeansPlusPlus => kmeans_pp_init(points, config.k, &mut restart_rng)?,
            InitStrategy::Random => random_init(points, config.k, &mut restart_rng),
        };

        let mut model = KMeans::new(config.clone(), centroids);
        let outcome = model.fit(points, &mut restart_rng);

        let inertia = outcome.inertia;
        if best_outcome
            .as_ref()
            .map(|current| inertia < current.inertia)
            .unwrap_or(true)
        {
            tracing::debug!(
                restart = restart_idx,
                inertia,
                converged = outcome.converged,
                iterations = outcome.iterations,
                "accepting new best kmeans solution"
            );
            best_model = Some(model);
            best_outcome = Some(outcome);
        }
    }

    match (best_model, best_outcome) {
        (Some(model), Some(outcome)) => Ok(KMeansRun { model, outcome }),
        _ => Err(KMeansError::InvalidData(
            "failed to train k-means on the provided dataset".into(),
        )),
    }
}

/// Combined model + diagnostics returned from [`kmeans_train_with_restarts`].
#[derive(Debug, Clone)]
pub struct KMeansRun {
    /// Best-scoring model.
    pub model: KMeans,
    /// Diagnostics and assignments for the best run.
    pub outcome: FitOutcome,
}

/// Generate random data matrix (n rows, dim columns) using a reproducible RNG.
pub fn generate_points(n: usize, dim: usize, rng: &mut ChaCha8Rng) -> DataMatrix {
    Array2::random_using((n, dim), Uniform::new(0.0, 1.0), rng)
}

/// Generate Gaussian-like clustered data useful for benchmarking.
pub fn generate_clustered_points(
    n_per_cluster: usize,
    centroids: &DataMatrix,
    spread: f64,
    rng: &mut ChaCha8Rng,
) -> DataMatrix {
    let k = centroids.nrows();
    let dim = centroids.ncols();
    let total = n_per_cluster * k;
    let mut points = Array2::zeros((total, dim));
    let normal = Normal::new(0.0, spread).unwrap();

    for (cluster_idx, centroid) in centroids.outer_iter().enumerate() {
        for sample_idx in 0..n_per_cluster {
            let row_idx = cluster_idx * n_per_cluster + sample_idx;
            let row = points.row_mut(row_idx);
            synthesise_sample(&centroid, row, &normal, rng);
        }
    }

    points
}

fn synthesise_sample<R: Rng + ?Sized>(
    centroid: &ArrayView1<f64>,
    mut row: ArrayViewMut1<'_, f64>,
    normal: &Normal<f64>,
    rng: &mut R,
) {
    for (value, &centre) in row.iter_mut().zip(centroid.iter()) {
        *value = centre + normal.sample(rng);
    }
}

fn kmeans_pp_init(points: &DataMatrix, k: usize, rng: &mut ChaCha8Rng) -> Result<DataMatrix> {
    let (n, dim) = (points.nrows(), points.ncols());
    if k > n {
        return Err(KMeansError::InvalidConfig(format!(
            "initialisation requires k <= n (k={k}, n={n})"
        )));
    }
    let mut centroids = Array2::zeros((k, dim));

    let first = rng.gen_range(0..n);
    centroids.row_mut(0).assign(&points.row(first));

    let mut distances: Vec<f64> = (0..n)
        .map(|i| squared_distance(&points.row(i), &centroids.row(0)))
        .collect();

    for cid in 1..k {
        let sum: f64 = distances.iter().sum();
        let mut pick = rng.gen::<f64>() * sum;
        let mut idx = 0usize;
        while pick > distances[idx] {
            pick -= distances[idx];
            idx += 1;
        }
        centroids.row_mut(cid).assign(&points.row(idx));

        for i in 0..n {
            let d = squared_distance(&points.row(i), &centroids.row(cid));
            if d < distances[i] {
                distances[i] = d;
            }
        }
    }
    Ok(centroids)
}

fn random_init(points: &DataMatrix, k: usize, rng: &mut ChaCha8Rng) -> DataMatrix {
    let n = points.nrows();
    let mut idxs: Vec<usize> = (0..n).collect();
    idxs.shuffle(rng);
    let mut centroids = Array2::zeros((k, points.ncols()));
    for (ci, &i) in idxs.iter().take(k).enumerate() {
        centroids.row_mut(ci).assign(&points.row(i));
    }
    centroids
}

fn squared_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

/// Data standardisation parameters (z-score normalisation).
#[derive(Debug, Clone)]
pub struct Standardization {
    /// Feature-wise mean.
    pub mean: Array1<f64>,
    /// Feature-wise standard deviation.
    pub std: Array1<f64>,
}

/// Output of [`standardize`] helper.
#[derive(Debug, Clone)]
pub struct StandardizedData {
    /// Transformed dataset.
    pub data: DataMatrix,
    /// Parameters required to transform new samples consistently.
    pub params: Standardization,
}

/// Apply z-score standardisation to each column of the dataset.
pub fn standardize(points: &DataMatrix) -> StandardizedData {
    if points.nrows() == 0 {
        let dim = points.ncols();
        return StandardizedData {
            data: points.clone(),
            params: Standardization {
                mean: Array1::zeros(dim),
                std: Array1::from_elem(dim, 1.0),
            },
        };
    }

    let n = points.nrows() as f64;
    let mean = points.sum_axis(Axis(0)) / n;
    let mut variance = Array1::<f64>::zeros(points.ncols());

    for row in points.rows() {
        let diff = &row - &mean;
        variance += &diff.mapv(|x| x * x);
    }
    variance /= n;

    let std = variance.mapv(|v| (v.max(1e-12)).sqrt());

    let mut data = points.clone();
    for mut row in data.rows_mut() {
        row -= &mean;
        row /= &std;
    }

    StandardizedData {
        data,
        params: Standardization { mean, std },
    }
}

/// DataLoader abstraction to load CSV/Parquet into [`DataMatrix`].
pub struct DataLoader;

impl DataLoader {
    /// Load a CSV file into memory assuming numeric columns.
    pub fn load_csv<P: AsRef<Path>>(path: P) -> Result<DataMatrix> {
        let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;
        let mut records: Vec<Vec<f64>> = Vec::new();
        let mut width = None;
        for record in rdr.records() {
            let record = record?;
            let mut row = Vec::with_capacity(record.len());
            for field in record.iter() {
                row.push(field.parse()?);
            }
            if let Some(expected) = width {
                if expected != row.len() {
                    return Err(KMeansError::InvalidData(format!(
                        "found inconsistent row width: expected {expected}, got {}",
                        row.len()
                    )));
                }
            } else {
                width = Some(row.len());
            }
            records.push(row);
        }
        let Some(dim) = width else {
            return Ok(Array2::zeros((0, 0)));
        };

        let n = records.len();
        let mut arr = Array2::zeros((n, dim));
        for (i, row) in records.into_iter().enumerate() {
            for (j, value) in row.into_iter().enumerate() {
                arr[(i, j)] = value;
            }
        }
        Ok(arr)
    }

    /// Load a Parquet file containing only numeric (int/float) columns.
    pub fn load_parquet<P: AsRef<Path>>(path: P) -> Result<DataMatrix> {
        let file = File::open(path)?;
        let reader = SerializedFileReader::new(file)?;
        let mut row_iter = reader.get_row_iter(None)?;
        let mut rows: Vec<Row> = Vec::new();
        while let Some(row) = row_iter.next() {
            rows.push(row);
        }
        if rows.is_empty() {
            return Ok(Array2::zeros((0, 0)));
        }
        let width = rows[0].len();
        let mut data = Array2::zeros((rows.len(), width));

        for (i, row) in rows.iter().enumerate() {
            if row.len() != width {
                return Err(KMeansError::InvalidData(format!(
                    "row {i} width mismatch: expected {width}, found {}",
                    row.len()
                )));
            }
            for (j, (_, field)) in row.get_column_iter().enumerate() {
                let value = match field {
                    Field::Double(v) => *v,
                    Field::Float(v) => *v as f64,
                    Field::Int(v) => *v as f64,
                    Field::Long(v) => *v as f64,
                    Field::Short(v) => *v as f64,
                    Field::Byte(v) => *v as f64,
                    Field::UInt(v) => *v as f64,
                    Field::ULong(v) => *v as f64,
                    Field::UShort(v) => *v as f64,
                    Field::UByte(v) => *v as f64,
                    Field::Null => {
                        return Err(KMeansError::InvalidData(format!(
                            "column {j} contained a NULL value which cannot be converted to f64"
                        )))
                    }
                    other => {
                        return Err(KMeansError::InvalidData(format!(
                            "unsupported parquet field at column {j}: {other:?}"
                        )))
                    }
                };
                data[(i, j)] = value;
            }
        }

        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;

    #[test]
    fn kmeans_basic_training() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let points = generate_points(256, 4, &mut rng);
        let config = KMeansConfig {
            k: 5,
            max_iter: 50,
            tol: 1e-6,
            init: InitStrategy::KMeansPlusPlus,
            n_init: 2,
        };
        let run =
            kmeans_train_with_restarts(&points, &config, &mut rng).expect("training succeeds");
        assert_eq!(run.outcome.assignments.len(), 256);
        assert_eq!(run.model.centroids.nrows(), 5);
        assert_eq!(run.model.centroids.ncols(), 4);
    }

    #[test]
    fn empty_dataset_is_rejected() {
        let points = Array2::<f64>::zeros((0, 0));
        let config = KMeansConfig::default();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let err = kmeans_train_with_restarts(&points, &config, &mut rng).unwrap_err();
        assert!(matches!(err, KMeansError::InvalidConfig(_)));
    }

    #[test]
    fn standardization_round_trip() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let data = generate_points(32, 3, &mut rng);
        let StandardizedData {
            data: transformed,
            params,
        } = standardize(&data);

        for col in 0..transformed.ncols() {
            let col_view = transformed.slice(s![.., col]);
            let count = col_view.len() as f64;
            let sum: f64 = col_view.iter().copied().sum();
            let mean = sum / count;
            let variance: f64 = col_view
                .iter()
                .map(|value| {
                    let diff = *value - mean;
                    diff * diff
                })
                .sum::<f64>()
                / count;

            // Column should be zero-centred with unit std within tolerance.
            assert!(mean.abs() < 1e-9);
            assert!((variance - 1.0).abs() < 1e-6);
            assert!(params.std[col] >= 0.0);
        }
    }
}
