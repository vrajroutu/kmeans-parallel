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
use rand::seq::index::sample;
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

/// Training execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum TrainingMode {
    /// Classic full-batch Lloyd iterations.
    #[default]
    FullBatch,
    /// Adaptive multiphase training with streaming seeding and minibatches.
    Adaptive,
}

/// Settings that control the adaptive multiphase pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveSettings {
    /// Factor applied to `k` to determine the reservoir sample size.
    pub reservoir_factor: f64,
    /// Fraction of the dataset used for the initial minibatch.
    pub initial_batch_fraction: f64,
    /// Maximum fraction of the dataset a minibatch may touch.
    pub max_batch_fraction: f64,
    /// Upper bound multiplier when expanding batch sizes.
    pub max_batch_multiplier: f64,
    /// Maximum number of adaptive minibatch iterations before polishing.
    pub max_adaptive_iters: usize,
    /// Number of consecutive low-shift iterations required before stopping stage 2.
    pub patience: usize,
    /// Target shift tolerance used to adapt batch sizes.
    pub convergence_tol: f64,
}

impl Default for AdaptiveSettings {
    fn default() -> Self {
        Self {
            reservoir_factor: 3.0,
            initial_batch_fraction: 0.1,
            max_batch_fraction: 0.6,
            max_batch_multiplier: 4.0,
            max_adaptive_iters: 25,
            patience: 3,
            convergence_tol: 1e-3,
        }
    }
}

impl AdaptiveSettings {
    fn validate(&self, points: &DataMatrix) -> Result<()> {
        if self.reservoir_factor < 1.0 {
            return Err(KMeansError::InvalidConfig(
                "adaptive.reservoir_factor must be >= 1.0".into(),
            ));
        }
        if !(0.0 < self.initial_batch_fraction && self.initial_batch_fraction <= 1.0) {
            return Err(KMeansError::InvalidConfig(
                "adaptive.initial_batch_fraction must be in (0, 1]".into(),
            ));
        }
        if !(0.0 < self.max_batch_fraction && self.max_batch_fraction <= 1.0) {
            return Err(KMeansError::InvalidConfig(
                "adaptive.max_batch_fraction must be in (0, 1]".into(),
            ));
        }
        if self.max_batch_fraction < self.initial_batch_fraction {
            return Err(KMeansError::InvalidConfig(
                "adaptive.max_batch_fraction must be >= adaptive.initial_batch_fraction".into(),
            ));
        }
        if self.max_batch_multiplier < 1.0 {
            return Err(KMeansError::InvalidConfig(
                "adaptive.max_batch_multiplier must be >= 1.0".into(),
            ));
        }
        if self.max_adaptive_iters == 0 {
            return Err(KMeansError::InvalidConfig(
                "adaptive.max_adaptive_iters must be > 0".into(),
            ));
        }
        if self.patience == 0 {
            return Err(KMeansError::InvalidConfig(
                "adaptive.patience must be > 0".into(),
            ));
        }
        if self.convergence_tol <= 0.0 {
            return Err(KMeansError::InvalidConfig(
                "adaptive.convergence_tol must be > 0".into(),
            ));
        }
        if points.nrows() < points.ncols() && self.initial_batch_fraction < 1.0 {
            // Warn via validation to avoid zero-sized batches on tiny datasets.
            tracing::debug!(
                "adaptive initial fraction may be too small for very tall datasets; continuing"
            );
        }
        Ok(())
    }
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
    /// Execution strategy controlling training behaviour.
    pub mode: TrainingMode,
    /// Optional tuning knobs for adaptive mode.
    pub adaptive: Option<AdaptiveSettings>,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 8,
            max_iter: 300,
            tol: 1e-4,
            init: InitStrategy::default(),
            n_init: 1,
            mode: TrainingMode::default(),
            adaptive: None,
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
        if let TrainingMode::Adaptive = self.mode {
            let settings = self.adaptive_settings();
            settings.validate(points)?;
        }
        Ok(())
    }

    /// Resolve adaptive settings, applying conservative defaults.
    pub fn adaptive_settings(&self) -> AdaptiveSettings {
        let mut settings = self.adaptive.clone().unwrap_or_default();
        if !settings.reservoir_factor.is_finite() || settings.reservoir_factor < 1.0 {
            settings.reservoir_factor = 3.0;
        }
        if !settings.initial_batch_fraction.is_finite() || settings.initial_batch_fraction <= 0.0 {
            settings.initial_batch_fraction = 0.1;
        }
        if !settings.max_batch_fraction.is_finite() || settings.max_batch_fraction <= 0.0 {
            settings.max_batch_fraction = 0.6;
        }
        if settings.max_batch_fraction < settings.initial_batch_fraction {
            settings.max_batch_fraction = settings.initial_batch_fraction;
        }
        if !settings.max_batch_multiplier.is_finite() || settings.max_batch_multiplier < 1.0 {
            settings.max_batch_multiplier = 4.0;
        }
        if settings.max_adaptive_iters == 0 {
            settings.max_adaptive_iters = 10;
        }
        if settings.patience == 0 {
            settings.patience = 2;
        }
        if !settings.convergence_tol.is_finite() || settings.convergence_tol <= 0.0 {
            settings.convergence_tol = (self.tol * 10.0).max(1e-6);
        }
        settings
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

/// Telemetry emitted by training procedures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingTelemetry {
    pub mode: TrainingMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adaptive: Option<AdaptiveTelemetry>,
}

/// Detailed stats for adaptive multiphase training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveTelemetry {
    pub reservoir_sample_size: usize,
    pub stage2_iterations: usize,
    pub stage2_last_batch_size: usize,
    pub stage2_final_shift: f64,
    pub stage2_total_updates: usize,
    pub stage3_iterations: usize,
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
    let mut best_telemetry: Option<TrainingTelemetry> = None;

    for restart_idx in 0..config.n_init {
        let mut restart_rng = ChaCha8Rng::seed_from_u64(rng.next_u64());
        let reservoir_size =
            compute_reservoir_size(points.nrows(), config.k, &config.adaptive_settings());
        let centroids = match config.mode {
            TrainingMode::Adaptive => {
                adaptive_initialise_centroids(points, config, reservoir_size, &mut restart_rng)?
            }
            TrainingMode::FullBatch => match config.init {
                InitStrategy::KMeansPlusPlus => kmeans_pp_init(points, config.k, &mut restart_rng)?,
                InitStrategy::Random => random_init(points, config.k, &mut restart_rng),
            },
        };

        let mut model = KMeans::new(config.clone(), centroids);
        let (outcome, telemetry) = match config.mode {
            TrainingMode::FullBatch => {
                let outcome = model.fit(points, &mut restart_rng);
                (outcome, None)
            }
            TrainingMode::Adaptive => {
                let settings = config.adaptive_settings();
                tracing::info!(
                    reservoir = reservoir_size,
                    "adaptive stage1 reservoir sampling initialised"
                );
                let stage2_stats =
                    adaptive_minibatch_refinement(&mut model, points, &settings, &mut restart_rng);
                tracing::info!(
                    iterations = stage2_stats.stage2_iterations,
                    last_batch = stage2_stats.stage2_last_batch_size,
                    shift = stage2_stats.stage2_final_shift,
                    updates = stage2_stats.stage2_total_updates,
                    "adaptive stage2 minibatch phase complete"
                );
                let mut telemetry = TrainingTelemetry {
                    mode: TrainingMode::Adaptive,
                    adaptive: Some(AdaptiveTelemetry {
                        reservoir_sample_size: reservoir_size,
                        stage2_iterations: stage2_stats.stage2_iterations,
                        stage2_last_batch_size: stage2_stats.stage2_last_batch_size,
                        stage2_final_shift: stage2_stats.stage2_final_shift,
                        stage2_total_updates: stage2_stats.stage2_total_updates,
                        stage3_iterations: 0,
                    }),
                };
                let outcome = model.fit(points, &mut restart_rng);
                tracing::info!(
                    iterations = outcome.iterations,
                    converged = outcome.converged,
                    inertia = outcome.inertia,
                    "adaptive stage3 full-batch polish complete"
                );
                telemetry.adaptive.as_mut().unwrap().stage3_iterations = outcome.iterations;
                (outcome, Some(telemetry))
            }
        };

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
            best_telemetry = telemetry;
        }
    }

    match (best_model, best_outcome) {
        (Some(model), Some(outcome)) => Ok(KMeansRun {
            model,
            outcome,
            telemetry: best_telemetry,
        }),
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
    /// Optional detailed telemetry describing how the model was trained.
    pub telemetry: Option<TrainingTelemetry>,
}

#[derive(Debug, Clone)]
struct AdaptiveStageStats {
    stage2_iterations: usize,
    stage2_last_batch_size: usize,
    stage2_final_shift: f64,
    stage2_total_updates: usize,
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

        distances
            .iter_mut()
            .zip(points.rows())
            .for_each(|(dist_slot, row)| {
                let d = squared_distance(&row, &centroids.row(cid));
                if d < *dist_slot {
                    *dist_slot = d;
                }
            });
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

fn compute_reservoir_size(n: usize, k: usize, settings: &AdaptiveSettings) -> usize {
    let desired = (k as f64 * settings.reservoir_factor).ceil() as usize;
    desired.clamp(k, n.max(1))
}

fn adaptive_initialise_centroids(
    points: &DataMatrix,
    config: &KMeansConfig,
    reservoir_size: usize,
    rng: &mut ChaCha8Rng,
) -> Result<DataMatrix> {
    let sample = reservoir_sample(points, reservoir_size, rng);
    match config.init {
        InitStrategy::KMeansPlusPlus => kmeans_pp_init(&sample, config.k, rng),
        InitStrategy::Random => Ok(random_init(&sample, config.k, rng)),
    }
}

fn reservoir_sample(points: &DataMatrix, sample_size: usize, rng: &mut ChaCha8Rng) -> DataMatrix {
    let (n, dim) = (points.nrows(), points.ncols());
    if sample_size >= n {
        return points.clone();
    }
    let mut sample: Vec<Array1<f64>> = Vec::with_capacity(sample_size);
    for i in 0..sample_size {
        sample.push(points.row(i).to_owned());
    }
    for i in sample_size..n {
        let j = rng.gen_range(0..=i);
        if j < sample_size {
            sample[j] = points.row(i).to_owned();
        }
    }
    let mut arr = Array2::zeros((sample_size, dim));
    for (i, row) in sample.into_iter().enumerate() {
        arr.row_mut(i).assign(&row);
    }
    arr
}

fn adaptive_minibatch_refinement(
    model: &mut KMeans,
    points: &DataMatrix,
    settings: &AdaptiveSettings,
    rng: &mut ChaCha8Rng,
) -> AdaptiveStageStats {
    let n = points.nrows();
    let k = model.centroids.nrows();
    let mut counts = vec![0usize; k];
    let mut prev_shift = f64::INFINITY;
    let mut patience = 0usize;
    let mut last_batch_size = 0usize;
    let mut total_updates = 0usize;

    for iteration in 0..settings.max_adaptive_iters {
        let batch_size =
            compute_batch_size(prev_shift, settings, n, k, model.config.tol.max(1e-12));
        last_batch_size = batch_size;
        let indices = sample_batch_indices(n, batch_size, rng);
        let mut max_shift = 0.0;

        for idx in indices {
            total_updates += 1;
            let point = points.row(idx);
            let cid = model.predict_point(&point);
            counts[cid] += 1;
            let eta = 1.0 / counts[cid] as f64;
            let mut centroid = model.centroids.row_mut(cid);
            for (coord, &value) in centroid.iter_mut().zip(point.iter()) {
                let old = *coord;
                let new_val = old + eta * (value - old);
                let shift = (new_val - old).abs();
                if shift > max_shift {
                    max_shift = shift;
                }
                *coord = new_val;
            }
        }

        if max_shift <= settings.convergence_tol {
            patience += 1;
            if patience >= settings.patience {
                return AdaptiveStageStats {
                    stage2_iterations: iteration + 1,
                    stage2_last_batch_size: last_batch_size,
                    stage2_final_shift: max_shift,
                    stage2_total_updates: total_updates,
                };
            }
        } else {
            patience = 0;
        }
        prev_shift = max_shift;
    }

    AdaptiveStageStats {
        stage2_iterations: settings.max_adaptive_iters,
        stage2_last_batch_size: last_batch_size,
        stage2_final_shift: prev_shift,
        stage2_total_updates: total_updates,
    }
}

fn compute_batch_size(
    prev_shift: f64,
    settings: &AdaptiveSettings,
    n: usize,
    k: usize,
    tol: f64,
) -> usize {
    let base = ((n as f64) * settings.initial_batch_fraction).round() as usize;
    let base = base.max(k).max(1);
    if !prev_shift.is_finite() {
        return base.min(n);
    }
    let ratio = (prev_shift / tol.max(1e-12))
        .clamp(0.5, settings.max_batch_multiplier)
        .max(0.25);
    let candidate = ((base as f64) * ratio).round() as usize;
    let max_allowed = ((n as f64) * settings.max_batch_fraction).ceil() as usize;
    candidate.max(base).min(max_allowed.max(base)).min(n).max(1)
}

fn sample_batch_indices(n: usize, batch_size: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
    if batch_size >= n {
        return (0..n).collect();
    }
    sample(rng, n, batch_size).into_vec()
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
        let rows: Vec<Row> = reader.get_row_iter(None)?.collect();
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
            ..KMeansConfig::default()
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

    #[test]
    fn adaptive_training_executes() {
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let points = generate_points(512, 6, &mut rng);
        let mut config = KMeansConfig {
            k: 6,
            max_iter: 60,
            tol: 1e-5,
            init: InitStrategy::KMeansPlusPlus,
            n_init: 1,
            ..KMeansConfig::default()
        };
        config.mode = TrainingMode::Adaptive;
        config.adaptive = Some(AdaptiveSettings {
            initial_batch_fraction: 0.2,
            max_batch_fraction: 0.8,
            convergence_tol: 5e-4,
            max_adaptive_iters: 12,
            patience: 2,
            ..AdaptiveSettings::default()
        });
        let run =
            kmeans_train_with_restarts(&points, &config, &mut rng).expect("adaptive training");
        assert_eq!(run.model.centroids.nrows(), 6);
        assert!(run.telemetry.is_some());
        let telemetry = run.telemetry.as_ref().unwrap();
        assert_eq!(telemetry.mode, TrainingMode::Adaptive);
        assert!(telemetry.adaptive.as_ref().unwrap().stage2_total_updates > 0);
    }
}
