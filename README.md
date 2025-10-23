# kmeans-parallel

[![Continuous Integration](https://github.com/vrajkishorerv/kmeans-parallel/actions/workflows/ci.yml/badge.svg)](https://github.com/vrajkishorerv/kmeans-parallel/actions/workflows/ci.yml)

High-performance, parallelised k-means clustering implemented in Rust by **Vraj Routu**. The crate provides:

- A reusable library with deterministic initialisation (`kmeans++` and random), restart handling, inertia tracking, cluster population metrics, and optional data standardisation helpers.
- A CLI (`kmeans-parallel`) that can ingest CSV or Parquet datasets, generate synthetic data when no input is supplied, and emit rich JSON reports plus optional assignment CSVs.
- Structured error handling and telemetry via `tracing`, making it straightforward to embed into larger observability pipelines.

## Breakthrough Highlights

- **Full-stack clustering toolkit** – same crate powers the CLI, library, release binaries, and multiple restarts with deterministic seeds.
- **Adaptive multiphase training** – optional `--mode adaptive` streams reservoir seeding, dynamic minibatches, and a final polish pass for rapid convergence.
- **Production-grade ingestion** – native CSV and Parquet loaders with schema validation, z-score standardisation, and serde-friendly config.
- **Observability-first design** – structured logging via `tracing`, diagnostic outputs (iterations, inertia, cluster sizes), and model persistence hooks.
- **Automated delivery** – GitHub Actions pipeline gates every change with linting, tests, and bench compilation, then publishes signed release tarballs on `v*` tags.

## Features

- **Parallel training** via `rayon`, exploiting all available cores (configurable with `--threads`).
- **Multiple restarts** with best-run selection based on inertia.
- **Resilient centroid management** that respawns empty clusters and exposes cluster sizes.
- **Data ingestion** from CSV or Parquet, with robust schema validation.
- **Standardisation utilities** for z-scoring features when clustering heterogeneous scales.
- **Model persistence** to JSON for reproducible deployments.

## CLI Usage

Generate synthetic data and train 6 clusters on three-dimensional points:

```bash
cargo run --release -- --k 6 --dim 3 --points 200_000 --standardize
```

Accelerate large datasets with the adaptive multiphase pipeline:

```bash
cargo run --release -- \
  --k 12 \
  --mode adaptive \
  --adaptive-initial-fraction 0.05 \
  --adaptive-max-fraction 0.8 \
  --adaptive-batch-tol 5e-4
```

Cluster an existing dataset while exporting centroids, diagnostics, and assignments:

```bash
cargo run --release -- \
  --input data/customers.parquet \
  --format parquet \
  --k 8 \
  --n-init 5 \
  --iterations 300 \
  --tol 1e-5 \
  --assignments outputs/assignments.csv \
  --output outputs/summary.json
```

Both commands emit a JSON summary containing configuration, convergence stats, elapsed runtime, and centroid coordinates. When `--standardize` is enabled, the z-score parameters are recorded for downstream inference.

## Benchmark Snapshot

Criterion 0.5 (macOS, release build) on 20k × 8 synthetic points:

- `kmeans_full_20k_8d`: **60.2 – 64.5 ms** (median ≈ 62.0 ms)
- `kmeans_adaptive_20k_8d`: **60.1 – 63.2 ms** (median ≈ 61.5 ms)

The adaptive pipeline keeps pace with full-batch Lloyd while reducing the amount of data touched during the early iterations.

## Library Usage

```rust
use kmeans_parallel::{
    kmeans_train_with_restarts, standardize, InitStrategy, KMeansConfig,
};
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

let mut rng = ChaCha8Rng::seed_from_u64(42);
let points = kmeans_parallel::generate_points(10_000, 4, &mut rng);
let StandardizedData { data, params: _ } = standardize(&points);

let config = KMeansConfig {
    k: 4,
    max_iter: 200,
    tol: 1e-5,
    init: InitStrategy::KMeansPlusPlus,
    n_init: 3,
    ..KMeansConfig::default()
};

let run = kmeans_train_with_restarts(&data, &config, &mut rng).expect("training succeeds");
println!("Inertia: {}", run.outcome.inertia);
println!("Cluster sizes: {:?}", run.outcome.cluster_sizes);
```

Want the adaptive pipeline from code? Set `config.mode = TrainingMode::Adaptive` and optionally tweak `config.adaptive`.

## Sample JSON Report

The CLI emits a rich JSON artefact capturing configuration, convergence, and adaptive telemetry:

```json
{
  "k": 3,
  "iterations": 3,
  "inertia": 34.06500554290972,
  "mode": "adaptive",
  "cluster_sizes": [75, 88, 93],
  "telemetry": {
    "mode": "adaptive",
    "adaptive": {
      "reservoir_sample_size": 9,
      "stage2_iterations": 25,
      "stage2_last_batch_size": 52,
      "stage2_total_updates": 1261,
      "stage3_iterations": 3
    }
  }
}
```

## Testing

Run the test suite with:

```bash
cargo test
```

The workspace requires access to crates.io for the first build to download dependencies.

## Release Pipeline

Tagged pushes matching `v*` trigger the CI workflow to build release binaries, attach them to GitHub Releases, and publish release notes automatically. To ship a new version:

```bash
cargo test && cargo bench --bench bench_kmeans
git tag v0.2.0
git push origin v0.2.0
```

Within minutes the macOS tarball appears on the release page ready for download. Continuous integration runs on every PR and push to `main`, ensuring the project stays production-ready.
