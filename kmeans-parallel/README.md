# kmeans-parallel

Production-ready, parallelised k-means clustering implemented in Rust. The crate provides:

- A reusable library with deterministic initialisation (`kmeans++` and random), restart handling, inertia tracking, cluster population metrics, and optional data standardisation helpers.
- A CLI (`kmeans-parallel`) that can ingest CSV or Parquet datasets, generate synthetic data when no input is supplied, and emit rich JSON reports plus optional assignment CSVs.
- Structured error handling and telemetry via `tracing`, making it straightforward to embed into larger observability pipelines.

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
};

let run = kmeans_train_with_restarts(&data, &config, &mut rng).expect("training succeeds");
println!("Inertia: {}", run.outcome.inertia);
println!("Cluster sizes: {:?}", run.outcome.cluster_sizes);
```

## Testing

Run the test suite with:

```bash
cargo test
```

The workspace requires access to crates.io for the first build to download dependencies.
