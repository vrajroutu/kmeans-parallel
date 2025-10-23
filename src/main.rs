use std::path::{Path, PathBuf};
use std::process;
use std::str::FromStr;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::ThreadPoolBuilder;
use serde_json::json;
use tracing::{error, info};
use tracing_subscriber::FmtSubscriber;

use kmeans_parallel::{
    generate_points, kmeans_train_with_restarts, standardize, DataLoader, DataMatrix, InitStrategy,
    KMeansConfig, KMeansRun, Result as KMeansResult, Standardization, StandardizedData,
};

#[derive(Parser, Debug)]
#[command(name = "kmeans-parallel")]
#[command(about = "Parallel K-Means trainer with production-ready ergonomics", long_about = None)]
struct Args {
    /// Number of clusters
    #[arg(short, long, default_value_t = 4)]
    k: usize,

    /// Number of points to generate when not using --input
    #[arg(short = 'n', long, default_value_t = 100_000)]
    points: usize,

    /// Dimensionality of points when generating
    #[arg(short, long, default_value_t = 2)]
    dim: usize,

    /// Maximum number of iterations
    #[arg(short, long, default_value_t = 50)]
    iterations: usize,

    /// RNG seed (optional)
    #[arg(long, default_value_t = 42u64)]
    seed: u64,

    /// Input dataset (CSV or Parquet)
    #[arg(long)]
    input: Option<PathBuf>,

    /// Explicitly specify the input file format (default: auto-detect from extension)
    #[arg(long, value_enum)]
    format: Option<InputFormat>,

    /// Output file for centroids and stats (JSON)
    #[arg(short, long, default_value = "kmeans_result.json")]
    output: PathBuf,

    /// Optional file containing per-sample cluster assignments (CSV)
    #[arg(long)]
    assignments: Option<PathBuf>,

    /// Initialization method
    #[arg(long, value_parser = parse_init_strategy, default_value = "kmeans++")]
    init: InitStrategy,

    /// Number of restarts (choose best by inertia)
    #[arg(long, default_value_t = 1)]
    n_init: usize,

    /// Convergence tolerance (absolute centroid shift)
    #[arg(long, default_value_t = 1e-6)]
    tol: f64,

    /// Save trained model JSON
    #[arg(long)]
    save_model: Option<PathBuf>,

    /// Whether to z-score standardize inputs prior to training
    #[arg(long, default_value_t = false)]
    standardize: bool,

    /// Override Rayon global thread pool size
    #[arg(long)]
    threads: Option<usize>,

    /// Verbosity: set RUST_LOG style level (info, debug, warn)
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum InputFormat {
    Csv,
    Parquet,
}

fn main() {
    let args = Args::parse();
    if let Err(err) = init_logging(&args.log_level) {
        eprintln!("failed to initialise logging: {err}");
    }

    if let Err(err) = run(args) {
        error!(error = %err, "kmeans run failed");
        process::exit(1);
    }
}

fn init_logging(level: &str) -> Result<(), String> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(match level {
            "debug" => tracing::Level::DEBUG,
            "warn" => tracing::Level::WARN,
            "error" => tracing::Level::ERROR,
            _ => tracing::Level::INFO,
        })
        .finish();
    tracing::subscriber::set_global_default(subscriber).map_err(|err| err.to_string())
}

fn run(args: Args) -> KMeansResult<()> {
    if let Some(threads) = args.threads {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|err| {
                kmeans_parallel::KMeansError::InvalidConfig(format!(
                    "failed to configure rayon threadpool: {err}"
                ))
            })?;
        info!(threads, "configured rayon global thread pool");
    }

    info!(
        k = args.k,
        n_init = args.n_init,
        tol = args.tol,
        max_iter = args.iterations,
        seed = args.seed,
        "starting kmeans optimisation"
    );

    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let raw_data = load_data(
        args.input.as_deref(),
        args.format,
        args.points,
        args.dim,
        &mut rng,
    )?;

    let (data, standardization): (DataMatrix, Option<Standardization>) = if args.standardize {
        let StandardizedData { data, params } = standardize(&raw_data);
        info!("applied z-score standardization to input features");
        (data, Some(params))
    } else {
        (raw_data, None)
    };

    let config = KMeansConfig {
        k: args.k,
        max_iter: args.iterations,
        tol: args.tol,
        init: args.init,
        n_init: args.n_init,
    };

    let start = Instant::now();
    let run: KMeansRun = kmeans_train_with_restarts(&data, &config, &mut rng)?;
    let elapsed = start.elapsed();
    info!(
        inertia = run.outcome.inertia,
        iterations = run.outcome.iterations,
        converged = run.outcome.converged,
        took_seconds = elapsed.as_secs_f64(),
        "kmeans optimisation finished"
    );
    info!(?run.outcome.cluster_sizes, "cluster population counts");

    write_result(
        &args,
        &data,
        &run,
        standardization,
        elapsed.as_secs_f64(),
        args.seed,
    )?;

    if let Some(path) = args.save_model.as_ref() {
        run.model.save_model(path)?;
        info!(path = ?path, "saved model snapshot");
    }

    if let Some(path) = args.assignments.as_ref() {
        write_assignments(path, &run)?;
    }

    Ok(())
}

fn load_data(
    input: Option<&Path>,
    format: Option<InputFormat>,
    points: usize,
    dim: usize,
    rng: &mut ChaCha8Rng,
) -> KMeansResult<DataMatrix> {
    if let Some(path) = input {
        let format_to_use =
            format.unwrap_or_else(|| infer_format(path).unwrap_or(InputFormat::Csv));
        info!(path = ?path, ?format_to_use, "loading input data");
        match format_to_use {
            InputFormat::Csv => DataLoader::load_csv(path),
            InputFormat::Parquet => DataLoader::load_parquet(path),
        }
    } else {
        if let Some(requested) = format {
            info!(
                ?requested,
                "ignoring --format because synthetic data will be generated"
            );
        }
        info!(points, dim, "generating synthetic uniform data");
        Ok(generate_points(points, dim, rng))
    }
}

fn infer_format(path: &Path) -> Option<InputFormat> {
    path.extension().and_then(|ext| ext.to_str()).map(|ext| {
        match ext.to_ascii_lowercase().as_str() {
            "parquet" | "pq" => InputFormat::Parquet,
            "csv" => InputFormat::Csv,
            _ => InputFormat::Csv,
        }
    })
}

fn write_result(
    args: &Args,
    data: &DataMatrix,
    run: &KMeansRun,
    standardization: Option<Standardization>,
    elapsed_secs: f64,
    seed: u64,
) -> KMeansResult<()> {
    let centroids: Vec<Vec<f64>> = (0..run.model.centroids.nrows())
        .map(|row| run.model.centroids.row(row).to_vec())
        .collect();
    let cluster_sizes = run.outcome.cluster_sizes.clone();

    let dump = json!({
        "k": run.model.centroids.nrows(),
        "dim": data.ncols(),
        "rows": data.nrows(),
        "iterations": run.outcome.iterations,
        "inertia": run.outcome.inertia,
        "converged": run.outcome.converged,
        "cluster_sizes": cluster_sizes,
        "init": run.model.config.init,
        "seed": seed,
        "elapsed_seconds": elapsed_secs,
        "standardized": args.standardize,
        "config": {
            "max_iter": run.model.config.max_iter,
            "tol": run.model.config.tol,
            "n_init": run.model.config.n_init,
        },
        "centroids": centroids,
        "data_source": if let Some(path) = args.input.as_ref() {
            let fmt = args
                .format
                .or_else(|| infer_format(path))
                .unwrap_or(InputFormat::Csv);
            json!({
                "type": "file",
                "path": path.display().to_string(),
                "format": format!("{fmt:?}").to_lowercase(),
            })
        } else {
            json!({
                "type": "synthetic",
                "points": args.points,
                "dim": args.dim,
            })
        },
        "assignments_path": args.assignments.as_ref().map(|p| p.display().to_string()),
        "standardization": standardization.as_ref().map(|params| {
            json!({
                "mean": params.mean.to_vec(),
                "std": params.std.to_vec(),
            })
        }),
    });

    std::fs::write(&args.output, serde_json::to_string_pretty(&dump)?)?;
    info!(path = ?args.output, "wrote clustering summary");
    Ok(())
}

fn write_assignments(path: &Path, run: &KMeansRun) -> KMeansResult<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record(["index", "cluster"])?;
    for (idx, cluster) in run.outcome.assignments.iter().enumerate() {
        writer.write_record([idx.to_string(), cluster.to_string()])?;
    }
    writer.flush()?;
    info!(path = ?path, "wrote assignments CSV");
    Ok(())
}

fn parse_init_strategy(raw: &str) -> Result<InitStrategy, String> {
    InitStrategy::from_str(raw).map_err(|err| err.to_string())
}
