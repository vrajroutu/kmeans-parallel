use criterion::{criterion_group, criterion_main, Criterion};
use kmeans_parallel::{
    generate_points, kmeans_train_with_restarts, AdaptiveSettings, InitStrategy, KMeansConfig,
    TrainingMode,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn bench_kmeans(c: &mut Criterion) {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let points = generate_points(20_000, 8, &mut rng);
    let config = KMeansConfig {
        k: 8,
        max_iter: 100,
        tol: 1e-6,
        init: InitStrategy::KMeansPlusPlus,
        n_init: 1,
        ..KMeansConfig::default()
    };
    c.bench_function("kmeans_full_20k_8d", |b| {
        b.iter(|| {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let _run =
                kmeans_train_with_restarts(&points, &config, &mut rng).expect("full bench run");
        });
    });

    let mut adaptive_config = config.clone();
    adaptive_config.mode = TrainingMode::Adaptive;
    adaptive_config.adaptive = Some(AdaptiveSettings {
        reservoir_factor: 2.5,
        initial_batch_fraction: 0.05,
        max_batch_fraction: 0.5,
        max_batch_multiplier: 3.0,
        max_adaptive_iters: 20,
        patience: 2,
        convergence_tol: 5e-4,
    });

    c.bench_function("kmeans_adaptive_20k_8d", |b| {
        b.iter(|| {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let _run = kmeans_train_with_restarts(&points, &adaptive_config, &mut rng)
                .expect("adaptive bench run");
        });
    });
}

criterion_group!(benches, bench_kmeans);
criterion_main!(benches);
