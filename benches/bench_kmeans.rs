use criterion::{criterion_group, criterion_main, Criterion};
use kmeans_parallel::{generate_points, kmeans_train_with_restarts, InitStrategy, KMeansConfig};
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
    };
    c.bench_function("kmeans_20k_8d", |b| {
        b.iter(|| {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let _run = kmeans_train_with_restarts(&points, &config, &mut rng).expect("bench run");
        });
    });
}

criterion_group!(benches, bench_kmeans);
criterion_main!(benches);
