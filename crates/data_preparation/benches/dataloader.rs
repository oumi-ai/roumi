use anyhow::Result;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use tch::Tensor;

use data_preparation::{
    dataloader::{DataLoader, DataLoaderConfig},
    dataset::{DataSource, InMemoryDataset, IterableDataset},
    sample::Sample,
    sampler::SequentialSampler,
    transforms::Transform,
};

// =========================== Helper functions  ===========================
// Simple transform implementations for benchmarking
struct IdentityTransform;
impl Transform<String, Sample> for IdentityTransform {
    fn apply(&self, input: String) -> Result<Sample> {
        Ok(Sample::from_single(
            "data",
            Tensor::from(input.len() as i64),
        ))
    }
}

/// Heavier tokenization transform for benchmarking
struct TokenizerTransform;
impl Transform<String, Sample> for TokenizerTransform {
    fn apply(&self, input: String) -> Result<Sample> {
        std::thread::sleep(std::time::Duration::from_micros(1500));

        let tokens: Vec<i64> = input.chars().take(128).map(|c| c as i64).collect();

        Ok(Sample::from_single(
            "input_ids",
            Tensor::from_slice(&tokens),
        ))
    }
}

// Mock data source that simulates I/O-bound operations (e.g., reading from disk/network).
// Each sample incurs a configurable delay to mimic real-world latency.
#[derive(Clone)]
struct DummySource {
    size: usize,
    delay_ms: u64,
}

impl DataSource<String> for DummySource {
    fn stream(&self) -> Result<Box<dyn Iterator<Item = Result<String>> + Send>> {
        let data: Vec<String> = (0..self.size)
            .map(|i| format!("Stream sample {}", i))
            .collect();

        let delay = self.delay_ms;

        Ok(Box::new(data.into_iter().map(move |string_input| {
            if delay > 0 {
                std::thread::sleep(std::time::Duration::from_millis(delay));
            }
            Ok(string_input)
        })))
    }
}

// =========================== Benchmarks  ===========================
fn bench_inmemory_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("inmemory_throughput");
    group.sample_size(10);

    let samples: Vec<String> = (0..1_000)
        .map(|i| format!("Sample text number {:05}", i))
        .collect();

    // Test light transform
    for num_workers in [0, 2, 4] {
        group.bench_function(
            format!("light_transform_workers_{}", num_workers),
            |bench| {
                let dataset =
                    InMemoryDataset::new(samples.clone()).with_transform(IdentityTransform);

                let config = DataLoaderConfig::builder()
                    .batch_size(32)
                    .num_workers(num_workers)
                    .prefetch_factor(num_workers.max(1) * 2)
                    .build();

                let sampler = SequentialSampler::new(dataset.len());
                let loader = DataLoader::new(dataset, sampler, config).unwrap();

                bench.iter(|| {
                    let mut total_samples = 0;
                    for batch in loader.iter().unwrap() {
                        let batch = batch.unwrap();
                        total_samples += batch.batch_size().unwrap() as usize;
                        black_box(batch);
                    }
                    assert_eq!(total_samples, 1000);
                });
            },
        );
    }

    // Test realistic tokenizer transform
    for num_workers in [0, 2, 4] {
        group.bench_function(
            format!("tokenizer_transform_workers_{}", num_workers),
            |bench| {
                let dataset =
                    InMemoryDataset::new(samples.clone()).with_transform(TokenizerTransform);

                let config = DataLoaderConfig::builder()
                    .batch_size(32)
                    .num_workers(num_workers)
                    .prefetch_factor(num_workers.max(1) * 2)
                    .build();

                let sampler = SequentialSampler::new(dataset.len());
                let loader = DataLoader::new(dataset, sampler, config).unwrap();

                bench.iter(|| {
                    let mut total_samples = 0;
                    for batch in loader.iter().unwrap() {
                        let batch = batch.unwrap();
                        total_samples += batch.batch_size().unwrap() as usize;
                        black_box(batch);
                    }
                    assert_eq!(total_samples, 1000);
                });
            },
        );
    }

    group.finish();
}

fn bench_iterable_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("iterable_throughput");
    group.sample_size(10);

    for num_workers in [0, 2, 4] {
        group.bench_function(format!("workers_{}", num_workers), |bench| {
            let sources: Vec<Box<dyn DataSource<String>>> = (0..4)
                .map(|_| {
                    Box::new(DummySource {
                        size: 125,
                        delay_ms: 1,
                    }) as Box<dyn DataSource<String>>
                })
                .collect();

            let dataset = IterableDataset::new(sources).with_transform(IdentityTransform);

            let config = DataLoaderConfig::builder()
                .batch_size(32)
                .num_workers(num_workers)
                .build();

            let loader = DataLoader::new_iterable(dataset, config).unwrap();

            bench.iter(|| {
                let mut count = 0;
                for batch in loader.iter().unwrap() {
                    let batch = batch.unwrap();
                    count += batch.batch_size().unwrap() as usize;
                    black_box(batch);
                }
                assert_eq!(count, 500);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_inmemory_throughput,
    bench_iterable_throughput
);
criterion_main!(benches);
