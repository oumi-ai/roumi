use criterion::{Criterion, criterion_group, criterion_main}; 
use std::hint::black_box; 
use tch::Tensor; 
use anyhow::Result; 

use data_preparation::{
    sample::Sample, 
    sampler::SequentialSampler, 
    transforms::Transform, 
    dataset::InMemoryDataset, 
    dataloader::{DataLoaderConfig, DataLoader},
};

// Simple transform implementations for benchmarking 
struct IdentityTransform; 
impl Transform<String, Sample> for IdentityTransform {
    fn apply(&self, input: String) -> Result<Sample> {
        Ok(Sample::from_single("data", Tensor::from(input.len() as i64)))
    }
}

// Heavier tokenization transform for benchmarking 
struct TokenizerTransform; 
impl Transform<String, Sample> for TokenizerTransform {
    fn apply(&self, input: String) -> Result<Sample> {
        std::thread::sleep(std::time::Duration::from_micros(1500)); 

        let tokens: Vec<i64> = input.chars() 
            .take(128)
            .map(|c| c as i64)
            .collect(); 

        Ok(Sample::from_single("input_ids", Tensor::from_slice(&tokens)))
    }
}

fn bench_inmemory_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("inmemory_throughput");
    group.sample_size(10);
    
    let samples: Vec<String> = (0..1_000)
        .map(|i| format!("Sample text number {:05}", i))
        .collect();
    
    // Test light transform
    for num_workers in [0, 2, 4] {
        group.bench_function(format!("light_transform_workers_{}", num_workers), |bench| {
            let dataset = InMemoryDataset::new(samples.clone())
                .with_transform(IdentityTransform);
            
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
        });
    }
    
    // Test realistic tokenizer transform
    for num_workers in [0, 2, 4] {
        group.bench_function(format!("tokenizer_transform_workers_{}", num_workers), |bench| {
            let dataset = InMemoryDataset::new(samples.clone())
                .with_transform(TokenizerTransform);
            
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
        });
    }
    
    group.finish();
}

criterion_group!(benches, bench_inmemory_throughput); 
criterion_main!(benches); 