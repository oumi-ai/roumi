//! Basic single-threaded tests for DataLoader functionality.
//!
//! Tests cover:
//! - InMemory datasets with auto-sampling and custom samplers
//! - Batch samplers including bucket sampling
//! - Iterable datasets with multiple data sources
//! - DataLoader configuration validation
//! - Shuffle determinism with seeds
//! - Transform error propagation

mod common;
use common::{StringToSample, TestDataSource};
use data_preparation::{
    collator::{PaddingCollator, PaddingRule},
    dataloader::{DataLoader, DataLoaderConfig},
    dataset::{DataSource, InMemoryDataset, IterableDataset},
    sample::Sample,
    sampler::{
        BatchBucketSampler, BatchSampler, RandomSampler, SequentialSampler, SubsetRandomSampler,
    },
    transforms::Transform,
};

use anyhow::{anyhow, Result};

// ================================================================================================
// 1. Basic InMemory Tests
// ================================================================================================
#[test]
fn test_dataloader_inmemory_basic() -> Result<()> {
    let data = vec!["hello".to_string(), "world".to_string(), "rust".to_string()];
    let dataset = InMemoryDataset::new(data).with_transform(StringToSample);

    let config = DataLoaderConfig::builder()
        .batch_size(2)
        .num_workers(0)
        .drop_last(false)
        .shuffle(false)
        .build();

    // Use DataLoader::new() for auto-sampling (no sampler needed)
    let dataloader = DataLoader::new(dataset, config)?;

    let batches: Vec<_> = dataloader.iter()?.collect::<Result<Vec<_>>>()?;
    assert_eq!(batches.len(), 2); // 3 samples, batch_size=2, drop_last=false -> 2 batches
    assert_eq!(batches[0].batch_size()?, 2); // First batch: full
    assert_eq!(batches[1].batch_size()?, 1); // Second batch: partial

    Ok(())
}

#[test]
fn test_dataloader_inmemory_drop_last() -> Result<()> {
    let data = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let dataset = InMemoryDataset::new(data).with_transform(StringToSample);

    let config = DataLoaderConfig::builder()
        .batch_size(2)
        .drop_last(true)
        .build();

    // Use DataLoader::new() for auto-sampling
    let dataloader = DataLoader::new(dataset, config)?;

    let batches: Vec<_> = dataloader.iter()?.collect::<Result<Vec<_>>>()?;
    assert_eq!(batches.len(), 1); // Only complete batches
    assert_eq!(batches[0].batch_size()?, 2);

    Ok(())
}

#[test]
fn test_dataloader_inmemory_with_custom_sampler() -> Result<()> {
    // Create data where each item has a unique, identifiable length
    let data: Vec<String> = (0..10).map(|i| "x".repeat(i + 1)).collect();
    let dataset = InMemoryDataset::new(data).with_transform(StringToSample);

    // Use SubsetRandomSampler to verify DataLoader uses the provided sampler
    let subset_indices = vec![2, 5, 7]; // Will have lengths 3, 6, 8
    let sampler = SubsetRandomSampler::new(dataset.len(), subset_indices, 42)?;

    let config = DataLoaderConfig::builder().batch_size(2).build();

    let dataloader = DataLoader::new_with_sampler(dataset, sampler, config)?;

    // Collect all lengths
    let mut seen_lengths = Vec::new();
    for batch in dataloader.iter()? {
        let batch = batch?;
        let lengths = batch.get("length")?;
        for i in 0..batch.batch_size()? {
            seen_lengths.push(lengths.int64_value(&[i]));
        }
    }
    assert_eq!(
        seen_lengths.len(),
        3,
        "Should only get samples from subset indices"
    );
    seen_lengths.sort_unstable();

    // Verify we only got data from the specified indices
    assert_eq!(
        seen_lengths,
        vec![3, 6, 8],
        "DataLoader should only yield samples from indices specified by the sampler"
    );

    Ok(())
}

#[test]
fn test_dataloader_inmemory_with_batch_sampler() -> Result<()> {
    let data: Vec<String> = (0..6).map(|i| format!("item{}", i)).collect();
    let dataset = InMemoryDataset::new(data).with_transform(StringToSample);

    let base_sampler = SequentialSampler::new(dataset.len());
    let batch_sampler = BatchSampler::new(base_sampler, 2, false)?;

    let config = DataLoaderConfig::builder().build();

    let loader = DataLoader::new_with_batch_sampler(dataset, batch_sampler, config)?;
    let batches: Vec<_> = loader.iter()?.collect::<Result<Vec<_>>>()?;

    // Should have 3 batches (6 samples / batch_size 2)
    assert_eq!(batches.len(), 3);

    // Verify each batch has correct size
    for batch in &batches {
        assert_eq!(batch.batch_size()?, 2);
    }

    Ok(())
}

#[test]
fn test_dataloader_batch_sampler_with_custom_collator() -> Result<()> {
    // Create dataset with variable-length strings
    let data = vec![
        "a".to_string(),          // length 1
        "bb".to_string(),         // length 2
        "ccccccccc".to_string(),  // length 9
        "dddddddddd".to_string(), // length 10
        "ee".to_string(),         // length 2
        "f".to_string(),          // length 1
    ];

    let dataset = InMemoryDataset::new(data.clone()).with_transform(StringToSample);

    // Create batch bucket sampler that groups by length
    let base_sampler = SequentialSampler::new(dataset.len());
    let data_ref = data.clone();
    let bucket_sampler = BatchBucketSampler::new(
        base_sampler,
        2,                                     // batch_size
        false,                                 // drop_last
        move |idx| data_ref[idx].len() as f64, // sort by string length
        2,                                     // bucket_multiplier (bucket_size = 4)
        42,                                    // seed
    )?;

    // Use PaddingCollator for variable lengths
    let padding_collator =
        PaddingCollator::new().pad("length", vec![(0, PaddingRule::MaxLength)], Some(0.0));

    let config = DataLoaderConfig::builder().build();

    let loader = DataLoader::new_with_batch_sampler_and_collator(
        dataset,
        bucket_sampler,
        config,
        padding_collator,
    )?;

    // Collect all batches and their lengths
    let mut batch_lengths = Vec::new();
    for batch in loader.iter()? {
        let batch = batch?;
        let tensor = batch.get("length")?;
        let batch_size = batch.batch_size()?;

        let mut lengths = Vec::new();
        for i in 0..batch_size {
            lengths.push(tensor.int64_value(&[i]));
        }
        batch_lengths.push(lengths);
    }

    // Verify batches group similar lengths together
    for (i, lengths) in batch_lengths.iter().enumerate() {
        if lengths.len() > 1 {
            let diff = (lengths[0] - lengths[1]).abs();
            assert!(
                diff <= 8,
                "Batch {} has too large length difference: {:?}",
                i,
                lengths
            );
        }
    }

    Ok(())
}

// ================================================================================================
// 2. Basic Iterable Tests
// ================================================================================================
#[test]
fn test_dataloader_iterable_basic() -> Result<()> {
    let source = TestDataSource {
        data: vec!["hello".to_string(), "world".to_string(), "rust".to_string()],
    };
    let dataset = IterableDataset::new(vec![Box::new(source) as Box<dyn DataSource<String>>])
        .with_transform(StringToSample);

    let config = DataLoaderConfig::builder().batch_size(2).build();

    let dataloader = DataLoader::new_iterable(dataset, config)?;
    let batches: Vec<_> = dataloader.iter()?.collect::<Result<Vec<_>>>()?;

    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].batch_size()?, 2);
    assert_eq!(batches[1].batch_size()?, 1);

    Ok(())
}

#[test]
fn test_dataloader_iterable_drop_last() -> Result<()> {
    let source = TestDataSource {
        data: vec!["a".to_string(), "b".to_string(), "c".to_string()],
    };
    let dataset = IterableDataset::new(vec![Box::new(source) as Box<dyn DataSource<String>>])
        .with_transform(StringToSample);

    let config = DataLoaderConfig::builder()
        .batch_size(2)
        .drop_last(true)
        .build();

    let dataloader = DataLoader::new_iterable(dataset, config)?;
    let batches: Vec<_> = dataloader.iter()?.collect::<Result<Vec<_>>>()?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].batch_size()?, 2);

    Ok(())
}

#[test]
fn test_dataloader_iterable_multiple_sources() -> Result<()> {
    // Create sources with identifiable data
    let source1 = TestDataSource {
        data: vec!["a".to_string(), "bb".to_string()], // lengths 1, 2
    };
    let source2 = TestDataSource {
        data: vec!["ccc".to_string(), "dddd".to_string()], // lengths 3, 4
    };
    let source3 = TestDataSource {
        data: vec!["eeeee".to_string(), "ffffff".to_string()], // lengths 5, 6
    };

    let dataset = IterableDataset::new(vec![
        Box::new(source1) as Box<dyn DataSource<String>>,
        Box::new(source2) as Box<dyn DataSource<String>>,
        Box::new(source3) as Box<dyn DataSource<String>>,
    ])
    .with_transform(StringToSample);

    // Test with batch_size=3 to force cross-source batching
    let config = DataLoaderConfig::builder().batch_size(3).build();

    let dataloader = DataLoader::new_iterable(dataset.clone(), config)?;

    let mut batches = Vec::new();
    for batch in dataloader.iter()? {
        let batch = batch?;
        let lengths = batch.get("length")?;
        let batch_lengths: Vec<i64> = (0..batch.batch_size()?)
            .map(|i| lengths.int64_value(&[i]))
            .collect();
        batches.push(batch_lengths);
    }

    // With 6 total samples and batch_size=3, we should get exactly 2 batches
    assert_eq!(batches.len(), 2, "Should have exactly 2 batches");

    // First batch should contain samples from source1 and part of source2
    assert_eq!(
        batches[0],
        vec![1, 2, 3],
        "First batch should contain: source1[a,bb] + source2[ccc]"
    );

    // Second batch should contain rest of source2 and source3
    assert_eq!(
        batches[1],
        vec![4, 5, 6],
        "Second batch should contain: source2[dddd] + source3[eeeee,ffffff]"
    );

    // Test with batch_size=4 for different cross-source pattern
    let config4 = DataLoaderConfig::builder().batch_size(4).build();

    let dataloader4 = DataLoader::new_iterable(dataset.clone(), config4)?;

    let mut batches4 = Vec::new();
    for batch in dataloader4.iter()? {
        let batch = batch?;
        let lengths = batch.get("length")?;
        let batch_lengths: Vec<i64> = (0..batch.batch_size()?)
            .map(|i| lengths.int64_value(&[i]))
            .collect();
        batches4.push(batch_lengths);
    }

    assert_eq!(batches4.len(), 2, "Should have 2 batches (4+2)");
    assert_eq!(
        batches4[0],
        vec![1, 2, 3, 4],
        "First batch spans source1 and source2"
    );
    assert_eq!(
        batches4[1],
        vec![5, 6],
        "Second batch contains only source3 (partial)"
    );

    // Test edge case: batch_size larger than any single source
    let config5 = DataLoaderConfig::builder().batch_size(5).build();

    let dataloader5 = DataLoader::new_iterable(dataset, config5)?;

    let mut batches5 = Vec::new();
    for batch in dataloader5.iter()? {
        let batch = batch?;
        let lengths = batch.get("length")?;
        let batch_lengths: Vec<i64> = (0..batch.batch_size()?)
            .map(|i| lengths.int64_value(&[i]))
            .collect();
        batches5.push(batch_lengths);
    }

    assert_eq!(batches5.len(), 2, "Should have 2 batches (5+1)");
    assert_eq!(
        batches5[0],
        vec![1, 2, 3, 4, 5],
        "First batch spans all three sources"
    );
    assert_eq!(batches5[1], vec![6], "Second batch has remaining sample");

    Ok(())
}

// ================================================================================================
// 3. Configuration Validation Tests
// ================================================================================================
#[test]
fn test_dataloader_invalid_config() -> Result<()> {
    let dataset = InMemoryDataset::new(vec!["test".to_string()]).with_transform(StringToSample);

    // Test batch_size = 0
    let config = DataLoaderConfig::builder().batch_size(0).build();

    let result = DataLoader::new(dataset.clone(), config);
    assert!(result.is_err(), "Should fail with batch_size = 0");

    // Test providing sampler with shuffle = true
    let sampler = SequentialSampler::new(dataset.len());
    let config = DataLoaderConfig::builder()
        .batch_size(1)
        .shuffle(true) // Invalid with custom sampler
        .build();

    let result = DataLoader::new_with_sampler(dataset.clone(), sampler, config);
    assert!(
        result.is_err(),
        "Should fail when shuffle=true with custom sampler"
    );

    Ok(())
}

#[test]
fn test_dataloader_batch_sampler_invalid_config() -> Result<()> {
    let data = vec!["test".to_string()];
    let dataset = InMemoryDataset::new(data).with_transform(StringToSample);

    let base_sampler = SequentialSampler::new(dataset.len());
    let batch_sampler = BatchSampler::new(base_sampler, 2, false)?;

    // Test: batch_size must not be specified with batch sampler
    let config = DataLoaderConfig::builder().batch_size(10).build();

    let result = DataLoader::new_with_batch_sampler(dataset.clone(), batch_sampler.clone(), config);
    assert!(
        result.is_err(),
        "Should fail when batch size is specified with batch sampler"
    );

    // Test: drop_last must not be specified with batch sampler
    let config = DataLoaderConfig::builder().drop_last(true).build();

    let result = DataLoader::new_with_batch_sampler(dataset.clone(), batch_sampler.clone(), config);
    assert!(
        result.is_err(),
        "Should fail when drop_last is specified with batch sampler"
    );

    // Test: shuffle must be false when specified with batch sampler
    let config = DataLoaderConfig::builder().shuffle(true).build();

    let result = DataLoader::new_with_batch_sampler(dataset, batch_sampler, config);
    assert!(
        result.is_err(),
        "Should fail when shuffle is called with batch sampler"
    );
    Ok(())
}

#[test]
fn test_dataloader_seed_mismatch_with_sampler_seed() -> Result<()> {
    let dataset = InMemoryDataset::new(vec!["test".to_string()]).with_transform(StringToSample);
    let sampler = RandomSampler::new(dataset.len(), false, None, 42)?;

    // Test seed mismatch
    let config = DataLoaderConfig::builder().batch_size(1).seed(1337).build();

    let result = DataLoader::new_with_sampler(dataset, sampler, config);
    assert!(result.is_err(), "Should fail with mismatched seeds");

    Ok(())
}

// ================================================================================================
// 4. DataLoader Behaviour Tests
// ================================================================================================
#[test]
fn test_dataloader_shuffles_determiniscally_with_seed() -> Result<()> {
    use std::collections::HashSet;

    // Create data with unique lengths to avoid collisions
    let data: Vec<String> = (0..100).map(|i| "x".repeat(i + 1)).collect();
    let dataset = InMemoryDataset::new(data).with_transform(StringToSample);

    // Test with multiple seeds
    for &seed in &[42, 1337, 0xdeadbeef] {
        let config = DataLoaderConfig::builder()
            .batch_size(10)
            .shuffle(true)
            .seed(seed)
            .build();

        // Create two dataloaders with same seed to test determinism
        let dataloader1 = DataLoader::new(dataset.clone(), config.clone())?;
        let dataloader2 = DataLoader::new(dataset.clone(), config)?;

        // Collect all values from first epoch of each dataloader
        let mut values1 = Vec::new();
        let mut values2 = Vec::new();

        for batch in dataloader1.iter()? {
            let batch = batch?;
            let lengths = batch.get("length")?;
            for i in 0..batch.batch_size()? {
                values1.push(lengths.int64_value(&[i]));
            }
        }

        for batch in dataloader2.iter()? {
            let batch = batch?;
            let lengths = batch.get("length")?;
            for i in 0..batch.batch_size()? {
                values2.push(lengths.int64_value(&[i]));
            }
        }

        // Verify identical order with same seed
        assert_eq!(
            values1, values2,
            "Shuffle results differ with same seed {}",
            seed
        );

        // Verify all samples are present (no duplicates/loss)
        let all_samples: HashSet<_> = values1.iter().cloned().collect();
        assert_eq!(all_samples.len(), 100, "Missing samples with seed {}", seed);

        // Verify we got the expected range of lengths (1 to 100)
        let min_length = all_samples.iter().min().unwrap();
        let max_length = all_samples.iter().max().unwrap();
        assert_eq!(*min_length, 1, "Minimum length should be 1");
        assert_eq!(*max_length, 100, "Maximum length should be 100");

        // Verify consecutive epochs have different order (shuffle works)
        let mut epoch2_values = Vec::new();
        for batch in dataloader1.iter()? {
            let batch = batch?;
            let lengths = batch.get("length")?;
            for i in 0..batch.batch_size()? {
                epoch2_values.push(lengths.int64_value(&[i]));
            }
        }

        // Should be different order in epoch 2
        assert_ne!(
            values1, epoch2_values,
            "Epochs should have different shuffle order with seed {}",
            seed
        );
    }

    Ok(())
}

#[test]
fn test_dataloader_transform_error_propagates() -> Result<()> {
    use tch::Tensor;

    // Create a transform that fails on certain inputs
    #[derive(Clone)]
    struct FailingTransform;

    impl Transform<String, Sample> for FailingTransform {
        fn apply(&self, input: String) -> Result<Sample> {
            if input.contains("fail") {
                Err(anyhow!("Transform failed on input: {}", input))
            } else {
                Ok(Sample::from_single(
                    "length",
                    Tensor::from(input.len() as i64),
                ))
            }
        }
    }

    unsafe impl Send for FailingTransform {}
    unsafe impl Sync for FailingTransform {}

    let data = vec![
        "good".to_string(),
        "fail_me".to_string(),
        "also_good".to_string(),
    ];
    let dataset = InMemoryDataset::new(data).with_transform(FailingTransform);

    let config = DataLoaderConfig::builder()
        .batch_size(1)
        .shuffle(false) // Ensure deterministic order
        .build();

    let dataloader = DataLoader::new(dataset, config)?;
    let mut iter = dataloader.iter()?;

    // First batch should succeed
    let batch1 = iter.next().expect("Should have first batch");
    assert!(batch1.is_ok(), "First batch should succeed");

    // Second batch should fail
    let batch2 = iter.next().expect("Should have second batch");
    assert!(batch2.is_err(), "Second batch should fail");

    // Check the error chain - the transform error should be in the chain
    let error = batch2.unwrap_err();
    let mut found_transform_error = false;

    // Walk the error chain
    for e in error.chain() {
        let msg = e.to_string();
        if msg.contains("Transform failed") || msg.contains("fail_me") {
            found_transform_error = true;
            break;
        }
    }

    assert!(
        found_transform_error,
        "Error chain should contain transform failure message. Full error: {:?}",
        error
    );

    // Third batch should succeed
    let batch3 = iter.next().expect("Should have third batch");
    assert!(batch3.is_ok(), "Third batch should succeed");

    Ok(())
}
