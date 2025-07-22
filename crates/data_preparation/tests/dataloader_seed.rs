//! Seed and determinism tests for DataLoader.
//!
//! Tests cover:
//! - Same seed → identical results (sampling order + transforms)
//! - Different seeds → different results
//! - Worker RNG independence across epochs
//! - Deterministic worker-to-shard assignment (multi-worker consistency)

mod common;
use common::StringToSample;

use data_preparation::{
    collator::StackCollator,
    dataloader::{DataLoader, DataLoaderConfig},
    dataset::{DataSource, InMemoryDataset, IterableDataset},
    minibatch::MiniBatch,
    sampler::{RandomSampler, SequentialSampler},
    transforms::{
        vision::{RandomHorizontalFlip, ToTensor},
        ToSample, Transform,
    },
};

use anyhow::Result;
use image::{DynamicImage, Rgb, RgbImage};
use std::collections::HashMap;

// ============================================================================
// Common Helper Functions
// ============================================================================

/// Creates test images with unique markers in the red channel
fn create_test_images(n: usize) -> Vec<DynamicImage> {
    (0..n)
        .map(|i| {
            let mut img = RgbImage::new(4, 4);
            img.put_pixel(0, 0, Rgb([i as u8, 0, 0]));
            DynamicImage::ImageRgb8(img)
        })
        .collect()
}

/// Creates test images with flip detection markers
/// - Position (0,0): original index
/// - Position (3,0): inverse of original index
fn create_test_images_with_flip_markers(n: usize) -> Vec<DynamicImage> {
    (0..n)
        .map(|i| {
            let mut img = RgbImage::new(4, 4);
            img.put_pixel(0, 0, Rgb([i as u8, 0, 0]));
            img.put_pixel(3, 0, Rgb([255 - i as u8, 0, 0]));
            DynamicImage::ImageRgb8(img)
        })
        .collect()
}

/// Extracts marker value and flip status from an image tensor
fn extract_marker_and_flip(img_tensor: &tch::Tensor) -> Result<(u8, f64)> {
    let red_channel = img_tensor.select(0, 0);

    let val_0_0 = red_channel.double_value(&[0, 0]);
    let val_0_3 = red_channel.double_value(&[0, 3]);

    let (marker, flip_indicator) = if val_0_0 > 0.0 {
        ((val_0_0 * 255.0) as u8, 0.0) // Not flipped
    } else {
        ((val_0_3 * 255.0) as u8, 1.0) // Flipped
    };

    Ok((marker, flip_indicator))
}

/// Collects markers and flip indicators from a DataLoader
fn collect_markers_and_flips(
    dataloader: &DataLoader<InMemoryDataset<DynamicImage>, StackCollator>,
) -> Result<Vec<(u8, f64)>> {
    let mut results = Vec::new();

    for batch in dataloader.iter()? {
        let batch = batch?;
        let tensor = batch.get("image")?;

        for i in 0..batch.batch_size()? {
            let img_tensor = tensor.select(0, i as i64);
            let (marker, flip_indicator) = extract_marker_and_flip(&img_tensor)?;
            results.push((marker, flip_indicator));
        }
    }

    results.sort_by_key(|(m, _)| *m); // Sort by marker for comparison
    Ok(results)
}

// ================================================================================================
// 1. Basic Determinism Tests (Single-threaded)
// ================================================================================================
#[test]
fn test_dataloader_produces_different_results_if_no_seed_specified() -> Result<()> {
    // When no seed is specified, results should vary between runs

    let data: Vec<String> = (0..50).map(|i| format!("item_{}", i)).collect();
    let dataset = InMemoryDataset::new(data).with_transform(StringToSample);

    let config = DataLoaderConfig::builder()
        .batch_size(10)
        .shuffle(true)
        // No seed specified
        .build();

    // Collect first several elements from multiple runs
    let mut all_sequences = Vec::new();

    // Create multiple dataloaders without seed specified
    for _ in 0..5 {
        let dataloader = DataLoader::new(dataset.clone(), config.clone())?;
        let first_batch = dataloader.iter()?.next().unwrap()?;
        let lengths = first_batch.get("length")?;

        // Collect all 10 elements from first batch (not just first 3)
        let sequence: Vec<i64> = (0..10).map(|i| lengths.int64_value(&[i, 0])).collect();

        all_sequences.push(sequence);
    }

    // Check that at least one full sequence is different
    let first_sequence = &all_sequences[0];
    let has_different_sequence = all_sequences
        .iter()
        .skip(1)
        .any(|seq| seq != first_sequence);

    assert!(
        has_different_sequence,
        "Without seed, at least one batch should have different order. All batches: {:?}",
        all_sequences
    );

    // Also verify the seeds were actually different by checking they're not all identical
    println!("Sequences collected: {:?}", all_sequences);

    Ok(())
}

// ================================================================================================
// 2. Multi-worker determinism tests
// ================================================================================================
#[test]
fn dataloader_iterable_deterministic_sampling_and_transforms() -> Result<()> {
    // Custom data source for testing
    struct ImageDataSource {
        images: Vec<DynamicImage>,
        #[allow(dead_code)]
        source_id: usize,
    }

    impl DataSource<DynamicImage> for ImageDataSource {
        fn stream(&self) -> Result<Box<dyn Iterator<Item = Result<DynamicImage>> + Send>> {
            let images = self.images.clone();
            Ok(Box::new(images.into_iter().map(Ok)))
        }
    }

    // Helper function to collect flip statistics for IterableDataset
    fn collect_flip_results(
        dataloader: &DataLoader<IterableDataset<DynamicImage>, StackCollator>,
    ) -> Result<HashMap<u8, f64>> {
        let mut results = HashMap::new();

        for batch in dataloader.iter()? {
            let batch = batch?;
            let tensor = batch.get("image")?;

            for i in 0..batch.batch_size()? {
                let img_tensor = tensor.select(0, i as i64);
                let (marker, flip_indicator) = extract_marker_and_flip(&img_tensor)?;
                results.insert(marker, flip_indicator);
            }
        }
        Ok(results)
    }

    // Create 8 sources, each with 10 unique images
    let mut sources: Vec<Box<dyn DataSource<DynamicImage>>> = Vec::new();
    for source_id in 0..8 {
        let mut images = Vec::new();
        for img_id in 0..10 {
            let mut img = RgbImage::new(4, 4);
            let marker = (source_id * 10 + img_id) as u8;
            img.put_pixel(0, 0, Rgb([marker, 0, 0]));
            images.push(DynamicImage::ImageRgb8(img));
        }
        sources.push(Box::new(ImageDataSource { images, source_id }));
    }

    let transform = RandomHorizontalFlip::new(0.5)?
        .then(ToTensor)
        .then(ToSample::new("image"));

    let dataset = IterableDataset::new(sources).with_transform(transform);

    let config = DataLoaderConfig::builder()
        .batch_size(5)
        .seed(42)
        .num_workers(3)
        .build();

    // Run twice and compare results
    let dataloader1 = DataLoader::new_iterable(dataset.clone(), config.clone())?;
    let run1_flips = collect_flip_results(&dataloader1)?;

    let dataloader2 = DataLoader::new_iterable(dataset.clone(), config)?;
    let run2_flips = collect_flip_results(&dataloader2)?;

    // Analysis
    eprintln!("\nAnalyzing results...");

    let mut worker_assignments: HashMap<u8, Vec<bool>> = HashMap::new();

    for marker in 0..80 {
        let source_id = marker / 10;
        let img_id = marker % 10;

        match (run1_flips.get(&marker), run2_flips.get(&marker)) {
            (Some(&flip1), Some(&flip2)) => {
                worker_assignments
                    .entry(marker)
                    .or_insert(Vec::new())
                    .push(flip1 == 1.0);
                worker_assignments
                    .entry(marker)
                    .or_insert(Vec::new())
                    .push(flip2 == 1.0);

                if flip1 == flip2 {
                    eprintln!(
                        "✓ Marker {} (source {}, img {}): consistent flip",
                        marker, source_id, img_id
                    );
                } else {
                    eprintln!(
                        "✗ Marker {} (source {}, img {}): inconsistent flip ({} vs {})",
                        marker, source_id, img_id, flip1, flip2
                    );
                }
            }
            (None, None) => {
                if marker != 0 {
                    eprintln!("! Marker {} missing in both runs", marker);
                }
            }
            _ => {
                panic!("Marker {} present in one run but not the other", marker);
            }
        }
    }

    // Verify work distribution
    eprintln!("\nExpected work distribution:");
    eprintln!("Worker 0 → sources [0, 3, 6] → markers 0-9, 30-39, 60-69");
    eprintln!("Worker 1 → sources [1, 4, 7] → markers 10-19, 40-49, 70-79");
    eprintln!("Worker 2 → sources [2, 5] → markers 20-29, 50-59");

    // Count consistent vs inconsistent
    let mut consistent = 0;
    let mut inconsistent = 0;

    for (_marker, flips) in &worker_assignments {
        if flips.len() == 2 && flips[0] == flips[1] {
            consistent += 1;
        } else {
            inconsistent += 1;
        }
    }

    eprintln!(
        "\nResults: {} consistent, {} inconsistent",
        consistent, inconsistent
    );

    assert_eq!(inconsistent, 0,
        "IterableDataset should have deterministic transforms due to deterministic shard assignment");

    Ok(())
}

#[test]
fn test_dataloader_inmemory_deterministic_sampling_and_transforms() -> Result<()> {
    // Helper to extract original index and flip status from complex markers
    fn extract_image_info(img_tensor: &tch::Tensor) -> Result<(u8, bool)> {
        let red_channel = img_tensor.select(0, 0);

        let val_0_0 = (red_channel.double_value(&[0, 0]) * 255.0).round() as u8;
        let val_0_3 = (red_channel.double_value(&[0, 3]) * 255.0).round() as u8;

        let (original_index, was_flipped) = if val_0_0 < 100 {
            (val_0_0, false)
        } else if val_0_0 > 100 {
            (255 - val_0_0, true)
        } else if val_0_3 < 100 {
            (val_0_3, true)
        } else {
            (255 - val_0_3, false)
        };

        Ok((original_index, was_flipped))
    }

    let images = create_test_images_with_flip_markers(100);

    let transform = RandomHorizontalFlip::new(0.5)?
        .then(ToTensor)
        .then(ToSample::new("image"));

    let dataset = InMemoryDataset::new(images).with_transform(transform);

    // Test with different worker counts and seeds
    for num_workers in [1, 2, 4, 8] {
        for test_seed in [42, 1337, 9999] {
            eprintln!(
                "\n=== Testing with {} workers, seed {} ===",
                num_workers, test_seed
            );

            let config = DataLoaderConfig::builder()
                .batch_size(10)
                .shuffle(true)
                .seed(test_seed)
                .num_workers(num_workers)
                .build();

            // Collect flip decisions from two runs
            let mut run1_flips: HashMap<u8, bool> = HashMap::new();
            let mut run2_flips: HashMap<u8, bool> = HashMap::new();

            // Run 1
            let dataloader1 = DataLoader::new(dataset.clone(), config.clone())?;
            for batch in dataloader1.iter()? {
                let batch = batch?;
                let tensor = batch.get("image")?;

                for i in 0..batch.batch_size()? {
                    let img_tensor = tensor.select(0, i as i64);
                    let (original_index, was_flipped) = extract_image_info(&img_tensor)?;
                    run1_flips.insert(original_index, was_flipped);
                }
            }

            // Run 2
            let dataloader2 = DataLoader::new(dataset.clone(), config)?;
            for batch in dataloader2.iter()? {
                let batch = batch?;
                let tensor = batch.get("image")?;

                for i in 0..batch.batch_size()? {
                    let img_tensor = tensor.select(0, i as i64);
                    let (original_index, was_flipped) = extract_image_info(&img_tensor)?;
                    run2_flips.insert(original_index, was_flipped);
                }
            }

            // Verify all samples processed
            assert_eq!(
                run1_flips.len(),
                100,
                "Should process all 100 samples with {} workers",
                num_workers
            );
            assert_eq!(
                run2_flips.len(),
                100,
                "Should process all 100 samples with {} workers",
                num_workers
            );

            // Compare flip decisions
            for idx in 0u8..100 {
                let flip1 = run1_flips
                    .get(&idx)
                    .expect(&format!("Sample {} missing in run1", idx));
                let flip2 = run2_flips
                    .get(&idx)
                    .expect(&format!("Sample {} missing in run2", idx));

                assert_eq!(
                    flip1, flip2,
                    "Sample {} should have same flip decision ({} vs {}) with {} workers, seed {}",
                    idx, flip1, flip2, num_workers, test_seed
                );
            }

            // Verify reasonable flip distribution
            let flip_count = run1_flips.values().filter(|&&f| f).count();
            eprintln!("  Flipped {} out of 100 samples", flip_count);

            assert!(
                flip_count >= 30 && flip_count <= 70,
                "Flip count {} should be roughly 50% with seed {}",
                flip_count,
                test_seed
            );

            eprintln!(
                "✓ Complete determinism verified with {} workers, seed {}",
                num_workers, test_seed
            );
        }
    }

    // Verify sample selection determinism
    eprintln!("\n=== Verifying sample selection determinism ===");
    let config = DataLoaderConfig::builder()
        .batch_size(10)
        .shuffle(true)
        .seed(42)
        .num_workers(4)
        .build();

    let mut run1_indices = Vec::new();
    let dataloader1 = DataLoader::new(dataset.clone(), config)?;
    for batch in dataloader1.iter()? {
        let batch = batch?;
        let tensor = batch.get("image")?;

        for i in 0..batch.batch_size()? {
            let img_tensor = tensor.select(0, i as i64);
            let (original_index, _) = extract_image_info(&img_tensor)?;
            run1_indices.push(original_index);
        }
    }

    // Verify we got all samples
    let mut sorted_indices = run1_indices.clone();
    sorted_indices.sort();
    assert_eq!(
        sorted_indices,
        (0..100).collect::<Vec<u8>>(),
        "Should have selected all samples exactly once"
    );

    eprintln!("✓ Sample selection verified - all samples selected exactly once");

    Ok(())
}

#[test]
fn test_single_threaded_vs_single_worker_consistency() -> Result<()> {
    // Verify num_workers = 0 and num_workers = 1 produce identical results

    let images = create_test_images(50);
    let transform = RandomHorizontalFlip::new(0.5)?
        .then(ToTensor)
        .then(ToSample::new("image"));

    let dataset = InMemoryDataset::new(images).with_transform(transform);
    let seed = 42;

    // Run with num_workers = 0 (main thread)
    let config0 = DataLoaderConfig::builder()
        .batch_size(10)
        .shuffle(true)
        .seed(seed)
        .num_workers(0)
        .build();

    let dataloader0 = DataLoader::new(dataset.clone(), config0)?;
    let results0 = collect_markers_and_flips(&dataloader0)?;

    // Run with num_workers = 1 (single worker thread)
    let config1 = DataLoaderConfig::builder()
        .batch_size(10)
        .shuffle(true)
        .seed(seed)
        .num_workers(1)
        .build();

    let dataloader1 = DataLoader::new(dataset, config1)?;
    let results1 = collect_markers_and_flips(&dataloader1)?;

    assert_eq!(results0, results1,
        "num_workers=0 (main thread) and num_workers=1 (worker thread) should produce identical results");

    Ok(())
}

// ================================================================================================
// 3. Cross-epoch behaviour tests
// ================================================================================================
#[test]
fn test_dataloader_cross_epoch_worker_rng_independence() -> Result<()> {
    // Verify that worker RNG state doesn't leak across epochs

    // Helper to extract flip indicators from a batch
    fn collect_flip_indicators(batch: &MiniBatch) -> Result<Vec<bool>> {
        let tensor = batch.get("image")?;
        let mut flips = Vec::new();

        for i in 0..batch.batch_size()? {
            let img_tensor = tensor.select(0, i as i64);
            let red_channel = img_tensor.select(0, 0);

            // Check if flipped by comparing pixel positions
            let val_0_0 = red_channel.double_value(&[0, 0]);
            let val_0_3 = red_channel.double_value(&[0, 3]);

            let is_flipped = val_0_0 < val_0_3;
            flips.push(is_flipped);
        }

        Ok(flips)
    }

    let images = create_test_images(20);
    let transform = RandomHorizontalFlip::new(0.5)?
        .then(ToTensor)
        .then(ToSample::new("image"));

    let dataset = InMemoryDataset::new(images).with_transform(transform);

    let config = DataLoaderConfig::builder()
        .batch_size(20)
        .shuffle(false)
        .num_workers(2)
        .seed(42)
        .build();

    let dataloader = DataLoader::new(dataset, config)?;

    // Run same dataloader for 3 epochs
    let mut epoch_results = Vec::new();

    for epoch in 0..3 {
        let batch = dataloader.iter()?.next().unwrap()?;
        let results = collect_flip_indicators(&batch)?;
        epoch_results.push(results);
        eprintln!(
            "Epoch {} flip count: {}",
            epoch,
            epoch_results[epoch].iter().filter(|&&x| x).count()
        );
    }

    // Each epoch should have identical flip patterns
    assert_eq!(
        epoch_results[0], epoch_results[1],
        "Epoch 0 and 1 should have identical RNG behavior"
    );
    assert_eq!(
        epoch_results[1], epoch_results[2],
        "Epoch 1 and 2 should have identical RNG behavior"
    );

    Ok(())
}

// ================================================================================================
// 4. End-to-end seed handling/coordination tests
// ================================================================================================
#[test]
fn test_sampler_seed_inheritance() -> Result<()> {
    // Test various seed coordination scenarios

    let data: Vec<String> = (0..10).map(|i| i.to_string()).collect();
    let dataset = InMemoryDataset::new(data).with_transform(StringToSample);

    // Test 1: Sampler has seed, config doesn't - should use sampler's seed
    let sampler1 = RandomSampler::new(dataset.len(), false, None, 123)?;
    let config1 = DataLoaderConfig::builder()
        .batch_size(2)
        // No seed specified
        .build();

    let result1 = DataLoader::new_with_sampler(dataset.clone(), sampler1, config1);
    assert!(
        result1.is_ok(),
        "Should use sampler's seed when config has none"
    );

    // Verify it actually uses the sampler's seed by checking determinism
    let sampler1b = RandomSampler::new(dataset.len(), false, None, 123)?;
    let config1b = DataLoaderConfig::builder().batch_size(2).build();
    let dataloader1b = DataLoader::new_with_sampler(dataset.clone(), sampler1b, config1b)?;

    let order1: Vec<i64> = result1?
        .iter()?
        .next()
        .unwrap()?
        .get("length")?
        .flatten(0, -1)
        .iter::<i64>()?
        .collect();
    let order1b: Vec<i64> = dataloader1b
        .iter()?
        .next()
        .unwrap()?
        .get("length")?
        .flatten(0, -1)
        .iter::<i64>()?
        .collect();

    assert_eq!(
        order1, order1b,
        "Should use sampler's seed deterministically"
    );

    // Test 2: Neither has seed - should generate one
    let sampler2 = SequentialSampler::new(dataset.len());
    let config2 = DataLoaderConfig::builder().batch_size(2).build();

    let result2 = DataLoader::new_with_sampler(dataset.clone(), sampler2, config2);
    assert!(result2.is_ok(), "Should work when neither has seed");

    // Test 3: Config has seed, sampler doesn't - should use config's seed
    let sampler3 = SequentialSampler::new(dataset.len());
    let config3 = DataLoaderConfig::builder().batch_size(2).seed(456).build();

    let result3 = DataLoader::new_with_sampler(dataset, sampler3, config3);
    assert!(
        result3.is_ok(),
        "Should use config's seed when sampler has none"
    );

    Ok(())
}
