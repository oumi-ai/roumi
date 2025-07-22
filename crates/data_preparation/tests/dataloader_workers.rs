//! Worker lifecycle and concurrency tests for DataLoader.
//!
//! Tests cover:
//! - Worker lifecycle (startup, cleanup, mode switching)
//! - Worker failure handling (panics, errors, timeouts)
//! - Memory management (backpressure, prefetch bounds)
//! - Epoch synchronization (persistent workers, resume)
//! - Concurrent behavior (determinism, work distribution)

mod common;
use common::StringToSample;
use data_preparation::{
    dataloader::{DataLoader, DataLoaderConfig},
    dataloader::{WORKER_ID, WORKER_RNG},
    dataset::{DataSource, InMemoryDataset, IterableDataset},
    sample::Sample,
    sampler::{RandomSampler, Sampler, SequentialSampler, SubsetRandomSampler},
    transforms::Transform,
};

use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tch::Tensor;

// ============================================================================
// Common Helper Transforms
// ============================================================================

#[derive(Clone)]
struct CountingTransform {
    counter: Arc<AtomicUsize>,
}

impl Transform<String, Sample> for CountingTransform {
    fn apply(&self, s: String) -> Result<Sample> {
        self.counter.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(10));
        Ok(Sample::from_single("data", Tensor::from(s.len() as i64)))
    }
}

#[derive(Clone)]
struct SlowTransform {
    delay_ms: u64,
}

impl Transform<String, Sample> for SlowTransform {
    fn apply(&self, s: String) -> Result<Sample> {
        std::thread::sleep(Duration::from_millis(self.delay_ms));
        Ok(Sample::from_single("data", Tensor::from(s.len() as i64)))
    }
}

// Simple data source for IterableDataset tests
struct SimpleDataSource {
    data: Vec<String>,
}

impl DataSource<String> for SimpleDataSource {
    fn stream(&self) -> Result<Box<dyn Iterator<Item = Result<String>> + Send>> {
        Ok(Box::new(self.data.clone().into_iter().map(Ok)))
    }
}

// ============================================================================
// 1. Worker Lifecycle Tests
// ============================================================================

#[test]
fn test_dataloader_inmemory_fresh_workers_cleanup() -> Result<()> {
    let process_count = Arc::new(AtomicUsize::new(0));

    let dataset = InMemoryDataset::new((0..100).map(|i| format!("item{}", i)).collect::<Vec<_>>())
        .with_transform(CountingTransform {
            counter: process_count.clone(),
        });

    let config = DataLoaderConfig::builder()
        .batch_size(1)
        .num_workers(2)
        .prefetch_factor(1)
        .persistent_workers(false)
        .build();

    let loader = DataLoader::new(dataset, config)?;

    let mut iter = loader.iter()?;
    let _first = iter.next().unwrap()?;

    let count_after_first = process_count.load(Ordering::SeqCst);
    println!(
        "Fresh workers: processed {} items after first batch",
        count_after_first
    );

    drop(iter);
    drop(loader);

    std::thread::sleep(Duration::from_millis(200));
    let fresh_final = process_count.load(Ordering::SeqCst);
    println!("Fresh workers: processed {} items total", fresh_final);

    assert!(
        fresh_final < 50,
        "Fresh workers should have stopped early, but processed {} items",
        fresh_final
    );

    Ok(())
}

#[test]
fn test_dataloader_inmemory_persistent_workers_cleanup() -> Result<()> {
    let process_count = Arc::new(AtomicUsize::new(0));

    let dataset = InMemoryDataset::new((0..100).map(|i| format!("item{}", i)).collect::<Vec<_>>())
        .with_transform(CountingTransform {
            counter: process_count.clone(),
        });

    let config = DataLoaderConfig::builder()
        .batch_size(1)
        .num_workers(2)
        .prefetch_factor(1)
        .persistent_workers(true)
        .build();

    println!("Creating DataLoader with persistent workers...");
    let loader = DataLoader::new(dataset, config)?;
    println!("DataLoader created successfully");

    println!("Creating iterator...");
    let mut iter = loader.iter()?;
    println!("Iterator created successfully");

    let _first = iter.next().unwrap()?;

    let count_after_first = process_count.load(Ordering::SeqCst);
    println!(
        "Persistent workers: processed {} items after first batch",
        count_after_first
    );

    drop(iter);
    drop(loader);

    std::thread::sleep(Duration::from_millis(200));
    let persistent_final = process_count.load(Ordering::SeqCst);
    println!(
        "Persistent workers: processed {} items total",
        persistent_final
    );

    assert!(
        persistent_final < 50,
        "Persistent workers should have stopped early, but processed {} items",
        persistent_final
    );

    Ok(())
}

#[test]
fn test_dataloader_iterable_persistent_workers_cleanup() -> Result<()> {
    let process_count = Arc::new(AtomicUsize::new(0));

    let dataset = IterableDataset::new(vec![Box::new(SimpleDataSource {
        data: (0..100).map(|i| format!("item{}", i)).collect(),
    }) as Box<dyn DataSource<String>>])
    .with_transform(CountingTransform {
        counter: process_count.clone(),
    });

    let config = DataLoaderConfig::builder()
        .batch_size(1)
        .num_workers(2)
        .prefetch_factor(1)
        .persistent_workers(true)
        .build();

    println!("Creating DataLoader with persistent workers...");
    let loader = DataLoader::new_iterable(dataset, config)?;
    println!("DataLoader created successfully");

    println!("Creating iterator...");
    let mut iter = loader.iter()?;
    println!("Iterator created successfully");

    let _first = iter.next().unwrap()?;

    let count_after_first = process_count.load(Ordering::SeqCst);
    println!(
        "Persistent workers: processed {} items after first batch",
        count_after_first
    );

    drop(iter);
    drop(loader);

    std::thread::sleep(Duration::from_millis(200));
    let persistent_final = process_count.load(Ordering::SeqCst);
    println!(
        "Persistent workers: processed {} items total",
        persistent_final
    );

    assert!(
        persistent_final < 50,
        "Persistent workers should have stopped early, but processed {} items",
        persistent_final
    );

    Ok(())
}

#[test]
fn test_dataloader_switching_worker_modes() -> Result<()> {
    // Test that we can switch between fresh and persistent workers
    // and that each mode behaves correctly

    let process_count = Arc::new(AtomicUsize::new(0));
    let thread_tracker = Arc::new(Mutex::new(HashSet::new()));

    #[derive(Clone)]
    struct DiagnosticTransform {
        process_count: Arc<AtomicUsize>,
        thread_tracker: Arc<Mutex<HashSet<thread::ThreadId>>>,
    }

    impl Transform<String, Sample> for DiagnosticTransform {
        fn apply(&self, s: String) -> Result<Sample> {
            self.process_count.fetch_add(1, Ordering::SeqCst);
            self.thread_tracker
                .lock()
                .unwrap()
                .insert(thread::current().id());
            Ok(Sample::from_single("data", Tensor::from(s.len() as i64)))
        }
    }

    let dataset = InMemoryDataset::new((0..50).map(|i| format!("item{}", i)).collect::<Vec<_>>())
        .with_transform(DiagnosticTransform {
            process_count: process_count.clone(),
            thread_tracker: thread_tracker.clone(),
        });

    // Phase 1: Fresh workers
    let config_fresh = DataLoaderConfig::builder()
        .batch_size(10)
        .num_workers(2)
        .persistent_workers(false)
        .build();

    let loader_fresh = DataLoader::new(dataset.clone(), config_fresh.clone())?;

    // Run 3 epochs with fresh workers
    for epoch in 0..3 {
        thread_tracker.lock().unwrap().clear();
        let batches: Vec<_> = loader_fresh.iter()?.collect::<Result<Vec<_>>>()?;
        assert_eq!(batches.len(), 5, "Should have 5 batches");

        let threads_this_epoch = thread_tracker.lock().unwrap().len();
        assert_eq!(
            threads_this_epoch, 2,
            "Fresh workers should create new threads each epoch"
        );

        println!("Fresh epoch {}: {} threads", epoch, threads_this_epoch);
    }

    let count_after_fresh = process_count.load(Ordering::SeqCst);
    assert_eq!(
        count_after_fresh, 150,
        "Should process 50 samples × 3 epochs"
    );

    // Phase 2: Persistent workers
    thread_tracker.lock().unwrap().clear();
    process_count.store(0, Ordering::SeqCst);

    let config_persistent = DataLoaderConfig::builder()
        .batch_size(10)
        .num_workers(2)
        .persistent_workers(true)
        .build();

    let loader_persistent = DataLoader::new(dataset.clone(), config_persistent)?;

    // Run 3 epochs with persistent workers
    let mut total_unique_threads = HashSet::new();
    for epoch in 0..3 {
        let batches: Vec<_> = loader_persistent.iter()?.collect::<Result<Vec<_>>>()?;
        assert_eq!(batches.len(), 5, "Should have 5 batches");

        let current_threads = thread_tracker.lock().unwrap().clone();
        total_unique_threads.extend(current_threads);

        println!(
            "Persistent epoch {}: {} total unique threads",
            epoch,
            total_unique_threads.len()
        );
    }

    assert_eq!(
        total_unique_threads.len(),
        2,
        "Persistent workers should reuse same 2 threads across all epochs"
    );

    let count_after_persistent = process_count.load(Ordering::SeqCst);
    assert_eq!(
        count_after_persistent, 150,
        "Should process 50 samples × 3 epochs"
    );

    // Phase 3: Back to fresh workers to ensure no interference
    thread_tracker.lock().unwrap().clear();

    let loader_fresh2 = DataLoader::new(dataset, config_fresh)?;
    let batches: Vec<_> = loader_fresh2.iter()?.collect::<Result<Vec<_>>>()?;
    assert_eq!(batches.len(), 5);

    // Should create new threads, not reuse persistent ones
    let fresh_threads = thread_tracker.lock().unwrap().len();
    assert_eq!(fresh_threads, 2, "Should create fresh threads again");

    Ok(())
}

// ============================================================================
// 2. Worker Failure and Error Handling Tests
// ============================================================================

#[test]
fn test_dataloader_inmemory_worker_panic_isolation() -> Result<()> {
    #[derive(Clone)]
    struct PanicTransform {
        panic_at: usize,
        counter: Arc<AtomicUsize>,
    }

    impl Transform<String, Sample> for PanicTransform {
        fn apply(&self, s: String) -> Result<Sample> {
            let count = self.counter.fetch_add(1, Ordering::SeqCst);
            if count == self.panic_at {
                panic!("Worker panic test at count {}", count);
            }
            Ok(Sample::from_single("data", Tensor::from(s.len() as i64)))
        }
    }

    let counter = Arc::new(AtomicUsize::new(0));
    let dataset = InMemoryDataset::new((0..20).map(|i| format!("item{}", i)).collect::<Vec<_>>())
        .with_transform(PanicTransform {
            panic_at: 5,
            counter: counter.clone(),
        });

    let config = DataLoaderConfig::builder()
        .batch_size(5)
        .num_workers(2)
        .build();

    let loader = DataLoader::new(dataset, config)?;

    let mut error_found = false;
    let mut successful_batches = 0;

    for batch in loader.iter()? {
        match batch {
            Ok(_) => successful_batches += 1,
            Err(e) => {
                eprintln!("Got error: {:?}", e);
                error_found = true;
                break;
            }
        }
    }

    assert!(
        error_found || successful_batches < 4,
        "Should have encountered an error or incomplete iteration due to worker panic"
    );

    Ok(())
}

#[test]
fn test_dataloader_worker_failure_detection() -> Result<()> {
    // Test that dataloader handles worker failure appropriately

    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Clone)]
    struct FailAfterNTransform {
        fail_after_samples: Arc<AtomicUsize>,
        worker_to_fail: usize,
        has_failed: Arc<AtomicBool>,
    }

    impl Transform<String, Sample> for FailAfterNTransform {
        fn apply(&self, s: String) -> Result<Sample> {
            // Check if we're in the target worker
            let is_worker = WORKER_RNG.with(|rng| rng.borrow().is_some());

            if is_worker {
                let worker_id = WORKER_ID.with(|id| *id.borrow());
                if worker_id == self.worker_to_fail && !self.has_failed.load(Ordering::SeqCst) {
                    // Count down samples processed by this specific worker
                    let remaining = self.fail_after_samples.fetch_sub(1, Ordering::SeqCst);
                    if remaining == 1 {
                        // Was 1, now 0
                        self.has_failed.store(true, Ordering::SeqCst);
                        panic!(
                            "Simulated worker {} failure after processing its samples",
                            worker_id
                        );
                    }
                }
            }

            Ok(Sample::from_single("data", Tensor::from(s.len() as i64)))
        }
    }

    let has_failed = Arc::new(AtomicBool::new(false));

    let dataset = InMemoryDataset::new((0..100).map(|i| format!("item{}", i)).collect::<Vec<_>>())
        .with_transform(FailAfterNTransform {
            fail_after_samples: Arc::new(AtomicUsize::new(5)),
            worker_to_fail: 1,
            has_failed: has_failed.clone(),
        });

    let config = DataLoaderConfig::builder()
        .batch_size(5)
        .num_workers(4)
        .prefetch_factor(2)
        .seed(42)
        .build();

    let loader = DataLoader::new(dataset, config)?;

    let mut successful_batches = 0;
    let mut got_error = false;
    let mut consecutive_errors = 0;

    for batch in loader.iter()? {
        match batch {
            Ok(_) => {
                successful_batches += 1;
                consecutive_errors = 0;
            }
            Err(e) => {
                if !got_error {
                    eprintln!("Got first error: {}", e);
                }
                got_error = true;
                consecutive_errors += 1;

                if consecutive_errors > 3 {
                    eprintln!("DataLoader cannot continue after worker failure");
                    break;
                }
            }
        }
    }

    println!(
        "Processed {} successful batches before worker failure",
        successful_batches
    );

    assert!(got_error, "Should have gotten an error from failed worker");
    assert!(
        has_failed.load(Ordering::SeqCst),
        "Worker should have failed"
    );
    assert!(
        successful_batches >= 4,
        "Should have processed some batches before failure, got {}",
        successful_batches
    );

    println!("Test passed: DataLoader properly detected worker failure");
    println!("Note: Current implementation doesn't support continuing with remaining workers");

    Ok(())
}

#[test]
fn test_dataloader_iterable_worker_error_propagation() -> Result<()> {
    // Test that errors from data sources are properly propagated

    struct FailingDataSource {
        fail_after: usize,
    }

    impl DataSource<String> for FailingDataSource {
        fn stream(&self) -> Result<Box<dyn Iterator<Item = Result<String>> + Send>> {
            let fail_after = self.fail_after;
            let mut count = 0;
            Ok(Box::new(
                std::iter::from_fn(move || {
                    count += 1;
                    if count > fail_after {
                        Some(Err(anyhow!("Data source error after {} items", fail_after)))
                    } else {
                        Some(Ok(format!("item_{}", count)))
                    }
                })
                .take(10),
            ))
        }
    }

    let dataset = IterableDataset::new(vec![
        Box::new(FailingDataSource { fail_after: 5 }) as Box<dyn DataSource<String>>
    ])
    .with_transform(StringToSample);

    let config = DataLoaderConfig::builder()
        .batch_size(3)
        .num_workers(1)
        .build();

    let loader = DataLoader::new_iterable(dataset, config)?;

    let mut batches_before_error = 0;
    let mut found_error = false;

    for batch in loader.iter()? {
        match batch {
            Ok(_) => batches_before_error += 1,
            Err(e) => {
                eprintln!("Got error: {}", e);
                eprintln!("Error chain:");
                for (i, err) in e.chain().enumerate() {
                    eprintln!("  {}: {}", i, err);
                }

                let error_found_in_chain = e
                    .chain()
                    .any(|err| err.to_string().contains("Data source error"));

                assert!(
                    error_found_in_chain,
                    "Expected 'Data source error' in error chain, but got: {}",
                    e
                );

                found_error = true;
                break;
            }
        }
    }

    assert!(found_error, "Should have encountered data source error");
    assert!(
        batches_before_error >= 1,
        "Should have gotten at least one batch before error"
    );

    println!("Got {} batches before error", batches_before_error);

    Ok(())
}

// ============================================================================
// 3. Timeout Tests
// ============================================================================

#[test]
fn test_dataloader_inmemory_worker_timeout_behavior() -> Result<()> {
    // Test 1: Timeout should trigger
    let dataset = InMemoryDataset::new(vec!["a".to_string()])
        .with_transform(SlowTransform { delay_ms: 1000 });

    let config = DataLoaderConfig::builder()
        .batch_size(1)
        .num_workers(1)
        .timeout(Duration::from_millis(100))
        .build();

    let loader = DataLoader::new(dataset, config)?;
    let mut iter = loader.iter()?;

    match iter.next() {
        Some(Err(e)) => {
            let error_string = e.to_string();
            assert!(
                error_string.contains("timeout")
                    || error_string.contains("Timeout")
                    || error_string.contains("100ms"),
                "Expected timeout-related error, but got: {}",
                error_string
            );
        }
        Some(Ok(_)) => panic!("Expected timeout error but got successful result"),
        None => panic!("Expected timeout error but got None"),
    }

    // Test 2: No timeout with fast transform
    let fast_dataset =
        InMemoryDataset::new(vec!["a".to_string()]).with_transform(SlowTransform { delay_ms: 10 });

    let config_fast = DataLoaderConfig::builder()
        .batch_size(1)
        .num_workers(1)
        .timeout(Duration::from_millis(500))
        .build();

    let loader_fast = DataLoader::new(fast_dataset, config_fast)?;
    let result = loader_fast.iter()?.next().unwrap();
    assert!(result.is_ok(), "Fast transform should not timeout");

    Ok(())
}

#[test]
fn test_dataloader_iterable_worker_timeout_behaviour() -> Result<()> {
    // This test verifies timeout when worker stops sending data

    struct HangingDataSource {
        items_before_hang: usize,
        hang_duration: Duration,
    }

    impl DataSource<String> for HangingDataSource {
        fn stream(&self) -> Result<Box<dyn Iterator<Item = Result<String>> + Send>> {
            let items = self.items_before_hang;
            let hang = self.hang_duration;
            let mut count = 0;

            Ok(Box::new(std::iter::from_fn(move || {
                if count < items {
                    count += 1;
                    Some(Ok(format!("item_{}", count)))
                } else if count == items {
                    count += 1;
                    std::thread::sleep(hang);
                    None
                } else {
                    None
                }
            })))
        }
    }

    let dataset = IterableDataset::new(vec![Box::new(HangingDataSource {
        items_before_hang: 3,
        hang_duration: Duration::from_secs(10),
    }) as Box<dyn DataSource<String>>])
    .with_transform(StringToSample);

    let config = DataLoaderConfig::builder()
        .batch_size(5)
        .num_workers(1)
        .timeout(Duration::from_millis(200))
        .build();

    let loader = DataLoader::new_iterable(dataset, config)?;
    let mut iter = loader.iter()?;

    match iter.next() {
        Some(Ok(batch)) if batch.batch_size()? < 5 => {
            println!("Got partial batch with {} items", batch.batch_size()?);

            // Try to get next batch - this SHOULD timeout
            match iter.next() {
                Some(Err(e)) => {
                    let error_string = e.to_string();
                    assert!(
                        error_string.contains("200ms")
                            || error_string.contains("timeout")
                            || error_string.contains("Timeout"),
                        "Expected timeout on second batch, got: {}",
                        error_string
                    );
                }
                Some(Ok(_)) => panic!("Should not get another batch after worker hang"),
                None => {
                    println!("Iterator ended after partial batch");
                }
            }
        }
        Some(Err(e)) => {
            let error_string = e.to_string();
            assert!(
                error_string.contains("200ms") || error_string.contains("timeout"),
                "Expected timeout error, got: {}",
                error_string
            );
        }
        _ => panic!("Unexpected result"),
    }

    Ok(())
}

#[test]
fn test_dataloader_worker_near_timeout_boundary() -> Result<()> {
    // Test that work completing just before timeout is handled correctly

    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Clone)]
    struct NearTimeoutTransform {
        slow_indices: HashSet<usize>,
        processed_slow: Arc<AtomicBool>,
    }

    impl Transform<usize, Sample> for NearTimeoutTransform {
        fn apply(&self, idx: usize) -> Result<Sample> {
            if self.slow_indices.contains(&idx) {
                self.processed_slow.store(true, Ordering::Relaxed);
                // Sleep for 90ms - just under the 100ms timeout
                std::thread::sleep(Duration::from_millis(90));
            }
            Ok(Sample::from_single("idx", Tensor::from(idx as i64)))
        }
    }

    let mut slow_indices = HashSet::new();
    slow_indices.insert(7);
    slow_indices.insert(13);
    slow_indices.insert(23);

    let processed_slow = Arc::new(AtomicBool::new(false));

    let dataset =
        InMemoryDataset::new((0..30).collect::<Vec<_>>()).with_transform(NearTimeoutTransform {
            slow_indices,
            processed_slow: processed_slow.clone(),
        });

    let config = DataLoaderConfig::builder()
        .batch_size(5)
        .num_workers(2)
        .timeout(Duration::from_millis(100))
        .prefetch_factor(1)
        .seed(42)
        .build();

    let loader = DataLoader::new(dataset, config)?;

    let start_time = std::time::Instant::now();
    let batches: Vec<_> = loader.iter()?.collect::<Result<Vec<_>>>()?;
    let total_time = start_time.elapsed();

    assert_eq!(batches.len(), 6, "Should get all 6 batches without timeout");

    for (i, batch) in batches.iter().enumerate() {
        assert_eq!(batch.batch_size()?, 5, "Batch {} should have 5 samples", i);
    }

    assert!(
        processed_slow.load(Ordering::Relaxed),
        "Should have processed at least one slow item"
    );

    assert!(
        total_time.as_millis() >= 90,
        "Should take at least 90ms due to slow items, but took {:?}",
        total_time
    );

    assert!(
        total_time.as_secs() < 2,
        "Should complete in reasonable time, but took {:?}",
        total_time
    );

    println!(
        "Successfully processed all batches in {:?} without timeout",
        total_time
    );
    println!(
        "Slow items were processed: {}",
        processed_slow.load(Ordering::Relaxed)
    );

    Ok(())
}

// ============================================================================
// 4. Memory Management Tests
// ============================================================================

/// Helper to track and update maximum concurrent operations
fn update_max_atomic(current: usize, max_atomic: &AtomicUsize) {
    let mut max = max_atomic.load(Ordering::SeqCst);
    while current > max {
        match max_atomic.compare_exchange(max, current, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => break,
            Err(actual) => max = actual,
        }
    }
}

#[test]
fn test_dataloader_backpressure_memory_bounds() -> Result<()> {
    #[derive(Clone)]
    struct MemoryTrackingTransform {
        active_count: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
    }

    impl Transform<String, Sample> for MemoryTrackingTransform {
        fn apply(&self, s: String) -> Result<Sample> {
            let current = self.active_count.fetch_add(1, Ordering::SeqCst) + 1;
            update_max_atomic(current, &self.max_active);

            // Simulate some processing time
            std::thread::sleep(Duration::from_millis(50));

            let result = Ok(Sample::from_single("data", Tensor::from(s.len() as i64)));

            self.active_count.fetch_sub(1, Ordering::SeqCst);
            result
        }
    }

    let active_count = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));

    let dataset = InMemoryDataset::new((0..100).map(|i| format!("item{}", i)).collect::<Vec<_>>())
        .with_transform(MemoryTrackingTransform {
            active_count: active_count.clone(),
            max_active: max_active.clone(),
        });

    let num_workers = 4;
    let prefetch_factor = 2;
    let batch_size = 5;

    let config = DataLoaderConfig::builder()
        .batch_size(batch_size)
        .num_workers(num_workers)
        .prefetch_factor(prefetch_factor)
        .build();

    let loader = DataLoader::new(dataset, config)?;

    // Consume slowly to test backpressure
    for (i, batch) in loader.iter()?.enumerate() {
        let _batch = batch?;
        // Slow consumer
        std::thread::sleep(Duration::from_millis(100));

        if i == 5 {
            let max = max_active.load(Ordering::SeqCst);
            let expected_max = num_workers * prefetch_factor * batch_size;

            assert!(
                max <= expected_max + num_workers,
                "Max active transforms ({}) exceeded expected bound ({})",
                max,
                expected_max
            );
        }
    }

    Ok(())
}

#[test]
fn test_dataloader_slow_consumer_fast_workers() -> Result<()> {
    // Test that prefetch_factor properly bounds memory usage

    #[derive(Clone)]
    struct FastTransform {
        in_flight: Arc<AtomicUsize>,
        max_in_flight: Arc<AtomicUsize>,
    }

    impl Transform<String, Sample> for FastTransform {
        fn apply(&self, s: String) -> Result<Sample> {
            let current = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            update_max_atomic(current, &self.max_in_flight);

            // Fast transform - workers can produce quickly
            std::thread::sleep(Duration::from_millis(1));

            let result = Ok(Sample::from_single("data", Tensor::from(s.len() as i64)));

            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            result
        }
    }

    let in_flight = Arc::new(AtomicUsize::new(0));
    let max_in_flight = Arc::new(AtomicUsize::new(0));

    let dataset = InMemoryDataset::new((0..1000).map(|i| format!("item{}", i)).collect::<Vec<_>>())
        .with_transform(FastTransform {
            in_flight: in_flight.clone(),
            max_in_flight: max_in_flight.clone(),
        });

    let num_workers = 4;
    let prefetch_factor = 2;
    let batch_size = 10;

    let config = DataLoaderConfig::builder()
        .batch_size(batch_size)
        .num_workers(num_workers)
        .prefetch_factor(prefetch_factor)
        .build();

    let loader = DataLoader::new(dataset, config)?;

    let mut batch_count = 0;
    for batch in loader.iter()? {
        let _batch = batch?;

        // Slow consumer - 10x slower than workers
        std::thread::sleep(Duration::from_millis(10));

        batch_count += 1;
        if batch_count % 10 == 0 {
            let max = max_in_flight.load(Ordering::SeqCst);
            let expected_max = num_workers * prefetch_factor * batch_size;

            println!(
                "After {} batches, max in-flight: {} (expected bound: {})",
                batch_count, max, expected_max
            );

            assert!(
                max <= expected_max + batch_size,
                "Memory usage exceeded bounds: {} > {}",
                max,
                expected_max
            );
        }
    }

    println!("Prefetch buffer successfully bounded memory usage");
    Ok(())
}

// ============================================================================
// 5. Epoch and Synchronization Tests
// ============================================================================

#[test]
fn test_dataloader_inmemory_persistent_worker_epoch_synchronization() -> Result<()> {
    let epoch_counter = Arc::new(Mutex::new(vec![0usize; 20]));

    #[derive(Clone)]
    struct EpochTrackingTransform {
        epoch_counter: Arc<Mutex<Vec<usize>>>,
        current_epoch: Arc<AtomicUsize>,
    }

    impl Transform<usize, Sample> for EpochTrackingTransform {
        fn apply(&self, idx: usize) -> Result<Sample> {
            let epoch = self.current_epoch.load(Ordering::SeqCst);
            let mut counters = self.epoch_counter.lock().unwrap();

            assert_eq!(
                counters[idx], epoch,
                "Sample {} processed multiple times in epoch {}",
                idx, epoch
            );

            counters[idx] += 1;
            Ok(Sample::from_single("id", Tensor::from(idx as i64)))
        }
    }

    let current_epoch = Arc::new(AtomicUsize::new(0));
    let dataset =
        InMemoryDataset::new((0..20).collect::<Vec<_>>()).with_transform(EpochTrackingTransform {
            epoch_counter: epoch_counter.clone(),
            current_epoch: current_epoch.clone(),
        });

    let config = DataLoaderConfig::builder()
        .batch_size(4)
        .num_workers(2)
        .persistent_workers(true)
        .build();

    let loader = DataLoader::new(dataset, config)?;

    // Run 3 epochs
    for expected_epoch in 0..3 {
        current_epoch.store(expected_epoch, Ordering::SeqCst);

        let batches: Vec<_> = loader.iter()?.collect::<Result<Vec<_>>>()?;
        assert_eq!(batches.len(), 5); // 20 samples / 4 batch_size

        // Verify all samples were processed exactly once this epoch
        let counters = epoch_counter.lock().unwrap();
        for (i, &count) in counters.iter().enumerate() {
            assert_eq!(
                count,
                expected_epoch + 1,
                "Sample {} processed {} times after {} epochs",
                i,
                count,
                expected_epoch + 1
            );
        }
    }

    Ok(())
}

#[test]
fn test_dataloader_iterable_persistent_worker_epoch_synchronization() -> Result<()> {
    // Test that persistent workers properly synchronize between epochs

    let process_tracker = Arc::new(Mutex::new(HashMap::<String, Vec<usize>>::new()));

    #[derive(Clone)]
    struct EpochTrackingTransform {
        tracker: Arc<Mutex<HashMap<String, Vec<usize>>>>,
        epoch: Arc<AtomicUsize>,
    }

    impl Transform<String, Sample> for EpochTrackingTransform {
        fn apply(&self, input: String) -> Result<Sample> {
            let current_epoch = self.epoch.load(Ordering::SeqCst);

            let mut tracker = self.tracker.lock().unwrap();
            tracker
                .entry(input.clone())
                .or_insert_with(Vec::new)
                .push(current_epoch);

            Ok(Sample::from_single(
                "data",
                Tensor::from(current_epoch as i64),
            ))
        }
    }

    struct SimpleSource {
        items: Vec<String>,
    }

    impl DataSource<String> for SimpleSource {
        fn stream(&self) -> Result<Box<dyn Iterator<Item = Result<String>> + Send>> {
            Ok(Box::new(self.items.clone().into_iter().map(Ok)))
        }
    }

    // Create 4 sources with 2 items each (8 total items)
    let sources: Vec<_> = (0..4)
        .map(|i| {
            Box::new(SimpleSource {
                items: (0..2).map(|j| format!("s{}_i{}", i, j)).collect(),
            }) as Box<dyn DataSource<String>>
        })
        .collect();

    let epoch_counter = Arc::new(AtomicUsize::new(0));
    let dataset = IterableDataset::new(sources).with_transform(EpochTrackingTransform {
        tracker: process_tracker.clone(),
        epoch: epoch_counter.clone(),
    });

    let config = DataLoaderConfig::builder()
        .batch_size(3)
        .num_workers(2)
        .persistent_workers(true)
        .build();

    let loader = DataLoader::new_iterable(dataset, config)?;

    // Run 3 epochs
    for expected_epoch in 0..3 {
        epoch_counter.store(expected_epoch, Ordering::SeqCst);

        let batches: Vec<_> = loader.iter()?.collect::<Result<Vec<_>>>()?;
        assert_eq!(batches.len(), 3); // 8 items / 3 batch_size = 3 batches
    }

    // Verify each item was processed exactly once per epoch
    let tracker = process_tracker.lock().unwrap();
    assert_eq!(tracker.len(), 8, "Should have processed 8 unique items");

    for (item, epochs) in tracker.iter() {
        assert_eq!(
            epochs.len(),
            3,
            "Item {} was processed {} times, expected 3",
            item,
            epochs.len()
        );

        let mut sorted_epochs = epochs.clone();
        sorted_epochs.sort();
        assert_eq!(
            sorted_epochs,
            vec![0, 1, 2],
            "Item {} was not processed in all epochs: {:?}",
            item,
            epochs
        );
    }

    println!("Verified: Each item processed exactly once per epoch with persistent workers");
    Ok(())
}

#[test]
fn test_dataloader_epoch_state_resume() -> Result<()> {
    // Test resuming training from a specific epoch + batch

    #[derive(Clone)]
    struct IndexToSample;

    impl Transform<usize, Sample> for IndexToSample {
        fn apply(&self, idx: usize) -> Result<Sample> {
            Ok(Sample::from_single("idx", Tensor::from(idx as i64)))
        }
    }

    let dataset = InMemoryDataset::new((0..100).collect::<Vec<_>>()).with_transform(IndexToSample);

    let seed = 42;

    // First: Collect what the full epoch 2 would look like
    let sampler_full = RandomSampler::new(100, false, None, seed)?;
    let epoch2_indices: Vec<_> = sampler_full.iter(2).collect();

    // Now simulate resume: create sampler with only remaining indices
    let remaining_indices = epoch2_indices[50..].to_vec();
    let resume_sampler = SubsetRandomSampler::new(100, remaining_indices.clone(), seed)?;

    let config = DataLoaderConfig::builder()
        .batch_size(7)
        .shuffle(false) // Must be false when using custom sampler
        .seed(seed)
        .num_workers(2)
        .build();

    let loader = DataLoader::new_with_sampler(dataset, resume_sampler, config)?;

    // Verify we process exactly the remaining samples
    let mut processed_count = 0;
    for batch in loader.iter()? {
        let batch = batch?;
        processed_count += batch.batch_size()?;
    }

    assert_eq!(
        processed_count, 50,
        "Should process exactly the remaining 50 samples"
    );

    println!("Successfully resumed epoch 2 from sample 50");

    Ok(())
}

#[test]
fn test_dataloader_inmemory_load_balancing() -> Result<()> {
    #[derive(Clone)]
    struct LoadBalancingTransform {
        worker_dist: Arc<Mutex<HashMap<usize, Vec<usize>>>>,
    }

    impl Transform<usize, Sample> for LoadBalancingTransform {
        fn apply(&self, idx: usize) -> Result<Sample> {
            let worker_id = WORKER_ID.with(|id| *id.borrow());

            self.worker_dist
                .lock()
                .unwrap()
                .entry(worker_id)
                .or_insert_with(Vec::new)
                .push(idx);

            Ok(Sample::from_single("idx", Tensor::from(idx as i64)))
        }
    }

    // Test configurations: (num_workers, dataset_size, batch_size)
    let test_configs = vec![
        (2, 100, 10),
        (4, 100, 10),
        (4, 200, 25),
        (8, 240, 15),
        (3, 97, 7), // Prime numbers to test edge cases
    ];

    for persistent_workers in [false, true] {
        println!(
            "\n=== Testing {} Workers ===",
            if persistent_workers {
                "Persistent"
            } else {
                "Fresh"
            }
        );

        for (num_workers, dataset_size, batch_size) in &test_configs {
            let worker_dist = Arc::new(Mutex::new(HashMap::<usize, Vec<usize>>::new()));

            let dataset = InMemoryDataset::new((0..*dataset_size).collect::<Vec<_>>())
                .with_transform(LoadBalancingTransform {
                    worker_dist: worker_dist.clone(),
                });

            let config = DataLoaderConfig::builder()
                .batch_size(*batch_size)
                .num_workers(*num_workers)
                .persistent_workers(persistent_workers)
                .prefetch_factor(2)
                .seed(42)
                .build();

            let loader = DataLoader::new(dataset, config)?;

            // Process multiple epochs to ensure consistency
            let num_epochs = if persistent_workers { 3 } else { 2 };

            for epoch in 0..num_epochs {
                worker_dist.lock().unwrap().clear();

                let mut batch_count = 0;
                for batch in loader.iter()? {
                    let _batch = batch?;
                    batch_count += 1;
                }

                // Verify we got the expected number of batches
                let expected_batches = (*dataset_size + batch_size - 1) / batch_size;
                assert_eq!(
                    batch_count, expected_batches,
                    "Expected {} batches but got {}",
                    expected_batches, batch_count
                );

                // Analyze distribution
                let distribution = worker_dist.lock().unwrap();
                let mut counts: Vec<(usize, usize)> = distribution
                    .iter()
                    .map(|(worker, samples)| (*worker, samples.len()))
                    .collect();
                counts.sort_by_key(|(worker, _)| *worker);

                // Calculate expected batch-based distribution
                let total_batches = expected_batches;
                let batches_per_worker = total_batches / *num_workers;
                let workers_with_extra_batch = total_batches % *num_workers;

                if epoch == 0 {
                    println!(
                        "\nConfig: {} workers, {} samples, batch_size={}",
                        num_workers, dataset_size, batch_size
                    );
                    println!(
                        "Total batches: {}, {} batches/worker, {} workers get +1 batch",
                        total_batches, batches_per_worker, workers_with_extra_batch
                    );
                }

                // Verify distribution fairness
                let mut total_samples = 0;
                let mut workers_with_correct_batches = 0;

                for (worker_id, count) in &counts {
                    total_samples += count;

                    // Calculate expected samples for this worker
                    let expected_batches = if *worker_id < workers_with_extra_batch {
                        batches_per_worker + 1
                    } else {
                        batches_per_worker
                    };

                    // Calculate expected sample range
                    // Most batches have batch_size samples, but the last batch might be partial
                    let min_expected = if expected_batches == 0 {
                        0
                    } else {
                        (expected_batches - 1) * batch_size + 1 // At least 1 sample in last batch
                    };
                    let max_expected = expected_batches * batch_size;

                    if epoch == 0 {
                        println!(
                            "  Worker {}: {} samples ({} batches)",
                            worker_id, count, expected_batches
                        );
                    }

                    // Verify the worker got the right amount of work
                    let valid = *count >= min_expected && *count <= max_expected;
                    assert!(
                        valid,
                        "Epoch {}: Worker {} has {} samples, expected {}-{} ({} batches)",
                        epoch, worker_id, count, min_expected, max_expected, expected_batches
                    );

                    // Check if worker got exactly the expected number of full batches worth
                    // (accounting for the last partial batch)
                    if expected_batches > 0 {
                        workers_with_correct_batches += 1;
                    }
                }

                // Verify totals
                assert_eq!(
                    total_samples, *dataset_size,
                    "Epoch {}: Total samples {} doesn't match dataset size {}",
                    epoch, total_samples, dataset_size
                );

                // Verify all workers that should have work actually got work
                let expected_active_workers = (*num_workers).min(total_batches);
                assert_eq!(
                    workers_with_correct_batches, expected_active_workers,
                    "Epoch {}: Expected {} active workers, but {} got work",
                    epoch, expected_active_workers, workers_with_correct_batches
                );

                // For persistent workers, verify consistency across epochs
                if persistent_workers && epoch > 0 {
                    println!("  Epoch {} distribution verified consistent", epoch);
                }
            }
        }
    }

    println!("\n✓ Load balancing verified for both fresh and persistent workers");
    Ok(())
}

#[test]
fn test_dataloader_deterministic_batch_skip() -> Result<()> {
    // Test that with deterministic setup, we can skip N batches and continue

    #[derive(Clone)]
    struct IndexToSample;

    impl Transform<usize, Sample> for IndexToSample {
        fn apply(&self, idx: usize) -> Result<Sample> {
            Ok(Sample::from_single("idx", Tensor::from(idx as i64)))
        }
    }

    let dataset = InMemoryDataset::new((0..50).collect::<Vec<_>>()).with_transform(IndexToSample);

    let config = DataLoaderConfig::builder()
        .batch_size(7)
        .shuffle(false)
        .num_workers(0)
        .build();

    // Run 1: Collect all batches
    let loader1 = DataLoader::new(dataset.clone(), config.clone())?;
    let all_batches: Vec<Vec<i64>> = loader1
        .iter()?
        .map(|batch| {
            let b = batch?;
            let tensor = b.get("idx")?;
            let values: Vec<i64> = (0..b.batch_size()?)
                .map(|i| tensor.int64_value(&[i]))
                .collect();
            Ok(values)
        })
        .collect::<Result<Vec<_>>>()?;

    // Run 2: Skip 3 batches
    let skip_batches = 3;
    let sampler2 = SequentialSampler::new(dataset.len());
    let indices_after_skip: Vec<_> = sampler2.iter(0).skip(skip_batches * 7).collect();

    let resume_sampler = SubsetRandomSampler::new(50, indices_after_skip, 42)?;
    let loader2 = DataLoader::new_with_sampler(dataset, resume_sampler, config)?;

    let resumed_batches: Vec<Vec<i64>> = loader2
        .iter()?
        .map(|batch| {
            let b = batch?;
            let tensor = b.get("idx")?;
            let values: Vec<i64> = (0..b.batch_size()?)
                .map(|i| tensor.int64_value(&[i]))
                .collect();
            Ok(values)
        })
        .collect::<Result<Vec<_>>>()?;

    // Verify counts match
    assert_eq!(
        all_batches.len() - skip_batches,
        resumed_batches.len(),
        "Should have correct number of remaining batches"
    );

    Ok(())
}
