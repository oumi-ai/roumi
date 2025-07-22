//! src/dataloader/iterator/mod.rs
//!
//! Iterator implementations for DataLoader.
//!
//! This module provides the core iteration logic that bridges datasets, samplers,
//! and workers to produce batches of data. The main types are:
//!
//! - `DataLoaderIter`: The public iterator type returned by `DataLoader::iter()`
//! - `IteratorImpl`: Internal enum with specialized implementations for different
//!                  scenarios (i.e., see Iterator Variants below)
//! - `IteratorConfig`: Shared configuration passed to all iterator variants
//!
//! # Iterator Variants
//!
//! The iterator implementation adapts based on:
//! - Dataset type: InMemory (index-based) vs Iterable (streaming)
//! - Threading: Single-threaded vs multi-threaded with workers
//! - Worker lifecycle: Fresh workers per epoch vs persistent workers
//!
//! This results in 6 variants:
//! - `InMemorySingle`: Single-threaded iteration over indexed data
//! - `InMemoryMulti`: Multi-threaded with fresh workers each epoch
//! - `InMemoryPersistent`: Multi-threaded with workers that survive across epochs
//! - `IterableSingle`: Single-threaded streaming
//! - `IterableMulti`: Multi-threaded streaming with fresh workers
//! - `IterablePersistent`: Multi-threaded streaming with persistent workers
//!
//! NOTE: Each variant is optimized for its specific use case rather than forcing all
//!       patterns through a generic abstraction. This allows for better performance
//!       and clearer code at the cost of some duplication.

use crate::collator::Collator;
use crate::dataset::InMemoryDataset;
use crate::minibatch::MiniBatch;
use crate::sample::Sample;
use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

use super::workers::inmemory::InMemoryWorkerManager;
use super::workers::pool::WorkerPool;
use super::workers::WorkerControl;
use crate::dataloader::common::thread::init_worker_rng;

// Declare submodules
pub(crate) mod inmemory;
pub(crate) mod iterable;

// Constants
const WORKER_RECV_TIMEOUT_MS: u64 = 100;

/// Shared configuration for all iterator implementations.
#[derive(Clone)]
struct IteratorConfig<'a, C> {
    batch_size: usize,
    drop_last: bool,
    collator: &'a C,
    timeout: Duration,
    prefetch_factor: usize,
    runtime_seed: Option<u64>,
    epoch: usize,
}

/// Iterator over batches of data.
///
/// Created by calling `dataloader.iter()`.
pub struct DataLoaderIter<'a, D, C, Raw = ()> {
    _dataset: std::marker::PhantomData<D>,
    inner: IteratorImpl<'a, C, Raw>,
}

/// Internal iterator implementation variants.
enum IteratorImpl<'a, C, Raw> {
    /// Single-threaded in-memory iteration.
    /// - `dataset`: Reference to the dataset
    /// - `batch_indices`: Iterator over batch indices from sampler
    /// - `config`: Shared iterator configuration
    /// - `rng_initialized`: Tracks whether RNG has been initialized for this epoch
    InMemorySingle {
        dataset: &'a InMemoryDataset<Raw>,
        batch_indices: Box<dyn Iterator<Item = Vec<usize>> + Send + 'a>,
        config: IteratorConfig<'a, C>,
        rng_initialized: bool,
    },

    /// Multi-threaded in-memory with fresh workers per epoch.
    /// - `worker_manager`: Manages the worker pool for this epoch
    /// - `batch_indices`: Iterator over batch indices from sampler
    /// - `config`: Shared iterator configuration
    /// - `pending_tasks`: Number of batches sent but not yet received
    /// - `batch_index`: Current batch number for deterministic worker assignment
    /// - `num_workers`: Total number of workers for round-robin calculation
    InMemoryMulti {
        worker_manager: Arc<InMemoryWorkerManager>,
        batch_indices: Box<dyn Iterator<Item = Vec<usize>> + Send + 'a>,
        config: IteratorConfig<'a, C>,
        pending_tasks: usize,
        batch_index: usize,
        num_workers: usize,
    },

    /// Multi-threaded in-memory with persistent workers.
    ///
    /// State tracking:
    /// - `pending_tasks`: Number of batches sent but not yet received
    /// - `workers_done`: Number of workers that acknowledged epoch end
    /// - `sentinels_sent`: Whether EndEpoch signals have been sent
    /// - `batches_exhausted`: All batches for this epoch have been sent
    /// - `batch_index`: Current batch number for deterministic worker assignment
    InMemoryPersistent {
        worker_manager: Arc<InMemoryWorkerManager>,
        batch_indices: Box<dyn Iterator<Item = Vec<usize>> + Send + 'a>,
        config: IteratorConfig<'a, C>,
        num_workers: usize,
        epoch_started: bool,
        workers_done: usize,
        pending_tasks: usize,
        sentinels_sent: bool,
        batches_exhausted: bool,
        batch_index: usize,
    },

    /// Single-threaded streaming iteration.
    /// - `dataset_iter`: Iterator over samples from the dataset
    /// - `config`: Shared iterator configuration
    /// - `rng_initialized`: Tracks whether RNG has been initialized for this epoch
    IterableSingle {
        dataset_iter: Box<dyn Iterator<Item = Result<Sample>> + Send + 'a>,
        config: IteratorConfig<'a, C>,
        rng_initialized: bool,
    },

    /// Multi-threaded streaming iteration with fresh workers
    /// - `worker_pool`: Pool of workers processing the stream
    /// - `sample_buffer`: Buffer for accumulating samples into batches
    /// - `config`: Shared iterator configuration
    IterableMulti {
        worker_pool: WorkerPool<(), Result<Sample>>,
        sample_buffer: Vec<Sample>,
        config: IteratorConfig<'a, C>,
    },

    /// Multi-threaded streaming with persistent workers
    /// - `output_rx`: Channel for receiving samples from workers
    /// - `control_channels`: Channels for sending control messages to workers
    /// - `sample_buffer`: Buffer for accumulating samples into batches
    /// - `config`: Shared iterator configuration
    /// - `num_workers`: Total number of workers
    /// - `epoch_started`: Whether StartEpoch has been sent
    /// - `workers_done`: Count of workers that finished their epoch
    /// - `current_epoch`: Current epoch number
    /// - `runtime_seed`: Base seed for RNG coordination
    IterablePersistent {
        output_rx: &'a Receiver<Result<Sample>>,
        control_channels: &'a Vec<Sender<WorkerControl>>,
        sample_buffer: Vec<Sample>,
        config: IteratorConfig<'a, C>,
        num_workers: usize,
        epoch_started: bool,
        workers_done: usize,
        current_epoch: usize,
        runtime_seed: Option<u64>,
    },
}

impl<'a, D, C, Raw> Iterator for DataLoaderIter<'a, D, C, Raw>
where
    C: Collator,
    Raw: Clone + Send + Sync + 'static,
{
    type Item = Result<MiniBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            // Single-threaded in-memory: Direct `Sample` fetching using `get_sample(index)`
            IteratorImpl::InMemorySingle {
                dataset,
                batch_indices,
                config,
                rng_initialized,
            } => {
                // Initialize RNG for main thread if we have a seed
                if !*rng_initialized {
                    if let Some(seed) = config.runtime_seed {
                        init_worker_rng(0, config.epoch, seed);
                    }
                    *rng_initialized = true;
                }

                let indices = batch_indices.next()?;

                let samples_result: Result<Vec<Sample>> = indices
                    .iter()
                    .map(|&idx| {
                        dataset.get_sample(idx).with_context(|| {
                            format!("Failed to get sample {} in single-threaded mode", idx)
                        })
                    })
                    .collect();

                match samples_result {
                    Ok(samples) => {
                        if samples.is_empty() {
                            Some(Err(anyhow!(
                                "Batch is empty - all {} indices failed to load",
                                indices.len()
                            )))
                        } else {
                            Some(config.collator.collate(&samples).with_context(|| {
                                format!("Collation failed for {} samples", samples.len())
                            }))
                        }
                    }
                    Err(e) => Some(Err(e)),
                }
            }

            // Multi-threaded in-memory with fresh workers
            IteratorImpl::InMemoryMulti {
                worker_manager,
                batch_indices,
                config,
                pending_tasks,
                batch_index,
                num_workers,
            } => {
                // Keep the pipeline full up to `prefetch_factor`
                while *pending_tasks < config.prefetch_factor {
                    match batch_indices.next() {
                        Some(indices) => {
                            // Deterministic round-robin assignment
                            let assigned_worker = *batch_index % *num_workers;

                            if let Err(e) =
                                worker_manager.send_task_to_worker(assigned_worker, indices)
                            {
                                return Some(Err(e.context(format!(
                                    "Failed to send batch {} to worker {}",
                                    *batch_index, assigned_worker
                                ))));
                            }

                            *batch_index += 1;
                            *pending_tasks += 1;
                        }
                        None => break, // No more batches to send
                    }
                }

                // If we have pending tasks, receive one
                if *pending_tasks > 0 {
                    match worker_manager.receive_task_result(config.timeout) {
                        Ok(result) => {
                            *pending_tasks -= 1;
                            Some(result)
                        }
                        Err(e) => Some(Err(e.context(format!(
                            "Failed to receive batch from workers after {:?} \
                            (pending tasks: {}, possible deadlock or slow transform)",
                            config.timeout, *pending_tasks
                        )))),
                    }
                } else {
                    None // All batches consumed
                }
            }

            // Multi-thread in-memory with persistent workers
            IteratorImpl::InMemoryPersistent {
                worker_manager,
                batch_indices,
                config,
                num_workers,
                epoch_started,
                workers_done,
                pending_tasks,
                sentinels_sent,
                batches_exhausted,
                batch_index,
            } => {
                // Keep the pipeline full up to `prefetch_factor`
                while *pending_tasks < config.prefetch_factor && !*batches_exhausted {
                    match batch_indices.next() {
                        Some(indices) => {
                            // Send batch to workers via shared channel (load balanced)
                            let worker_id = *batch_index % *num_workers;
                            *batch_index += 1;

                            if let Err(e) = worker_manager.send_task_to_worker(worker_id, indices) {
                                return Some(Err(e));
                            }
                            *pending_tasks += 1;
                        }
                        None => {
                            // No more batches in this epoch
                            *batches_exhausted = true;
                            break;
                        }
                    }
                }

                // Process completed batches
                if *pending_tasks > 0 {
                    // Still have real work pending
                    match worker_manager.receive_task_result(config.timeout) {
                        Ok(Ok(batch)) => {
                            *pending_tasks -= 1;
                            Some(Ok(batch))
                        }
                        Ok(Err(e)) => {
                            if e.to_string().contains("WORKER_EPOCH_DONE") {
                                // Worker finished early
                                *workers_done += 1;
                                self.next()
                            } else {
                                // Propagate error from worker
                                *pending_tasks -= 1;
                                Some(Err(e))
                            }
                        }
                        Err(e) => Some(Err(e)),
                    }
                }
                // Send end-of-epoch signals to all workers
                else if !*sentinels_sent && *batches_exhausted {
                    for i in 0..*num_workers {
                        if let Err(e) = worker_manager.send_task_to_worker(i, vec![]) {
                            return Some(Err(e));
                        }
                    }
                    *sentinels_sent = true;
                    self.next()
                }
                // Wait for all workers to confirm epoch completion.
                else if *workers_done < *num_workers {
                    match worker_manager.receive_task_result(config.timeout) {
                        Ok(Err(e)) if e.to_string().contains("WORKER_EPOCH_DONE") => {
                            *workers_done += 1;
                            if *workers_done >= *num_workers {
                                // All workers done, epoch complete
                                *epoch_started = false;
                                None
                            } else {
                                self.next()
                            }
                        }
                        Ok(_other) => self.next(), // Unexpected result after sentinels
                        Err(e) => Some(Err(e)),
                    }
                } else {
                    // Epoch complete, reset for next iteration
                    *epoch_started = false;
                    *batches_exhausted = false;
                    None
                }
            }

            // Single-threaded streaming
            IteratorImpl::IterableSingle {
                dataset_iter,
                config,
                rng_initialized,
            } => {
                // Initialize RNG for main thread
                if !*rng_initialized {
                    if let Some(seed) = config.runtime_seed {
                        init_worker_rng(0, config.epoch, seed);
                    }
                    *rng_initialized = true;
                }

                let mut samples = Vec::with_capacity(config.batch_size);

                // Collect samples up to batch_size
                for _ in 0..config.batch_size {
                    match dataset_iter.next() {
                        Some(Ok(sample)) => samples.push(sample),
                        Some(Err(e)) => {
                            return Some(Err(
                                e.context("Failed to load sample from iterable dataset")
                            ))
                        }
                        None => break, // End of dataset
                    }
                }

                // Return batch if we have samples
                if samples.is_empty() || (config.drop_last && samples.len() < config.batch_size) {
                    None
                } else {
                    Some(config.collator.collate(&samples).with_context(|| {
                        format!(
                            "Failed to collate streaming batch of {} samples",
                            samples.len()
                        )
                    }))
                }
            }

            // Multi-threaded streaming with fresh workers
            IteratorImpl::IterableMulti {
                worker_pool,
                sample_buffer,
                config,
            } => {
                // Buffer samples until we have a full batch
                while sample_buffer.len() < config.batch_size {
                    match worker_pool
                        .output_rx
                        .recv_timeout(Duration::from_millis(WORKER_RECV_TIMEOUT_MS))
                    {
                        Ok(Ok(sample)) => sample_buffer.push(sample),
                        Ok(Err(e)) => {
                            // If we have some samples, return partial batch
                            if !sample_buffer.is_empty() && !config.drop_last {
                                break;
                            }
                            // Otherwise propagate error with context
                            return Some(Err(e.context(format!(
                                "Failed while buffering samples (had {} samples, needed {})",
                                sample_buffer.len(),
                                config.batch_size
                            ))));
                        }
                        Err(_) => {
                            // Timeout is normal when workers finish
                            break;
                        }
                    }
                }

                // Return batch if we have samples
                if sample_buffer.is_empty()
                    || (config.drop_last && sample_buffer.len() < config.batch_size)
                {
                    None
                } else {
                    let batch_end = sample_buffer.len().min(config.batch_size);
                    let batch: Vec<_> = sample_buffer.drain(0..batch_end).collect();

                    Some(config.collator.collate(&batch)
                        .with_context(|| format!(
                            "Failed to collate streaming batch of {} samples (expected batch_size: {})",
                            batch.len(),
                            config.batch_size
                        ))
                    )
                }
            }

            // Multi-threaded streaming with persistent workers
            IteratorImpl::IterablePersistent {
                output_rx,
                control_channels,
                sample_buffer,
                config,
                num_workers,
                epoch_started,
                workers_done,
                current_epoch,
                runtime_seed,
            } => {
                // Send StartEpoch signal to all workers at the beginning of each epoch
                if !*epoch_started {
                    *workers_done = 0; // Reset completion counter
                    for tx in control_channels.iter() {
                        if let Err(e) = tx.send(WorkerControl::StartEpoch {
                            epoch: *current_epoch,
                            base_seed: (*runtime_seed).unwrap_or(0),
                        }) {
                            return Some(Err(anyhow!("Failed to signal epoch start: {}", e)));
                        }
                    }
                    *epoch_started = true;
                }

                // Buffer samples until we have a full batch
                while sample_buffer.len() < config.batch_size {
                    match output_rx.recv_timeout(Duration::from_millis(WORKER_RECV_TIMEOUT_MS)) {
                        Ok(Ok(sample)) => {
                            sample_buffer.push(sample);
                        }
                        Ok(Err(e)) => {
                            // Check if this is the epoch completion marker
                            if e.to_string().contains("WORKER_EPOCH_DONE") {
                                *workers_done += 1;

                                if *workers_done >= *num_workers {
                                    // All workers completed - either return last partial batch or end epoch
                                    if sample_buffer.is_empty() {
                                        *epoch_started = false;
                                        *workers_done = 0;
                                        return None;
                                    }
                                    // Process remaining samples as final batch
                                    break;
                                }
                            } else {
                                // Return partial batch before error from worker
                                if !sample_buffer.is_empty() && !config.drop_last {
                                    break;
                                }
                                // Propagate the worker error as it is - it already has context from the worker
                                return Some(Err(e));
                            }
                        }
                        Err(_) => {
                            // Timeout - only valid if all workers are done
                            if *workers_done >= *num_workers {
                                if sample_buffer.is_empty() {
                                    *epoch_started = false;
                                    *workers_done = 0;
                                    return None;
                                }
                                // Process final partial batch
                                break;
                            }
                            // Otherwise ignore timeout and keep waiting for samples
                        }
                    }
                }

                // Decide whether to return a batch or end the epoch
                if sample_buffer.is_empty()
                    || (config.drop_last && sample_buffer.len() < config.batch_size)
                {
                    // Reset epoch state if all workers are done
                    if *workers_done >= *num_workers {
                        *epoch_started = false;
                        *workers_done = 0;
                    }
                    None
                } else {
                    // Create and return batch
                    let batch_end = sample_buffer.len().min(config.batch_size);
                    let batch: Vec<_> = sample_buffer.drain(0..batch_end).collect();

                    Some(config.collator.collate(&batch).with_context(|| {
                        format!("Failed to collate batch of {} samples", batch.len())
                    }))
                }
            }
        }
    }
}
