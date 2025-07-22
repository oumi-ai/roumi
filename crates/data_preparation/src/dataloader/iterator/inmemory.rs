//! src/dataloader/iterator/inmemory.rs
//!
//! Iterator implementation for InMemoryDataset.
//!
//! This module contains the `iter()` method that creates iterators for datasets
//! with random access capabilities.
//!
//! The `IteratorImpl` variant created depends on the DataLoader configuration:
//! - `num_workers = 0`: IteratorImpl::InMemorySingle
//! - `num_workers > 0 && !persistent_workers`: IteratorImpl::InMemoryMulti
//! - `num_workers > 0 && persistent_workers`: IteratorImpl::InMemoryPersistent

use crate::collator::Collator;
use crate::dataset::InMemoryDataset;
use crate::minibatch::MiniBatch;
use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::{DataLoaderIter, IteratorConfig, IteratorImpl};
use crate::dataloader::common::thread::{init_worker_rng, WORKER_ID};
use crate::dataloader::config::DataLoaderConfig;
use crate::dataloader::loader::{DataLoader, LoaderType};
use crate::dataloader::workers::inmemory::{InMemoryWorkerManager, InMemoryWorkerTask};
use crate::dataloader::workers::pool::WorkerPool;

impl<Raw, C> DataLoader<InMemoryDataset<Raw>, C>
where
    Raw: Clone + Send + Sync + 'static,
    C: Collator + Clone + Send + Sync + 'static,
{
    /// Creates an iterator over batches for the current epoch.
    ///
    /// If `shuffle` is true, increments the epoch counter for deterministic shuffling.
    pub fn iter(&self) -> Result<DataLoaderIter<'_, InMemoryDataset<Raw>, C, Raw>> {
        // Get runtime_seed from DataLoader
        let runtime_seed = self.runtime_seed;

        // Update epoch for shuffling and worker coordination
        let current_epoch_value = self.current_epoch.fetch_add(1, Ordering::SeqCst);

        let sampler_epoch = if self.config.shuffle.unwrap() {
            current_epoch_value // Varies for different shuffle per epoch
        } else {
            0 // Fixed for consistent order
        };

        let worker_epoch = current_epoch_value; // Always varies for transforms

        let config = IteratorConfig {
            batch_size: self.config.batch_size.unwrap_or(1),
            drop_last: self.config.drop_last.unwrap_or(false),
            collator: &self.collator,
            timeout: self.config.timeout,
            prefetch_factor: self.config.prefetch_factor,
            runtime_seed,
            epoch: worker_epoch,
        };

        match &self.loader_type {
            LoaderType::InMemory {
                batch_sampler,
                worker_manager,
            } => {
                let batch_indices = batch_sampler.iter(sampler_epoch);

                let inner = if let Some(manager) = worker_manager {
                    if self.config.persistent_workers {
                        if let Some(base_seed) = self.runtime_seed {
                            manager
                                .set_epoch(worker_epoch, base_seed, self.config.num_workers)
                                .context("Failed to send epoch info to persistent workers")?;
                        }
                        // Persistent workers
                        IteratorImpl::InMemoryPersistent {
                            worker_manager: manager.clone(), // Clone the Arc
                            batch_indices,
                            config,
                            num_workers: self.config.num_workers,
                            epoch_started: false,
                            workers_done: 0,
                            pending_tasks: 0,
                            sentinels_sent: false,
                            batches_exhausted: false,
                            batch_index: 0,
                        }
                    } else {
                        // Fresh workers per epoch
                        let dataset = Arc::new(self.dataset.clone());
                        let collator = self.collator.clone();
                        let loader_config = self.config.clone();

                        create_fresh_inmemory_workers(
                            dataset,
                            collator,
                            &loader_config,
                            config,
                            batch_indices,
                            worker_epoch,
                            runtime_seed,
                        )?
                    }
                } else {
                    // Single-threaded
                    IteratorImpl::InMemorySingle {
                        dataset: &self.dataset,
                        batch_indices,
                        config,
                        rng_initialized: false,
                    }
                };

                Ok(DataLoaderIter {
                    _dataset: std::marker::PhantomData,
                    inner,
                })
            }
            _ => Err(anyhow!(
                "Internal error: InMemoryDataset has incorrect loader type. \
                             This is a bug in the DataLoader implementation."
            )),
        }
    }
}

/// Creates fresh worker threads for a single epoch.
///
/// Workers are created per epoch to:
/// - Release memory between epochs
/// - Avoid synchronization overhead for simple workloads
/// - Provide isolation between epochs
///
/// Best for: Quick transforms, few epochs, memory-constrained environments
fn create_fresh_inmemory_workers<'a, Raw, C>(
    dataset: Arc<InMemoryDataset<Raw>>,
    collator: C,
    loader_config: &DataLoaderConfig,
    iter_config: IteratorConfig<'a, C>,
    batch_indices: Box<dyn Iterator<Item = Vec<usize>> + Send + 'a>,
    epoch: usize,
    runtime_seed: Option<u64>,
) -> Result<IteratorImpl<'a, C, Raw>>
where
    Raw: Clone + Send + Sync + 'static,
    C: Collator + Clone + Send + Sync + 'static,
{
    let buffer_size = loader_config.num_workers * loader_config.prefetch_factor;
    let worker_timeout = loader_config.worker_timeout;

    let worker_pool = WorkerPool::new_deterministic(
        loader_config.num_workers,
        buffer_size,
        move |task_rx: Receiver<InMemoryWorkerTask>,
              output_tx: Sender<Result<MiniBatch>>,
              shutdown| {
            // Changed to InMemoryWorkerTask
            let worker_id = WORKER_ID.with(|id| *id.borrow());

            if let Some(base_seed) = runtime_seed {
                init_worker_rng(worker_id, epoch, base_seed);
            }

            while !shutdown.load(Ordering::Relaxed) {
                match task_rx.recv_timeout(worker_timeout) {
                    Ok(InMemoryWorkerTask::SetEpoch { epoch, base_seed }) => {
                        // Fresh workers are created per epoch, so this message is unexpected.
                        // However, we handle it for completeness and reinitialize RNG if received.
                        init_worker_rng(worker_id, epoch, base_seed);
                    }
                    Ok(InMemoryWorkerTask::Batch(indices)) => {
                        // Handle InMemoryWorkerTask::Batch
                        let batch_size = indices.len();
                        let result = InMemoryWorkerManager::process_batch_lazy(
                            &dataset, &indices, &collator,
                        )
                        .with_context(|| {
                            format!(
                                "Worker {} failed to process batch with {} indices",
                                worker_id, batch_size
                            )
                        });
                        if output_tx.send(result).is_err() {
                            break;
                        }
                    }
                    Ok(InMemoryWorkerTask::EndEpoch) => {
                        // Fresh workers just exit on epoch end
                        break;
                    }
                    Err(RecvTimeoutError::Timeout) => continue,
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            }
        },
    )
    .context("Failed to create fresh worker pool for in-memory dataset")?;

    Ok(IteratorImpl::InMemoryMulti {
        worker_manager: Arc::new(InMemoryWorkerManager { worker_pool }),
        batch_indices,
        config: iter_config,
        pending_tasks: 0,
        batch_index: 0,
        num_workers: loader_config.num_workers,
    })
}
