//! src/dataloader/iterator/iterable.rs
//!
//! Iterator implementation for IterableDataset.
//!
//! This module contains the `iter()` method that creates iterators for streaming
//! datasets without random access.
//!
//! The `IteratorImpl` variant created depends on the DataLoader configuration:
//! - `num_workers = 0`: IteratorImpl::IterableSingle
//! - `num_workers > 0 && !persistent_workers`: IteratorImpl::IterableMulti
//! - `num_workers > 0 && persistent_workers`: IteratorImpl::IterablePersistent
//!
//! # Worker Distribution
//! When using multiple workers, the dataset's data sources are distributed using
//! round-robin assignment. This is handled by `IterableDataset::iter_sharded()`.
//!
//! NOTE: If `num_workers > num_sources`, some workers will be idle. For example,
//!       with 8 workers and 3 data sources:
//! - Workers 0, 1, 2 will each process one source
//! - Workers 3-7 will remain idle
//!
//! To avoid idle workers, ensure you have at least as many data sources as workers,
//! or use data formats that support concurrent reads within a single source. Any
//! advanced sharding strategies should be implemented at the Dataset or DataSource
//! level before reaching the DataLoader.

use crate::collator::Collator;
use crate::dataset::{Dataset, IterableDataset};
use anyhow::{anyhow, Context, Result};
use std::sync::atomic::Ordering;

use super::{DataLoaderIter, IteratorConfig, IteratorImpl};
use crate::dataloader::common::thread::{init_worker_rng, WORKER_ID};
use crate::dataloader::config::DataLoaderConfig;
use crate::dataloader::loader::{DataLoader, LoaderType};
use crate::dataloader::workers::pool::WorkerPool;

impl<Raw, C> DataLoader<IterableDataset<Raw>, C>
where
    Raw: Clone + Send + Sync + 'static,
    C: Collator,
{
    /// Creates an iterator over batches for IterableDataset.
    ///
    /// Note: IterableDataset does not support shuffling since they have no random access.
    /// Workers will process data sources in parallel using shard distribution.
    pub fn iter(&self) -> Result<DataLoaderIter<'_, IterableDataset<Raw>, C, Raw>> {
        // Warns if there are more workers than data sources
        if self.config.num_workers > 0 {
            let num_sources = self.dataset.num_sources();
            if num_sources < self.config.num_workers {
                eprintln!(
                    "Performance Warning: {} workers configured but only {} data source(s) available.\n   \
                     {} worker(s) will be idle. For better performance:\n   \
                     - Provide at least {} data sources (e.g., multiple files)\n   \
                     - Or reduce num_workers to {}\n   \
                     - Or use data formats that support concurrent reads (e.g., Parquet)",
                    self.config.num_workers,
                    num_sources,
                    self.config.num_workers - num_sources,
                    self.config.num_workers,
                    num_sources
                );
            }
        }

        // Get runtime seed from DataLoader
        let runtime_seed = self.runtime_seed;

        // Get current epoch
        let epoch = self.current_epoch.fetch_add(1, Ordering::SeqCst);

        let config = IteratorConfig {
            batch_size: self.config.batch_size.unwrap_or(1),
            drop_last: self.config.drop_last.unwrap_or(false),
            collator: &self.collator,
            timeout: self.config.timeout,
            prefetch_factor: self.config.prefetch_factor,
            runtime_seed,
            epoch,
        };

        match &self.loader_type {
            LoaderType::Iterable { worker_manager } => {
                let inner = if let Some(manager) = worker_manager {
                    if self.config.persistent_workers {
                        // Use persistent workers
                        if let Some(pool) = &manager.persistent_pool {
                            if let Some(control_channels) = &manager.control_channels {
                                // Note: We'll need to update the iterator to send epoch/seed info
                                // when epoch starts (this happens in the Iterator implementation)
                                IteratorImpl::IterablePersistent {
                                    output_rx: &pool.output_rx,
                                    control_channels,
                                    sample_buffer: Vec::new(),
                                    config,
                                    num_workers: self.config.num_workers,
                                    epoch_started: false,
                                    workers_done: 0,
                                    current_epoch: epoch,
                                    runtime_seed: self.runtime_seed,
                                }
                            } else {
                                // Control channels not available, fall back to fresh workers
                                create_fresh_iterable_workers(
                                    &self.dataset,
                                    &self.config,
                                    config,
                                    epoch,
                                    runtime_seed,
                                )?
                            }
                        } else {
                            // Persistent worker pool fail to be initialized, fallback to fresh workers
                            create_fresh_iterable_workers(
                                &self.dataset,
                                &self.config,
                                config,
                                epoch,
                                runtime_seed,
                            )?
                        }
                    } else {
                        // User sets `persistent_workers = false`, use fresh workers.
                        create_fresh_iterable_workers(
                            &self.dataset,
                            &self.config,
                            config,
                            epoch,
                            runtime_seed,
                        )?
                    }
                } else {
                    // Single-threaded iteration
                    IteratorImpl::IterableSingle {
                        dataset_iter: self.dataset.iter(),
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
                "Internal error: Invalid loader implementation for IterableDataset"
            )),
        }
    }
}

/// Creates fresh worker threads for streaming datasets.
///
/// Uses round-robin shard distribution to ensure no duplicate reads
/// and balanced workload across workers.
fn create_fresh_iterable_workers<'a, Raw, C>(
    dataset: &IterableDataset<Raw>,
    loader_config: &DataLoaderConfig,
    iter_config: IteratorConfig<'a, C>,
    epoch: usize,
    runtime_seed: Option<u64>,
) -> Result<IteratorImpl<'a, C, Raw>>
where
    Raw: Clone + Send + Sync + 'static,
    C: Collator,
{
    let buffer_size = loader_config.num_workers
        * loader_config.prefetch_factor
        * loader_config.batch_size.unwrap();

    let dataset = dataset.clone();
    let num_workers = loader_config.num_workers;
    let num_sources = dataset.num_sources();

    let worker_pool = WorkerPool::new(
        num_workers,
        buffer_size,
        move |_task_rx, output_tx, shutdown| {
            let worker_id = WORKER_ID.with(|id| *id.borrow());

            if let Some(base_seed) = runtime_seed {
                init_worker_rng(worker_id, epoch, base_seed);
            }

            // Process assigned shards
            for sample_result in dataset.iter_sharded(worker_id, num_workers) {
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }

                let sample_result_with_context = sample_result.with_context(|| {
                    format!(
                        "Worker {} failed to load sample from stream (processing sources {:?})",
                        worker_id,
                        (worker_id..dataset.num_sources())
                            .step_by(num_workers)
                            .collect::<Vec<_>>()
                    )
                });

                if output_tx.send(sample_result_with_context).is_err() {
                    // Channel closed, stop processing
                    break;
                }
            }
        },
    )
    .with_context(|| {
        format!(
            "Failed to create {} fresh workers for streaming dataset with {} sources. ",
            num_workers, num_sources,
        )
    })?;

    Ok(IteratorImpl::IterableMulti {
        worker_pool,
        sample_buffer: Vec::new(),
        config: iter_config,
    })
}
