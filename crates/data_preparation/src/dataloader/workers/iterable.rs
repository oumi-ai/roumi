//! src/dataloader/workers/iterable.rs
//!
//! Worker implementation for IterableDataset.
//!
//! Provides parallel streaming for datasets without random access. Each worker processes assigned
//! data sources (shards) independently using round-robin distribution.
//!
//! # Architecture
//! - Workers receive shard assignments via `iter_sharded(worker_id, num_workers)`
//! - Fresh workers: Created per epoch: simple lifecycle
//! - Persistent workers: Reused across epochs with control channel coordination
//!
//! # Control Protocol (Persistent Workers)
//! - `StartEpoch`: Begin processing assigned shards
//! - `WORKER_EPOCH_DONE`: Worker signals completion (sent as error message)

use crate::dataset::IterableDataset;
use crate::sample::Sample;
use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{bounded, RecvTimeoutError, Sender};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use super::pool::WorkerPool;
use super::WorkerControl;
use crate::dataloader::common::thread::{init_worker_rng, WORKER_ID};
use crate::dataloader::config::DataLoaderConfig;

/// Manages workers for streaming datasets.
///
/// Coordinates worker lifecycle and epoch synchronization. Unlike in-memory
/// workers that share a task queue, streaming workers independently process
/// their assigned shards.
pub(crate) struct IterableWorkerManager {
    // Pool of persistent workers that survive across epochs.
    // Only populated when `persistent_workers = true`.
    pub(crate) persistent_pool: Option<WorkerPool<WorkerControl, Result<Sample>>>,

    // Control channels for broadcasting epoch start signals to persistent workers.
    // One channel per worker to ensure deterministic broadcast delivery.
    // Only populated when `persistent_workers = true`.
    pub(crate) control_channels: Option<Vec<Sender<WorkerControl>>>,
}

impl IterableWorkerManager {
    /// Creates a new worker manager.
    ///
    /// If `persistent_workers` is true, spawns workers immediately that will be reused.
    /// If false, workers will be created fresh for each iteration in `DataLoader::iter()`.
    pub(crate) fn new<Raw>(
        num_workers: usize,
        dataset: IterableDataset<Raw>,
        config: &DataLoaderConfig,
    ) -> Result<Self>
    where
        Raw: Clone + Send + Sync + 'static,
    {
        let (persistent_pool, control_channels) = if config.persistent_workers && num_workers > 0 {
            let (pool, channels) = Self::create_persistent_pool(
                dataset,
                num_workers,
                config.prefetch_factor,
                config.batch_size.unwrap(),
            )
            .context("Failed to create persistent worker pool")?;
            (Some(pool), Some(channels))
        } else {
            (None, None)
        };

        Ok(Self {
            persistent_pool,
            control_channels,
        })
    }

    /// Creates persistent workers that survive across epochs.
    ///
    /// # Design
    /// Unlike fresh workers that use a shared task channel (load-balancing),
    /// persistent workers each get their own control channel to ensure
    /// every worker receives the StartEpoch signal (broadcast pattern).
    ///
    /// # Worker Lifecycle
    /// 1. Wait for StartEpoch signal
    /// 2. Process assigned shards via `iter_sharded()`
    /// 3. Send WORKER_EPOCH_DONE when complete
    /// 4. Return to step 1 for next epoch
    ///
    /// # Channel Architecture
    /// - Control channels: One per worker for broadcasting epoch signals
    /// - Output channel: Shared by all workers for sending samples
    fn create_persistent_pool<Raw>(
        dataset: IterableDataset<Raw>,
        num_workers: usize,
        prefetch_factor: usize,
        batch_size: usize,
    ) -> Result<(
        WorkerPool<WorkerControl, Result<Sample>>,
        Vec<Sender<WorkerControl>>,
    )>
    where
        Raw: Clone + Send + Sync + 'static,
    {
        // TODO: Adjust buffer size when num_workers > num_sources to avoid
        // over-allocation for idle workers. Currently we allocate assuming all workers
        // will be active, but the iterator already warns users about this scenario.
        let buffer_size = num_workers * prefetch_factor * batch_size;

        // Create dedicated control channels for broadcast pattern.
        let mut control_senders = Vec::with_capacity(num_workers);
        let mut control_receivers = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            let (tx, rx) = bounded::<WorkerControl>(1);
            control_senders.push(tx);
            control_receivers.push(rx);
        }

        // Shared output channel for all workers
        let (output_tx, output_rx) = bounded(buffer_size);
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut workers = Vec::with_capacity(num_workers);

        // Spawn persistent workers
        for (worker_id, control_rx) in control_receivers.into_iter().enumerate() {
            let dataset = dataset.clone();
            let output_tx = output_tx.clone();
            let shutdown_clone = shutdown.clone();

            let handle = thread::Builder::new()
                .name(format!("persistent-worker-{}", worker_id))
                .spawn(move || {
                    WORKER_ID.with(|id| *id.borrow_mut() = worker_id);

                    // Persistent workers run continuously until shutdown
                    loop {
                        if shutdown_clone.load(Ordering::Relaxed) {
                            break;
                        }

                        // Wait for epoch start signal
                        match control_rx.recv_timeout(Duration::from_millis(100)) {
                            Ok(WorkerControl::StartEpoch { epoch, base_seed }) => {
                                // Initialize RNG for this epoch
                                init_worker_rng(worker_id, epoch, base_seed);

                                // Process this worker's assigned shards
                                for sample_result in dataset.iter_sharded(worker_id, num_workers) {
                                    if shutdown_clone.load(Ordering::Relaxed) {
                                        break;
                                    }

                                    let sample_result_with_context =
                                        sample_result.with_context(|| {
                                            format!(
                                                "Worker {} failed to load sample from stream",
                                                worker_id
                                            )
                                        });

                                    if output_tx.send(sample_result_with_context).is_err() {
                                        break; // Main thread dropped receiver
                                    }
                                }

                                // Signal epoch completion
                                let _ = output_tx.send(Err(anyhow!("WORKER_EPOCH_DONE")));
                            }
                            Err(RecvTimeoutError::Timeout) => {
                                continue; // Check shutdown and wait again
                            }
                            Err(RecvTimeoutError::Disconnected) => {
                                break; // Control channel closed, exit
                            }
                        }
                    }
                })
                .with_context(|| format!("Failed to spawn persistent worker {}", worker_id))?;

            workers.push(handle);
        }

        let pool = WorkerPool {
            workers,
            task_tx: None, // Persistent workers use control_senders instead
            worker_task_txs: None,
            output_rx,
            shutdown,
        };

        Ok((pool, control_senders))
    }
}
