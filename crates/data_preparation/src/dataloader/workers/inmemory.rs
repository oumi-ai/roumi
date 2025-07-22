//! src/dataloader/workers/inmemory.rs
//!
//! Worker implementation for InMemoryDataset.
//!
//! Workers fetch transformed data (`Sample`s) on demand using indices
//! and collate them into mini-batches.
//!
//! # Architecture:
//! - Workers share the dataset via `Arc` for zero-copy access.
//! - Tasks are distributed through a shared channel (load balancing)
//! - Supports both fresh workers per epoch and persistent workers
//!
//! # Task Types
//! - `Batch(Vec<usize>)`: Process samples at the given indices
//! - `EndEpoch`: Signal to persistent workers that epoch is complete

use crate::collator::Collator;
use crate::dataset::InMemoryDataset;
use crate::minibatch::MiniBatch;
use crate::sample::Sample;
use anyhow::{anyhow, Context, Result};
use crossbeam_channel::RecvTimeoutError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::pool::WorkerPool;
use crate::dataloader::common::thread::{init_worker_rng, WORKER_ID};
use crate::dataloader::config::DataLoaderConfig;

// Task types for in-memory worker communication.
#[derive(Debug)]
pub enum InMemoryWorkerTask {
    /// Process a batch of indices
    Batch(Vec<usize>),
    /// Signal end of epoch
    EndEpoch,
    /// Set epoch for RNG initialization
    SetEpoch { epoch: usize, base_seed: u64 },
}

/// Manages workers for in-memory datasets.
pub(crate) struct InMemoryWorkerManager {
    pub(crate) worker_pool: WorkerPool<InMemoryWorkerTask, Result<MiniBatch>>,
}

impl InMemoryWorkerManager {
    /// Creates a new manager with Arc-shared dataset access.
    ///
    /// Each worker receives batch indices and fetches samples on-demand.
    /// Returns error if persistent workers are requested but initialization fails.
    pub(crate) fn new<Raw, C>(
        num_workers: usize,
        dataset: Arc<InMemoryDataset<Raw>>,
        collator: C,
        config: &DataLoaderConfig,
    ) -> Result<Self>
    where
        Raw: Clone + Send + Sync + 'static,
        C: Collator + Clone + Send + Sync + 'static,
    {
        if config.persistent_workers && num_workers > 0 {
            let pool = Self::create_persistent_pool(dataset, collator, num_workers, config)?;
            Ok(Self { worker_pool: pool })
        } else {
            // Fresh worker
            let dummy_pool = WorkerPool {
                workers: vec![],
                task_tx: None,
                worker_task_txs: None,
                output_rx: crossbeam_channel::never(),
                shutdown: Arc::new(AtomicBool::new(false)),
            };

            Ok(Self {
                worker_pool: dummy_pool,
            })
        }
    }

    /// Creates a pool of persistent workers that survive across epochs.
    ///
    /// # Worker Design
    /// Each worker runs a simple event loop:
    /// 1. Receive InMemoryWorkerTask from shared channel
    /// 2. Process Batch tasks or handle EndEpoch signals
    /// 3. Send results back via output channel.
    pub(crate) fn create_persistent_pool<Raw, C>(
        dataset: Arc<InMemoryDataset<Raw>>,
        collator: C,
        num_workers: usize,
        config: &DataLoaderConfig,
    ) -> Result<WorkerPool<InMemoryWorkerTask, Result<MiniBatch>>>
    where
        Raw: Clone + Send + Sync + 'static,
        C: Collator + Clone + Send + Sync + 'static,
    {
        let buffer_size = config.prefetch_factor;

        WorkerPool::new_deterministic(
            num_workers,
            buffer_size,
            move |task_rx, output_tx, shutdown| {
                let worker_id = WORKER_ID.with(|id| *id.borrow());
                WORKER_ID.with(|id| *id.borrow_mut() = worker_id);

                loop {
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }

                    match task_rx.recv() {
                        Ok(InMemoryWorkerTask::SetEpoch { epoch, base_seed }) => {
                            init_worker_rng(worker_id, epoch, base_seed);
                        }
                        Ok(InMemoryWorkerTask::Batch(indices)) => {
                            let result = Self::process_batch_lazy(&dataset, &indices, &collator)
                                .with_context(|| format!("Persistent worker {} failed", worker_id));

                            if output_tx.send(result).is_err() {
                                break;
                            }
                        }
                        Ok(InMemoryWorkerTask::EndEpoch) => {
                            let _ = output_tx.send(Err(anyhow!("WORKER_EPOCH_DONE")));
                        }
                        Err(_) => {
                            break;
                        }
                    }
                }
            },
        )
    }

    /// Process a batch by fetching samples on-demand using O(1) index access.
    ///
    /// This avoids pre-caching all samples in each worker. Called by both
    /// fresh and persistent workers.
    pub(crate) fn process_batch_lazy<Raw, C>(
        dataset: &InMemoryDataset<Raw>,
        indices: &[usize],
        collator: &C,
    ) -> Result<MiniBatch>
    where
        Raw: Clone + Send + Sync + 'static,
        C: Collator,
    {
        let samples: Result<Vec<Sample>> = indices
            .iter()
            .map(|&index| {
                dataset.get_sample(index).with_context(|| {
                    format!(
                        "Failed to load sample at index {} (dataset size: {})",
                        index,
                        dataset.len()
                    )
                })
            })
            .collect();

        let samples = samples?;
        collator
            .collate(&samples)
            .with_context(|| format!("Failed to collate batch of {} samples", samples.len()))
    }

    /// Sends a batch of indices to the worker pool for processing.
    ///
    /// Workers will fetch samples at these indices and create a MiniBatch.
    /// An empty vec! signals EndEpoch for persistent workers
    pub(crate) fn send_task(&self, indices: Vec<usize>) -> Result<()> {
        let task = if indices.is_empty() {
            InMemoryWorkerTask::EndEpoch
        } else {
            InMemoryWorkerTask::Batch(indices)
        };

        self.worker_pool
            .task_tx
            .as_ref()
            .ok_or_else(|| anyhow!("Worker pool task channel is closed"))?
            .send(task)
            .map_err(|_| anyhow!("Failed to send task to workers"))
    }

    /// Sends a batch to specific worker (deterministic mode)
    pub(crate) fn send_task_to_worker(&self, worker_id: usize, indices: Vec<usize>) -> Result<()> {
        if let Some(ref worker_txs) = self.worker_pool.worker_task_txs {
            // Create task directly here
            let task = if indices.is_empty() {
                InMemoryWorkerTask::EndEpoch
            } else {
                InMemoryWorkerTask::Batch(indices)
            };

            // Deterministic mode - send to specific worker
            worker_txs[worker_id]
                .send(task)
                .map_err(|_| anyhow!("Failed to send task to worker {}", worker_id))
        } else {
            // Fallback to shared channel
            self.send_task(indices)
        }
    }

    /// Receives a processed MiniBatch from the worker pool.
    /// Blocks until a result is available or timeout occurs.
    pub(crate) fn receive_task_result(&self, timeout: Duration) -> Result<Result<MiniBatch>> {
        self.worker_pool
            .output_rx
            .recv_timeout(timeout)
            .map_err(|e| match e {
                RecvTimeoutError::Timeout => anyhow!(
                    "Worker timeout after {:?} - possible deadlock or slow data loading",
                    timeout
                ),
                RecvTimeoutError::Disconnected => {
                    anyhow!("Worker channel disconnected - workers may have crashed")
                }
            })
    }

    /// Send epoch information to all workers
    pub(crate) fn set_epoch(&self, epoch: usize, base_seed: u64, num_workers: usize) -> Result<()> {
        if let Some(ref worker_txs) = self.worker_pool.worker_task_txs {
            // Deterministic mode - send to each worker directly
            for tx in worker_txs.iter() {
                tx.send(InMemoryWorkerTask::SetEpoch { epoch, base_seed })
                    .map_err(|_| anyhow!("Failed to send epoch info to workers"))?;
            }
        } else if let Some(ref task_tx) = self.worker_pool.task_tx {
            // Shared channel mode - broadcast to all
            for _ in 0..num_workers {
                task_tx
                    .send(InMemoryWorkerTask::SetEpoch { epoch, base_seed })
                    .map_err(|_| anyhow!("Failed to send epoch info to workers"))?;
            }
        } else {
            return Err(anyhow!("Worker pool has no task channels"));
        }
        Ok(())
    }
}
