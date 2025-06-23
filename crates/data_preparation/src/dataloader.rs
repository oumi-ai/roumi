use crate::collator::{Collator, StackCollator};
use crate::dataset::{Dataset, InMemoryDataset, IterableDataset};
use crate::minibatch::MiniBatch;
use crate::sample::Sample;
use crate::sampler::{BatchSampler, Sampler};
use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// This module implements the `DataLoader` that efficiently loads data,
/// with support for both in-memory and iterable datasets.
///
/// The `DataLoader` coordinates the following components:
/// - `Dataset`: Provide data access (in-memory or iterable)
/// - `Sampler`: Control iteration order (e.g., sequential, random, bucketed)
/// - `Collator`: Batch samples together (e.g., stacking, padding)
///
/// Internally, the `DataLoader` uses a worker management system that coordinates
/// parallel data loading for improved throughput:
/// - WorkerManager: Coordinates task distribution and result collection between the main thread and workers,
/// - WorkerPool: Manages the lifecycle of worker threads and their communication channels.
///
/// # Example usage
/// ```ignore
/// let dataset = InMemoryDataset::new(data).with_transform(transform);
/// let sampler = SequentialSampler::new(dataset.len());
/// let config = DataLoaderConfig::builder()
///     .batch_size(32)
///     .num_workers(4)
///     .build();
///
/// let loader = DataLoader::new(dataset, sampler, config)?;
/// for batch in loader.iter()? {
///     let batch = batch?;
///     // Process batch...
/// }
/// ```
///
/// # Performance considerations
///
/// ## Worker Configuration
/// - `num_workers = 0`: Single-threaded
/// - `num_workers > 0`: Parallel loading
///
/// ## Guidelines for setting `num_workers`:  
/// - For I/O bound workloads (e.g., loading from disk/network), use num_workers = 2-4x CPU cores
/// - For CPU-bound workloads (e.g., heavy transforms), use num_workers = all CPU cores
/// - For mixed workloads, start with 4 workers and tune based on profiling.
///
/// ## Memory usage
/// - Single-threaded: O(batch_size) - minimal memory footprint
/// - Multi-threaded: O(num_workers x prefetch_factor x batch_size)
///
/// ## Optimization notes:
/// - Enable `shuffle = true` for better model generalization
/// - Use `drop_last = true` for consistent batch sizes in training.

// ================================================================================================
// 1. Core Types (DataLoader, LoaderType)
// ================================================================================================

/// The main DataLoader struct that coordinates data loading
///
/// Supports two modes:
/// - In-memory: Random access with sampling strategies
/// - Iterable: Sequential access with shard distribution
///
/// # Thread safety:
/// - `DataLoader` itself is Send + Sync and can be shared across threads.
/// - Iterators are not Send and must be used on a single thread.
/// - Multiple iterators can be created from the same DataLoader safely.
///
/// # Type parameters:
/// - `D`: Dataset type (InMemoryDataset or IterableDataset)
/// - `C`: Collator type (defaults to StackCollator)
pub struct DataLoader<D, C = StackCollator> {
    dataset: D,
    collator: C,
    config: DataLoaderConfig,
    current_epoch: std::cell::Cell<usize>,
    loader_type: LoaderType,
}

/// Internal enum representing the data loading strategy based on dataset type.
///
/// This separation allows optimal implementations for different access patterns.
/// - InMemory: Uses samplers for flexible iteration order (random, sequential, etc.)
/// - Iterable: Direct iteration without sampling (no random access)
enum LoaderType {
    InMemory {
        batch_sampler: Box<dyn Sampler<Item = Vec<usize>> + Send + Sync>,
        worker_manager: Option<Arc<InMemoryWorkerManager>>,
    },
    Iterable {
        worker_manager: Option<Arc<IterableWorkerManager>>,
    },
}

// ================================================================================================
// 2. DataLoader Configuration
// ================================================================================================

/// Configuration for DataLoader
/// ```ignore
/// let config = DataLoaderConfig::builder()
///     .batch_size(32)
///     .num_workers(4)
///     .build();
/// ```
#[derive(Clone)]
pub struct DataLoaderConfig {
    /// Number of samples per batch
    pub batch_size: usize,
    /// Number of parallel workers (0 = single-threaded)
    pub num_workers: usize,
    /// Whether to drop the last incomplete batch
    pub drop_last: bool,
    /// Whether to shuffle data each epoch (in-memory only)
    pub shuffle: bool,
    /// Number of batches to prefetch per worker
    pub prefetch_factor: usize,
    /// Timeout for batch operations.
    /// Prevents indefinite blocking if workers stall.
    pub timeout: Duration,
    /// Timeout for worker receive operations.
    /// Controls how often workers check for shutdown.
    pub worker_timeout: Duration,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            num_workers: 0,
            drop_last: false,
            shuffle: false,
            prefetch_factor: 2,
            timeout: Duration::from_secs(30),
            worker_timeout: Duration::from_millis(100),
        }
    }
}

impl DataLoaderConfig {
    pub fn builder() -> DataLoaderConfigBuilder {
        DataLoaderConfigBuilder::default()
    }
}

/// Builder for DataLoaderConfig with method chaining
#[derive(Default)]
pub struct DataLoaderConfigBuilder {
    config: DataLoaderConfig,
}

impl DataLoaderConfigBuilder {
    /// Set the batch size (must be > 0)
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    /// Set the number of workers
    pub fn num_workers(mut self, workers: usize) -> Self {
        self.config.num_workers = workers;
        self
    }

    /// Set whether to drop_last
    pub fn drop_last(mut self, drop: bool) -> Self {
        self.config.drop_last = drop;
        self
    }

    /// Set whether to shuffle dataset every epoch
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.config.shuffle = shuffle;
        self
    }

    /// Set the prefetch factor for the dataset
    pub fn prefetch_factor(mut self, factor: usize) -> Self {
        self.config.prefetch_factor = factor;
        self
    }

    /// Set the timeout for batch operations.
    ///
    /// - Too low: May cancel batches during legitimate heavy processing
    /// - Too high: Delays detection of stuck workers.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Set the worker timeout for receiving operations
    /// Lower values = more responsive shutdown, higher CPU usage.
    pub fn worker_timeout(mut self, worker_timeout: Duration) -> Self {
        self.config.worker_timeout = worker_timeout;
        self
    }

    /// Build the final configuration.
    pub fn build(self) -> DataLoaderConfig {
        self.config
    }
}

// ================================================================================================
// 3a. DataLoader Constructors for InMemoryDataset
// ================================================================================================

impl<Raw> DataLoader<InMemoryDataset<Raw>, StackCollator>
where
    Raw: Clone + Send + Sync + 'static,
{
    /// Creates a new DataLoader for in-memory datasets with default StackCollator.
    ///
    /// # Errors:
    /// - Returns error if `batch_size` is 0
    /// - Returns error if `prefetch_factor` is 0 when using workers
    /// - Worker thread creation failure
    ///
    /// # Example
    /// ```ignore
    /// let dataloader = DataLoader::new(dataset, sampler, config)?;
    /// ```
    pub fn new(
        dataset: InMemoryDataset<Raw>,
        sampler: impl Sampler<Item = usize> + Send + Sync + 'static,
        config: DataLoaderConfig,
    ) -> Result<Self> {
        Self::new_with_collator(dataset, sampler, config, StackCollator)
    }
}

impl<Raw, C> DataLoader<InMemoryDataset<Raw>, C>
where
    Raw: Clone + Send + Sync + 'static,
    C: Collator + Clone + Send + Sync + 'static,
{
    /// Creates a new DataLoader with a custom collator.
    ///
    /// # Arguments
    /// - `dataset`: The dataset to load from.
    /// - `sampler`: Sampling strategy (e.g., SequentialSampler, RandomSampler)
    /// - `config`: DataLoader configuration
    /// - `collator`: Custom collator for batching samples
    ///
    /// # Errors
    /// - Returns error if `batch_size` is 0
    /// - Returns error if `prefetch_factor` is 0 when using workers
    /// - Worker thread creation failure
    ///
    /// # Thread safety
    /// The dataset is wrapped in Arc for zero-copy sharing across workers.
    pub fn new_with_collator(
        dataset: InMemoryDataset<Raw>,
        sampler: impl Sampler<Item = usize> + Send + Sync + 'static,
        config: DataLoaderConfig,
        collator: C,
    ) -> Result<Self> {
        if config.batch_size == 0 {
            return Err(anyhow!("Batch size must be greater than 0"));
        }

        if config.prefetch_factor == 0 && config.num_workers > 0 {
            return Err(anyhow!(
                "Prefetch factor must be > 0 when using {} workers",
                config.num_workers
            ));
        }

        let batch_sampler = BatchSampler::new(sampler, config.batch_size, config.drop_last)
            .context("Failed to wrap sampler with BaseSampler")?;

        let worker_manager = if config.num_workers > 0 {
            // Wrap dataset in Arc for sharing across workers
            let shared_dataset = Arc::new(dataset.clone());
            Some(Arc::new(
                InMemoryWorkerManager::new(
                    config.num_workers,
                    shared_dataset,
                    collator.clone(),
                    &config,
                )
                .context("Failed to initialize worker manager")?,
            ))
        } else {
            None
        };

        Ok(Self {
            dataset,
            collator,
            config,
            current_epoch: std::cell::Cell::new(0),
            loader_type: LoaderType::InMemory {
                batch_sampler: Box::new(batch_sampler),
                worker_manager,
            },
        })
    }
}

impl<Raw> DataLoader<InMemoryDataset<Raw>, StackCollator>
where
    Raw: Clone + Send + Sync + 'static,
{
    /// Creates a DataLoader with custom batch sampler for advanced batching strategies.
    /// Uses the default StackCollator.
    ///
    /// Use this when you need:
    /// - Length-based bucketing (BatchBucketSampler) to minimize padding
    /// - Class-balanced batching for imbalanced datasets
    /// - Custom batching logic while using standard collation
    ///
    /// Note: config.batch_size is ignored since the sampler controls batching.
    ///
    /// # Example
    /// ```ignore
    /// let bucket_sampler = BatchBucketSampler::new(sampler, 32, |i| lengths[i], 100)?;
    /// let loader = DataLoader::new_with_batch_sampler(dataset, bucket_sampler, config)?;
    /// ```
    pub fn new_with_batch_sampler(
        dataset: InMemoryDataset<Raw>,
        batch_sampler: impl Sampler<Item = Vec<usize>> + Send + Sync + 'static,
        config: DataLoaderConfig,
    ) -> Result<Self> {
        if config.batch_size != 1 {
            eprintln!(
                "Warning: DataLoader created with batch_sampler ignores config.batch_size={}. \
                The batch_sampler controls batching.",
                config.batch_size
            );
        }
        Self::new_with_batch_sampler_and_collator(dataset, batch_sampler, config, StackCollator)
    }
}

impl<Raw, C> DataLoader<InMemoryDataset<Raw>, C>
where
    Raw: Clone + Send + Sync + 'static,
    C: Collator + Clone + Send + Sync + 'static,
{
    /// Creates a DataLoader with custom batch sampler and custom collator.
    ///
    /// Use this when you need:
    /// - Custom batching strategy (e.g., length-based bucketing)
    /// - Custom collation (e.g., PaddingCollator for variable-length sequences)
    /// This is common in NLP where you may want to group similar-length sequences
    /// AND pad them efficiently.
    ///
    /// Note: config.batch_size is ignored since the batch sampler controls batching.
    ///
    /// # Example
    /// ```ignore
    /// let bucket_sampler = BatchBucketSampler::new(sampler, 32, |i| lengths[i], 100)?;
    /// let padding_collator = PaddingCollator::new().pad("input_ids", vec![(0, PaddingRule::MaxLength)], None);
    /// let loader = DataLoader::new_with_batch_sampler_and_collator(
    ///     dataset, bucket_sampler, config, padding_collator
    /// )?;
    /// ```
    pub fn new_with_batch_sampler_and_collator(
        dataset: InMemoryDataset<Raw>,
        batch_sampler: impl Sampler<Item = Vec<usize>> + Send + Sync + 'static,
        config: DataLoaderConfig,
        collator: C,
    ) -> Result<Self> {
        if config.prefetch_factor == 0 && config.num_workers > 0 {
            return Err(anyhow!(
                "Prefetch factor must be >0 when using {} workers",
                config.num_workers
            ));
        }

        if config.batch_size != 1 {
            eprintln!(
                "Warning: DataLoader created with batch_sampler ignores config.batch_size={}. \
                The batch_sampler controls batching.",
                config.batch_size
            );
        }

        let worker_manager = if config.num_workers > 0 {
            let shared_dataset = Arc::new(dataset.clone());
            Some(Arc::new(
                InMemoryWorkerManager::new(
                    config.num_workers,
                    shared_dataset,
                    collator.clone(),
                    &config,
                )
                .context("Failed to initialize worker pool")?,
            ))
        } else {
            None
        };

        Ok(Self {
            dataset,
            collator,
            config,
            current_epoch: std::cell::Cell::new(0),
            loader_type: LoaderType::InMemory {
                batch_sampler: Box::new(batch_sampler),
                worker_manager,
            },
        })
    }
}

// ================================================================================================
// 3b. DataLoadeer Constructor for IterableDataset
// ================================================================================================
impl<Raw> DataLoader<IterableDataset<Raw>, StackCollator>
where
    Raw: Clone + Send + Sync + 'static,
{
    /// Creates  a DataLoader for iterable datasets with the default StackCollator.
    ///
    /// Use this for large datasets that don't fit in memory or infinite data streams.
    ///
    /// # Note:
    /// IterableDataset do not support generic index-based shuffling or random access.
    pub fn new_iterable(dataset: IterableDataset<Raw>, config: DataLoaderConfig) -> Result<Self> {
        Self::new_iterable_with_collator(dataset, config, StackCollator)
    }
}

impl<Raw, C> DataLoader<IterableDataset<Raw>, C>
where
    Raw: Clone + Send + Sync + 'static,
    C: Collator + Clone + Send + Sync + 'static,
{
    /// Creates a DataLoader for iterable datasets with a custom collator.
    ///
    /// Allows custom batching logic (e.g., PaddingCollator for variable-length sequences).
    ///
    /// # Worker Distribution
    /// Uses round-robin shard distribution across data sources:
    /// - Worker 0 processes sources [0, N, 2N, ...] where N = num_workers
    /// - Worker 1 processes sources [1, N+1, 2N+1, ...]
    /// - etc.
    ///
    /// This ensures:
    /// 1. No duplicate reads across workers
    /// 2. Balanced workload when sources have similar sizes
    /// 3. Deterministic assignment for reproducibility
    ///
    /// Example with 3 workers and 7 sources:
    /// - Worker 0: sources [0, 3, 6]
    /// - Worker 1: sources [1, 4]
    /// - Worker 2: sources [2, 5]
    pub fn new_iterable_with_collator(
        dataset: IterableDataset<Raw>,
        config: DataLoaderConfig,
        collator: C,
    ) -> Result<Self> {
        let worker_manager = if config.num_workers > 0 {
            Some(Arc::new(
                IterableWorkerManager::new(config.num_workers, dataset.clone(), &config)
                    .context("Failed to create worker manager for iterable dataset")?,
            ))
        } else {
            None
        };

        Ok(Self {
            dataset,
            collator,
            config,
            current_epoch: std::cell::Cell::new(0),
            loader_type: LoaderType::Iterable { worker_manager },
        })
    }
}

// ================================================================================================
// 4. Iterator definition
// ================================================================================================

/// Shared configuration for all iterator implementations.
/// Extracted to avoid passing multiple parameters through iterator variants.
#[derive(Clone)]
struct IteratorConfig<'a, C> {
    batch_size: usize,
    drop_last: bool,
    collator: &'a C,
    timeout: Duration,
    prefetch_factor: usize,
}

/// Iterator over batches of data.
///
/// Created by calling `dataloader.iter()`.
///
/// # Lifetime
/// Borrows the DataLoader for the iteration duration.
/// Multiple iterators can exist simultaneously.
///
/// # Thread safety
/// Iterators are not Send - use on a single thread only.
pub struct DataLoaderIter<'a, D, C, Raw = ()> {
    _dataset: std::marker::PhantomData<D>,
    inner: IteratorImpl<'a, C, Raw>,
}

/// Internal iterator implementation variants for different dataset/threading combinations.
///
/// Separates concerns between:
/// - Dataset type: InMemory(index-based) vs Iterable(sequential)
/// - Threading: Single(simple path) vs Multi(worker pool)
enum IteratorImpl<'a, C, Raw> {
    /// Single-threaded in-memory iteration.
    /// Fetches samples on-demand without prefetching.
    InMemorySingle {
        dataset: &'a InMemoryDataset<Raw>,
        batch_indices: Box<dyn Iterator<Item = Vec<usize>> + Send + 'a>,
        config: IteratorConfig<'a, C>,
    },
    /// Multi-threaded in-memory iteration
    /// Workers prefetch batches for better throughput.
    InMemoryMulti {
        worker_manager: Arc<InMemoryWorkerManager>,
        batch_indices: Box<dyn Iterator<Item = Vec<usize>> + Send + 'a>,
        config: IteratorConfig<'a, C>,
        pending_tasks: usize,
    },
    /// Single-threaded streaming iteration.
    /// Simple sequential processing.
    IterableSingle {
        dataset_iter: Box<dyn Iterator<Item = Result<Sample>> + Send + 'a>,
        config: IteratorConfig<'a, C>,
    },
    /// Multi-threaded streaming iteration.
    /// Workers process shards in parallel.
    IterableMulti {
        worker_pool: WorkerPool<(), Result<Sample>>,
        sample_buffer: Vec<Sample>,
        config: IteratorConfig<'a, C>,
    },
}

/// In the iterator implementation:
impl<'a, D, C, Raw> Iterator for DataLoaderIter<'a, D, C, Raw>
where
    C: Collator,
    Raw: Clone + Send + Sync + 'static,
{
    type Item = Result<MiniBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            // Single-threaded path: fetch samples directly using O(1) `get_sample(index)`
            IteratorImpl::InMemorySingle {
                dataset,
                batch_indices,
                config,
            } => {
                let indices = batch_indices.next()?;

                // Use O(1) direct access instead of skip
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

            // Multi-threaded path: delegate to workers with prefetching
            IteratorImpl::InMemoryMulti {
                worker_manager,
                batch_indices,
                config,
                pending_tasks,
            } => {
                // Keep the pipeline full up to prefetch_factor
                while *pending_tasks < config.prefetch_factor {
                    match batch_indices.next() {
                        Some(indices) => {
                            let batch_size = indices.len();
                            if let Err(e) = worker_manager.send_task(indices) {
                                return Some(Err(e.context(format!(
                                    "Failed to send batch of {} indices to workers \
                                    (pending tasks: {}, workers may be overloaded)",
                                    batch_size, *pending_tasks
                                ))));
                            }
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

            // Sequential streaming without workers
            IteratorImpl::IterableSingle {
                dataset_iter,
                config,
            } => {
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

                // Check if we should return this batch
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

            // Parallel streaming with worker pool
            IteratorImpl::IterableMulti {
                worker_pool,
                sample_buffer,
                config,
            } => {
                // Buffer samples until we have a full batch
                while sample_buffer.len() < config.batch_size {
                    match worker_pool
                        .output_rx
                        .recv_timeout(Duration::from_millis(100))
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

                // Decide whether to return a batch
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
        }
    }
}

// ================================================================================================
// 4a. Iterator for InMemoryDataset
// ================================================================================================

impl<Raw, C> DataLoader<InMemoryDataset<Raw>, C>
where
    Raw: Clone + Send + Sync + 'static,
    C: Collator + Clone + Send + Sync + 'static,
{
    /// Creates an iterator over batches for the current epoch.
    ///
    /// If `shuffle` is true, increments the epoch counter for deterministic shuffling.
    pub fn iter(&self) -> Result<DataLoaderIter<'_, InMemoryDataset<Raw>, C, Raw>> {
        // Update epoch for shuffling
        let epoch = if self.config.shuffle {
            let current = self.current_epoch.get();
            self.current_epoch.set(current + 1);
            current
        } else {
            0
        };

        let config = IteratorConfig {
            batch_size: self.config.batch_size,
            drop_last: self.config.drop_last,
            collator: &self.collator,
            timeout: self.config.timeout,
            prefetch_factor: self.config.prefetch_factor,
        };

        match &self.loader_type {
            LoaderType::InMemory {
                batch_sampler,
                worker_manager,
            } => {
                let batch_indices = batch_sampler.iter(epoch);

                let inner = if let Some(manager) = worker_manager {
                    // Multi-worker case
                    IteratorImpl::InMemoryMulti {
                        worker_manager: manager.clone(),
                        batch_indices,
                        config,
                        pending_tasks: 0,
                    }
                } else {
                    // Single-threaded: no caching, fetch on demand
                    IteratorImpl::InMemorySingle {
                        dataset: &self.dataset,
                        batch_indices,
                        config,
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

// ================================================================================================
// 4b. Iterator for IterableDataset
// ================================================================================================
impl<Raw, C> DataLoader<IterableDataset<Raw>, C>
where
    Raw: Clone + Send + Sync + 'static,
    C: Collator,
{
    /// Creates an iterator over batches for iterating datasets.
    ///
    /// Note: Iterable datasets do not support shuffling since they have no random access.
    /// Workers will process data sources in parallel using shard distribution.
    pub fn iter(&self) -> Result<DataLoaderIter<'_, IterableDataset<Raw>, C>> {
        let config = IteratorConfig {
            batch_size: self.config.batch_size,
            drop_last: self.config.drop_last,
            collator: &self.collator,
            timeout: self.config.timeout,
            prefetch_factor: self.config.prefetch_factor,
        };

        match &self.loader_type {
            LoaderType::Iterable { worker_manager } => {
                let inner = if let Some(_manager) = worker_manager {
                    // Create fresh workers for this iteration
                    // TODO: Future optimization - use persistent_workers from manager
                    let buffer_size = self.config.num_workers
                        * self.config.prefetch_factor
                        * self.config.batch_size;
                    let dataset = self.dataset.clone();
                    let num_workers = self.config.num_workers;

                    // Uses shard distribution: worker 0 gets sources [0, N, 2N, ...], worker 1 gets [1, N+1, 2N +1, ...], etc.
                    // This ensures no duplicate reads and balanced workload across workers.
                    let worker_pool = WorkerPool::new(
                        num_workers,
                        buffer_size,
                        move |_task_rx, output_tx, shutdown| {
                            let worker_id = WORKER_ID.with(|id| *id.borrow());

                            // Process assigned shards
                            for sample_result in dataset.iter_sharded(worker_id, num_workers) {
                                if shutdown.load(Ordering::Relaxed) {
                                    break;
                                }

                                let sample_result_with_context = sample_result.with_context(|| {
                                    format!(
                                        "Worker {} failed to load sample from stream",
                                        worker_id
                                    )
                                });

                                if output_tx.send(sample_result_with_context).is_err() {
                                    // Channel closed, stop processing
                                    break;
                                }
                            }
                        },
                    )
                    .context("Failed to create worker pool for streaming")?;
                    IteratorImpl::IterableMulti {
                        worker_pool,
                        sample_buffer: Vec::new(),
                        config,
                    }
                } else {
                    // Single-threaded iteration
                    IteratorImpl::IterableSingle {
                        dataset_iter: self.dataset.iter(),
                        config,
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

// ================================================================================================
// 5. Worker Management
// ================================================================================================

// Thread-local storage for worker identification.
// Each worker thread gets a unique ID (0 to num_workers - 1) for:
// - Shard distribution in iterable datasets
// - Debugging and error messages
// - Future features like worker-specific initialization.
thread_local! {
    static WORKER_ID: std::cell::RefCell<usize> = std::cell::RefCell::new(0);
}

/// Thread pool for parallel data loading.
/// Manages worker lifecycle and communication channels.
///
/// # Design:
/// - Task channel: Main thread → Workers
/// - Output channel: Workers → Main thread
/// - Shutdown flag: Graceful termination
///
/// # Generic Parameters
/// - `Task`: Work items sent to workers
/// - `Output`: Results returned from workers
struct WorkerPool<Task, Output> {
    workers: Vec<thread::JoinHandle<()>>,
    task_tx: Sender<Task>,
    output_rx: Receiver<Output>,
    shutdown: Arc<AtomicBool>,
}

impl<Task, Output> WorkerPool<Task, Output>
where
    Task: Send + 'static,
    Output: Send + 'static,
{
    /// Creates a new worker pool with the specified number of workers.
    ///
    /// # Arguments
    /// - `num_workers`: Number of worker threads (must be > 0)
    /// - `buffer_size`: Channel buffer size (must be > 0)
    /// - `worker_fn`: Function each worker executes.
    ///
    /// # Errors
    /// - Invalid parameters (0 workers or buffer size)
    /// - Thread spawn failure
    fn new<F>(num_workers: usize, buffer_size: usize, worker_fn: F) -> Result<Self>
    where
        F: Fn(Receiver<Task>, Sender<Output>, Arc<AtomicBool>) + Send + Sync + 'static,
    {
        if num_workers == 0 {
            return Err(anyhow!(
                "Cannot create WorkerPool with 0 workers. \
                Either set num_workers > 0 or use single-threaded mode. "
            ));
        }

        if buffer_size == 0 {
            return Err(anyhow!(
                "Cannot create WorkerPool with buffer_size 0. \
                Buffer size must be > 0 to prevent deadlocks."
            ));
        }

        let (task_tx, task_receiver) = bounded(buffer_size);
        let (task_result_sender, output_rx) = bounded(buffer_size);
        let shutdown = Arc::new(AtomicBool::new(false));

        let worker_fn = Arc::new(worker_fn);
        let mut workers = Vec::with_capacity(num_workers);

        for worker_id in 0..num_workers {
            let task_receiver_clone = task_receiver.clone();
            let task_result_sender_clone = task_result_sender.clone();
            let shutdown_clone = shutdown.clone();
            let worker_fn_clone = worker_fn.clone();

            let handle = thread::Builder::new()
                .name(format!("dataloader-worker-{}", worker_id))
                .spawn(move || {
                    // Set thread-local worker ID
                    WORKER_ID.with(|id| *id.borrow_mut() = worker_id);

                    // Run worker function
                    worker_fn_clone(
                        task_receiver_clone,
                        task_result_sender_clone,
                        shutdown_clone,
                    );
                })
                .with_context(|| format!("Failed to spawn worker thread {}", worker_id))?;

            workers.push(handle);
        }
        Ok(Self {
            workers,
            task_tx,
            output_rx,
            shutdown,
        })
    }
}

/// Ensures clean shutdown of worker threads when the pool is dropped.
/// Sets the shutdown flag and waits for all workers to finish gracefully.
impl<Task, Output> Drop for WorkerPool<Task, Output> {
    fn drop(&mut self) {
        // Signal shutdown to all workers
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for workers to finish
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

// ================================================================================================
// 5a. Worker Management for InMemoryDataset
// ================================================================================================
/// Manages workers for in-memory datasets.
/// Workers share the dataset via Arc to avoid memory duplication.
///
/// # Design
/// - Zero-copy dataset sharing via Arc
/// - Lazy sample fetching (no pre-caching)
/// - Bounded channels prevent memory bloat
struct InMemoryWorkerManager {
    worker_pool: WorkerPool<Vec<usize>, Result<MiniBatch>>,
}

impl InMemoryWorkerManager {
    /// Creates a new manager with Arc-shared dataset access.
    /// Each worker receives batch indices and fetches samples on-demand.
    fn new<Raw, C>(
        num_workers: usize,
        dataset: Arc<InMemoryDataset<Raw>>,
        collator: C,
        config: &DataLoaderConfig,
    ) -> Result<Self>
    where
        Raw: Clone + Send + Sync + 'static,
        C: Collator + Clone + Send + Sync + 'static,
    {
        let buffer_size = num_workers * config.prefetch_factor;
        let worker_timeout = config.worker_timeout;

        let worker_pool = WorkerPool::new(
            num_workers,
            buffer_size,
            move |task_rx: Receiver<Vec<usize>>, output_tx: Sender<Result<MiniBatch>>, shutdown| {
                let worker_id = WORKER_ID.with(|id| *id.borrow());
                // Each worker has a shared reference to the dataset
                let dataset = dataset.clone(); // Arc clone - cheap!
                let collator = collator.clone();

                while !shutdown.load(Ordering::Relaxed) {
                    match task_rx.recv_timeout(worker_timeout) {
                        Ok(indices) => {
                            let batch_size = indices.len();
                            let result = Self::process_batch_lazy(&dataset, &indices, &collator)
                                .with_context(|| {
                                    format!(
                                        "Worker {} failed to process batch with {} indices",
                                        worker_id, batch_size
                                    )
                                });

                            if output_tx.send(result).is_err() {
                                // Output channel closed, exit
                                break;
                            }
                        }
                        Err(RecvTimeoutError::Timeout) => continue, // Normal timeout, keep waiting
                        Err(RecvTimeoutError::Disconnected) => break, // Channel closed, exit cleanly
                    }
                }
            },
        )
        .context("Failed to create worker pool for in-memory dataset")?;
        Ok(Self { worker_pool })
    }

    /// Process a batch by fetching samples on-demand using O(1) index access.
    /// This avoids pre-caching all samples in each worker.
    fn process_batch_lazy<Raw, C>(
        dataset: &InMemoryDataset<Raw>,
        indices: &[usize],
        collator: &C,
    ) -> Result<MiniBatch>
    where
        Raw: Clone + Send + Sync + 'static,
        C: Collator,
    {
        // Fetch and transform samples on-demand using O(1) access
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
    /// Workers will fetch samples at these indices and create a MiniBatch.
    fn send_task(&self, indices: Vec<usize>) -> Result<()> {
        let batch_size = indices.len();
        self.worker_pool.task_tx.send(indices).map_err(|_| {
            anyhow!(
                "Failed to send batch of {} indices to workers - worker pool may be shutting down",
                batch_size
            )
        })
    }

    /// Receives a processed MiniBatch from the worker pool.
    /// Blocks until a result is available or timeout occurs.
    fn receive_task_result(&self, timeout: Duration) -> Result<Result<MiniBatch>> {
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
}

// ================================================================================================
// 5b. Worker Management for IterableDataset
// ================================================================================================
/// Placeholder for future persistent worker implementation.
struct IterableWorkerManager {
    // Remove the worker_pool field - we will create workers fresh for each iteration for now.
    // TODO: implement persistent_workers that can be reused across iterations.
    // Future fields:
    // - persistent_pool: Option<WorkerPool<...>>
    // - reset_strategy: WorkerResetStrategy
}

impl IterableWorkerManager {
    // Placeholder constructor - actual worker creation happens in DataLoader::iter()
    fn new<Raw>(
        _num_workers: usize,
        _dataset: IterableDataset<Raw>,
        _config: &DataLoaderConfig,
    ) -> Result<Self>
    where
        Raw: Clone + Send + Sync + 'static,
    {
        Ok(Self {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{DataSource, InMemoryDataset, IterableDataset};
    use crate::sampler::SequentialSampler;
    use crate::transforms::Transform;
    use tch::Tensor;

    struct StringToSample;
    impl Transform<String, Sample> for StringToSample {
        fn apply(&self, input: String) -> Result<Sample> {
            let length = input.len() as i64;
            Ok(Sample::from_single("length", Tensor::from_slice(&[length])))
        }
    }

    struct TestDataSource {
        data: Vec<String>,
    }

    impl DataSource<String> for TestDataSource {
        fn stream(&self) -> Result<Box<dyn Iterator<Item = Result<String>> + Send>> {
            Ok(Box::new(self.data.clone().into_iter().map(Ok)))
        }
    }

    #[test]
    fn test_dataloader_inmemory_basic() -> Result<()> {
        let data = vec!["hello".to_string(), "world".to_string(), "rust".to_string()];
        let dataset = InMemoryDataset::new(data).with_transform(StringToSample);
        let sampler = SequentialSampler::new(dataset.len());

        let config = DataLoaderConfig::builder()
            .batch_size(2)
            .num_workers(0)
            .drop_last(false)
            .shuffle(false)
            .build();

        let dataloader = DataLoader::new(dataset, sampler, config)?;

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
        let sampler = SequentialSampler::new(dataset.len());

        let config = DataLoaderConfig::builder()
            .batch_size(2)
            .drop_last(true)
            .build();

        let dataloader = DataLoader::new(dataset, sampler, config)?;

        let batches: Vec<_> = dataloader.iter()?.collect::<Result<Vec<_>>>()?;
        assert_eq!(batches.len(), 1); // Only complete batches
        assert_eq!(batches[0].batch_size()?, 2);

        Ok(())
    }

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
    fn test_dataloader_iterable_multi_worker() -> Result<()> {
        let source = TestDataSource {
            data: vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
        };
        let dataset = IterableDataset::new(vec![Box::new(source) as Box<dyn DataSource<String>>])
            .with_transform(StringToSample);

        let config = DataLoaderConfig::builder()
            .batch_size(2)
            .num_workers(2)
            .build();

        let dataloader = DataLoader::new_iterable(dataset, config)?;
        let batches: Vec<_> = dataloader.iter()?.collect::<Result<Vec<_>>>()?;

        assert!(batches.len() >= 2);
        Ok(())
    }

    #[test]
    fn test_dataloader_iterable_with_collator() -> Result<()> {
        let source = TestDataSource {
            data: vec!["test1".to_string(), "test2".to_string()],
        };
        let dataset = IterableDataset::new(vec![Box::new(source) as Box<dyn DataSource<String>>])
            .with_transform(StringToSample);

        let config = DataLoaderConfig::builder().batch_size(1).build();

        let dataloader = DataLoader::new_iterable_with_collator(dataset, config, StackCollator)?;

        let batches: Vec<_> = dataloader.iter()?.collect::<Result<Vec<_>>>()?;
        assert_eq!(batches.len(), 2);

        Ok(())
    }

    #[test]
    fn test_dataloader_epoch_determinism() -> Result<()> {
        let data = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let dataset = InMemoryDataset::new(data).with_transform(StringToSample);
        let sampler = SequentialSampler::new(dataset.len());

        let config = DataLoaderConfig::builder()
            .batch_size(2)
            .shuffle(true)
            .build();

        let dataloader = DataLoader::new(dataset, sampler, config)?;

        let epoch0_batch1: Vec<_> = dataloader.iter()?.collect::<Result<Vec<_>>>()?;
        let epoch0_batch2: Vec<_> = dataloader.iter()?.collect::<Result<Vec<_>>>()?;

        assert_eq!(epoch0_batch1.len(), epoch0_batch2.len());
        Ok(())
    }

    #[test]
    fn test_dataloader_single_vs_multi_worker() -> Result<()> {
        let data = vec![
            "hello".to_string(),
            "world".to_string(),
            "rust".to_string(),
            "test".to_string(),
        ];
        let dataset = InMemoryDataset::new(data.clone()).with_transform(StringToSample);
        let sampler = SequentialSampler::new(dataset.len());

        // Single-threaded
        let config_single = DataLoaderConfig::builder()
            .batch_size(2)
            .num_workers(0)
            .build();
        let dataloader_single = DataLoader::new(dataset.clone(), sampler.clone(), config_single)?;
        let batches_single: Vec<_> = dataloader_single.iter()?.collect::<Result<Vec<_>>>()?;

        // Multi-threaded
        let dataset2 = InMemoryDataset::new(data).with_transform(StringToSample);
        let sampler2 = SequentialSampler::new(dataset2.len());
        let config_multi = DataLoaderConfig::builder()
            .batch_size(2)
            .num_workers(2)
            .build();
        let dataloader_multi = DataLoader::new(dataset2, sampler2, config_multi)?;
        let batches_multi: Vec<_> = dataloader_multi.iter()?.collect::<Result<Vec<_>>>()?;

        assert_eq!(batches_single.len(), batches_multi.len());
        assert_eq!(batches_single.len(), 2);

        for (single_batch, multi_batch) in batches_single.iter().zip(batches_multi.iter()) {
            assert_eq!(single_batch.batch_size()?, multi_batch.batch_size()?);
        }

        Ok(())
    }

    #[test]
    fn test_builder_pattern() -> Result<()> {
        let data = vec!["test".to_string()];
        let dataset = InMemoryDataset::new(data).with_transform(StringToSample);
        let sampler = SequentialSampler::new(dataset.len());

        let config = DataLoaderConfig::builder()
            .batch_size(1)
            .num_workers(4)
            .timeout(Duration::from_secs(60))
            .prefetch_factor(3)
            .build();

        let dataloader = DataLoader::new(dataset, sampler, config)?;

        // Can't access config directly, but we can verify it works
        let batches: Vec<_> = dataloader.iter()?.collect::<Result<Vec<_>>>()?;
        assert_eq!(batches.len(), 1);

        Ok(())
    }

    #[test]
    fn test_transform_error_propagation() -> Result<()> {
        struct FailingTransform;
        impl Transform<String, Sample> for FailingTransform {
            fn apply(&self, _: String) -> Result<Sample> {
                Err(anyhow!("Transform failed"))
            }
        }

        let dataset =
            InMemoryDataset::new(vec!["test".to_string()]).with_transform(FailingTransform);
        let sampler = SequentialSampler::new(1);
        let config = DataLoaderConfig::default();

        let dataloader = DataLoader::new(dataset, sampler, config)?;
        let result = dataloader.iter()?.next().unwrap();
        assert!(result.is_err());

        // Check error chain
        let err = result.unwrap_err();
        let error_chain = err.chain().map(|e| e.to_string()).collect::<Vec<_>>();
        assert!(
            error_chain.iter().any(|e| e.contains("Transform failed")),
            "Expected 'Transform failed' in error chain, but got: {:?}",
            error_chain
        );
        Ok(())
    }

    #[test]
    fn test_worker_shutdown_cleanup() -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        // Track how many samples were processed
        let process_count = Arc::new(AtomicUsize::new(0));

        #[derive(Clone)]
        struct CountingTransform {
            counter: Arc<AtomicUsize>,
        }

        impl Transform<String, Sample> for CountingTransform {
            fn apply(&self, s: String) -> Result<Sample> {
                self.counter.fetch_add(1, Ordering::SeqCst);
                // Add small delay to make timing more predictable
                std::thread::sleep(Duration::from_millis(10));
                Ok(Sample::from_single("data", Tensor::from(s.len() as i64)))
            }
        }

        let transform = CountingTransform {
            counter: process_count.clone(),
        };

        // Test that dropiing loader stops all processing
        {
            let dataset =
                InMemoryDataset::new((0..100).map(|i| format!("item{}", i)).collect::<Vec<_>>())
                    .with_transform(transform.clone());

            let loader = DataLoader::new(
                dataset,
                SequentialSampler::new(100),
                DataLoaderConfig::builder()
                    .batch_size(1)
                    .num_workers(2)
                    .prefetch_factor(1) // Minimal prefetch
                    .build(),
            )?;

            // Start iteration but only consume first batch
            let mut iter = loader.iter()?;
            let _first = iter.next().unwrap()?;

            // Record count after first batch
            let count_after_first = process_count.load(Ordering::SeqCst);
            assert!(
                count_after_first >= 1,
                "Should have processed at least 1 item"
            );

            // Drop everything (loader and iterator)
            drop(iter);
            drop(loader);
        } // Everything cleaned up here

        // Give time for any lingering worker threads to finish
        std::thread::sleep(Duration::from_millis(200));

        let final_count = process_count.load(Ordering::SeqCst);

        // Workers should have stopped early, not processed all 100 items
        assert!(
            final_count < 50,
            "Workers should have stopped early, but processed {} items",
            final_count
        );

        Ok(())
    }

    #[test]
    fn test_worker_timeout() -> Result<()> {
        struct SlowTransform;
        impl Transform<String, Sample> for SlowTransform {
            fn apply(&self, _: String) -> Result<Sample> {
                std::thread::sleep(Duration::from_secs(1));
                Ok(Sample::from_single("data", Tensor::from_slice(&[0])))
            }
        }

        let dataset = InMemoryDataset::new(vec!["a".to_string()]).with_transform(SlowTransform);

        let config = DataLoaderConfig::builder()
            .num_workers(1)
            .timeout(Duration::from_millis(10))
            .build();

        let loader = DataLoader::new(dataset, SequentialSampler::new(1), config)?;
        let mut iter = loader.iter()?;

        match iter.next() {
            Some(Err(e)) => {
                let error_string = format!("{:?}", e);
                assert!(
                    error_string.to_lowercase().contains("timeout"),
                    "Expected timeout error, but got: {:?}",
                    e
                );
                Ok(())
            }
            _ => Err(anyhow!("Expected timeout error")),
        }
    }
}

#[cfg(test)]
mod batch_sampler_tests {
    use super::*;
    use crate::collator::{PaddingCollator, PaddingRule};
    use crate::sampler::{BatchBucketSampler, SequentialSampler};
    use crate::transforms::Transform;
    use tch::{Kind, Tensor};

    // Transform that creates variable-length samples
    struct VariableLengthTransform;
    impl Transform<String, Sample> for VariableLengthTransform {
        fn apply(&self, input: String) -> Result<Sample> {
            // Create tensor with length matching string length
            let len = input.len();
            let data = vec![1i64; len];
            Ok(Sample::from_single("input_ids", Tensor::from_slice(&data)))
        }
    }

    #[test]
    fn test_new_with_batch_sampler() -> Result<()> {
        // Create dataset with SAME-LENGTH strings for StackCollator
        let data: Vec<String> = vec![
            "12345".into(), // length 5
            "abcde".into(), // length 5
            "hello".into(), // length 5
            "world".into(), // length 5
            "tests".into(), // length 5
            "rustc".into(), // length 5
        ];

        let dataset = InMemoryDataset::new(data.clone()).with_transform(VariableLengthTransform);
        let base_sampler = SequentialSampler::new(dataset.len());
        let batch_sampler = BatchSampler::new(base_sampler, 2, false)?;

        let config = DataLoaderConfig::builder().num_workers(0).build();

        let loader = DataLoader::new_with_batch_sampler(dataset, batch_sampler, config)?;

        // Collect all batches
        let batches: Vec<_> = loader.iter()?.collect::<Result<Vec<_>>>()?;

        // Should have 3 batches (6 samples / batch_size 2)
        assert_eq!(batches.len(), 3);

        // Verify each batch has correct size
        for (i, batch) in batches.iter().enumerate() {
            let tensor = batch.get("input_ids")?;
            assert_eq!(tensor.size(), &[2, 5], "Batch {} has wrong shape", i);
        }

        Ok(())
    }

    #[test]
    fn test_new_with_batch_sampler_and_collator() -> Result<()> {
        // Create dataset with variable-length strings
        let data: Vec<String> = vec![
            "short".into(),
            "medium text".into(),
            "a much longer text".into(),
            "tiny".into(),
            "another medium one".into(),
            "the longest text of all".into(),
        ];

        let dataset = InMemoryDataset::new(data.clone()).with_transform(VariableLengthTransform);

        // Create a batch bucket sampler that groups by length
        let base_sampler = SequentialSampler::new(dataset.len());
        let data_clone = data.clone();
        let bucket_sampler = BatchBucketSampler::new(
            base_sampler,
            2,                                       // batch_size
            false,                                   // drop_last
            move |idx| data_clone[idx].len() as f64, // sort by string length
            2,                                       // bucket_size_multiplier (bucket_size = 4)
            42,                                      // seed
        )?;

        // Use PaddingCollator since sequences have different lengths
        let padding_collator =
            PaddingCollator::new().pad("input_ids", vec![(0, PaddingRule::MaxLength)], Some(0.0));

        let config = DataLoaderConfig::builder()
            .num_workers(0) // Single-threaded for deterministic test
            .build();

        let loader = DataLoader::new_with_batch_sampler_and_collator(
            dataset,
            bucket_sampler,
            config,
            padding_collator, // Need padding for variable lengths
        )?;

        // Collect all batches
        let batches: Vec<_> = loader.iter()?.collect::<Result<Vec<_>>>()?;

        // Should have 3 batches (6 samples / batch_size 2)
        assert_eq!(batches.len(), 3);

        // Verify batches contain similar-length sequences
        for batch in batches {
            let tensor = batch.get("input_ids")?;
            let batch_size = tensor.size()[0];

            if batch_size > 1 {
                // Get original lengths (non-padding values)
                let seq0 = tensor.select(0, 0);
                let seq1 = tensor.select(0, 1);

                let len0 = seq0.ne(0).sum(Kind::Int64).int64_value(&[]);
                let len1 = seq1.ne(0).sum(Kind::Int64).int64_value(&[]);

                let diff = (len0 - len1).abs();

                // Bucket sampler should group similar lengths
                assert!(
                    diff <= 10,
                    "Length difference {} too large, bucket sampler not working",
                    diff
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_batch_sampler_ignores_config_batch_size() -> Result<()> {
        let data = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let dataset = InMemoryDataset::new(data).with_transform(VariableLengthTransform);

        // Batch sampler with batch_size=1
        let base = SequentialSampler::new(dataset.len());
        let batch_sampler = BatchSampler::new(base, 1, false)?;

        // Config with different batch_size=10 (should be ignored)
        let config = DataLoaderConfig::builder().batch_size(10).build();

        let loader = DataLoader::new_with_batch_sampler(dataset, batch_sampler, config)?;

        let batches: Vec<_> = loader.iter()?.collect::<Result<Vec<_>>>()?;

        // Should have 3 batches of size 1, not 1 batch of size 3
        assert_eq!(batches.len(), 3);
        for batch in batches {
            assert_eq!(batch.batch_size()?, 1);
        }

        Ok(())
    }

    #[test]
    fn test_batch_sampler_with_workers() -> Result<()> {
        let data: Vec<String> = (0..100).map(|i| format!("sample_{}", i)).collect();

        let dataset = InMemoryDataset::new(data).with_transform(VariableLengthTransform);

        // Custom batch sampler
        let base = SequentialSampler::new(dataset.len());
        let batch_sampler = BatchSampler::new(base, 10, false)?;

        let config = DataLoaderConfig::builder()
            .num_workers(2)
            .prefetch_factor(2)
            .build();

        let loader = DataLoader::new_with_batch_sampler(dataset, batch_sampler, config)?;

        let mut total_samples = 0;
        for batch in loader.iter()? {
            let batch = batch?;
            total_samples += batch.batch_size()? as usize;
        }

        assert_eq!(total_samples, 100);
        Ok(())
    }
}
