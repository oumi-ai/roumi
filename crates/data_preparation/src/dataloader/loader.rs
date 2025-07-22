//! src/dataloader/loader.rs
//!
//! DataLoader constructors for different dataset types and batching needs.
//!
//! # Constructor Overview
//!
//! ## For InMemoryDataset:
//!
//! ### Automatic Sampling
//! DataLoader creates the appropriate sampler based on `config.shuffle`:
//! - If `config.shuffle = false`, DataLoader creates a SequentialSampler.
//! - If `config.shuffle = true`, DataLoader creates a RandomSampler with `config.seed`.
//!
//! **Methods:**
//! - `new()` - Auto-sampling + default StackCollator
//! - `new_with_collator()` - Auto-sampling + custom Collator
//!
//! ### Custom sampling
//! Users provide the sampler, DataLoader just coordinates.
//!
//! **Methods:**
//! - `new_with_sampler()` - User provided sampler + default StackCollator
//! - `new_with_sampler_and_collator()` - User provided sampler + custom Collator
//! - `new_with_batch_sampler()` - User provided batch sampler + default StackCollator
//! - `new_with_batch_sampler_and_collator()` - User provided batch sampler + custom Collator
//!
//! NOTE: For index-based (not pre-batched) samplers, the DataLoader will automatically wrap them with
//!       `BatchSampler` using the configured batch_size and drop_last.
//!
//! ## For IterableDataset:
//! - `new_iterable()` - With default StackCollator
//! - `new_iterable_with_collator()` - With custom Collator
//!
//!
//! # Seed Coordination
//!
//! For full reproducibility when using custom samplers, use the same seed for both
//! the sampler and DataLoader's config:
//!
//! ```ignore
//! let seed = 42;
//! let sampler = WeightedRandomSampler::new(weights, false, None, seed)?;
//! let config = DataLoaderConfig::builder()
//!     .batch_size(32)
//!     .seed(seed)  // Same seed for coordinated randomness!
//!     .build();
//! let dataloader = DataLoader::new_with_sampler(dataset, sampler, config)?;
//! ```
//!
//! The sampler's seed controls sampling order, while config.seed controls worker
//! initialization and transform randomness. Using the same seed ensures all
//! randomness is coordinated.

use crate::collator::{Collator, StackCollator};
use crate::dataset::{InMemoryDataset, IterableDataset};
use crate::sampler::{BatchSampler, RandomSampler, Sampler, SequentialSampler};
use anyhow::{anyhow, Context, Result};
use rand::Rng;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use super::config::DataLoaderConfig;
use super::workers::inmemory::InMemoryWorkerManager;
use super::workers::iterable::IterableWorkerManager;

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
    pub(crate) dataset: D,
    pub(crate) collator: C,
    pub(crate) config: DataLoaderConfig,
    pub(crate) current_epoch: AtomicUsize,
    pub(crate) runtime_seed: Option<u64>,
    pub(crate) loader_type: LoaderType,
}

/// Internal enum representing the data loading strategy based on dataset type.
///
/// This separation allows optimal implementations for different access patterns.
/// - InMemory: Uses samplers for flexible iteration order (random, sequential, etc.)
/// - Iterable: Direct iteration without sampling (no random access)
pub(crate) enum LoaderType {
    InMemory {
        batch_sampler: Box<dyn Sampler<Item = Vec<usize>> + Send + Sync>,
        worker_manager: Option<Arc<InMemoryWorkerManager>>,
    },
    Iterable {
        worker_manager: Option<Arc<IterableWorkerManager>>,
    },
}

// ================================================================================================
// 2. DataLoader Constructors for InMemoryDataset
// ================================================================================================
impl<Raw> DataLoader<InMemoryDataset<Raw>, StackCollator>
where
    Raw: Clone + Send + Sync + 'static,
{
    /// Creates a new DataLoader for in-memory datasets with default StackCollator
    /// and that automatically manages sampling.
    ///
    /// The DataLoader creates an appropriate sampler based on `config.shuffle`.
    /// - If `shuffle = true`: Creates a RandomSampler with `config.seed`
    /// - If `shuffle = false`: Creates a SequentialSampler
    ///
    /// # Example
    /// ```ignore
    /// // Simple shuffled training
    /// let config = DataLoaderConfig::builder()
    ///     .batch_size(32)
    ///     .shuffle(true)
    ///     .seed(42)
    ///     .build();
    /// let dataloader = DataLoader::new(dataset, config)?;
    /// ```
    pub fn new(dataset: InMemoryDataset<Raw>, config: DataLoaderConfig) -> Result<Self> {
        Self::new_with_collator(dataset, config, StackCollator)
    }
}

impl<Raw, C> DataLoader<InMemoryDataset<Raw>, C>
where
    Raw: Clone + Send + Sync + 'static,
    C: Collator + Clone + Send + Sync + 'static,
{
    /// Creates a new DataLoader with automatic sampling and a custom collator.
    ///
    /// # Arguments
    /// - `dataset`: The dataset to load from.
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
        mut config: DataLoaderConfig,
        collator: C,
    ) -> Result<Self> {
        config.batch_size = Some(config.batch_size.unwrap_or(1));
        config.drop_last = Some(config.drop_last.unwrap_or(false));
        config.shuffle = Some(config.shuffle.unwrap_or(false));

        if config.batch_size.unwrap() == 0 {
            return Err(anyhow!("Batch size must be greater than 0"));
        }

        if config.prefetch_factor == 0 && config.num_workers > 0 {
            return Err(anyhow!(
                "Prefetch factor must be > 0 when using {} workers",
                config.num_workers
            ));
        }

        // Create sampler based on shuffle setting
        let effective_seed = config.seed.unwrap_or_else(|| rand::rng().random());

        let sampler: Box<dyn Sampler<Item = usize> + Send + Sync> = if config.shuffle.unwrap() {
            Box::new(RandomSampler::new(
                dataset.len(),
                false,
                None,
                effective_seed,
            )?)
        } else {
            Box::new(SequentialSampler::new(dataset.len()))
        };

        // Wrap in BatchSampler
        let batch_sampler = BatchSampler::new(
            sampler,
            config.batch_size.unwrap(),
            config.drop_last.unwrap(),
        )
        .context("Failed to wrap sampler with BatchSampler")?;

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
            current_epoch: AtomicUsize::new(0),
            runtime_seed: Some(effective_seed),
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
    /// Creates a DataLoader with user-provided sampler and default StackCollator.
    ///
    /// # Errors
    /// - Returns error if `config.shuffle = true`
    /// - Returns error if `config.seed` and sampler's seed do not match.
    ///
    /// # Example
    /// ```ignore
    /// let sampler = WeightedRandomSampler::new(weights, false, None, 42)?;
    /// let config = DataLoaderConfig::builder()
    ///     .batch_size(32)
    ///     .build(); // No shuffle or seed!
    /// let dataloader = DataLoader::new_with_sampler(dataset, sampler, config)?;
    /// ```
    pub fn new_with_sampler(
        dataset: InMemoryDataset<Raw>,
        sampler: impl Sampler<Item = usize> + Send + Sync + 'static,
        config: DataLoaderConfig,
    ) -> Result<Self> {
        Self::new_with_sampler_and_collator(dataset, sampler, config, StackCollator)
    }
}

impl<Raw, C> DataLoader<InMemoryDataset<Raw>, C>
where
    Raw: Clone + Send + Sync + 'static,
    C: Collator + Clone + Send + Sync + 'static,
{
    pub fn new_with_sampler_and_collator(
        dataset: InMemoryDataset<Raw>,
        sampler: impl Sampler<Item = usize> + Send + Sync + 'static,
        mut config: DataLoaderConfig,
        collator: C,
    ) -> Result<Self> {
        if let Some(true) = config.shuffle {
            return Err(anyhow!(
                "Cannot specify shuffle = true when providing a custom sampler.\n\
                Either:\n\
                1. Use DataLoader::new() with shuffle=true to let DataLoader manage sampling\n\
                2. Use DataLoader::new_with_sampler() with shuffle=false and your own sampler"
            ));
        }

        config.batch_size = Some(config.batch_size.unwrap_or(1));
        config.drop_last = Some(config.drop_last.unwrap_or(false));
        config.shuffle = Some(config.shuffle.unwrap_or(false));

        if config.batch_size.unwrap() == 0 {
            return Err(anyhow!("Batch size must be greater than 0"));
        }

        let runtime_seed = match (sampler.seed(), config.seed) {
            (Some(sampler_seed), Some(config_seed)) => {
                if sampler_seed != config_seed {
                    return Err(anyhow!(
                        "Seed mismatch: sampler uses seed {} but dataloader's config.seed is {}.\n\
                        For proper coordination, use the same seed value:\n\
                        let seed = 42;\n\
                        let sampler = RandomSampler::new(..., seed)?;\n\
                        let config = DataLoaderConfig::builder().seed(seed).build();",
                        sampler_seed,
                        config_seed
                    ));
                }
                sampler_seed
            }
            (Some(sampler_seed), None) => {
                // Use sampler's seed for everything
                sampler_seed
            }
            (None, Some(config_seed)) => {
                // Non-random sampler, use config seed for workers
                config_seed
            }
            (None, None) => {
                // Generate seed for workers/transforms
                rand::rng().random()
            }
        };

        if config.prefetch_factor == 0 && config.num_workers > 0 {
            return Err(anyhow!(
                "Prefetch factor must be >0 when using {} workers",
                config.num_workers
            ));
        }

        let batch_sampler = BatchSampler::new(
            sampler,
            config.batch_size.unwrap(),
            config.drop_last.unwrap(),
        )
        .context("Failed to wrap sampler with BatchSampler")?;

        let worker_manager = if config.num_workers > 0 {
            let shared_dataset = Arc::new(dataset.clone());
            Some(Arc::new(
                InMemoryWorkerManager::new(
                    config.num_workers,
                    shared_dataset,
                    collator.clone(),
                    &config,
                )
                .context("Failed to initialize worker manager for in-memory dataset")?,
            ))
        } else {
            None
        };

        Ok(Self {
            dataset,
            collator,
            config,
            current_epoch: AtomicUsize::new(0),
            runtime_seed: Some(runtime_seed),
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
    /// Creates a DataLoader with custom batch sampler and default StackCollator.
    ///
    /// Use this for advanced batching strategies such as:
    /// - Length-based bucketing (BatchBucketSampler)
    /// - Class-balanced batching
    /// - Custom grouping logic
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
    /// Common in NLP for grouping similar-length sequences and padding efficiently.
    ///
    /// Note:
    /// - `config.batch_size` is ignored since the batch sampler controls batching.
    /// - `config.shuffle` must be false
    /// - `config.seed` must be None
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
        mut config: DataLoaderConfig,
        collator: C,
    ) -> Result<Self> {
        if config.batch_size.is_some() {
            return Err(anyhow!(
                "batch_size must not be specified when using batch_sampler.\n\
                The batch_sampler controls batch size."
            ));
        }

        if config.drop_last.is_some() {
            return Err(anyhow!(
                "drop_last must not be specified when using batch_sampler.\n\
                The batch_sampler controls whether to drop the last batch."
            ));
        }

        if let Some(true) = config.shuffle {
            return Err(anyhow!(
                "Cannot specify shuffle=true when providing a batch sampler.\n\
                The batch sampler should handle its own shuffling logic."
            ));
        }

        config.shuffle = Some(config.shuffle.unwrap_or(false));

        // Coordinate seeds between batch sampler and DataLoader
        let runtime_seed = match (batch_sampler.seed(), config.seed) {
            (Some(sampler_seed), Some(config_seed)) => {
                if sampler_seed != config_seed {
                    return Err(anyhow!(
                        "Seed mismatch: batch sampler uses seed {} but dataloader's config.seed is {}.\n\
                        For proper coordination, use the same seed value.",
                        sampler_seed,
                        config_seed,
                    ));
                }
                sampler_seed
            }
            (Some(sampler_seed), None) => sampler_seed,
            (None, Some(config_seed)) => config_seed,
            (None, None) => rand::rng().random(),
        };

        if config.prefetch_factor == 0 && config.num_workers > 0 {
            return Err(anyhow!(
                "Prefetch factor must be >0 when using {} workers",
                config.num_workers
            ));
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
                .context("Failed to initialize worker pool for in-memory dataset")?,
            ))
        } else {
            None
        };

        Ok(Self {
            dataset,
            collator,
            config,
            current_epoch: AtomicUsize::new(0),
            runtime_seed: Some(runtime_seed),
            loader_type: LoaderType::InMemory {
                batch_sampler: Box::new(batch_sampler),
                worker_manager,
            },
        })
    }
}

// ================================================================================================
// 3. DataLoader Constructor for IterableDataset
// ================================================================================================
impl<Raw> DataLoader<IterableDataset<Raw>, StackCollator>
where
    Raw: Clone + Send + Sync + 'static,
{
    /// Creates a DataLoader for iterable datasets with the default StackCollator.
    ///
    /// Use this for large datasets that don't fit in memory or infinite data streams.
    ///
    /// # Note:
    /// IterableDataset do not support generic index-based shuffling or random access.
    ///
    /// # Example:
    /// ```ignore
    /// let dataset = IterableDataset::new(vec![source1, source2]);
    /// let config = DataLoaderConfig::builder()
    ///     .batch_size(32)
    ///     .num_workers(4)
    ///     .build();
    /// let dataloader = DataLoader::new_iterable(dataset, config)?;
    /// ```
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
        mut config: DataLoaderConfig,
        collator: C,
    ) -> Result<Self> {
        if let Some(0) = config.batch_size {
            return Err(anyhow!("Batch size must be greater than 0"));
        }

        config.batch_size = Some(config.batch_size.unwrap_or(1));
        config.drop_last = Some(config.drop_last.unwrap_or(false));

        if let Some(true) = config.shuffle {
            eprintln!(
                "Warning: shuffle = true is ignored for IterableDataset. \
                Consider implementing shuffling at the DataSource level."
            );
        }
        config.shuffle = Some(config.shuffle.unwrap_or(false));

        // Generate runtime seed for workers if none provided
        let effective_seed = config.seed.unwrap_or_else(|| rand::rng().random());

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
            current_epoch: AtomicUsize::new(0),
            runtime_seed: Some(effective_seed),
            loader_type: LoaderType::Iterable { worker_manager },
        })
    }
}
