//! src/dataloader/config.rs
//!
//! Configuration for DataLoader behaviour
//!
//! The `DataLoaderConfig` struct stores the parameters that control
//! how data is loaded.
//!
//! Example:
//! ```ignore
//! let config = DataLoaderConfig::builder()
//!     .batch_size(32)
//!     .num_workers(4)
//!     .persistent_workers(true)
//!     .prefetch_factor(2)
//!     .drop_last(true)
//!     .build();
//! ```
//!
//! # Performance considerations:
//! - `num_workers`: More workers can improve throughput but increase memory usage
//! - `prefetch_factor`: Higher values reduce GPU starvation but use more memory
//! - `persistent_workers`: Reduces overhead between epochs but requires careful handling
//!                         of synchronization among workers to avoid deadlock etc.

use std::time::Duration;

/// Configuration for DataLoader
#[derive(Clone)]
pub struct DataLoaderConfig {
    /// Number of samples per batch (defaults to 1 if not specified)
    pub batch_size: Option<usize>,
    /// Number of parallel workers (0 = single-threaded)
    pub num_workers: usize,
    /// Whether to drop the last incomplete batch (defaults to false if not specified)
    pub drop_last: Option<bool>,
    /// Whether to shuffle data each epoch (in-memory datasets only)
    /// Mutually exclusive with providing a sampler or batch sampler.
    pub shuffle: Option<bool>,
    /// Random seed for reproducible shuffling and transforms
    pub seed: Option<u64>,
    /// Number of batches to prefetch per worker (must be >0 when using workers)
    pub prefetch_factor: usize,
    /// Maximum time to wait for a complete batch from workers.
    /// If exceeded, returns an error (assuming workers are stuck). Default: 30s
    pub timeout: Duration,
    /// How often idle workers check for shutdown signal.
    /// Not an error timeout - just a polling interval. Default: 100ms.
    pub worker_timeout: Duration,
    /// Whether to reuse workers across iterations
    pub persistent_workers: bool,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: None,
            num_workers: 0,
            drop_last: None,
            shuffle: None,
            seed: None,
            prefetch_factor: 2,
            timeout: Duration::from_secs(30),
            worker_timeout: Duration::from_millis(100),
            persistent_workers: false,
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
        self.config.batch_size = Some(size);
        self
    }

    /// Set the number of workers
    pub fn num_workers(mut self, workers: usize) -> Self {
        self.config.num_workers = workers;
        self
    }

    /// Set whether to drop_last
    pub fn drop_last(mut self, drop: bool) -> Self {
        self.config.drop_last = Some(drop);
        self
    }

    /// Set whether to shuffle dataset every epoch
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.config.shuffle = Some(shuffle);
        self
    }

    /// Set the random seed for reproducible data loading.
    ///
    /// When set, this seed controls:
    /// - Data shuffling (if shuffle = true)
    /// - Random transforms in workers
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Set the prefetch factor for the dataset.
    /// Higher values help prevent GPU starvation but use more memory.
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

    /// Set the worker polling interval
    ///
    /// - Too low: More responsive shutdown, higher CPU usage.
    /// - Too high: Less CPU overhead, slower shutdown response
    pub fn worker_timeout(mut self, worker_timeout: Duration) -> Self {
        self.config.worker_timeout = worker_timeout;
        self
    }

    /// Enable persistent workers that survive across iterations.
    pub fn persistent_workers(mut self, persistent: bool) -> Self {
        self.config.persistent_workers = persistent;
        self
    }

    /// Build the final configuration.
    pub fn build(self) -> DataLoaderConfig {
        self.config
    }
}
