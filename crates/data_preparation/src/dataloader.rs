use crate::collator::StackCollator;
use crate::sampler::Sampler;
use anyhow::{anyhow, Result};
use crossbeam_channel::{bounded, Receiver, Sender};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// ================================================================================================
// 1. Core Types (DataLoader, LoaderType)
// ================================================================================================

/// The main DataLoader struct that coordinates data loading
///
/// Supports two modes:
/// - In-memory: Random access with sampling strategies
/// - Iterable: Sequential access with shard distribution
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
    /// Timeout for batch operations
    pub timeout: Duration,
    /// Timeout for worker receive operations
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

/// Builder for DataLoaderConfig
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

    /// Set the timeout for batch operations
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Set the worker timeout for receiving operations
    pub fn worker_timeout(mut self, worker_timeout: Duration) -> Self {
        self.config.worker_timeout = worker_timeout;
        self
    }

    pub fn build(self) -> DataLoaderConfig {
        self.config
    }
}

// ================================================================================================
// 3. Worker Management
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
/// Managers worker lifecycle and communication channels.
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
    fn new<F>(num_workers: usize, buffer_size: usize, worker_fn: F) -> Result<Self>
    where
        F: Fn(Receiver<Task>, Sender<Output>, Arc<AtomicBool>) + Send + Sync + 'static,
    {
        if num_workers == 0 {
            return Err(anyhow!("Cannot create WorkerPool with 0 workers"));
        }

        if buffer_size == 0 {
            return Err(anyhow!("Cannot create WorkerPool with buffer_size 0"));
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

            let handle = thread::spawn(move || {
                WORKER_ID.with(|id| *id.borrow_mut() = worker_id);
                worker_fn_clone(
                    task_receiver_clone,
                    task_result_sender_clone,
                    shutdown_clone,
                );
            });

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
        self.shutdown.store(true, Ordering::Relaxed);
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}
