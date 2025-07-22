use anyhow::{anyhow, ensure, Result};
use rand::distr::{weighted::WeightedIndex, Distribution};
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashSet;

/// A `Sampler` defines the strategy for how to iterate and draw samples from
/// a dataset.
///
/// # Associated type
/// - `Item`: The handle yielded by the iterator
///    - For in-memory datasets this is a `usize` index
///    - For streaming datasets it could be just the next item `()` or
///      a custom handle like `(shard, offset)`.
///
/// # Method
/// - `iter(epoch)`: returns a sequential or shuffled sequence for that epoch.
///    - Users pass the `epoch` parameter so internally the sampler uses it
///      together with the base RNG seed to shuffle in a reproducible way across epochs.
///
/// Implementations must be `Send + Sync` so the same sampler instance can be
/// safely shared across DataLoader worker threads.
pub trait Sampler: Send + Sync {
    type Item: Send + Sync;

    fn iter(&self, epoch: usize) -> Box<dyn Iterator<Item = Self::Item> + Send + '_>;
}

/// ============================================================================
/// Yields indices sequentially in order `(0,1,2,...,dataset_size-1)`.
///
/// # Arguments:
/// - `dataset_size`: Total number of samples in a dataset
///
/// # Examples
/// ```ignore
/// let sampler = SequentialSampler::new(5);
/// let indices: Vec<_> = sampler.iter(0).collect();
/// assert_eq!(indices, vec![0, 1, 2, 3, 4]);
/// ```
#[derive(Debug, Clone)]
pub struct SequentialSampler {
    dataset_size: usize,
}

impl SequentialSampler {
    pub fn new(dataset_size: usize) -> Self {
        Self { dataset_size }
    }
}

impl Sampler for SequentialSampler {
    type Item = usize;

    fn iter(&self, _epoch: usize) -> Box<dyn Iterator<Item = usize> + Send + '_> {
        Box::new(0..self.dataset_size)
    }
}

/// ============================================================================
/// Random uniform sampling over `0..dataset_size`, with optional replacement.
///
/// # Arguments:
/// - `dataset_size`: Total number of samples in a dataset.
/// - `replacement`: If `true`, each draw is independent and indices may repeat;
///                  If `false`, each index can only appear once.
/// - `num_samples`: Number of samples to draw from (defaults to the full-range `dataset_size` if `None`)
///                  If `replacement=false`, users must have num_samples <= dataset_size.
/// - `base_seed`: Base RNG seed.
///
/// # Seed Handling
/// 1. Seeds matter because:
/// - Fixed seed -> Identical order every run (good for reproducibility)
/// - Varying seed -> Better model generalization (sees data in different orders)
///
/// # Design rationale
/// 1. Per-epoch variation
/// - For each epoch, we derive a new RNG as `base_seed + epoch`. So at epoch = 0, the RNG = base_seed;
///   At epoch = 1, the RNG = base_seed + 1.
/// - Benefits:
///   - Fresh shuffle each epoch (better training)
///   - Still reproducible with the same `base_seed`.
///
/// 2. Multi-worker safety
/// - For multi-worker setups, the DataLoader initializes a separate `RandomSampler` instance per worker.
/// - The DataLoader provides each instance with a distinct `base_seed` (typically `global_seed + worker_id`).
/// - So the resulting RNG seed for each worker at each epoch is: `(global_seed + worker_id) + epoch`
/// - This prevents duplicate samples across workers and biased gradient updates.
/// - Note: The sampler itself does not manage worker_id. Worker coordination is handled by `DataLoader`.
///
/// 3. For further reading, please refer to this notebook on seeds and determinism:
///    https://gist.github.com/NicolasHug/96a75c2d754ff2a7c52afca2c0b628d4
///
/// # Example usage
/// ```ignore
/// // Without replacement
/// let sampler1 = RandomSampler::new(1000, false, None, 42)?;
///
/// // With replacement
/// let sampler2 = RandomSampler::new(1000, true, None, 42)?;
/// ```
#[derive(Debug, Clone)]
pub struct RandomSampler {
    dataset_size: usize,
    replacement: bool,
    num_samples: usize,
    base_seed: u64,
}

impl RandomSampler {
    pub fn new(
        dataset_size: usize,
        replacement: bool,
        num_samples: Option<usize>,
        base_seed: u64,
    ) -> Result<Self> {
        let num_samples = num_samples.unwrap_or(dataset_size);
        ensure!(
            num_samples > 0,
            "num_samples must be a positive integer value, but got num_samples={}",
            num_samples
        );

        if !replacement {
            ensure!(
                num_samples <= dataset_size,
                "num_samples ({}) exceeds dataset size ({}) without replacement",
                num_samples,
                dataset_size
            );
        }

        Ok(Self {
            dataset_size,
            replacement,
            num_samples,
            base_seed,
        })
    }

    /// Derives a deterministic random number generator for the given epoch
    #[inline]
    fn derive_rng_for_epoch(&self, epoch: usize) -> StdRng {
        StdRng::seed_from_u64(self.base_seed.wrapping_add(epoch as u64))
    }
}

impl Sampler for RandomSampler {
    type Item = usize;

    fn iter(&self, epoch: usize) -> Box<dyn Iterator<Item = usize> + Send + '_> {
        let mut rng = self.derive_rng_for_epoch(epoch);
        if self.replacement {
            Box::new((0..self.num_samples).map(move |_| rng.random_range(0..self.dataset_size)))
        } else {
            let mut indices: Vec<_> = (0..self.dataset_size).collect();
            indices.shuffle(&mut rng);
            indices.truncate(self.num_samples);
            Box::new(indices.into_iter())
        }
    }
}

/// ============================================================================
/// Samples elements randomly from a predefined subset of indices, without replacement.
///
/// Each call to `iter(epoch)` returns a new deterministic permutation of the provided
/// indices for that epoch. See `RandomSampler` docs for details.
///
/// # Arguments:
/// - `dataset_size`: Total number of samples in a dataset.
/// - `indices`: The subset of indices to shuffle and sample from. There should be
///              no duplicates and each index should be within bounds (`<dataset_size`).
/// - `base_seed`: Base RNG seed.
///
/// # Example
/// ```ignore
/// // Create sampler for indices 10-19 of a 1000-sample dataset
/// let sampler = SubsetRandomSampler::new(1000, (10..20).collect(), 42)?;
/// let order0: Vec<_> = sampler.iter(0).collect(); // one permutation of 10..19
/// let order1: Vec<_> = sampler.iter(1).collect(); // a different permutation of 10..19
/// ```
#[derive(Debug, Clone)]
pub struct SubsetRandomSampler {
    _dataset_size: usize,
    indices: Vec<usize>,
    base_seed: u64,
}

impl SubsetRandomSampler {
    pub fn new(dataset_size: usize, indices: Vec<usize>, base_seed: u64) -> Result<Self> {
        ensure!(!indices.is_empty(), "Indices must not be empty");

        let mut seen_indices = HashSet::with_capacity(indices.len());
        for &index in &indices {
            ensure!(
                index < dataset_size,
                "Index {} out of bounds for dataset of size {}",
                index,
                dataset_size,
            );
            ensure!(
                seen_indices.insert(index),
                "Duplicate index {} found in SubsetRandomSampler",
                index
            );
        }
        Ok(Self {
            _dataset_size: dataset_size,
            indices,
            base_seed,
        })
    }

    #[inline]
    fn derive_rng_for_epoch(&self, epoch: usize) -> StdRng {
        StdRng::seed_from_u64(self.base_seed.wrapping_add(epoch as u64))
    }
}

impl Sampler for SubsetRandomSampler {
    type Item = usize;

    fn iter(&self, epoch: usize) -> Box<dyn Iterator<Item = usize> + Send + '_> {
        let mut rng = self.derive_rng_for_epoch(epoch);
        let mut shuffled = self.indices.clone();
        shuffled.shuffle(&mut rng);
        Box::new(shuffled.into_iter())
    }
}

/// ============================================================================
/// Sample elements according to the given weights (probabilities), with optional replacement.
///
/// # Arguments:
/// - `dataset_size`: Total number of samples in a dataset.
/// - `weights`: Relative weight for each index (need not sum to 1). Length must match `dataset_size`.
/// - `replacement`: If `true`, each draw is independent and indices may repeat;
///                  If `false`, each index can only appear once.
/// - `num_samples`: Number of samples to draw from (defaults to the full-range `dataset_size` if `None`)
///                  If `replacement=false`, users must have num_samples <= dataset_size.
/// - `base_seed`: Base RNG seed. See `RandomSampler` docs for details.
///
/// # Example
/// ```ignore
/// // Draw 3 samples according to weights
/// let sampler = WeightedRandomSampler::new(
///     5,                              // dataset_size
///     vec![1.0, 2.0, 0.5, 4.0, 1.5],  // weights
///     false,                          // without replacement
///     Some(3),                        // num_samples
///     42                              // seed
/// )?;
/// ```
#[derive(Debug, Clone)]
pub struct WeightedRandomSampler {
    _dataset_size: usize,
    weights: Vec<f64>,
    replacement: bool,
    num_samples: usize,
    base_seed: u64,
}

impl WeightedRandomSampler {
    pub fn new(
        dataset_size: usize,
        weights: Vec<f64>,
        replacement: bool,
        num_samples: Option<usize>,
        base_seed: u64,
    ) -> Result<Self> {
        // Validate the `weights` input
        ensure!(
            weights.len() == dataset_size,
            "The length of the weights sequence ({}) does not match the dataset size ({})",
            weights.len(),
            dataset_size,
        );
        ensure!(
            !weights.is_empty(),
            "The weights sequence must not be empty"
        );
        ensure!(
            weights.iter().all(|&w| w >= 0.0 && w.is_finite()),
            "All weights must be finite and non-negative"
        );
        let total_weight: f64 = weights.iter().sum();
        ensure!(
            total_weight > 0.0,
            "All weights are zero - at least one weight must be positive"
        );

        // Validate the `num_samples` input
        let num_samples = num_samples.unwrap_or(dataset_size);
        ensure!(
            num_samples > 0,
            "num_samples must be a positive integer value, but got num_samples={}",
            num_samples,
        );
        ensure!(
            replacement || num_samples <= dataset_size,
            "num_samples ({}) must be <= dataset_size ({}) when replacement = false",
            num_samples,
            dataset_size,
        );
        if !replacement {
            let positive_count = weights.iter().filter(|&&w| w > 0.0).count();
            ensure!(
                num_samples <= positive_count,
                "num_samples ({}) must be <= number of positive-weight indices ({}) when replacement = false",
                num_samples,
                positive_count,
            );
        }

        Ok(Self {
            _dataset_size: dataset_size,
            weights,
            replacement,
            num_samples,
            base_seed,
        })
    }

    #[inline]
    fn derive_rng_for_epoch(&self, epoch: usize) -> StdRng {
        StdRng::seed_from_u64(self.base_seed.wrapping_add(epoch as u64))
    }
}

impl Sampler for WeightedRandomSampler {
    type Item = usize;

    fn iter(&self, epoch: usize) -> Box<dyn Iterator<Item = usize> + Send + '_> {
        let mut rng = self.derive_rng_for_epoch(epoch);

        if self.replacement {
            let distribution =
                WeightedIndex::new(&self.weights).expect("Weights should be non-negative");
            Box::new((0..self.num_samples).map(move |_| distribution.sample(&mut rng)))
        } else {
            // Without replacement:
            //
            // We cannot reuse `WeightedIndex` here, because it only supports independent
            // draws (with replacement) in constant time. To remove indices once sampled,
            // one would have to rebuild the alias table after each pick-turning an O(1)
            // draw into O(n) per sample, i.e., O(n^2) total.
            //
            // So instead we use weighted sampling with reservoir here:
            // 1. For each index `i` with weight `w_i`, draw `u ~ Uniform(0, 1)`
            // 2. Compute a score for each index `i` as `u.powf(1.0 /w_i)`.
            let mut scored_indices: Vec<(usize, f64)> = self
                .weights
                .iter()
                .enumerate()
                .filter(|(_, &weight)| weight > 0.0) // Skip index with zero-weight
                .map(|(index, &weight)| {
                    let u = rng.random::<f64>();
                    (index, u.powf(1.0 / weight))
                })
                .collect();

            // 3. Sort scored_indices (`(index, score)`) by score descending and
            //    take the top N, where N = num_samples
            // Performance note: The sort is O(n log n) in the dataset size.
            // We can potentially use a priority queue later for better performance if
            // dataset_size is large but num_samples is small.
            scored_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("No NaNs due to filtering"));
            scored_indices.truncate(self.num_samples);
            Box::new(scored_indices.into_iter().map(|(index, _)| index))
        }
    }
}

/// ============================================================================
/// Wraps a [`Sampler`] to yield mini-batches of items.
///
/// Each call to `.iter(epoch)` produces successive `Vec<S::Item>` batches, where each
/// mini-batch contains up to `batch_size` elements drawn from the underlying sampler.
/// If `drop_last` is `true`, any final mini-batch smaller than `batch_size` will be discarded.
///
/// # Type parameters
/// - `S`: The underlying sampler implementing [`Sampler`] (e.g., `RandomSampler`)
///
/// # Arguments:
/// - `sampler`: Base sampler to wrap
/// - `batch_size`: Number of items per batch. Must be >= 1
/// - `drop_last`: If true, discards mini-batches smaller than `batch_size`
///
/// # Example
/// ```ignore
/// // Create mini-batches of 32 indices from a 1000-item dataset.
/// let base_sampler = SequentialSampler::new(1000);
/// let batch_sampler = BatchSampler::new(base_sampler, 32, false).unwrap();
///
/// for mini_batch in batch_sampler.iter(0) {
///     // `mini_batch` is Vec<usize> of length 32, except the last mini-batch
///     println!("Batch size: {}", mini_batch.len());
/// }
/// ```
#[derive(Debug, Clone)]
pub struct BatchSampler<S> {
    sampler: S,
    batch_size: usize,
    drop_last: bool,
}

impl<S: Sampler> BatchSampler<S> {
    pub fn new(sampler: S, batch_size: usize, drop_last: bool) -> Result<Self> {
        ensure!(
            batch_size > 0,
            "batch_size must be > 0, but got batch_size={}",
            batch_size
        );
        Ok(Self {
            sampler,
            batch_size,
            drop_last,
        })
    }
}

impl<S: Sampler> Sampler for BatchSampler<S> {
    type Item = Vec<S::Item>;

    fn iter(&self, epoch: usize) -> Box<dyn Iterator<Item = Self::Item> + Send + '_> {
        let mut sampler_iter = self.sampler.iter(epoch);
        let batch_size = self.batch_size;
        let drop_last = self.drop_last;

        Box::new(std::iter::from_fn(move || {
            let mut mini_batch = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                if let Some(item) = sampler_iter.next() {
                    mini_batch.push(item);
                } else {
                    break;
                }
            }
            if mini_batch.len() == batch_size || (!drop_last && !mini_batch.is_empty()) {
                Some(mini_batch)
            } else {
                None
            }
        }))
    }
}

/// ============================================================================
/// A sampler that iterates over a list of shard handles (e.g., filenames, URLs, offsets),
/// either in sequential or shuffled order.
///
/// If `shuffle = true`, the order is randomized in a deterministic way by seeding an
/// RNG with `base_seed + epoch`. See `RandomSampler` for details on seed derivation.
///
/// # Type parameters:
/// - `H`: Shard handle type (e.g., `PathBuf`, `String`, `(PathBuf, Offset)`, etc.).
///        Must implement `Clone + Send + Sync` so it can be cheaply duplicated across workers.
///
/// # Arguments:
/// - `shards`: The list of shard handles to iterate.
/// - `shuffle`: Whether to shuffle the order every epoch.
/// - `base_seed`: Base RNG seed.
///
/// # Example
/// ```ignore
///
/// // Sequential iteration over Parquet files
/// let seq_sampler = ShardSampler::new(
///     vec![
///         PathBuf::from("data/part1.parquet"),
///         PathBuf::from("data/part2.parquet"),
///     ],
///     false, // shuffle
///     123, // base RNG seed
/// );
///
/// // Shuffled iteration over S3 URIs
/// let s3_sampler = ShardSampler::new(
///     vec![
///         "s3://bucket/part1".to_string(),
///         "s3://bucket/part2".to_string(),
///     ],
///     true,
///     42
/// );
/// ```
#[derive(Debug, Clone)]
pub struct ShardSampler<H: Clone + Send + Sync> {
    shards: Vec<H>,
    shuffle: bool,
    base_seed: u64,
}

impl<H: Clone + Send + Sync> ShardSampler<H> {
    pub fn new(shards: Vec<H>, shuffle: bool, base_seed: u64) -> Result<Self> {
        ensure!(!shards.is_empty(), "Shard list must not be empty");
        Ok(Self {
            shards,
            shuffle,
            base_seed,
        })
    }
}

impl<H: Clone + Send + Sync> Sampler for ShardSampler<H> {
    type Item = H;

    fn iter(&self, epoch: usize) -> Box<dyn Iterator<Item = H> + Send + '_> {
        let mut shards = self.shards.clone();

        if self.shuffle {
            let mut rng = StdRng::seed_from_u64(self.base_seed.wrapping_add(epoch as u64));
            shards.shuffle(&mut rng);
        }
        Box::new(shards.into_iter())
    }
}

/// ============================================================================
/// A sampler that partitions a dataset's indices evenly across multiple workers for
/// parallel training.
///
/// # Arguments:
/// - `dataset_size`: Total number of samples in dataset.
/// - `num_replicas`: Total number of parallel workers.
/// - `rank`: Unique ID for this worker. Must satisfy `0 <= rank < num_replicas`.
/// - `shuffle`: Whether to shuffle the indices.
/// - `drop_last`: If true, any extra indices beyond an exact multiple of `num_replicas` are dropped.
///                If false, the list of indices is padded by cycling from the front until its length
///                is evenly divisible among workers.
/// - `base_seed`: Base RNG seed shared by all workers.
///
/// # Worker allocation
/// - For dataset with 10 samples across 3 workers:
/// ```text
/// Original: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
///
/// drop_last=true (truncate to 9 samples)
///   Worker 0: [0, 3, 6]
///   Worker 1: [1, 4, 7]
///   Worker 2: [2, 5, 8] // 9 is dropped
///
/// drop_last=false (pad to 12 samples)
///   Worker 0: [0, 3, 6, 9]
///   Worker 1: [1, 4, 7, 0] // padded with index 0
///   Worker 2: [2, 5, 8, 1] // padded with index 1
/// ```
///
/// # Deterministic shuffling:
/// - Each sampler instance uses `base_seed + epoch` to seed its RNG.
/// - Note: when running N workers, DataLoader should give each worker a different `starting seed`.
/// - Example usage in DataLoader:
/// ```ignore
/// // Suppose base_seed = 42, num_workers = 4
/// // Worker 0: starting_seed = 42 + 0 = 42
/// // Worker 1: starting_seed = 42 + 1 = 43
/// // ...
/// let sampler_for_worker_i = DistributedSampler::new(
///     dataset_size, num_workers, i, shuffle, drop_last, starting_seed
/// )?;
///
/// // Then, each epoch:
/// let indices = sampler_for_worker_i.iter(epoch);  
/// // Internally seeds RNG with (starting_seed + epoch)
/// ```
#[derive(Debug, Clone)]
pub struct DistributedSampler {
    dataset_size: usize,
    num_replicas: usize,
    rank: usize,
    shuffle: bool,
    drop_last: bool,
    base_seed: u64,
}

impl DistributedSampler {
    pub fn new(
        dataset_size: usize,
        num_replicas: usize,
        rank: usize,
        shuffle: bool,
        drop_last: bool,
        base_seed: u64,
    ) -> Result<Self> {
        ensure!(dataset_size > 0, "Dataset size must not be empty");
        ensure!(num_replicas > 0, "Number of workers must be > 0");
        ensure!(
            rank < num_replicas,
            "Invalid rank {rank}, rank should be in the interval [0, {}]",
            num_replicas.saturating_sub(1)
        );
        Ok(Self {
            dataset_size,
            num_replicas,
            rank,
            shuffle,
            drop_last,
            base_seed,
        })
    }

    #[inline]
    fn derive_rng_for_epoch(&self, epoch: usize) -> StdRng {
        StdRng::seed_from_u64(self.base_seed.wrapping_add(epoch as u64))
    }

    /// Compute the total number of indices to evenly distribute across workers.
    ///
    /// If `drop_last = true`:
    /// - Truncate to the largest multiple of `num_replicas` that does not exceed `dataset_size`.
    /// - Example: `dataset_size = 10`, `num_replicas = 3` → truncate to `9` (because `10 % 3 = 1`).
    ///            The index `9` is dropped before splitting.
    ///
    /// If `drop_last = false`:
    /// - Pad up to the smallest multiple of `num_replicas` that is at least `dataset_size`.
    /// - Example: `dataset_size = 10`, `num_replicas = 3` → pad to `12` (next multiple of 3).
    ///             We will append two extra indices (cycling from the front).         
    fn total_size(&self) -> usize {
        if self.drop_last {
            self.dataset_size - (self.dataset_size % self.num_replicas)
        } else {
            ((self.dataset_size + self.num_replicas - 1) / self.num_replicas) * self.num_replicas
        }
    }
}

impl Sampler for DistributedSampler {
    type Item = usize;

    fn iter(&self, epoch: usize) -> Box<dyn Iterator<Item = usize> + Send + '_> {
        let mut indices: Vec<usize> = (0..self.dataset_size).collect();

        if self.shuffle {
            indices.shuffle(&mut self.derive_rng_for_epoch(epoch));
        }

        let total_size = self.total_size();

        if self.drop_last {
            indices.truncate(total_size);
        } else if total_size > indices.len() {
            let padding: Vec<_> = indices
                .iter()
                .cycle()
                .take(total_size - indices.len())
                .cloned()
                .collect();
            indices.extend(padding);
        }

        // Partition `indices` evenly across `num_replicas`.
        // Each worker picks every `num_replicas`th element, starting at its own `rank`.
        // This guarantees disjoint subsets with no overlaps.
        Box::new(
            indices
                .into_iter()
                .skip(self.rank)
                .step_by(self.num_replicas),
        )
    }
}

/// ============================================================================
/// A sampler that groups indices into length-based buckets before shuffling and batching.
/// This reduces padding overhead when handling variable-length sequences.
///
/// # Type parameters
/// - `S`: An existing sampler that yields `usize` indices (e.g., `RandomSampler`)
/// - `F`: A function mapping `usize -> f64`, used to sort indices (e.g., sequence length).
///
/// # Arguments:
/// - `sampler`: Base sampler providing indices per epoch
/// - `batch_size`: Number of indices per final mini-batch (must be >= 1)
/// - `drop_last`:  If `true`, incomplete final mini-batches are dropped.
///                If `false`, they are kept as is.
/// - `sort_key`: Function mapping `usize` → `f64` used to sort indices within a bucket.
///               Sorted in descending order: higher key ⇒ earlier in bucket.
/// - `bucket_size_multiplier`: Multiplier for bucket size. Each bucket’s size is
///                             `batch_size * bucket_size_multiplier`. Must be ≥ 1.
/// - `base_seed`: Base RNG seed for deterministic shuffling.
///
/// # Algorithm Overview
/// 1. Pre-bucketing
///     - Collects indices in groups of size `batch_size * bucket_multiplier`.
///
/// 2. Sorting (within each bucket)
///     - Sorts indices by a user-provided key (e.g., sequence length) in descending order.
///     - Longer sequences come first, minimizing padding wasted on shorter ones.
///
/// 3. Batching
///     - Splits each sorted bucket into consecutive mini-batches of exactly `batch_size`.
///     - Partial final batches are kept or dropped based on `drop_last`.
///
/// 4. Intra-bucket shuffling
///     - Randomizes the order of those final mini-batches (not the individual examples)
///       using a deterministic RNG seed per bucket. This preserves some randomness while
///       still grouping similar lengths together.
///
/// # Example: NLP Training
/// ```ignore
/// // Suppose you have a collection of text sequences of varying lengths:
/// let texts = vec!["short", "medium length", "a much longer piece of text…"];
/// // Precompute “length” as the sort key for each example:
/// let lengths: Vec<f64> = texts.iter().map(|t| t.len() as f64).collect();
///
/// // Build a BatchBucketSampler that:
/// // 1. Uses a SequentialSampler over text indices [0..texts.len()).
/// // 2. Groups indices into buckets of size = 32 * 100 = 3200.
/// // 3. Sorts each bucket by length (descending).
/// // 4. Splits into mini‐batches of size 32 (dropping any remainder if `drop_last` is true).
/// // 5. Shuffles the resulting mini‐batches using a deterministic RNG seed.
/// let sampler = BatchBucketSampler::new(
///     SequentialSampler::new(texts.len()),  // base sampler
///     32,                                   // batch_size
///     false,                                // drop_last = false
///     |idx| lengths[idx],                  // sort_key: “length of texts[idx]”
///     100,                                  // bucket_multiplier (bucket_size = 3200)
///     42,                                   // base_seed
/// )?;
///
/// // In your training loop:
/// for epoch in 0..num_epochs {
///     for batch_indices in sampler.iter(epoch) {
///         // `batch_indices` is a Vec<usize> of length 32 (except perhaps the last bucket’s remainder)
///         let batch_texts: Vec<&str> = batch_indices.iter().map(|&i| &texts[i]).collect();
///         // ... now you can pad/encode batch_texts and pass them to your model ...
///     }
/// }
/// ```
///
/// # Design Considerations
/// 1. Bucket Size: Larger buckets → better length grouping but more memory
/// 2. Sort Key: Should correlate with example length/complexity
/// 3. Seed Handling:
///     a. For reproducibility, we combine `base_seed` with both the epoch and
///        the bucket’s position:
///         ```text
///         bucket_rng_seed = base_seed + epoch + bucket_id
///         ```  
///     b. This ensures the same bucket‐level shuffle order across runs when
///        `base_seed` and `epoch` match.
///
/// TODO: benchmarks/ for performance characteristics with different bucket sizes.
#[derive(Debug, Clone)]
pub struct BatchBucketSampler<S, F>
where
    S: Sampler<Item = usize>,
    F: Fn(usize) -> f64 + Send + Sync,
{
    bucket_sampler: BatchSampler<S>,
    batch_size: usize,
    drop_last: bool,
    sort_key: F,
    base_seed: u64,
}

impl<S, F> BatchBucketSampler<S, F>
where
    S: Sampler<Item = usize>,
    F: Fn(usize) -> f64 + Send + Sync,
{
    pub fn new(
        sampler: S,
        batch_size: usize,
        drop_last: bool,
        sort_key: F,
        bucket_size_multiplier: usize,
        base_seed: u64,
    ) -> Result<Self> {
        ensure!(
            batch_size > 0,
            "Batch size must be >= 1, but got batch_size={}",
            batch_size
        );
        ensure!(
            bucket_size_multiplier > 0,
            "Bucket size multipler must be >= 1, but got bucket_size_multipler={}",
            bucket_size_multiplier
        );

        let bucket_size = batch_size
            .checked_mul(bucket_size_multiplier)
            .ok_or_else(|| anyhow!("Bucket size overflow"))?;

        let bucket_sampler = BatchSampler::new(sampler, bucket_size, false)?;
        Ok(Self {
            bucket_sampler,
            batch_size,
            drop_last,
            sort_key,
            base_seed,
        })
    }
}

impl<S, F> Sampler for BatchBucketSampler<S, F>
where
    S: Sampler<Item = usize>,
    F: Fn(usize) -> f64 + Send + Sync,
{
    type Item = Vec<usize>;

    /// Produces an iterator of index batches for the given epoch.
    fn iter(&self, epoch: usize) -> Box<dyn Iterator<Item = Vec<usize>> + Send + '_> {
        // 1. Obtain an iterator over pre-buckets (each bucket is Vec<usize> of size `bucket_size`).
        //    Stream one pre-bucket at a time to avoid allocation of all pre-buckets at once.
        let pre_buckets = self.bucket_sampler.iter(epoch);

        // 2. For each pre-bucket, sort by sort_key, split into mini-batches, then shuffle those batches.
        let iter = pre_buckets
            .enumerate()
            .flat_map(move |(bucket_id, mut bucket_indices)| {
                // a. Sort bucket by sort_key descending
                bucket_indices.sort_unstable_by(|&a, &b| {
                    (self.sort_key)(b)
                        .partial_cmp(&(self.sort_key)(a))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // b. Split sorted indices into chunks of `batch_size`.
                let mut batches_in_bucket = Vec::new();
                let mut start = 0;
                let len = bucket_indices.len();
                while start < len {
                    let end = (start + self.batch_size).min(len);
                    if end - start == self.batch_size {
                        batches_in_bucket.push(bucket_indices[start..end].to_vec());
                    } else if !self.drop_last {
                        batches_in_bucket.push(bucket_indices[start..].to_vec());
                    }
                    start += self.batch_size;
                }

                // 3. Shuffle those mini-batches within the bucket, using a
                //    deterministic RNG seed = base_seed + epoch + bucket_id.
                if batches_in_bucket.len() > 1 {
                    let mut rng = StdRng::seed_from_u64(
                        self.base_seed
                            .wrapping_add(epoch as u64)
                            .wrapping_add(bucket_id as u64),
                    );
                    batches_in_bucket.shuffle(&mut rng);
                }

                // 4. Yield the mini-batches for this pre-bucket
                batches_in_bucket.into_iter()
            });
        Box::new(iter)
    }
}

/// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::InMemoryDataset;
    use crate::sample::Sample;
    use tch::Tensor;

    const TEST_SEED: u64 = 42;
    const TEST_DATASET_SIZE: usize = 100;

    mod sequential_sampler_tests {
        use super::*;

        #[test]
        fn yields_sequential_indices() {
            let sampler = SequentialSampler::new(100);
            let indices: Vec<usize> = sampler.iter(0).collect();
            assert_eq!(indices, (0..100).collect::<Vec<_>>());
        }

        #[test]
        fn handles_empty_dataset() {
            let sampler = SequentialSampler::new(0);
            assert_eq!(sampler.iter(0).count(), 0);
        }

        #[test]
        fn works_with_inmemory_dataset() {
            let samples: Vec<Sample> = (0..100)
                .map(|i| Sample::from_single("data", Tensor::from(i)))
                .collect();
            let dataset = InMemoryDataset::new(samples);

            let sampler = SequentialSampler::new(dataset.len());
            let indices: Vec<usize> = sampler.iter(0).collect();
            assert_eq!(indices, (0..100).collect::<Vec<_>>());
        }
    }

    mod random_sampler_tests {
        use super::*;
        use std::collections::HashSet;

        #[test]
        fn validates_parameters() {
            assert!(RandomSampler::new(10, false, None, TEST_SEED).is_ok());

            // Invalid initialization: empty dataset and nothing to sample from
            assert!(RandomSampler::new(0, false, None, TEST_SEED).is_err());

            // Invalid initialization: `num_samples` > `dataset_size` when `replacement = false`
            assert!(RandomSampler::new(10, false, Some(11), TEST_SEED).is_err());
        }

        #[test]
        fn without_replacement_contains_all_indices() {
            let sampler = RandomSampler::new(TEST_DATASET_SIZE, false, None, TEST_SEED).unwrap();
            let samples: Vec<_> = sampler.iter(0).collect();
            assert_eq!(samples.len(), TEST_DATASET_SIZE);
            assert_eq!(HashSet::<_>::from_iter(samples).len(), TEST_DATASET_SIZE);
        }

        #[test]
        fn with_replacement_allows_duplicates() {
            let sampler = RandomSampler::new(10, true, Some(100), TEST_SEED).unwrap();
            let samples: Vec<_> = sampler.iter(0).collect();
            assert!(HashSet::<_>::from_iter(&samples).len() < 100);
        }

        #[test]
        fn produces_deterministic_results() {
            let sampler = RandomSampler::new(TEST_DATASET_SIZE, false, None, TEST_SEED).unwrap();
            let epoch1 = sampler.iter(1).collect::<Vec<_>>();
            assert_eq!(epoch1, sampler.iter(1).collect::<Vec<_>>());
            assert_ne!(epoch1, sampler.iter(2).collect::<Vec<_>>());
        }
    }

    mod subset_random_sampler_tests {
        use super::*;

        #[test]
        fn rejects_invalid_indices() {
            // No indices to sample from
            assert!(SubsetRandomSampler::new(TEST_DATASET_SIZE, vec![], TEST_SEED).is_err());

            // Duplicate index
            assert!(SubsetRandomSampler::new(3, vec![1, 1, 2], TEST_SEED).is_err());

            // Index out of bounds
            assert!(SubsetRandomSampler::new(3, vec![1, 2, 3], TEST_SEED).is_err());
        }

        #[test]
        fn shuffles_provided_indices() {
            let indices = vec![10, 20, 30, 40];
            let sampler =
                SubsetRandomSampler::new(TEST_DATASET_SIZE, indices.clone(), TEST_SEED).unwrap();
            let samples: Vec<_> = sampler.iter(0).collect();
            assert_eq!(
                HashSet::<_>::from_iter(samples),
                HashSet::from_iter(indices)
            );
        }

        #[test]
        fn different_epochs_produce_different_orders() {
            let sampler = SubsetRandomSampler::new(10, vec![1, 2, 3, 4], TEST_SEED).unwrap();
            assert_ne!(
                sampler.iter(1).collect::<Vec<_>>(),
                sampler.iter(2).collect::<Vec<_>>()
            );
        }
    }

    mod weighted_random_sampler_tests {
        use super::*;

        #[test]
        fn validate_weights() {
            // Empty weights
            assert!(WeightedRandomSampler::new(3, vec![], false, Some(1), TEST_SEED).is_err());

            // Negative weights
            assert!(
                WeightedRandomSampler::new(3, vec![0.1, -0.5, 0.2], true, Some(1), TEST_SEED)
                    .is_err()
            );

            // Zero weights
            assert!(
                WeightedRandomSampler::new(3, vec![0.0, 0.0, 0.0], false, Some(1), TEST_SEED)
                    .is_err()
            );
        }

        #[test]
        fn respects_zero_weights() {
            let weights = vec![1.0, 0.0, 2.0];
            let sampler =
                WeightedRandomSampler::new(weights.len(), weights, true, Some(10), TEST_SEED)
                    .unwrap();
            assert!(!sampler.iter(0).any(|index| index == 1));
        }

        #[test]
        fn with_replacement_samples_correctly() {
            let weights = vec![0.1, 0.9];
            let sampler =
                WeightedRandomSampler::new(weights.len(), weights, true, Some(1000), TEST_SEED)
                    .unwrap();
            let samples = sampler.iter(0).collect::<Vec<_>>();
            let count_1 = samples.iter().filter(|&&index| index == 1).count();
            assert!(count_1 > 800); // Very likely to have many more 1s than 0s
        }

        #[test]
        fn without_replacement_uses_all_indices() {
            let weights = vec![0.5, 0.5, 0.5, 0.5];
            let weights_len = weights.len();
            let sampler =
                WeightedRandomSampler::new(weights_len, weights, false, None, TEST_SEED).unwrap();
            let samples = sampler.iter(0).collect::<Vec<_>>();
            assert_eq!(HashSet::<_>::from_iter(samples).len(), weights_len);
        }
    }

    mod batch_sampler_tests {
        use super::*;
        use crate::collator::{Collator, StackCollator};

        #[test]
        fn test_batches_full() {
            let base_sampler = SequentialSampler::new(10);
            let batch_sampler = BatchSampler::new(base_sampler, 2, false).unwrap();
            let mini_batches: Vec<_> = batch_sampler.iter(0).collect();
            assert_eq!(
                mini_batches,
                vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7], vec![8, 9]]
            );
        }

        #[test]
        fn test_batches_drop_last() {
            let base_sampler = SequentialSampler::new(10);
            let batch_sampler = BatchSampler::new(base_sampler, 3, true).unwrap();
            let mini_batches: Vec<_> = batch_sampler.iter(0).collect();
            assert_eq!(
                mini_batches,
                vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]
            );
        }

        #[test]
        fn test_integration_with_dataset() -> Result<()> {
            let samples = vec![
                Sample::from_single("data", Tensor::from(1)),
                Sample::from_single("data", Tensor::from(2)),
                Sample::from_single("data", Tensor::from(3)),
            ];
            let dataset = InMemoryDataset::new(samples);
            let base_sampler = SequentialSampler::new(dataset.len());
            let sampler = BatchSampler::new(base_sampler, 2, false)?;

            let mut all_values = Vec::new();
            for batch_indices in sampler.iter(0) {
                for idx in batch_indices {
                    let sample = dataset.get(idx).unwrap();
                    all_values.push(sample.get("data")?.int64_value(&[]));
                }
            }
            assert_eq!(all_values, vec![1, 2, 3]);
            Ok(())
        }

        #[test]
        fn test_integration_with_dataset_and_collator() -> Result<()> {
            let samples: Vec<Sample> = (0..100)
                .map(|i| Sample::from_single("data", Tensor::from_slice(&[i])))
                .collect();
            let dataset = InMemoryDataset::new(samples);

            // Create pipeline: Sampler -> BaseSampler -> Collator
            let base_sampler = SequentialSampler::new(dataset.len());
            let sampler = BatchSampler::new(base_sampler, 10, false)?;
            let collator = StackCollator;

            // Process batches
            let mut expected_value = 0;
            for batch_indices in sampler.iter(0) {
                let samples: Vec<_> = batch_indices
                    .iter()
                    .map(|&i| dataset.get(i).cloned().unwrap())
                    .collect();
                let minibatch = collator.collate(&samples)?;

                // Verify batch contents
                let tensor = minibatch.get("data")?;
                assert_eq!(tensor.size(), &[10, 1]);

                // Check values
                let values = tensor.size()[0];
                for i in 0..values {
                    assert_eq!(tensor.int64_value(&[i, 0]), expected_value);
                    expected_value += 1;
                }
            }
            Ok(())
        }
    }

    mod shard_sampler_tests {
        use super::*;
        use std::path::PathBuf;

        #[test]
        fn rejects_empty_shards() {
            assert!(ShardSampler::new(Vec::<PathBuf>::new(), false, TEST_SEED).is_err());
        }

        #[test]
        fn yields_sequential_order() -> Result<()> {
            let shards = vec![PathBuf::from("shard1"), PathBuf::from("shard2")];
            let sampler = ShardSampler::new(shards.clone(), false, 42)?;
            let collected: Vec<_> = sampler.iter(0).collect();
            assert_eq!(collected, shards); // Order preserved
            Ok(())
        }

        #[test]
        fn yields_shuffled_order() -> Result<()> {
            let shards = (0..10)
                .map(|i| PathBuf::from(format!("shard{}", i)))
                .collect();
            let sampler = ShardSampler::new(shards, true, TEST_SEED)?;

            let out0: Vec<_> = sampler.iter(0).collect();
            let out1: Vec<_> = sampler.iter(1).collect();
            assert_ne!(out0, out1); // Different permutation per epoch
            assert_eq!(out0, sampler.iter(0).collect::<Vec<_>>()); // Same permutation for same epoch
            Ok(())
        }
    }

    mod distributed_sampler_tests {
        use super::*;

        #[test]
        fn rejects_invalid_args() {
            assert!(DistributedSampler::new(0, 1, 0, false, false, TEST_SEED).is_err()); // dataset_size = 0
            assert!(DistributedSampler::new(10, 0, 0, false, false, TEST_SEED).is_err()); // num_replicas = 0
            assert!(DistributedSampler::new(10, 2, 2, false, false, TEST_SEED).is_err());
            // rank >= num_replicas
        }

        #[test]
        fn drop_last_true_truncates_indices() -> Result<()> {
            // With drop_last = true, should truncate to 9 elements (3 workers x 3 samples)
            let sampler = DistributedSampler::new(10, 3, 2, false, true, TEST_SEED)?;
            let indices: Vec<_> = sampler.iter(0).collect();
            assert_eq!(indices.len(), 3);
            assert_eq!(indices, vec![2, 5, 8]);
            Ok(())
        }

        #[test]
        fn drop_last_false_correctly_pads_indices_by_cycling() -> Result<()> {
            // With drop_last = false, should pad to 12 elements (3 workers x 4 samples)
            let sampler = DistributedSampler::new(10, 3, 2, false, false, TEST_SEED)?;
            let indices: Vec<_> = sampler.iter(0).collect();
            assert_eq!(indices, vec![2, 5, 8, 1]); // Last element is padded from front
            Ok(())
        }

        #[test]
        fn shuffles_deterministically() -> Result<()> {
            let sampler = DistributedSampler::new(100, 4, 0, true, false, TEST_SEED)?;

            // Same epoch = same order
            let epoch1_a: Vec<_> = sampler.iter(1).collect();
            let epoch1_b: Vec<_> = sampler.iter(1).collect();
            assert_eq!(epoch1_a, epoch1_b);

            // Different epoch = different order
            let epoch2: Vec<_> = sampler.iter(2).collect();
            assert_ne!(epoch1_a, epoch2);
            Ok(())
        }

        #[test]
        fn no_duplicates_across_workers() -> Result<()> {
            // Simulate 3 workers with the same dataset
            let workers = (0..3)
                .map(|rank| {
                    DistributedSampler::new(10, 3, rank, true, false, TEST_SEED)
                        .unwrap()
                        .iter(0)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            // Verify no overlapping indices between workers
            let all_indices: HashSet<_> = workers.iter().flatten().collect();
            assert_eq!(all_indices.len(), 10); // All original indices covered
            Ok(())
        }

        #[test]
        fn single_worker_gets_all_indices() -> Result<()> {
            let sampler = DistributedSampler::new(5, 1, 0, true, false, TEST_SEED)?;
            let mut indices: Vec<_> = sampler.iter(0).collect();
            indices.sort_unstable();
            assert_eq!(indices, vec![0, 1, 2, 3, 4]); // Gets all indices
            Ok(())
        }
    }

    mod batch_bucket_sampler_tests {
        use super::*;
        use crate::dataset::{DataSource, Dataset, IterableDataset};
        use crate::readers::TxtSource;
        use crate::transforms::text::Tokenize;
        use std::io::Write;
        use tempfile::NamedTempFile;
        use tokenizers::tokenizer::Tokenizer;

        // Helper function
        fn create_batch_bucket_sampler(
            dataset_size: usize,
            batch_size: usize,
            bucket_multiplier: usize,
        ) -> BatchBucketSampler<SequentialSampler, impl Fn(usize) -> f64> {
            let base_sampler = SequentialSampler::new(dataset_size);
            let sort_key = |i: usize| i as f64; // Use index as sort key for deterministic tests
            BatchBucketSampler::new(
                base_sampler,
                batch_size,
                false, // Don't drop last
                sort_key,
                bucket_multiplier,
                TEST_SEED,
            )
            .unwrap()
        }

        #[test]
        fn create_buckets_successfully() -> Result<()> {
            let sampler = create_batch_bucket_sampler(100, 8, 4); // bucket_size = 32
            let epoch = 0;
            let mut batches = sampler.iter(epoch);

            // First bucket should contain indices 0..32
            let first_batch = batches.next().unwrap();
            assert_eq!(first_batch.len(), 8);
            assert!(first_batch.iter().all(|&x| x < 32));
            Ok(())
        }

        #[test]
        fn sorts_withins_buckets() -> Result<()> {
            let sampler = create_batch_bucket_sampler(100, 8, 4); // bucket_size = 32
            let epoch = 0;
            let batches: Vec<_> = sampler.iter(epoch).collect();

            // Verify sorted order within each *individual mini-batch*
            for batch in batches {
                // Each `batch` is a Vec<usize>
                let mut prev = None;
                for &idx in batch.iter() {
                    if let Some(p) = prev {
                        assert!(
                            idx <= p,
                            "Indices must be in descending order within each mini-batch, but saw {} > {}",
                            idx,
                            p
                        );
                    }
                    prev = Some(idx);
                }
            }
            Ok(())
        }

        #[test]
        fn shuffling_across_batches() -> Result<()> {
            let sampler = create_batch_bucket_sampler(100, 8, 4);
            let epoch = 0;

            // Get first bucket's batches
            let bucket_batches: Vec<_> = sampler.iter(epoch).take(4).collect();

            // Verify batches are shuffled (not sequential)
            let sequential_order: Vec<_> = (0..8)
                .rev()
                .chain(8..16)
                .rev()
                .chain(16..24)
                .rev()
                .chain(24..32)
                .rev()
                .collect();
            let actual_order: Vec<_> = bucket_batches.iter().flatten().copied().collect();

            assert_ne!(
                actual_order, sequential_order,
                "Batches should be shuffled within bucket"
            );
            Ok(())
        }

        #[test]
        fn deterministic_shuffling_across_epochs() -> Result<()> {
            let sampler = create_batch_bucket_sampler(100, 8, 4);

            let epoch1: Vec<_> = sampler.iter(1).flatten().collect();
            let epoch1_repeat: Vec<_> = sampler.iter(1).flatten().collect();
            let epoch2: Vec<_> = sampler.iter(2).flatten().collect();

            assert_eq!(
                epoch1, epoch1_repeat,
                "Same epoch should produce identical sequences"
            );
            assert_ne!(
                epoch1, epoch2,
                "Different epochs should produce different sequences"
            );
            Ok(())
        }

        #[test]
        fn partial_buckets_remain() -> Result<()> {
            let sampler = create_batch_bucket_sampler(50, 8, 4);
            let epoch = 0;

            let all_indices: Vec<_> = sampler.iter(epoch).flatten().collect();
            assert_eq!(all_indices.len(), 50, "Should include all indices");

            let last_bucket_start = 32;
            let last_bucket_indices: Vec<_> = all_indices
                .iter()
                .filter(|&&x| x >= last_bucket_start)
                .copied()
                .collect();
            assert_eq!(last_bucket_indices.len(), 18);
            Ok(())
        }

        #[test]
        fn works_with_in_memory_dataset() -> Result<()> {
            let texts = vec![
                "short".to_string(),
                "medium_length".to_string(),
                "very very long sequence".to_string(),
                "tiny".to_string(),
                "another medium".to_string(),
                "tinyest".to_string(),
            ];

            let dataset = InMemoryDataset::new(texts.clone());
            let sampler = BatchBucketSampler::new(
                SequentialSampler::new(dataset.len()),
                2,                                        // batch_size
                false,                                    // drop_last
                |i| dataset.get(i).unwrap().len() as f64, // sort by string length
                2,                                        // bucket_multiplier
                TEST_SEED,
            )?;

            // Verify that each mini-batch contains similarly-sized sequences
            for batch_indices in sampler.iter(0) {
                let batch_strings: Vec<_> = batch_indices
                    .iter()
                    .map(|&i| dataset.get(i).unwrap())
                    .cloned()
                    .collect();
                let lengths: Vec<_> = batch_strings.iter().map(|s| s.len()).collect();
                let max_len = *lengths.iter().max().unwrap();
                let min_len = *lengths.iter().min().unwrap();
                assert!(
                    max_len - min_len <= 10,
                    "In-memory bucket test: batch lengths too far apart: max={}, min={}",
                    max_len,
                    min_len
                );
            }

            // Confirm that every item appeared exactly once when flattened
            let all_indices: Vec<usize> = sampler.iter(0).flatten().collect();
            assert_eq!(all_indices.len(), dataset.len());
            let unique_indices: HashSet<_> = all_indices.iter().cloned().collect();
            assert_eq!(unique_indices.len(), dataset.len());
            Ok(())
        }

        #[test]
        fn works_with_iterable_dataset_and_txt_datasource() -> Result<()> {
            let mut file = NamedTempFile::new()?;
            writeln!(file, "short")?;
            writeln!(file, "medium length line")?;
            writeln!(file, "this is a very very long line of text")?;
            writeln!(file, "another medium line")?;
            writeln!(file, "tiny")?;
            writeln!(file, "yet another reasonably sized line")?;
            let path = file.into_temp_path();

            let txt_source = TxtSource::new(&path);
            let tokenizer = Tokenizer::from_pretrained("bert-base-uncased", None)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from pretrained: {}", e))?;
            let transform = Tokenize::new(tokenizer);
            let dataset =
                IterableDataset::new(vec![Box::new(txt_source) as Box<dyn DataSource<String>>])
                    .with_transform(transform);

            let sampler = BatchBucketSampler::new(
                SequentialSampler::new(6),
                2, // batch_size
                false,
                |i| {
                    // provide a mock sort key for index i (six lines total):
                    //   0 (short)  → 4.0
                    //   1 (medium) → 15.0
                    //   2 (long)   → 30.0
                    //   3 (med)    → 14.0
                    //   4 (tiny)   → 3.0
                    //   5 (med)    → 25.0
                    match i {
                        0 => 4.0,
                        1 => 15.0,
                        2 => 30.0,
                        3 => 14.0,
                        4 => 3.0,
                        5 => 25.0,
                        _ => 0.0,
                    }
                },
                3,         // bucket_multiplier (bucket_size = 2 * 3 = 6)
                TEST_SEED, // seed
            )?;

            // Iterate through sampler.iter(0) and, for each mini-batch, pull the data
            // out of the IterableDataset, tokenizing on the fly. Then verify that within
            // each batch the tokenized lengths are close to one another.
            let all_samples: Vec<Sample> = dataset.iter().collect::<Result<_>>()?;
            let mut total_seen = 0;

            for batch_indices in sampler.iter(0) {
                let mut batch_token_lens = Vec::new();
                for &idx in &batch_indices {
                    let sample = &all_samples[idx];
                    let tensor = sample.get("input_ids").unwrap();
                    let token_len = tensor.size()[0] as usize;
                    batch_token_lens.push(token_len);
                    total_seen += 1;
                }
                let &max_tok = batch_token_lens.iter().max().unwrap();
                let &min_tok = batch_token_lens.iter().min().unwrap();
                assert!(
                    max_tok - min_tok <= 4,
                    "Line lengths in batch should be similar: max={}, min={}",
                    max_tok,
                    min_tok
                );
            }
            assert_eq!(total_seen, all_samples.len());
            Ok(())
        }
    }
}
