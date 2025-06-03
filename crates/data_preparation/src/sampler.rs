use anyhow::{ensure, Result};
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
/// - `iter(epoch)`: returns a shuffled sequence for that epoch.
///    - Users pass the `epoch` parameter so internally the sampler uses it
///      together with the base seed to shuffle in a reproducible way across epochs.
///
/// Implementations must be `Send + Sync` so the same sampler instance can be
/// safely shared across DataLoader worker threads.
pub trait Sampler: Send + Sync {
    type Item;

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
/// - `base_seed`: Master seed.
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
/// - `base_seed`: Master seed.
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
/// - `base_seed`: Master seed. See `RandomSampler` docs for details.
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
/// - `base_seed`: Master seed.
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
///     123, // base seed
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
}
