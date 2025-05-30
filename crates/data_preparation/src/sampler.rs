use anyhow::{ensure, Result};
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, Rng, SeedableRng};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::InMemoryDataset;
    use crate::sample::Sample;
    use tch::Tensor;

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

        const TEST_SEED: u64 = 42;
        const TEST_DATASET_SIZE: usize = 100;

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
}
