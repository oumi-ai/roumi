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
}
