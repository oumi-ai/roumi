use crate::sample::Sample;
use anyhow::{anyhow, bail, Result};
use std::collections::{HashMap, HashSet};
use tch::{Device, Tensor};

/// The `MiniBatch` struct represents a batch of data examples grouped for model input.
///
/// It is constructed by stacking multiple [`Sample`]s together along the
/// batch dimension (dim 0). Internally, it holds a map from feature names
/// (e.g., `"input_ids"`, `"labels"`) to batched tensors.
///
/// Each tensor in the map has shape `[batch_size, ...]`, where:
/// - `batch_size` = number of samples in the batch
/// - Remaining dimensions must match across all samples for stacking.
///
/// # Examples
/// Suppose we have 4 `Samples` and each 'Sample' contains the following features:
/// - `"input_ids"` -> shape `[128]` (tokenized sequence)
/// - `"pixel_values"` -> shape `[3, 224, 224]` (RGB image)
///
/// Then the resulting `MiniBatch` will contain:
/// - `"input_ids"` -> shape `[4, 128]`
/// - `"pixel_values"` -> shape `[4, 3, 224, 224]`
#[derive(Debug)]
pub struct MiniBatch {
    tensors: HashMap<String, Tensor>,
}

impl MiniBatch {
    /// Constructs a `MiniBatch` from a list of individual [`Sample`]s.
    ///
    /// This stacks the tensors from each sample along the batch dimension,
    /// producing one batched tensor per feature.
    ///
    /// # Validation checks:
    /// - All `Sample`s must contain the same set of feature keys.
    /// - For each feature key, the associated tensors must have the same
    ///   shape across samples to be stackable.
    pub fn from_samples(samples: Vec<Sample>) -> Result<Self> {
        if samples.is_empty() {
            return Err(anyhow!("Cannot create mini-batch from empty sample list"));
        }

        // Use the first sample to extract the expected set of feature keys
        let expected_keys: HashSet<&String> = samples[0].features.keys().collect();

        // Ensure all samples have the same keys
        if samples[1..].iter().any(|s| {
            s.features.len() != expected_keys.len()
                || !s.features.keys().all(|k| expected_keys.contains(k))
        }) {
            bail!("Mismatched feature keys across samples");
        }

        // Stack tensors for each feature
        let mut tensors = HashMap::with_capacity(expected_keys.len());
        for key in expected_keys {
            // Gather tensor references for this feature across all samples
            let tensors_to_stack: Vec<&Tensor> = samples
                .iter()
                .map(|s| s.features.get(key).expect("Validated key"))
                .collect();

            // Validate that tensor shapes are compatabile for stacking
            let reference_shape = tensors_to_stack[0].size();
            for (i, tensor) in tensors_to_stack.iter().enumerate() {
                if tensor.size() != reference_shape {
                    bail!(
                        "Shape mismatch in sample {} for feature '{}': expected {:?}, got {:?}",
                        i,
                        key,
                        reference_shape,
                        tensor.size()
                    );
                }
            }

            // Stack along dimension 0 to form the batched tensor.
            // Shape validation check above ensures that this call is safe.
            let stacked = Tensor::stack(&tensors_to_stack, 0);
            tensors.insert(key.clone(), stacked);
        }
        Ok(Self { tensors })
    }

    /// Returns the number of samples in the batch.
    pub fn batch_size(&self) -> Result<i64> {
        self.tensors
            .values()
            .next()
            .map(|t| t.size()[0])
            .ok_or(anyhow!("Empty mini-batch"))
    }

    /// Returns a reference to the tensor for a given feature key.
    pub fn get(&self, feature: &str) -> Result<&Tensor> {
        self.tensors
            .get(feature)
            .ok_or_else(|| anyhow!("Feature '{}' not found in mini-batch", feature))
    }

    /// Returns an iterator over all feature keys in the batch.
    pub fn features(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Transfers all tensors to the target device (CPU/GPU)
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            tensors: self
                .tensors
                .iter()
                .map(|(feature_name, tensor)| (feature_name.clone(), tensor.to_device(device)))
                .collect(),
        }
    }
}

#[cfg(test)]
mod minibatch_test {
    use super::*;
    use anyhow::Result;
    use tch::{Device, Kind, Tensor};

    /// Helper function: Creates a sample with predictable values
    fn make_sample(value: i64) -> Sample {
        Sample::from_single(
            "input_ids",
            Tensor::from_slice(&[value]).to_kind(Kind::Int64),
        )
        .with_feature(
            "labels",
            Tensor::from_slice(&[value % 2]).to_kind(Kind::Int64),
        )
        .with_feature("mask", Tensor::ones(&[1], (Kind::Float, Device::Cpu)))
    }

    #[test]
    fn test_minibatch_from_samples() -> Result<()> {
        let samples = vec![make_sample(1), make_sample(2), make_sample(3)];
        let batch = MiniBatch::from_samples(samples)?;

        assert_eq!(batch.batch_size()?, 3);

        // Check all features are batched correctly
        for feature in batch.features() {
            assert_eq!(batch.get(feature)?.size(), &[3, 1]);
        }

        // Check correct values
        let labels: Vec<i64> = batch.get("labels")?.squeeze_dim(1).try_into()?;
        assert_eq!(labels, vec![1, 0, 1]);
        Ok(())
    }

    #[test]
    fn test_minibatch_to_device() -> Result<()> {
        let cpu_batch = MiniBatch::from_samples(vec![make_sample(9), make_sample(10)])?;
        let target_device = Device::cuda_if_available();
        let moved_batch = cpu_batch.to_device(target_device);

        for feature in moved_batch.features() {
            assert_eq!(moved_batch.get(feature)?.device(), target_device);
            assert_eq!(cpu_batch.get(feature)?.device(), Device::Cpu);
        }
        Ok(())
    }

    #[test]
    fn test_minibatch_shape_mismatch() {
        let empty = MiniBatch::from_samples(vec![]);
        assert!(empty.is_err());

        let s1 = Sample::from_single("input_ids", Tensor::zeros(&[2], (Kind::Float, Device::Cpu)));
        let s2 = Sample::from_single("input_ids", Tensor::zeros(&[3], (Kind::Float, Device::Cpu)));
        let result = MiniBatch::from_samples(vec![s1, s2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_minibatch_key_mismatch() {
        let s1 = Sample::from_single("input_ids", Tensor::from_slice(&[1]));
        let s2 = Sample::from_single("labels", Tensor::from_slice(&[0]));
        let result = MiniBatch::from_samples(vec![s1, s2]);
        assert!(result.is_err());
    }
}
