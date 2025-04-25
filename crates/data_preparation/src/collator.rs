use crate::minibatch::MiniBatch;
use crate::sample::Sample;
use anyhow::{bail, Result};
use std::collections::{HashMap, HashSet};
use tch::Tensor;

/// A `Collator` defines how to pad and combine multiple [`Sample`]s into a [`MiniBatch`].
pub trait Collator {
    fn collate(&self, samples: &[Sample]) -> Result<MiniBatch>;
}

/// A `Collator` that simply stacks tensors with identical shapes.
/// along the batch dimension (dim 0). It does not implement any
/// padding logic here, so if any sample has inconsistent shape,
/// an error is returned.
#[derive(Debug)]
pub struct StackCollator;

impl Collator for StackCollator {
    fn collate(&self, samples: &[Sample]) -> Result<MiniBatch> {
        if samples.is_empty() {
            bail!("Cannot collate empty sample list");
        }

        // Validate feature keys
        let first_keys: HashSet<&String> = samples[0].features.keys().collect();
        for (i, sample) in samples.iter().enumerate().skip(1) {
            let missing_keys: Vec<&String> = first_keys
                .iter()
                .filter(|&&k| !sample.features.contains_key(k))
                .cloned()
                .collect();

            let extra_keys: Vec<&String> = sample
                .features
                .keys()
                .filter(|k| !first_keys.contains(k))
                .collect();

            if !missing_keys.is_empty() || !extra_keys.is_empty() {
                bail!(
                    "Sample #{} has mismatch feature keys:\n -Missing: {:?}\n -Extra: {:?}",
                    i,
                    missing_keys,
                    extra_keys
                )
            }
        }

        // Stack tensors for each feature
        let mut tensors = HashMap::with_capacity(first_keys.len());
        for key in first_keys {
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
        Ok(MiniBatch { tensors })
    }
}

//=======================================================================================================
/// Defines how a tensor should be padded across a batch
#[derive(Debug)]
pub enum PaddingRule {
    // Pad to maximum size of batch
    MaxLength,
    // Pad (or truncate) to the right to exactly this size
    FixedRight(i64),
    // Pad to the left to exactly this size
    FixedLeft(i64),
    // Center-pad to reach exactly this size.
    // For odd padding amounts, the extra unit goes to the right/bottom.
    Symmetric(i64),
}

/// A `Collator` that pads variable-length tensors across multiple
/// [`Sample`]s so they can be stacked into a [`MiniBatch`].
///
/// Padding rules are configured per-feature and per-dimension using
/// the `.pad(...)` method. You can optionally specify a padding
/// value for each feature, or rely on the default (0.0).
///
/// Any feature that is not explicitly configured for padding must
/// already have the same shape across all samples, otherwise the
/// collation will return a shape mismatch error.
///
/// # Example
/// ```ignore
/// let collator = PaddingCollator::new()
///     
///     //MaxLength padding on dimension 0, with default pad_value = 0.0
///     .pad("input_ids", vec![(0, PaddingRule::MaxLength)], None)    
///
///     //fixed length padding to the right on dimension 1 to 22, with custom pad_value = -100.0
///     .pad("labels", vec![(1, PaddingRule::FixedRight(22))], Some(-100.0))   
/// ```
#[derive(Debug)]
pub struct PaddingCollator {
    // Per-feature padding configuration:
    // maps feature names to list of `(dimension, padding rule)` pairs
    pad_config: HashMap<String, Vec<(usize, PaddingRule)>>,

    // Optional per-feature pad values (if not set, defaults to 0.0)
    pad_values: HashMap<String, f64>,
}

impl PaddingCollator {
    /// Creates a new `PaddingCollator`
    pub fn new() -> Self {
        Self {
            pad_config: HashMap::new(),
            pad_values: HashMap::new(),
        }
    }

    /// Registers padding rules for a given feature
    ///
    /// # Arguments:
    /// - `feature`: feature name (e.g., `"input_ids"`)
    /// - `rules`: list of `(dimension, target_length)`:
    ///     - `PaddingRule::MaxLength` = pad to maximum size in the batch (right-side/bottom-side)
    ///     - `PaddingRule::FixedRight(n)` = pad to a fixed length `n` (right-side/bottom-side)
    ///     - `PaddingRule::FixedLeft(n)` = pad to a fixed length `n` (left-side/topside)
    ///     -`PaddingRule::Symmetric(n)` = pad to a fixed length `n` with equal padding on both sides.
    /// - `pad_value`: optional padding value (defaults to 0.0)
    pub fn pad<I>(mut self, feature: impl ToString, rules: I, pad_value: Option<f64>) -> Self
    where
        I: IntoIterator<Item = (usize, PaddingRule)>,
    {
        let key = feature.to_string();
        self.pad_config
            .entry(key.clone())
            .or_default()
            .extend(rules);

        if let Some(v) = pad_value {
            self.pad_values.insert(key, v);
        }
        self
    }

    fn compute_target_shape(
        &self,
        tensors: &[&Tensor],
        rules: &[(usize, PaddingRule)],
    ) -> Result<Vec<i64>> {
        let mut target = tensors[0].size().to_vec();
        for &(dim, ref rule) in rules {
            if dim >= target.len() {
                bail!(
                    "Invalid padding dimension {} for tensor with {} dims",
                    dim,
                    target.len()
                );
            }
            target[dim] = match rule {
                PaddingRule::MaxLength => tensors.iter().map(|t| t.size()[dim]).max().unwrap_or(0),
                PaddingRule::FixedRight(n)
                | PaddingRule::FixedLeft(n)
                | PaddingRule::Symmetric(n) => *n,
            };
        }
        Ok(target)
    }

    fn pad_single(
        &self,
        tensor: &Tensor,
        target_shape: &[i64],
        rules: &[(usize, PaddingRule)],
        pad_value: f64,
    ) -> Result<Tensor> {
        let mut result = tensor.shallow_clone();
        for &(dim, ref rule) in rules {
            let dim = dim as i64;
            let current_len = result.size()[dim as usize];
            let target_len = target_shape[dim as usize];

            match current_len.cmp(&target_len) {
                std::cmp::Ordering::Less => {
                    let pad_total = target_len - current_len;
                    let (pad_before, pad_after) = match rule {
                        PaddingRule::Symmetric(_) => (pad_total / 2, pad_total - pad_total / 2),
                        PaddingRule::FixedLeft(_) => (pad_total, 0),
                        _ => (0, pad_total),
                    };

                    if pad_before > 0 || pad_after > 0 {
                        let mut new_shape = result.size();
                        new_shape[dim as usize] = target_len;
                        let padded =
                            Tensor::full(&new_shape, pad_value, (result.kind(), result.device()));

                        // Calculate copy ranges
                        let copy_start = pad_before as i64;
                        let copy_len = current_len as i64;

                        let source_view = result.narrow(dim, 0, copy_len);
                        let mut dest_view = padded.narrow(dim, copy_start, copy_len);
                        dest_view.copy_(&source_view);
                        result = padded;
                    }
                }
                std::cmp::Ordering::Greater => {
                    result = result.narrow(dim, 0, target_len);
                }
                std::cmp::Ordering::Equal => {}
            }
        }
        Ok(result)
    }
}

/// Collate [`Sample`]s into a [`MiniBatch`], applying padding as configured per feature
impl Collator for PaddingCollator {
    fn collate(&self, samples: &[Sample]) -> Result<MiniBatch> {
        if samples.is_empty() {
            bail!("Cannot collate empty sample list");
        }

        // Validate feature keys
        let first_keys: HashSet<&String> = samples[0].features.keys().collect();
        for (i, sample) in samples.iter().enumerate().skip(1) {
            let missing_keys: Vec<&String> = first_keys
                .iter()
                .filter(|&&k| !sample.features.contains_key(k))
                .cloned()
                .collect();

            let extra_keys: Vec<&String> = sample
                .features
                .keys()
                .filter(|k| !first_keys.contains(k))
                .collect();

            if !missing_keys.is_empty() || !extra_keys.is_empty() {
                bail!(
                    "Sample #{} has mismatch feature keys:\n -Missing: {:?}\n -Extra: {:?}",
                    i,
                    missing_keys,
                    extra_keys
                )
            }
        }

        let mut batched = HashMap::with_capacity(first_keys.len());
        for key in first_keys {
            let tensors: Vec<&Tensor> = samples.iter().map(|s| &s.features[key]).collect();

            if let Some(rules) = self.pad_config.get(key) {
                // 1. Compute target shape
                let target_shape = self.compute_target_shape(&tensors, rules)?;
                let pad_value = *self.pad_values.get(key).unwrap_or(&0.0);

                // 2. Pre-allocate batch tensor
                let batch_tensor = Tensor::empty(
                    &{
                        let mut shape = vec![tensors.len() as i64];
                        shape.extend(target_shape.iter().copied());
                        shape
                    },
                    (tensors[0].kind(), tensors[0].device()),
                );

                // 3. Pad and copy
                for (i, t) in tensors.into_iter().enumerate() {
                    let padded = self.pad_single(t, &target_shape, rules, pad_value)?;
                    batch_tensor.select(0, i as i64).copy_(&padded);
                }
                batched.insert(key.clone(), batch_tensor);
            } else {
                // Stack-only
                let reference_shape = tensors[0].size();
                if tensors.iter().any(|t| t.size() != reference_shape) {
                    bail!("Shape mismatch for non-padded feature '{}'", key);
                }
                batched.insert(key.clone(), Tensor::stack(&tensors, 0));
            }
        }
        Ok(MiniBatch { tensors: batched })
    }
}

#[cfg(test)]
mod paddingcollator_tests {
    use super::*;
    use crate::minibatch::MiniBatch;
    use crate::sample::Sample;
    use anyhow::Result;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn test_padding_collator_dynamic_1d() -> Result<()> {
        let samples = vec![
            Sample::from_single("input_ids", Tensor::from_slice(&[1, 2, 3])),
            Sample::from_single("input_ids", Tensor::from_slice(&[4, 5])),
        ];
        let collator =
            PaddingCollator::new().pad("input_ids", vec![(0, PaddingRule::MaxLength)], None);
        let batch = MiniBatch::collate(samples, collator)?;
        let input_ids = batch.get("input_ids")?;
        assert_eq!(input_ids.size(), &[2, 3]);

        let expected = Tensor::from_slice(&[1, 2, 3, 4, 5, 0]).reshape(&[2, 3]);
        assert!(input_ids.equal(&expected));
        Ok(())
    }

    #[test]
    fn test_padding_collator_multi_feature_1d() -> Result<()> {
        let s1 = Sample::from_single("input_ids", Tensor::from_slice(&[1, 2, 3]))
            .with_feature("attention_mask", Tensor::from_slice(&[1, 1, 1]));
        let s2 = Sample::from_single("input_ids", Tensor::from_slice(&[4, 5]))
            .with_feature("attention_mask", Tensor::from_slice(&[1, 1]));

        let collator = PaddingCollator::new()
            .pad("input_ids", vec![(0, PaddingRule::MaxLength)], None)
            .pad("attention_mask", vec![(0, PaddingRule::MaxLength)], None);

        let batch = MiniBatch::collate(vec![s1, s2], collator)?;
        let input_ids = batch.get("input_ids")?;
        let attention_mask = batch.get("attention_mask")?;

        assert_eq!(input_ids.size(), &[2, 3]);
        assert_eq!(attention_mask.size(), &[2, 3]);

        // Value assertions
        let expected_input_ids = Tensor::from_slice(&[1, 2, 3, 4, 5, 0]).reshape(&[2, 3]);
        assert!(input_ids.equal(&expected_input_ids));

        let expected_attention_mask = Tensor::from_slice(&[1, 1, 1, 1, 1, 0]).reshape(&[2, 3]);
        assert!(attention_mask.equal(&expected_attention_mask));
        Ok(())
    }

    #[test]
    fn test_padding_collator_fixed_right_3d() -> Result<()> {
        let img1 = Tensor::zeros(&[3, 28, 30], (Kind::Float, Device::Cpu));
        let img2 = Tensor::ones(&[3, 32, 32], (Kind::Float, Device::Cpu));
        let samples = vec![
            Sample::from_single("pixel_values", img1),
            Sample::from_single("pixel_values", img2),
        ];
        let collator = PaddingCollator::new().pad(
            "pixel_values",
            vec![
                (1, PaddingRule::FixedRight(32)),
                (2, PaddingRule::FixedRight(32)),
            ],
            Some(0.0),
        );
        let batch = MiniBatch::collate(samples, collator)?;
        let pixel_values = batch.get("pixel_values")?;
        assert_eq!(pixel_values.size(), &[2, 3, 32, 32]);
        Ok(())
    }

    #[test]
    fn test_padding_collator_fixed_right_padding_content() -> Result<()> {
        // Single channel 2x2 image with distinct values
        // [ [1, 2],
        //   [3, 4]]
        let img = Tensor::from_slice(&[1i64, 2, 3, 4]).reshape(&[1, 2, 2]);
        let samples = vec![Sample::from_single("img", img)];
        let collator = PaddingCollator::new().pad(
            "img",
            vec![
                (1, PaddingRule::FixedRight(4)),
                (2, PaddingRule::FixedRight(4)),
            ],
            Some(0.0),
        );
        let batch = MiniBatch::collate(samples, collator)?;
        let output = batch.get("img")?;
        assert_eq!(output.size(), &[1, 1, 4, 4]);

        let expected = Tensor::from_slice(&[1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            .reshape(&[1, 1, 4, 4]);
        assert!(output.equal(&expected));
        Ok(())
    }

    #[test]
    fn test_padding_collator_fixed_left_padding_content() -> Result<()> {
        let img = Tensor::from_slice(&[1i64, 2, 3, 4]).reshape(&[1, 2, 2]);
        let samples = vec![Sample::from_single("img", img)];

        let collator = PaddingCollator::new().pad(
            "img",
            vec![
                (1, PaddingRule::FixedLeft(4)),
                (2, PaddingRule::FixedLeft(4)),
            ],
            Some(0.0),
        );
        let batch = MiniBatch::collate(samples, collator)?;
        let output = batch.get("img")?;
        assert_eq!(output.size(), &[1, 1, 4, 4]);

        let expected = Tensor::from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4])
            .reshape(&[1, 1, 4, 4]);
        assert!(output.equal(&expected));
        Ok(())
    }

    #[test]
    fn test_padding_collator_symmetric_padding_content() -> Result<()> {
        let img = Tensor::from_slice(&[1i64, 2, 3, 4]).reshape(&[1, 2, 2]);
        let samples = vec![Sample::from_single("img", img)];

        let collator = PaddingCollator::new().pad(
            "img",
            vec![
                (1, PaddingRule::Symmetric(4)),
                (2, PaddingRule::Symmetric(4)),
            ],
            Some(0.0),
        );
        let batch = MiniBatch::collate(samples, collator)?;
        let output = batch.get("img")?;
        assert_eq!(output.size(), &[1, 1, 4, 4]);

        let expected = Tensor::from_slice(&[0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0])
            .reshape(&[1, 1, 4, 4]);
        assert!(output.equal(&expected));
        Ok(())
    }

    #[test]
    fn test_padding_collator_invalid_dim() {
        let samples = vec![Sample::from_single(
            "input_ids",
            Tensor::from_slice(&[1, 2]),
        )];
        let collator =
            PaddingCollator::new().pad("input_ids", vec![(1, PaddingRule::MaxLength)], None);
        assert!(MiniBatch::collate(samples, collator).is_err());
    }

    #[test]
    fn test_padding_collator_shape_mismatch_unpadded() {
        let samples = vec![
            Sample::from_single("labels", Tensor::from_slice(&[1])),
            Sample::from_single("labels", Tensor::from_slice(&[2, 3])),
        ];
        let collator = PaddingCollator::new();
        assert!(MiniBatch::collate(samples, collator).is_err());
    }
}
