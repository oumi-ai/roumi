use crate::minibatch::MiniBatch;
use crate::sample::Sample;
use anyhow::{anyhow, bail, Result};
use std::collections::{HashMap, HashSet};
use tch::Tensor;

/// A `Collator` defines how to pad and combine multiple [`Sample`]s into a [`MiniBatch`].
pub trait Collator {
    fn collate(&self, samples: Vec<Sample>) -> Result<MiniBatch>;
}

/// A `Collator` that simply stacks tensors with identical shapes.
/// along the batch dimension (dim 0). It does not implement any
/// padding logic here, so if any sample has inconsistent shape,
/// an error is returned.
#[derive(Debug)]
pub struct StackCollator;

impl Collator for StackCollator {
    fn collate(&self, samples: Vec<Sample>) -> Result<MiniBatch> {
        if samples.is_empty() {
            bail!("Cannot collate empty sample list");
        }

        // Validate feature keys
        let first_keys: HashSet<&String> = samples[0].features.keys().collect();
        for sample in &samples {
            if sample.features.keys().collect::<HashSet<_>>() != first_keys {
                bail!("Mismatched feature keys across samples");
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
