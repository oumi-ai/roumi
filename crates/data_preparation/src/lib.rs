use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use tch::Tensor;

/// The `Sample` struct represents a single data example in a machine learning pipeline.
///
/// It contains a mapping from feature names (e.g., `"input_ids"`, `"labels"`)
/// to their corresponding tensor values.
///
/// Internally, the `features` map stores:
/// - **Keys**(`String`): Feature names
/// - **Values**(`Tensor`): The data tensors associated with each feature
///
/// # Examples:
/// - For a text sample: `{"input_ids": Tensor([1, 32, 128]), "attention_mask": Tensor([1, 1, 0]), "labels": Tensor([0])}`
/// - For an image sample: `{"pixel_values": Tensor([3, 224, 224]), "labels": Tensor([5])}`
#[derive(Debug)]
pub struct Sample {
    pub features: HashMap<String, Tensor>,
}

impl Sample {
    /// Creates a new `Sample` from a full feature map.
    ///
    /// This constructor is intended for use cases where the full `HashMap<String, Tensor>`
    /// is already available. It does not perform any conversions - callers are responsible
    /// for ensuring keys(feature names) are `String`.
    pub fn new(features: HashMap<String, Tensor>) -> Self {
        Self { features }
    }

    /// Creates a `Sample` from a single `(feature_name, tensor)` pair.
    ///
    /// This is a convenience constructor for simple samples (e.g., inference with one input).
    /// Accepts both `&str` and `String` for the feature name via `Into<String>`.
    ///
    /// Chain with [`with_feature`](Self::with_feature) to add more features.
    pub fn from_single(name: impl Into<String>, tensor: Tensor) -> Self {
        Self {
            features: HashMap::from([(name.into(), tensor)]),
        }
    }

    /// Adds or overwrites a feature in the `Sample`.
    pub fn with_feature(mut self, name: impl Into<String>, tensor: Tensor) -> Self {
        self.features.insert(name.into(), tensor);
        self
    }

    /// Returns a reference to the tensor by feature name.
    pub fn get(&self, feature: &str) -> Result<&Tensor> {
        self.features
            .get(feature)
            .ok_or_else(|| anyhow!("Feature {} not found", feature))
    }

    /// Returns an iterator over all feature names in this `Sample`.
    pub fn features(&self) -> impl Iterator<Item = &str> {
        self.features.keys().map(String::as_str)
    }
}

#[cfg(test)]
mod sample_test {
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
    fn test_sample_basic_construction() -> Result<()> {
        let sample = make_sample(42);

        assert_eq!(sample.get("input_ids")?.int64_value(&[0]), 42);
        assert_eq!(sample.get("labels")?.int64_value(&[0]), 0);
        assert!(sample.get("missing").is_err());

        let features: Vec<_> = sample.features().collect();
        assert!(features.contains(&"input_ids"));
        assert!(features.contains(&"labels"));
        assert!(features.contains(&"mask"));
        Ok(())
    }
}

/// The `MiniBatch` struct serves as a container for machine learning data,
/// storing a full batch of samples as tensors.
///
/// Data is organized in a `HashMap` where:
/// - Keys are feature names (e.g., "tokens", "labels")
/// - Values are tensors with shape `[batch_size,...]`
///
/// For example:
/// ```text
/// Batch of 4 samples:
/// {
///     "tokens": Tensor([4, 128])   //4 sequences, 128 tokens each
///     "labels": Tensor([4])       //4 corresponding labels
/// }
/// ```
///
/// **Notes:**
/// 1. All tensors must share the same `batch_size` (first dimension).
/// 2. For single examples, use `batch_size = 1`.
/// 3. The batch size here refers to the total samples stored. During
///    training/inference, this batch can be split into smaller mini-
///    batches. *(Implementation note: Mini-batch splitting for future work.)*
#[derive(Debug)]
pub struct MiniBatch {
    tensors: HashMap<String, Tensor>,
}

impl MiniBatch {
    /// Creates a new MiniBatch from a HashMap of tensors.
    /// Validates that:
    /// - No tensors are scalar (must have at least one dimension).
    /// - All tensors share the same size for the first dimension (batch size).
    pub fn new(tensors: HashMap<String, Tensor>) -> Result<Self> {
        if tensors.is_empty() {
            return Ok(MiniBatch { tensors });
        }

        // Check for scalars
        let batch_sizes: Vec<usize> = tensors
            .iter()
            .map(|(key, t)| {
                let size = t.size();
                if size.is_empty() {
                    bail!(
                        "Scalar tensor '{}' not allowed; tensors must have a batch dimension (e.g., [batch_size], [batch_size, seq_len])",
                        key
                    )
                } else {
                    Ok(size[0] as usize)
                }
            })
            .collect::<Result<Vec<usize>>>()?;

        // Ensure the batch size is consistent
        let first_batch_size = batch_sizes[0];
        if !batch_sizes.iter().all(|&size| size == first_batch_size) {
            bail!(
                "Inconsistent batch sizes: expected {}, found {:?}",
                first_batch_size,
                batch_sizes
            );
        }

        Ok(MiniBatch { tensors })
    }

    /// Returns the number of samples in the batch.
    pub fn len(&self) -> usize {
        self.tensors
            .values()
            .next()
            .map_or(0, |t| t.size()[0] as usize)
    }

    /// Checks if the MiniBatch contains no tensors
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Returns an immutable reference to a tensor by key, if it exists.
    pub fn get(&self, key: &str) -> Option<&Tensor> {
        self.tensors.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Kind, Tensor};

    #[test]
    fn test_accept_1d_batch_tensor() {
        let tensor = Tensor::from_slice(&[1, 2, 3, 4]).to_kind(Kind::Int);
        let mut data_map = HashMap::new();
        data_map.insert("tensor".to_string(), tensor);

        match MiniBatch::new(data_map) {
            Ok(_mini_batch) => println!("Successfully created MiniBatch for 1D data."),
            Err(e) => panic!("Error creating MiniBatch for 1D data: {}", e),
        }
    }

    #[test]
    fn test_accept_padded_variable_length_tensors() {
        let max_length = 5;
        let sequences = vec![
            vec![1, 2, 3],
            vec![4, 5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14],
        ];

        let mut padded_data = Vec::new();
        for seq in sequences {
            let mut padded_seq = seq;
            padded_seq.extend(vec![0; max_length - padded_seq.len()]);
            padded_data.extend(padded_seq);
        }

        let tensor = Tensor::from_slice(&padded_data)
            .reshape(&[4, max_length as i64])
            .to_kind(Kind::Int);

        let mut data_map = HashMap::new();
        data_map.insert("tensor".to_string(), tensor);

        match MiniBatch::new(data_map) {
            Ok(_mini_batch) => {
                println!("Successfully created MiniBatch for variable-length data after applying zero-padding.");
                println!("MiniBatch: {:?}", _mini_batch);
            }
            Err(e) => {
                panic!(
                    "Error creating MiniBatch for variable-length data even after zero-padding: {}",
                    e
                );
            }
        }
    }

    #[test]
    fn test_reject_unpadded_variable_length_tensors() {
        let sequences = vec![
            vec![1, 2, 3],
            vec![4, 5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14],
        ];

        let tensors: Vec<Tensor> = sequences
            .into_iter()
            .map(|seq| Tensor::from_slice(&seq).to_kind(Kind::Int))
            .collect();

        let result = Tensor::f_stack(&tensors, 0); // Should fail.

        match result {
            Ok(t) => {
                let mut data_map = HashMap::new();
                data_map.insert("tokens".to_string(), t);
                match MiniBatch::new(data_map) {
                    Ok(_) => panic!("Expected MiniBatch::new to fail due to inconsistent shapes, but it succeeded."),
                    Err(e) => println!("Correctly rejected by MiniBatch::new: {}", e),
                }
            }
            Err(e) => {
                println!("Correctly failed to stack variable-length tensors: {}", e);
            }
        }
    }
}
