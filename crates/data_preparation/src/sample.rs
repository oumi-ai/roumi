use anyhow::{anyhow, Context, Result};
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

/// Creates a shallow clone of the `Sample`
impl Clone for Sample {
    fn clone(&self) -> Self {
        let features = self
            .features
            .iter()
            .map(|(k, v)| (k.clone(), v.shallow_clone()))
            .collect();
        Self { features }
    }
}

/// Safety:
/// The `unsafe impl` here indicates we manually verified thread-safety conditions.
///
/// - The `Send` implementation is safe because:
/// 1. `tch::Tensor` is marked as `Send` in its source (see [tensor.rs])
/// 2. `HashMap<String, Tensor>` composes only `Send` types:
///    * `String` is `Send` (standard library guarantee)
///    * `Tensor` is `Send` (as verified in tch-rs source)
///    * `HashMap` is `Send` when its key/value are `Send`
///
/// - The `Sync` implementation is safe because:
/// 1. `tch::Tensor` is marked as `Sync` in its source
/// 2. All operations require `&mut self` for mutation
/// 3. Immutable references allow concurrent reads.
///
/// [tensor.rs]: https://docs.rs/tch/latest/src/tch/wrappers/tensor.rs.html
unsafe impl Send for Sample {}
unsafe impl Sync for Sample {}

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

    /// Merges two samples, returning an error if any feature names collide.
    #[inline]
    pub fn merge(self, other: Sample) -> Result<Self> {
        let mut features = self.features;
        features.reserve(other.features.len()); // Pre-allocate for merge

        for (key, tensor) in other.features {
            if let Some(_existing) = features.insert(key.clone(), tensor) {
                return Err(anyhow::anyhow!("Duplicate feature: {}", key))
                    .context("Failed merging samples");
            }
        }
        Ok(Self { features })
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
