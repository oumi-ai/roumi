pub mod core;
pub mod multimodal;
pub mod text;
pub mod vision;

pub use core::Transform;
pub use multimodal::MultimodalPipeline;

/// ===========================================================================
use crate::sample::Sample;
use anyhow::Result;
use tch::Tensor;
/// Converts a tensor to a `Sample` with specified feature name
#[derive(Debug)]
pub struct ToSample {
    feature_name: String,
}

impl ToSample {
    pub fn new(feature_name: impl Into<String>) -> Self {
        Self {
            feature_name: feature_name.into(),
        }
    }
}

impl Transform<Tensor, Sample> for ToSample {
    fn apply(&self, tensor: Tensor) -> Result<Sample> {
        Ok(Sample::from_single(&self.feature_name, tensor))
    }
}
