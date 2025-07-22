use data_preparation::{dataset::DataSource, sample::Sample, transforms::Transform};

use anyhow::Result;
use tch::Tensor;

/// Simple transform that converts String to Sample with length as tensor.  
pub struct StringToSample;
impl Transform<String, Sample> for StringToSample {
    fn apply(&self, input: String) -> Result<Sample> {
        let length = input.len() as i64;
        Ok(Sample::from_single("length", Tensor::from_slice(&[length])))
    }
}

/// Test data source that yields predefined strings
pub struct TestDataSource {
    pub data: Vec<String>,
}

impl DataSource<String> for TestDataSource {
    fn stream(&self) -> Result<Box<dyn Iterator<Item = Result<String>> + Send>> {
        Ok(Box::new(self.data.clone().into_iter().map(Ok)))
    }
}
