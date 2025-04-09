// src/info.rs
use std::collections::HashMap; 
use tch::Kind; 

/// Describes the layout consistency of tensors associated with a specific key 
/// within a `Dataset`
#[derive(Debug, PartialEq)]
pub enum TensorLayout {
    // All tensors have the same shape and dtype. 
    Standard {
        shape: Vec<i64>, 
        dtype: Kind,
    },
    // Tensors have varying shapes but the same dtype 
    VaryingDimSize {
        dtype: Kind, 
    },
    // Tensors have varying dtypes (and possibly varying shapes).
    VaryingDtype, 
}

/// Contains metadata describing the overall structure of a `SafetensorsDataset`
/// This is returned by the [`SafetensorsDataset::info()`] method. 
#[derive(Debug, PartialEq)]
pub struct DatasetInfo{
    // The number of rows in the dataset. 
    pub len: usize, 
    // A map where keys are the dataset's string keys, and values describe
    // the [`TensorLayout`] for the tensors associated with that key. 
    pub layouts: HashMap<String, TensorLayout>, 
}