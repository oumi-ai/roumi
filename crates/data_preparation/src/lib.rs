// src/lib.rs

// Declare modules
mod dataset; 
mod safetensors_dataset; 
mod error; 

pub use dataset::Dataset; 
pub use safetensors_dataset::SafetensorsDataset;
pub use error::{DataPrepError, Result};