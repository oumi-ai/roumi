// src/lib.rs

//! # data_preparation Crate
//!
//! This crate provides support for managing collections of tensors, 
//! in the '.safetensors' file format. 
//!
//! ## Key Features
//! ====================================================================================
//! * `Dataset`: A basic in-memory representation (`HashMap<String, Vec<tch::Tensor>>`).
//! * `SafetensorsDataset`: A wrapper around `Dataset` providing methods (`save_to_file`, `load_from_file`)
//!     to serialize/deserialize the list-of-tensors structure to/from `.safetensors` files.
//!     Handles tensor byte conversion and metadata persistence.
//! * `DataPrepError`: Custom error type for specific and informative error handling.
//!
//! ## Example Usage
//! ====================================================================================
//! use data_preparation::{SafetensorsDataset, Result}; // Use the crate's Result type
//! use tch::{Tensor, kind, Device};
//! use std::collections::HashMap;
//! # use tempfile::NamedTempFile; // Use tempfile for a runnable doc test example
//!
//! # fn run_example() -> Result<()> { // Return your crate's Result
//! // 1. Create some tensor data (e.g., loaded from another source)
//! let key = "embeddings";
//! let device = Device::Cpu; // Or Device::cuda_if_available();
//! let tensors: Vec<Tensor> = (0..5) // Create 5 tensors
//!     .map(|i| Tensor::randn(&[1, 32], (kind::FLOAT_CPU, device)) * (i as f64 + 1.0))
//!     .collect();
//!
//! let mut data_map = HashMap::new();
//! data_map.insert(key.to_string(), tensors);
//!
//! // 2. Create the dataset (use '?' for error handling)
//! let dataset = SafetensorsDataset::from_dict(data_map)?;
//!
//! // 3. Save the dataset to a temporary file
//! let temp_file = NamedTempFile::new().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
//! let file_path = temp_file.path();
//! // In real usage: let file_path = "path/to/my_dataset.safetensors";
//!
//! println!("Saving dataset to: {:?}", file_path);
//! dataset.save_to_file(file_path)?;
//! println!("Dataset saved successfully.");
//!
//! // 4. Load the dataset back
//! let loaded_dataset = SafetensorsDataset::load_from_file(file_path)?;
//! println!("Dataset loaded successfully.");
//!
//! // 5. Verify contents
//! assert_eq!(dataset.len(), loaded_dataset.len());
//! assert!(loaded_dataset.keys().contains(&key.to_string()));
//! let original_tensors = dataset.get_tensors(key).unwrap();
//! let loaded_tensors = loaded_dataset.get_tensors(key).unwrap();
//! assert_eq!(loaded_tensors.len(), original_tensors.len());
//!
//! // Optional: Check tensor equality (might require CPU transfer if on GPU - not yet implemented)
//! for (orig, loaded) in original_tensors.iter().zip(loaded_tensors.iter()) {
//!    assert!(orig.allclose(loaded, 1e-6, 1e-6, false));
//! }
//! println!("Dataset verification successful.");
//!
//! Ok(())
//! # }
//! # fn main() { run_example().expect("Example failed"); } // Simple runner for doc test
//!
//! ## Modules
//! ====================================================================================
//! - `dataset`: Contains the core `Dataset` struct definition.
//! - `safetensors_dataset`: Contains the `SafetensorsDataset` struct and its
//!   associated save/load logic for the Safetensors format.
//! - `error`: Defines the crate's custom error type `DataPrepError` and `Result` alias.


// Declare modules
mod dataset; 
mod safetensors_dataset; 
mod error; 

pub use dataset::Dataset; 
pub use safetensors_dataset::SafetensorsDataset;
pub use safetensors_dataset::TensorLayout;
pub use error::{DataPrepError, Result};