// src/error.rs
use safetensors; // Assuming safetensors crate is accessible
use serde_json;
use std::{fmt, io};
use tch; // Assuming tch crate is accessible for potential tensor errors

#[derive(Debug)]
pub enum DataPrepError {
    Io(io::Error),
    SerdeJson(serde_json::Error),
    Safetensor(safetensors::SafeTensorError),
    Tch(tch::TchError), // If needed for tensor creation errors later
    InvalidKey(String),
    InconsistentTensorList(String), // For mixed dtypes or shapes if checked
    MetadataNotFound(String),
    MetadataFormat(String), // If metadata parsing fails
    UnsupportedDtype(String),
    FileFormat(String), // Generic safetensor format errors (e.g., too short)
    Other(String),      // Catch-all
}

impl fmt::Display for DataPrepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataPrepError::Io(e) => write!(f, "I/O error: {}", e),
            DataPrepError::SerdeJson(e) => {
                write!(f, "JSON serialization/deserialization error: {}", e)
            }
            DataPrepError::Safetensor(e) => write!(f, "Safetensor error: {}", e),
            DataPrepError::Tch(e) => write!(f, "LibTorch error: {}", e),
            DataPrepError::InvalidKey(e) => write!(f, "Invalid key: {}", e),
            DataPrepError::InconsistentTensorList(e) => {
                write!(f, "Inconsistent tensor list: {}", e)
            }
            DataPrepError::MetadataNotFound(e) => write!(f, "Metadata not found: {}", e),
            DataPrepError::MetadataFormat(e) => write!(f, "Metadata format error: {}", e),
            DataPrepError::UnsupportedDtype(e) => write!(f, "Unsupported dtype: {}", e),
            DataPrepError::FileFormat(e) => write!(f, "Invalid file format: {}", e),
            DataPrepError::Other(e) => write!(f, "An error occurred: {}", e),
        }
    }
}

impl std::error::Error for DataPrepError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DataPrepError::Io(e) => Some(e),
            DataPrepError::SerdeJson(e) => Some(e),
            DataPrepError::Safetensor(e) => Some(e),
            DataPrepError::Tch(e) => Some(e),
            _ => None,
        }
    }
}

// Implement From traits for easy conversion using '?'
impl From<io::Error> for DataPrepError {
    fn from(err: io::Error) -> Self {
        DataPrepError::Io(err)
    }
}

impl From<serde_json::Error> for DataPrepError {
    fn from(err: serde_json::Error) -> Self {
        DataPrepError::SerdeJson(err)
    }
}

impl From<safetensors::SafeTensorError> for DataPrepError {
    fn from(err: safetensors::SafeTensorError) -> Self {
        DataPrepError::Safetensor(err)
    }
}

// Add From<tch::TchError> if needed

// Helper type alias for Results within this crate
pub type Result<T> = std::result::Result<T, DataPrepError>;
