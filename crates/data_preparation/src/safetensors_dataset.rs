// src/safetensors.rs
use crate::dataset::Dataset; 
use crate::error::{DataPrepError, Result}; // Use custom error types
use safetensors::{serialize, Dtype, SafeTensors};
use safetensors::tensor::TensorView; 
use serde_json::{self, Value};
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fs::File;
use std::io::{Read, Write}; // Required traits
use std::path::Path;
use tch::{Kind, Tensor};

/// A wrapper around `Dataset` providing methods to save/load 
/// using the safetensors format 
/// 
/// This implementation handles the specific structure where each key 
/// maps to a `Vec<Tensor>`, saving them as numbered tensors (e.g., key.0)
/// and storing list metadata. 
#[derive(Debug)]
pub struct SafetensorsDataset {
    // Keep dataset private, expose access via methods 
    dataset: Dataset, 
}

impl SafetensorsDataset {
    /// Creaates a new SafetensorsDataset from a map of tensor lists. 
    ///
    /// # Errors 
    /// Returns `DataPrepError::InconsistentTensorList` if any key maps to an empty Vec<Tensor>.
    /// Returns 'DataPrepError::InconstistentTensorList' if tensors within a list have different dtypes. 
    pub fn from_dict(tensors: HashMap<String, Vec<Tensor>>) -> Result<Self> {
        for (key, value) in &tensors {
            if value.is_empty() {
                // Return Result instead of panic 
                return Err(DataPrepError::InconsistentTensorList(format!(
                    "Found empty tensor list for key '{}'", key
                )));
            }
            // Dtype homogeneity check 
            let first_kind = value[0].kind(); 
            if !value.iter().all(|t| t.kind() == first_kind) {
                return Err(DataPrepError::InconsistentTensorList(format!(
                    "Inconsistent dtypes found in list for key '{}'", key
                )));
            }
        }
        Ok(SafetensorsDataset{
            dataset: Dataset::new(tensors),
        })
    }

    /// Returns the number of items in the dataset. 
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Checks if the dataset contains any data 
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }

    /// Returns a set of borrowed keys in the dataset.
    pub fn keys(&self) -> HashSet<&String> {
        self.dataset.tensors.keys().collect()
    }
    
    /// Returns a reference to the list of tensors for a given key, if it exists. 
    pub fn get_tensors(&self, key: &str) -> Option<&Vec<Tensor>> {
        self.dataset.tensors.get(key)
    }

    /// Access the inner Dataset immutably.
    pub fn inner_dataset(&self) -> &Dataset {
        &self.dataset
    }

    /// Saves the dataset to a safetensors file. 
    /// 
    /// # Errors 
    /// Return `DataPrepError` on I/O issues, serialization errors, 
    /// invalid keys, or unsupported tensor types. 
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut tensor_data_map: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)> = HashMap::new(); 
        let mut metadata: HashMap<String, String> = HashMap::new(); // Values must be string 

        // Add overall dataset size metadata
        metadata.insert(
            "size".to_string(), 
            serde_json::to_string(&self.dataset.len())? // Use ? with From <serde.json> error 
        );

        for (key, tensor_list) in &self.dataset.tensors {
            if key.contains('.') {
                //Return error 
                return Err(DataPrepError::InvalidKey(format!(
                    "'.' is not allowed in key '{}'", key
                )));
            }

            if tensor_list.is_empty() {
                // This case might be unreachable if from_dict enforces non-empty lists. But, 
                // TODO: how to represent empty list metadata. 
                // For now, skip saving tensors and metadata for empty lists. 
            }

            // Determine list dtype 
            let list_kind = tensor_list[0].kind(); 
            let list_safetensor_dtype = match list_kind {
                Kind::Float => Dtype::F32, 
                Kind::Int64 => Dtype::I64,
                Kind::Bool => Dtype::BOOL, 
                // TODO: Add other supported types here 
                _ => return Err(DataPrepError::UnsupportedDtype(format!(
                    "Dtype {:?} in list for key '{}' is not supported for saving.",  list_kind, key
                )))
            };
            let list_dtype_str = format!("{:?}", list_safetensor_dtype); 

            // Process individual tensors 
            for (i, tensor) in tensor_list.iter().enumerate() {
                // Dtype homogeneity check
                if tensor.kind() != list_kind {
                    return Err(DataPrepError::InconsistentTensorList(format!(
                        "Inconsistent dtypes in list for key '{}': expected {:?}, found {:?} at index {}",
                        key, list_kind, tensor.kind(), i
                    )));
                } 

                let tensor_key = format!("{}.{}", key, i); 
                let num_elements = tensor.numel(); 
                let shape: Vec<usize> = tensor.size().iter().map(|&x| x as usize).collect(); 

                let (dtype_enum, bytes) = match list_kind {
                    Kind::Float => {
                        let mut data = vec![0.0f32; num_elements];
                        tensor.copy_data(&mut data, num_elements);
                        (Dtype::F32, data.into_iter().flat_map(|x|x.to_le_bytes()).collect())
                    }
                    Kind::Int64 => {
                        let mut data = vec![0i64; num_elements];
                        tensor.copy_data(&mut data, num_elements);
                        (Dtype::I64, data.into_iter().flat_map(|x|x.to_le_bytes()).collect())
                    }
                    Kind::Bool => {
                        let mut data = vec![0u8; num_elements]; 
                        tensor.copy_data(&mut data, num_elements);
                        (Dtype::BOOL, data)
                    }
                    // This case is most likely unreachable due to the dtype check before the loop 
                    _ => unreachable!("Unsupported dtype checked earlier"),
                };

                tensor_data_map.insert(tensor_key.clone(), (dtype_enum, shape, bytes));
            }

            // Constuct list metadata correctly 
            let tensor_meta_map: HashMap<&str, Value> = HashMap::from([
                ("list", Value::Bool(true)),
                ("numel", Value::Number(tensor_list.len().into())),
                ("dtype", Value::String(list_dtype_str)),
            ]);
            metadata.insert(key.clone(), serde_json::to_string(&tensor_meta_map)?);
        
        } // End outer loop 

        // Create TensorViews
        let tensors_for_serialization: HashMap<String, TensorView> = tensor_data_map
            .iter()
            .map(|(key, (dtype, shape, bytes))| {
                TensorView::new(*dtype, shape.clone(), bytes)
                    .map(|view| (key.clone(), view))
                    // Map safetensor error to custom error
                    .map_err(|e| DataPrepError::Safetensor(e))
            })
            .collect::<Result<HashMap<_, _>>>()?; // Provide the success T to Result

        // Serialize using safetensors crate 
        let serialized = serialize(&tensors_for_serialization, &Some(metadata))?; 

        // Write to file 
        let mut file = File::create(path)?;
        file.write_all(&serialized)?;
        Ok(())
    }


    /// Loads a dataset from a safetensors file using manual parsing. 
    /// 
    /// # Errors 
    /// Returns `DataPrepError` on I/O issues, file format errors, parsing errors, 
    /// missing metadata, or unsupported tensor types found in the file. 
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Read file 
        let mut file = File::open(path)?;
        let mut buffer = Vec::new(); 
        file.read_to_end(&mut buffer)?;

        if buffer.len() < 8 {
            return Err(DataPrepError::FileFormat("File too short for header size".into()));
        }

        let header_len = u64::from_le_bytes(buffer[..8]
            .try_into()
            .map_err(|_| DataPrepError::FileFormat("Invalid header size bytes".into()))?) as usize;
        let header_end = 8 + header_len;
        if buffer.len() < header_end {
            return Err(DataPrepError::FileFormat("File too short for header content".into()));
       }

       // Parse header JSON
       let header_bytes = &buffer[8..header_end];
       let header: Value = serde_json::from_slice(header_bytes)?;

       // Extract __metadata__ section
       let top_level_metadata = header
       .get("__metadata__")
       .ok_or_else(|| DataPrepError::MetadataNotFound("__metadata__ section missing".into()))?;
        // Ensure __metadata__ is an object/map
        let top_level_metadata_map: HashMap<String, Value> = serde_json::from_value(top_level_metadata.clone())?; 

        // Deserialize tensor views using safetensors library
        let safetensor_views = SafeTensors::deserialize(&buffer)?; 

        // Group tensor keys (e.g., "key.0", "key.1" -> "key")
        let mut grouped_keys: HashMap<String, Vec<String>> = HashMap::new();
        for name in safetensor_views.names() {
             if let Some((base_key, index_str)) = name.rsplit_once('.') {
                 // Basic validation of index part if needed
                 if index_str.parse::<usize>().is_ok() {
                     grouped_keys
                        .entry(base_key.to_string())
                        .or_default()
                        .push(name.clone());
                 } else {
                     // Handle keys with '.' but not numeric suffix? Or ignore?
                     // Ignore for now, assuming format from save_to_file
                 }
             } else {
                 // Ignore keys without '.' separator for now
             }
        }

        // Reconstruct dataset
        let mut dataset_tensors = HashMap::new();
        for (base_key, mut tensor_keys) in grouped_keys {
            // Sort keys by index ("key.0", "key.1", "key.10")
            tensor_keys.sort_by_key(|k| k.rsplit_once('.').unwrap().1.parse::<usize>().unwrap_or(usize::MAX));

            // Get list metadata string from top_level_metadata_map
            let meta_value = top_level_metadata_map
                .get(&base_key)
                .ok_or_else(|| DataPrepError::MetadataNotFound(format!("Metadata missing for key '{}'", base_key)))?;
            let meta_str = meta_value.as_str()
                .ok_or_else(|| DataPrepError::MetadataFormat(format!("Metadata for key '{}' is not a string", base_key)))?;

            // Parse the list metadata JSON string
             let list_meta: HashMap<String, Value> = serde_json::from_str(meta_str)?; // Use '?'

            // Extract fields using correct types
            let is_list = list_meta.get("list")
                .and_then(|v| v.as_bool())
                .ok_or_else(|| DataPrepError::MetadataFormat(format!("Missing/invalid 'list' boolean for key '{}'", base_key)))?;

            let numel = list_meta.get("numel")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| DataPrepError::MetadataFormat(format!("Missing/invalid 'numel' number for key '{}'", base_key)))? as usize;

            let dtype_str = list_meta.get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| DataPrepError::MetadataFormat(format!("Missing/invalid 'dtype' string for key '{}'", base_key)))?;

            if !is_list {
                // Handle non-list case? Currently save_to_file always sets list=true
                continue;
            }

            if numel != tensor_keys.len() {
                 return Err(DataPrepError::FileFormat(format!(
                     "Metadata 'numel' ({}) mismatches found tensor count ({}) for key '{}'",
                     numel, tensor_keys.len(), base_key
                 )));
            }    

            // Reconstruct tensor list
            let mut tensor_list = Vec::with_capacity(numel);
            for tensor_key in tensor_keys { // Iterate sorted keys
                let tensor_view = safetensor_views.tensor(&tensor_key)?; // Use '?'

                // Check TensorView dtype matches metadata (optional but good)
                let view_dtype_str = format!("{:?}", tensor_view.dtype());
                 if view_dtype_str != dtype_str {
                      return Err(DataPrepError::FileFormat(format!(
                         "Metadata dtype ('{}') mismatches TensorView dtype ('{}') for tensor '{}'",
                         dtype_str, view_dtype_str, tensor_key
                     )));
                 }

                // Reconstruct tch::Tensor
                let tensor = match dtype_str {
                    "F32" => {
                        let data: Vec<f32> = tensor_view.data()
                            .chunks_exact(4)
                            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())) // Consider error handling for try_into
                            .collect();
                         Tensor::from_slice(&data)
                            .reshape(&tensor_view.shape().iter().map(|&d| d as i64).collect::<Vec<_>>())
                    }
                     "I64" => {
                        let data: Vec<i64> = tensor_view.data()
                            .chunks_exact(8)
                            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                            .collect();
                         Tensor::from_slice(&data)
                            .reshape(&tensor_view.shape().iter().map(|&d| d as i64).collect::<Vec<_>>())
                    }
                     "BOOL" => {
                        let data: Vec<u8> = tensor_view.data().to_vec(); // Assume BOOL is u8
                         Tensor::from_slice(&data)
                             .reshape(&tensor_view.shape().iter().map(|&d| d as i64).collect::<Vec<_>>())
                             .to_kind(Kind::Bool) // Explicitly convert to Bool kind
                    }
                    // TODO: Add other types here if supported by save_to_file
                    unsupported_dtype => {
                        return Err(DataPrepError::UnsupportedDtype(format!(
                             "Dtype '{}' specified in metadata for tensor '{}' is not supported for loading.",
                             unsupported_dtype, tensor_key
                        )));
                    }
                };
                tensor_list.push(tensor);
            }
            dataset_tensors.insert(base_key.clone(), tensor_list); // Clone base_key needed
        } // End loop over grouped keys

        Ok(SafetensorsDataset {
            dataset: Dataset::new(dataset_tensors),
        })
    }  
     
}