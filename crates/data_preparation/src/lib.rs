use tch::{Tensor, Kind};
use std::collections::{HashMap, HashSet};
use safetensors::{serialize, Dtype, tensor::TensorView}; 
use std::fs::File; 
use std::path::Path; 
use std::io::{self, Read, Write}; 
use serde_json::Value; 
use safetensors::SafeTensors;


// Represents a dataset as a dictionary of lists of tensors 
#[derive(Debug)]
pub struct Dataset{
    pub tensors: HashMap<String, Vec<Tensor>>,
}

impl Dataset{
    pub fn new(tensors: HashMap<String, Vec<Tensor>>) -> Self{
        Dataset{tensors}
    }

    pub fn len(&self) -> usize {
        self.tensors.values().next().map_or(0, |v| v.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}


#[derive(Debug)]
pub struct SafetensorsDataset {
    dataset: Dataset,
}

impl SafetensorsDataset{
    pub fn from_dict(tensors: HashMap<String, Vec<Tensor>>) -> Self {
        for (key, value) in &tensors {
            if value.is_empty() {
                panic!("Found empty list for key '{}'", key);
            }
        }
        SafetensorsDataset {
            dataset: Dataset::new(tensors),
        }
    }

    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }

    /// Returns the set of keys in the dataset.
    pub fn keys(&self) -> HashSet<&String> {
        self.dataset.tensors.keys().collect()
    }

    /// Returns the list of tensors for a given key, if it exists.
    pub fn get_tensors(&self, key: &str) -> Option<&Vec<Tensor>> {
        self.dataset.tensors.get(key)
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        
        let mut tensor_data_map: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)> = HashMap::new();
        let mut metadata = HashMap::new();

        metadata.insert(
             "size".to_string(),
             serde_json::to_string(&self.dataset.len())
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("serde error: {}", e)))?
        );

        for (key, tensor_list) in &self.dataset.tensors {
            if key.contains('.') {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("'.' is not allowed in key '{}'", key),
                ));
            }

            for (i, tensor) in tensor_list.iter().enumerate() {
                let tensor_key = format!("{}.{}", key, i);
                let num_elements = tensor.numel(); // num_elements is usize
                let shape: Vec<usize> = tensor.size().iter().map(|&x| x as usize).collect();

                let (dtype, bytes) = match tensor.kind() {
                    Kind::Float => {
                        let mut data = vec![0.0f32; num_elements];
                        tensor.copy_data(&mut data, num_elements);
                        let byte_data = data.into_iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
                        (Dtype::F32, byte_data)
                    }
                    Kind::Int64 => {
                         let mut data = vec![0i64; num_elements];
                         tensor.copy_data(&mut data, num_elements);
                         let byte_data = data.into_iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
                        (Dtype::I64, byte_data)
                    }
                    Kind::Bool => {
                        let mut data = vec![0u8; num_elements];
                        tensor.copy_data(&mut data, num_elements);
                        (Dtype::BOOL, data)
                    }
                    _ => return Err(io::Error::new(
                        io::ErrorKind::Other, // Uses io::ErrorKind now correctly
                        format!("Unsupported dtype for tensor '{}': {:?}", tensor_key, tensor.kind()),
                    )),
                };

                 tensor_data_map.insert(tensor_key.clone(), (dtype, shape, bytes));
            }

            let tensor_meta = HashMap::from([
                ("list".to_string(), serde_json::to_string(&true)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("serde error: {}", e)))?), // Uses io::Error correctly
                ("numel".to_string(), serde_json::to_string(&tensor_list.len())
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("serde error: {}", e)))?), // Uses io::Error correctly
                ("dtype".to_string(), serde_json::to_string("F32") // Still hardcoded F32 - potential issue
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("serde error: {}", e)))?), // Uses io::Error correctly
            ]);
            metadata.insert(
                key.clone(),
                serde_json::to_string(&tensor_meta)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("serde error: {}", e)))? // Uses io::Error correctly
             );
        }

        let tensors_for_serialization: HashMap<String, TensorView> = tensor_data_map // Uses tensor_data_map correctly
             .iter()
             .map(|(key, (dtype, shape, bytes))| {
                  TensorView::new(*dtype, shape.clone(), bytes)
                      .map(|view| (key.clone(), view))
                      .map_err(|e| {
                          io::Error::new( 
                              io::ErrorKind::Other, 
                              format!("Failed to create TensorView for {}: {}", key, e),
                          )
                      })
             })
             .collect::<Result<HashMap<_, _>, io::Error>>()?; 


        let serialized = serialize(&tensors_for_serialization, &Some(metadata))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("safetensor error: {}", e)))?; 

        let mut file = File::create(path)?;
        file.write_all(&serialized)?;
        Ok(())
    }

    /// Load a dataset from a safetensors file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        // Read the safetensors file
        let mut file = File::open(&path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        // Parse the safetensors header manually to extract metadata
        // The first 8 bytes are the header size (64-bit little-endian integer)
        if buffer.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "Safetensors file too short to contain header size",
            ));
        }
        let header_size = u64::from_le_bytes(buffer[..8].try_into().unwrap()) as usize;
        if buffer.len() < 8 + header_size {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "Safetensors file too short to contain header",
            ));
        }
        let header_bytes = &buffer[8..8 + header_size];
        let header: serde_json::Value = serde_json::from_slice(header_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Failed to parse safetensors header: {}", e)))?;

        // Extract metadata from the header
        let metadata = header
            .get("__metadata__")
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Missing __metadata__ in safetensors header"))?;
        let metadata: HashMap<String, Value> = serde_json::from_value(metadata.clone())
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("serde error: {}", e)))?;

        // The rest of the buffer is the tensor data
        let safetensors = SafeTensors::deserialize(&buffer)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("safetensor error: {}", e)))?;

        // Group tensors by their base key (e.g., "test.0", "test.1" -> "test")
        let mut dataset_tensors = HashMap::new();
        let mut grouped_keys = HashMap::new();

        for key in safetensors.names() {
            let parts: Vec<&str> = key.splitn(2, '.').collect();
            if parts.len() == 2 {
                let base_key = parts[0];
                grouped_keys
                    .entry(base_key.to_string())
                    .or_insert_with(Vec::new)
                    .push(key.clone());
            } else {
                // Single tensor (not implemented for now)
                continue;
            }
        }

        // Process each group of tensors
        for (base_key, _tensor_keys) in grouped_keys {
            let meta = metadata
                .get(&base_key)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::Other,
                        format!("Missing metadata for key '{}'", base_key),
                    )
                })?;
            // The metadata value is a JSON string, so parse it first
            let meta_str = meta.as_str()
                .ok_or_else(|| io::Error::new(io::ErrorKind::Other, format!("Metadata for key '{}' is not a string", base_key)))?;
            let meta: Value = serde_json::from_str(meta_str)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("serde error: {}", e)))?;
            //let meta: HashMap<String, Value> = serde_json::from_value(meta.clone())
            //    .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("serde error: {}", e)))?;
            let meta: HashMap<String, Value> = serde_json::from_value(meta)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("serde error: {}", e)))?;

            let is_list = meta.get("list")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::Other,
                        format!("Missing 'list' in metadata for '{}'", base_key),
                    )
                })?
                .parse::<bool>()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Failed to parse 'list' as bool: {}", e)))?;
    
            let numel = meta.get("numel")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::Other,
                        format!("Missing 'numel' in metadata for '{}'", base_key),
                    )
                })?
                .parse::<u64>()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Failed to parse 'numel' as u64: {}", e)))? as usize;

            let dtype = meta
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::Other,
                        format!("Missing 'dtype' in metadata for '{}'", base_key),
                    )
                })?;
            // Strip quotes from dtype (e.g., "\"F32\"" -> "F32")
            let dtype = dtype.trim_matches('"');

            if is_list {
                let mut tensor_list = Vec::with_capacity(numel);
                for i in 0..numel {
                    let tensor_key = format!("{}.{}", base_key, i);
                    let tensor_view = safetensors
                        .tensor(&tensor_key)
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("safetensor error: {}", e)))?;

                    // Convert bytes back to tensor based on dtype
                    let tensor = match dtype {
                        "F32" => {
                            let f32_data: Vec<f32> = tensor_view
                                .data()
                                .chunks_exact(4)
                                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                                .collect();
                            Tensor::from_slice(&f32_data)
                                .reshape(&tensor_view.shape().iter().map(|&x| x as i64).collect::<Vec<_>>())
                        }
                        "I64" => {
                            let i64_data: Vec<i64> = tensor_view
                                .data()
                                .chunks_exact(8)
                                .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                                .collect();
                            Tensor::from_slice(&i64_data)
                                .reshape(&tensor_view.shape().iter().map(|&x| x as i64).collect::<Vec<_>>())
                        }
                        "BOOL" => {
                            let bool_data: Vec<u8> = tensor_view.data().to_vec();
                            Tensor::from_slice(&bool_data)
                                .reshape(&tensor_view.shape().iter().map(|&x| x as i64).collect::<Vec<_>>())
                        }
                        _ => return Err(io::Error::new(
                            io::ErrorKind::Other,
                            format!("Unsupported dtype '{}' for tensor '{}'", dtype, tensor_key),
                        )),
                    };
                    tensor_list.push(tensor);
                }
                dataset_tensors.insert(base_key, tensor_list);
            }
        }

        Ok(SafetensorsDataset {
            dataset: Dataset::new(dataset_tensors),
        })
    }
}


