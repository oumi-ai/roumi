use tch::{Tensor, Kind};
use std::collections::HashMap;
use safetensors::{serialize, Dtype, tensor::TensorView}; 
use std::fs::File; 
use std::path::Path; 
use std::io::{self, Write}; 


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

    
}


