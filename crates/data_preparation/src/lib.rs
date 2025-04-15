use anyhow::{bail, Result};
use std::collections::HashMap;
use tch::Tensor;

#[derive(Debug)]
pub struct Dataset {
    tensors: HashMap<String, Tensor>,
}

impl Dataset {
    /// Creates a new Dataset from a HashMap of tensors.
    /// Validates that:
    /// - No tensors are scalar (must have at least one dimension).
    /// - All tensors share the same size for the first dimension (batch size).
    pub fn new(tensors: HashMap<String, Tensor>) -> Result<Self> {
        if tensors.is_empty() {
            return Ok(Dataset { tensors });
        }

        // Check for scalars
        let batch_sizes: Vec<usize> = tensors
            .iter()
            .map(|(key, t)| {
                let size = t.size();
                if size.is_empty() {
                    bail!(
                        "Scalar tensor '{}' not allowed; tensors must have a batch dimension (e.g., [batch_size], [batch_size, seq_len])",
                        key
                    )
                } else {
                    Ok(size[0] as usize)
                }
            })
            .collect::<Result<Vec<usize>>>()?;

        // Ensure the batch size is consistent
        let first_batch_size = batch_sizes[0];
        if !batch_sizes.iter().all(|&size| size == first_batch_size) {
            bail!(
                "Inconsistent batch sizes: expected {}, found {:?}",
                first_batch_size,
                batch_sizes
            );
        }

        Ok(Dataset { tensors })
    }

    /// Returns the number of samples in the batch.
    pub fn len(&self) -> usize {
        self.tensors
            .values()
            .next()
            .map_or(0, |t| t.size()[0] as usize)
    }

    /// Checks if the dataset contains no tensors
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Returns an immutable reference to a tensor by key, if it exists.
    pub fn get(&self, key: &str) -> Option<&Tensor> {
        self.tensors.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Kind, Tensor}; 

    #[test]
    fn test_accept_1d_batch_tensor() {
        let tensor = Tensor::from_slice(&[1, 2, 3, 4]).to_kind(Kind::Int);
        let mut data_map = HashMap::new();
        data_map.insert("tensor".to_string(), tensor);

        match Dataset::new(data_map) {
            Ok(_dataset) => println!("Successfully created dataset for 1D data."),
            Err(e) => panic!("Error creating dataset for 1D data: {}", e),
        }
    }

    #[test]
    fn test_accept_padded_variable_length_tensors() {
        let max_length = 5;
        let sequences = vec![
            vec![1, 2, 3],
            vec![4, 5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14],
        ];

        let mut padded_data = Vec::new();
        for seq in sequences {
            let mut padded_seq = seq;
            padded_seq.extend(vec![0; max_length - padded_seq.len()]);
            padded_data.extend(padded_seq);
        }

        let tensor = Tensor::from_slice(&padded_data)
            .reshape(&[4, max_length as i64])
            .to_kind(Kind::Int);

        let mut data_map = HashMap::new();
        data_map.insert("tensor".to_string(), tensor);

        match Dataset::new(data_map) {
            Ok(dataset) => {
                println!("Successfully created dataset for variable-length data after applying zero-padding.");
                println!("Dataset: {:?}", dataset);
            }
            Err(e) => {
                panic!(
                    "Error creating dataset for variable-length data even after zero-padding: {}",
                    e
                );
            }
        }
    }

    #[test]
    fn test_reject_unpadded_variable_length_tensors() {
        let sequences = vec![
            vec![1, 2, 3],
            vec![4, 5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14],
        ];

        let tensors: Vec<Tensor> = sequences
            .into_iter()
            .map(|seq| Tensor::from_slice(&seq).to_kind(Kind::Int))
            .collect();

        let result = Tensor::f_stack(&tensors, 0); // Should fail. 

        match result {
            Ok(t) => {
                let mut data_map = HashMap::new();
                data_map.insert("tokens".to_string(), t);
                match Dataset::new(data_map) {
                    Ok(_) => panic!("Expected Dataset::new to fail due to inconsistent shapes, but it succeeded."),
                    Err(e) => println!("Correctly rejected by Dataset::new: {}", e),
                }
            }
            Err(e) => {
                println!("Correctly failed to stack variable-length tensors: {}", e);
            }
        }
    }
}
