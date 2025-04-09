// src/dataset.rs
use std::collections::HashMap;
use tch::Tensor;

// Represents a dataset as a dictionary of lists of tensors.
// Assumes that for a given key, all tensors in the Vec have the same shape and dtype.

#[derive(Debug)]
pub struct Dataset {
    // The data structure holding tensor lists keyed by strings.
    // Make field public for now for easy access from SafetensorsDataset,
    // or keep private and add necessary accessors. Let's keep it puublic for now for simplicity.
    pub tensors: HashMap<String, Vec<Tensor>>,
}

impl Dataset {
    // Creates a new Dataset instance.
    pub fn new(tensors: HashMap<String, Vec<Tensor>>) -> Self {
        Dataset { tensors }
    }

    // Returns the number of tensor lists in the dataset.
    // Assumes all lists have the same length. Return 0 if empty.
    pub fn len(&self) -> usize {
        self.tensors
            .values()
            .find(|v| !v.is_empty())
            .map_or(0, |v| v.len())
    }

    // Checks if the dataset contains any data
    pub fn is_empty(&self) -> bool {
        // More robust check: are all lists empty or is the map empty?
        self.tensors.values().all(|v| v.is_empty())
    }
}

// --- Unit Tests for Dataset ---
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tch::{kind, Tensor};

    #[test]
    fn test_empty_dataset() {
        let empty_dataset = Dataset::new(HashMap::new());
        assert_eq!(empty_dataset.len(), 0);
        assert!(empty_dataset.is_empty());
    }

    #[test]
    fn test_dataset_with_empty_list() {
        // Test behaviour with keys mapped to empty lists
        let mut tensors = HashMap::new();
        tensors.insert("empty_key".to_string(), vec![]);
        let dataset = Dataset::new(tensors);
        assert_eq!(dataset.len(), 0); //len() finds no non-empty lists
        assert!(dataset.is_empty()); // is_empty() checks all lists
    }

    #[test]
    fn test_dataset_len() {
        let inputs = Tensor::ones(&[32, 16], kind::FLOAT_CPU);
        let input_tensors: Vec<Tensor> = (0..32)
            .map(|i| inputs.index_select(0, &Tensor::from_slice(&[i])))
            .collect();
        let mut tensors = HashMap::new();
        tensors.insert("inputs".to_string(), input_tensors);
        let dataset = Dataset::new(tensors);

        assert_eq!(dataset.len(), 32);
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_dataset_getitem_by_key() {
        // Test accessing the underlying map directly
        let inputs = Tensor::ones(&[5, 10], kind::FLOAT_CPU);
        let mut tensors = HashMap::new();
        tensors.insert("inputs".to_string(), vec![inputs.shallow_clone()]);
        let dataset = Dataset::new(tensors);

        let retrieved = dataset.tensors.get("inputs").unwrap();
        assert_eq!(retrieved.len(), 1);
        // Use tch tensor comparison
        assert!(retrieved[0].allclose(&inputs, 1e-6, 1e-6, false));
    }
}
