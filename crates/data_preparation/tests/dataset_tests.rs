use data_preparation::Dataset; 
use std::collections::HashMap; 
use tch::{kind, Tensor};
use data_preparation::SafetensorsDataset; 

#[test]
fn test_empty_dataset(){
    // Test empty dataset
    let empty_dataset = Dataset::new(HashMap::new());
    assert_eq!(empty_dataset.len(), 0);
    assert!(empty_dataset.is_empty());
}

#[test]
fn test_dataset_len() {
    // Create a tensor with shape [32, 16, 16]
    let inputs = Tensor::ones(&[32, 16, 16], kind::FLOAT_CPU);

    // Create a dataset with 32 rows
    let input_tensors: Vec<Tensor> = (0..32)
        .map(|i| {
            inputs.index_select(0, &Tensor::from_slice(&[i as i64]))
        })
        .collect();

    let mut tensors = HashMap::new();
    tensors.insert("inputs".to_string(), input_tensors);
    let dataset = Dataset::new(tensors);

    // Test length
    assert_eq!(dataset.len(), 32);
    assert!(!dataset.is_empty());
}

#[test]
fn test_dataset_getitem_by_key() {
    // Create a tensor with shape [32, 16, 16]
    let inputs = Tensor::ones(&[32, 16, 16], kind::FLOAT_CPU);

    // Create a dataset
    let mut tensors = HashMap::new();
    tensors.insert("inputs".to_string(), vec![inputs.shallow_clone()]);
    let dataset = Dataset::new(tensors);

    // Test getitem by key
    let retrieved = dataset.tensors.get("inputs").unwrap();
    assert_eq!(retrieved.len(), 1);
    let are_equal = retrieved[0].eq_tensor(&inputs).all().int64_value(&[]) != 0;
    assert!(are_equal);
}

#[test]
fn test_datset_getitem_by_index() {
    // Create a tensor with shape [32, 16, 16]
    let inputs = Tensor::ones(&[32, 16, 16], kind::FLOAT_CPU); 

    // Create a dataset with 32 rows (split the tensor into 32 tensors of shape [16, 16])
    let input_tensors: Vec<Tensor> = (0..32)
        .map(|i| {
            inputs.index_select(0, &Tensor::from_slice(&[i as i64]))
        })
        .collect(); 

    let mut tensors = HashMap::new(); 
    tensors.insert("inputs".to_string(), input_tensors); 
    let dataset = Dataset::new(tensors); 

    // Test getitem by index 
    let input_tensors_ref = dataset.tensors.get("inputs").unwrap(); // Reference the tensors from dataset
    for i in 0..32 {
        let mut elem = HashMap::new();
        for (key, tensors) in &dataset.tensors {
            elem.insert(key.clone(), &tensors[i]);
        }
        let input_elem = elem.get("inputs").unwrap();
        let are_equal = input_elem.eq_tensor(&input_tensors_ref[i]).all().int64_value(&[]) != 0;
        assert!(are_equal);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io;
    // Import only what's needed and available without features
    use tempfile::TempDir;
    use tch::{kind, Tensor};

    // --- Helper Function (remains the same) ---
    fn setup_float_dataset(num_tensors: i64, dim_size: i64, key: &str) -> SafetensorsDataset {
        let tensor_shape = [num_tensors, dim_size];
        let test_data = Tensor::ones(&tensor_shape, kind::FLOAT_CPU);
        let test_tensors: Vec<Tensor> = (0..num_tensors)
            .map(|i| test_data.index_select(0, &Tensor::from_slice(&[i])))
            .collect();
        let mut tensors_map = HashMap::new();
        tensors_map.insert(key.to_string(), test_tensors);
        SafetensorsDataset::from_dict(tensors_map)
    }

    // --- Test Functions ---

    #[test]
    fn test_save_creates_file() {
        let temp_dir = TempDir::new().unwrap();
        //let temp_dir = TempDir::new("create_file_test").unwrap();
        let file_path = temp_dir.path().join("output.safetensors");
        let dataset = setup_float_dataset(5, 10, "test_float");
        dataset.save_to_file(&file_path).expect("Failed to save dataset");
        assert!(file_path.exists(), "Safetensors file was not created");
    }

    #[test]
    fn test_save_and_load_verifies_data_f32() {
        let temp_dir = TempDir::new().unwrap();
        //let temp_dir = TempDir::new("verify_data_f32").unwrap();
        let file_path = temp_dir.path().join("data_f32.safetensors");
        let num_tensors = 32i64;
        let dim_size = 128i64;
        let key = "test_float";
        let dataset = setup_float_dataset(num_tensors, dim_size, key);
        dataset.save_to_file(&file_path).expect("Failed to save dataset");
    
        // Load the dataset using SafetensorsDataset::load_from_file
        let loaded_dataset = SafetensorsDataset::load_from_file(&file_path)
            .expect("Failed to load dataset");
    
        // Verify the loaded dataset
        assert_eq!(loaded_dataset.len(), num_tensors as usize, "Incorrect number of tensors loaded");
        assert_eq!(loaded_dataset.keys().len(), 1, "Incorrect number of keys in loaded dataset");
        assert!(loaded_dataset.keys().contains(&key.to_string()), "Key '{}' not found in loaded dataset", key);
    
        let original_tensors = dataset.get_tensors(key).unwrap();
        let loaded_tensors = loaded_dataset.get_tensors(key).unwrap();
        assert_eq!(loaded_tensors.len(), original_tensors.len(), "Incorrect number of tensors under key '{}'", key);
    
        for (i, (orig, loaded)) in original_tensors.iter().zip(loaded_tensors.iter()).enumerate() {
            // Verify shape
            let orig_shape: Vec<i64> = orig.size().into_iter().collect();
            let loaded_shape: Vec<i64> = loaded.size().into_iter().collect();
            assert_eq!(loaded_shape, orig_shape, "Shape mismatch for tensor {} under key '{}'", i, key);
    
            // Verify data
            let are_equal = orig.eq_tensor(loaded).all().int64_value(&[]) != 0;
            assert!(are_equal, "Data mismatch for tensor {} under key '{}'", i, key);
        }
    }

    #[test]
    fn test_save_invalid_key_fails() {
        let mut tensors_map = HashMap::new();
        let dummy_tensor = Tensor::randn(&[10], kind::FLOAT_CPU);
        tensors_map.insert("invalid.key".to_string(), vec![dummy_tensor]);
        let dataset = SafetensorsDataset::from_dict(tensors_map);
        let temp_dir = TempDir::new().unwrap();
        //let temp_dir = TempDir::new("invalid_key").unwrap();
        let file_path = temp_dir.path().join("should_not_save.safetensors");
        let result = dataset.save_to_file(&file_path);
        assert!(result.is_err(), "Saving with invalid key '.' should fail");
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidInput, "Error kind should be InvalidInput for bad key");
    }

    // TODO: Add more tests for other data types (I64, Bool)... these data tests should work.
}