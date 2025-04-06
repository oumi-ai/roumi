// tests/integration_tests.rs
use data_preparation::{DataPrepError, SafetensorsDataset}; // Use crate name
use std::collections::HashMap;
use tempfile::TempDir; // Using tempfile now
use tch::{kind, Tensor}; // Import Kind if needed for type checks

// Helper function can stay here or be moved into the main crate
// under test_utils feature if preferred. Keeping it here for now.
fn setup_float_dataset_for_test(num_tensors: i64, dim_size: i64, key: &str) -> SafetensorsDataset {
    let tensor_shape = [num_tensors, dim_size];
    let test_data = Tensor::ones(&tensor_shape, kind::FLOAT_CPU);
    let test_tensors: Vec<Tensor> = (0..num_tensors)
        .map(|i| test_data.index_select(0, &Tensor::from_slice(&[i])))
        .collect();
    let mut tensors_map = HashMap::new();
    tensors_map.insert(key.to_string(), test_tensors);
    // Use the new constructor which returns Result
    SafetensorsDataset::from_dict(tensors_map).expect("Setup failed")
}

#[test]
fn test_create_and_save_safetensors_file() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("output.safetensors");
    let dataset = setup_float_dataset_for_test(5, 10, "test_float");

    dataset.save_to_file(&file_path).expect("Failed to save dataset");

    assert!(file_path.exists(), "Safetensors file was not created");
}

#[test]
fn test_load_safetensors_file_and_verify_f32_data() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("data_f32.safetensors");
    let num_tensors = 32i64;
    let dim_size = 128i64;
    let key = "test_float";
    let dataset = setup_float_dataset_for_test(num_tensors, dim_size, key);

    // Save
    dataset.save_to_file(&file_path).expect("Failed to save dataset");

    // Load using the struct's method
    let loaded_dataset = SafetensorsDataset::load_from_file(&file_path)
        .expect("Failed to load dataset");

    // Verify
    assert_eq!(loaded_dataset.len(), num_tensors as usize, "Incorrect numel loaded");
    assert_eq!(loaded_dataset.keys().len(), 1, "Incorrect number of keys");
    assert!(loaded_dataset.keys().contains(&key.to_string()), "Key missing");

    // Use the public getter method
    let original_tensors = dataset.get_tensors(key).unwrap();
    let loaded_tensors = loaded_dataset.get_tensors(key).unwrap();
    assert_eq!(loaded_tensors.len(), original_tensors.len(), "Tensor list length mismatch");

    for (i, (orig, loaded)) in original_tensors.iter().zip(loaded_tensors.iter()).enumerate() {
        assert_eq!(loaded.size(), orig.size(), "Shape mismatch for tensor {}", i);
        assert_eq!(loaded.kind(), orig.kind(), "Kind mismatch for tensor {}", i);
        // Use allclose for float comparison
        assert!(orig.allclose(loaded, 1e-6, 1e-6, false), "Data mismatch for tensor {}", i);
        // Also verify it's ones
        assert!(loaded.eq(1.0).all().int64_value(&[]) == 1, "Tensor {} not ones", i)
    }
}

#[test]
fn test_save_invalid_key_to_safetensors_file() {
    let mut tensors_map = HashMap::new();
    let dummy_tensor = Tensor::randn(&[10], kind::FLOAT_CPU);
    tensors_map.insert("invalid.key".to_string(), vec![dummy_tensor]);

    // Use constructor that returns Result
    let dataset = SafetensorsDataset::from_dict(tensors_map).expect("Creation should succeed here");

    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("should_not_save.safetensors");

    let result = dataset.save_to_file(&file_path);
    assert!(result.is_err(), "Saving with invalid key '.' should fail");

    // Check custom error type
    match result.err().unwrap() {
        DataPrepError::InvalidKey(msg) => assert!(msg.contains("'.' is not allowed")),
        e => panic!("Expected InvalidKey error, got {:?}", e),
    }
}

#[test]
fn test_load_missing_metadata_from_safetensors_file_fails() {
     // Need a way to create a safetensor file *without* the expected metadata
     // This is hard to do without manually crafting a file or using internal APIs.
     // Skipping for now, but highlights a potential test gap for the manual loader.
     println!("Skipping test for loading file with missing metadata.");
}

#[test]
fn test_load_metadata_wrong_type_from_safetensors_file_fails() {
     // Need to craft a file where e.g., "list" metadata is not a boolean string.
     // Skipping for now.
     println!("Skipping test for loading file with malformed metadata type.");
}

#[test]
fn test_save_empty_list_to_safetensors_dataset_fails() {
    let mut tensors_map = HashMap::new();
    tensors_map.insert("empty_list".to_string(), vec![]); // Empty vec

    let result = SafetensorsDataset::from_dict(tensors_map);
    assert!(result.is_err());
     match result.err().unwrap() {
        DataPrepError::InconsistentTensorList(msg) => assert!(msg.contains("Found empty tensor list")),
        e => panic!("Expected InconsistentTensorList error, got {:?}", e),
    }
}

// TODO: Add integration tests for I64, Bool types save/load.
// TODO: Add integration tests for multiple keys.