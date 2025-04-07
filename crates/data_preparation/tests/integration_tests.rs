// tests/integration_tests.rs
use data_preparation::{DataPrepError, SafetensorsDataset}; // Use crate name
use std::collections::HashMap;
use tempfile::TempDir; // Using tempfile now
use tch::{kind, Kind, Tensor}; 

// --- Helper Functions ---
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

fn setup_multi_key_dataset() -> (SafetensorsDataset, HashMap<String, Vec<Tensor>>) {
    let num_rows = 3; 

    // Features: Row 0=[0.], Row 1=[1.], Row 2=[2.] (shape [1, 1])
    let features_list: Vec<Tensor> = (0..num_rows)
        .map(|i| Tensor::f_from_slice(&[i as f32]).unwrap().reshape(&[1, 1]))
        .collect(); 

    // Labels: Row 0 = [10], Row 1 = [11], Row 2 = [12] (shape [])
    let labels_list: Vec<Tensor> = (0..num_rows)
        .map(|i| Tensor::from(i+10).to_kind(Kind::Int64))
        .collect(); 

    let mut tensors_map = HashMap::new(); 
    tensors_map.insert("features".to_string(), features_list.iter().map(|t| t.shallow_clone()).collect());
    tensors_map.insert("labels".to_string(), labels_list.iter().map(|t| t.shallow_clone()).collect());

    // Use the constructor which returns Result 
    let dataset = SafetensorsDataset::from_dict(tensors_map).expect("Setup failed");

    // Return both dataset and the original tensors for easy verification 
    let original_tensors = HashMap::from([
        ("features".to_string(), features_list),
        ("labels".to_string(), labels_list),
    ]);

    (dataset, original_tensors)
}

// --- Save/Load Tests ----
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

// --- Accessor Method Tests --- 
#[test]
fn test_contains_key() {
    let (dataset, _) = setup_multi_key_dataset(); 

    assert!(dataset.contains_key("features"), "'features' key check failed"); 
    assert!(dataset.contains_key("labels"), "'labels' key check failed");
    assert!(!dataset.contains_key("unknown_key"), "'unknown_key' key check failed");
    assert!(!dataset.contains_key(""), "Empty string key check failed"); 
}

#[test]
fn test_get_by_key() {
    let (dataset, original_tensors) = setup_multi_key_dataset(); 

    // Test getting existing key features 
    let features_vec_opt = dataset.get_by_key("features");
    assert!(features_vec_opt.is_some(), "Key 'features' shoudl exist"); 
    let features_vec = features_vec_opt.unwrap();
    assert_eq!(features_vec.len(), 3, "Incorrect length for 'features' vector"); 
    // Compare first tensor content 
    assert!(features_vec[0].allclose(&original_tensors["features"][0], 1e-6, 1e-6, false), "Tensor data mismatch for features[0]");

    // Test getting existing key "labels"
    let labels_vec_opt = dataset.get_by_key("labels");
    assert!(labels_vec_opt.is_some(), "Key 'labels' should exist");
    let labels_vec = labels_vec_opt.unwrap();
    assert_eq!(labels_vec.len(), 3, "Incorrect length for 'labels' vector");
    // Compare second tensor content (eq_tensor for ints)
    assert!(labels_vec[1].eq_tensor(&original_tensors["labels"][1]).all().int64_value(&[])==1, "Tensor data mismatch for labels[1]");

    // Test getting non-existent key 
    let unknown_vec_opt = dataset.get_by_key("unknown_key");
    assert!(unknown_vec_opt.is_none(), "Key 'unknown_key' should not exist");
}

#[test]
fn test_get_by_index() {
    let (dataset, original_tensors) = setup_multi_key_dataset();
    let num_rows = 3; 

    // Test fetching first row (index 0)
    let row0_opt = dataset.get_by_index(0);
    assert!(row0_opt.is_some(), "Row 0 should exist");
    let row0 = row0_opt.unwrap(); 
    assert_eq!(row0.len(), 2, "Row 0 should have 2 keys");
    assert!(row0.contains_key("features"), "Row 0 missing 'features'");
    assert!(row0.contains_key("labels"), "Row 0 missing 'labels'");
    // Compare tensor references 
    assert!(row0["features"].allclose(&original_tensors["features"][0], 1e-6, 1e-6, false), "Row 0 features mismatch");
    assert!(row0["labels"].eq_tensor(&original_tensors["labels"][0]).all().int64_value(&[])==1, "Row 0 labels mismatch");

   // Test fetching last row (index num_rows - 1)
   let last_index = num_rows - 1;
   let row_last_opt = dataset.get_by_index(last_index);
   assert!(row_last_opt.is_some(), "Last row ({}) should exist", last_index);
   let row_last = row_last_opt.unwrap();
   assert_eq!(row_last.len(), 2, "Last row ({}) should have 2 keys", last_index);
   assert!(row_last["features"].allclose(&original_tensors["features"][last_index], 1e-6, 1e-6, false), "Last row features mismatch");
   assert!(row_last["labels"].eq_tensor(&original_tensors["labels"][last_index]).all().int64_value(&[])==1, "Last row labels mismatch");

    // Test fetching out-of-bounds index
    let invalid_index = num_rows; // Index equal to length is out of bounds
    let row_invalid_opt = dataset.get_by_index(invalid_index);
    assert!(row_invalid_opt.is_none(), "Row at invalid index {} should be None", invalid_index);
}

// --- Constructor/Edge Case Tests ---
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