mod common; 
use common::{setup_float_dataset_for_test, setup_typed_dataset};
use data_preparation::{DataPrepError, SafetensorsDataset};
use std::collections::HashMap;
use tempfile::TempDir;
use tch::{Kind, Tensor};

// Tests for save_to_file, load_from_file
#[test]
fn test_create_and_save_safetensors_file() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("output.safetensors");

    let dataset = setup_float_dataset_for_test(5, 10, "test_float");
    dataset.save_to_file(&file_path).expect("Failed to save dataset");

    assert!(file_path.exists(), "Safetensors file was not created");
}

#[test]
fn test_save_invalid_key_to_safetensors_file() {
    let mut map = HashMap::new();
    map.insert("invalid.key".to_string(), vec![Tensor::randn(&[10], (Kind::Float, tch::Device::Cpu))]);

    let dataset = SafetensorsDataset::from_dict(map).expect("Creation should succeed here");

    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("should_not_save.safetensors");

    let result = dataset.save_to_file(&file_path);
    assert!(result.is_err(), "Saving with invalid key '.' should fail");

    if let Err(DataPrepError::InvalidKey(msg)) = result {
        assert!(
            msg.contains("'.' is not allowed in key 'invalid.key'"),
            "Wrong error message: {}",
            msg
        );
    } else {
        panic!("Expected InvalidKey, got something else");
    }
}

#[test]
fn test_load_safetensors_file_and_verify_dtypes() {
    // We'll do multiple dtype checks: int64, bool, double, etc.
    let temp_dir = TempDir::new().unwrap();

    // 1) Int64
    let file_path_i64 = temp_dir.path().join("data_i64.safetensors");
    let (ds_i64, original_map_i64) =
        setup_typed_dataset(4, &[2], "i64_key", &(0..8).collect::<Vec<i64>>(), Kind::Int64);
    ds_i64.save_to_file(&file_path_i64).expect("Save i64 failed");
    let loaded_i64 = SafetensorsDataset::load_from_file(&file_path_i64).expect("Load i64 failed");
    assert_eq!(loaded_i64.len(), 4);
    let loaded_tensors = loaded_i64.get_tensors("i64_key").unwrap();
    let original_tensors = original_map_i64.get("i64_key").unwrap();
    assert_eq!(loaded_tensors.len(), original_tensors.len());
    // Spot check
    assert_eq!(loaded_tensors[0].kind(), Kind::Int64, "Should be int64 kind");

    // 2) Bool
    let file_path_bool = temp_dir.path().join("data_bool.safetensors");
    let bool_values: Vec<bool> = (0..10).map(|x| x % 2 == 0).collect();
    let (ds_bool, original_map_bool) =
        setup_typed_dataset(2, &[5], "bool_key", &bool_values, Kind::Bool);
    ds_bool.save_to_file(&file_path_bool).expect("Save bool failed");
    let loaded_bool = SafetensorsDataset::load_from_file(&file_path_bool).expect("Load bool failed");
    assert_eq!(loaded_bool.len(), 2);
    let bool_tensors = loaded_bool.get_tensors("bool_key").unwrap();
    let orig_bool_tensors = original_map_bool.get("bool_key").unwrap();
    assert_eq!(bool_tensors.len(), orig_bool_tensors.len());
    // Spot check
    assert_eq!(bool_tensors[0].kind(), Kind::Bool, "Should be bool kind");
}

#[test]
fn test_load_from_nonexistent_file() {
    let result = SafetensorsDataset::load_from_file("this_file_does_not_exist.safetensors");
    assert!(result.is_err(), "Loading nonexistent file should fail");
    // Could match on DataPrepError::Io or DataPrepError::FileFormat
}