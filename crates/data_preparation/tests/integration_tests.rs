// tests/integration_tests.rs
use data_preparation::{DataPrepError, SafetensorsDataset, TensorLayout}; // Use crate name
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

// --- Tests for Various Dtypes ---
// --- Helper Function ---
fn setup_typed_dataset<T>(
    num_tensors: i64,
    dims: &[i64],
    key: &str,
    values: &[T],
    kind: Kind,
) -> (SafetensorsDataset, HashMap<String, Vec<Tensor>>) 
where
    T: tch::kind::Element + Copy,
{
    assert_eq!(values.len() as i64, num_tensors * dims.iter().product::<i64>(), "Incorrect number of values provided for shape");
    
    // 1. Create the original list of tensors
    let original_tensor_list: Vec<Tensor> = values.chunks(dims.iter().product::<i64>() as usize)
        .map(|chunk| Tensor::from_slice(chunk).to_kind(kind).reshape(dims)) // Assuming default device is okay now
        .collect();
    assert_eq!(original_tensor_list.len(), num_tensors as usize);
    
    // 2. Create the map to return for test verification purposes.
    let mut verification_tensors_map = HashMap::new();
    verification_tensors_map.insert(
        key.to_string(),
        original_tensor_list.iter().map(|t| t.shallow_clone()).collect()
    );
    
    // 3. Create the map that will be PASSED TO and OWNED BY from_dict.
    let mut dataset_tensors_map = HashMap::new();
    // Move the original_tensor_list into this map
    dataset_tensors_map.insert(key.to_string(), original_tensor_list);
    
    // 4. Create the dataset by PASSING OWNERSHIP of dataset_tensors_map. 
    let dataset = SafetensorsDataset::from_dict(dataset_tensors_map)
        .expect("Setup failed");
    
    // 5. Return the created dataset and the map containing shallow clones for verification
    (dataset, verification_tensors_map)
    }


#[test]
fn test_load_safetensors_file_and_verify_i64_data() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("data_i64.safetensors");
    let num_tensors = 10i64;
    let dims = [4i64, 2i64];
    let key = "test_i64";
    let values: Vec<i64> = (0..(num_tensors * dims.iter().product::<i64>())).map(|i| i * 10).collect();

    // setup 
    let (dataset, original_tensors_map) = setup_typed_dataset(
        num_tensors, &dims, key, &values, Kind::Int64
    );

    dataset.save_to_file(&file_path).expect("Failed to save I64 dataset");
    let loaded_dataset = SafetensorsDataset::load_from_file(&file_path)
        .expect("Failed to load I64 dataset");

    // Verify basics
    assert_eq!(loaded_dataset.len(), num_tensors as usize);
    assert!(loaded_dataset.contains_key(key));

    let loaded_tensors_vec = loaded_dataset.get_tensors(key).unwrap();
    let original_tensors_vec = original_tensors_map.get(key).expect("Original key missing");
    assert_eq!(loaded_tensors_vec.len(), original_tensors_vec.len());
    for (i, (orig_tensor, loaded_tensor)) in original_tensors_vec.iter().zip(loaded_tensors_vec.iter()).enumerate() {
        assert_eq!(loaded_tensor.size(), orig_tensor.size(), "I64 Shape mismatch tensor {}", i);
        assert_eq!(loaded_tensor.kind(), Kind::Int64, "I64 Kind mismatch tensor {}", i);
        assert!(orig_tensor.eq_tensor(loaded_tensor).all().int64_value(&[]) == 1, "I64 Data mismatch tensor {}", i);
    }
}

#[test]
fn test_load_safetensors_file_and_verify_bool_data() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("data_bool.safetensors");
    let num_tensors = 5i64;
    let dims = [10i64]; // Shape [10]
    let key = "test_bool";
    let context = "Bool Save/Load";

    // 1. Create original boolean data and tensors directly
    let values: Vec<bool> = (0..(num_tensors * dims.iter().product::<i64>()))
        .map(|i| i % 2 == 0) // Alternating true/false
        .collect();
    let chunk_size = dims.iter().product::<i64>() as usize;

    let original_tensors_list: Vec<Tensor> = values.chunks(chunk_size)
        .map(|chunk| Tensor::from_slice(chunk).reshape(dims))
        .collect();
    assert_eq!(original_tensors_list[0].kind(), Kind::Bool);

    let mut tensors_map = HashMap::new();
    tensors_map.insert(key.to_string(), original_tensors_list.iter().map(|t| t.shallow_clone()).collect());
    let dataset_to_save = SafetensorsDataset::from_dict(tensors_map).expect(&format!("[{}] Setup failed", context));

    dataset_to_save.save_to_file(&file_path).expect(&format!("[{}] Failed to save Bool dataset", context));

    let loaded_dataset = SafetensorsDataset::load_from_file(&file_path)
        .expect(&format!("[{}] Failed to load Bool dataset", context));

    // 5. Verify 
    assert_eq!(loaded_dataset.len(), num_tensors as usize, "[{}] Length mismatch", context);
    assert!(loaded_dataset.contains_key(key), "[{}] Key missing", context);
    let loaded_tensors_vec = loaded_dataset.get_tensors(key).unwrap();
    assert_eq!(loaded_tensors_vec.len(), original_tensors_list.len(), "[{}] List length mismatch", context);

    // Compare each tensor
    for (i, (orig_tensor, loaded_tensor)) in original_tensors_list.iter().zip(loaded_tensors_vec.iter()).enumerate() {
        let test_ctx = format!("[{} Tensor {}]", context, i);
        assert_eq!(loaded_tensor.size(), orig_tensor.size(), "{} Shape mismatch", test_ctx);
        assert_eq!(loaded_tensor.kind(), Kind::Bool, "{} Kind mismatch", test_ctx); // Check Kind
        // Use eq_tensor for exact comparison
        assert!(orig_tensor.eq_tensor(loaded_tensor).all().int64_value(&[]) == 1, "{} Data mismatch", test_ctx);
    }
}

#[test]
fn test_load_safetensors_file_and_verify_f64_data() { // Double
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("data_f64.safetensors");
    let num_tensors = 8i64;
    let dims = [3i64];
    let key = "test_f64";
    let values: Vec<f64> = (0..(num_tensors * dims.iter().product::<i64>())).map(|i| i as f64 * 1.1).collect();

    let (dataset, original_tensors_map) = setup_typed_dataset( 
        num_tensors, &dims, key, &values, Kind::Double
    );

    dataset.save_to_file(&file_path).expect("Failed to save F64 dataset");
    let loaded_dataset = SafetensorsDataset::load_from_file(&file_path)
        .expect("Failed to load F64 dataset");

    // Verify basics
    assert_eq!(loaded_dataset.len(), num_tensors as usize);
    assert!(loaded_dataset.contains_key(key));

    let loaded_tensors_vec = loaded_dataset.get_tensors(key).unwrap();
    let original_tensors_vec = original_tensors_map.get(key).expect("Original key missing");
    assert_eq!(loaded_tensors_vec.len(), original_tensors_vec.len());
    for (i, (orig_tensor, loaded_tensor)) in original_tensors_vec.iter().zip(loaded_tensors_vec.iter()).enumerate() {
        assert_eq!(loaded_tensor.size(), orig_tensor.size(), "F64 Shape mismatch tensor {}", i);
        assert_eq!(loaded_tensor.kind(), Kind::Double, "F64 Kind mismatch tensor {}", i);
        assert!(orig_tensor.allclose(loaded_tensor, 1e-6, 1e-6, false), "F64 Data mismatch tensor {}", i);
    }
}

#[test]
fn test_load_safetensors_file_and_verify_i32_data() { // Int
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("data_i32.safetensors");
    let num_tensors = 12i64;
    let dims = [2i64, 2i64];
    let key = "test_i32";
    let values: Vec<i32> = (0..(num_tensors * dims.iter().product::<i64>())).map(|i| (i as i32) - 10).collect();

    let (dataset, original_tensors_map) = setup_typed_dataset( 
        num_tensors, &dims, key, &values, Kind::Int
    );

    dataset.save_to_file(&file_path).expect("Failed to save I32 dataset");
    let loaded_dataset = SafetensorsDataset::load_from_file(&file_path)
        .expect("Failed to load I32 dataset");

    // Verify basics
    assert_eq!(loaded_dataset.len(), num_tensors as usize);
    assert!(loaded_dataset.contains_key(key));

    let loaded_tensors_vec = loaded_dataset.get_tensors(key).unwrap();
    let original_tensors_vec = original_tensors_map.get(key).expect("Original key missing");
    assert_eq!(loaded_tensors_vec.len(), original_tensors_vec.len());
    for (i, (orig_tensor, loaded_tensor)) in original_tensors_vec.iter().zip(loaded_tensors_vec.iter()).enumerate() {
         assert_eq!(loaded_tensor.size(), orig_tensor.size(), "I32 Shape mismatch tensor {}", i);
         assert_eq!(loaded_tensor.kind(), Kind::Int, "I32 Kind mismatch tensor {}", i);
         assert!(orig_tensor.eq_tensor(loaded_tensor).all().int64_value(&[]) == 1, "I32 Data mismatch tensor {}", i);
     }
}

#[test]
fn test_load_safetensors_file_and_verify_u8_data() { // Uint8
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("data_u8.safetensors");
    let num_tensors = 6i64;
    let dims = [5i64];
    let key = "test_u8";
    let values: Vec<u8> = (0..(num_tensors * dims.iter().product::<i64>()) as u8).collect();

    let (dataset, original_tensors_map) = setup_typed_dataset( 
        num_tensors, &dims, key, &values, Kind::Uint8
    );

    dataset.save_to_file(&file_path).expect("Failed to save U8 dataset");
    let loaded_dataset = SafetensorsDataset::load_from_file(&file_path)
        .expect("Failed to load U8 dataset");

    // Verify basics
     assert_eq!(loaded_dataset.len(), num_tensors as usize);
     assert!(loaded_dataset.contains_key(key));

    let loaded_tensors_vec = loaded_dataset.get_tensors(key).unwrap();
    let original_tensors_vec = original_tensors_map.get(key).expect("Original key missing");
     assert_eq!(loaded_tensors_vec.len(), original_tensors_vec.len());
      for (i, (orig_tensor, loaded_tensor)) in original_tensors_vec.iter().zip(loaded_tensors_vec.iter()).enumerate() {
         assert_eq!(loaded_tensor.size(), orig_tensor.size(), "U8 Shape mismatch tensor {}", i);
         assert_eq!(loaded_tensor.kind(), Kind::Uint8, "U8 Kind mismatch tensor {}", i);
         assert!(orig_tensor.eq_tensor(loaded_tensor).all().int64_value(&[]) == 1, "U8 Data mismatch tensor {}", i);
     }
}

#[test]
fn test_load_safetensors_file_and_verify_i8_data() { // Int8
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("data_i8.safetensors");
    let num_tensors = 4i64;
    let dims = [2i64];
    let key = "test_i8";
    let values: Vec<i8> = (0..(num_tensors * dims.iter().product::<i64>()) as i8).map(|i| i - 3).collect();

    let (dataset, original_tensors_map) = setup_typed_dataset( 
        num_tensors, &dims, key, &values, Kind::Int8
    );

    dataset.save_to_file(&file_path).expect("Failed to save I8 dataset");
    let loaded_dataset = SafetensorsDataset::load_from_file(&file_path)
        .expect("Failed to load I8 dataset");

    // Verify basics
    assert_eq!(loaded_dataset.len(), num_tensors as usize);
    assert!(loaded_dataset.contains_key(key));

    let loaded_tensors_vec = loaded_dataset.get_tensors(key).unwrap();
    let original_tensors_vec = original_tensors_map.get(key).expect("Original key missing");
     assert_eq!(loaded_tensors_vec.len(), original_tensors_vec.len());
      for (i, (orig_tensor, loaded_tensor)) in original_tensors_vec.iter().zip(loaded_tensors_vec.iter()).enumerate() {
         assert_eq!(loaded_tensor.size(), orig_tensor.size(), "I8 Shape mismatch tensor {}", i);
         assert_eq!(loaded_tensor.kind(), Kind::Int8, "I8 Kind mismatch tensor {}", i);
         assert!(orig_tensor.eq_tensor(loaded_tensor).all().int64_value(&[]) == 1, "I8 Data mismatch tensor {}", i);
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

// --- Rename Tests --- 
#[test]
fn test_rename_single_key() {
    let (mut dataset, original_tensors) = setup_multi_key_dataset();
    let key_mapping = vec![("features".to_string(), "inputs".to_string())];
    let context = "Rename single key";

    dataset.rename(&key_mapping).expect(&format!("[{}] Rename failed", context));
    
    assert_eq!(dataset.keys().len(), 2, "[{}] Dataset should still have 2 keys", context);
    assert!(!dataset.contains_key("features"), "[{}] Key 'features' should no longer exist", context);
    assert!(dataset.contains_key("inputs"), "[{}] Key 'inputs' should exist", context);
    assert!(dataset.contains_key("labels"), "[{}] Key 'labels' should still exist", context);

    // Verify tensor data remains correct 
    let input_tensors = dataset.get_tensors("inputs").unwrap(); 
    assert_eq!(input_tensors.len(), 3);
    for i in 0..3 {
        assert!(input_tensors[i].allclose(&original_tensors["features"][i], 1e-6, 1e-6, false),
        "[{}] Mismatch at inputs[{}]", context, i);
    }
    let labels_tensors = dataset.get_tensors("labels").unwrap(); // Verify unchanged key too
    assert_eq!(labels_tensors.len(), 3);
    for i in 0..3 {
         assert!(labels_tensors[i].eq_tensor(&original_tensors["labels"][i]).all().int64_value(&[]) == 1,
                "[{}] Mismatch at labels[{}]", context, i);
    }
}

#[test]
fn test_rename_multiple_keys() {
    let (mut dataset, original_tensors) = setup_multi_key_dataset();
    let key_mapping = vec![
        ("features".to_string(), "inputs".to_string()),
        ("labels".to_string(), "targets".to_string()),
    ];
    let context = "Rename multiple keys";

    dataset.rename(&key_mapping).expect(&format!("[{}] Rename failed", context));

    assert_eq!(dataset.keys().len(), 2, "[{}] Dataset should still have 2 keys", context);
    // Use contains_key
    assert!(!dataset.contains_key("features"), "[{}] Key 'features' should no longer exist", context);
    assert!(!dataset.contains_key("labels"), "[{}] Key 'labels' should no longer exist", context);
    assert!(dataset.contains_key("inputs"), "[{}] Key 'inputs' should exist", context);
    assert!(dataset.contains_key("targets"), "[{}] Key 'targets' should exist", context);

    // Verify tensor data remains correct
    let inputs_tensors = dataset.get_tensors("inputs").unwrap();
    assert_eq!(inputs_tensors.len(), 3);
    for i in 0..3 {
        assert!(inputs_tensors[i].allclose(&original_tensors["features"][i], 1e-6, 1e-6, false),
                "[{}] Mismatch at inputs[{}]", context, i);
    }
    let targets_tensors = dataset.get_tensors("targets").unwrap();
    assert_eq!(targets_tensors.len(), 3);
     for i in 0..3 {
         assert!(targets_tensors[i].eq_tensor(&original_tensors["labels"][i]).all().int64_value(&[]) == 1,
                "[{}] Mismatch at targets[{}]", context, i);
    }
}

#[test]
fn test_rename_error_nonexistent_old() {
    let (mut dataset, _) = setup_multi_key_dataset();
    let key_mapping = vec![("nonexistent".to_string(), "newkey".to_string())];
    let context = "Rename Err Nonexistent Old";

    let result = dataset.rename(&key_mapping);
    assert!(result.is_err(), "[{}] Renaming non-existent key should fail", context);

    // Check error type without panic!
    if let Err(err) = result {
        if let DataPrepError::InvalidKey(msg) = err {
            assert!(msg.contains("Key 'nonexistent' to rename does not exist"),
                    "[{}] Incorrect error message: {}", context, msg);
        } else {
            assert!(false, "[{}] Expected InvalidKey error, got {:?}", context, err);
        }
    }
}

#[test]
fn test_rename_error_conflict_existing() {
    let (mut dataset, _) = setup_multi_key_dataset();
    let key_mapping = vec![("features".to_string(), "labels".to_string())]; // "labels" already exists
    let context = "Rename Err Conflict Existing";

    let result = dataset.rename(&key_mapping);
    assert!(result.is_err(), "[{}] Renaming to existing key should fail", context);

    // Check error type without panic!
    if let Err(err) = result {
        if let DataPrepError::InvalidKey(msg) = err {
             assert!(msg.contains("New key 'labels' already exists"),
                    "[{}] Incorrect error message: {}", context, msg);
        } else {
             assert!(false, "[{}] Expected InvalidKey error, got {:?}", context, err);
        }
    }
}

#[test]
fn test_rename_error_duplicate_new() {
    let (mut dataset, _) = setup_multi_key_dataset();
    let key_mapping = vec![
        ("features".to_string(), "newkey".to_string()),
        ("labels".to_string(), "newkey".to_string()), // Duplicate "newkey"
    ];
     let context = "Rename Err Duplicate New";

    let result = dataset.rename(&key_mapping);
    assert!(result.is_err(), "[{}] Renaming with duplicate new keys should fail", context);

    // Check error type without panic!
    if let Err(err) = result {
        if let DataPrepError::InvalidKey(msg) = err {
             assert!(msg.contains("Duplicate new key 'newkey'"),
                    "[{}] Incorrect error message: {}", context, msg);
        } else {
             assert!(false, "[{}] Expected InvalidKey error, got {:?}", context, err);
        }
    }
}

// --- Test functions for info() ----
#[test]
fn test_info_standard_layout() {
    let (dataset, _) = setup_multi_key_dataset(); 
    let info = dataset.info(); 
    let context = "Standard layout";

    assert_eq!(info.len, 3, "[{}] Dataset length mismatch", context);
    assert_eq!(info.layouts.len(), 2, "[{}] Layout key count mismatch", context); 

    // Check "features" layout 
    if let Some(layout) = info.layouts.get("features") {
        if let TensorLayout::Standard {shape, dtype} = layout {
            assert_eq!(shape, &vec![1, 1], "[{}] Features shape mismatch", context);
            assert_eq!(*dtype, Kind::Float, "[{}] Features dtype mismatch", context);
        } else {
            assert!(false, "[{}] Expected Standard layout for 'features', got {:?}", context, layout);
        }
    } else{
        assert!(false, "[{}] Layout for 'features' missing", context);
    }

    // Check "labels" layout 
    if let Some(layout) = info.layouts.get("labels") {
        if let TensorLayout::Standard {shape, dtype} = layout {
            assert_eq!(shape, &Vec::<i64>::new(), "[{}] Labels shape mismatch", context);
            assert_eq!(*dtype, Kind::Int64, "[{}] Labels dtype mismatch", context);
        } else{
            assert!(false, "[{}] Expected Standard layout for 'labels', got {:?}", context, layout);
        }
    } else{
        assert!(false, "[{}] Layout for 'labels' missing", context);
    }
}

#[test]
fn test_info_varying_shape_layout() {
    let context = "Varying Shape Layout";
    // Setup dataset with varying shapes for "features"
    let features_list: Vec<Tensor> = vec![
        // *** FIX: Add f32 suffix to literals ***
        Tensor::f_from_slice(&[0.0f32]).unwrap().reshape(&[1, 1]), // Shape [1, 1]
        Tensor::f_from_slice(&[1.0f32, 2.0f32]).unwrap().reshape(&[2, 1]), // Shape [2, 1]
        Tensor::f_from_slice(&[3.0f32]).unwrap().reshape(&[1, 1]), // Shape [1, 1]
    ];
    // Use standard labels (no explicit device)
    let labels_list: Vec<Tensor> = (0..3)
        .map(|i| Tensor::from(i + 10).to_kind(Kind::Int64))
        .collect();

    let mut tensors_map = HashMap::new();
    tensors_map.insert("features".to_string(), features_list);
    tensors_map.insert("labels".to_string(), labels_list);
    let dataset = SafetensorsDataset::from_dict(tensors_map).expect("Setup failed");

    // Get info
    let info = dataset.info();
    assert_eq!(info.len, 3, "[{}] Dataset length mismatch", context);
    assert_eq!(info.layouts.len(), 2, "[{}] Layout key count mismatch", context);

    // Check "features" layout
    if let Some(layout) = info.layouts.get("features") {
        // Check if it's the correct variant and extract the dtype
        if let TensorLayout::VaryingDimSize { dtype } = layout {
             // Assert the dtype is Float (Kind::Float)
             assert_eq!(*dtype, Kind::Float, "[{}] Features dtype mismatch", context); // This should now pass
        } else {
             // Fail the test if the layout variant is not VaryingDimSize
             assert!(false, "[{}] Expected VaryingDimSize layout for 'features', got {:?}", context, layout);
        }
    } else {
         // Fail the test if the key "features" is missing
         assert!(false, "[{}] Layout for 'features' missing", context);
    }

    // Check "labels" layout (should still be standard)
     if let Some(layout) = info.layouts.get("labels") {
         if let TensorLayout::Standard { shape, dtype } = layout {
            // Check shape and dtype for labels
            assert_eq!(shape, &Vec::<i64>::new(), "[{}] Labels shape mismatch", context); // Explicit empty Vec<i64>
            assert_eq!(*dtype, Kind::Int64, "[{}] Labels dtype mismatch", context);
        } else {
            assert!(false, "[{}] Expected Standard layout for 'labels', got {:?}", context, layout);
        }
    } else {
         assert!(false, "[{}] Layout for 'labels' missing", context);
    }
}

#[test]
fn test_info_empty_dataset() {
    let keys = vec!["features".to_string(), "labels".to_string()];
    let empty_dataset = SafetensorsDataset::empty(keys.clone());
    let context = "Empty Dataset Info";

    let info = empty_dataset.info();
    assert_eq!(info.len, 0, "[{}] Empty dataset should have len 0", context);
    assert_eq!(info.layouts.len(), keys.len(), "[{}] Layout key count mismatch", context);

    // Empty lists should result in VaryingDtype according to info() implementation
    for key in keys {
        if let Some(layout) = info.layouts.get(&key) {
            if let TensorLayout::VaryingDtype = layout { /* Correct */ }
            else { assert!(false, "[{}] Expected VaryingDtype for key '{}', got {:?}", context, key, layout); }
        } else { assert!(false, "[{}] Layout for key '{}' missing", context, key); }
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

#[test]
fn test_select_indices() {
    let (dataset, original_tensors) = setup_multi_key_dataset(); 

    // Helper function to verify a selected dataset 
    fn verify_selected(
        selected: &SafetensorsDataset, 
        expected_len: usize, 
        expected_indices: &[usize], 
        original_tensors: &HashMap<String, Vec<Tensor>>,
    ) {
        assert_eq!(selected.len(), expected_len, "Selected dataset length mismatch");
        assert_eq!(selected.keys().len(), 2, "Selected dataset should have 2 keys");

        // Verify features 
        let features = selected.get_tensors("features").unwrap(); 
        assert_eq!(features.len(), expected_len, "Features length mismatch");
        for (i, &orig_idx) in expected_indices.iter().enumerate() {
            assert!(
                features[i].allclose(&original_tensors["features"][orig_idx], 1e-6, 1e-6, false),
                "Mismatch at features[{}] (original index {})", i, orig_idx
            );
        }

        // Verify labels 
        let labels = selected.get_tensors("labels").unwrap(); 
        assert_eq!(labels.len(), expected_len, "Labels length mismatch"); 
        for (i, &orig_idx) in expected_indices.iter().enumerate() {
            assert!(
                labels[i].eq_tensor(&original_tensors["labels"][orig_idx]).all().int64_value(&[]) == 1,
                "Mismatch at labels [{}] (original index {})", i, orig_idx
            );
        }
    }

    // Test Case 1: Select a subset in order
    let indices1 = [0, 2]; 
    let selected1 = dataset.select(&indices1).expect("Selecting valid indices [0, 2] failed");
    verify_selected(&selected1, 2, &indices1, &original_tensors);

    // Test Case 2: Select subset out of order with duplicates 
    let indices2 = [1, 0, 1]; 
    let selected2 = dataset.select(&indices2).expect("Selecting valid indices [1, 0, 1] failed");
    verify_selected(&selected2, 3, &indices2, &original_tensors);

    // Test Case 3: Select single element 
    let indices3 = [1]; 
    let selected3 = dataset.select(&indices3).expect("Selecting single index [1] failed");
    verify_selected(&selected3, 1, &indices3, &original_tensors);

    // Test Case 4: Select empty list 
    let indices4: [usize; 0] = [];
    let selected4 = dataset.select(&indices4).expect("Selecting empty indices [] failed");
    assert_eq!(selected4.len(), 0, "Selected dataset should have length 0");
    assert_eq!(selected4.keys().len(), 2, "Selected dataset should have 2 keys");
    assert!(selected4.get_tensors("features").unwrap().is_empty(), "Features should be empty");
    assert!(selected4.get_tensors("labels").unwrap().is_empty(), "Labels should be empty");

    // Test Case 5: Select with out-of-bounds index 
    let indices5 = [0, 3];
    let result = dataset.select(&indices5);
    assert!(result.is_err(), "Selecting indices [0, 3] should fail");
    let err = result.unwrap_err(); 
    match err {
        DataPrepError::Other(msg) => {
            assert!(
                msg.contains("Index 3 is out of bounds for dataset of length 3"),
                "Expected error message to contain 'Index 3 is out of bounds for dataset of length 3', got '{}'",
                msg
            );
        }
        _ => assert!(false, "Expected DataPrepError::Other, got {:?}", err),
    }
}

#[test]
fn test_map() {
    // Helper function to compare tensor values against expected values
    fn assert_tensor_values(
        tensors: &[Tensor],
        expected_values: &[f64],
        key: &str,
        kind: Kind,
    ) 
    {
        assert_eq!(
            tensors.len(),
            expected_values.len(),
            "Length mismatch for key '{}': expected {}, got {}",
            key,
            expected_values.len(),
            tensors.len()
        );
        for (i, (&value, tensor)) in expected_values.iter().zip(tensors.iter()).enumerate() {
            assert_eq!(
                tensor.kind(),
                kind,
                "Dtype mismatch for key '{}'[{}]: expected {:?}, got {:?}",
                key,
                i,
                kind,
                tensor.kind()
            );
            let tensor_value = if tensor.size().is_empty() {
                tensor.double_value(&[])
            } else {
                tensor.double_value(&[0, 0])
            };
            assert!(
                (tensor_value - value).abs() < 1e-6,
                "Value mismatch for key '{}'[{}]: expected {}, got {}",
                key,
                i,
                value,
                tensor_value
            );
        }
    }

    // Test Case 1: Map transforms values (add 10 to features, add 0.5 to labels and cast to float)
    let (dataset, _) = setup_multi_key_dataset(); // Original: features=[[0.]],[[1.]],[[2.]]; labels=10, 11, 12
    let mapped = dataset
        .map(|_i, row| {
            let mut new_row = HashMap::new();
            let features = *row.get("features").unwrap();
            new_row.insert("features".to_string(), features + 10.0f64);
            let labels = *row.get("labels").unwrap();
            let new_labels = labels.to_kind(Kind::Float) + 0.5f64;
            new_row.insert("labels".to_string(), new_labels);
            new_row
        })
        .expect("Mapping dataset failed");
    assert_eq!(mapped.len(), 3, "Mapped dataset should have length 3");
    assert_eq!(mapped.keys().len(), 2, "Mapped dataset should have 2 keys");
    assert_tensor_values(
        mapped.get_tensors("features").unwrap(),
        &[10.0, 11.0, 12.0],
        "features",
        Kind::Float,
    );
    assert_tensor_values(
        mapped.get_tensors("labels").unwrap(),
        &[10.5, 11.5, 12.5],
        "labels",
        Kind::Float,
    );

    // Test Case 2: Map fails on inconsistent keys (removes "labels" for second row)
    let (dataset, _) = setup_multi_key_dataset();
    let mapped_res = dataset.map(|i, row| {
        let mut new_row = HashMap::new();
        let features = *row.get("features").unwrap();
        new_row.insert("features".to_string(), features.shallow_clone());
        if i != 1 { // Second row has index 1 (0-based)
            let labels = *row.get("labels").unwrap();
            new_row.insert("labels".to_string(), labels.shallow_clone());
        }
        new_row
    });
    assert!(mapped_res.is_err(), "Map should fail with inconsistent keys");
    match mapped_res.err().unwrap() {
        DataPrepError::InvalidKey(msg) => {
            assert!(
                msg.contains("produced a row with 1 keys, expected 2"),
                "Expected error message about inconsistent keys, got '{}'",
                msg
            );
        }
        e => assert!(
            false,
            "Expected DataPrepError::InvalidKey, got {:?}", e
        ),
    }

    // Test Case 3: Map fails on inconsistent dtype (changes features dtype for second row)
    let (dataset, _) = setup_multi_key_dataset();
    let mapped_res = dataset.map(|i, row| {
        let mut new_row = HashMap::new();
        let features = *row.get("features").unwrap();
        if i == 1 { // Second row has index 1 (0-based)
            new_row.insert("features".to_string(), features.to_kind(Kind::Int64));
        } else {
            new_row.insert("features".to_string(), features.shallow_clone());
        }
        let labels = *row.get("labels").unwrap();
        new_row.insert("labels".to_string(), labels.shallow_clone());
        new_row
    });
    assert!(mapped_res.is_err(), "Map should fail with inconsistent dtypes");
    match mapped_res.err().unwrap() {
        DataPrepError::InconsistentTensorList(msg) => {
            assert!(
                msg.contains("produced a tensor with dtype Int64 for key 'features', expected dtype Float"),
                "Expected error message about dtype mismatch, got '{}'",
                msg
            );
        }
        e => assert!(
            false,
            "Expected DataPrepError::InconsistentTensorList, got {:?}", e
        ),
    }

    // Test Case 4: Map on empty dataset
    let empty_dataset = SafetensorsDataset::empty(vec!["features".to_string(), "labels".to_string()]);
    let mapped = empty_dataset
        .map(|_i, row| {
            let mut new_row = HashMap::new();
            let features = *row.get("features").unwrap();
            new_row.insert("features".to_string(), features + 1.0f64);
            let labels = *row.get("labels").unwrap();
            new_row.insert("labels".to_string(), labels.shallow_clone());
            new_row
        })
        .expect("Mapping empty dataset failed");
    assert_eq!(mapped.len(), 0, "Mapped empty dataset should have length 0");
    assert_eq!(mapped.keys().len(), 2, "Mapped empty dataset should have 2 keys");
    assert!(mapped.get_tensors("features").unwrap().is_empty(), "Features should be empty");
    assert!(mapped.get_tensors("labels").unwrap().is_empty(), "Labels should be empty");
}

#[test]
fn test_get_items() {
    let (dataset, original_tensors) = setup_multi_key_dataset(); // 3 rows: 0, 1, 2

    // --- Internal Helper Function for verifying get_items results ---
    fn verify_get_items_result(
        selected_items: &Vec<HashMap<String, &Tensor>>, // Input is the Vec of rows/items
        expected_len: usize,
        expected_indices: &[usize],
        original_tensors: &HashMap<String, Vec<Tensor>>,
        context: &str, // Context for error messages
    ) {
        // Check overall Vec length
        assert_eq!(selected_items.len(), expected_len, "[{}] Incorrect number of items returned", context);
        assert_eq!(selected_items.len(), expected_indices.len(), "[{}] Mismatch between expected len and expected indices len", context);

        // Check each item (row map) in the result vector
        for (i, item_map) in selected_items.iter().enumerate() {
            let original_index = expected_indices[i]; // Get the corresponding original index

            // --- Start of inlined logic from assert_row_matches ---
            assert_eq!(item_map.len(), original_tensors.len(), "[{}] Row {} key count mismatch", context, i);
            for (key, expected_list) in original_tensors {
                let row_tensor = item_map.get(key).expect(&format!("[{}] Key '{}' missing in row map {}", context, key, i));
                let original_tensor = &expected_list[original_index];

                // Compare shape, kind, and data
                assert_eq!(row_tensor.size(), original_tensor.size(), "[{}] Shape mismatch key '{}', row index {}", context, key, i);
                assert_eq!(row_tensor.kind(), original_tensor.kind(), "[{}] Kind mismatch key '{}', row index {}", context, key, i);

                if row_tensor.kind() == Kind::Float || row_tensor.kind() == Kind::Double {
                     assert!(original_tensor.allclose(row_tensor, 1e-6, 1e-6, false), "[{}] Data mismatch key '{}', row index {}", context, key, i);
                } else {
                     assert!(original_tensor.eq_tensor(row_tensor).all().int64_value(&[]) == 1, "[{}] Data mismatch key '{}', row index {}", context, key, i);
                }
            }
        }
    }

    // --- Test Case 1: Select subset [0, 2] ---
    let indices1 = [0, 2];
    let selected_items1 = dataset.get_items(&indices1).expect("Selecting [0, 2] failed");
    verify_get_items_result(&selected_items1, 2, &indices1, &original_tensors, "Select [0, 2]");

    // --- Test Case 2: Select subset [1, 0, 1] (reorder, duplicate) ---
     let indices2 = [1, 0, 1];
     let selected_items2 = dataset.get_items(&indices2).expect("Selecting [1, 0, 1] failed");
     verify_get_items_result(&selected_items2, 3, &indices2, &original_tensors, "Select [1, 0, 1]");

    // --- Test Case 3: Select single [1] ---
    let indices3 = [1];
    let selected_items3 = dataset.get_items(&indices3).expect("Selecting [1] failed");
    verify_get_items_result(&selected_items3, 1, &indices3, &original_tensors, "Select [1]");

    // --- Test Case 4: Select empty list ---
    let indices4: [usize; 0] = [];
    let selected_items4 = dataset.get_items(&indices4).expect("Selecting empty [] failed");
    assert!(selected_items4.is_empty(), "Selecting empty indices should yield empty Vec");

    // --- Test Case 5: Select with out-of-bounds index ---
    let indices5 = [0, 3]; // Dataset length is 3, so index 3 is invalid
    let result5 = dataset.get_items(&indices5);
    assert!(result5.is_err(), "Selecting indices [0, 3] should fail");

    // Check error type using if let and assert!
    if let Err(err) = result5 { // Check if it's an Err
        if let DataPrepError::Other(msg) = err { // Check if it's the correct variant
            // Optionally check the message content
            println!("Got expected error: {}", msg); // Optional print for debugging
            assert!(msg.contains("Index 3 is out of bounds"), 
            "Expected message about index out of bounds, got: {}", msg);
        } else {
            // If it's an Err, but not the right variant, fail the test using assert!
            assert!(false, "Expected DataPrepError::Other for out of bounds error, got {:?}", err);
        }
    } 
}

#[test]
fn test_filter() {
    // Helper function to compare two tensor lists 
    fn assert_tensor_lists_match(
        filtered: &[Tensor],
        original: &[Tensor],
        expected_indices: &[usize],
        key: &str,
    ){
        assert_eq!(
            filtered.len(), expected_indices.len(), 
            "Length mismatch for key '{}': expected {}, got {}", key, expected_indices.len(), filtered.len());
        
        for (i, &orig_idx) in expected_indices.iter().enumerate() {
            if key == "features" {
                assert!(
                    filtered[i].allclose(&original[orig_idx], 1e-6, 1e-6, false),
                    "Mismatch at {}[{}] (original index {})", key, i, orig_idx
                );
            } else {
                assert!(
                    filtered[i].eq_tensor(&original[orig_idx]).all().int64_value(&[]) == 1, 
                    "Mismatch at {}[{}] (original index {}", key, i, orig_idx
                );
            }
        }
    }

    // Test Case 1: Filter keeps a subset (labels > 10, keeps rows 1 and 2)
    let (dataset, original_tensors) = setup_multi_key_dataset(); 
    let filtered = dataset
        .filter(|row| row.get("labels").unwrap().int64_value(&[]) > 10) 
        .expect("Filtering dataset failed");
    assert_eq!(filtered.len(), 2, "Filtered dataset should have length 2");
    assert_eq!(filtered.keys().len(), 2, "Filtered dataset should have 2 keys");
    assert_tensor_lists_match(
        filtered.get_tensors("features").unwrap(), 
        &original_tensors["features"], 
        &[1, 2], 
        "features",
    );
    assert_tensor_lists_match(
        filtered.get_tensors("labels").unwrap(), 
        &original_tensors["labels"],
        &[1, 2],
        "labels",
    );

    // Test Case 2: Filter keeps none (labels < 0, keeps no rows)
    let filtered = dataset
        .filter(|row| row.get("labels").unwrap().int64_value(&[]) < 0)
        .expect("Filtering dataset failed");
    assert_eq!(filtered.len(), 0, "Filtered dataset should have length 0");
    assert_eq!(filtered.keys().len(), 2, "Filtered dataset should have 2 keys");
    assert!(filtered.get_tensors("features").unwrap().is_empty(), "Features should be empty");
    assert!(filtered.get_tensors("labels").unwrap().is_empty(), "Labels should be empty");

    // Test Case 3: Filter keeps all (always true)
    let filtered = dataset
        .filter(|_| true)
        .expect("Filtering dataset failed");
    assert_eq!(filtered.len(), 3, "Filtered dataset should have length 3");
    assert_eq!(filtered.keys().len(), 2, "Filtered dataset should have 2 keys");
    assert_tensor_lists_match(
        filtered.get_tensors("features").unwrap(), 
        &original_tensors["features"],
        &[0, 1, 2],
        "features",
    );
    assert_tensor_lists_match(
        filtered.get_tensors("labels").unwrap(), 
        &original_tensors["labels"],
        &[0, 1, 2], 
        "labels",
    );

    // Test Case 4: Filter on empty dataset 
    let empty_dataset = SafetensorsDataset::empty(
        vec!["features".to_string(), 
        "labels".to_string()]
    );
    let filtered = empty_dataset
        .filter(|_| true)
        .expect("Filtering empty dataset failed");
    assert_eq!(filtered.len(), 0, "Filtered empty dataset should have length 0");
    assert_eq!(filtered.keys().len(), 2, "Filtered empty dataset should have 2 keys");
    assert!(filtered.get_tensors("features").unwrap().is_empty(), "Features should be empty");
    assert!(filtered.get_tensors("labels").unwrap().is_empty(), "Labels should be empty");
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
// TODO: test_info_varying_dtype_layout: Not yet supported as we assume that our tensors have the same dtype; 
