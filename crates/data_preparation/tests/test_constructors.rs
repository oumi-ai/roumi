use data_preparation::{DataPrepError, SafetensorsDataset};
use std::collections::HashMap;
use tch::Tensor; 

/// Basic constructor tests for from_dict, empty, etc. 

#[test]
fn test_from_dict_allows_empty_lists() {
    let mut tensors_map = HashMap::new();
    tensors_map.insert("empty_list".to_string(), vec![]); // Now allowed
    let result = SafetensorsDataset::from_dict(tensors_map);
    assert!(result.is_ok(), "Expected from_dict to allow empty lists");
}

#[test]
fn test_empty_dataset() {
    let dataset = SafetensorsDataset::empty(vec!["features".to_string(), "labels".to_string()]);
    assert!(dataset.is_empty(), "Newly created empty dataset should be empty");
    assert_eq!(dataset.keys().len(), 2, "Should have 2 keys with empty vectors");
}

#[test]
fn test_inconsistent_dtype_fails() {
    // Create a map with two different dtypes under the same key
    let mut map = HashMap::new();
    let t1 = Tensor::from(1i64);
    let t2 = Tensor::from(2.0f32); // different dtype
    map.insert("inconsistent".to_string(), vec![t1, t2]);

    let result = SafetensorsDataset::from_dict(map);
    assert!(result.is_err(), "Expected error for inconsistent dtypes");

    if let Err(err) = result {
        if let DataPrepError::InconsistentTensorList(msg) = err {
            assert!(
                msg.contains("Inconsistent dtypes found in list for key 'inconsistent'"),
                "Wrong error message: {}",
                msg
            );
        } else {
            panic!("Expected InconsistentTensorList, got {:?}", err);
        }
    }
}
