mod common;
use common::setup_multi_key_dataset;
use data_preparation::{DataPrepError, SafetensorsDataset};

/// Tests for len, is_empty, keys, contains_key, get_tensors, rename, etc.

#[test]
fn test_len_and_is_empty() {
    // Using multi-key dataset from helper
    let (dataset, _) = setup_multi_key_dataset();
    assert_eq!(dataset.len(), 3, "Expected 3 rows");
    assert!(!dataset.is_empty(), "Should not be empty");

    let empty = SafetensorsDataset::empty(vec!["features".to_string()]);
    assert_eq!(empty.len(), 0, "Empty dataset has len 0");
    assert!(empty.is_empty(), "Empty dataset is empty");
}

#[test]
fn test_keys_and_contains_key() {
    let (dataset, _) = setup_multi_key_dataset();
    let keys = dataset.keys();

    assert_eq!(keys.len(), 2, "Should have 'features' and 'labels'");
    assert!(dataset.contains_key("features"), "Key 'features' missing");
    assert!(dataset.contains_key("labels"), "Key 'labels' missing");
    assert!(!dataset.contains_key("unknown"), "Unexpected key found");
}

#[test]
fn test_get_tensors() {
    let (dataset, original) = setup_multi_key_dataset();
    let feats_opt = dataset.get_tensors("features");
    assert!(feats_opt.is_some(), "Should have 'features' key");
    let feats = feats_opt.unwrap();
    assert_eq!(feats.len(), 3, "Expected 3 rows of features");
    assert!(
        feats[0].allclose(&original["features"][0], 1e-6, 1e-6, false),
        "Mismatch in first features row"
    );
}

#[test]
fn test_rename_single_key() {
    let (mut dataset, original) = setup_multi_key_dataset();
    let mappings = [("features".to_string(), "inputs".to_string())];

    dataset.rename(&mappings).expect("Rename should succeed");
    assert!(dataset.contains_key("inputs"), "Key 'inputs' missing");
    assert!(
        !dataset.contains_key("features"),
        "Old key 'features' should be gone"
    );

    let inputs = dataset.get_tensors("inputs").unwrap();
    assert_eq!(inputs.len(), 3);
    for i in 0..3 {
        assert!(
            inputs[i].allclose(&original["features"][i], 1e-6, 1e-6, false),
            "Mismatch at inputs[{}]",
            i
        );
    }
}

#[test]
fn test_rename_error_nonexistent_old() {
    let (mut dataset, _) = setup_multi_key_dataset();
    let result = dataset.rename(&[("nonexistent".to_string(), "newkey".to_string())]);
    assert!(result.is_err(), "Renaming nonexistent key should fail");

    if let Err(err) = result {
        match err {
            DataPrepError::InvalidKey(msg) => {
                assert!(
                    msg.contains("Key 'nonexistent' to rename does not exist"),
                    "Wrong error message: {}",
                    msg
                );
            }
            _ => panic!("Expected InvalidKey, got {:?}", err),
        }
    }
}
