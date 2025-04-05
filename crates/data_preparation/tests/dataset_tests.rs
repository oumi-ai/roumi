use data_preparation::Dataset; 
use std::collections::HashMap; 
use tch::{kind, Tensor};

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
