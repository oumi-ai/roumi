// tests/helpers.rs

#![allow(dead_code)]
use data_preparation::*;
use std::collections::HashMap;
use tch::{kind, Kind, Tensor};

/// Creates a `SafetensorsDataset` with a single key containing
/// `num_tensors` tensors, each of shape `[dim_size]`, filled with ones.
///
/// # Panics
/// - If dataset creation fails.
pub fn setup_float_dataset_for_test(num_tensors: i64, dim_size: i64, key: &str) -> SafetensorsDataset {
    let tensor_shape = [num_tensors, dim_size];
    let test_data = Tensor::ones(&tensor_shape, kind::FLOAT_CPU);

    let test_tensors: Vec<Tensor> = (0..num_tensors)
        .map(|i| test_data.index_select(0, &Tensor::from_slice(&[i])))
        .collect();
    
    let mut tensors_map = HashMap::new();
    tensors_map.insert(key.to_string(), test_tensors);
    
    SafetensorsDataset::from_dict(tensors_map).expect("Setup failed")
}

/// Returns a simple multi-key `SafetensorsDataset` containing:
/// - "features": 3 rows of shape [1,1], values: [0.], [1.], [2.]
/// - "labels": 3 scalar tensors: 10, 11, 12
/// Also returns a clone of the original tensors for easy verification.
pub fn setup_multi_key_dataset() -> (SafetensorsDataset, HashMap<String, Vec<Tensor>>) {
    let num_rows = 3; 
    
    let features_list: Vec<Tensor> = (0..num_rows)
        .map(|i| Tensor::f_from_slice(&[i as f32]).unwrap().reshape(&[1, 1]))
        .collect(); 

    let labels_list: Vec<Tensor> = (0..num_rows)
        .map(|i| Tensor::from(i+10).to_kind(Kind::Int64))
        .collect(); 

    let mut tensors_map = HashMap::new(); 
    tensors_map.insert(
        "features".to_string(), 
        features_list.iter().map(|t| t.shallow_clone()).collect());
    tensors_map.insert(
        "labels".to_string(), 
        labels_list.iter().map(|t| t.shallow_clone()).collect());
    
    let dataset = SafetensorsDataset::from_dict(tensors_map).expect("Setup failed");
    
    let original_tensors = HashMap::from([
        ("features".to_string(), features_list),
        ("labels".to_string(), labels_list),
    ]);
    
    (dataset, original_tensors)
}

/// Creates a typed `SafetensorsDataset` for a given dtype and shape using raw values.
///
/// # Panics
/// - If the number of values doesn't match `num_tensors * product(dims)`.
/// - If dataset creation fails.
pub fn setup_typed_dataset<T>(
    num_tensors: i64,
    dims: &[i64],
    key: &str,
    values: &[T],
    kind: Kind,
) -> (SafetensorsDataset, HashMap<String, Vec<Tensor>>)
where
    T: tch::kind::Element + Copy,
{
    let total_needed = num_tensors * dims.iter().product::<i64>();
    assert_eq!(
        values.len() as i64, total_needed,
        "Incorrect number of values provided: need {}, got {}",
        total_needed, values.len()
    );
    let original_tensor_list: Vec<Tensor> = values
        .chunks(dims.iter().product::<i64>() as usize)
        .map(|chunk| {
            Tensor::from_slice(chunk)
                .to_kind(kind)
                .reshape(dims)
        })
        .collect();

    let mut verification_map = HashMap::new();
    verification_map.insert(
        key.to_string(),
        original_tensor_list.iter().map(|t| t.shallow_clone()).collect(),
    );

    let mut dataset_map = HashMap::new();
    dataset_map.insert(key.to_string(), original_tensor_list);

    let dataset = SafetensorsDataset::from_dict(dataset_map).expect("Setup failed");
    (dataset, verification_map)
}