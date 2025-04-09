mod common;
use common::setup_multi_key_dataset;
use data_preparation::{SafetensorsDataset, TensorLayout};
use std::collections::HashMap;
use tch::Tensor;

/// Tests for info()

#[test]
fn test_info_standard_layout() {
    let (dataset, _) = setup_multi_key_dataset();
    let info = dataset.info();

    assert_eq!(info.len, 3, "Should have 3 rows");
    assert_eq!(info.layouts.len(), 2, "Should have 2 keys in layout");

    // features => Standard shape [1,1], dtype Float
    let layout_feats = info
        .layouts
        .get("features")
        .expect("features layout missing");
    match layout_feats {
        TensorLayout::Standard { shape, dtype } => {
            assert_eq!(shape, &vec![1, 1], "Shape mismatch for features");
            assert_eq!(*dtype, tch::Kind::Float, "Dtype mismatch for features");
        }
        _ => panic!("Expected Standard layout for features"),
    }

    // labels => Standard shape [], dtype Int64
    let layout_labels = info.layouts.get("labels").expect("labels layout missing");
    match layout_labels {
        TensorLayout::Standard { shape, dtype } => {
            assert_eq!(shape, &Vec::<i64>::new(), "Shape mismatch for labels");
            assert_eq!(*dtype, tch::Kind::Int64, "Dtype mismatch for labels");
        }
        _ => panic!("Expected Standard layout for labels"),
    }
}

#[test]
fn test_info_varying_dim_size() {
    // features have varying shapes => VaryingDimSize
    let feats = vec![
        Tensor::from_slice(&[0.0f32]).reshape(&[1, 1]),
        Tensor::from_slice(&[1.0f32, 2.0]).reshape(&[2, 1]),
    ];
    let labels = vec![Tensor::from(10i64), Tensor::from(11i64)];

    let mut map = HashMap::new();
    map.insert("features".to_string(), feats);
    map.insert("labels".to_string(), labels);

    let ds = SafetensorsDataset::from_dict(map).expect("Setup failed");
    let info = ds.info();

    let feats_layout = info
        .layouts
        .get("features")
        .expect("Missing features layout");
    match feats_layout {
        TensorLayout::VaryingDimSize { dtype } => {
            assert_eq!(*dtype, tch::Kind::Float, "Dtype mismatch for varying dims");
        }
        _ => panic!("Expected VaryingDimSize for features"),
    }
}
