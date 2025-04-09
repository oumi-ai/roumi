mod common; 
use common::setup_multi_key_dataset; 
use std::collections::HashMap;
use tch::Kind;

/// Tests for get_row, get_rows, select, filter, map
#[test]
fn test_get_row_and_get_rows() {
    let (dataset, original) = setup_multi_key_dataset();

    // get_row(0)
    let row0_opt = dataset.get_row(0);
    assert!(row0_opt.is_some(), "Row 0 should exist");
    let row0 = row0_opt.unwrap();
    assert_eq!(row0.len(), 2, "Should have 2 keys in row 0");
    assert!(
        row0["features"].allclose(&original["features"][0], 1e-6, 1e-6, false),
        "Mismatch in row 0 features"
    );

    // get_rows([1, 0, 2])
    let rows = dataset.get_rows(&[1, 0, 2]).expect("get_rows should succeed");
    assert_eq!(rows.len(), 3, "Should have 3 row-maps");
    // Spot check
    assert_eq!(rows[0]["labels"].int64_value(&[]), 11, "Mismatch at row 1's labels");
}

#[test]
fn test_select() {
    let (dataset, original) = setup_multi_key_dataset();
    let subset = dataset.select(&[0, 2]).expect("Select should succeed");
    assert_eq!(subset.len(), 2, "Should have 2 rows in the new dataset");

    let feats = subset.get_tensors("features").unwrap();
    assert_eq!(feats.len(), 2);
    assert!(feats[0].allclose(&original["features"][0], 1e-6, 1e-6, false));
    assert!(feats[1].allclose(&original["features"][2], 1e-6, 1e-6, false));

    // Out-of-bounds
    let result = dataset.select(&[3]);
    assert!(result.is_err(), "Index 3 out of range should fail");
}

#[test]
fn test_filter() {
    let (dataset, original) = setup_multi_key_dataset();
    // labels > 10 => keep rows [1, 2]
    let filtered = dataset
        .filter(|row| row["labels"].int64_value(&[]) > 10)
        .expect("Filter failed");
    assert_eq!(filtered.len(), 2, "Should keep 2 rows");
    assert!(
        filtered.get_tensors("labels").unwrap()[0]
            .eq_tensor(&original["labels"][1])
            .all()
            .int64_value(&[])
            == 1
    );

    // filter => none
    let none_filtered = dataset
        .filter(|row| row["labels"].int64_value(&[]) < 0)
        .expect("Filter of none failed");
    assert_eq!(none_filtered.len(), 0, "Should keep 0 rows");
}

#[test]
fn test_map_transform() {
    let (dataset, _) = setup_multi_key_dataset();
    // features + 10, labels -> to float + 0.5
    let mapped = dataset
        .map(|_i, row| {
            let mut new_map = HashMap::new();
            let feats = row["features"].shallow_clone() + 10.0f64;
            new_map.insert("features".to_string(), feats);

            let labs_float = row["labels"].to_kind(Kind::Float) + 0.5;
            new_map.insert("labels".to_string(), labs_float);

            new_map
        })
        .expect("Map failed");

    assert_eq!(mapped.len(), 3, "Mapped dataset should have 3 rows");
    let feats = mapped.get_tensors("features").unwrap();
    let labs = mapped.get_tensors("labels").unwrap();
    assert_eq!(feats[0].double_value(&[0, 0]), 10.0, "features row0 mismatch");
    assert_eq!(labs[0].double_value(&[]), 10.5, "labels row0 mismatch");
}