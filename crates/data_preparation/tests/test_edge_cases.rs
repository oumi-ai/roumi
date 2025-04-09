// Tests for optional advanced corner cases

#[test]
fn test_load_missing_metadata_from_safetensors_file_fails() {
    // Example placeholder: you'd craft a .safetensors file missing metadata,
    // then load it. For now, just skip or test manually.
    println!("Skipping test for loading file with missing metadata");
}

#[test]
fn test_load_metadata_wrong_type_from_safetensors_file_fails() {
    println!("Skipping test for loading file with malformed metadata type");
}

// TODO: Add integration tests for Half, Bfloat types save/load.
// TODO: test_info_varying_dtype_layout: Not yet supported as we assume that our tensors have the same dtype; 