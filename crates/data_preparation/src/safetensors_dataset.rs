// src/safetensors.rs
use crate::dataset::Dataset;
use crate::error::{DataPrepError, Result};
use crate::info::{DatasetInfo, TensorLayout};

use safetensors::{serialize, tensor::TensorView, Dtype, SafeTensors};
use serde_json::{self, Value};
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use tch::{Kind, Tensor};

/// --- Struct Definition ---
///
/// A wrapper around `Dataset` that stores 'HashMap<String, Vec<Tensor>>`
/// and supports saving/loading to the `.safetensors` format.
///
/// Provides row-level access/manipulation (e.g., `get_row`, `filter`, `map`)
/// and enforces consistent dtypes within each key.
#[derive(Debug)]
pub struct SafetensorsDataset {
    // The internal dataset storage.
    dataset: Dataset,
}

impl SafetensorsDataset {
    // ---------------------------------------------------------------------------------------
    // Constructors
    // ---------------------------------------------------------------------------------------

    /// Creates a new `SafetensorsDataset` from a map of tensor lists.
    ///
    /// # Arguments
    /// - 'tensors': A map where each key is a feature name (e.g., "inputs", "labels") and
    ///              the value is a non-empty `Vec<Tensor>` with consistent `Kind` (dtype).
    ///
    /// # Errors
    /// - ` DataPrepError::InconsistentTensorList` if:
    ///    1) Any list is empty,
    ///    2) Any list contains mixed dtypes.
    ///
    /// # Example
    /// ```rust
    /// # use data_preparation::SafetensorsDataset;
    /// # use tch::Tensor;
    /// # use std::collections::HashMap;
    /// let mut map = HashMap::new();
    /// map.insert("key1".to_string(), vec![Tensor::from(1i64)]);
    /// let dataset = SafetensorsDataset::from_dict(map).unwrap();
    /// assert_eq!(dataset.len(), 1);
    /// ```
    pub fn from_dict(tensors: HashMap<String, Vec<Tensor>>) -> Result<Self> {
        // Check each key for non-emptiness and dtype consistency.
        for (key, value) in &tensors {
            if value.is_empty() {
                continue;
            }
            // if value.is_empty() {
            //   return Err(DataPrepError::InconsistentTensorList(format!(
            //        "Input tensor list for key '{}' cannot be empty.", key
            //    )));
            //}
            let first_kind = value[0].kind();
            if !value.iter().all(|t| t.kind() == first_kind) {
                return Err(DataPrepError::InconsistentTensorList(format!(
                    "Inconsistent dtypes found in list for key '{}'. Expected {:?}",
                    key, first_kind
                )));
            }
        }
        Ok(SafetensorsDataset {
            dataset: Dataset::new(tensors),
        })
    }

    /// Creates an empty `SafetensorsDataset`, initializing each provided key
    /// with an empty vector of tensors.
    ///
    /// This is useful for setting up a dataset structure that you plan to fill later
    /// or for testing code that expects a dataset but does not require data.
    ///
    /// # Example
    /// ```rust
    /// # use data_preparation::SafetensorsDataset;
    /// let dataset = SafetensorsDataset::empty(vec!["input".to_string()]);
    /// assert!(dataset.is_empty());
    /// ```
    pub fn empty(keys: Vec<String>) -> Self {
        let mut empty_tensors = HashMap::new();
        for key in keys {
            empty_tensors.insert(key, Vec::new());
        }
        SafetensorsDataset {
            dataset: Dataset::new(empty_tensors),
        }
    }

    // ---------------------------------------------------------------------------------------
    // Basic Accessors
    // ---------------------------------------------------------------------------------------

    /// Returns the number of rows in the dataset (based on the first non-empty tensor list).
    ///
    /// # Warning
    /// This method does not ensure all keys have the same length. It simply returns the
    /// length of the first non-empty list it encounters. If different keys have differen
    /// lengths, row-based operations (e.g., ['get_row']) may fail or return `None`
    /// for out-of-bound indices.
    ///
    /// # TODO
    /// - Add support for variable-length tensors
    ///
    /// # Example
    /// ```rust
    /// # use data_preparation::SafetensorsDataset;
    /// # use tch::Tensor;
    /// # use std::collections::HashMap;
    /// let dataset = SafetensorsDataset::from_dict(HashMap::from([
    ///     ("key".to_string(), vec![Tensor::from(1), Tensor::from(2)])
    /// ])).unwrap();
    /// assert_eq!(dataset.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Returns `true` if the dataset contains no rows.
    ///
    /// A dataset is empty if:
    /// - It has no keys, or
    /// - Every key's vector of tensors is empty.
    ///
    /// # Example
    /// ```rust
    /// # use data_preparation::SafetensorsDataset;
    /// let dataset = SafetensorsDataset::empty(vec!["key".to_string()]);
    /// assert!(dataset.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }

    /// Returns a set of references to all the keys in this dataset.
    ///
    /// # Example
    /// ```rust
    /// # use data_preparation::SafetensorsDataset;
    /// # use tch::Tensor;
    /// # use std::collections::{HashMap, HashSet};
    /// let dataset = SafetensorsDataset::from_dict(HashMap::from([
    ///     ("key1".to_string(), vec![Tensor::from(1)]),
    ///     ("key2".to_string(), vec![Tensor::from(2)])
    /// ])).unwrap();
    /// let keys = dataset.keys();
    /// assert!(keys.iter().any(|k| *k == "key1"));
    /// assert!(keys.iter().any(|k| *k == "key2"));
    /// ```
    pub fn keys(&self) -> HashSet<&String> {
        self.dataset.tensors.keys().collect()
    }

    /// Returns `true` if the dataset contains the specified key.
    ///
    /// # Arguments
    /// - `key`: The feature name to check for
    ///
    /// # Example
    /// ```rust
    /// # use data_preparation::SafetensorsDataset;
    /// # use tch::Tensor;
    /// # use std::collections::HashMap;
    /// let dataset = SafetensorsDataset::from_dict(HashMap::from([
    ///     ("key1".to_string(), vec![Tensor::from(1)])
    /// ])).unwrap();
    /// assert!(dataset.contains_key("key1"));
    /// assert![!dataset.contains_key("missing")];
    /// ```
    pub fn contains_key(&self, key: &str) -> bool {
        self.dataset.tensors.contains_key(key)
    }

    /// Returns a reference to the `Vec<Tensor>` for a given key, if it exists.
    ///
    /// # Example
    /// ```rust
    /// # use data_preparation::SafetensorsDataset;
    /// # use tch::Tensor;
    /// # use std::collections::HashMap;
    /// let dataset = SafetensorsDataset::from_dict(HashMap::from([
    ///     ("features".to_string(), vec![Tensor::from(1), Tensor::from(2)])
    /// ])).unwrap();
    /// let maybe_tensors = dataset.get_tensors("features");
    /// assert!(maybe_tensors.is_some());
    /// assert_eq!(maybe_tensors.unwrap().len(), 2);
    /// ```
    pub fn get_tensors(&self, key: &str) -> Option<&Vec<Tensor>> {
        self.dataset.tensors.get(key)
    }

    // ---------------------------------------------------------------------------------------
    // Row-level Access / Manipulation
    // ---------------------------------------------------------------------------------------

    /// Retrieves a single "row" by index, returning a `HashMap<String, &Tensor>`
    /// where each entry is a reference tensor of that index for the
    /// corresponding key.
    ///
    /// # Warning
    /// If your dataset has keys with different lengths, this returns
    /// `None` for any index that is out of bounds for at least one key.
    ///
    /// # Differences vs. [`get_rows`]
    /// - `get_row(index)` returns exactly one `Option<HashMap<String, &Tensor>>`.
    /// - `get_rows(indices)` can getch multiple rows at once (returning a `Vec`).
    ///
    /// # Differences vs. [`select`]
    /// - `get_row` is read-only and returns a single row by reference.
    /// - `select` produces an entirely new `SafetensorsDataset` tha
    ///    contains only the specified rows (by shallow copy)
    ///
    /// # Example
    ///  ```rust
    /// # use data_preparation::SafetensorsDataset;
    /// # use tch::Tensor;
    /// # use std::collections::HashMap;
    /// let dataset = SafetensorsDataset::from_dict(HashMap::from([
    ///     ("key".to_string(), vec![Tensor::from(1), Tensor::from(2)])
    /// ])).unwrap();
    ///
    /// // Retrieve the second row (index 1)
    /// let row = dataset.get_row(1).unwrap();
    /// assert_eq!(row["key"].int64_value(&[]), 2);
    /// ```
    pub fn get_row(&self, index: usize) -> Option<HashMap<String, &Tensor>> {
        if index >= self.len() {
            return None;
        }
        let mut row = HashMap::new();
        for (key, tensors) in &self.dataset.tensors {
            if index < tensors.len() {
                row.insert(key.clone(), &tensors[index]);
            } else {
                return None;
            }
        }
        Some(row)
    }

    /// Retrieves multiple rows at once, returning a `Vec<HashMap<String, &Tensor>>`.
    ///
    /// Each element in the returned `Vec` corresponds to one row at the specified
    /// index across all keys. If any requested index is out of range for any key,
    /// an error is returned.
    ///
    /// # Differences vs. [`get_row`]
    /// - `get_rows` allows you to fetch many rows simultaneously (`&[0. 2. 5]`, etc. )
    /// - `get_row(index)` only fetches a single row and returns an `Option` rather than
    /// an entire `Vec`.
    ///
    /// # Differences vs. [`select`]
    /// - `get_rows` is read only: you get references to existing tensors.
    /// - `select` constructs a **new** `SafetensorsDataset` that has exactly those rows.
    ///   In other words, `select(&[0, 2])` returns a `SafetensorsDataset` with 2 rows,
    ///   while `get_rows(&[0, 2])` returns a `Vec` of 2 row-maps for the *existing* dataset.
    ///
    /// # Example
    /// ```rust
    /// # use data_preparation::SafetensorsDataset;
    /// # use tch::Tensor;
    /// # use std::collections::HashMap;
    /// let dataset = SafetensorsDataset::from_dict(HashMap::from([
    ///     ("key1".to_string(), vec![Tensor::from(10), Tensor::from(20), Tensor::from(30)]),
    ///     ("key2".to_string(), vec![Tensor::from(100), Tensor::from(200), Tensor::from(300)]),
    /// ])).unwrap();
    ///
    /// // Retrieve the 0th and 2nd rows
    /// let rows = dataset.get_rows(&[0, 2]).unwrap();
    /// assert_eq!(rows.len(), 2);
    /// assert_eq!(rows[0]["key1"].int64_value(&[]), 10);
    /// assert_eq!(rows[0]["key2"].int64_value(&[]), 100);
    /// assert_eq!(rows[1]["key1"].int64_value(&[]), 30);
    /// assert_eq!(rows[1]["key2"].int64_value(&[]), 300);
    /// ```
    pub fn get_rows(&self, indices: &[usize]) -> Result<Vec<HashMap<String, &Tensor>>> {
        let len = self.len();
        for &index in indices {
            if index >= len {
                return Err(DataPrepError::Other(format!(
                    "Index {} is out of bounds for dataset of length {}.",
                    index, len
                )));
            }
        }
        let mut result = Vec::with_capacity(indices.len());
        for &index in indices {
            let row = self.get_row(index).ok_or_else(|| {
                DataPrepError::Other(format!("Failed to retrieve row at index {}", index))
            })?;
            result.push(row);
        }
        Ok(result)
    }

    /// Returns a new `SafetensorsDataset` containing only the rows at the
    /// specified indices. For each key, only the tensors at those indices are
    /// shallow-cloned into the new datase
    ///
    /// Note if you just want to read a few rows by reference, use `get_rows`
    /// or `get_row` instead.
    ///
    /// # Example
    /// ```rust
    /// # use data_preparation::SafetensorsDataset;
    /// # use tch::Tensor;
    /// # use std::collections::HashMap;
    /// let dataset = SafetensorsDataset::from_dict(HashMap::from([
    ///     ("k".to_string(), vec![Tensor::from(0), Tensor::from(1), Tensor::from(2)])
    /// ])).unwrap();
    ///
    /// // Create a new dataset with only rows 1 and 2
    /// let subset = dataset.select(&[1, 2]).unwrap();
    /// assert_eq!(subset.len(), 2);
    /// ```
    pub fn select(&self, indices: &[usize]) -> Result<Self> {
        let len = self.len();
        for &index in indices {
            if index >= len {
                return Err(DataPrepError::Other(format!(
                    "Index {} is out of bounds for dataset of length {}.",
                    index, len
                )));
            }
        }

        // Prepare an empty structure to store the selected rows
        let mut selected_tensors: HashMap<String, Vec<Tensor>> = HashMap::new();
        for key in self.keys() {
            selected_tensors.insert(key.to_string(), Vec::with_capacity(indices.len()));
        }

        // If no indices are given, return an empty dataset with the same keys
        if indices.is_empty() {
            return Ok(SafetensorsDataset {
                dataset: Dataset::new(selected_tensors),
            });
        }

        // Retrieve each row, then PUSH its tensors with a deep copy into the new map
        for &index in indices {
            let row = self.get_row(index).ok_or_else(|| {
                DataPrepError::InconsistentTensorList(format!(
                    "Failed to access row at index {}.",
                    index
                ))
            })?;
            for (key, tensor) in row {
                selected_tensors
                    .get_mut(key.as_str())
                    .unwrap()
                    .push(tensor.shallow_clone())
            }
        }
        Self::from_dict(selected_tensors)
    }

    /// Returns a new `SafetensorsDataset` containing only rows for which the
    /// provided predicate function returns `true`.
    ///
    /// # Type Parameters
    /// - `F`: A closure or function that takes a map representing the row
    ///        (key -> &Tensor) and returns a `bool`.
    ///
    /// # Example
    /// ```rust
    /// # use data_preparation::SafetensorsDataset;
    /// # use tch::Tensor;
    /// # use std::collections::HashMap;
    /// let dataset = SafetensorsDataset::from_dict(HashMap::from([
    ///     ("x".to_string(), vec![Tensor::from(1), Tensor::from(2), Tensor::from(3)]),
    ///     ("y".to_string(), vec![Tensor::from(10), Tensor::from(20), Tensor::from(30)]),
    /// ])).unwrap();
    ///
    /// // Keep only rows where x > 1
    /// let filtered = dataset.filter(|row| row["x"].int64_value(&[]) > 1).unwrap();
    /// assert_eq!(filtered.len(), 2); // The rows with x = 2 and x = 3 remain
    /// ```
    pub fn filter<F>(&self, f: F) -> Result<Self>
    where
        F: Fn(&HashMap<String, &Tensor>) -> bool,
    {
        let mut filtered_tensors: HashMap<String, Vec<Tensor>> = self
            .keys()
            .into_iter()
            .map(|key| (key.to_string(), Vec::new()))
            .collect();

        // Test each row: if it pases, shallow-clone the tensors into filtered tensors
        for i in 0..self.len() {
            let row = self.get_row(i).ok_or_else(|| {
                DataPrepError::InconsistentTensorList(format!(
                    "Failed to access row at index {}",
                    i
                ))
            })?;
            if f(&row) {
                for (key, tensor) in row {
                    filtered_tensors
                        .get_mut(key.as_str())
                        .unwrap()
                        .push(tensor.shallow_clone());
                }
            }
        }
        // Note:
        // `from_dict` enforces dtype consistency for non-empty lists.
        // If all rows are filtered out, some lists may remain empty.
        // We allow that here for now.
        Self::from_dict(filtered_tensors)
    }

    /// Applies a transformation function to each row, returning a new `SafetensorsDataset`
    ///
    /// The function `f` is called with `(row_index, row_map)`, where
    /// - `row_index` is the 0-based index of the row,
    /// - `row_map` is the key -> &Tensor map.
    ///
    /// `f` must return a map of the same keys with the new (shallow) `Tensor`s.
    /// - The shape and dtype of the new tensors must be consistent (all rows must match
    ///   dtypes for a given key).
    ///
    /// # Errors
    /// - Returns `DataPrepError::InvalidKey` if `f` produces a key not in the original dataset.
    /// - Returns `DataPrepError::InconsistentTensorList` if different rows produce different dtypes.
    ///
    /// # Example
    /// ```rust
    /// # use data_preparation::SafetensorsDataset;
    /// # use tch::{Tensor, Kind};
    /// # use std::collections::HashMap;
    ///
    /// // Our original dataset has two keys: 'x' and 'y'.
    /// // Each has two rows: [1, 2] for x, [10, 20] for y.
    /// let dataset = SafetensorsDataset::from_dict(HashMap::from([
    ///     ("x".to_string(), vec![Tensor::from(1), Tensor::from(2)]),
    ///     ("y".to_string(), vec![Tensor::from(10), Tensor::from(20)])
    /// ])).unwrap();
    ///
    /// // We want to combine 'x' and 'y' for each row, and also update 'y':
    /// // x_new = x + y
    /// // y_new = y*2
    /// let mapped = dataset.map(|_row_index, row| {
    ///     let x_val = row["x"].int64_value(&[]);
    ///     let y_val = row["y"].int64_value(&[]);
    ///
    ///     HashMap::from([
    ///         ("x".to_string(), Tensor::from(x_val + y_val)),
    ///         ("y".to_string(), Tensor::from(y_val * 2)),
    ///     ])
    /// }).unwrap();
    ///
    /// // Now the new dataset has the same keys ('x' and 'y'), but transformed values:
    /// // x-> [11, 22] and y -> [20, 40].
    ///
    /// let row0 = mapped.get_row(0).unwrap();
    /// assert_eq!(row0["x"].int64_value(&[]), 11);
    /// assert_eq!(row0["y"].int64_value(&[]), 20);
    /// let row1 = mapped.get_row(1).unwrap();
    /// assert_eq!(row1["x"].int64_value(&[]), 22);
    /// assert_eq!(row1["y"].int64_value(&[]), 40);
    /// ```
    pub fn map<F>(&self, f: F) -> Result<Self>
    where
        F: Fn(usize, &HashMap<String, &Tensor>) -> HashMap<String, Tensor>,
    {
        let len = self.len();
        if len == 0 {
            // Return an empty dataset with the same keys.
            let mut empty_tensors = HashMap::new();
            for key in self.keys() {
                empty_tensors.insert(key.to_string(), Vec::new());
            }
            return Ok(Self {
                dataset: Dataset::new(empty_tensors),
            });
        }

        let mut transformed_tensors: HashMap<String, Vec<Tensor>> = HashMap::new();
        for key in self.keys() {
            transformed_tensors.insert(key.to_string(), Vec::with_capacity(len));
        }

        // Apply the transformation function to each row
        for i in 0..len {
            let row = self.get_row(i).ok_or_else(|| {
                DataPrepError::InconsistentTensorList(format!(
                    "Failed to access row at index {}",
                    i
                ))
            })?;
            let transformed_row = f(i, &row);

            // Check that the transformed row matches the original key set.
            if transformed_row.len() != row.len() {
                return Err(DataPrepError::InvalidKey(format!(
                    "Transformation at index {} produced a row with {} keys, expected {}.",
                    i,
                    transformed_row.len(),
                    row.len()
                )));
            }

            // Insert each transformed tensor into the new map, ensuring dtype consistency
            for (key, tensor) in transformed_row {
                if !self.contains_key(&key) {
                    return Err(DataPrepError::InvalidKey(format!(
                        "Transformation at index {} produced unknown key '{}'.",
                        i, key
                    )));
                }
                let existing_list = transformed_tensors.get_mut(&key).unwrap();
                // If there is already data for this key, ensure dtype matches.
                if !existing_list.is_empty() && tensor.kind() != existing_list[0].kind() {
                    return Err(DataPrepError::InconsistentTensorList(format!(
                        "Transformation at index {} produced a tensor with dtype {:?} for key '{}', expected dtype {:?}",
                        i,
                        tensor.kind(),
                        key,
                        existing_list[0].kind()
                    )));
                }
                existing_list.push(tensor);
            }
        }

        // Each key's length should match the dataset length now.
        for (key, tensors) in &transformed_tensors {
            if tensors.len() != len {
                return Err(DataPrepError::InconsistentTensorList(format!(
                    "Transformation produced {} tensors for key '{}', expected {}. ",
                    tensors.len(),
                    key,
                    len
                )));
            }
        }
        Self::from_dict(transformed_tensors)
    }

    /// Renames keys in place according to the provided mapping.
    ///
    /// # Arguments
    /// - `key_mapping`: A slice of `(old_key, new_key)` pairs. Each `old_key` in
    ///   the dataset is renamed to `new_key`. If `new_key` already exists
    ///   (and is not itself being renamed away), an error is returned.
    ///
    /// # Errors
    /// - `DataPrepError::InvalidKey` if a mapping references a non-existent key
    ///   or introduces a duplicate collision.
    ///
    /// # Example
    /// ```rust
    /// # use data_preparation::SafetensorsDataset;
    /// # use tch::Tensor;
    /// # use std::collections::HashMap;
    /// let mut dataset = SafetensorsDataset::from_dict(HashMap::from([
    ///     ("old".to_string(), vec![Tensor::from(1)])
    /// ])).unwrap();
    ///
    /// // Rename "old" to "new"
    /// dataset.rename(&[("old".to_string(), "new".to_string())]).unwrap();
    /// assert!(dataset.contains_key("new"));
    /// assert!(!dataset.contains_key("old"));
    /// ```
    pub fn rename(&mut self, key_mapping: &[(String, String)]) -> Result<()> {
        // Detect duplicates in the new keys.
        let mut new_keys = HashSet::new();
        for (_, new_key) in key_mapping {
            if !new_keys.insert(new_key) {
                return Err(DataPrepError::InvalidKey(format!(
                    "Duplicate new key '{}' in key mapping",
                    new_key
                )));
            }
        }

        // Gather the sets of old and new keys.
        let current_keys: HashSet<&String> = self.dataset.tensors.keys().collect();
        let old_keys: HashSet<&String> = key_mapping.iter().map(|(old, _)| old).collect();
        let new_keys: HashSet<&String> = key_mapping.iter().map(|(_, new)| new).collect();

        // Check that all old keys exist in the dataset.
        for old_key in &old_keys {
            if !current_keys.contains(old_key) {
                return Err(DataPrepError::InvalidKey(format!(
                    "Key '{}' to rename does not exist in the dataset",
                    old_key
                )));
            }
        }

        // Check for collisions: a new key that already exists but is not being replaced.
        let keys_to_remove: HashSet<&String> = old_keys;
        for new_key in &new_keys {
            if current_keys.contains(new_key) && !keys_to_remove.contains(new_key) {
                return Err(DataPrepError::InvalidKey(format!(
                    "New key '{}' already exists in the dataset and is not being renamed",
                    new_key
                )));
            }
        }

        // Perform the renaming by draining the old map and rebuilding it.
        let mut new_tensors = HashMap::with_capacity(self.dataset.tensors.len());
        for (old_key, tensor_list) in self.dataset.tensors.drain() {
            let new_key = key_mapping
                .iter()
                .find(|(old, _)| old == &old_key)
                .map(|(_, new)| new.clone())
                .unwrap_or(old_key);
            new_tensors.insert(new_key, tensor_list);
        }
        self.dataset.tensors = new_tensors;

        Ok(())
    }

    /// Returns an immutable reference to the underlying `Dataset`.
    ///
    /// You can use this if you need lower-level operations. However, be careful
    /// with direct mutation since it can bypass the safety checks provided by
    /// `SafetensorsDataset`.
    pub fn inner_dataset(&self) -> &Dataset {
        &self.dataset
    }

    // ---------------------------------------------------------------------------------------
    // Metadata & I/O
    // ---------------------------------------------------------------------------------------

    /// Returns metadata about the dataset's structure, including
    /// - The number of rows (`len`),
    /// - A map from each key to its `TensorLayout`.
    ///
    /// # Example
    /// ``` rust
    /// use data_preparation::SafetensorsDataset;
    /// use tch::Tensor;
    /// use std::collections::HashMap;
    ///
    /// let dataset = SafetensorsDataset::from_dict(HashMap::from([
    ///     ("key".to_string(), vec![Tensor::from(1)]),
    /// ])).unwrap();
    ///
    /// let info = dataset.info();
    /// println!("{:?}", info);
    /// ```
    pub fn info(&self) -> DatasetInfo {
        let len = self.len();
        let mut layouts = HashMap::new();

        for (key, tensors) in &self.dataset.tensors {
            if tensors.is_empty() {
                // This might be rare unless user called `empty(...)` or mutated
                // the dataset directly, given that from_dict enforces non-empty
                layouts.insert(key.clone(), TensorLayout::VaryingDtype);
                continue;
            }
            let first_tensor = &tensors[0];
            let first_shape = first_tensor.size();
            let first_dtype = first_tensor.kind();

            let mut all_same_shape = true;
            let mut all_same_dtype = true;

            for tensor in tensors.iter().skip(1) {
                if tensor.kind() != first_dtype {
                    all_same_dtype = false;
                }
                if tensor.size() != first_shape {
                    all_same_shape = false;
                }
                if !all_same_dtype && !all_same_shape {
                    break;
                }
            }

            let layout = if !all_same_dtype {
                TensorLayout::VaryingDtype
            } else if !all_same_shape {
                TensorLayout::VaryingDimSize { dtype: first_dtype }
            } else {
                TensorLayout::Standard {
                    shape: first_shape,
                    dtype: first_dtype,
                }
            };
            layouts.insert(key.clone(), layout);
        }
        DatasetInfo { len, layouts }
    }

    /// Saves the dataset to a `.safetensors` file on disk.
    ///
    /// Each key’s tensors are written under the names `key.0`, `key.1`, etc.
    /// Metadata is stored under `__metadata__` in JSON format, including:
    /// - The dataset length,
    /// - Per-key metadata (dtype, etc.).
    ///
    /// # Errors
    /// - `DataPrepError::InvalidKey` if any key contains the character `'.'`.
    /// - `DataPrepError::UnsupportedDtype` if a tensor’s dtype is not ye
    ///   supported (e.g., `F16`).
    /// - `DataPrepError::FileFormat` or `DataPrepError::Other` on I/O failures,
    ///   JSON issues, etc.
    ///
    /// # Example
    /// ```rust,no_run
    /// use data_preparation::SafetensorsDataset;
    /// use tch::Tensor;
    /// use std::collections::HashMap;
    ///
    /// let ds = SafetensorsDataset::from_dict(HashMap::from([
    ///     ("x".to_string(), vec![Tensor::from(1)]),
    /// ])).unwrap();
    ///
    /// ds.save_to_file("my_dataset.safetensors").unwrap();
    /// ```
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut tensor_data_map: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)> = HashMap::new();
        let mut metadata: HashMap<String, String> = HashMap::new(); // Values must be string

        // Store total dataset size in metadata
        metadata.insert(
            "size".to_string(),
            serde_json::to_string(&self.dataset.len())?,
        );

        for (key, tensor_list) in &self.dataset.tensors {
            // Keys cannot contain '.' (conflicts with safetensors naming)
            if key.contains('.') {
                return Err(DataPrepError::InvalidKey(format!(
                    "'.' is not allowed in key '{}'",
                    key
                )));
            }

            if tensor_list.is_empty() {
                // This case might be unreachable if from_dict enforces non-empty lists. But,
                // TODO: how to represent empty list metadata.
                // For now, skip saving tensors and metadata for empty lists.
                continue;
            }

            // Determine dtype for this lis
            let list_kind = tensor_list[0].kind();
            let (safetensor_dtype, dtype_str) = match list_kind {
                Kind::Float => (Dtype::F32, "F32"),
                Kind::Double => (Dtype::F64, "F64"),
                Kind::Int64 => (Dtype::I64, "I64"),
                Kind::Int => (Dtype::I32, "I32"),
                Kind::Int8 => (Dtype::I8, "I8"),
                Kind::Uint8 => (Dtype::U8, "U8"),
                Kind::Bool => (Dtype::BOOL, "BOOL"),
                // TODO: Add F16, and BF16 here
                _ => {
                    return Err(DataPrepError::UnsupportedDtype(format!(
                        "Dtype {:?} in list for key '{}' is not supported for saving.",
                        list_kind, key
                    )))
                }
            };

            // Serialize each tensor in the vector as key.index
            for (i, tensor) in tensor_list.iter().enumerate() {
                if tensor.kind() != list_kind {
                    return Err(DataPrepError::InconsistentTensorList(format!(
                        "Inconsistent dtypes in list for key '{}': expected {:?}, found {:?} at index {}",
                        key,
                        list_kind,
                        tensor.kind(),
                        i
                    )));
                }

                let tensor_key = format!("{}.{}", key, i);
                let num_elements = tensor.numel();
                let shape: Vec<usize> = tensor.size().iter().map(|&x| x as usize).collect();

                // Flatten the data to bytes in little-endian form.
                let bytes: Vec<u8> = match list_kind {
                    Kind::Float => {
                        let mut data = vec![0.0f32; num_elements];
                        tensor.copy_data(&mut data, num_elements);
                        data.into_iter().flat_map(|x| x.to_le_bytes()).collect()
                    }
                    Kind::Double => {
                        let mut data = vec![0.0f64; num_elements];
                        tensor.copy_data(&mut data, num_elements);
                        data.into_iter().flat_map(|x| x.to_le_bytes()).collect()
                    }
                    Kind::Int64 => {
                        let mut data = vec![0i64; num_elements];
                        tensor.copy_data(&mut data, num_elements);
                        data.into_iter().flat_map(|x| x.to_le_bytes()).collect()
                    }
                    Kind::Int => {
                        let mut data = vec![0i32; num_elements];
                        tensor.copy_data(&mut data, num_elements);
                        data.into_iter().flat_map(|x| x.to_le_bytes()).collect()
                    }
                    Kind::Int8 => {
                        let mut data = vec![0i8; num_elements];
                        tensor.copy_data(&mut data, num_elements);
                        data.into_iter().flat_map(|x| x.to_le_bytes()).collect()
                    }
                    Kind::Uint8 => {
                        let mut data = vec![0u8; num_elements];
                        tensor.copy_data(&mut data, num_elements);
                        data
                    }
                    Kind::Bool => {
                        let mut bool_data = vec![false; num_elements];
                        tensor.copy_data::<bool>(&mut bool_data, num_elements);
                        let byte_data: Vec<u8> = bool_data.into_iter().map(|b| b as u8).collect();
                        byte_data
                    }
                    // This case is most likely unreachable due to the dtype check before the loop
                    _ => unreachable!("Unsupported dtype checked earlier"),
                };

                tensor_data_map.insert(tensor_key, (safetensor_dtype, shape, bytes));
            }

            // Create per-ket metadata(dtype, etc. )
            let meta_map = HashMap::from([
                ("list", Value::Bool(true)),
                ("numel", Value::Number(tensor_list.len().into())),
                ("dtype", Value::String(dtype_str.to_string())),
            ]);
            metadata.insert(key.clone(), serde_json::to_string(&meta_map)?);
        } // End outer loop

        // Create tensor data into safetensors view
        let tensors_for_serialization: HashMap<String, TensorView> = tensor_data_map
            .iter()
            .map(|(name, (dt, shape, bytes))| {
                TensorView::new(*dt, shape.clone(), &bytes)
                    .map(|view| (name.clone(), view))
                    .map_err(DataPrepError::Safetensor)
            })
            .collect::<Result<_>>()?;

        // Serialize everything into bytes
        let serialized = serialize(&tensors_for_serialization, &Some(metadata))?;

        // Write to file
        let mut file = File::create(path)?;
        file.write_all(&serialized)?;
        Ok(())
    }

    /// Loads a `SafetensorsDataset` from a `.safetensors` file.
    ///
    /// # Errors
    /// - `DataPrepError::FileFormat` or `DataPrepError::MetadataNotFound` for
    ///   malformed or missing metadata.
    /// - `DataPrepError::UnsupportedDtype` for dtypes not covered (F16, BF16, etc.).
    /// - Other I/O or serde errors.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use data_preparation::SafetensorsDataset;
    /// let ds = SafetensorsDataset::load_from_file("my_dataset.safetensors").unwrap();
    /// println!("Loaded dataset with {} rows.", ds.len());
    /// ```
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Read entire file into memory.
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        // Use safetensors to parse the views.
        let safetensor_views = SafeTensors::deserialize(&buffer)?;

        // Extract the top-level JSON metadata (the stuff under __metadata__).
        let header_len = u64::from_le_bytes(buffer[..8].try_into().map_err(|_| {
            DataPrepError::FileFormat("Invalid file header - cannot read length.".into())
        })?) as usize;
        let header_end = 8 + header_len;
        if buffer.len() < header_end {
            return Err(DataPrepError::FileFormat("Truncated file header.".into()));
        }
        let header_bytes = &buffer[8..header_end];
        let header_val: Value = serde_json::from_slice(header_bytes)?;
        let top_level_metadata = header_val.get("__metadata__").ok_or_else(|| {
            DataPrepError::MetadataNotFound("__metadata__ section missing.".into())
        })?;
        let top_level_map: HashMap<String, Value> =
            serde_json::from_value(top_level_metadata.clone())?;

        // Group tensor keys by base key (e.g. "mykey.0", "mykey.1" -> "mykey").
        let mut grouped_keys: HashMap<String, Vec<String>> = HashMap::new();
        for name in safetensor_views.names() {
            if let Some((base_key, index_str)) = name.rsplit_once('.') {
                if index_str.parse::<usize>().is_ok() {
                    grouped_keys
                        .entry(base_key.to_string())
                        .or_default()
                        .push(name.clone());
                }
            }
        }

        // Rebuild the dataset from grouped keys.
        let mut dataset_tensors = HashMap::new();
        for (base_key, mut suffix_keys) in grouped_keys {
            // Sort by numeric suffix: key.0, key.1, key.2
            suffix_keys.sort_by_key(|k| {
                k.rsplit_once('.')
                    .unwrap()
                    .1
                    .parse::<usize>()
                    .unwrap_or(usize::MAX)
            });

            let meta_str = top_level_map
                .get(&base_key)
                .ok_or_else(|| {
                    DataPrepError::MetadataNotFound(format!(
                        "No metadata found for key '{}'.",
                        base_key
                    ))
                })?
                .as_str()
                .ok_or_else(|| {
                    DataPrepError::MetadataFormat(format!(
                        "Metadata for key '{}' is not a string.",
                        base_key
                    ))
                })?;

            let list_meta: HashMap<String, Value> = serde_json::from_str(meta_str)?;
            let is_list = list_meta
                .get("list")
                .and_then(|v| v.as_bool())
                .ok_or_else(|| {
                    DataPrepError::MetadataFormat(format!(
                        "'list' boolean missing for '{}'.",
                        base_key
                    ))
                })?;
            let numel = list_meta
                .get("numel")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| {
                    DataPrepError::MetadataFormat(format!(
                        "'numel' missing or invalid for '{}'.",
                        base_key
                    ))
                })? as usize;
            let dtype_str = list_meta
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    DataPrepError::MetadataFormat(format!(
                        "'dtype' missing or invalid for '{}'.",
                        base_key
                    ))
                })?;

            if !is_list {
                // If the data was not marked as a list, you could handle differently or ignore.
                // For now, skip or treat as single-tensor. Adjust as needed.
                continue;
            }
            if numel != suffix_keys.len() {
                return Err(DataPrepError::FileFormat(format!(
                    "Metadata says key '{}' has {} tensors, but found {}.",
                    base_key,
                    numel,
                    suffix_keys.len()
                )));
            }

            // Reconstruct each tensor by decoding bytes from the safetensor view.
            let mut tensor_list = Vec::with_capacity(numel);
            for tensor_name in suffix_keys {
                let view = safetensor_views.tensor(&tensor_name)?;
                let shape = view.shape();
                let raw_data = view.data();

                // Helper to shape a Tensor from raw bytes with the correct type.
                let t = match dtype_str {
                    "F32" => {
                        let expected_bytes =
                            shape.iter().product::<usize>() * std::mem::size_of::<f32>();
                        if raw_data.len() != expected_bytes {
                            return Err(DataPrepError::FileFormat(format!(
                                "Incorrect byte size for F32 '{}' ({} vs expected {}).",
                                tensor_name,
                                raw_data.len(),
                                expected_bytes
                            )));
                        }
                        let floats: Vec<f32> = raw_data
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                            .collect();
                        Tensor::from_slice(&floats)
                            .reshape(&shape.iter().map(|&d| d as i64).collect::<Vec<_>>())
                    }
                    "F64" => {
                        let expected_bytes =
                            shape.iter().product::<usize>() * std::mem::size_of::<f64>();
                        if raw_data.len() != expected_bytes {
                            return Err(DataPrepError::FileFormat(format!(
                                "Incorrect byte size for F64 '{}'.",
                                tensor_name
                            )));
                        }
                        let doubles: Vec<f64> = raw_data
                            .chunks_exact(8)
                            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                            .collect();
                        Tensor::from_slice(&doubles)
                            .reshape(&shape.iter().map(|&d| d as i64).collect::<Vec<_>>())
                    }
                    "I8" => {
                        if raw_data.len() != shape.iter().product::<usize>() {
                            return Err(DataPrepError::FileFormat(format!(
                                "Incorrect byte size for I8 '{}'.",
                                tensor_name
                            )));
                        }
                        let i8_data: Vec<i8> = raw_data.iter().map(|&b| b as i8).collect();
                        Tensor::from_slice(&i8_data)
                            .reshape(&shape.iter().map(|&d| d as i64).collect::<Vec<_>>())
                    }
                    "I32" => {
                        let expected_bytes =
                            shape.iter().product::<usize>() * std::mem::size_of::<i32>();
                        if raw_data.len() != expected_bytes {
                            return Err(DataPrepError::FileFormat(format!(
                                "Incorrect byte size for I32 '{}'.",
                                tensor_name
                            )));
                        }
                        let i32_data: Vec<i32> = raw_data
                            .chunks_exact(4)
                            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                            .collect();
                        Tensor::from_slice(&i32_data)
                            .reshape(&shape.iter().map(|&d| d as i64).collect::<Vec<_>>())
                    }
                    "I64" => {
                        let expected_bytes =
                            shape.iter().product::<usize>() * std::mem::size_of::<i64>();
                        if raw_data.len() != expected_bytes {
                            return Err(DataPrepError::FileFormat(format!(
                                "Incorrect byte size for I64 '{}'.",
                                tensor_name
                            )));
                        }
                        let i64_data: Vec<i64> = raw_data
                            .chunks_exact(8)
                            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                            .collect();
                        Tensor::from_slice(&i64_data)
                            .reshape(&shape.iter().map(|&d| d as i64).collect::<Vec<_>>())
                    }
                    "U8" => {
                        let expected_bytes =
                            shape.iter().product::<usize>() * std::mem::size_of::<u8>();
                        if raw_data.len() != expected_bytes {
                            return Err(DataPrepError::FileFormat(format!(
                                "Incorrect byte size for U8 '{}'.",
                                tensor_name
                            )));
                        }
                        let u8_data = raw_data.to_vec();
                        Tensor::from_slice(&u8_data)
                            .reshape(&shape.iter().map(|&d| d as i64).collect::<Vec<_>>())
                            .to_kind(Kind::Uint8)
                    }
                    "BOOL" => {
                        let expected_bytes =
                            shape.iter().product::<usize>() * std::mem::size_of::<u8>();
                        if raw_data.len() != expected_bytes {
                            return Err(DataPrepError::FileFormat(format!(
                                "Incorrect byte size for BOOL '{}'.",
                                tensor_name
                            )));
                        }
                        let bool_data: Vec<u8> = raw_data.to_vec();
                        let t = Tensor::from_slice(&bool_data)
                            .reshape(&shape.iter().map(|&d| d as i64).collect::<Vec<_>>());
                        t.to_kind(Kind::Bool)
                    }
                    _ => {
                        return Err(DataPrepError::UnsupportedDtype(format!(
                            "Dtype '{}' is not supported during load.",
                            dtype_str
                        )))
                    }
                };
                tensor_list.push(t);
            }
            dataset_tensors.insert(base_key.clone(), tensor_list);
        }

        Ok(SafetensorsDataset {
            dataset: Dataset::new(dataset_tensors),
        })
    }
}
