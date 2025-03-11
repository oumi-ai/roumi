use pyo3::{exceptions::PyValueError, prelude::*};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyclass]
pub struct GrpoRewards {
    #[pyo3(get)]
    pub prompts: Vec<String>,

    #[pyo3(get)]
    pub completions: Vec<String>,

    #[pyo3(get)]
    pub function_name: String,
    // TODO: Add kwargs dict.
}

#[pymethods]
impl GrpoRewards {
    #[new]
    fn new(function_name: &str, prompts: Vec<String>, completions: Vec<String>) -> PyResult<Self> {
        if completions.is_empty() {
            return Err(PyValueError::new_err("Completions cannot be empty."));
        } else if !prompts.is_empty() && (prompts.len() != completions.len()) {
            return Err(PyValueError::new_err(
                "Prompts and completions must have the same length.",
            ));
        }

        Ok(GrpoRewards {
            prompts,
            completions,
            function_name: function_name.to_string(),
        })
    }
}


/// A Python module implemented in Rust.
#[pymodule]
fn grpo_rewards(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<GrpoRewards>()?;
    Ok(())
}
