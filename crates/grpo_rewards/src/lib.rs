use pyo3::{exceptions::PyValueError, prelude::*};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

trait Calculator: Send + Sync {
    fn new() -> Self
    where
        Self: Sized;

    fn compute_rewards(&self, prompts: &Vec<String>, completions: &Vec<String>) -> Vec<f32>;
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
    calculator: Box<dyn Calculator>,
}

struct CompletionNegativeLengthCalculator;

impl Calculator for CompletionNegativeLengthCalculator {
    fn new() -> Self {
        CompletionNegativeLengthCalculator
    }

    fn compute_rewards(&self, _prompts: &Vec<String>, completions: &Vec<String>) -> Vec<f32> {
        let mut result: Vec<f32> = Vec::<f32>::with_capacity(completions.len());
        for completion in completions {
            result.push(-(completion.len() as f32));
        }
        result
    }
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

    #[pyo3(signature = ())]
    fn compute(&self) -> PyResult<Vec<f32>> {
        let rewards: Vec<f32> = self
            .calculator
            .compute_rewards(&self.prompts, &self.completions);
        Ok(0.0)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn grpo_rewards(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<GrpoRewards>()?;
    Ok(())
}
