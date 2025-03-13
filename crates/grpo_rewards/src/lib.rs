use pyo3::types::PyDict;
use pyo3::{exceptions::PyTypeError, exceptions::PyValueError, prelude::*};
use std::collections::HashMap;

trait Calculator: Send + Sync {
    fn new(params: HashMap<String, String>) -> Self
    where
        Self: Sized;

    fn compute_rewards(
        &self,
        prompts: &Vec<String>,
        completions: &Vec<String>,
    ) -> anyhow::Result<Vec<f32>>;
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

// Dummy sample calculator.
struct CompletionNegativeLengthCalculator;

impl Calculator for CompletionNegativeLengthCalculator {
    fn new(_params: HashMap<String, String>) -> Self {
        CompletionNegativeLengthCalculator
    }

    fn compute_rewards(
        &self,
        _prompts: &Vec<String>,
        completions: &Vec<String>,
    ) -> anyhow::Result<Vec<f32>> {
        let mut result: Vec<f32> = Vec::<f32>::with_capacity(completions.len());
        for completion in completions {
            result.push(-(completion.len() as f32));
        }
        Ok(result)
    }
}

fn convert_pydict_to_str2str_map(
    d: Option<&Bound<'_, PyDict>>,
) -> anyhow::Result<HashMap<String, String>> {
    if let Some(params) = d {
        let mut x: HashMap<String, String> = HashMap::with_capacity(params.len());

        for (key, value) in params.iter() {
            let key: String = key
                .str()
                .map_err(|_| {
                    PyTypeError::new_err("function_params's key is not convertible to string")
                })?
                .to_string();
            let value: String = value
                .str()
                .map_err(|_| {
                    PyTypeError::new_err("function_params's value is not convertible to string")
                })?
                .to_string();
            x.insert(key, value);
        }
        Ok(x)
    } else {
        Ok(HashMap::new())
    }
}

#[pymethods]
impl GrpoRewards {
    #[new]
    #[pyo3(signature = (function_name, prompts, completions, *, function_params=None))]
    fn py_new(
        function_name: &str,
        prompts: Vec<String>,
        completions: Vec<String>,
        function_params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if completions.is_empty() {
            return Err(PyValueError::new_err("Completions cannot be empty."));
        } else if !prompts.is_empty() && (prompts.len() != completions.len()) {
            return Err(PyValueError::new_err(
                "Prompts and completions must have the same length.",
            ));
        }

        let internal_func_params = convert_pydict_to_str2str_map(function_params)?;

        // TODO Refactor into calculator builder function.
        let calculator: Box<dyn Calculator>;
        match function_name {
            "CompletionNegativeLengthCalculator" => {
                calculator = Box::new(CompletionNegativeLengthCalculator::new(
                    internal_func_params,
                ))
            }
            _ => return Err(PyValueError::new_err("Unknown calculator.")),
        }

        Ok(GrpoRewards {
            prompts,
            completions,
            function_name: function_name.to_string(),
            calculator,
        })
    }

    #[pyo3(signature = ())]
    fn compute_rewards(&self) -> PyResult<Vec<f32>> {
        let rewards: Vec<f32> = self
            .calculator
            .compute_rewards(&self.prompts, &self.completions)?;
        Ok(rewards)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn grpo_rewards(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GrpoRewards>()?;
    Ok(())
}
