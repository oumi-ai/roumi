use pyo3::{exceptions::PyValueError, prelude::*};

trait Calculator: Send + Sync {
    fn new() -> Self
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
    fn new() -> Self {
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

        // TODO Refactor into calculator builder function.
        let calculator: Box<dyn Calculator>;
        match function_name {
            "CompletionNegativeLengthCalculator" => {
                calculator = Box::new(CompletionNegativeLengthCalculator::new())
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
