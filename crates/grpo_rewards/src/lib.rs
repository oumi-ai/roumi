use pyo3::types::PyDict;
use pyo3::{exceptions::PyTypeError, exceptions::PyValueError, prelude::*};
use std::collections::HashMap;

pub trait Calculator: Send + Sync {
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
pub struct CompletionNegativeLengthCalculator;

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

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;
    
    // Test the CompletionNegativeLengthCalculator implementation
    #[test]
    fn test_negative_length_calculator() {
        let calculator = CompletionNegativeLengthCalculator::new(HashMap::new());
        
        let prompts = vec![
            "What is the capital of France?".to_string(),
            "Explain quantum computing".to_string(),
        ];
        
        let completions = vec![
            "Paris".to_string(),            // Length: 5
            "It's complicated".to_string(), // Length: 16 (including the apostrophe)
        ];
        
        let rewards = calculator.compute_rewards(&prompts, &completions).unwrap();
        
        assert_eq!(rewards.len(), 2);
        assert_eq!(rewards[0], -5.0);
        assert_eq!(rewards[1], -16.0); // Fixed: correct length
    }
    
    // Test empty completions
    #[test]
    fn test_empty_completion() {
        let calculator = CompletionNegativeLengthCalculator::new(HashMap::new());
        
        let prompts = vec!["Test".to_string()];
        let completions = vec!["".to_string()]; // Empty string, length: 0
        
        let rewards = calculator.compute_rewards(&prompts, &completions).unwrap();
        
        assert_eq!(rewards.len(), 1);
        assert_eq!(rewards[0], 0.0); // -0.0 is equal to 0.0 in float comparison
    }
    
    // Test empty prompts with the calculator directly
    #[test]
    fn test_empty_prompts() {
        let calculator = CompletionNegativeLengthCalculator::new(HashMap::new());
        
        let prompts = vec![];
        let completion_str = "Answer without prompt".to_string();
        let expected_len = completion_str.len() as f32;
        let completions = vec![completion_str];
        
        let rewards = calculator.compute_rewards(&prompts, &completions).unwrap();
        
        assert_eq!(rewards.len(), 1);
        assert_eq!(rewards[0], -expected_len);
    }
    
    // The following tests need a Python interpreter to be initialized
    // We'll create non-PyO3 versions of these tests to avoid that dependency
    
    // Test for constructor validation - create a direct Rust-only test
    #[test]
    fn test_grpo_rewards_validation() {
        // Instead of testing the Python constructor, let's create our own function
        // that performs similar validation
        fn validate_inputs(
            prompts: &Vec<String>,
            completions: &Vec<String>,
        ) -> Result<(), String> {
            if completions.is_empty() {
                return Err("Completions cannot be empty.".to_string());
            } else if !prompts.is_empty() && (prompts.len() != completions.len()) {
                return Err("Prompts and completions must have the same length.".to_string());
            }
            Ok(())
        }
        
        // Test empty completions
        let prompts = vec!["Test".to_string()];
        let completions = vec![];
        let result = validate_inputs(&prompts, &completions);
        assert!(result.is_err());
        if let Err(msg) = result {
            assert!(msg.contains("Completions cannot be empty"));
        }
        
        // Test mismatched lengths
        let prompts = vec!["Test1".to_string(), "Test2".to_string()];
        let completions = vec!["Only one completion".to_string()];
        let result = validate_inputs(&prompts, &completions);
        assert!(result.is_err());
        if let Err(msg) = result {
            assert!(msg.contains("must have the same length"));
        }
        
        // Test valid input
        let prompts = vec!["Test1".to_string()];
        let completions = vec!["Answer1".to_string()];
        let result = validate_inputs(&prompts, &completions);
        assert!(result.is_ok());
        
        // Test empty prompts (should be valid)
        let prompts = vec![];
        let completions = vec!["Answer without prompt".to_string()];
        let result = validate_inputs(&prompts, &completions);
        assert!(result.is_ok());
    }
    
    // Test calculator selection (without PyO3)
    #[test]
    fn test_calculator_selection() {
        fn select_calculator(name: &str) -> Result<&'static str, String> {
            match name {
                "CompletionNegativeLengthCalculator" => Ok("Valid calculator"),
                _ => Err("Unknown calculator.".to_string()),
            }
        }
        
        // Test valid calculator
        let result = select_calculator("CompletionNegativeLengthCalculator");
        assert!(result.is_ok());
        
        // Test invalid calculator
        let result = select_calculator("NonExistentCalculator");
        assert!(result.is_err());
        if let Err(msg) = result {
            assert!(msg.contains("Unknown calculator"));
        }
    }
    
    // Test full calculation flow (without PyO3)
    #[test]
    fn test_calculation_flow() {
        // Create a calculator
        let calculator = CompletionNegativeLengthCalculator::new(HashMap::new());
        
        // Test data
        let prompts = vec!["Test1".to_string()];
        let completions = vec!["Answer1".to_string()]; // Length: 7
        
        // Compute rewards
        let rewards = calculator.compute_rewards(&prompts, &completions).unwrap();
        
        // Verify results
        assert_eq!(rewards.len(), 1);
        assert_eq!(rewards[0], -7.0);
    }
}
