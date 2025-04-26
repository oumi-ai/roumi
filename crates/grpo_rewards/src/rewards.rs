// crates/grpo_rewards/src/rewards.rs
use anyhow;
use std::collections::HashMap;

// Update trait definition to require Send + Sync for thread safety with plugins
pub trait Calculator: Send + Sync {
    fn compute_rewards(
        &self,
        prompts: &Vec<String>,
        completions: &Vec<String>,
    ) -> anyhow::Result<Vec<f32>>;

    // Factory method (keep for compatibility)
    fn new(params: HashMap<String, String>) -> Self
    where
        Self: Sized;
}

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

pub struct CompletionSameLengthAsPromptCalculator;

impl Calculator for CompletionSameLengthAsPromptCalculator {
    fn new(_params: HashMap<String, String>) -> Self {
        CompletionSameLengthAsPromptCalculator
    }

    fn compute_rewards(
        &self,
        prompts: &Vec<String>, // Remove the underscore to use this parameter
        completions: &Vec<String>,
    ) -> anyhow::Result<Vec<f32>> {
        let mut result: Vec<f32> = Vec::<f32>::with_capacity(completions.len());

        // Handle the case where prompts is empty but completions is not
        if prompts.is_empty() && !completions.is_empty() {
            // Either use a default value or just process completions
            for completion in completions {
                result.push(-((completion.len()) as f32));
            }
        } else {
            // Zip the two vectors together and iterate through the pairs
            for (prompt, completion) in prompts.iter().zip(completions.iter()) {
                let length_diff = (completion.len() as i32) - (prompt.len() as i32);
                result.push(-((length_diff.abs()) as f32));
            }
        }

        Ok(result)
    }
}

// Add other calculator implementations here
// For example:
// pub struct ComplexityRewardCalculator { ... }
// impl Calculator for ComplexityRewardCalculator { ... }
