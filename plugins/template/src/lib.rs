use anyhow::Result;
use std::collections::HashMap;

// This needs to match the Calculator trait in the main crate
pub trait Calculator: Send + Sync {
    fn compute_rewards(&self, prompts: &Vec<String>, completions: &Vec<String>)
        -> Result<Vec<f32>>;
}

// Example calculator implementation
pub struct KeywordMatchCalculator {
    keywords: Vec<String>,
    reward_per_match: f32,
}

impl KeywordMatchCalculator {
    fn new(params: HashMap<String, String>) -> Self {
        let keywords = params
            .get("keywords")
            .map(|s| s.split(',').map(|k| k.trim().to_string()).collect())
            .unwrap_or_default();

        let reward_per_match = params
            .get("reward_per_match")
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1.0);

        KeywordMatchCalculator {
            keywords,
            reward_per_match,
        }
    }
}

impl Calculator for KeywordMatchCalculator {
    fn compute_rewards(
        &self,
        _prompts: &Vec<String>,
        completions: &Vec<String>,
    ) -> Result<Vec<f32>> {
        let mut rewards = Vec::with_capacity(completions.len());

        for completion in completions {
            let mut reward = 0.0;
            for keyword in &self.keywords {
                if completion.to_lowercase().contains(&keyword.to_lowercase()) {
                    reward += self.reward_per_match;
                }
            }
            rewards.push(reward);
        }

        Ok(rewards)
    }
}

// Factory function to create the calculator
fn create_keyword_match_calculator(params: HashMap<String, String>) -> Box<dyn Calculator> {
    Box::new(KeywordMatchCalculator::new(params))
}

// Registration function that will be called by the plugin loader
#[no_mangle]
pub fn register_calculators() -> Vec<(
    &'static str,
    fn(HashMap<String, String>) -> Box<dyn Calculator>,
)> {
    vec![
        (
            "KeywordMatchCalculator",
            create_keyword_match_calculator as fn(HashMap<String, String>) -> Box<dyn Calculator>,
        ),
        // Add more calculators here
    ]
}
