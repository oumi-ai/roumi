use std::collections::HashMap;

// Import both the struct and the trait
use grpo_rewards::CompletionNegativeLengthCalculator;
use grpo_rewards::Calculator; // This is the important addition

fn main() -> anyhow::Result<()> {
    // Create sample data
    let prompts = vec![
        "What is the capital of France?".to_string(),
        "Explain quantum computing".to_string(),
    ];
    
    let completions = vec![
        "Paris is the capital of France.".to_string(),
        "Quantum computing uses quantum mechanics principles like superposition and entanglement to process information.".to_string(),
    ];
    
    // Now this should work because Calculator trait is in scope
    let calculator = CompletionNegativeLengthCalculator::new(HashMap::new());
    let rewards = calculator.compute_rewards(&prompts, &completions)?;
    
    println!("Prompts:");
    for (i, prompt) in prompts.iter().enumerate() {
        println!("  {}: {}", i, prompt);
    }
    
    println!("\nCompletions:");
    for (i, completion) in completions.iter().enumerate() {
        println!("  {}: {}", i, completion);
    }
    
    println!("\nRewards:");
    for (i, reward) in rewards.iter().enumerate() {
        println!("  {}: {}", i, reward);
    }
    
    Ok(())
}
