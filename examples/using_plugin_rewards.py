#!/usr/bin/env python3

from grpo_rewards import GrpoRewards, load_reward_plugin

def main():
    # Load the plugin
    calculator_names = load_reward_plugin("../plugins/template/target/release/libgrpo_rewards_template_plugin.so")
    print(f"Loaded calculators from plugin: {calculator_names}")
    
    # Example prompts and completions
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing"
    ]
    
    completions = [
        "Paris is the capital of France.",
        "Quantum physics uses superposition and entanglement to perform calculations."
    ]
    
    # Create a GrpoRewards instance with the KeywordMatchCalculator from the plugin
    rewards_calculator = GrpoRewards(
        function_name="KeywordMatchCalculator",
        function_params={
            "keywords": "capital,France,quantum,superposition",
            "reward_per_match": "2.5"
        },
        from_plugin=True  # This tells GrpoRewards to look for the calculator in loaded plugins
    )
    
    # Compute the rewards
    rewards = rewards_calculator.compute_rewards(prompts, completions)
    
    # Print the results
    print("\nRewards (keyword matching):")
    for i, reward in enumerate(rewards):
        print(f"  {i}: {reward}")

if __name__ == "__main__":
    main()