#!/usr/bin/env python3

from grpo_rewards import GrpoRewards

def main():
    # Example prompts and completions
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing"
    ]
    
    completions = [
        "Paris is the capital of France.",
        "Quantum computing uses quantum mechanics principles like superposition and entanglement to process information."
    ]
    
    # Create a GrpoRewards instance with the CompletionNegativeLengthCalculator
    rewards_calculator = GrpoRewards(
        function_name="CompletionNegativeLengthCalculator",
        prompts=prompts,
        completions=completions,
        function_params={"param1": "value1"}  # Optional parameters as dict
    )
    
    # Compute the rewards
    rewards = rewards_calculator.compute_rewards()
    
    # Print the results
    print("Prompts:")
    for i, prompt in enumerate(rewards_calculator.prompts):
        print(f"  {i}: {prompt}")
    
    print("\nCompletions:")
    for i, completion in enumerate(rewards_calculator.completions):
        print(f"  {i}: {completion}")
    
    print("\nRewards (negative length of completions):")
    for i, reward in enumerate(rewards):
        print(f"  {i}: {reward}")
    
    # Verify that the rewards are indeed the negative lengths of completions
    for i, (completion, reward) in enumerate(zip(completions, rewards)):
        expected = -float(len(completion))
        assert reward == expected, f"Reward {reward} does not match expected {expected} for completion {i}"
        print(f"  Verified: Length of completion {i} is {len(completion)}, reward is {reward}")
    
    print("\nAll rewards verified correctly!")

if __name__ == "__main__":
    main()