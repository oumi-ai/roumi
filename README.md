# roumi
Rust libraries for Oumi

This is a repository for Oumi's Rust libraries. It will hopefully grow substantially in the near future, but for now there is a grpo_rewards crate that calculates rewards based on completion length. There is a main.rs in the crates directory for an example of how to call it from rust and a python example in the examples directory for how to call it from python.


## Prerequisites

- **Rust:** [Install Rust](https://www.rust-lang.org/tools/install)
- **Python:** Version 3.10 or higher is recommended.
- **Maturin:** install via pip below

## Setting up a Python Virtual Environment 

(Optional, but Recommended)

```bash
python -m venv venv
venv\Scripts\activate # on Windows
source venv/bin/activate # on macOS/Linux
pip install -U pip
```

## Install Maturin
```bash
pip install maturin
```

## Build each crate for Python to use
```bash
cd crates/grpo_rewards
maturin develop
```

## Run the Python Example
```bash
cd ../../examples
python example_grpo_rewards.py
```

## Building and Running the Rust Code
```bash
cargo run
```

## Testing
```bash
cargo test --lib --bins --tests
```




