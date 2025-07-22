//! src/dataloader/mod.rs
//!
//! This module implements the `DataLoader`.
//!
//! The `DataLoader` coordinates the `Dataset`, `Sampler`, and `Collator` to efficiently
//! load and batch data for training. It supports both in-memory and streaming datasets
//! with optional parallel loading.
//!
//! # Architecture Overview
//!
//! ```text
//!                    ┌─────────┐
//!                    │ Dataset │ (InMemory or Iterable)
//!                    └────┬────┘
//!                         │ provides raw data
//!                         ↓
//!                  ┌──────────────┐
//!                  │  Transform   │ (Raw → Sample)
//!                  └──────┬───────┘
//!                         │ converts to Sample
//!                         ↓
//!                    ┌─────────┐
//!                    │ Sampler │ (defines iteration order)
//!                    └────┬────┘
//!                         │ provides indices/order
//!                         ↓
//!                  ┌──────────────┐
//!                  │  DataLoader  │ ←───── Config (batch_size, workers, etc.)
//!                  └──────┬───────┘
//!                         │ coordinates everything
//!                         ↓
//!                   [Worker Threads] (optional parallelism)
//!                         │
//!                         │ fetches & transforms samples
//!                         ↓
//!                    ┌─────────┐
//!                    │ Collator │ (combines samples into batch)
//!                    └────┬────┘
//!                         │ stacks/pads tensors
//!                         ↓
//!                   ┌───────────┐
//!                   │ MiniBatch │ (ready for model)
//!                   └───────────┘
//! ```
//!
//! # Module Structure
//!
//! ```text
//! src/dataloader/
//! ├── mod.rs             # Public API exports + module-level architecture docs
//! ├── config.rs          # DataLoaderConfig, builder, and validation
//! ├── loader.rs          # DataLoader struct and constructors
//! ├── iterator/          
//! │   ├── mod.rs         # DataLoaderIter and common iteration logic
//! │   ├── inmemory.rs    # InMemory-specific iterator implementations
//! │   └── iterable.rs    # Iterable-specific iterator implementations
//! ├── workers/           
//! │   ├── mod.rs         # Worker-related types and traits
//! │   ├── pool.rs        # Generic `WorkerPool<Task, Output>` implementation
//! │   ├── inmemory.rs    # InMemoryWorkerManager and task types
//! │   └── iterable.rs    # IterableWorkerManager and control types
//! └── common/            
//!     ├── mod.rs         # Module declarations for shared utilities
//!     └── thread.rs      # Thread-local worker ID
//! ```
//!
//! # Example Usage
//!
//! ## Basic single-threaded usage:
//! ```ignore
//! let dataset = InMemoryDataset::new(data).with_transform(transform);
//! let sampler = RandomSampler::new(dataset.len(), false, None, seed)?;
//! let config = DataLoaderConfig::builder()
//!     .batch_size(32)
//!     .build();
//!
//! let dataloader = DataLoader::new(dataset, sampler, config)?;
//!
//! for batch in dataloader.iter()?{
//!     let batch: MiniBatch = batch?;
//!     // Use batch.get("input_ids")?, etc.
//! }
//! ```
//!
//! ## Multi-threaded with persistent workers
//! ```ignore
//! let config = DataLoaderConfig::builder()
//!     .batch_size(32)
//!     .num_workers(4)
//!     .persistent_workers(true)
//!     .prefetch_factor(2)
//!     .build();
//!
//! let dataloader = DataLoader::new(dataset, sampler, config)?;
//! ```
//!
//! ## Streaming large datasets
//! ```ignore
//! let dataset = IterableDataset::new(data_sources).with_transform(transform);
//! let config = DataLoaderConfig::builder()
//!     .batch_size(64)
//!     .num_workers(8)
//!     .build();
//!
//! let dataloader = DataLoader::new_iterable(dataset, config)?;
//! ```
//!
//! # Performance Guidelines
//!
//! ## Worker Configuration
//! - `num_workers = 0`: Single-threaded, lowest memory usage but no parallelism
//! - `num_workers > 0`: Multi-threaded parallel loading
//!
//! ## Memory Usage
//! - Single-threaded: O(batch_size)
//! - Multi-threaded: O(num_workers x prefetch_factor x batch_size)
//!
//! ## Notes:
//! - Enable `shuffle = true` for better model generalization
//! - Enable `persistent_workers = true` for multi-epoch training
//! - Increase `prefetch_factor` if GPU is starved for data
//! - Reduce `batch_size`, `num_workers` or `prefetch_factor` if out-of-memory.

// Module declarations
mod common;
mod config;
mod iterator;
mod loader;
mod workers;

// Public re-exports
pub use config::{DataLoaderConfig, DataLoaderConfigBuilder};
pub use loader::DataLoader;

pub use common::thread::{worker_gen_bool, WORKER_ID, WORKER_RNG};
