//! Thread-local storage for worker identification.
//!
//! Provides a thread-local worker ID that allows workers to identify themselves
//! for debugging, error messages, and task distribution.

use rand::rngs::StdRng;
use rand::Rng as _;
use rand::SeedableRng;
use std::cell::RefCell;

thread_local! {
    /// Thread-local worker ID.
    ///
    /// Each worker thread is assigned a unique ID (0 to num_workers-1) when spawned.
    /// This ID is used for:
    /// - Task distribution
    /// - Debugging and error messages
    /// - Worker-specific configuration (future)
    pub static WORKER_ID: RefCell<usize> = RefCell::new(0);

    /// Thread-local RNG for deterministic randomness in workers
    pub static WORKER_RNG: RefCell<Option<StdRng>> = RefCell::new(None);
}

/// Initialize worker's RNG based on worker_id, epoch, and base seed.
/// Seed formula: base_seed + (epoch << 32) + worker_id
/// This ensures each worker has unique but deterministic randomness.
pub fn init_worker_rng(worker_id: usize, epoch: usize, base_seed: u64) {
    WORKER_RNG.with(|rng| {
        let seed = base_seed
            .wrapping_add((epoch as u64) << 32)
            .wrapping_add(worker_id as u64);
        *rng.borrow_mut() = Some(StdRng::seed_from_u64(seed));
    })
}

/// Get a random bool from worker RNG, or thread_rng if not in worker context.
/// Used by transforms like RandomHorizontalFlip to ensure determinism.
pub fn worker_gen_bool(p: f64) -> bool {
    WORKER_RNG.with(|rng| {
        let mut rng_ref = rng.borrow_mut();
        match rng_ref.as_mut() {
            Some(rng) => rng.random_bool(p),
            None => rand::rng().random_bool(p),
        }
    })
}
