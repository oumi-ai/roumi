/// A `Sampler` defines the strategy for how to iterate and draw samples from
/// a dataset.
///
/// # Associated type
/// - `Item`: The handle yielded by the iterator
///    - For in-memory datasets this is a `usize` index
///    - For streaming datasets it could be just the next item `()` or
///      a custom handle like `(shard, offset)`.
///
/// # Method
/// - `iter(epoch)`: returns a shuffled sequence for that epoch.
///    - Users pass the `epoch` parameter so internally the sampler uses it
///      together with the base seed to shuffle in a reproducible way across epochs.
///
/// Implementations must be `Send + Sync` so the same sampler instance can be
/// safely shared across DataLoader worker threads.
pub trait Sampler: Send + Sync {
    type Item;

    fn iter(&self, epoch: usize) -> Box<dyn Iterator<Item = Self::Item> + Send + '_>;
}
