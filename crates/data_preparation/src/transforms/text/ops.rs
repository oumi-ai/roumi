use crate::sample::Sample;
use crate::transforms::Transform;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tch::Tensor;
use tokenizers::Tokenizer;

/// ===========================================================================
/// Tokenize text data
///
/// Wraps a HuggingFace [`Tokenizer`](tokenizers::Tokenizer)[1] to convert
/// raw `String` input into a `Sample` containing:
/// - `input_ids` (i64 tensor)
/// - `attention_mask` (i64 tensor)
/// - `token_type_ids` (i64 tensor, if present)
///
/// 1. https://docs.rs/tokenizers/latest/tokenizers/#modules
///
/// # Example
/// ```ignore
/// let hf = Tokenizer::from_pretrained("bert-base-uncased", None)?;
/// let pipeline = Tokenize::new(hf);
/// let sample = pipeline.apply("Hello world!".to_string())?;
/// assert!(sample.get("text_input_ids").is_ok());
/// assert!(sample.get("text_attention_mask").is_ok());
/// ```
#[derive(Debug)]
pub struct Tokenize {
    tokenizer: Tokenizer,
}

impl Tokenize {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }
}

impl Transform<String, Sample> for Tokenize {
    fn apply(&self, text: String) -> Result<Sample> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let mut features = HashMap::new();
        let convert =
            |data: &[u32]| Tensor::from_slice(&data.iter().map(|&x| x as i64).collect::<Vec<_>>());

        features.insert("input_ids".into(), convert(encoding.get_ids()));

        features.insert(
            "attention_mask".into(),
            convert(encoding.get_attention_mask()),
        );

        // Only add type IDs if non-empty
        if !encoding.get_type_ids().is_empty() {
            features.insert("token_type_ids".into(), convert(encoding.get_type_ids()));
        }

        Ok(Sample::new(features))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dataset::{Dataset, InMemoryDataset},
        sample::Sample,
        transforms::Transform,
    };
    use anyhow::{anyhow, Context, Result};
    use tokenizers::tokenizer::Tokenizer;

    fn load_tokenizer() -> Result<Tokenizer> {
        Tokenizer::from_file("src/tokenizer.json")
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))
    }

    #[test]
    fn test_tokenize_text() -> Result<()> {
        let tokenizer = load_tokenizer().context("Failed to initialize tokenizer for test")?;
        let pipeline = Tokenize::new(tokenizer);
        let sample = pipeline.apply("Hello Rust!".to_string())?;

        assert!(sample.get("input_ids").is_ok());
        assert!(sample.get("attention_mask").is_ok());
        Ok(())
    }

    #[test]
    fn test_tokenize_text_dataset() -> Result<()> {
        let tokenizer = load_tokenizer().context("Failed to initialize tokenizer for test")?;
        let pipeline = Tokenize::new(tokenizer);

        let texts = vec!["foo".into(), "bar".into()];
        let dataset = InMemoryDataset::new(texts).with_transform(pipeline);

        // It should yield two Samples with the right keys
        let out: Vec<Sample> = dataset.iter().collect::<Result<_>>()?;
        assert_eq!(out.len(), 2);
        for s in out {
            assert!(s.get("input_ids").is_ok());
            assert!(s.get("attention_mask").is_ok());
        }
        Ok(())
    }
}
