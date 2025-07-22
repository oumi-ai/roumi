use crate::{sample::Sample, transforms::Transform};
use anyhow::{Context, Result};
use image::DynamicImage;

/// Container for vision + text multimodal input data
#[derive(Clone)]
pub struct MultimodalInput {
    pub image: DynamicImage,
    pub text: String,
}

/// Multimodal transform pipeline
///
/// # Type Parameters
/// - `V`: Vision transform pipeline (implements `Transform<DynamicImage, Sample>`)
/// - `T`: Text transform pipeline (implements `Transform<String, Sample>`)
///
/// Processes input data by:
/// 1. Running `vision_pipeline` on the `input.image` (DynamicImage)
/// 2. Running `text_pipeline` on the `input.text` (String)
/// 3. Merging the resulting Samples
///
/// # Example
/// ```ignore
/// let multimodal = MultimodalPipeline::new(vision_pipeline, text_pipeline);
/// let sample = pipeline.apply(MultimodalInput {
///     image: loaded_image,  // DynamicImage
///     text: "description".into()  // String
/// })?;
/// ```
#[derive(Debug, Clone)]
pub struct MultimodalPipeline<V, T> {
    vision_pipeline: V,
    text_pipeline: T,
}

impl<V, T> MultimodalPipeline<V, T> {
    /// Combine a vision and a text pipeline into one multimodal transform.
    pub fn new(vision_pipeline: V, text_pipeline: T) -> Self {
        Self {
            vision_pipeline,
            text_pipeline,
        }
    }
}

impl<V, T> Transform<MultimodalInput, Sample> for MultimodalPipeline<V, T>
where
    V: Transform<DynamicImage, Sample>,
    T: Transform<String, Sample>,
{
    fn apply(&self, input: MultimodalInput) -> Result<Sample> {
        let vision_sample = self
            .vision_pipeline
            .apply(input.image)
            .context("Vision processing failed")?;
        let text_sample = self
            .text_pipeline
            .apply(input.text)
            .context("Text processing failed")?;
        vision_sample.merge(text_sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dataset::{Dataset, InMemoryDataset},
        transforms::{text::Tokenize, vision::*, ToSample},
    };
    use anyhow::Result;
    use image::{imageops::FilterType, RgbImage};
    use tch::Kind;
    use tokenizers::Tokenizer;

    // Test RGB image (2x2 pixels)
    fn test_image() -> image::DynamicImage {
        let buf = vec![
            10u8, 20, 30, 40, 50, 60, // Pixel 1, Pixel 2
            70u8, 80, 90, 100, 110, 120, // Pixel 3, Pixel 4
        ];
        image::DynamicImage::ImageRgb8(RgbImage::from_raw(2, 2, buf).unwrap())
    }

    #[test]
    fn test_multimodal_pipeline() -> Result<()> {
        let vision_pipelineline = Resize::new(2, 2, FilterType::Nearest)?
            .then(ToTensor)
            .then(Normalize::imagenet())
            .then(ToSample::new("pixels"));

        let tokenizer = Tokenizer::from_file("src/tokenizer.json")
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
        let text_pipelineline = Tokenize::new(tokenizer);

        // Create multimodal pipeline
        let multimodal = MultimodalPipeline::new(vision_pipelineline, text_pipelineline);

        // Create test dataset
        let dataset = InMemoryDataset::new(vec![
            MultimodalInput {
                image: test_image(),
                text: "red square".into(),
            },
            MultimodalInput {
                image: test_image(),
                text: "blue circle".into(),
            },
        ])
        .with_transform(multimodal);

        // Verify processing
        for sample in dataset.iter() {
            let sample = sample?;

            // Check vision output
            let pixels = sample.get("pixels")?;
            assert_eq!(pixels.size(), &[3, 2, 2]); // CHW format
            assert_eq!(pixels.kind(), Kind::Float);

            // Check text output
            let input_ids = sample.get("input_ids")?;
            assert!(input_ids.size()[0] > 0); // At least 1 token
            assert_eq!(input_ids.kind(), Kind::Int64);
        }

        Ok(())
    }
}
