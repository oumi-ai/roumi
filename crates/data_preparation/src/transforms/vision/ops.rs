use crate::dataloader::worker_gen_bool;
use crate::transforms::Transform;
use anyhow::{ensure, Context, Result};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use tch::{Kind, Tensor};

/// This script contains transformations useful for image preprocessing.
///
/// # Example Pipeline
/// ```ignore
/// use crate::transforms::vision::*;
///
/// let pipeline = Resize::new(256, 256, FilterType::Lanczos3)?
///     .then(RandomHorizontalFlip::new(0.5)?)
///     .then(ToTensor)
///     .then(Normalize::imagenet());
/// ```
///
/// ===========================================================================
/// Resizes an image to the specified dimension while preserving
/// the aspect ratio. Users must specify the filter type.
///
/// # Filter Types
/// - `Nearest`: Nearest neighbour, fastest
/// - `Triangle`: Bilinear filter, good all-round default
/// - `CatmullRom`: Bicubic sharpening
/// - `Gaussian`: Blurring/smoothing
/// - `Lanczos3`: Lanczos with window 3, highest quality re-sampling but slowest.
///
/// # Examples
/// ``` ignore
/// # use image::imageops::FilterType;
/// let resize = Resize::new(256, 256, FilterType::Triangle)?;
/// let img = load_image("cat.jpg")?;
/// let resized = resize.apply(img)?;
/// ```
#[derive(Debug)]
pub struct Resize {
    width: u32,
    height: u32,
    filter: FilterType,
}

impl Resize {
    /// Creates a new Resize transform.
    pub fn new(width: u32, height: u32, filter: FilterType) -> Result<Self> {
        ensure!(
            width > 0 && height > 0,
            "Image dimensions must be positive after resizing (got {}x{})",
            width,
            height
        );
        Ok(Self {
            width,
            height,
            filter,
        })
    }
}

impl Transform<DynamicImage, DynamicImage> for Resize {
    fn apply(&self, img: DynamicImage) -> Result<DynamicImage> {
        Ok(img.resize(self.width, self.height, self.filter))
    }
}

/// ===========================================================================
/// Randomly flips images horizontally during training.
/// Panics if probability `p` is outside [0.0, 1.0] range.
///
/// # Example
/// ```ignore
/// let flip = RandomHorizontalFlip::new(0.5)?; // 50% flip chance
/// let augmented = flip.apply(image)?;
/// ```
#[derive(Debug)]
pub struct RandomHorizontalFlip {
    p: f64,
}

impl RandomHorizontalFlip {
    pub fn new(p: f64) -> Result<Self> {
        ensure!(
            (0.0..=1.0).contains(&p),
            "Probability must be in [0.0, 1.0] range (got {})",
            p
        );
        Ok(Self { p })
    }
}

impl Transform<DynamicImage, DynamicImage> for RandomHorizontalFlip {
    fn apply(&self, img: DynamicImage) -> Result<DynamicImage> {
        Ok(if worker_gen_bool(self.p) {
            img.fliph()
        } else {
            img
        })
    }
}

/// ===========================================================================
/// Converts an image to a channel-first f32 tensor in [0.0, 1.0] range.
///
/// Channel Handling
/// | Input Format  | Output Shape |
/// |---------------|--------------|
/// | Grayscale (L) | `[1, H, W]`  |
/// | RGB           | `[3, H, W]`  |
/// | RGBA          | `[4, H, W]`  |
/// | Other         | `[3, H, W]`  |
/// Note: *CMYK, BGR, etc. will undergo implicit conversion to RGB.
///       For precise format control, pre-convert your images.
///
/// # Example
/// ```ignore
/// let converter = ToTensor;
/// let tensor = converter.apply(image)?;
/// ```
#[derive(Debug)]
pub struct ToTensor;

impl Transform<DynamicImage, Tensor> for ToTensor {
    fn apply(&self, img: DynamicImage) -> Result<Tensor> {
        let (width, height) = img.dimensions();
        ensure!(
            width > 0 && height > 0,
            "Image dimensions must be positive (got {}x{})",
            width,
            height
        );

        let tensor = match img {
            DynamicImage::ImageLuma8(img) => {
                Tensor::from_slice(img.as_raw()).reshape(&[1, height as i64, width as i64])
            }
            DynamicImage::ImageRgb8(img) => {
                Tensor::from_slice(img.as_raw()).reshape(&[3, height as i64, width as i64])
            }
            DynamicImage::ImageRgba8(img) => {
                Tensor::from_slice(img.as_raw()).reshape(&[4, height as i64, width as i64])
            }
            // Handle all other cases via conversion to RGB
            _ => {
                let rgb = img.to_rgb8();
                Tensor::from_slice(rgb.as_raw()).reshape(&[3, height as i64, width as i64])
            }
        };

        // Normalize to [0,1] range
        tensor
            .to_kind(Kind::Float)
            .f_div_scalar(255.0)
            .context("Failed to normalize tensor values")
    }
}

/// ===========================================================================
/// Normalizes tensors using channel-wise statistics.
///
/// # Arguments:
/// - `mean`: per-channel means
/// - `std`: per-channel standard deviation.
/// The dimensions of mean and and std should match the input tensor's
/// number of channels.
///
/// # Mathematical Operation:
/// ```text
/// output[...,c,h,w] = (input[...,c,h,w] - mean[c]) / std[c]
/// ```
///
/// # Example
/// ```ignore
/// let norm = Normalize::imagenet();
/// let normalized = norm.apply(tensor)?;
/// ```
#[derive(Debug)]
pub struct Normalize {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl Normalize {
    /// Creates new normalization parameters.
    pub fn new(mean: &[f32], std: &[f32]) -> Result<Self> {
        ensure!(!mean.is_empty(), "Normalization mean cannot be empty");
        ensure!(
            mean.len() == std.len(),
            "The mean and standard deviation for normalization must match in dimension. 
            The dimension of mean is {} but the dimension of std is {}. ",
            mean.len(),
            std.len()
        );
        Ok(Self {
            mean: mean.to_vec(),
            std: std.to_vec(),
        })
    }

    /// ImageNet standard normalization (RGB)
    pub fn imagenet() -> Self {
        Self {
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
        }
    }
}

impl Transform<Tensor, Tensor> for Normalize {
    fn apply(&self, tensor: Tensor) -> Result<Tensor> {
        let (num_channels, _height, _width) = tensor
            .size3()
            .context("Input must be 3D tensor [C, H, W]")?;

        ensure!(
            num_channels as usize == self.mean.len(),
            "Channel count mismatch: input has {} channels but normalization expects {} ",
            num_channels,
            self.mean.len()
        );

        let mean_t = Tensor::from_slice(&self.mean)
            .reshape(&[num_channels, 1, 1])
            .to_kind(tensor.kind());

        let std_t = Tensor::from_slice(&self.std)
            .reshape(&[num_channels, 1, 1])
            .to_kind(tensor.kind());

        Ok((tensor - mean_t) / std_t)
    }
}

/// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Dataset, InMemoryDataset};
    use crate::transforms::{ToSample, Transform};
    use image::{Rgb, RgbImage};
    use tch::{Device, Kind, Tensor};

    // Helper: Create a test RGB image (3x3 pixels)
    fn test_rgb_image() -> DynamicImage {
        let mut img = RgbImage::new(3, 3);
        for x in 0..3 {
            for y in 0..3 {
                img.put_pixel(x, y, Rgb([(x * 85) as u8, (y * 85) as u8, 128]));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_vision_resize_transform() -> Result<()> {
        let img = test_rgb_image();
        let resize = Resize::new(6, 6, FilterType::Triangle)?;

        let resized = resize.apply(img)?;
        assert_eq!(resized.dimensions(), (6, 6));
        Ok(())
    }

    #[test]
    fn test_vision_image_to_tensor_transform() -> Result<()> {
        let img = test_rgb_image();
        let converter = ToTensor;

        let tensor = converter.apply(img)?;
        assert_eq!(tensor.size(), vec![3, 3, 3]); // CHW format
        assert_eq!(tensor.kind(), Kind::Float);

        // Verify normalization to [0,1]
        let min = tensor.f_min()?.double_value(&[]);
        let max = tensor.f_max()?.double_value(&[]);
        assert!(min >= 0.0 && max <= 1.0);
        Ok(())
    }

    #[test]
    fn test_vision_tensor_data_normalize_transform() -> Result<()> {
        let tensor = Tensor::ones(&[3, 32, 32], (Kind::Float, Device::Cpu));
        let norm = Normalize::new(&[1.0; 3], &[1.0; 3])?;

        let normalized = norm.apply(tensor)?;

        // Check each channel's mean separately
        for c in 0..3 {
            let channel_mean = normalized.select(0, c).mean(Kind::Float);
            assert!(channel_mean.double_value(&[]) < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn test_vision_random_horizontal_flip_transform() -> Result<()> {
        // construct a 2×1 image where left = red, right = blue
        let mut img = RgbImage::new(2, 1);
        img.put_pixel(0, 0, Rgb([255, 0, 0])); // red
        img.put_pixel(1, 0, Rgb([0, 0, 255])); // blue

        let flip = RandomHorizontalFlip::new(1.0)?; // Always flip
        let flipped = flip.apply(DynamicImage::ImageRgb8(img))?;

        // After flip, left should be blue, right should be red:
        assert_eq!(flipped.as_bytes(), &[0, 0, 255, 255, 0, 0]);
        Ok(())
    }

    #[test]
    fn test_vision_transforms_chaining() -> Result<()> {
        let pipeline = Resize::new(16, 16, FilterType::Nearest)?
            .then(RandomHorizontalFlip::new(0.0)?) // No flip
            .then(ToTensor)
            .then(Normalize::imagenet());
        let img = test_rgb_image();
        let tensor = pipeline.apply(img)?;
        assert_eq!(tensor.size(), vec![3, 16, 16]);
        assert_eq!(tensor.kind(), Kind::Float);
        Ok(())
    }

    #[test]
    fn test_vision_pipeline_with_dataset() -> Result<()> {
        // 1. Create pipeline
        let pipeline = Resize::new(16, 16, FilterType::Nearest)?
            .then(RandomHorizontalFlip::new(0.5)?) // 50% flip chance
            .then(ToTensor)
            .then(Normalize::imagenet())
            .then(ToSample::new("pixels"));

        // 2. Create dataset with test images
        let images = vec![test_rgb_image(), test_rgb_image(), test_rgb_image()];
        let dataset = InMemoryDataset::new(images).with_transform(pipeline);

        // 3. Verify transformed samples
        for sample in dataset.iter() {
            let sample = sample?;
            let tensor = sample.get("pixels")?;

            assert_eq!(tensor.size(), vec![3, 16, 16]); // CHW format
            assert_eq!(tensor.kind(), Kind::Float);
        }

        Ok(())
    }
}
