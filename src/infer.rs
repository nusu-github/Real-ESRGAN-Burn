use std::path::Path;

use anyhow::{Result, anyhow, bail};
use burn::prelude::*;
use image::{Rgb, RgbImage};

use crate::arch::rrdb::RrdbNet;
use crate::arch::srvgg::SrvggNetCompact;

pub enum GeneratorModel<B: Backend> {
    Rrdb(Box<RrdbNet<B>>),
    Srvgg(Box<SrvggNetCompact<B>>),
}

impl<B: Backend> GeneratorModel<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Self::Rrdb(model) => model.forward(input),
            Self::Srvgg(model) => model.forward(input),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_generator_inference<B: Backend>(
    model: &GeneratorModel<B>,
    scale: usize,
    input_path: &Path,
    output_path: &Path,
    max_input_edge: Option<usize>,
    tile_size: usize,
    tile_pad: usize,
    device: &B::Device,
) -> Result<()> {
    let input_image = load_rgb_image(input_path, max_input_edge)?;
    let input_tensor = rgb_image_to_tensor::<B>(&input_image, device);
    let output_tensor =
        run_generator_forward_tiled(model, input_tensor, scale, tile_size, tile_pad, device);
    let output_image = tensor_to_rgb_image(output_tensor)?;
    save_rgb_image(&output_image, output_path)
}

fn run_generator_forward_tiled<B: Backend>(
    model: &GeneratorModel<B>,
    input: Tensor<B, 4>,
    scale: usize,
    tile_size: usize,
    tile_pad: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let [batch, channels, height, width] = input.dims();
    if tile_size == 0 || (height <= tile_size && width <= tile_size) {
        return model.forward(input);
    }

    let mut output =
        Tensor::<B, 4>::zeros([batch, channels, height * scale, width * scale], device);

    let mut top = 0usize;
    while top < height {
        let bottom = (top + tile_size).min(height);
        let mut left = 0usize;

        while left < width {
            let right = (left + tile_size).min(width);

            let pad_top = top.saturating_sub(tile_pad);
            let pad_left = left.saturating_sub(tile_pad);
            let pad_bottom = (bottom + tile_pad).min(height);
            let pad_right = (right + tile_pad).min(width);

            let tile = input.clone().slice([
                0..batch,
                0..channels,
                pad_top..pad_bottom,
                pad_left..pad_right,
            ]);
            let tile_output = model.forward(tile);

            let crop_top = (top - pad_top) * scale;
            let crop_left = (left - pad_left) * scale;
            let crop_bottom = crop_top + (bottom - top) * scale;
            let crop_right = crop_left + (right - left) * scale;

            let tile_output = tile_output.slice([
                0..batch,
                0..channels,
                crop_top..crop_bottom,
                crop_left..crop_right,
            ]);

            let out_top = top * scale;
            let out_left = left * scale;
            let out_bottom = bottom * scale;
            let out_right = right * scale;

            output = output.slice_assign(
                [
                    0..batch,
                    0..channels,
                    out_top..out_bottom,
                    out_left..out_right,
                ],
                tile_output,
            );

            left += tile_size;
        }

        top += tile_size;
    }

    output
}

pub fn load_rgb_image(path: &Path, max_input_edge: Option<usize>) -> Result<RgbImage> {
    let image = image::open(path).map_err(|e| anyhow!("failed to open {:?}: {e}", path))?;
    let image = image.to_rgb8();

    if let Some(max_edge) = max_input_edge {
        let (w, h) = image.dimensions();
        let current_max = w.max(h) as usize;
        if current_max > max_edge {
            let scale = max_edge as f32 / current_max as f32;
            let new_w = (w as f32 * scale).round().max(1.0) as u32;
            let new_h = (h as f32 * scale).round().max(1.0) as u32;
            return Ok(image::imageops::resize(
                &image,
                new_w,
                new_h,
                image::imageops::FilterType::CatmullRom,
            ));
        }
    }

    Ok(image)
}

pub fn save_rgb_image(image: &RgbImage, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    image
        .save(path)
        .map_err(|e| anyhow!("failed to save {:?}: {e}", path))
}

pub fn rgb_image_to_tensor<B: Backend>(image: &RgbImage, device: &B::Device) -> Tensor<B, 4> {
    let (width, height) = image.dimensions();
    let width = width as usize;
    let height = height as usize;
    let mut data = vec![0.0f32; 3 * width * height];

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x as u32, y as u32);
            let idx = y * width + x;
            data[idx] = f32::from(pixel[0]) / 255.0;
            data[width * height + idx] = f32::from(pixel[1]) / 255.0;
            data[2 * width * height + idx] = f32::from(pixel[2]) / 255.0;
        }
    }

    let tensor_data = TensorData::new(data, [1, 3, height, width]);
    Tensor::<B, 4>::from_data(tensor_data, device)
}

pub fn tensor_to_rgb_image<B: Backend>(tensor: Tensor<B, 4>) -> Result<RgbImage> {
    let [batch, channels, height, width] = tensor.dims();
    if batch != 1 || channels != 3 {
        bail!("expected output shape [1, 3, H, W], got [{batch}, {channels}, {height}, {width}]");
    }

    let values = tensor
        .to_data()
        .into_vec::<f32>()
        .map_err(|e| anyhow!("failed to decode tensor data: {e}"))?;

    let expected_len = 3 * height * width;
    if values.len() != expected_len {
        bail!(
            "unexpected tensor data len {}, expected {}",
            values.len(),
            expected_len
        );
    }

    let mut image = RgbImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let r = (values[idx].clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (values[height * width + idx].clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (values[2 * height * width + idx].clamp(0.0, 1.0) * 255.0).round() as u8;
            image.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn::backend::NdArray;

    #[test]
    fn rgb_image_tensor_roundtrip_preserves_shape() {
        let mut image = RgbImage::new(2, 2);
        image.put_pixel(0, 0, Rgb([10, 20, 30]));
        image.put_pixel(1, 0, Rgb([40, 50, 60]));
        image.put_pixel(0, 1, Rgb([70, 80, 90]));
        image.put_pixel(1, 1, Rgb([100, 110, 120]));

        let device = Default::default();
        let tensor = rgb_image_to_tensor::<B>(&image, &device);
        assert_eq!(tensor.dims(), [1, 3, 2, 2]);

        let output = tensor_to_rgb_image::<B>(tensor).expect("tensor conversion should succeed");
        assert_eq!(output.dimensions(), (2, 2));
    }
}
