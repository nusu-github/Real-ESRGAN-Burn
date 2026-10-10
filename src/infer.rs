//! Image pre/post-processing and tiled inference, following `realesrgan.utils.RealESRGANer`.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use burn::prelude::*;
use burn::tensor::ops::PadMode;
use image::{Rgb, RgbImage};

use crate::arch::Generator;

/// Splits the input into `tile_size` squares, each run with `tile_pad` pixels of context on
/// every side, to bound memory use. `tile_size == 0` runs the whole image at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tiling {
    pub tile_size: usize,
    pub tile_pad: usize,
}

/// One tile of the input: `core` is the region it produces, `padded` the region it reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tile {
    pub core: Region,
    pub padded: Region,
}

/// A rectangle `[top, bottom) x [left, right)` in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub top: usize,
    pub left: usize,
    pub bottom: usize,
    pub right: usize,
}

impl Region {
    pub fn height(&self) -> usize {
        self.bottom - self.top
    }

    pub fn width(&self) -> usize {
        self.right - self.left
    }
}

impl Tiling {
    pub const NONE: Self = Self {
        tile_size: 0,
        tile_pad: 0,
    };

    /// The tiles covering a `height x width` image, row by row.
    pub fn tiles(&self, height: usize, width: usize) -> Vec<Tile> {
        let step = if self.tile_size == 0 {
            height.max(width).max(1)
        } else {
            self.tile_size
        };
        let pad = self.tile_pad;

        let mut tiles = Vec::new();
        for top in (0..height).step_by(step) {
            for left in (0..width).step_by(step) {
                let core = Region {
                    top,
                    left,
                    bottom: (top + step).min(height),
                    right: (left + step).min(width),
                };
                let padded = Region {
                    top: top.saturating_sub(pad),
                    left: left.saturating_sub(pad),
                    bottom: (core.bottom + pad).min(height),
                    right: (core.right + pad).min(width),
                };
                tiles.push(Tile { core, padded });
            }
        }
        tiles
    }
}

/// Upscales `image` by `model.scale()`.
pub fn upscale(model: &Generator, image: &RgbImage, tiling: Tiling, device: &Device) -> RgbImage {
    let scale = model.scale();
    let input = rgb_image_to_tensor(image, device);
    let [_, _, height, width] = input.dims();
    let mut output = RgbImage::new((width * scale) as u32, (height * scale) as u32);

    for Tile { core, padded } in tiling.tiles(height, width) {
        let patch = input.clone().slice(s![
            ..,
            ..,
            padded.top..padded.bottom,
            padded.left..padded.right
        ]);
        let top = (core.top - padded.top) * scale;
        let left = (core.left - padded.left) * scale;
        let result = forward_padded(model, patch).slice(s![
            ..,
            ..,
            top..top + core.height() * scale,
            left..left + core.width() * scale
        ]);
        write_tensor_into(&mut output, result, core.left * scale, core.top * scale);
    }

    output
}

/// Runs `model` on an input of any size: pads the bottom and right edges up to the input
/// multiple the network needs (reflecting, like `RealESRGANer`), then crops the result.
fn forward_padded(model: &Generator, x: Tensor<4>) -> Tensor<4> {
    let [_, _, height, width] = x.dims();
    let multiple = model.input_multiple();
    let pad_h = height.next_multiple_of(multiple) - height;
    let pad_w = width.next_multiple_of(multiple) - width;
    if pad_h == 0 && pad_w == 0 {
        return model.forward(x);
    }

    // Reflection needs more pixels than it pads; tiny inputs fall back to edge replication.
    let mode = if pad_h < height && pad_w < width {
        PadMode::Reflect
    } else {
        PadMode::Edge
    };
    let scale = model.scale();
    model
        .forward(x.pad([(0, pad_h), (0, pad_w)], mode))
        .slice(s![.., .., 0..height * scale, 0..width * scale])
}

pub fn load_rgb_image(path: &Path, max_edge: Option<usize>) -> Result<RgbImage> {
    let image = image::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .to_rgb8();

    let (w, h) = image.dimensions();
    let longest = w.max(h) as usize;
    match max_edge {
        Some(max_edge) if longest > max_edge => {
            let ratio = max_edge as f64 / longest as f64;
            let new_w = ((f64::from(w) * ratio).round() as u32).max(1);
            let new_h = ((f64::from(h) * ratio).round() as u32).max(1);
            Ok(image::imageops::resize(
                &image,
                new_w,
                new_h,
                image::imageops::FilterType::CatmullRom,
            ))
        }
        _ => Ok(image),
    }
}

pub fn save_rgb_image(image: &RgbImage, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    image
        .save(path)
        .with_context(|| format!("failed to save {}", path.display()))
}

/// `[1, 3, height, width]` with values in `[0, 1]`.
pub fn rgb_image_to_tensor(image: &RgbImage, device: &Device) -> Tensor<4> {
    let (width, height) = image.dimensions();
    let (width, height) = (width as usize, height as usize);
    let plane = width * height;

    let mut data = vec![0.0f32; 3 * plane];
    for (idx, pixel) in image.pixels().enumerate() {
        for (channel, &value) in pixel.0.iter().enumerate() {
            data[channel * plane + idx] = f32::from(value) / 255.0;
        }
    }

    Tensor::from_data(TensorData::new(data, [1, 3, height, width]), device)
}

/// Inverse of [`rgb_image_to_tensor`]; values are clamped to `[0, 1]`.
pub fn tensor_to_rgb_image(tensor: Tensor<4>) -> Result<RgbImage> {
    let [batch, channels, height, width] = tensor.dims();
    ensure!(
        batch == 1 && channels == 3,
        "expected shape [1, 3, H, W], got [{batch}, {channels}, {height}, {width}]"
    );
    let mut image = RgbImage::new(width as u32, height as u32);
    write_tensor_into(&mut image, tensor, 0, 0);
    Ok(image)
}

/// Copies a `[1, 3, h, w]` tensor into `image` with its top-left corner at `(x0, y0)`.
fn write_tensor_into(image: &mut RgbImage, tensor: Tensor<4>, x0: usize, y0: usize) {
    let [_, _, height, width] = tensor.dims();
    let plane = height * width;
    let values = tensor
        .into_data()
        .try_to_vec_as::<f32>()
        .expect("float tensor data converts to f32");
    let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let pixel = Rgb([
                to_u8(values[idx]),
                to_u8(values[plane + idx]),
                to_u8(values[2 * plane + idx]),
            ]);
            image.put_pixel((x0 + x) as u32, (y0 + y) as u32, pixel);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_cover_the_image_once() {
        let tiling = Tiling {
            tile_size: 4,
            tile_pad: 1,
        };
        let tiles = tiling.tiles(6, 9);
        assert_eq!(tiles.len(), 2 * 3);

        let mut covered = vec![0; 6 * 9];
        for tile in &tiles {
            for y in tile.core.top..tile.core.bottom {
                for x in tile.core.left..tile.core.right {
                    covered[y * 9 + x] += 1;
                }
            }
        }
        assert!(covered.iter().all(|&n| n == 1));

        let last = tiles.last().unwrap();
        let expected_core = Region {
            top: 4,
            left: 8,
            bottom: 6,
            right: 9,
        };
        let expected_padded = Region {
            top: 3,
            left: 7,
            bottom: 6,
            right: 9,
        };
        assert_eq!(last.core, expected_core);
        assert_eq!(last.padded, expected_padded);
    }

    #[test]
    fn no_tiling_is_one_tile() {
        let tiles = Tiling::NONE.tiles(5, 7);
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].core, tiles[0].padded);
        assert_eq!((tiles[0].core.height(), tiles[0].core.width()), (5, 7));
    }

    #[cfg(feature = "flex")]
    mod flex {
        use super::super::*;
        use crate::arch::{GeneratorConfig, RrdbNetConfig, SrvggNetCompactConfig};

        fn test_image(width: u32, height: u32) -> RgbImage {
            RgbImage::from_fn(width, height, |x, y| {
                Rgb([(x * 20) as u8, (y * 30) as u8, ((x + y) * 7) as u8])
            })
        }

        #[test]
        fn rgb_tensor_roundtrip_is_lossless() {
            let device = Device::flex();
            let image = test_image(5, 3);
            let tensor = rgb_image_to_tensor(&image, &device);
            assert_eq!(tensor.dims(), [1, 3, 3, 5]);
            assert_eq!(tensor_to_rgb_image(tensor).unwrap(), image);
        }

        #[test]
        fn tiled_and_untiled_outputs_match() {
            let device = Device::flex();
            let model = GeneratorConfig::Srvgg(
                SrvggNetCompactConfig::new(4)
                    .with_num_feat(4)
                    .with_num_conv(1),
            )
            .init(&device);
            let image = test_image(11, 7);

            let whole = upscale(&model, &image, Tiling::NONE, &device);
            // The receptive field (3 convs, 3x3) fits in the padding, so tiles are seamless.
            let tiling = Tiling {
                tile_size: 4,
                tile_pad: 3,
            };
            let tiled = upscale(&model, &image, tiling, &device);
            assert_eq!(whole.dimensions(), (44, 28));
            assert_eq!(tiled.dimensions(), whole.dimensions());
            // Allow one step of rounding noise from different summation orders.
            let max_diff = tiled
                .as_raw()
                .iter()
                .zip(whole.as_raw())
                .map(|(a, b)| a.abs_diff(*b))
                .max();
            assert!(max_diff <= Some(1), "max pixel difference {max_diff:?}");
        }

        #[test]
        fn odd_sizes_are_padded_for_pixel_unshuffle() {
            let device = Device::flex();
            let model = GeneratorConfig::Rrdb(
                RrdbNetConfig::new(2)
                    .with_num_feat(4)
                    .with_num_block(1)
                    .with_num_grow_ch(2),
            )
            .init(&device);
            let tiling = Tiling {
                tile_size: 4,
                tile_pad: 1,
            };
            let output = upscale(&model, &test_image(7, 5), tiling, &device);
            assert_eq!(output.dimensions(), (14, 10));
        }
    }
}
