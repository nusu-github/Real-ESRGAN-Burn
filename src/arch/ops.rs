//! Small building blocks shared by the architectures.

use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::*;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};

/// `nn.Conv2d(channels_in, channels_out, 3, 1, 1)`.
pub fn conv3x3(channels_in: usize, channels_out: usize, device: &Device) -> Conv2d {
    Conv2dConfig::new([channels_in, channels_out], [3, 3])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .init(device)
}

/// `F.interpolate(x, scale_factor=factor, mode="nearest")`.
pub fn upsample_nearest(x: Tensor<4>, factor: usize) -> Tensor<4> {
    let [_, _, height, width] = x.dims();
    interpolate(
        x,
        InterpolateOptions::new(InterpolateMode::Nearest)
            .with_output_size([height * factor, width * factor]),
    )
}

/// `F.interpolate(x, scale_factor=factor, mode="bilinear", align_corners=False)`.
///
/// Burn aligns corners by default; PyTorch does not.
pub fn upsample_bilinear(x: Tensor<4>, factor: usize) -> Tensor<4> {
    let [_, _, height, width] = x.dims();
    interpolate(
        x,
        InterpolateOptions::new(InterpolateMode::Bilinear)
            .with_align_corners(false)
            .with_output_size([height * factor, width * factor]),
    )
}
