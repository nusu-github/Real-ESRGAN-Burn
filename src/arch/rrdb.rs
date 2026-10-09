//! RRDBNet, the ESRGAN generator (`basicsr.archs.rrdbnet_arch.RRDBNet`).
//!
//! Field names mirror the PyTorch module, so official checkpoints load without key remapping.

use burn::nn::conv::Conv2d;
use burn::nn::{PixelUnshuffle, PixelUnshuffleConfig};
use burn::prelude::*;
use burn::tensor::activation::leaky_relu;

use super::ops::{conv3x3, upsample_nearest};

const NEGATIVE_SLOPE: f64 = 0.2;
/// Residual scaling applied inside every dense block and RRDB.
const RESIDUAL_SCALE: f64 = 0.2;

#[derive(Config, Debug)]
pub struct RrdbNetConfig {
    /// Upscaling factor: 1, 2 or 4.
    pub scale: usize,
    #[config(default = 3)]
    pub num_in_ch: usize,
    #[config(default = 3)]
    pub num_out_ch: usize,
    #[config(default = 64)]
    pub num_feat: usize,
    #[config(default = 23)]
    pub num_block: usize,
    #[config(default = 32)]
    pub num_grow_ch: usize,
}

impl RrdbNetConfig {
    /// # Panics
    ///
    /// If `scale` is not 1, 2 or 4.
    pub fn init(&self, device: &Device) -> RrdbNet {
        // The trunk always upsamples 4x; smaller scales fold pixels into channels first.
        let unshuffle_factor = match self.scale {
            4 => None,
            2 => Some(2),
            1 => Some(4),
            scale => panic!("RRDBNet supports scale 1, 2 or 4, got {scale}"),
        };
        let in_ch = self.num_in_ch * unshuffle_factor.map_or(1, |f| f * f);
        let feat = self.num_feat;

        RrdbNet {
            unshuffle: unshuffle_factor.map(|f| PixelUnshuffleConfig::new(f).init()),
            conv_first: conv3x3(in_ch, feat, device),
            body: (0..self.num_block)
                .map(|_| Rrdb::new(feat, self.num_grow_ch, device))
                .collect(),
            conv_body: conv3x3(feat, feat, device),
            conv_up1: conv3x3(feat, feat, device),
            conv_up2: conv3x3(feat, feat, device),
            conv_hr: conv3x3(feat, feat, device),
            conv_last: conv3x3(feat, self.num_out_ch, device),
            scale: self.scale,
        }
    }
}

#[derive(Module, Debug)]
pub struct RrdbNet {
    unshuffle: Option<PixelUnshuffle>,
    conv_first: Conv2d,
    body: Vec<Rrdb>,
    conv_body: Conv2d,
    conv_up1: Conv2d,
    conv_up2: Conv2d,
    conv_hr: Conv2d,
    conv_last: Conv2d,
    scale: usize,
}

impl RrdbNet {
    pub fn scale(&self) -> usize {
        self.scale
    }

    /// Input height and width must be multiples of this value.
    pub fn input_multiple(&self) -> usize {
        self.unshuffle.as_ref().map_or(1, |u| u.downscale_factor)
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = match &self.unshuffle {
            Some(unshuffle) => unshuffle.forward(x),
            None => x,
        };

        let feat = self.conv_first.forward(x);
        let body = self
            .body
            .iter()
            .fold(feat.clone(), |h, block| block.forward(h));
        let feat = feat + self.conv_body.forward(body);

        let feat = leaky_relu(
            self.conv_up1.forward(upsample_nearest(feat, 2)),
            NEGATIVE_SLOPE,
        );
        let feat = leaky_relu(
            self.conv_up2.forward(upsample_nearest(feat, 2)),
            NEGATIVE_SLOPE,
        );
        self.conv_last
            .forward(leaky_relu(self.conv_hr.forward(feat), NEGATIVE_SLOPE))
    }
}

/// Residual in Residual Dense Block.
#[derive(Module, Debug)]
pub struct Rrdb {
    rdb1: ResidualDenseBlock,
    rdb2: ResidualDenseBlock,
    rdb3: ResidualDenseBlock,
}

impl Rrdb {
    fn new(num_feat: usize, num_grow_ch: usize, device: &Device) -> Self {
        Self {
            rdb1: ResidualDenseBlock::new(num_feat, num_grow_ch, device),
            rdb2: ResidualDenseBlock::new(num_feat, num_grow_ch, device),
            rdb3: ResidualDenseBlock::new(num_feat, num_grow_ch, device),
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let out = self
            .rdb3
            .forward(self.rdb2.forward(self.rdb1.forward(x.clone())));
        x + out * RESIDUAL_SCALE
    }
}

#[derive(Module, Debug)]
pub struct ResidualDenseBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv4: Conv2d,
    conv5: Conv2d,
}

impl ResidualDenseBlock {
    fn new(num_feat: usize, num_grow_ch: usize, device: &Device) -> Self {
        let conv_in = |i: usize| num_feat + i * num_grow_ch;
        Self {
            conv1: conv3x3(conv_in(0), num_grow_ch, device),
            conv2: conv3x3(conv_in(1), num_grow_ch, device),
            conv3: conv3x3(conv_in(2), num_grow_ch, device),
            conv4: conv3x3(conv_in(3), num_grow_ch, device),
            conv5: conv3x3(conv_in(4), num_feat, device),
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let lrelu = |t| leaky_relu(t, NEGATIVE_SLOPE);

        let x1 = lrelu(self.conv1.forward(x.clone()));
        let x2 = lrelu(
            self.conv2
                .forward(Tensor::cat(vec![x.clone(), x1.clone()], 1)),
        );
        let x3 = lrelu(
            self.conv3
                .forward(Tensor::cat(vec![x.clone(), x1.clone(), x2.clone()], 1)),
        );
        let x4 = lrelu(self.conv4.forward(Tensor::cat(
            vec![x.clone(), x1.clone(), x2.clone(), x3.clone()],
            1,
        )));
        let x5 = self
            .conv5
            .forward(Tensor::cat(vec![x.clone(), x1, x2, x3, x4], 1));

        x + x5 * RESIDUAL_SCALE
    }
}

#[cfg(all(test, feature = "flex"))]
mod tests {
    use super::*;

    fn tiny(scale: usize) -> RrdbNetConfig {
        RrdbNetConfig::new(scale)
            .with_num_feat(8)
            .with_num_block(1)
            .with_num_grow_ch(4)
    }

    #[test]
    fn output_shape_matches_scale() {
        let device = Device::flex();
        for scale in [1, 2, 4] {
            let model = tiny(scale).init(&device);
            let y = model.forward(Tensor::zeros([1, 3, 8, 12], &device));
            assert_eq!(y.dims(), [1, 3, 8 * scale, 12 * scale], "scale {scale}");
        }
    }

    #[test]
    fn input_multiple_follows_unshuffle() {
        let device = Device::flex();
        let multiples: Vec<_> = [1, 2, 4]
            .map(|scale| tiny(scale).init(&device).input_multiple())
            .into();
        assert_eq!(multiples, [4, 2, 1]);
    }

    #[test]
    #[should_panic(expected = "RRDBNet supports scale 1, 2 or 4")]
    fn unsupported_scale_panics() {
        tiny(3).init(&Device::flex());
    }
}
