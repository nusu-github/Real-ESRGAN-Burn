use burn::module::Ignored;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::interpolate::{Interpolate2dConfig, InterpolateMode};
use burn::nn::{LeakyRelu, LeakyReluConfig, PaddingConfig2d};
use burn::prelude::*;

fn conv3x3<B: Backend>(channels_in: usize, channels_out: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([channels_in, channels_out], [3, 3])
        .with_padding(PaddingConfig2d::Explicit(1, 1))
        .init(device)
}

fn nearest_upsample_2x<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    Interpolate2dConfig::new()
        .with_mode(InterpolateMode::Nearest)
        .with_scale_factor(Some([2.0, 2.0]))
        .init()
        .forward(x)
}

pub fn pixel_unshuffle<B: Backend>(x: Tensor<B, 4>, scale: usize) -> Tensor<B, 4> {
    let [b, c, hh, hw] = x.dims();
    assert!(hh % scale == 0, "height must be divisible by scale");
    assert!(hw % scale == 0, "width must be divisible by scale");

    let h = hh / scale;
    let w = hw / scale;
    let out_c = c * scale * scale;

    x.reshape([b, c, h, scale, w, scale])
        .permute([0, 1, 3, 5, 2, 4])
        .reshape([b, out_c, h, w])
}

#[derive(Module, Debug)]
pub struct ResidualDenseBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    conv5: Conv2d<B>,
    lrelu: Ignored<LeakyRelu>,
}

impl<B: Backend> ResidualDenseBlock<B> {
    pub fn new(num_feat: usize, num_grow_ch: usize, device: &B::Device) -> Self {
        let lrelu = LeakyReluConfig::new().with_negative_slope(0.2).init();

        Self {
            conv1: conv3x3(num_feat, num_grow_ch, device),
            conv2: conv3x3(num_feat + num_grow_ch, num_grow_ch, device),
            conv3: conv3x3(num_feat + 2 * num_grow_ch, num_grow_ch, device),
            conv4: conv3x3(num_feat + 3 * num_grow_ch, num_grow_ch, device),
            conv5: conv3x3(num_feat + 4 * num_grow_ch, num_feat, device),
            lrelu: Ignored(lrelu),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x1 = self.lrelu.0.forward(self.conv1.forward(x.clone()));
        let x2 = self.lrelu.0.forward(
            self.conv2
                .forward(Tensor::cat(vec![x.clone(), x1.clone()], 1)),
        );
        let x3 = self.lrelu.0.forward(
            self.conv3
                .forward(Tensor::cat(vec![x.clone(), x1.clone(), x2.clone()], 1)),
        );
        let x4 = self.lrelu.0.forward(self.conv4.forward(Tensor::cat(
            vec![x.clone(), x1.clone(), x2.clone(), x3.clone()],
            1,
        )));
        let x5 = self
            .conv5
            .forward(Tensor::cat(vec![x.clone(), x1, x2, x3, x4], 1));

        x + x5 * 0.2
    }
}

#[derive(Module, Debug)]
pub struct Rrdb<B: Backend> {
    rdb1: ResidualDenseBlock<B>,
    rdb2: ResidualDenseBlock<B>,
    rdb3: ResidualDenseBlock<B>,
}

impl<B: Backend> Rrdb<B> {
    pub fn new(num_feat: usize, num_grow_ch: usize, device: &B::Device) -> Self {
        Self {
            rdb1: ResidualDenseBlock::new(num_feat, num_grow_ch, device),
            rdb2: ResidualDenseBlock::new(num_feat, num_grow_ch, device),
            rdb3: ResidualDenseBlock::new(num_feat, num_grow_ch, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let out = self
            .rdb3
            .forward(self.rdb2.forward(self.rdb1.forward(x.clone())));
        x + out * 0.2
    }
}

#[derive(Module, Debug)]
pub struct RrdbNet<B: Backend> {
    conv_first: Conv2d<B>,
    body: Vec<Rrdb<B>>,
    conv_body: Conv2d<B>,
    conv_up1: Conv2d<B>,
    conv_up2: Conv2d<B>,
    conv_hr: Conv2d<B>,
    conv_last: Conv2d<B>,
    scale: Ignored<usize>,
    lrelu: Ignored<LeakyRelu>,
}

impl<B: Backend> RrdbNet<B> {
    pub fn new(
        num_in_ch: usize,
        num_out_ch: usize,
        scale: usize,
        num_block: usize,
        num_feat: usize,
        num_grow_ch: usize,
        device: &B::Device,
    ) -> Self {
        let adjusted_in_ch = match scale {
            2 => num_in_ch * 4,
            1 => num_in_ch * 16,
            _ => num_in_ch,
        };
        let lrelu = LeakyReluConfig::new().with_negative_slope(0.2).init();

        Self {
            conv_first: conv3x3(adjusted_in_ch, num_feat, device),
            body: (0..num_block)
                .map(|_| Rrdb::new(num_feat, num_grow_ch, device))
                .collect(),
            conv_body: conv3x3(num_feat, num_feat, device),
            conv_up1: conv3x3(num_feat, num_feat, device),
            conv_up2: conv3x3(num_feat, num_feat, device),
            conv_hr: conv3x3(num_feat, num_feat, device),
            conv_last: conv3x3(num_feat, num_out_ch, device),
            scale: Ignored(scale),
            lrelu: Ignored(lrelu),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut feat = match self.scale.0 {
            2 => pixel_unshuffle(x, 2),
            1 => pixel_unshuffle(x, 4),
            _ => x,
        };

        feat = self.conv_first.forward(feat);

        let mut body_feat = feat.clone();
        for block in &self.body {
            body_feat = block.forward(body_feat);
        }
        body_feat = self.conv_body.forward(body_feat);

        feat = feat + body_feat;
        feat = self
            .lrelu
            .0
            .forward(self.conv_up1.forward(nearest_upsample_2x(feat)));
        feat = self
            .lrelu
            .0
            .forward(self.conv_up2.forward(nearest_upsample_2x(feat)));

        self.conv_last
            .forward(self.lrelu.0.forward(self.conv_hr.forward(feat)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn::backend::NdArray;

    #[test]
    fn pixel_unshuffle_scale_2_shape() {
        let device = Default::default();
        let x = Tensor::<B, 4>::zeros([1, 3, 8, 8], &device);
        let y = pixel_unshuffle(x, 2);
        assert_eq!(y.dims(), [1, 12, 4, 4]);
    }

    #[test]
    fn rrdb_x2_output_shape() {
        let device = Default::default();
        let model = RrdbNet::<B>::new(3, 3, 2, 2, 64, 32, &device);
        let x = Tensor::<B, 4>::zeros([1, 3, 8, 8], &device);
        let y = model.forward(x);
        assert_eq!(y.dims(), [1, 3, 16, 16]);
    }
}
