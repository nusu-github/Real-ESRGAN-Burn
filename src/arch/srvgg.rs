use burn::module::Ignored;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::interpolate::{Interpolate2dConfig, InterpolateMode};
use burn::nn::{PRelu, PReluConfig, PaddingConfig2d};
use burn::prelude::*;

fn conv3x3<B: Backend>(channels_in: usize, channels_out: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([channels_in, channels_out], [3, 3])
        .with_padding(PaddingConfig2d::Explicit(1, 1))
        .init(device)
}

pub fn pixel_shuffle<B: Backend>(x: Tensor<B, 4>, scale: usize) -> Tensor<B, 4> {
    let [n, c, h, w] = x.dims();
    let denom = scale * scale;
    assert!(c % denom == 0, "channels must be divisible by scale^2");
    let out_c = c / denom;

    x.reshape([n, out_c, scale, scale, h, w])
        .permute([0, 1, 4, 2, 5, 3])
        .reshape([n, out_c, h * scale, w * scale])
}

fn nearest_upsample<B: Backend>(x: Tensor<B, 4>, scale: usize) -> Tensor<B, 4> {
    Interpolate2dConfig::new()
        .with_mode(InterpolateMode::Nearest)
        .with_scale_factor(Some([scale as f32, scale as f32]))
        .init()
        .forward(x)
}

#[derive(Module, Debug)]
pub struct SrvggNetCompact<B: Backend> {
    conv_first: Conv2d<B>,
    acts: Vec<PRelu<B>>,
    convs: Vec<Conv2d<B>>,
    conv_last: Conv2d<B>,
    upscale: Ignored<usize>,
}

impl<B: Backend> SrvggNetCompact<B> {
    pub fn new(
        num_in_ch: usize,
        num_out_ch: usize,
        num_feat: usize,
        num_conv: usize,
        upscale: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            conv_first: conv3x3(num_in_ch, num_feat, device),
            acts: (0..(num_conv + 1))
                .map(|_| {
                    PReluConfig::new()
                        .with_num_parameters(num_feat)
                        .init(device)
                })
                .collect(),
            convs: (0..num_conv)
                .map(|_| conv3x3(num_feat, num_feat, device))
                .collect(),
            conv_last: conv3x3(num_feat, num_out_ch * upscale * upscale, device),
            upscale: Ignored(upscale),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut out = self.conv_first.forward(x.clone());
        out = self.acts[0].forward(out);

        for (i, conv) in self.convs.iter().enumerate() {
            out = conv.forward(out);
            out = self.acts[i + 1].forward(out);
        }

        out = self.conv_last.forward(out);
        out = pixel_shuffle(out, self.upscale.0);
        out + nearest_upsample(x, self.upscale.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn::backend::NdArray;

    #[test]
    fn pixel_shuffle_scale_4_shape() {
        let device = Default::default();
        let x = Tensor::<B, 4>::zeros([1, 48, 8, 8], &device);
        let y = pixel_shuffle(x, 4);
        assert_eq!(y.dims(), [1, 3, 32, 32]);
    }

    #[test]
    fn srvgg_output_shape() {
        let device = Default::default();
        let model = SrvggNetCompact::<B>::new(3, 3, 64, 2, 4, &device);
        let x = Tensor::<B, 4>::zeros([1, 3, 12, 10], &device);
        let y = model.forward(x);
        assert_eq!(y.dims(), [1, 3, 48, 40]);
    }
}
