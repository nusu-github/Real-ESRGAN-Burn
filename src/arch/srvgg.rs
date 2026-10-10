//! SRVGGNetCompact, the compact VGG-style generator of `realesr-*v3` models
//! (`realesrgan.archs.srvgg_arch.SRVGGNetCompact`, PReLU activations).
//!
//! `body` mirrors the PyTorch `nn.ModuleList` index by index, so official checkpoints load
//! without key remapping.

use burn::module::Param;
use burn::nn::conv::Conv2d;
use burn::nn::{Initializer, PixelShuffle, PixelShuffleConfig};
use burn::prelude::*;
use burn::tensor::activation::prelu;

use super::ops::{conv3x3, upsample_nearest};

#[derive(Config, Debug)]
pub struct SrvggNetCompactConfig {
    pub upscale: usize,
    #[config(default = 3)]
    pub num_in_ch: usize,
    #[config(default = 3)]
    pub num_out_ch: usize,
    #[config(default = 64)]
    pub num_feat: usize,
    #[config(default = 16)]
    pub num_conv: usize,
}

impl SrvggNetCompactConfig {
    pub fn init(&self, device: &Device) -> SrvggNetCompact {
        let feat = self.num_feat;
        let act = || Layer::Act(TorchPRelu::new(feat, device));

        let mut body = vec![Layer::Conv(conv3x3(self.num_in_ch, feat, device)), act()];
        for _ in 0..self.num_conv {
            body.push(Layer::Conv(conv3x3(feat, feat, device)));
            body.push(act());
        }
        let out_ch = self.num_out_ch * self.upscale * self.upscale;
        body.push(Layer::Conv(conv3x3(feat, out_ch, device)));

        SrvggNetCompact {
            body,
            upsampler: PixelShuffleConfig::new(self.upscale).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct SrvggNetCompact {
    body: Vec<Layer>,
    upsampler: PixelShuffle,
}

impl SrvggNetCompact {
    pub fn scale(&self) -> usize {
        self.upsampler.upscale_factor
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let out = self
            .body
            .iter()
            .fold(x.clone(), |h, layer| layer.forward(h));
        // The network learns the residual over a nearest-neighbour upsampling.
        self.upsampler.forward(out) + upsample_nearest(x, self.scale())
    }
}

/// One entry of `body`. Variant names are not part of PyTorch keys and are skipped on load.
#[derive(Module, Debug)]
pub enum Layer {
    Conv(Conv2d),
    Act(TorchPRelu),
}

impl Layer {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        match self {
            Self::Conv(conv) => conv.forward(x),
            Self::Act(act) => act.forward(x),
        }
    }
}

/// `nn.PReLU(num_parameters)`. Unlike `burn::nn::PRelu` (`alpha`), the parameter is named
/// `weight` like in PyTorch.
#[derive(Module, Debug)]
pub struct TorchPRelu {
    weight: Param<Tensor<1>>,
}

impl TorchPRelu {
    fn new(num_parameters: usize, device: &Device) -> Self {
        Self {
            weight: Initializer::Constant { value: 0.25 }.init([num_parameters], device),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        prelu(x, self.weight.val())
    }
}

#[cfg(all(test, feature = "flex"))]
mod tests {
    use super::*;

    #[test]
    fn output_shape_matches_scale() {
        let device = Device::flex();
        let model = SrvggNetCompactConfig::new(4)
            .with_num_feat(8)
            .with_num_conv(2)
            .init(&device);
        let y = model.forward(Tensor::zeros([1, 3, 12, 10], &device));
        assert_eq!(y.dims(), [1, 3, 48, 40]);
    }

    #[test]
    fn body_layout_matches_pytorch_module_list() {
        let model = SrvggNetCompactConfig::new(4)
            .with_num_conv(2)
            .init(&Device::flex());
        let kinds: String = model
            .body
            .iter()
            .map(|layer| match layer {
                Layer::Conv(_) => 'c',
                Layer::Act(_) => 'a',
            })
            .collect();
        assert_eq!(kinds, "cacacac");
    }
}
