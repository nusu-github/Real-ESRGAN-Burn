use burn::module::{Ignored, Param};
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::interpolate::{Interpolate2dConfig, InterpolateMode};
use burn::prelude::*;
use burn::tensor::activation::leaky_relu;
use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;

#[derive(Module, Debug)]
pub struct SpectralConv2d<B: Backend> {
    weight_orig: Param<Tensor<B, 4>>,
    bias: Option<Param<Tensor<B, 1>>>,
    stride: Ignored<[usize; 2]>,
    padding: Ignored<[usize; 2]>,
}

impl<B: Backend> SpectralConv2d<B> {
    pub fn new(
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
        device: &B::Device,
    ) -> Self {
        let weight_shape = [channels_out, channels_in, kernel_size, kernel_size];
        let weight_orig = Param::from_tensor(Tensor::<B, 4>::zeros(weight_shape, device));
        let bias = if bias {
            Some(Param::from_tensor(Tensor::<B, 1>::zeros(
                [channels_out],
                device,
            )))
        } else {
            None
        };

        Self {
            weight_orig,
            bias,
            stride: Ignored([stride, stride]),
            padding: Ignored([padding, padding]),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        conv2d(
            x,
            self.weight_orig.val(),
            self.bias.as_ref().map(|v| v.val()),
            ConvOptions::new(self.stride.0, self.padding.0, [1, 1], 1),
        )
    }
}

fn conv2d_regular<B: Backend>(
    channels_in: usize,
    channels_out: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    bias: bool,
    device: &B::Device,
) -> Conv2d<B> {
    Conv2dConfig::new([channels_in, channels_out], [kernel_size, kernel_size])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(padding, padding))
        .with_bias(bias)
        .init(device)
}

fn upsample_bilinear_2x<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    Interpolate2dConfig::new()
        .with_mode(InterpolateMode::Linear)
        .with_scale_factor(Some([2.0, 2.0]))
        .init()
        .forward(x)
}

#[derive(Module, Debug)]
pub struct UnetDiscriminatorSn<B: Backend> {
    conv0: Conv2d<B>,
    conv1: SpectralConv2d<B>,
    conv2: SpectralConv2d<B>,
    conv3: SpectralConv2d<B>,
    conv4: SpectralConv2d<B>,
    conv5: SpectralConv2d<B>,
    conv6: SpectralConv2d<B>,
    conv7: SpectralConv2d<B>,
    conv8: SpectralConv2d<B>,
    conv9: Conv2d<B>,
    skip_connection: Ignored<bool>,
}

impl<B: Backend> UnetDiscriminatorSn<B> {
    pub fn new(
        num_in_ch: usize,
        num_feat: usize,
        skip_connection: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            conv0: conv2d_regular(num_in_ch, num_feat, 3, 1, 1, true, device),
            conv1: SpectralConv2d::new(num_feat, num_feat * 2, 4, 2, 1, false, device),
            conv2: SpectralConv2d::new(num_feat * 2, num_feat * 4, 4, 2, 1, false, device),
            conv3: SpectralConv2d::new(num_feat * 4, num_feat * 8, 4, 2, 1, false, device),
            conv4: SpectralConv2d::new(num_feat * 8, num_feat * 4, 3, 1, 1, false, device),
            conv5: SpectralConv2d::new(num_feat * 4, num_feat * 2, 3, 1, 1, false, device),
            conv6: SpectralConv2d::new(num_feat * 2, num_feat, 3, 1, 1, false, device),
            conv7: SpectralConv2d::new(num_feat, num_feat, 3, 1, 1, false, device),
            conv8: SpectralConv2d::new(num_feat, num_feat, 3, 1, 1, false, device),
            conv9: conv2d_regular(num_feat, 1, 3, 1, 1, true, device),
            skip_connection: Ignored(skip_connection),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x0 = leaky_relu(self.conv0.forward(x), 0.2);
        let x1 = leaky_relu(self.conv1.forward(x0.clone()), 0.2);
        let x2 = leaky_relu(self.conv2.forward(x1.clone()), 0.2);
        let x3 = leaky_relu(self.conv3.forward(x2.clone()), 0.2);

        let x3 = upsample_bilinear_2x(x3);
        let mut x4 = leaky_relu(self.conv4.forward(x3), 0.2);
        if self.skip_connection.0 {
            x4 = x4 + x2;
        }

        let x4 = upsample_bilinear_2x(x4);
        let mut x5 = leaky_relu(self.conv5.forward(x4), 0.2);
        if self.skip_connection.0 {
            x5 = x5 + x1;
        }

        let x5 = upsample_bilinear_2x(x5);
        let mut x6 = leaky_relu(self.conv6.forward(x5), 0.2);
        if self.skip_connection.0 {
            x6 = x6 + x0;
        }

        let out = leaky_relu(self.conv7.forward(x6), 0.2);
        let out = leaky_relu(self.conv8.forward(out), 0.2);
        self.conv9.forward(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn::backend::NdArray;

    #[test]
    fn unet_discriminator_shape() {
        let device = Default::default();
        let model = UnetDiscriminatorSn::<B>::new(3, 4, true, &device);
        let x = Tensor::<B, 4>::zeros([1, 3, 32, 32], &device);
        let y = model.forward(x);
        assert_eq!(y.dims(), [1, 1, 32, 32]);
    }
}
