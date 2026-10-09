//! UNetDiscriminatorSN, the Real-ESRGAN discriminator
//! (`basicsr.archs.discriminator_arch.UNetDiscriminatorSN`).

use burn::module::Param;
use burn::nn::Initializer;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::*;
use burn::tensor::activation::leaky_relu;
use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;

use super::ops::{conv3x3, upsample_bilinear};

const NEGATIVE_SLOPE: f64 = 0.2;

#[derive(Config, Debug)]
pub struct UnetDiscriminatorSnConfig {
    #[config(default = 3)]
    pub num_in_ch: usize,
    #[config(default = 64)]
    pub num_feat: usize,
    #[config(default = true)]
    pub skip_connection: bool,
}

impl UnetDiscriminatorSnConfig {
    pub fn init(&self, device: &Device) -> UnetDiscriminatorSn {
        let f = self.num_feat;
        let down = |ch_in, ch_out| SpectralNormConv2dConfig::new([ch_in, ch_out], 4).with_stride(2);
        let same = |ch_in, ch_out| SpectralNormConv2dConfig::new([ch_in, ch_out], 3);

        UnetDiscriminatorSn {
            conv0: conv3x3(self.num_in_ch, f, device),
            conv1: down(f, f * 2).init(device),
            conv2: down(f * 2, f * 4).init(device),
            conv3: down(f * 4, f * 8).init(device),
            conv4: same(f * 8, f * 4).init(device),
            conv5: same(f * 4, f * 2).init(device),
            conv6: same(f * 2, f).init(device),
            conv7: same(f, f).init(device),
            conv8: same(f, f).init(device),
            conv9: conv3x3(f, 1, device),
            skip_connection: self.skip_connection,
        }
    }
}

#[derive(Module, Debug)]
pub struct UnetDiscriminatorSn {
    conv0: Conv2d,
    conv1: SpectralNormConv2d,
    conv2: SpectralNormConv2d,
    conv3: SpectralNormConv2d,
    conv4: SpectralNormConv2d,
    conv5: SpectralNormConv2d,
    conv6: SpectralNormConv2d,
    conv7: SpectralNormConv2d,
    conv8: SpectralNormConv2d,
    conv9: Conv2d,
    skip_connection: bool,
}

impl UnetDiscriminatorSn {
    /// Per-pixel realness logits, `[batch, 1, height, width]`.
    ///
    /// Height and width must be multiples of 8.
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let lrelu = |t| leaky_relu(t, NEGATIVE_SLOPE);
        let skip = |x: Tensor<4>, s: Tensor<4>| if self.skip_connection { x + s } else { x };

        // Downsample.
        let x0 = lrelu(self.conv0.forward(x));
        let x1 = lrelu(self.conv1.forward(x0.clone()));
        let x2 = lrelu(self.conv2.forward(x1.clone()));
        let x3 = lrelu(self.conv3.forward(x2.clone()));

        // Upsample.
        let x4 = skip(lrelu(self.conv4.forward(upsample_bilinear(x3, 2))), x2);
        let x5 = skip(lrelu(self.conv5.forward(upsample_bilinear(x4, 2))), x1);
        let x6 = skip(lrelu(self.conv6.forward(upsample_bilinear(x5, 2))), x0);

        // Extra convolutions.
        let out = lrelu(self.conv7.forward(x6));
        let out = lrelu(self.conv8.forward(out));
        self.conv9.forward(out)
    }
}

#[derive(Config, Debug)]
pub struct SpectralNormConv2dConfig {
    pub channels: [usize; 2],
    pub kernel_size: usize,
    #[config(default = 1)]
    pub stride: usize,
    #[config(default = 1)]
    pub padding: usize,
}

impl SpectralNormConv2dConfig {
    pub fn init(&self, device: &Device) -> SpectralNormConv2d {
        let [ch_in, ch_out] = self.channels;
        let k = self.kernel_size;
        let fan_in = ch_in * k * k;
        // Vectors of unit expected norm, like the normalized Gaussians PyTorch starts from.
        let unit = |n: usize| Initializer::Normal {
            mean: 0.0,
            std: 1.0 / (n as f64).sqrt(),
        };

        SpectralNormConv2d {
            // Same initialization as a plain convolution.
            weight_orig: Conv2dConfig::new([ch_in, ch_out], [k, k])
                .with_bias(false)
                .init(device)
                .weight,
            weight_u: unit(ch_out).init([ch_out], device),
            weight_v: unit(fan_in).init([fan_in], device),
            stride: self.stride,
            padding: self.padding,
        }
    }
}

/// `spectral_norm(nn.Conv2d(.., bias=False))` in evaluation mode.
///
/// The weight is divided by its largest singular value, estimated as `u · (W v)` from the
/// power-iteration vectors stored in the checkpoint. No power iteration runs here, which is
/// what PyTorch does in `eval()` mode.
#[derive(Module, Debug)]
pub struct SpectralNormConv2d {
    weight_orig: Param<Tensor<4>>,
    weight_u: Param<Tensor<1>>,
    weight_v: Param<Tensor<1>>,
    stride: usize,
    padding: usize,
}

impl SpectralNormConv2d {
    /// The normalized convolution kernel, `weight_orig / sigma`.
    pub fn weight(&self) -> Tensor<4> {
        let weight = self.weight_orig.val();
        let [ch_out, ch_in, kh, kw] = weight.dims();
        let fan_in = ch_in * kh * kw;

        let u = self.weight_u.val().reshape([1, ch_out]);
        let v = self.weight_v.val().reshape([fan_in, 1]);
        let sigma = u.matmul(weight.clone().reshape([ch_out, fan_in]).matmul(v));
        weight / sigma.reshape([1, 1, 1, 1])
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let options = ConvOptions::new([self.stride; 2], [self.padding; 2], [1, 1], 1);
        conv2d(x, self.weight(), None, options)
    }
}

#[cfg(all(test, feature = "flex"))]
mod tests {
    use super::*;

    #[test]
    fn output_is_a_full_resolution_map() {
        let device = Device::flex();
        let model = UnetDiscriminatorSnConfig::new()
            .with_num_feat(4)
            .init(&device);
        let y = model.forward(Tensor::zeros([1, 3, 32, 24], &device));
        assert_eq!(y.dims(), [1, 1, 32, 24]);
    }

    #[test]
    fn spectral_norm_divides_by_u_w_v() {
        let device = Device::flex();
        let mut conv = SpectralNormConv2dConfig::new([1, 2], 1).init(&device);
        // W = [[1], [3]] (2 x 1), u = [1, 0], v = [2] => sigma = u · (W v) = 2.
        conv.weight_orig = Param::from_tensor(Tensor::from_floats([[[[1.0]]], [[[3.0]]]], &device));
        conv.weight_u = Param::from_tensor(Tensor::from_floats([1.0, 0.0], &device));
        conv.weight_v = Param::from_tensor(Tensor::from_floats([2.0], &device));

        let expected = TensorData::from([[[[0.5f32]]], [[[1.5]]]]);
        conv.weight().into_data().assert_eq(&expected, false);
    }
}
