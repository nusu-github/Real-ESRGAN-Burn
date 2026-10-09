//! Upscale one image with one model on the CPU.
//!
//! ```sh
//! cargo run --release --example upscale -- RealESRGAN_x4plus models/RealESRGAN_x4plus.pth in.png out.png
//! ```

use anyhow::{Context, Result, bail};
use burn::prelude::*;

use real_esrgan_burn::infer::{Tiling, load_rgb_image, upscale};
use real_esrgan_burn::loader::load_pytorch_weights;
use real_esrgan_burn::model_variant::{Architecture, ModelVariant};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model, weights, input, output] = args.as_slice() else {
        bail!("usage: upscale <model name> <weights.pth> <input image> <output image>");
    };

    let device = Device::flex();
    let variant: ModelVariant = model.parse()?;
    let Architecture::Generator(config) = variant.architecture() else {
        bail!("{variant} is a discriminator");
    };

    let mut model = config.init(&device);
    load_pytorch_weights(&mut model, weights.as_ref(), variant.top_level_keys())?;

    let image = load_rgb_image(input.as_ref(), None)?;
    let tiling = Tiling {
        tile_size: 128,
        tile_pad: 10,
    };
    upscale(&model, &image, tiling, &device)
        .save(output)
        .with_context(|| format!("failed to save {output}"))?;
    Ok(())
}
