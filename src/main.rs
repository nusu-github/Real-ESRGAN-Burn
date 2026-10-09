use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use burn::prelude::*;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use burn::tensor::DeviceType;
use clap::{Parser, ValueEnum};
use image::RgbImage;

use real_esrgan_burn::infer::{Tiling, load_rgb_image, save_rgb_image, upscale};
use real_esrgan_burn::loader::load_pytorch_weights;
use real_esrgan_burn::model_variant::{Architecture, ModelVariant};
use real_esrgan_burn::report::{ModelRunEntry, RunReport};

#[cfg(not(any(feature = "flex", feature = "cuda", feature = "wgpu")))]
compile_error!("enable at least one backend feature: `flex`, `cuda` or `wgpu`");

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceChoice {
    /// The first available of: CUDA, a hardware wgpu adapter, the CPU.
    Auto,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "wgpu")]
    Wgpu,
    /// The pure-Rust CPU backend.
    #[cfg(feature = "flex")]
    Flex,
}

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Upscale an image with the official Real-ESRGAN models"
)]
struct Cli {
    /// Directory containing the official `.pth` weight files.
    #[arg(long, default_value = "models")]
    models_dir: PathBuf,

    #[arg(long, default_value = "assets/test.jpg")]
    input: PathBuf,

    /// Each model writes `<output-dir>/<model>/<input stem>.png`.
    #[arg(long, default_value = "outputs")]
    output_dir: PathBuf,

    /// Models to run, by official name (repeatable). Defaults to all of them.
    #[arg(long = "model", value_name = "NAME")]
    models: Vec<ModelVariant>,

    #[arg(long, value_enum, default_value_t = DeviceChoice::Auto)]
    device: DeviceChoice,

    /// Shrink the input so that its longest edge is at most this many pixels.
    #[arg(long)]
    input_max_edge: Option<usize>,

    /// Tile size for tiled inference (0 disables tiling).
    #[arg(long, default_value_t = 128)]
    tile_size: usize,

    /// Context added around each tile.
    #[arg(long, default_value_t = 10)]
    tile_pad: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = select_device(cli.device)?;
    println!("Using device: {device:?}");

    let image = load_rgb_image(&cli.input, cli.input_max_edge)?;
    let job = Job {
        models_dir: &cli.models_dir,
        output_name: output_file_name(&cli.input),
        output_dir: &cli.output_dir,
        image: &image,
        tiling: Tiling {
            tile_size: cli.tile_size,
            tile_pad: cli.tile_pad,
        },
        device: &device,
    };

    let variants = if cli.models.is_empty() {
        ModelVariant::ALL.to_vec()
    } else {
        cli.models
    };
    let entries = variants
        .into_iter()
        .map(|variant| {
            println!("{variant}...");
            job.run(variant)
        })
        .collect();

    let report = RunReport { entries };
    let report_path = cli.output_dir.join("run_report.json");
    report.write(&report_path)?;

    match report.failures() {
        0 => {
            println!("All models succeeded. Report: {}", report_path.display());
            Ok(())
        }
        failures => bail!(
            "{failures} model(s) failed; see report at {}",
            report_path.display()
        ),
    }
}

struct Job<'a> {
    models_dir: &'a Path,
    output_dir: &'a Path,
    output_name: PathBuf,
    image: &'a RgbImage,
    tiling: Tiling,
    device: &'a Device,
}

impl Job<'_> {
    fn run(&self, variant: ModelVariant) -> ModelRunEntry {
        let mut entry = ModelRunEntry::new(variant);
        if let Err(err) = self.try_run(variant, &mut entry) {
            eprintln!("{variant}: {err:#}");
            entry.fail(&err);
        }
        entry
    }

    fn try_run(&self, variant: ModelVariant, entry: &mut ModelRunEntry) -> Result<()> {
        let weights = self.models_dir.join(variant.file_name());
        ensure!(
            weights.exists(),
            "missing weight file: {}",
            weights.display()
        );
        let keys = variant.top_level_keys();

        match variant.architecture() {
            Architecture::Generator(config) => {
                let mut model = config.init(self.device);
                entry.top_level_key = Some(load_pytorch_weights(&mut model, &weights, keys)?);

                entry.inference_executed = true;
                let output = upscale(&model, self.image, self.tiling, self.device);
                let path = self.output_dir.join(variant.name()).join(&self.output_name);
                save_rgb_image(&output, &path)?;
                entry.output_path = Some(path.display().to_string());
            }
            // The discriminator scores images rather than producing them; check that it loads.
            Architecture::Discriminator(config) => {
                let mut model = config.init(self.device);
                entry.top_level_key = Some(load_pytorch_weights(&mut model, &weights, keys)?);
            }
        }
        Ok(())
    }
}

fn output_file_name(input: &Path) -> PathBuf {
    let stem = input.file_stem().unwrap_or("output".as_ref());
    Path::new(stem).with_extension("png")
}

fn select_device(choice: DeviceChoice) -> Result<Device> {
    match choice {
        DeviceChoice::Auto => auto_device().context("no usable device found"),
        // `Device::cuda(0)` never fails; a missing GPU would only surface on first use.
        #[cfg(feature = "cuda")]
        DeviceChoice::Cuda => first_cuda_device().context("no CUDA device found"),
        #[cfg(feature = "wgpu")]
        DeviceChoice::Wgpu => Ok(Device::wgpu(DeviceKind::DefaultDevice)),
        #[cfg(feature = "flex")]
        DeviceChoice::Flex => Ok(Device::flex()),
    }
}

#[cfg(feature = "cuda")]
fn first_cuda_device() -> Option<Device> {
    Device::enumerate(DeviceType::Cuda).into_iter().next()
}

/// Asks the machine rather than relying on `Device::default()`, which only looks at the
/// enabled features.
fn auto_device() -> Option<Device> {
    #[cfg(feature = "cuda")]
    if let Some(device) = first_cuda_device() {
        return Some(device);
    }

    // Skip software adapters such as llvmpipe, which are slower than Flex.
    #[cfg(feature = "wgpu")]
    if let Some(device) = Device::enumerate(DeviceType::Wgpu)
        .into_iter()
        .find(|device| device.identity().is_some_and(|id| id.physical.is_some()))
    {
        return Some(device);
    }

    cpu_device()
}

#[cfg(feature = "flex")]
fn cpu_device() -> Option<Device> {
    Some(Device::flex())
}

#[cfg(not(feature = "flex"))]
fn cpu_device() -> Option<Device> {
    None
}
