#![recursion_limit = "256"]

use std::path::PathBuf;

use anyhow::{Result, bail};
use burn::prelude::*;
use clap::{Parser, ValueEnum};

use real_esrgan_burn::arch::discriminator::UnetDiscriminatorSn;
use real_esrgan_burn::arch::rrdb::RrdbNet;
use real_esrgan_burn::arch::srvgg::SrvggNetCompact;
use real_esrgan_burn::infer::{GeneratorModel, run_generator_inference};
use real_esrgan_burn::loader::{
    load_model_from_pytorch, reserialize_pytorch_checkpoint_with_python, srvgg_key_remaps,
};
use real_esrgan_burn::model_variant::{Architecture, ModelVariant};
use real_esrgan_burn::report::{ModelRunEntry, RunReport, write_report};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendChoice {
    Ndarray,
    Cuda,
}

#[derive(Debug, Parser)]
#[command(name = "real-esrgan-burn")]
#[command(about = "Load all official Real-ESRGAN variants and run inference")]
struct Cli {
    #[arg(long, default_value = "models")]
    models_dir: PathBuf,

    #[arg(long, default_value = "assets/test.jpg")]
    input: PathBuf,

    #[arg(long, default_value = "outputs")]
    output_dir: PathBuf,

    #[arg(long)]
    input_max_edge: Option<usize>,

    #[arg(long, default_value_t = 128)]
    tile_size: usize,

    #[arg(long, default_value_t = 10)]
    tile_pad: usize,

    #[arg(long, value_enum, default_value_t = BackendChoice::Cuda)]
    backend: BackendChoice,

    #[arg(long)]
    run_all: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if !cli.run_all {
        bail!("this tool only supports --run-all")
    }

    match cli.backend {
        BackendChoice::Ndarray => run_all::<burn::backend::NdArray>(&cli),
        BackendChoice::Cuda => run_all::<burn::backend::Cuda>(&cli),
    }
}

fn run_all<B: Backend>(cli: &Cli) -> Result<()> {
    let device = B::Device::default();
    println!("Using backend: {}", B::name(&device));

    std::fs::create_dir_all(&cli.output_dir)?;

    let mut entries = Vec::new();
    let mut failures = 0usize;

    for variant in ModelVariant::all() {
        let weight_path = cli.models_dir.join(variant.file_name());

        if !weight_path.exists() {
            failures += 1;
            entries.push(ModelRunEntry {
                model_name: variant.name().to_string(),
                weight_file: variant.file_name().to_string(),
                top_level_key: None,
                status: "failed".to_string(),
                inference_executed: false,
                output_path: None,
                error: Some(format!("missing weight file: {:?}", weight_path)),
            });
            continue;
        }

        let load_weight_path = if matches!(variant, ModelVariant::EsrganSrX4Df2kostOfficial) {
            reserialize_pytorch_checkpoint_with_python(&weight_path)?
        } else {
            weight_path.clone()
        };

        match variant.architecture() {
            Architecture::Rrdb { scale, num_block } => {
                let remaps = Vec::new();
                let loaded = load_model_from_pytorch::<B, _, _>(
                    &load_weight_path,
                    variant.top_level_keys(),
                    &remaps,
                    &device,
                    || RrdbNet::new(3, 3, scale, num_block, 64, 32, &device),
                );

                match loaded {
                    Ok((model, loaded_key)) => {
                        let generator = GeneratorModel::Rrdb(model);
                        let output_path = cli
                            .output_dir
                            .join(variant.output_slug())
                            .join("test_out.png");

                        match run_generator_inference(
                            &generator,
                            scale,
                            &cli.input,
                            &output_path,
                            cli.input_max_edge,
                            cli.tile_size,
                            cli.tile_pad,
                            &device,
                        ) {
                            Ok(()) => entries.push(ModelRunEntry {
                                model_name: variant.name().to_string(),
                                weight_file: variant.file_name().to_string(),
                                top_level_key: Some(loaded_key),
                                status: "success".to_string(),
                                inference_executed: true,
                                output_path: Some(output_path.to_string_lossy().to_string()),
                                error: None,
                            }),
                            Err(err) => {
                                failures += 1;
                                entries.push(ModelRunEntry {
                                    model_name: variant.name().to_string(),
                                    weight_file: variant.file_name().to_string(),
                                    top_level_key: Some(loaded_key),
                                    status: "failed".to_string(),
                                    inference_executed: true,
                                    output_path: None,
                                    error: Some(err.to_string()),
                                });
                            }
                        }
                    }
                    Err(err) => {
                        failures += 1;
                        entries.push(ModelRunEntry {
                            model_name: variant.name().to_string(),
                            weight_file: variant.file_name().to_string(),
                            top_level_key: None,
                            status: "failed".to_string(),
                            inference_executed: false,
                            output_path: None,
                            error: Some(err.to_string()),
                        });
                    }
                }
            }
            Architecture::Srvgg { scale, num_conv } => {
                let remaps = srvgg_key_remaps(num_conv);
                let loaded = load_model_from_pytorch::<B, _, _>(
                    &load_weight_path,
                    variant.top_level_keys(),
                    &remaps,
                    &device,
                    || SrvggNetCompact::new(3, 3, 64, num_conv, 4, &device),
                );

                match loaded {
                    Ok((model, loaded_key)) => {
                        let generator = GeneratorModel::Srvgg(model);
                        let output_path = cli
                            .output_dir
                            .join(variant.output_slug())
                            .join("test_out.png");

                        match run_generator_inference(
                            &generator,
                            scale,
                            &cli.input,
                            &output_path,
                            cli.input_max_edge,
                            cli.tile_size,
                            cli.tile_pad,
                            &device,
                        ) {
                            Ok(()) => entries.push(ModelRunEntry {
                                model_name: variant.name().to_string(),
                                weight_file: variant.file_name().to_string(),
                                top_level_key: Some(loaded_key),
                                status: "success".to_string(),
                                inference_executed: true,
                                output_path: Some(output_path.to_string_lossy().to_string()),
                                error: None,
                            }),
                            Err(err) => {
                                failures += 1;
                                entries.push(ModelRunEntry {
                                    model_name: variant.name().to_string(),
                                    weight_file: variant.file_name().to_string(),
                                    top_level_key: Some(loaded_key),
                                    status: "failed".to_string(),
                                    inference_executed: true,
                                    output_path: None,
                                    error: Some(err.to_string()),
                                });
                            }
                        }
                    }
                    Err(err) => {
                        failures += 1;
                        entries.push(ModelRunEntry {
                            model_name: variant.name().to_string(),
                            weight_file: variant.file_name().to_string(),
                            top_level_key: None,
                            status: "failed".to_string(),
                            inference_executed: false,
                            output_path: None,
                            error: Some(err.to_string()),
                        });
                    }
                }
            }
            Architecture::Discriminator => {
                let remaps = Vec::new();
                let loaded = load_model_from_pytorch::<B, _, _>(
                    &load_weight_path,
                    variant.top_level_keys(),
                    &remaps,
                    &device,
                    || UnetDiscriminatorSn::new(3, 64, true, &device),
                );

                match loaded {
                    Ok((_model, loaded_key)) => {
                        entries.push(ModelRunEntry {
                            model_name: variant.name().to_string(),
                            weight_file: variant.file_name().to_string(),
                            top_level_key: Some(loaded_key),
                            status: "success".to_string(),
                            inference_executed: false,
                            output_path: None,
                            error: None,
                        });
                    }
                    Err(err) => {
                        failures += 1;
                        entries.push(ModelRunEntry {
                            model_name: variant.name().to_string(),
                            weight_file: variant.file_name().to_string(),
                            top_level_key: None,
                            status: "failed".to_string(),
                            inference_executed: false,
                            output_path: None,
                            error: Some(err.to_string()),
                        });
                    }
                }
            }
        }
    }

    let report = RunReport { entries };
    let report_path = cli.output_dir.join("run_report.json");
    write_report(&report_path, &report)?;

    if failures > 0 {
        bail!(
            "{} model(s) failed; see report at {:?}",
            failures,
            report_path
        );
    }

    println!("All models succeeded. Report: {:?}", report_path);
    Ok(())
}
