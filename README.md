# Real-ESRGAN-Burn

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) super-resolution inference implemented in Rust with the [Burn](https://burn.dev/) deep learning framework (0.22).

Loads the official PyTorch weight files directly and runs tiled inference to upscale images.
Outputs match the PyTorch reference implementation (checked in `tests/pytorch_parity.rs`).

## Supported Models

| Model                           | Architecture               | Scale |
| ------------------------------- | -------------------------- | ----- |
| RealESRGAN_x4plus               | RRDB (23 blocks)           | 4x    |
| RealESRGAN_x2plus               | RRDB (23 blocks)           | 2x    |
| RealESRNet_x4plus               | RRDB (23 blocks)           | 4x    |
| ESRGAN_SRx4_DF2KOST_official    | RRDB (23 blocks)           | 4x    |
| RealESRGAN_x4plus_anime_6B      | RRDB (6 blocks)            | 4x    |
| realesr-animevideov3            | SRVGGNetCompact (16 convs) | 4x    |
| realesr-general-x4v3            | SRVGGNetCompact (32 convs) | 4x    |
| RealESRGAN_x4plus_netD          | UNet Discriminator (SN)    | -     |
| RealESRGAN_x2plus_netD          | UNet Discriminator (SN)    | -     |
| RealESRGAN_x4plus_anime_6B_netD | UNet Discriminator (SN)    | -     |

Discriminators are only loaded (to check the weights), not run on the input.

## Prerequisites

- **Rust** 1.95 or newer
- **Model weight files** (`.pth`) from the [official Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases)
- For the `cuda` backend: an NVIDIA driver at run time, and network access on the first build
  (CubeCL downloads an LLVM bundle, ~45 MB, into `~/.cache/tracel` and `~/.local/share/tracel`)

Both checkpoint formats (zip, PyTorch >= 1.6, and the legacy pickle format of
`ESRGAN_SRx4_DF2KOST_official`) are read natively; Python is not needed.

## Backends

Backends are Cargo features of this crate. `flex` and `cuda` are enabled by default.

| Feature | Device                          | Notes                                   |
| ------- | ------------------------------- | --------------------------------------- |
| `flex`  | CPU (pure Rust)                 | No build or runtime requirements        |
| `cuda`  | NVIDIA GPUs                     | Downloads the LLVM bundle on first build |
| `wgpu`  | Any GPU (Vulkan, Metal, DX12)   | Needs a Vulkan/Metal/DX12 driver        |

```sh
# CPU only (fastest to build)
cargo build --release --no-default-features --features flex

# CPU + any GPU through wgpu
cargo build --release --no-default-features --features flex,wgpu
```

## Usage

```sh
mkdir -p models
# Download the weights you want from https://github.com/xinntao/Real-ESRGAN/releases

# Run every model (auto-selects the device); missing weight files are reported as failures
cargo run --release -- --input path/to/image.jpg

# Only some models, on the CPU
cargo run --release -- --input image.jpg --device flex \
    --model RealESRGAN_x4plus --model realesr-general-x4v3
```

Each model writes `<output-dir>/<model>/<input stem>.png`, and a summary is written to
`<output-dir>/run_report.json`.

### CLI Options

| Option             | Default           | Description                                                     |
| ------------------ | ----------------- | --------------------------------------------------------------- |
| `--input`          | `assets/test.jpg` | Input image path                                                |
| `--model`          | all               | Model to run, by official name (repeatable)                     |
| `--device`         | `auto`            | `auto`, `cuda`, `wgpu` or `flex` (depending on enabled features) |
| `--models-dir`     | `models`          | Directory containing `.pth` weight files                        |
| `--output-dir`     | `outputs`         | Output directory                                                |
| `--tile-size`      | `128`             | Tile size for tiled inference (0 = no tiling)                   |
| `--tile-pad`       | `10`              | Padding around each tile                                        |
| `--input-max-edge` | none              | Resize input so longest edge <= this value                      |

`--device auto` picks CUDA if a CUDA device is present, then a hardware wgpu adapter, then the CPU.

## Library

The crate is also a library: build a network from `ModelVariant::architecture()`, load the
official weights with `loader::load_pytorch_weights`, and run `infer::upscale` on any `Device`.
See [`examples/upscale.rs`](examples/upscale.rs):

```sh
cargo run --release --example upscale -- RealESRGAN_x4plus models/RealESRGAN_x4plus.pth in.png out.png
```

Library users who want to choose backends themselves can depend on the crate with
`default-features = false`.

## Development

```sh
cargo test                                 # unit tests and PyTorch parity tests (on Flex)
python tests/fixtures/gen_fixtures.py      # regenerate the parity fixtures (needs torch)
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Acknowledgments

This project is a Rust reimplementation of [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al., which is licensed under the BSD 3-Clause License.
