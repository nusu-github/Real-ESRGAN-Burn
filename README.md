# Real-ESRGAN-Burn

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) super-resolution inference implemented in Rust with the [Burn](https://burn.dev/) deep learning framework.

Loads official PyTorch weight files directly and runs tiled inference to upscale images.

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
| RealESRGAN_x4plus_netD          | UNet Discriminator         | -     |
| RealESRGAN_x2plus_netD          | UNet Discriminator         | -     |
| RealESRGAN_x4plus_anime_6B_netD | UNet Discriminator         | -     |

## Prerequisites

- **Rust** (edition 2024)
- **CUDA toolkit** (for GPU backend; NdArray backend works on CPU)
- **Python + PyTorch** (only needed for `ESRGAN_SRx4_DF2KOST_official` checkpoint conversion)
- **Model weight files** (`.pth`) from the [official Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases)

## Setup

1. Clone the repository:

```sh
git clone https://github.com/<your-username>/Real-ESRGAN-Burn.git
cd Real-ESRGAN-Burn
```

1. Download model weights into the `models/` directory:

```sh
mkdir -p models
# Download the weights you want from:
# https://github.com/xinntao/Real-ESRGAN/releases
```

1. Place a test image at `assets/test.jpg` (or specify `--input`).

## Usage

```sh
# Run all models with CUDA backend (default)
cargo run --release -- --run-all

# Use NdArray (CPU) backend
cargo run --release -- --run-all --backend ndarray

# Custom options
cargo run --release -- --run-all \
    --models-dir models \
    --input path/to/image.jpg \
    --output-dir outputs \
    --tile-size 128 \
    --tile-pad 10 \
    --input-max-edge 512
```

### CLI Options

| Option             | Default           | Description                                   |
| ------------------ | ----------------- | --------------------------------------------- |
| `--run-all`        | required          | Run inference with all available models       |
| `--backend`        | `cuda`            | Backend: `cuda` or `ndarray`                  |
| `--models-dir`     | `models`          | Directory containing `.pth` weight files      |
| `--input`          | `assets/test.jpg` | Input image path                              |
| `--output-dir`     | `outputs`         | Output directory                              |
| `--tile-size`      | `128`             | Tile size for tiled inference (0 = no tiling) |
| `--tile-pad`       | `10`              | Padding around each tile                      |
| `--input-max-edge` | none              | Resize input so longest edge <= this value    |

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Acknowledgments

This project is a Rust reimplementation of [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al., which is licensed under the BSD 3-Clause License.
