//! Burn ports of the networks behind the official Real-ESRGAN checkpoints.

pub mod discriminator;
pub mod ops;
pub mod rrdb;
pub mod srvgg;

use burn::prelude::*;

pub use discriminator::{UnetDiscriminatorSn, UnetDiscriminatorSnConfig};
pub use rrdb::{RrdbNet, RrdbNetConfig};
pub use srvgg::{SrvggNetCompact, SrvggNetCompactConfig};

/// A super-resolution network: maps `[batch, 3, h, w]` in `[0, 1]` to
/// `[batch, 3, h * scale, w * scale]`.
#[derive(Module, Debug)]
// `Box<T>` is not a `Module`, and a generator is built once per run.
#[allow(clippy::large_enum_variant)]
pub enum Generator {
    Rrdb(RrdbNet),
    Srvgg(SrvggNetCompact),
}

impl Generator {
    pub fn scale(&self) -> usize {
        match self {
            Self::Rrdb(model) => model.scale(),
            Self::Srvgg(model) => model.scale(),
        }
    }

    /// Input height and width must be multiples of this value.
    pub fn input_multiple(&self) -> usize {
        match self {
            Self::Rrdb(model) => model.input_multiple(),
            Self::Srvgg(_) => 1,
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        match self {
            Self::Rrdb(model) => model.forward(x),
            Self::Srvgg(model) => model.forward(x),
        }
    }
}

#[derive(Debug, Clone)]
pub enum GeneratorConfig {
    Rrdb(RrdbNetConfig),
    Srvgg(SrvggNetCompactConfig),
}

impl GeneratorConfig {
    pub fn init(&self, device: &Device) -> Generator {
        match self {
            Self::Rrdb(config) => Generator::Rrdb(config.init(device)),
            Self::Srvgg(config) => Generator::Srvgg(config.init(device)),
        }
    }

    pub fn scale(&self) -> usize {
        match self {
            Self::Rrdb(config) => config.scale,
            Self::Srvgg(config) => config.upscale,
        }
    }
}
