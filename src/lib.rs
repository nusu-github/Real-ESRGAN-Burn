//! Real-ESRGAN inference with Burn.
//!
//! Backends are compiled in through the `flex`, `cuda` and `wgpu` features; every API takes
//! the `burn::prelude::Device` to run on.

pub mod arch;
pub mod infer;
pub mod loader;
pub mod model_variant;
pub mod report;
