//! Loading official PyTorch checkpoints (`.pth`).

use std::path::Path;

use anyhow::{Result, bail};
use burn::store::{ModuleSnapshot, PytorchStore};

/// Loads the state dict stored under the first matching key of `top_level_keys` into `model`
/// and returns that key.
///
/// Both the zip format (PyTorch >= 1.6) and the legacy pickle format are supported.
/// Loading is strict: every parameter of `model` must be present with the right shape, and
/// every tensor in the state dict must be used.
pub fn load_pytorch_weights<M: ModuleSnapshot>(
    model: &mut M,
    path: &Path,
    top_level_keys: &[&str],
) -> Result<String> {
    let mut errors = Vec::new();

    for &key in top_level_keys {
        let mut store = PytorchStore::from_file(path)
            .with_top_level_key(key)
            // Module structures mirror the PyTorch ones index by index, including
            // parameter-free entries, so indices must be kept as they are.
            .map_indices_contiguous(false);

        match model.load_from(&mut store) {
            Ok(result) if result.unused.is_empty() => return Ok(key.to_string()),
            Ok(result) => errors.push(format!("{key}: unused tensors {:?}", result.unused)),
            Err(err) => errors.push(format!("{key}: {err}")),
        }
    }

    bail!("failed to load {}: {}", path.display(), errors.join(" | "))
}
