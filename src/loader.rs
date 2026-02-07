use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Result, bail};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

#[derive(Debug, Clone)]
pub struct KeyRemap {
    pub pattern: String,
    pub replacement: String,
}

impl KeyRemap {
    pub fn new(pattern: impl Into<String>, replacement: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            replacement: replacement.into(),
        }
    }
}

pub fn load_model_from_pytorch<B, M, F>(
    weight_path: &Path,
    top_level_keys: &[&str],
    key_remaps: &[KeyRemap],
    device: &B::Device,
    build_model: F,
) -> Result<(M, String)>
where
    B: Backend,
    M: Module<B>,
    F: Fn() -> M,
{
    let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
    let mut errors = Vec::new();

    for top_level_key in top_level_keys {
        let mut args = LoadArgs::new(weight_path.to_path_buf()).with_top_level_key(top_level_key);

        for remap in key_remaps {
            args = args.with_key_remap(remap.pattern.as_str(), remap.replacement.as_str());
        }

        match recorder.load(args, device) {
            Ok(record) => {
                let model = build_model().load_record(record);
                return Ok((model, (*top_level_key).to_string()));
            }
            Err(err) => {
                errors.push(format!("{top_level_key}: {err}"));
            }
        }
    }

    bail!(
        "failed to load {:?}; attempts: {}",
        weight_path,
        errors.join(" | ")
    )
}

pub fn srvgg_key_remaps(num_conv: usize) -> Vec<KeyRemap> {
    let mut remaps = Vec::new();

    remaps.push(KeyRemap::new(r"^body\.0\.(.*)$", "conv_first.$1"));
    remaps.push(KeyRemap::new(r"^body\.1\.weight$", "acts.0.alpha"));

    for i in 0..num_conv {
        let conv_idx = 2 + i * 2;
        let act_idx = conv_idx + 1;

        remaps.push(KeyRemap::new(
            format!(r"^body\.{conv_idx}\.(.*)$"),
            format!("convs.{i}.$1"),
        ));

        remaps.push(KeyRemap::new(
            format!(r"^body\.{act_idx}\.weight$"),
            format!("acts.{}.alpha", i + 1),
        ));
    }

    let last_idx = 2 + 2 * num_conv;
    remaps.push(KeyRemap::new(
        format!(r"^body\.{last_idx}\.(.*)$"),
        "conv_last.$1",
    ));

    remaps
}

pub fn reserialize_pytorch_checkpoint_with_python(weight_path: &Path) -> Result<PathBuf> {
    let converted_path = weight_path.with_extension("burnfix.pth");

    if converted_path.exists() {
        let src_meta = std::fs::metadata(weight_path)?;
        let dst_meta = std::fs::metadata(&converted_path)?;
        if dst_meta.modified()? >= src_meta.modified()? {
            return Ok(converted_path);
        }
    }

    let script = r#"
import sys
import torch

src, dst = sys.argv[1], sys.argv[2]
ckpt = torch.load(src, map_location="cpu")

out = {}
if isinstance(ckpt, dict):
    if isinstance(ckpt.get("params_ema"), dict):
        out["params_ema"] = ckpt["params_ema"]
    if isinstance(ckpt.get("params"), dict):
        out["params"] = ckpt["params"]

if not out:
    if isinstance(ckpt, dict):
        out["params"] = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format for burnfix conversion.")

torch.save(out, dst)
"#;

    let output = Command::new("python")
        .arg("-c")
        .arg(script)
        .arg(weight_path)
        .arg(&converted_path)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "failed to convert checkpoint {:?} with python: {}",
            weight_path,
            stderr.trim()
        );
    }

    Ok(converted_path)
}

#[cfg(test)]
mod tests {
    use super::srvgg_key_remaps;

    #[test]
    fn srvgg_remap_count_is_expected() {
        let remaps = srvgg_key_remaps(16);
        assert_eq!(remaps.len(), 35);
        assert_eq!(remaps[0].replacement, "conv_first.$1");
        assert_eq!(remaps[34].replacement, "conv_last.$1");
    }
}
