//! Loads small PyTorch checkpoints and compares outputs with what PyTorch computed.
//! The fixtures come from `tests/fixtures/gen_fixtures.py`.
#![cfg(feature = "flex")]

use std::path::{Path, PathBuf};

use burn::prelude::*;
use burn::store::ModuleSnapshot;
use burn::tensor::Tolerance;
use serde::Deserialize;

use real_esrgan_burn::arch::{
    GeneratorConfig, RrdbNetConfig, SrvggNetCompactConfig, UnetDiscriminatorSnConfig,
};
use real_esrgan_burn::loader::load_pytorch_weights;

#[derive(Deserialize)]
struct Reference {
    input_shape: [usize; 4],
    input: Vec<f32>,
    output_shape: [usize; 4],
    output: Vec<f32>,
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Loads `<name>.pth` into `model` and checks `forward` against `<name>.json`.
fn assert_matches_pytorch<M: ModuleSnapshot>(
    name: &str,
    mut model: M,
    key: &str,
    forward: impl Fn(&M, Tensor<4>) -> Tensor<4>,
) {
    let device = Device::flex();
    let loaded_key = load_pytorch_weights(
        &mut model,
        &fixture(&format!("{name}.pth")),
        &["missing", key],
    )
    .unwrap();
    assert_eq!(loaded_key, key);

    let json = std::fs::read_to_string(fixture(&format!("{name}.json"))).unwrap();
    let reference: Reference = serde_json::from_str(&json).unwrap();
    let input = Tensor::from_data(
        TensorData::new(reference.input, reference.input_shape),
        &device,
    );
    let expected = TensorData::new(reference.output, reference.output_shape);

    forward(&model, input)
        .into_data()
        .assert_approx_eq::<f32>(&expected, Tolerance::absolute(1e-4));
}

#[test]
fn rrdb_x2_matches_pytorch() {
    let config = RrdbNetConfig::new(2)
        .with_num_feat(8)
        .with_num_block(2)
        .with_num_grow_ch(4);
    let model = GeneratorConfig::Rrdb(config).init(&Device::flex());
    assert_matches_pytorch("rrdb_x2", model, "params_ema", |m, x| m.forward(x));
}

#[test]
fn rrdb_from_legacy_checkpoint_matches_pytorch() {
    let config = RrdbNetConfig::new(4)
        .with_num_feat(8)
        .with_num_block(1)
        .with_num_grow_ch(4);
    let model = config.init(&Device::flex());
    assert_matches_pytorch("rrdb_x4_legacy", model, "params", |m, x| m.forward(x));
}

#[test]
fn srvgg_matches_pytorch() {
    let config = SrvggNetCompactConfig::new(4)
        .with_num_feat(8)
        .with_num_conv(3);
    let model = GeneratorConfig::Srvgg(config).init(&Device::flex());
    assert_matches_pytorch("srvgg_x4", model, "params", |m, x| m.forward(x));
}

#[test]
fn spectral_norm_discriminator_matches_pytorch() {
    let model = UnetDiscriminatorSnConfig::new()
        .with_num_feat(4)
        .init(&Device::flex());
    assert_matches_pytorch("unet_disc", model, "params", |m, x| m.forward(x));
}

#[test]
fn architecture_mismatch_is_rejected() {
    // One RRDB block too many: `body.1.*` is missing from the checkpoint.
    let mut model = RrdbNetConfig::new(4)
        .with_num_feat(8)
        .with_num_block(2)
        .with_num_grow_ch(4)
        .init(&Device::flex());
    let err =
        load_pytorch_weights(&mut model, &fixture("rrdb_x4_legacy.pth"), &["params"]).unwrap_err();
    assert!(err.to_string().contains("body.1"), "{err:#}");

    // One block too few: `body.1.*` in the checkpoint would be ignored.
    let mut model = RrdbNetConfig::new(2)
        .with_num_feat(8)
        .with_num_block(1)
        .with_num_grow_ch(4)
        .init(&Device::flex());
    let err =
        load_pytorch_weights(&mut model, &fixture("rrdb_x2.pth"), &["params_ema"]).unwrap_err();
    assert!(err.to_string().contains("unused tensors"), "{err:#}");
}
