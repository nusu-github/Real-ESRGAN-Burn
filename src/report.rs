use std::path::Path;

use anyhow::{Result, anyhow};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ModelRunEntry {
    pub model_name: String,
    pub weight_file: String,
    pub top_level_key: Option<String>,
    pub status: String,
    pub inference_executed: bool,
    pub output_path: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RunReport {
    pub entries: Vec<ModelRunEntry>,
}

pub fn write_report(path: &Path, report: &RunReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let json = serde_json::to_string_pretty(report)
        .map_err(|e| anyhow!("failed to serialize report: {e}"))?;
    std::fs::write(path, json).map_err(|e| anyhow!("failed to write {:?}: {e}", path))
}
