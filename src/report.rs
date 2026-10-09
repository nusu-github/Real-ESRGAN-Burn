//! The JSON report written after a run.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::model_variant::ModelVariant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Success,
    Failed,
}

#[derive(Debug, Serialize)]
pub struct ModelRunEntry {
    pub model_name: String,
    pub weight_file: String,
    /// The checkpoint key the weights were loaded from.
    pub top_level_key: Option<String>,
    pub status: RunStatus,
    pub inference_executed: bool,
    pub output_path: Option<String>,
    pub error: Option<String>,
}

impl ModelRunEntry {
    /// A successful entry with nothing recorded yet.
    pub fn new(variant: ModelVariant) -> Self {
        Self {
            model_name: variant.name().to_string(),
            weight_file: variant.file_name().to_string(),
            top_level_key: None,
            status: RunStatus::Success,
            inference_executed: false,
            output_path: None,
            error: None,
        }
    }

    pub fn fail(&mut self, error: &anyhow::Error) {
        self.status = RunStatus::Failed;
        self.error = Some(format!("{error:#}"));
    }
}

#[derive(Debug, Serialize)]
pub struct RunReport {
    pub entries: Vec<ModelRunEntry>,
}

impl RunReport {
    pub fn failures(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.status == RunStatus::Failed)
            .count()
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self).context("failed to serialize report")?;
        std::fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_serializes_lowercase() {
        let mut entry = ModelRunEntry::new(ModelVariant::RealEsrganX4Plus);
        entry.fail(&anyhow::anyhow!("boom"));
        let report = RunReport {
            entries: vec![entry, ModelRunEntry::new(ModelVariant::RealEsrganX2Plus)],
        };
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["entries"][0]["status"], "failed");
        assert_eq!(json["entries"][0]["error"], "boom");
        assert_eq!(json["entries"][1]["status"], "success");
        assert_eq!(report.failures(), 1);
    }
}
