//! Persist a `SimulationResult` to disk as `.npy` arrays plus a
//! `metadata.json` sidecar. Consumed by `python/scripts/plot_run.py`.

use std::fs;
use std::path::Path;

use anyhow::Result;
use ndarray::{Array1, Array2};
use ndarray_npy::write_npy;
use serde::{Deserialize, Serialize};

use crate::SimulationResult;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    pub steps_run: usize,
    pub dt_s: f64,
    pub duration_s: f64,
    pub score: f64,
    pub terminated_reason: String,
    pub target_m: [f64; 3],
    pub tolerance_m: f64,
    pub mean_wind_ms: f64,
    pub notes: Option<String>,
}

pub fn save_run(result: &SimulationResult, dir: &Path, meta: &RunMetadata) -> Result<()> {
    fs::create_dir_all(dir)?;

    let times = Array1::from_vec(result.times_s.clone());
    let positions = state_field(result, |s| s.position);
    let velocities = state_field(result, |s| s.velocity);
    let batteries = Array1::from_vec(result.states.iter().map(|s| s.battery_wh).collect());
    let commands = row_stack(&result.commanded_force_n);
    let winds = row_stack(&result.wind_at_drone);

    write_npy(dir.join("times.npy"), &times)?;
    write_npy(dir.join("positions.npy"), &positions)?;
    write_npy(dir.join("velocities.npy"), &velocities)?;
    write_npy(dir.join("battery_wh.npy"), &batteries)?;
    write_npy(dir.join("commanded_force_n.npy"), &commands)?;
    write_npy(dir.join("wind_at_drone_ms.npy"), &winds)?;

    fs::write(dir.join("metadata.json"), serde_json::to_string_pretty(meta)?)?;
    Ok(())
}

fn state_field<F: Fn(&drone::DroneState) -> [f64; 3]>(
    result: &SimulationResult,
    accessor: F,
) -> Array2<f64> {
    let rows = result.states.len();
    let mut arr = Array2::<f64>::zeros((rows, 3));
    for (i, s) in result.states.iter().enumerate() {
        let v = accessor(s);
        arr[[i, 0]] = v[0];
        arr[[i, 1]] = v[1];
        arr[[i, 2]] = v[2];
    }
    arr
}

fn row_stack(rows: &[[f64; 3]]) -> Array2<f64> {
    let n = rows.len();
    let mut arr = Array2::<f64>::zeros((n, 3));
    for (i, r) in rows.iter().enumerate() {
        arr[[i, 0]] = r[0];
        arr[[i, 1]] = r[1];
        arr[[i, 2]] = r[2];
    }
    arr
}
