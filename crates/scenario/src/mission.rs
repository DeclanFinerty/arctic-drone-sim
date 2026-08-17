//! Built-in mission implementations.

use serde::{Deserialize, Serialize};

use drone::DroneState;

use crate::Mission;

/// Hover at a fixed target position for a fixed duration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationKeep {
    pub target_m: [f64; 3],
    pub tolerance_m: f64,
    pub duration_s: f64,
}

impl Mission for StationKeep {
    fn is_complete(&self, _state: &DroneState, time_s: f64) -> bool {
        time_s >= self.duration_s
    }

    /// Root-mean-square position error over the full history.
    fn score(&self, history: &[DroneState]) -> f64 {
        if history.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = history
            .iter()
            .map(|s| {
                let dx = s.position[0] - self.target_m[0];
                let dy = s.position[1] - self.target_m[1];
                let dz = s.position[2] - self.target_m[2];
                dx * dx + dy * dy + dz * dz
            })
            .sum();
        (sum_sq / history.len() as f64).sqrt()
    }

    fn target(&self, _time_s: f64) -> Option<[f64; 3]> {
        Some(self.target_m)
    }
}
