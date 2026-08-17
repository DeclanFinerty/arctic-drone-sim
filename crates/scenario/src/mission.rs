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

/// Fly from `start_m` to `goal_m` and stop within `tolerance_m` of the goal.
/// Terminates on arrival or when `timeout_s` elapses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointToPoint {
    pub start_m: [f64; 3],
    pub goal_m: [f64; 3],
    pub tolerance_m: f64,
    pub timeout_s: f64,
}

impl Mission for PointToPoint {
    fn is_complete(&self, state: &DroneState, time_s: f64) -> bool {
        if time_s >= self.timeout_s {
            return true;
        }
        let dx = state.position[0] - self.goal_m[0];
        let dy = state.position[1] - self.goal_m[1];
        let dz = state.position[2] - self.goal_m[2];
        (dx * dx + dy * dy + dz * dz).sqrt() < self.tolerance_m
    }

    /// Terminal distance from the goal — 0 for a clean arrival, otherwise the
    /// shortfall.
    fn score(&self, history: &[DroneState]) -> f64 {
        let Some(last) = history.last() else { return 0.0 };
        let dx = last.position[0] - self.goal_m[0];
        let dy = last.position[1] - self.goal_m[1];
        let dz = last.position[2] - self.goal_m[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    fn target(&self, _time_s: f64) -> Option<[f64; 3]> {
        Some(self.goal_m)
    }
}
