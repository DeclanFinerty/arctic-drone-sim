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

/// Parametric-course tracking mission for the leader in a formation flight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FollowCourse {
    pub course: CourseKind,
    pub duration_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CourseKind {
    /// Lissajous 1:2 figure-8. `x = cx + Ax * sin(w t)`, `y = cy + Ay * sin(2 w t)`.
    Figure8 {
        center_m: [f64; 3],
        amplitude_x_m: f64,
        amplitude_y_m: f64,
        period_s: f64,
    },
    /// Circle in the horizontal plane at the course center's altitude.
    Circle {
        center_m: [f64; 3],
        radius_m: f64,
        period_s: f64,
    },
}

impl CourseKind {
    pub fn position_at(&self, t: f64) -> [f64; 3] {
        match *self {
            Self::Figure8 { center_m, amplitude_x_m, amplitude_y_m, period_s } => {
                let w = 2.0 * std::f64::consts::PI / period_s;
                [
                    center_m[0] + amplitude_x_m * (w * t).sin(),
                    center_m[1] + amplitude_y_m * (2.0 * w * t).sin(),
                    center_m[2],
                ]
            }
            Self::Circle { center_m, radius_m, period_s } => {
                let w = 2.0 * std::f64::consts::PI / period_s;
                [
                    center_m[0] + radius_m * (w * t).cos(),
                    center_m[1] + radius_m * (w * t).sin(),
                    center_m[2],
                ]
            }
        }
    }
}

impl Mission for FollowCourse {
    fn is_complete(&self, _state: &DroneState, time_s: f64) -> bool {
        time_s >= self.duration_s
    }

    /// Cross-track RMS: at each recorded state, compare to the course target
    /// at the corresponding proportional time.
    fn score(&self, history: &[DroneState]) -> f64 {
        if history.len() < 2 {
            return 0.0;
        }
        let dt = self.duration_s / (history.len() - 1) as f64;
        let mut sum_sq = 0.0;
        for (i, state) in history.iter().enumerate() {
            let t = i as f64 * dt;
            let target = self.course.position_at(t);
            let dx = state.position[0] - target[0];
            let dy = state.position[1] - target[1];
            let dz = state.position[2] - target[2];
            sum_sq += dx * dx + dy * dy + dz * dz;
        }
        (sum_sq / history.len() as f64).sqrt()
    }

    fn target(&self, time_s: f64) -> Option<[f64; 3]> {
        Some(self.course.position_at(time_s))
    }
}

/// Simple time-limited mission with no target — for satellite drones whose
/// controller ignores the mission target (e.g. FollowLeader).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutMission {
    pub duration_s: f64,
}

impl Mission for TimeoutMission {
    fn is_complete(&self, _state: &DroneState, time_s: f64) -> bool {
        time_s >= self.duration_s
    }
    fn score(&self, _history: &[DroneState]) -> f64 { 0.0 }
    fn target(&self, _time_s: f64) -> Option<[f64; 3]> { None }
}
