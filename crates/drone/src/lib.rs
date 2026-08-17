pub mod config;
pub mod dynamics;
pub mod model;

use serde::{Deserialize, Serialize};

/// Instantaneous drone state in world coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DroneState {
    /// Position in meters, world frame.
    pub position: [f64; 3],
    /// Linear velocity in m/s, world frame.
    pub velocity: [f64; 3],
    /// Orientation as a unit quaternion `[w, x, y, z]`. Point-mass models
    /// leave this at identity `[1, 0, 0, 0]`.
    pub orientation: [f64; 4],
    /// Body-frame angular velocity in rad/s. Zero for point-mass models.
    pub angular_velocity: [f64; 3],
    /// Remaining battery energy in Wh.
    pub battery_wh: f64,
}

impl DroneState {
    pub fn at_rest(position: [f64; 3], battery_wh: f64) -> Self {
        Self {
            position,
            velocity: [0.0; 3],
            orientation: [1.0, 0.0, 0.0, 0.0],
            angular_velocity: [0.0; 3],
            battery_wh,
        }
    }
}

/// Command sent to the drone at each control step.
///
/// For point-mass models this is simply a 3D thrust vector in the world frame.
/// A future rigid-body model can add body-frame torques as separate fields.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ControlInput {
    /// 3D commanded force in N, world frame.
    pub force_n: [f64; 3],
}

/// Static, drone-model-level capability envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DroneCapabilities {
    pub mass_kg: f64,
    pub max_thrust_n: f64,
    pub max_wind_speed_ms: f64,
    pub max_speed_ms: f64,
    pub battery_capacity_wh: f64,
}

/// Step drone physics forward under an external wind field.
pub trait DroneDynamics {
    fn step(
        &self,
        state: &DroneState,
        control: &ControlInput,
        wind: [f64; 3],
        dt: f64,
    ) -> DroneState;

    fn capabilities(&self) -> &DroneCapabilities;
}
