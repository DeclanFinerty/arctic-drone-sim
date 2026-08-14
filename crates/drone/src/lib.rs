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
    /// Orientation as a unit quaternion `[w, x, y, z]`.
    pub orientation: [f64; 4],
    /// Body-frame angular velocity in rad/s.
    pub angular_velocity: [f64; 3],
    /// Remaining battery energy in Wh.
    pub battery_wh: f64,
}

/// Command sent to the drone at each control step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlInput {
    /// Collective thrust in N.
    pub thrust: f64,
    /// Body-frame torques in N·m.
    pub torque: [f64; 3],
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
