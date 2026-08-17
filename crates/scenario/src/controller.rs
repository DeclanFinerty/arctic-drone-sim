//! Built-in controllers.

use serde::{Deserialize, Serialize};

use drone::{ControlInput, DroneState};

use crate::{Controller, Mission};

/// PID station-keeping controller with velocity feedback (a PDI+V form):
/// the D term is on measured velocity, not on the error derivative, which
/// avoids the derivative-kick from a step-change target and gives cleaner
/// damping. Includes a gravity feed-forward on the z-axis so `kp` doesn't
/// have to compensate for weight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PidStationKeep {
    /// Position error gain (N/m).
    pub kp: f64,
    /// Velocity feedback gain (N.s/m).
    pub kv: f64,
    /// Integral gain on position error (N/(m.s)).
    pub ki: f64,
    /// Drone mass, used for the gravity feed-forward.
    pub mass_kg: f64,
    pub gravity_ms2: f64,
    /// Accumulated position-error integral.
    pub integral_m_s: [f64; 3],
    /// Anti-windup clamp on the integral term (m*s).
    pub integral_clamp_m_s: f64,
}

impl PidStationKeep {
    pub fn new(kp: f64, kv: f64, ki: f64, mass_kg: f64) -> Self {
        Self {
            kp,
            kv,
            ki,
            mass_kg,
            gravity_ms2: 9.81,
            integral_m_s: [0.0; 3],
            integral_clamp_m_s: 50.0,
        }
    }
}

impl Controller for PidStationKeep {
    fn compute_control(
        &mut self,
        state: &DroneState,
        mission: &dyn Mission,
        _wind: [f64; 3],
        dt_s: f64,
    ) -> ControlInput {
        let target = mission.target(0.0).unwrap_or([0.0; 3]);
        let error = [
            target[0] - state.position[0],
            target[1] - state.position[1],
            target[2] - state.position[2],
        ];

        for i in 0..3 {
            self.integral_m_s[i] += error[i] * dt_s;
            self.integral_m_s[i] = self
                .integral_m_s[i]
                .clamp(-self.integral_clamp_m_s, self.integral_clamp_m_s);
        }

        let gravity_ff = self.mass_kg * self.gravity_ms2;
        let force_n = [
            self.kp * error[0] - self.kv * state.velocity[0] + self.ki * self.integral_m_s[0],
            self.kp * error[1] - self.kv * state.velocity[1] + self.ki * self.integral_m_s[1],
            gravity_ff + self.kp * error[2] - self.kv * state.velocity[2]
                + self.ki * self.integral_m_s[2],
        ];

        ControlInput { force_n }
    }
}
