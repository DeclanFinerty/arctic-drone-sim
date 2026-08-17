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
    /// Optional cruise-speed cap. When set, horizontal position error is
    /// clamped so the steady-state balance `kp*err == kv*v` gives cruise
    /// speed at the cap. Prevents the drone from ballistically accelerating
    /// past its rated speed on long point-to-point legs.
    pub max_horiz_speed_ms: Option<f64>,
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
            max_horiz_speed_ms: None,
        }
    }

    pub fn with_max_horiz_speed(mut self, v_ms: f64) -> Self {
        self.max_horiz_speed_ms = Some(v_ms);
        self
    }
}

impl Controller for PidStationKeep {
    fn compute_control(
        &mut self,
        state: &DroneState,
        _all_states: &[DroneState],
        mission: &dyn Mission,
        _wind: [f64; 3],
        time_s: f64,
        dt_s: f64,
    ) -> ControlInput {
        let target = mission.target(time_s).unwrap_or([0.0; 3]);
        let mut error = [
            target[0] - state.position[0],
            target[1] - state.position[1],
            target[2] - state.position[2],
        ];

        // Horizontal-error clamp: caps kp*|err_h| at kv*v_cruise so steady-state
        // cruise speed is bounded. Vertical error is left alone.
        if let Some(v_cruise) = self.max_horiz_speed_ms {
            let horiz_mag = (error[0].powi(2) + error[1].powi(2)).sqrt();
            let max_horiz = self.kv * v_cruise / self.kp;
            if horiz_mag > max_horiz {
                let s = max_horiz / horiz_mag;
                error[0] *= s;
                error[1] *= s;
            }
        }

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

/// Tracks another drone's position offset by a fixed world-frame vector.
/// Reuses the same PID gains and cruise-speed limit shape as PidStationKeep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FollowLeader {
    pub leader_idx: usize,
    pub offset_m: [f64; 3],
    pub kp: f64,
    pub kv: f64,
    pub mass_kg: f64,
    pub gravity_ms2: f64,
    pub max_horiz_speed_ms: Option<f64>,
}

impl FollowLeader {
    pub fn new(leader_idx: usize, offset_m: [f64; 3], kp: f64, kv: f64, mass_kg: f64) -> Self {
        Self {
            leader_idx,
            offset_m,
            kp,
            kv,
            mass_kg,
            gravity_ms2: 9.81,
            max_horiz_speed_ms: None,
        }
    }

    pub fn with_max_horiz_speed(mut self, v_ms: f64) -> Self {
        self.max_horiz_speed_ms = Some(v_ms);
        self
    }
}

impl Controller for FollowLeader {
    fn compute_control(
        &mut self,
        state: &DroneState,
        all_states: &[DroneState],
        _mission: &dyn Mission,
        _wind: [f64; 3],
        _time_s: f64,
        _dt_s: f64,
    ) -> ControlInput {
        // If the leader index is out of range (shouldn't happen; guarded at
        // sim start) fall back to the follower's current position -> no force
        // change other than gravity feed-forward.
        let leader = all_states.get(self.leader_idx).unwrap_or(state);
        let target = [
            leader.position[0] + self.offset_m[0],
            leader.position[1] + self.offset_m[1],
            leader.position[2] + self.offset_m[2],
        ];
        let mut error = [
            target[0] - state.position[0],
            target[1] - state.position[1],
            target[2] - state.position[2],
        ];

        if let Some(v_cruise) = self.max_horiz_speed_ms {
            let horiz_mag = (error[0].powi(2) + error[1].powi(2)).sqrt();
            let max_horiz = self.kv * v_cruise / self.kp;
            if horiz_mag > max_horiz {
                let s = max_horiz / horiz_mag;
                error[0] *= s;
                error[1] *= s;
            }
        }

        let gravity_ff = self.mass_kg * self.gravity_ms2;
        // Velocity-tracking term: match the leader's velocity for smoother
        // formation flying (subtract leader v from follower v before feedback).
        let vrel = [
            state.velocity[0] - leader.velocity[0],
            state.velocity[1] - leader.velocity[1],
            state.velocity[2] - leader.velocity[2],
        ];
        let force_n = [
            self.kp * error[0] - self.kv * vrel[0],
            self.kp * error[1] - self.kv * vrel[1],
            gravity_ff + self.kp * error[2] - self.kv * vrel[2],
        ];
        ControlInput { force_n }
    }
}
