//! Flight dynamics: point-mass quadrotor with quadratic drag and thrust-limited
//! command tracking. Suitable for trajectory-level studies where rotational
//! dynamics are not the focus.

use serde::{Deserialize, Serialize};

use crate::{ControlInput, DroneCapabilities, DroneDynamics, DroneState};

/// Simplified 3-DOF quadrotor: a point mass in world coordinates driven by a
/// commanded thrust vector, subject to quadratic aerodynamic drag against the
/// local wind and gravity along -z.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointMassQuad {
    pub capabilities: DroneCapabilities,
    /// Effective drag reference area (m^2). Combined with `drag_coefficient`
    /// gives the drag scaling C_d * A.
    pub drag_area_m2: f64,
    /// Drag coefficient (dimensionless). Typical order of magnitude ~1 for a
    /// bluff quadrotor airframe with exposed rotors.
    pub drag_coefficient: f64,
    /// Air density in kg/m^3. Arctic ~0 C sea level ~= 1.29.
    pub air_density_kgm3: f64,
    /// Battery drain model: watts per newton of applied thrust magnitude.
    /// Linear approximation; a full model would use P ~ T^(3/2) via momentum
    /// theory. Calibrate so hover thrust gives realistic hover power.
    pub power_per_thrust_w_per_n: f64,
    pub gravity_ms2: f64,
}

impl PointMassQuad {
    /// Small commercial-class quadrotor with Arctic air density.
    pub fn arctic_small_quad() -> Self {
        Self {
            capabilities: DroneCapabilities {
                mass_kg: 1.5,
                max_thrust_n: 30.0,
                max_wind_speed_ms: 15.0,
                max_speed_ms: 20.0,
                battery_capacity_wh: 100.0,
            },
            drag_area_m2: 0.05,
            drag_coefficient: 1.0,
            air_density_kgm3: 1.29,
            power_per_thrust_w_per_n: 6.7,
            gravity_ms2: 9.81,
        }
    }

    fn clamp_thrust(&self, force: [f64; 3]) -> [f64; 3] {
        let mag = (force[0].powi(2) + force[1].powi(2) + force[2].powi(2)).sqrt();
        if mag > self.capabilities.max_thrust_n && mag > 0.0 {
            let s = self.capabilities.max_thrust_n / mag;
            [force[0] * s, force[1] * s, force[2] * s]
        } else {
            force
        }
    }
}

impl DroneDynamics for PointMassQuad {
    fn step(
        &self,
        state: &DroneState,
        control: &ControlInput,
        wind: [f64; 3],
        dt: f64,
    ) -> DroneState {
        let m = self.capabilities.mass_kg;
        let thrust = self.clamp_thrust(control.force_n);

        // Quadratic drag opposes velocity relative to the local air motion.
        let v_rel = [
            state.velocity[0] - wind[0],
            state.velocity[1] - wind[1],
            state.velocity[2] - wind[2],
        ];
        let v_rel_mag = (v_rel[0].powi(2) + v_rel[1].powi(2) + v_rel[2].powi(2)).sqrt();
        let drag_scale = 0.5 * self.air_density_kgm3 * self.drag_area_m2 * self.drag_coefficient * v_rel_mag;
        let drag = [-drag_scale * v_rel[0], -drag_scale * v_rel[1], -drag_scale * v_rel[2]];

        let acc = [
            (thrust[0] + drag[0]) / m,
            (thrust[1] + drag[1]) / m,
            (thrust[2] + drag[2]) / m - self.gravity_ms2,
        ];

        // Semi-implicit (symplectic) Euler: update velocity first, then position.
        let new_velocity = [
            state.velocity[0] + acc[0] * dt,
            state.velocity[1] + acc[1] * dt,
            state.velocity[2] + acc[2] * dt,
        ];
        let new_position = [
            state.position[0] + new_velocity[0] * dt,
            state.position[1] + new_velocity[1] * dt,
            state.position[2] + new_velocity[2] * dt,
        ];

        let thrust_mag = (thrust[0].powi(2) + thrust[1].powi(2) + thrust[2].powi(2)).sqrt();
        let energy_wh = self.power_per_thrust_w_per_n * thrust_mag * dt / 3600.0;
        let new_battery = (state.battery_wh - energy_wh).max(0.0);

        DroneState {
            position: new_position,
            velocity: new_velocity,
            orientation: state.orientation,
            angular_velocity: state.angular_velocity,
            battery_wh: new_battery,
        }
    }

    fn capabilities(&self) -> &DroneCapabilities {
        &self.capabilities
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hover_thrust_balances_gravity_in_still_air() {
        let quad = PointMassQuad::arctic_small_quad();
        let state = DroneState::at_rest([0.0, 0.0, 30.0], 100.0);
        let hover = [0.0, 0.0, quad.capabilities.mass_kg * quad.gravity_ms2];
        let next = quad.step(&state, &ControlInput { force_n: hover }, [0.0; 3], 0.01);
        assert!(next.velocity[2].abs() < 1e-9);
        assert!((next.position[2] - 30.0).abs() < 1e-9);
    }

    #[test]
    fn thrust_is_clipped_at_capability_limit() {
        let quad = PointMassQuad::arctic_small_quad();
        let state = DroneState::at_rest([0.0, 0.0, 30.0], 100.0);
        // Command 100 N (way over the 30 N limit).
        let control = ControlInput { force_n: [100.0, 0.0, 0.0] };
        let next = quad.step(&state, &control, [0.0; 3], 0.1);
        // Applied acceleration should reflect the clipped 30 N thrust.
        let expected_ax = 30.0 / quad.capabilities.mass_kg;
        assert!((next.velocity[0] - expected_ax * 0.1).abs() < 1e-9);
    }

    #[test]
    fn battery_drains_during_hover() {
        let quad = PointMassQuad::arctic_small_quad();
        let state = DroneState::at_rest([0.0, 0.0, 30.0], 100.0);
        let hover = [0.0, 0.0, quad.capabilities.mass_kg * quad.gravity_ms2];
        let mut s = state;
        for _ in 0..100 {
            s = quad.step(&s, &ControlInput { force_n: hover }, [0.0; 3], 0.01);
        }
        assert!(s.battery_wh < 100.0);
        assert!(s.battery_wh > 99.0);
    }
}
