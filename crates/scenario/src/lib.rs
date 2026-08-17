pub mod controller;
pub mod metrics;
pub mod mission;

use drone::{ControlInput, DroneState};

/// A mission defines the objective and success criteria for a run.
pub trait Mission {
    fn is_complete(&self, state: &DroneState, time: f64) -> bool;
    fn score(&self, history: &[DroneState]) -> f64;
    /// Target position at a given time, if the mission has a moving/static target.
    fn target(&self, time: f64) -> Option<[f64; 3]>;
}

/// A controller produces control inputs from the drone's own state, the
/// broader multi-drone state (empty slice for single-drone runs), mission
/// context, local wind, current sim time, and the timestep it is being
/// asked to integrate over. `time_s` is passed so time-varying missions can
/// be sampled at the right instant.
pub trait Controller {
    fn compute_control(
        &mut self,
        state: &DroneState,
        all_states: &[DroneState],
        mission: &dyn Mission,
        wind: [f64; 3],
        time_s: f64,
        dt_s: f64,
    ) -> ControlInput;
}
