pub mod output;
pub mod state;

use drone::{DroneDynamics, DroneState};
use scenario::{Controller, Mission};
use wind_field::WindFieldQuery;

/// Full time series of a single simulation run.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub times_s: Vec<f64>,
    pub states: Vec<DroneState>,
    pub commanded_force_n: Vec<[f64; 3]>,
    pub wind_at_drone: Vec<[f64; 3]>,
    pub score: f64,
    pub steps_run: usize,
    pub terminated_reason: &'static str,
}

/// Advance the simulation until the mission reports complete, the drone's
/// battery drains, or `max_duration_s` is reached — whichever comes first.
pub fn run<W, D, C>(
    wind: &W,
    drone_model: &D,
    mission: &dyn Mission,
    controller: &mut C,
    initial: DroneState,
    dt_s: f64,
    max_duration_s: f64,
) -> SimulationResult
where
    W: WindFieldQuery + ?Sized,
    D: DroneDynamics + ?Sized,
    C: Controller + ?Sized,
{
    let capacity = (max_duration_s / dt_s).ceil() as usize + 1;
    let mut times = Vec::with_capacity(capacity);
    let mut states = Vec::with_capacity(capacity);
    let mut commands = Vec::with_capacity(capacity);
    let mut winds = Vec::with_capacity(capacity);

    let mut state = initial;
    let mut t = 0.0;
    times.push(t);
    states.push(state.clone());

    let terminated = loop {
        if mission.is_complete(&state, t) {
            break "mission_complete";
        }
        if t >= max_duration_s {
            break "max_duration";
        }
        if state.battery_wh <= 0.0 {
            break "battery_empty";
        }

        let wind_v = wind.wind_at(state.position, t);
        let control = controller.compute_control(&state, mission, wind_v, dt_s);
        state = drone_model.step(&state, &control, wind_v, dt_s);
        t += dt_s;

        commands.push(control.force_n);
        winds.push(wind_v);
        times.push(t);
        states.push(state.clone());
    };

    let score = mission.score(&states);
    SimulationResult {
        steps_run: states.len(),
        score,
        times_s: times,
        states,
        commanded_force_n: commands,
        wind_at_drone: winds,
        terminated_reason: terminated,
    }
}
