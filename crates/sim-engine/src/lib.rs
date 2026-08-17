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
        let control = controller.compute_control(&state, &[], mission, wind_v, t, dt_s);
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

/// Advance N drones simultaneously through one shared wind field. Each drone
/// has its own dynamics, controller, mission, and initial state. Controllers
/// see all drone states each step so followers can reference the leader.
/// Terminates when any leader mission completes, any drone drains its
/// battery, or `max_duration_s` is reached. Returns one SimulationResult per
/// drone, in the input order.
pub fn run_multi<W, D>(
    wind: &W,
    drone_models: &[&D],
    missions: &[&dyn Mission],
    controllers: &mut [Box<dyn Controller>],
    initials: Vec<DroneState>,
    dt_s: f64,
    max_duration_s: f64,
) -> Vec<SimulationResult>
where
    W: WindFieldQuery + ?Sized,
    D: DroneDynamics + ?Sized,
{
    let n = initials.len();
    assert_eq!(drone_models.len(), n);
    assert_eq!(missions.len(), n);
    assert_eq!(controllers.len(), n);

    let capacity = (max_duration_s / dt_s).ceil() as usize + 1;
    let mut times = Vec::with_capacity(capacity);
    let mut states: Vec<Vec<DroneState>> = (0..n).map(|_| Vec::with_capacity(capacity)).collect();
    let mut commands: Vec<Vec<[f64; 3]>> = (0..n).map(|_| Vec::with_capacity(capacity)).collect();
    let mut winds: Vec<Vec<[f64; 3]>> = (0..n).map(|_| Vec::with_capacity(capacity)).collect();

    let mut current: Vec<DroneState> = initials;
    let mut t = 0.0;
    times.push(t);
    for i in 0..n { states[i].push(current[i].clone()); }

    let terminated: &'static str = loop {
        // Termination: leader (index 0) mission completes, or any drone dead battery, or timeout.
        if missions[0].is_complete(&current[0], t) {
            break "mission_complete";
        }
        if t >= max_duration_s {
            break "max_duration";
        }
        if current.iter().any(|s| s.battery_wh <= 0.0) {
            break "battery_empty";
        }

        // Sample wind for each drone at its current position.
        let winds_now: Vec<[f64; 3]> = current.iter()
            .map(|s| wind.wind_at(s.position, t))
            .collect();

        // Compute controls with access to the shared state snapshot.
        let mut controls = Vec::with_capacity(n);
        for i in 0..n {
            let c = controllers[i].compute_control(
                &current[i], &current, missions[i], winds_now[i], t, dt_s,
            );
            controls.push(c);
        }

        // Advance each drone.
        let mut next = Vec::with_capacity(n);
        for i in 0..n {
            let s_next = drone_models[i].step(&current[i], &controls[i], winds_now[i], dt_s);
            next.push(s_next);
        }
        current = next;
        t += dt_s;

        for i in 0..n {
            commands[i].push(controls[i].force_n);
            winds[i].push(winds_now[i]);
            states[i].push(current[i].clone());
        }
        times.push(t);
    };

    let mut results = Vec::with_capacity(n);
    for i in 0..n {
        let score = missions[i].score(&states[i]);
        results.push(SimulationResult {
            steps_run: states[i].len(),
            score,
            times_s: times.clone(),
            states: std::mem::take(&mut states[i]),
            commanded_force_n: std::mem::take(&mut commands[i]),
            wind_at_drone: std::mem::take(&mut winds[i]),
            terminated_reason: terminated,
        });
    }
    results
}
