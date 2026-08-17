use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;

use drone::dynamics::PointMassQuad;
use drone::DroneState;
use scenario::controller::PidStationKeep;
use scenario::mission::{PointToPoint, StationKeep};
use scenario::Mission;
use sim_engine::output::{RunMetadata, save_run};
use wind_field::grid::AdvectedField;
use wind_field::io::load_grid;
use wind_field::WindFieldQuery;

#[derive(Debug, Parser)]
#[command(name = "sim-engine", about = "Arctic drone simulation runner")]
struct Cli {
    /// Path to the simulation config TOML.
    #[arg(short, long)]
    config: PathBuf,
    /// Output directory for run artifacts (default: output/runs/{run_name}).
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Named subdirectory under output/runs/.
    #[arg(long, default_value = "station_keep")]
    run_name: String,
}

#[derive(Debug, Deserialize)]
struct Config {
    seed: Option<u64>,
    wind_field: WindFieldConfig,
    drone: DroneConfig,
    controller: ControllerConfig,
    mission: MissionConfig,
    simulation: SimulationConfig,
}

#[derive(Debug, Deserialize)]
struct WindFieldConfig {
    #[serde(default = "default_wind_field_dir")]
    load_from: String,
    /// Advect the snapshot past query points at this mean flow (m/s, world
    /// frame). Taylor's frozen-turbulence hypothesis. `[0, 0, 0]` disables.
    #[serde(default = "default_advection")]
    taylor_advection_ms: [f64; 3],
}

fn default_wind_field_dir() -> String {
    "output/wind_fields/mann_anisotropic".to_string()
}

fn default_advection() -> [f64; 3] {
    [6.4, 0.0, 0.0]
}

#[derive(Debug, Deserialize)]
struct DroneConfig {
    mass_kg: f64,
    max_thrust_n: f64,
    max_wind_speed_ms: f64,
    max_speed_ms: f64,
    battery_capacity_wh: f64,
    drag_area_m2: f64,
    drag_coefficient: f64,
    air_density_kgm3: f64,
    power_per_thrust_w_per_n: f64,
}

#[derive(Debug, Deserialize)]
struct ControllerConfig {
    kp: f64,
    kv: f64,
    ki: f64,
    /// Optional horizontal cruise-speed cap. When set the PID clamps position
    /// error so long point-to-point legs cruise at this speed instead of
    /// accelerating to the drag-equilibrium ballistic speed.
    #[serde(default)]
    max_horiz_speed_ms: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum MissionConfig {
    StationKeep {
        target_m: [f64; 3],
        tolerance_m: f64,
        initial_position_m: Option<[f64; 3]>,
    },
    PointToPoint {
        start_m: [f64; 3],
        goal_m: [f64; 3],
        tolerance_m: f64,
    },
}

impl MissionConfig {
    fn initial_position(&self) -> [f64; 3] {
        match self {
            Self::StationKeep { target_m, initial_position_m, .. } => {
                initial_position_m.unwrap_or(*target_m)
            }
            Self::PointToPoint { start_m, .. } => *start_m,
        }
    }
    fn target(&self) -> [f64; 3] {
        match self {
            Self::StationKeep { target_m, .. } => *target_m,
            Self::PointToPoint { goal_m, .. } => *goal_m,
        }
    }
    fn tolerance(&self) -> f64 {
        match self {
            Self::StationKeep { tolerance_m, .. } | Self::PointToPoint { tolerance_m, .. } => *tolerance_m,
        }
    }
}

#[derive(Debug, Deserialize)]
struct SimulationConfig {
    duration_s: f64,
    dt_s: f64,
    #[serde(default)]
    output_dir: Option<String>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let cfg_str = fs::read_to_string(&cli.config)
        .with_context(|| format!("reading config {}", cli.config.display()))?;
    let cfg: Config = toml::from_str(&cfg_str).context("parsing config TOML")?;

    let field_dir = PathBuf::from(&cfg.wind_field.load_from);
    let base_grid = load_grid(&field_dir).with_context(|| {
        format!("loading wind field from {}", field_dir.display())
    })?;
    let (min_c, max_c) = base_grid.domain_bounds();
    tracing::info!(
        "wind field {} loaded: shape {:?}, domain x=[{:.0},{:.0}] y=[{:.0},{:.0}] z=[{:.0},{:.0}]  Taylor advection {:?} m/s",
        field_dir.display(),
        base_grid.shape(),
        min_c[0], max_c[0], min_c[1], max_c[1], min_c[2], max_c[2],
        cfg.wind_field.taylor_advection_ms,
    );
    let wind = AdvectedField {
        inner: base_grid,
        mean_flow_ms: cfg.wind_field.taylor_advection_ms,
    };

    let drone_model = PointMassQuad {
        capabilities: drone::DroneCapabilities {
            mass_kg: cfg.drone.mass_kg,
            max_thrust_n: cfg.drone.max_thrust_n,
            max_wind_speed_ms: cfg.drone.max_wind_speed_ms,
            max_speed_ms: cfg.drone.max_speed_ms,
            battery_capacity_wh: cfg.drone.battery_capacity_wh,
        },
        drag_area_m2: cfg.drone.drag_area_m2,
        drag_coefficient: cfg.drone.drag_coefficient,
        air_density_kgm3: cfg.drone.air_density_kgm3,
        power_per_thrust_w_per_n: cfg.drone.power_per_thrust_w_per_n,
        gravity_ms2: 9.81,
    };

    let mission: Box<dyn Mission> = match &cfg.mission {
        MissionConfig::StationKeep { target_m, tolerance_m, .. } => Box::new(StationKeep {
            target_m: *target_m,
            tolerance_m: *tolerance_m,
            duration_s: cfg.simulation.duration_s,
        }),
        MissionConfig::PointToPoint { start_m: _, goal_m, tolerance_m } => Box::new(PointToPoint {
            start_m: cfg.mission.initial_position(),
            goal_m: *goal_m,
            tolerance_m: *tolerance_m,
            timeout_s: cfg.simulation.duration_s,
        }),
    };

    let mut controller = PidStationKeep::new(
        cfg.controller.kp,
        cfg.controller.kv,
        cfg.controller.ki,
        cfg.drone.mass_kg,
    );
    if let Some(v) = cfg.controller.max_horiz_speed_ms {
        controller = controller.with_max_horiz_speed(v);
    }

    let start = cfg.mission.initial_position();
    let initial = DroneState::at_rest(start, cfg.drone.battery_capacity_wh);

    let result = sim_engine::run(
        &wind,
        &drone_model,
        mission.as_ref(),
        &mut controller,
        initial,
        cfg.simulation.dt_s,
        cfg.simulation.duration_s,
    );

    let mean_wind_ms = mean_wind_magnitude(&result);
    tracing::info!(
        "simulation done: steps={}, score(rms error)={:.3} m, terminated={}, mean|wind|={:.3} m/s",
        result.steps_run, result.score, result.terminated_reason, mean_wind_ms
    );

    let output_dir = cli
        .output
        .or_else(|| cfg.simulation.output_dir.as_ref().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("output").join("runs").join(&cli.run_name));

    let meta = RunMetadata {
        steps_run: result.steps_run,
        dt_s: cfg.simulation.dt_s,
        duration_s: cfg.simulation.duration_s,
        score: result.score,
        terminated_reason: result.terminated_reason.to_string(),
        target_m: cfg.mission.target(),
        tolerance_m: cfg.mission.tolerance(),
        mean_wind_ms,
        notes: Some(format!(
            "wind={}, seed={:?}, kp={}, kv={}, ki={}",
            cfg.wind_field.load_from, cfg.seed, cfg.controller.kp, cfg.controller.kv, cfg.controller.ki
        )),
    };
    save_run(&result, &output_dir, &meta)?;
    tracing::info!("wrote {}", output_dir.display());
    Ok(())
}

fn mean_wind_magnitude(result: &sim_engine::SimulationResult) -> f64 {
    if result.wind_at_drone.is_empty() {
        return 0.0;
    }
    let sum: f64 = result
        .wind_at_drone
        .iter()
        .map(|w| (w[0].powi(2) + w[1].powi(2) + w[2].powi(2)).sqrt())
        .sum();
    sum / result.wind_at_drone.len() as f64
}
