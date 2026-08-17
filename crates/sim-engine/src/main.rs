use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;

use drone::dynamics::PointMassQuad;
use drone::DroneState;
use scenario::controller::{FollowLeader, PidStationKeep};
use scenario::mission::{CourseKind, FollowCourse, PointToPoint, StationKeep, TimeoutMission};
use scenario::{Controller, Mission};
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
    #[serde(default)]
    mission: Option<MissionConfig>,
    #[serde(default)]
    formation: Option<FormationConfig>,
    simulation: SimulationConfig,
}

#[derive(Debug, Deserialize)]
struct FormationConfig {
    leader: LeaderCourseConfig,
    #[serde(default)]
    followers: Vec<FollowerConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum LeaderCourseConfig {
    Figure8 {
        center_m: [f64; 3],
        amplitude_x_m: f64,
        amplitude_y_m: f64,
        period_s: f64,
    },
    Circle {
        center_m: [f64; 3],
        radius_m: f64,
        period_s: f64,
    },
}

impl LeaderCourseConfig {
    fn to_course(&self) -> CourseKind {
        match *self {
            Self::Figure8 { center_m, amplitude_x_m, amplitude_y_m, period_s } =>
                CourseKind::Figure8 { center_m, amplitude_x_m, amplitude_y_m, period_s },
            Self::Circle { center_m, radius_m, period_s } =>
                CourseKind::Circle { center_m, radius_m, period_s },
        }
    }
}

#[derive(Debug, Deserialize)]
struct FollowerConfig {
    name: String,
    offset_m: [f64; 3],
}

#[derive(Debug, Deserialize)]
struct WindFieldConfig {
    #[serde(default = "default_wind_field_dir")]
    load_from: String,
    /// Advect the snapshot past query points at this mean flow (m/s, world
    /// frame). Taylor's frozen-turbulence hypothesis. `[0, 0, 0]` disables.
    #[serde(default = "default_advection")]
    taylor_advection_ms: [f64; 3],
    /// Shift the loaded field's per-component mean by this vector. Enables
    /// mean-wind sweeps over a single saved Mann field.
    #[serde(default)]
    mean_offset_ms: [f64; 3],
    /// Scale turbulent fluctuations about the loaded mean. Enables TI sweeps
    /// over a single saved Mann field. 1.0 = no change.
    #[serde(default = "one")]
    turbulence_scale: f64,
}

fn default_wind_field_dir() -> String {
    "output/wind_fields/mann_anisotropic".to_string()
}

fn default_advection() -> [f64; 3] {
    [6.4, 0.0, 0.0]
}

fn one() -> f64 { 1.0 }

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
    let base_grid = base_grid.transform(
        cfg.wind_field.mean_offset_ms,
        cfg.wind_field.turbulence_scale,
    );
    let (min_c, max_c) = base_grid.domain_bounds();
    tracing::info!(
        "wind field {} loaded: shape {:?}, domain x=[{:.0},{:.0}] y=[{:.0},{:.0}] z=[{:.0},{:.0}]  Taylor advection {:?} m/s  mean_offset {:?}  turbulence_scale {}",
        field_dir.display(),
        base_grid.shape(),
        min_c[0], max_c[0], min_c[1], max_c[1], min_c[2], max_c[2],
        cfg.wind_field.taylor_advection_ms,
        cfg.wind_field.mean_offset_ms,
        cfg.wind_field.turbulence_scale,
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

    let output_dir = cli
        .output
        .or_else(|| cfg.simulation.output_dir.as_ref().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("output").join("runs").join(&cli.run_name));

    if let Some(formation) = &cfg.formation {
        run_formation(&cfg, formation, &wind, &drone_model, &output_dir)?;
    } else {
        let mission_cfg = cfg.mission.as_ref()
            .context("config has neither [mission] nor [formation]")?;
        run_single(&cfg, mission_cfg, &wind, &drone_model, &output_dir)?;
    }
    tracing::info!("wrote {}", output_dir.display());
    Ok(())
}

fn run_single<W>(
    cfg: &Config,
    mission_cfg: &MissionConfig,
    wind: &W,
    drone_model: &PointMassQuad,
    output_dir: &PathBuf,
) -> Result<()>
where
    W: WindFieldQuery + ?Sized,
{
    let mission: Box<dyn Mission> = match mission_cfg {
        MissionConfig::StationKeep { target_m, tolerance_m, .. } => Box::new(StationKeep {
            target_m: *target_m,
            tolerance_m: *tolerance_m,
            duration_s: cfg.simulation.duration_s,
        }),
        MissionConfig::PointToPoint { start_m: _, goal_m, tolerance_m } => Box::new(PointToPoint {
            start_m: mission_cfg.initial_position(),
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

    let start = mission_cfg.initial_position();
    let initial = DroneState::at_rest(start, cfg.drone.battery_capacity_wh);

    let result = sim_engine::run(
        wind, drone_model, mission.as_ref(), &mut controller, initial,
        cfg.simulation.dt_s, cfg.simulation.duration_s,
    );

    let mean_wind_ms = mean_wind_magnitude(&result);
    tracing::info!(
        "simulation done: steps={}, score={:.3} m, terminated={}, mean|wind|={:.3} m/s",
        result.steps_run, result.score, result.terminated_reason, mean_wind_ms
    );

    let meta = RunMetadata {
        steps_run: result.steps_run,
        dt_s: cfg.simulation.dt_s,
        duration_s: cfg.simulation.duration_s,
        score: result.score,
        terminated_reason: result.terminated_reason.to_string(),
        target_m: mission_cfg.target(),
        tolerance_m: mission_cfg.tolerance(),
        mean_wind_ms,
        notes: Some(format!(
            "wind={}, seed={:?}, kp={}, kv={}, ki={}",
            cfg.wind_field.load_from, cfg.seed, cfg.controller.kp, cfg.controller.kv, cfg.controller.ki
        )),
    };
    save_run(&result, output_dir, &meta)?;
    Ok(())
}

fn run_formation<W>(
    cfg: &Config,
    formation: &FormationConfig,
    wind: &W,
    drone_model: &PointMassQuad,
    output_dir: &PathBuf,
) -> Result<()>
where
    W: WindFieldQuery + ?Sized,
{
    let course = formation.leader.to_course();
    let leader_start = course.position_at(0.0);
    tracing::info!(
        "formation: 1 leader + {} followers, duration {} s, leader starts at {:?}",
        formation.followers.len(), cfg.simulation.duration_s, leader_start,
    );

    // Build missions. Leader has FollowCourse; followers have TimeoutMission.
    let leader_mission = FollowCourse {
        course: course.clone(),
        duration_s: cfg.simulation.duration_s,
    };
    let follower_mission = TimeoutMission { duration_s: cfg.simulation.duration_s };

    // Collect boxed missions into a Vec so we can take stable refs to them.
    let mut missions_boxed: Vec<Box<dyn Mission>> = Vec::new();
    missions_boxed.push(Box::new(leader_mission));
    for _ in 0..formation.followers.len() {
        missions_boxed.push(Box::new(follower_mission.clone()));
    }
    let missions_refs: Vec<&dyn Mission> = missions_boxed.iter().map(|m| m.as_ref()).collect();

    // Controllers: leader uses PidStationKeep (targets the moving course
    // position via mission.target(t)); followers use FollowLeader.
    let mut controllers: Vec<Box<dyn Controller>> = Vec::new();
    let mut leader_ctrl = PidStationKeep::new(
        cfg.controller.kp, cfg.controller.kv, cfg.controller.ki, cfg.drone.mass_kg,
    );
    if let Some(v) = cfg.controller.max_horiz_speed_ms {
        leader_ctrl = leader_ctrl.with_max_horiz_speed(v);
    }
    controllers.push(Box::new(leader_ctrl));
    for (i, f) in formation.followers.iter().enumerate() {
        let leader_idx = 0;
        let mut ctrl = FollowLeader::new(
            leader_idx, f.offset_m,
            cfg.controller.kp, cfg.controller.kv, cfg.drone.mass_kg,
        );
        if let Some(v) = cfg.controller.max_horiz_speed_ms {
            ctrl = ctrl.with_max_horiz_speed(v);
        }
        controllers.push(Box::new(ctrl));
        let _ = i;
    }

    // Initial states: leader at course(0), followers at leader(0) + offset.
    let mut initials = Vec::with_capacity(1 + formation.followers.len());
    initials.push(DroneState::at_rest(leader_start, cfg.drone.battery_capacity_wh));
    for f in &formation.followers {
        let p = [
            leader_start[0] + f.offset_m[0],
            leader_start[1] + f.offset_m[1],
            leader_start[2] + f.offset_m[2],
        ];
        initials.push(DroneState::at_rest(p, cfg.drone.battery_capacity_wh));
    }

    let drone_models: Vec<&PointMassQuad> = std::iter::repeat(drone_model)
        .take(1 + formation.followers.len())
        .collect();

    let results = sim_engine::run_multi(
        wind, &drone_models, &missions_refs, &mut controllers, initials,
        cfg.simulation.dt_s, cfg.simulation.duration_s,
    );

    // Write per-drone outputs under <output_dir>/<name>/.
    let names: Vec<String> = std::iter::once("leader".to_string())
        .chain(formation.followers.iter().map(|f| f.name.clone()))
        .collect();
    for (i, result) in results.iter().enumerate() {
        let mean_wind_ms = mean_wind_magnitude(result);
        tracing::info!(
            "  {:>8}: steps={}, score={:.3} m, mean|wind|={:.3} m/s",
            names[i], result.steps_run, result.score, mean_wind_ms,
        );
        let per_dir = output_dir.join(&names[i]);
        let target_at_start = missions_refs[i].target(0.0).unwrap_or([0.0; 3]);
        let meta = RunMetadata {
            steps_run: result.steps_run,
            dt_s: cfg.simulation.dt_s,
            duration_s: cfg.simulation.duration_s,
            score: result.score,
            terminated_reason: result.terminated_reason.to_string(),
            target_m: target_at_start,
            tolerance_m: 0.0,
            mean_wind_ms,
            notes: Some(format!(
                "formation/{}, wind={}, seed={:?}",
                names[i], cfg.wind_field.load_from, cfg.seed,
            )),
        };
        save_run(result, &per_dir, &meta)?;
    }

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
