use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "sim-engine", about = "Arctic drone simulation runner")]
struct Cli {
    /// Path to a simulation config TOML.
    #[arg(short, long)]
    config: PathBuf,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    tracing::info!(config = %cli.config.display(), "scaffold: simulation loop not implemented yet");
    Ok(())
}
