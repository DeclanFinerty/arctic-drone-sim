# arctic-drone-sim

Modular simulation framework for drone operations under Arctic wind conditions.

- **Rust** — simulation engine (wind field, drone physics, scenarios).
- **Python** (managed with `uv`) — data ingestion (Environment Canada), processing, and scientific visualization.

See [`CLAUDE.md`](./CLAUDE.md) for architecture, phase plan, and conventions.

## Layout

```
crates/       Rust workspace (wind-field, drone, scenario, sim-engine)
python/       uv-managed Python package (arctic_sim)
configs/      Simulation configs (TOML)
data/         Downloaded + processed input data (gitignored)
output/       Simulation results (gitignored)
```

## Build & run

```bash
# Rust
cargo build --release
cargo run --release -- --config configs/default.toml
cargo test

# Python
cd python
uv sync
uv run python -m arctic_sim.ingest.env_canada --help
uv run jupyter lab
```

## Status

Phase 1 in progress: wind field foundation (Environment Canada ingestion → Mann turbulence model → validation).
