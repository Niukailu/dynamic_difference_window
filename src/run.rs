use crate::{
    engine::{kernel_hash, Matrix, Stats},
    model::{number, Config, Plan},
    multi::Cluster,
};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};
pub enum State {
    Single(Matrix),
    Shards(Vec<Matrix>),
}
impl State {
    pub fn value(&self, row: usize, col: usize) -> Result<f64> {
        match self {
            Self::Single(m) => m.value(row, col),
            Self::Shards(m) => m[row / m[0].rows].value(row % m[0].rows, col),
        }
    }
}
pub struct Options {
    pub devices: Vec<i32>,
    pub memory_gib: f64,
    pub candidates: usize,
    pub objective: String,
    pub extend: bool,
    pub expand: bool,
}
pub fn execute(
    config: &Config,
    plan: Option<&Plan>,
    options: &Options,
    output: &Path,
) -> Result<()> {
    config.validate()?;
    ensure!(!output.exists(), "output already exists");
    ensure!(
        (1..=64).contains(&options.candidates),
        "candidates must be 1..64"
    );
    ensure!(
        ["mass", "peak"].contains(&options.objective.as_str()),
        "objective must be mass or peak"
    );
    if let Some(plan) = plan {
        ensure!(
            config.rounds <= plan.windows.len() || options.extend,
            "reference does not cover requested rounds; use --extend"
        );
        ensure!(
            plan.windows
                .iter()
                .take(config.rounds)
                .all(|w| w.bits.len() <= config.width),
            "reference window exceeds width"
        );
    }
    let setup = Instant::now();
    let mut cluster = Cluster::new(config, &options.devices, options.memory_gib)?;
    let (mut left, mut right) = config.initial();
    let mut state = State::Single(cluster.engines[0].point(1, 1, 0)?);
    let mut spare = None;
    let mut scale = 0.;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?,
    );
    writeln!(
        writer,
        "{}",
        json!({
            "config": config,
            "backend": "rust-cuda",
            "version": env!("CARGO_PKG_VERSION"),
            "devices": options.devices,
            "memory_gib": options.memory_gib,
            "kernel_sha256": kernel_hash(),
            "setup_seconds": setup.elapsed().as_secs_f64(),
            "selection": if options.expand { "nested windows" }
                else if plan.is_some() { "fixed replay then optional continuation" }
                else { "greedy candidates" },
            "candidates": options.candidates,
            "objective": options.objective,
        })
    )?;
    writer.flush()?;
    let all = Instant::now();
    for round in 1..=config.rounds {
        let start = Instant::now();
        let reference = plan.and_then(|p| p.records.get(round - 1));
        let include = plan.and_then(|p| p.windows.get(round - 1));
        let mut candidates = if let Some(target) = include.filter(|_| !options.expand) {
            vec![target.clone()]
        } else {
            match &state {
                State::Single(m) => cluster.engines[0].candidates(
                    m,
                    &left,
                    &right,
                    config.width,
                    include,
                    options.candidates,
                )?,
                State::Shards(m) => cluster.candidates(
                    m,
                    &left,
                    &right,
                    config.width,
                    include,
                    options.candidates,
                )?,
            }
        };
        let mut best = 0;
        if candidates.len() > 1 {
            let scores = match &state {
                State::Single(m) => {
                    let mut values = vec![];
                    for target in &candidates {
                        values.push(if options.objective == "mass" {
                            cluster.engines[0].retained_mass(m, &left, &right, target)?
                        } else {
                            let (_, s) = cluster.engines[0].step(m, &left, &right, target, None)?;
                            s.peak
                        });
                    }
                    values
                }
                State::Shards(m) => {
                    if options.objective == "mass" {
                        cluster.masses(m, &left, &right, &candidates)?
                    } else {
                        let mut values = vec![];
                        for target in &candidates {
                            values.push(cluster.compute(m, &left, &right, target)?.peak);
                        }
                        values
                    }
                }
            };
            for i in 1..scores.len() {
                if scores[i] > scores[best] {
                    best = i;
                }
            }
        }
        let evaluated = candidates.len();
        let target = candidates.swap_remove(best);
        let selection = start.elapsed().as_secs_f64();
        if let State::Single(prob) = &state {
            if options.devices.len() > 1
                && left.size() >= options.devices.len()
                && target.size() >= options.devices.len()
            {
                state = State::Shards(cluster.split(prob)?);
                spare = None;
            }
        }
        let compute = Instant::now();
        let stats: Stats;
        let mut exchange = 0.;
        state = match state {
            State::Single(prob) => {
                let (next, s) =
                    cluster.engines[0].step(&prob, &left, &right, &target, spare.take())?;
                ensure!(s.peak > 0., "no paths remain at round {round}");
                cluster.engines[0].scale(&next, 1. / s.peak)?;
                cluster.engines[0].context.sync()?;
                spare = Some(prob);
                stats = s;
                State::Single(next)
            }
            State::Shards(prob) => {
                stats = cluster.compute(&prob, &left, &right, &target)?;
                ensure!(stats.peak > 0., "no paths remain at round {round}");
                let started = Instant::now();
                let next = if target.size() < options.devices.len() {
                    State::Single(cluster.collapse(&left, &target, stats.peak)?)
                } else {
                    State::Shards(cluster.exchange(prob, &left, &target, stats.peak)?)
                };
                exchange = started.elapsed().as_secs_f64();
                next
            }
        };
        scale += stats.peak.log2();
        let endpoint = config.physical((
            target.value(stats.index / left.size()),
            left.value(stats.index % left.size()),
        ));
        let mut record = json!({
            "round": round,
            "log2_max": scale,
            "log2_mass": scale + (stats.total / stats.peak).log2(),
            "output": [format!("{:#x}", endpoint.0), format!("{:#x}", endpoint.1)],
            "window_base": format!("{:#x}", target.base),
            "window_bits": target.bits,
            "selection_seconds": selection,
            "compute_seconds": compute.elapsed().as_secs_f64() - exchange,
            "exchange_seconds": exchange,
            "active_devices": match &state { State::Single(_) => 1, State::Shards(m) => m.len() },
            "candidates_evaluated": evaluated,
        });
        if let Some(reference) = reference {
            let physical = (
                number(reference["output"][0].as_str().unwrap())?,
                number(reference["output"][1].as_str().unwrap())?,
            );
            let internal = config.physical(physical);
            let value = match (target.index(internal.0), left.index(internal.1)) {
                (Some(i), Some(j)) => state.value(i, j)?,
                _ => 0.,
            };
            record["reference_output"] = reference["output"].clone();
            record["reference_log2_max"] = reference["log2_max"].clone();
            record["log2_at_reference_output"] = if value > 0. {
                json!(scale + value.log2())
            } else {
                Value::Null
            };
        }
        record["seconds"] = json!(start.elapsed().as_secs_f64());
        writeln!(writer, "{record}")?;
        writer.flush()?;
        println!(
            "round={round} log2_max={scale:.12} seconds={:.4}",
            start.elapsed().as_secs_f64()
        );
        right = left;
        left = target;
    }
    writeln!(
        writer,
        "{}",
        json!({"completed_rounds":config.rounds,"search_seconds":all.elapsed().as_secs_f64()})
    )?;
    Ok(())
}
pub fn validate(path: &Path, reference: Option<&Path>, tolerance: f64) -> Result<Value> {
    ensure!(
        tolerance.is_finite() && tolerance >= 0.,
        "invalid tolerance"
    );
    let plan = Plan::read(path)?;
    let mut prior = 0.;
    for row in &plan.records {
        let peak = row["log2_max"]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("missing peak"))?;
        ensure!(peak.is_finite() && peak <= 1e-10, "invalid peak");
        if let Some(mass) = row["log2_mass"].as_f64() {
            ensure!(
                mass <= prior + 1e-10 && mass >= peak - 1e-10,
                "probability mass violation"
            );
            prior = mass;
        }
        if row.get("log2_at_reference_output").is_some() {
            let value = row["log2_at_reference_output"]
                .as_f64()
                .ok_or_else(|| anyhow::anyhow!("lost reference endpoint"))?;
            ensure!(
                value
                    >= row["reference_log2_max"]
                        .as_f64()
                        .ok_or_else(|| anyhow::anyhow!("missing reference peak"))?
                        - tolerance,
                "reference endpoint probability decreased"
            );
        }
    }
    let mut error = 0.;
    if let Some(reference) = reference {
        let other = Plan::read(reference)?;
        ensure!(
            plan.config.cipher == other.config.cipher
                && plan.config.mode == other.config.mode
                && plan.config.word_bits == other.config.word_bits
                && plan.config.left == other.config.left
                && plan.config.right == other.config.right,
            "incompatible references"
        );
        ensure!(
            plan.records.len() <= other.records.len(),
            "reference too short"
        );
        for (a, b) in plan.records.iter().zip(&other.records) {
            error = f64::max(
                error,
                (a["log2_max"].as_f64().unwrap() - b["log2_max"].as_f64().unwrap()).abs(),
            );
        }
        ensure!(
            error <= tolerance,
            "max log2 error {error} exceeds {tolerance}"
        );
    }
    Ok(
        json!({"rounds":plan.records.len(),"max_log2_error":if reference.is_some(){Some(error)}else{None}}),
    )
}
