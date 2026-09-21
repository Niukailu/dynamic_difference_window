//! Single-GPU fixed replay / nested expansion without materializing dense states.
use crate::{
    engine::{kernel_hash, Engine},
    model::{number, Plan},
};
use anyhow::{ensure, Result};
use serde_json::json;
use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};
pub fn execute(
    plan: &Plan,
    width: Option<usize>,
    rounds: Option<usize>,
    device: i32,
    memory: f64,
    output: &Path,
) -> Result<()> {
    ensure!(!output.exists(), "output exists");
    let mut config = plan.config.clone();
    if let Some(w) = width {
        config.width = w;
    }
    if let Some(r) = rounds {
        config.rounds = r;
    }
    config.validate()?;
    ensure!(config.rounds <= plan.windows.len(), "reference too short");
    ensure!(
        plan.windows
            .iter()
            .take(config.rounds)
            .all(|w| w.bits.len() <= config.width),
        "reference exceeds width"
    );
    let setup = Instant::now();
    let mut engine = Engine::new(&config, device, memory)?;
    let mut prob = engine.compressed_point()?;
    let (mut left, mut right) = config.initial();
    let mut scale = 0.;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?,
    );
    writeln!(
        out,
        "{}",
        json!({"config":config,"backend":"rust-cuda-implicit","device":device,"memory_gib":memory,"kernel_sha256":kernel_hash(),"setup_seconds":setup.elapsed().as_secs_f64(),"selection":if width.is_some(){"nested windows"}else{"fixed replay"}})
    )?;
    out.flush()?;
    let start = Instant::now();
    for round in 0..config.rounds {
        let at = Instant::now();
        let include = &plan.windows[round];
        let target = if width.is_some() {
            engine
                .compressed_candidates(&prob, &left, &right, config.width, Some(include))?
                .remove(0)
        } else {
            include.clone()
        };
        let (next, stats) = engine.compressed_step(&prob, &left, &right, &target)?;
        scale += stats.peak.log2();
        let peak = config.physical((
            target.value(stats.index / left.size()),
            left.value(stats.index % left.size()),
        ));
        let mut row = json!({"round":round+1,"log2_max":scale,"log2_mass":scale+(stats.total/stats.peak).log2(),"output":[format!("{:#x}",peak.0),format!("{:#x}",peak.1)],"window_base":format!("{:#x}",target.base),"window_bits":target.bits,"state_bytes":next.bytes(),"dense_state_bytes":next.rows*next.cols*8,"buckets":next.buckets()});
        if let Some(reference) = plan.records.get(round) {
            let physical = (
                number(reference["output"][0].as_str().unwrap())?,
                number(reference["output"][1].as_str().unwrap())?,
            );
            let point = config.physical(physical);
            let value = match (target.index(point.0), left.index(point.1)) {
                (Some(i), Some(j)) => engine.compressed_value(&next, i, j)?,
                _ => 0.,
            };
            row["reference_output"] = reference["output"].clone();
            row["reference_log2_max"] = reference["log2_max"].clone();
            row["log2_at_reference_output"] = if value > 0. {
                json!(scale + value.log2())
            } else {
                serde_json::Value::Null
            };
        }
        if round + 1 == config.rounds {
            let physical = if let Some(reference) = plan.records.get(round) {
                (
                    number(reference["output"][0].as_str().unwrap())?,
                    number(reference["output"][1].as_str().unwrap())?,
                )
            } else {
                (config.right, config.left)
            };
            let point = config.physical(physical);
            let value = match (target.index(point.0), left.index(point.1)) {
                (Some(i), Some(j)) => engine.compressed_value(&next, i, j)?,
                _ => 0.,
            };
            row["target_output"] =
                json!([format!("{:#x}", physical.0), format!("{:#x}", physical.1)]);
            row["log2_target"] = if value > 0. {
                json!(scale + value.log2())
            } else {
                serde_json::Value::Null
            };
        }
        row["seconds"] = json!(at.elapsed().as_secs_f64());
        writeln!(out, "{row}")?;
        out.flush()?;
        println!(
            "round={} log2_max={scale:.12} state_gib={:.4} dense_gib={:.1} seconds={:.3}",
            round + 1,
            next.bytes() as f64 / 1073741824.,
            (next.rows * next.cols * 8) as f64 / 1073741824.,
            at.elapsed().as_secs_f64()
        );
        prob = next;
        right = left;
        left = target;
    }
    writeln!(
        out,
        "{}",
        json!({"completed_rounds":config.rounds,"search_seconds":start.elapsed().as_secs_f64()})
    )?;
    Ok(())
}
