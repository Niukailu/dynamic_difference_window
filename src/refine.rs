//! Exact endpoint scoring: three local transitions followed by an adjoint suffix.
use crate::{
    engine::{Engine, Matrix},
    model::{Plan, Window},
    run::{self, Options},
};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{path::Path, time::Instant};
struct Message {
    values: Vec<f64>,
    rows: usize,
    cols: usize,
    scale: f64,
}
fn windows(initial: &(Window, Window), schedule: &[Window], completed: usize) -> (Window, Window) {
    let left = if completed == 0 {
        initial.0.clone()
    } else {
        schedule[completed - 1].clone()
    };
    let right = if completed == 0 {
        initial.1.clone()
    } else if completed == 1 {
        initial.0.clone()
    } else {
        schedule[completed - 2].clone()
    };
    (left, right)
}
fn endpoint(prob: &Matrix, left: &Window, right: &Window, point: (u64, u64)) -> Result<f64> {
    match (left.index(point.0), right.index(point.1)) {
        (Some(i), Some(j)) => prob.value(i, j),
        _ => Ok(0.),
    }
}
fn backward(
    engine: &mut Engine,
    initial: &(Window, Window),
    schedule: &[Window],
    point: (u64, u64),
) -> Result<Vec<Message>> {
    let (left, right) = windows(initial, schedule, schedule.len());
    let i = left
        .index(point.0)
        .ok_or_else(|| anyhow::anyhow!("endpoint outside window"))?;
    let j = right
        .index(point.1)
        .ok_or_else(|| anyhow::anyhow!("endpoint outside window"))?;
    let mut prob = engine.point(left.size(), right.size(), i * right.size() + j)?;
    let mut scale = 0.;
    let mut messages = vec![Message {
        values: prob.buffer.download(prob.size())?,
        rows: prob.rows,
        cols: prob.cols,
        scale,
    }];
    for completed in (0..schedule.len()).rev() {
        let (left, right) = windows(initial, schedule, completed);
        prob = engine.adjoint(&prob, &left, &right, &schedule[completed])?;
        let stats = engine.summary(prob.ptr(), prob.size())?;
        ensure!(stats.peak > 0., "endpoint has no retained paths");
        engine.scale(&prob, 1. / stats.peak)?;
        scale += stats.peak.log2();
        messages.push(Message {
            values: prob.buffer.download(prob.size())?,
            rows: prob.rows,
            cols: prob.cols,
            scale,
        });
    }
    messages.reverse();
    Ok(messages)
}
fn score(
    engine: &mut Engine,
    prob: &Matrix,
    state: (&Window, &Window),
    tail: (&[Window], usize),
    target: &Window,
    suffix: Option<(&Matrix, f64)>,
    point: (u64, u64),
) -> Result<f64> {
    let (left, right) = state;
    let (schedule, position) = tail;
    let stop = (position + 3).min(schedule.len());
    let mut value = None;
    let (mut left, mut right) = (left.clone(), right.clone());
    let mut scale = 0.;
    for (index, next) in schedule.iter().enumerate().take(stop).skip(position) {
        let target = if index == position { target } else { next };
        let (next, stats) =
            engine.step(value.as_ref().unwrap_or(prob), &left, &right, target, None)?;
        if stats.peak <= 0. {
            return Ok(f64::NEG_INFINITY);
        }
        engine.scale(&next, 1. / stats.peak)?;
        scale += stats.peak.log2();
        value = Some(next);
        right = left;
        left = target.clone();
    }
    let mass = if let Some((suffix, suffix_scale)) = suffix {
        scale += suffix_scale;
        engine.dot(value.as_ref().unwrap(), suffix)?
    } else {
        endpoint(value.as_ref().unwrap(), &left, &right, point)?
    };
    Ok(if mass > 0. {
        scale + mass.log2()
    } else {
        f64::NEG_INFINITY
    })
}
pub fn execute(
    plan: &Plan,
    device: i32,
    memory: f64,
    host_memory: f64,
    candidates: usize,
    passes: usize,
    output: &Path,
) -> Result<()> {
    ensure!(!output.exists(), "output exists");
    ensure!(
        (1..=64).contains(&candidates) && passes > 0 && host_memory.is_finite() && host_memory > 0.,
        "invalid refinement parameters"
    );
    let initial = plan.config.initial();
    let point = plan.config.physical(plan.endpoint()?);
    let mut schedule = plan.windows.clone();
    let cache_bytes: usize = (0..=schedule.len())
        .map(|i| {
            let (l, r) = windows(&initial, &schedule, i);
            l.size() * r.size() * 8
        })
        .sum();
    ensure!(
        cache_bytes as f64 <= host_memory * 1073741824.,
        "adjoint cache exceeds host memory budget"
    );
    let workspace = 6usize * 8 * (1usize << (2 * plan.config.width));
    ensure!(
        workspace as f64 <= memory * 1073741824.,
        "conservative GPU workspace estimate exceeds budget"
    );
    let started = Instant::now();
    let mut engine = Engine::new(&plan.config, device, memory)?;
    let mut history = vec![];
    for sweep in 0..passes {
        let messages = backward(&mut engine, &initial, &schedule, point)?;
        let before = messages[0].scale + messages[0].values[0].log2();
        let mut prob = engine.point(1, 1, 0)?;
        let (mut left, mut right) = initial.clone();
        let mut prefix_scale = 0.;
        let mut accepted = 0;
        for position in 0..schedule.len() {
            let original = schedule[position].clone();
            let proposed =
                engine.candidates(&prob, &left, &right, plan.config.width, None, candidates)?;
            let mut choices = vec![original.clone()];
            for candidate in proposed {
                if !choices.contains(&candidate) {
                    choices.push(candidate);
                }
            }
            let stop = (position + 3).min(schedule.len());
            let suffix = if stop < schedule.len() {
                let m = &messages[stop];
                Some((engine.from_host(m.rows, m.cols, &m.values)?, m.scale))
            } else {
                None
            };
            let mut best = 0;
            let mut best_score = f64::NEG_INFINITY;
            let mut original_score = 0.;
            for (i, candidate) in choices.iter().enumerate() {
                let current = score(
                    &mut engine,
                    &prob,
                    (&left, &right),
                    (&schedule, position),
                    candidate,
                    suffix.as_ref().map(|(m, s)| (m, *s)),
                    point,
                )?;
                if i == 0 {
                    original_score = current;
                }
                if current > best_score + 1e-12 {
                    best = i;
                    best_score = current;
                }
            }
            let target = choices.swap_remove(best);
            if target != original {
                accepted += 1;
                let change = json!({"sweep":sweep+1,"round":position+1,"before":prefix_scale+original_score,"after":prefix_scale+best_score,"old":original,"new":target});
                println!("{change}");
                history.push(change);
            }
            schedule[position] = target.clone();
            let (next, stats) = engine.step(&prob, &left, &right, &target, None)?;
            ensure!(stats.peak > 0., "empty optimized distribution");
            engine.scale(&next, 1. / stats.peak)?;
            prefix_scale += stats.peak.log2();
            prob = next;
            right = left;
            left = target;
        }
        let after = prefix_scale + endpoint(&prob, &left, &right, point)?.log2();
        ensure!(
            after >= before - 1e-10,
            "refinement decreased endpoint probability"
        );
        let summary = json!({"sweep":sweep+1,"before":before,"after":after,"accepted":accepted});
        println!("{summary}");
        history.push(summary);
        if accepted == 0 {
            break;
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    drop(engine);
    let optimized = Plan {
        config: plan.config.clone(),
        windows: schedule,
        records: vec![],
    };
    let options = Options {
        devices: vec![device],
        memory_gib: memory,
        candidates: 1,
        objective: "mass".into(),
        extend: false,
        expand: false,
    };
    run::execute(&optimized.config, Some(&optimized), &options, output)?;
    let text = std::fs::read_to_string(output)?;
    let mut records: Vec<Value> = text
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    records[0]["optimization"] = json!({"method":"exact endpoint adjoint coordinate ascent","endpoint":plan.endpoint()?,"seconds":seconds,"host_cache_bytes":cache_bytes,"history":history});
    let data = records
        .iter()
        .map(|r| serde_json::to_string(r).unwrap())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    std::fs::write(output, data)?;
    Ok(())
}
