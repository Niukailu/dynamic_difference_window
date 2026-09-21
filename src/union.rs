//! Reflection preserves path weights for endpoints that swap the input branches.
//! Intersect each scalar window, replay that intersection, then apply 2P(A)-P(I).
use crate::{
    compressed,
    model::{Config, Plan, Window},
};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
pub fn execute(
    source: &Path,
    rounds: Option<usize>,
    device: i32,
    memory: f64,
    output: &Path,
) -> Result<()> {
    ensure!(!output.exists(), "output exists");
    let mut plan = Plan::read(source)?;
    if let Some(r) = rounds {
        ensure!(r >= 2 && r <= plan.windows.len(), "invalid round count");
        plan.windows.truncate(r);
        plan.records.truncate(r);
        plan.config.rounds = r;
    }
    ensure!(plan.windows.len() >= 2, "at least two rounds required");
    ensure!(
        plan.endpoint()? == (plan.config.right, plan.config.left),
        "reflection requires swapped endpoint"
    );
    let a = plan.records.last().unwrap()["log2_max"].as_f64().unwrap();
    let initial = plan.config.initial();
    let mut reverse = vec![initial.1, initial.0];
    reverse.extend_from_slice(&plan.windows[..plan.windows.len() - 2]);
    reverse.reverse();
    let common: Option<Vec<_>> = plan
        .windows
        .iter()
        .zip(&reverse)
        .map(|(a, b)| a.intersect(b))
        .collect();
    let intersection_path = output.with_extension("intersection.jsonl");
    ensure!(!intersection_path.exists(), "intersection output exists");
    let start = Instant::now();
    let intersection = if let Some(windows) = common {
        evaluate(&plan.config, windows, &intersection_path, device, memory)?
    } else {
        None
    };
    let overlap = intersection.map(|i| 2f64.powf(i - a)).unwrap_or(0.);
    ensure!(
        overlap <= 1. + 1e-10,
        "intersection exceeds constituent probability"
    );
    let probability = a + (2. - overlap.min(1.)).log2();
    let result = json!({"config":plan.config,"reference":source,"rounds":plan.config.rounds,"endpoint":plan.endpoint()?,"method":"2P(A)-P(A intersection reflected(A)); exact virtual FP64 replay","log2_a":a,"log2_intersection":intersection,"log2_union":probability,"threshold_log2":-2.*plan.config.word_bits as f64,"above_threshold":probability > -2.*plan.config.word_bits as f64,"intersection_file":if intersection_path.exists(){Some(&intersection_path)}else{None},"seconds":start.elapsed().as_secs_f64()});
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    writeln!(out, "{}", serde_json::to_string_pretty(&result)?)?;
    println!("{result}");
    Ok(())
}

fn evaluate(
    config: &Config,
    windows: Vec<Window>,
    path: &Path,
    device: i32,
    memory: f64,
) -> Result<Option<f64>> {
    ensure!(!path.exists(), "intersection output exists");
    let value = {
        let subset = Plan {
            config: config.clone(),
            windows,
            records: vec![],
        };
        let empty = match compressed::execute(&subset, None, None, device, memory, path) {
            Ok(()) => false,
            Err(error)
                if error
                    .downcast_ref::<crate::engine::EmptyDistribution>()
                    .is_some() =>
            {
                true
            }
            Err(error) => return Err(error),
        };
        if empty {
            let mut log = OpenOptions::new().append(true).open(path)?;
            writeln!(
                log,
                "{}",
                json!({"empty_intersection":true,"log2_target":null})
            )?;
            None
        } else {
            let text = std::fs::read_to_string(path)?;
            let data: Vec<Value> = text
                .lines()
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?;
            let last = data.iter().rfind(|r| r.get("round").is_some()).unwrap();
            last["log2_target"].as_f64()
        }
    };
    Ok(value)
}

pub fn multiple(
    sources: &[PathBuf],
    reflect: bool,
    device: i32,
    memory: f64,
    output: &Path,
) -> Result<()> {
    ensure!(!output.exists(), "output exists");
    ensure!(
        (1..=4).contains(&(sources.len() * if reflect { 2 } else { 1 })),
        "requires 1..4 total path sets"
    );
    let plans: Vec<_> = sources
        .iter()
        .map(|p| Plan::read(p))
        .collect::<Result<_>>()?;
    let mut config = plans[0].config.clone();
    let mut sets = vec![];
    let mut logs = vec![];
    for plan in &plans {
        let c = &plan.config;
        ensure!(
            c.rounds == config.rounds
                && c.rounds >= 2
                && c.left == config.left
                && c.right == config.right
                && c.word_bits == config.word_bits
                && c.mode == config.mode
                && c.cipher == config.cipher,
            "incompatible plans"
        );
        ensure!(
            plan.endpoint()? == (c.right, c.left),
            "current union command requires swapped endpoints"
        );
        config.width = config.width.max(c.width);
        sets.push(plan.windows.clone());
        logs.push(plan.records.last().unwrap()["log2_max"].as_f64().unwrap());
        if reflect {
            let initial = c.initial();
            let mut reversed = vec![initial.1, initial.0];
            reversed.extend_from_slice(&plan.windows[..c.rounds - 2]);
            reversed.reverse();
            sets.push(reversed);
            logs.push(*logs.last().unwrap());
        }
    }
    let started = Instant::now();
    let mut cache: HashMap<Vec<Window>, (Option<f64>, PathBuf)> = HashMap::new();
    let mut terms = vec![];
    for mask in 1usize..1usize << sets.len() {
        let indices: Vec<_> = (0..sets.len()).filter(|i| mask & (1 << i) != 0).collect();
        let mut file = None;
        let probability = if indices.len() == 1 {
            Some(logs[indices[0]])
        } else {
            let mut common = Some(sets[indices[0]].clone());
            for &index in &indices[1..] {
                common = common.and_then(|v| {
                    v.iter()
                        .zip(&sets[index])
                        .map(|(a, b)| a.intersect(b))
                        .collect::<Option<Vec<_>>>()
                });
            }
            if let Some(windows) = common {
                if let Some((value, path)) = cache.get(&windows) {
                    file = Some(path.clone());
                    *value
                } else {
                    let path = output.with_extension(format!("intersection-{mask:02}.jsonl"));
                    let value = evaluate(&config, windows.clone(), &path, device, memory)?;
                    file = Some(path.clone());
                    cache.insert(windows, (value, path));
                    value
                }
            } else {
                None
            }
        };
        if let Some(value) = probability {
            ensure!(
                indices.iter().all(|&i| value <= logs[i] + 1e-10),
                "intersection exceeds constituent"
            );
        }
        let term = json!({"subset":indices,"sign":if mask.count_ones()%2==1{1}else{-1},"log2_probability":probability,"intersection_file":file});
        println!("{term}");
        terms.push(term);
    }
    let offset = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.;
    let mut correction = 0.;
    for term in &terms {
        if let Some(value) = term["log2_probability"].as_f64() {
            let add = term["sign"].as_i64().unwrap() as f64 * 2f64.powf(value - offset);
            let adjusted = add - correction;
            let next = sum + adjusted;
            correction = (next - sum) - adjusted;
            sum = next;
        }
    }
    ensure!(sum >= 1. - 1e-10, "union lost constituent mass");
    let log = offset + sum.log2();
    let result = json!({"config":config,"references":sources,"reflected":reflect,"rounds":config.rounds,"endpoint":[config.right,config.left],"terms":terms,"method":"exact path-set inclusion-exclusion with virtual FP64 replay","log2_union":log,"threshold_log2":-2.*config.word_bits as f64,"above_threshold":log > -2.*config.word_bits as f64,"seconds":started.elapsed().as_secs_f64()});
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    writeln!(out, "{}", serde_json::to_string_pretty(&result)?)?;
    println!("{result}");
    Ok(())
}
