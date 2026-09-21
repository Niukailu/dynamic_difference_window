//! Exact one-round parity-window search using two disjoint coordinate branches.
//! This tests a larger family at the same state count, without requiring a new
//! transition kernel or claiming support for arbitrary affine window sequences.
use crate::{
    engine::Engine,
    model::{Plan, Window},
    posterior::{log_weight, Local},
    refine::{backward_suffix, replay_endpoint},
    run::{self, Options},
};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{path::Path, time::Instant};

fn branch(expanded: &Window, first: i32, second: i32, fixed: usize) -> Result<Window> {
    Window::new(
        expanded.base | (((fixed & 1) as u64) << first) | (((fixed >> 1) as u64) << second),
        expanded
            .bits
            .iter()
            .copied()
            .filter(|&b| b != first && b != second)
            .collect(),
    )
}
fn sum_logs(a: f64, b: f64) -> f64 {
    let largest = a.max(b);
    if !largest.is_finite() {
        return largest;
    }
    largest + (2f64.powf(a - largest) + 2f64.powf(b - largest)).log2()
}
/// Search every pairwise XOR constraint after adding one coordinate bit.
/// The union has exactly as many states as the original coordinate window.
pub fn execute(plan: &Plan, round: usize, device: i32, memory: f64, output: &Path) -> Result<()> {
    ensure!(!output.exists(), "output exists");
    ensure!(round > 0 && round <= plan.windows.len(), "invalid round");
    ensure!(
        plan.config.width < 20 && memory.is_finite() && memory > 0.,
        "invalid budget"
    );
    let mut previous = plan.config.initial().0.size();
    let mut largest_state = 1usize;
    for window in &plan.windows {
        largest_state = largest_state.max(previous * window.size());
        previous = window.size();
    }
    ensure!(
        16. * 8. * largest_state as f64 <= memory * 1073741824.,
        "expanded affine workspace exceeds budget"
    );
    let position = round - 1;
    let original = &plan.windows[position];
    ensure!(
        !original.bits.is_empty(),
        "parity search needs a nontrivial window"
    );
    let start = Instant::now();
    let point = plan.config.physical(plan.endpoint()?);
    let initial = plan.config.initial();
    let mut engine = Engine::new(&plan.config, device, memory)?;
    let stop = (position + 3).min(plan.windows.len());
    let suffix = if stop < plan.windows.len() {
        Some(backward_suffix(
            &mut engine,
            &initial,
            &plan.windows,
            point,
            stop,
        )?)
    } else {
        None
    };
    let (mut left, mut right) = initial;
    let mut prob = engine.point(1, 1, 0)?;
    let mut prefix_scale = 0.;
    for target in &plan.windows[..position] {
        let (next, stats) = engine.step(&prob, &left, &right, target, None)?;
        ensure!(stats.peak > 0., "empty prefix");
        engine.scale(&next, 1. / stats.peak)?;
        prefix_scale += stats.peak.log2();
        prob = next;
        right = left;
        left = target.clone();
    }
    let local = Local {
        prob: &prob,
        left: &left,
        right: &right,
        schedule: &plan.windows,
        position,
        suffix: suffix.as_ref().map(|(m, s)| (m, *s)),
        point,
    };
    let baseline = local.baseline(&mut engine, original)?;
    let before = prefix_scale + baseline;
    if let Some(reference) = plan.records.last() {
        ensure!(
            (before - reference["log2_max"].as_f64().unwrap()).abs() < 1e-9,
            "baseline replay mismatch"
        );
    }
    let mut best = baseline;
    let mut selected = None;
    let mut evaluated = 0;
    let mut expansion_profile = vec![];
    for added in 0..plan.config.word_bits as i32 {
        if original.bits.contains(&added) {
            continue;
        }
        let mut bits = original.bits.clone();
        bits.push(added);
        bits.sort();
        let expanded = Window::new(original.base, bits)?;
        let (quarters, scale) = local.partition(&mut engine, &expanded, true)?;
        let expanded_score = log_weight(quarters[..4].iter().sum(), scale) + prefix_scale;
        ensure!(
            expanded_score >= before - 1e-10,
            "expanded window lost paths"
        );
        expansion_profile.push(json!({"added_bit":added,"window":expanded,
            "log2_target":expanded_score,"gain_bits":expanded_score-before,
            "state_count":expanded.size()}));
        let mut pair = 0;
        for i in 0..expanded.bits.len() {
            for j in i + 1..expanded.bits.len() {
                for parity in 0..2 {
                    // parity 0 selects 00/11, parity 1 selects 01/10.
                    let fixed = if parity == 0 { [0, 3] } else { [1, 2] };
                    let score = log_weight(
                        quarters[4 * pair + fixed[0]] + quarters[4 * pair + fixed[1]],
                        scale,
                    );
                    evaluated += 1;
                    if score > best + 1e-12 {
                        best = score;
                        selected = Some((
                            expanded.clone(),
                            expanded.bits[i],
                            expanded.bits[j],
                            parity,
                            fixed,
                        ));
                    }
                }
                pair += 1;
            }
        }
        println!(
            "affine round={round} added={added} best={:.12}",
            prefix_scale + best
        );
    }
    let mut branches = vec![];
    let mut relation = Value::Null;
    if let Some((expanded, first, second, parity, fixed)) = &selected {
        relation = json!({"expanded":expanded,"first":first,"second":second,"xor":parity});
        let mut checked = vec![];
        for value in fixed {
            let window = branch(expanded, *first, *second, *value)?;
            let score = local.baseline(&mut engine, &window)?;
            checked.push(score);
            branches.push((window, score + prefix_scale));
        }
        ensure!(
            (sum_logs(checked[0], checked[1]) - best).abs() < 1e-9,
            "parity/direct score mismatch"
        );
        ensure!(
            branches[0].0.intersect(&branches[1].0).is_none(),
            "branches must be disjoint"
        );
        ensure!(
            branches.iter().map(|(w, _)| w.size()).sum::<usize>() == original.size(),
            "state budget changed"
        );
    }
    drop(suffix);
    drop(prob);
    let mut saved = vec![];
    for (index, (window, score)) in branches.iter().enumerate() {
        let path = output.with_extension(format!("branch-{index}.jsonl"));
        ensure!(!path.exists(), "branch output exists");
        let mut windows = plan.windows.clone();
        windows[position] = window.clone();
        let candidate = Plan {
            config: plan.config.clone(),
            windows,
            records: vec![],
        };
        let full = replay_endpoint(&mut engine, &candidate, point)?;
        ensure!(
            full == *score || (full - score).abs() < 1e-9,
            "full endpoint replay mismatch"
        );
        // Empty branches contribute zero and need no ordinary replay (which
        // deliberately rejects empty distributions).
        if full.is_finite() {
            let options = Options {
                devices: vec![device],
                memory_gib: memory,
                candidates: 1,
                objective: "mass".into(),
                extend: false,
                expand: false,
            };
            run::execute(&candidate.config, Some(&candidate), &options, &path)?;
            saved.push(json!({"window":window,"log2_target":full,"path":path}));
        } else {
            saved.push(json!({"window":window,"log2_target":null,"path":null}));
        }
    }
    // Save the best one-bit expansion as a separately replayed candidate. A
    // coordinator can compare these across rounds instead of widening every round.
    let mut best_expansion = Value::Null;
    if let Some(expanded) = expansion_profile.iter().max_by(|a, b| {
        a["log2_target"]
            .as_f64()
            .unwrap()
            .total_cmp(&b["log2_target"].as_f64().unwrap())
    }) {
        let expected = expanded["log2_target"].as_f64().unwrap();
        if expected > before + 1e-12 {
            let window: Window = serde_json::from_value(expanded["window"].clone())?;
            let mut windows = plan.windows.clone();
            windows[position] = window.clone();
            let mut config = plan.config.clone();
            config.width = config.width.max(window.bits.len());
            let candidate = Plan {
                config,
                windows,
                records: vec![],
            };
            let full = replay_endpoint(&mut engine, &candidate, point)?;
            ensure!(
                (full - expected).abs() < 1e-9,
                "expansion full replay mismatch"
            );
            let path = output.with_extension("expanded.jsonl");
            let options = Options {
                devices: vec![device],
                memory_gib: memory,
                candidates: 1,
                objective: "mass".into(),
                extend: false,
                expand: false,
            };
            run::execute(&candidate.config, Some(&candidate), &options, &path)?;
            best_expansion = json!({"path":path,"window":window,"log2_target":full,
                "added_bit":expanded["added_bit"]});
        }
    }
    let summary = json!({"method":"equal-cardinality single-round pairwise-XOR window",
        "config":plan.config,"round":round,"device":device,"endpoint":plan.endpoint()?,
        "before":before,"after":prefix_scale+best,"improved":selected.is_some(),
        "old_window":original,"state_count":original.size(),"relation":relation,
        "branches":saved,"disjoint":true,"candidates_evaluated":evaluated,
        "expansion_profile":expansion_profile,
        "best_expansion":best_expansion,
        "seconds":start.elapsed().as_secs_f64()});
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    writeln!(file, "{}", serde_json::to_string_pretty(&summary)?)?;
    Ok(())
}
