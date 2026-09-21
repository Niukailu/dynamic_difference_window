//! Adjacent-window beam search, optionally maximizing paths outside a reference.
use crate::{
    engine::Engine,
    model::{Plan, Window},
    refine::replay_endpoint,
    run::{self, Options as ReplayOptions},
    tagged::{self, Local, Score, Tagged},
};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{collections::HashMap, path::Path, time::Instant};

pub struct Options {
    pub round: usize,
    pub pool: usize,
    pub device: i32,
    pub memory: f64,
}
fn log_add(a: f64, b: f64) -> f64 {
    let largest = a.max(b);
    if largest.is_finite() {
        largest + (2f64.powf(a - largest) + 2f64.powf(b - largest)).log2()
    } else {
        largest
    }
}
fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() < 1e-9
}
fn candidates(
    engine: &mut Engine,
    local: &Local<'_>,
    limit: usize,
    baseline: f64,
) -> Result<Vec<(Window, f64)>> {
    let original = &local.schedule[local.position];
    let mut scores = HashMap::from([(original.clone(), baseline)]);
    for added in 0..engine.config.word_bits as i32 {
        if original.bits.contains(&added) {
            continue;
        }
        let mut bits = original.bits.clone();
        bits.push(added);
        bits.sort();
        let expanded = Window::new(original.base, bits)?;
        let (halves, scale) = local.halves(engine, &expanded)?;
        let k = expanded.bits.iter().position(|&b| b == added).unwrap();
        let original_score = tagged::log_mass(
            halves[2 * k + ((original.base >> added) & 1) as usize],
            scale,
        );
        ensure!(
            close(original_score, baseline),
            "tagged posterior baseline mismatch"
        );
        for (index, &removed) in expanded.bits.iter().enumerate() {
            for fixed in 0..2 {
                let target = Window::new(
                    expanded.base | ((fixed as u64) << removed),
                    expanded
                        .bits
                        .iter()
                        .copied()
                        .filter(|&b| b != removed)
                        .collect(),
                )?;
                let value = tagged::log_mass(halves[2 * index + fixed], scale);
                scores
                    .entry(target)
                    .and_modify(|v| *v = v.max(value))
                    .or_insert(value);
            }
        }
    }
    let mut sorted: Vec<_> = scores.into_iter().filter(|(w, _)| w != original).collect();
    sorted.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then(a.0.base.cmp(&b.0.base))
            .then(a.0.bits.cmp(&b.0.bits))
    });
    let mut selected = vec![(original.clone(), baseline)];
    // Preserve high scorers and representatives for different newly free bits.
    for candidate in sorted.iter().take((limit - 1) / 2) {
        selected.push(candidate.clone());
    }
    let mut groups = std::collections::HashSet::new();
    for (w, _) in &selected {
        groups.insert(w.mask() & !original.mask());
    }
    for candidate in &sorted {
        if selected.len() == limit {
            break;
        }
        if groups.insert(candidate.0.mask() & !original.mask()) {
            selected.push(candidate.clone());
        }
    }
    for candidate in &sorted {
        if selected.len() == limit {
            break;
        }
        if !selected.iter().any(|(w, _)| w == &candidate.0) {
            selected.push(candidate.clone());
        }
    }
    Ok(selected)
}
fn tail_score(
    engine: &mut Engine,
    first: &Tagged,
    first_window: &Window,
    prior_left: &Window,
    local: &Local<'_>,
    second: &Window,
) -> Result<Score> {
    let mut value = first.step(
        engine,
        first_window,
        prior_left,
        second,
        local.reference.map(|r| &r[local.position + 1]),
    )?;
    let (mut left, mut right) = (second.clone(), first_window.clone());
    for index in local.position + 2..local.stop {
        if !value.scale.is_finite() {
            break;
        }
        let target = &local.schedule[index];
        value = value.step(
            engine,
            &left,
            &right,
            target,
            local.reference.map(|r| &r[index]),
        )?;
        right = left;
        left = target.clone();
    }
    value.score(engine, local.suffix, &left, &right, local.point)
}
pub fn execute(
    plan: &Plan,
    reference: Option<&Plan>,
    options: &Options,
    output: &Path,
) -> Result<()> {
    ensure!(!output.exists(), "output exists");
    ensure!(
        options.round > 0
            && options.round < plan.windows.len()
            && (2..=128).contains(&options.pool),
        "invalid block or pool"
    );
    ensure!(
        options.memory.is_finite() && options.memory > 0.,
        "invalid memory budget"
    );
    let position = options.round - 1;
    ensure!(
        plan.windows[position..position + 2]
            .iter()
            .all(|w| w.bits.len() < 20),
        "candidate expansion exceeds width 20"
    );
    let target = plan.target_endpoint()?;
    if let Some(a) = reference {
        ensure!(
            a.config.cipher == plan.config.cipher
                && a.config.mode == plan.config.mode
                && a.config.word_bits == plan.config.word_bits
                && a.config.left == plan.config.left
                && a.config.right == plan.config.right
                && a.windows.len() == plan.windows.len()
                && a.target_endpoint()? == target,
            "reference must have same cipher/input/rounds/target"
        );
    }
    let mut prior = 1usize;
    let mut largest = 1;
    for w in &plan.windows {
        largest = largest.max(prior * w.size());
        prior = w.size();
    }
    ensure!(
        24. * 8. * largest as f64 <= options.memory * 1073741824.,
        "tagged workspace exceeds conservative budget"
    );
    let start = Instant::now();
    let point = plan.config.physical(target);
    let allowed = reference.map(|p| p.windows.as_slice());
    let stop = (position + 4).min(plan.windows.len());
    let mut engine = Engine::new(&plan.config, options.device, options.memory)?;
    let h = if stop < plan.windows.len() {
        Some(tagged::suffix(
            &mut engine,
            &plan.config,
            &plan.windows,
            allowed,
            point,
            stop,
        )?)
    } else {
        None
    };
    let (mut left, mut right) = plan.config.initial();
    let mut prefix = Tagged::seed(&mut engine, reference.is_some())?;
    for (index, target) in plan.windows.iter().enumerate().take(position) {
        prefix = prefix.step(
            &mut engine,
            &left,
            &right,
            target,
            allowed.map(|r| &r[index]),
        )?;
        right = left;
        left = target.clone();
    }
    let first_original = prefix.step(
        &mut engine,
        &left,
        &right,
        &plan.windows[position],
        allowed.map(|r| &r[position]),
    )?;
    let local = Local {
        prob: &prefix,
        left: &left,
        right: &right,
        schedule: &plan.windows,
        reference: allowed,
        position,
        stop,
        suffix: h.as_ref(),
        point,
    };
    let before = tail_score(
        &mut engine,
        &first_original,
        &plan.windows[position],
        &left,
        &local,
        &plan.windows[position + 1],
    )?;
    ensure!(before.total.is_finite(), "source has no target paths");
    let first_pool = candidates(&mut engine, &local, options.pool, before.novel)?;
    let second_local = Local {
        prob: &first_original,
        left: &plan.windows[position],
        right: &left,
        position: position + 1,
        ..local
    };
    let second_pool = candidates(&mut engine, &second_local, options.pool, before.novel)?;
    let mut best = before;
    let mut choice = (0, 0);
    let mut evaluated = 0;
    for (i, (first, _)) in first_pool.iter().enumerate() {
        let f = prefix.step(
            &mut engine,
            &left,
            &right,
            first,
            allowed.map(|r| &r[position]),
        )?;
        if !f.scale.is_finite() {
            continue;
        }
        for (j, (second, _)) in second_pool.iter().enumerate() {
            let score = tail_score(&mut engine, &f, first, &left, &local, second)?;
            evaluated += 1;
            if score.novel > best.novel + 1e-12 {
                best = score;
                choice = (i, j);
            }
        }
        println!(
            "block round={} first={}/{} evaluated={} target={:.12} novel={:.12}",
            options.round,
            i + 1,
            first_pool.len(),
            evaluated,
            best.total,
            best.novel
        );
    }
    let mut schedule = plan.windows.clone();
    schedule[position] = first_pool[choice.0].0.clone();
    schedule[position + 1] = second_pool[choice.1].0.clone();
    drop(first_original);
    drop(prefix);
    drop(h);
    let verified = tagged::replay(&mut engine, &plan.config, &schedule, allowed, point)?;
    ensure!(
        close(verified.total, best.total) && close(verified.novel, best.novel),
        "full tagged replay mismatch"
    );
    let optimized = Plan {
        config: plan.config.clone(),
        windows: schedule,
        records: vec![],
    };
    let total = replay_endpoint(&mut engine, &optimized, point)?;
    ensure!(close(total, best.total), "ordinary full replay mismatch");
    let overlap = if let Some(a) = reference {
        let common: Option<Vec<_>> = optimized
            .windows
            .iter()
            .zip(&a.windows)
            .map(|(a, b)| a.intersect(b))
            .collect();
        if let Some(windows) = common {
            replay_endpoint(
                &mut engine,
                &Plan {
                    config: plan.config.clone(),
                    windows,
                    records: vec![],
                },
                point,
            )?
        } else {
            f64::NEG_INFINITY
        }
    } else {
        f64::NEG_INFINITY
    };
    // Independent inclusion identity; the search itself never subtracts weights.
    ensure!(
        close(log_add(overlap, best.novel), total),
        "novel plus intersection differs from total"
    );
    let union = if let Some(a) = reference {
        Some(log_add(a.target_log2()?, best.novel))
    } else {
        None
    };
    let seconds = start.elapsed().as_secs_f64();
    drop(engine);
    let replay = ReplayOptions {
        devices: vec![options.device],
        memory_gib: options.memory,
        candidates: 1,
        objective: "mass".into(),
        extend: false,
        expand: false,
    };
    run::execute(&optimized.config, Some(&optimized), &replay, output)?;
    let mut records: Vec<Value> = std::fs::read_to_string(output)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    records[0]["optimization"] = json!({"method":"adjacent two-window search with nonnegative path novelty tags",
        "round":options.round,"pool_limit":options.pool,"candidate_pools":[first_pool,second_pool],
        "selected_indices":[choice.0,choice.1],"candidates_evaluated":evaluated,"seconds":seconds,
        "objective":if reference.is_some(){"new paths outside reference"}else{"endpoint weight"},
        "reference_windows":reference.map(|a|&a.windows),"reference_log2":reference.map(|a|a.target_log2()).transpose()?,
        "before_target":before.total,"after_target":best.total,"before_novel":before.novel,"after_novel":best.novel,
        "log2_intersection":overlap,"log2_union":union,"target":target});
    let last = records
        .iter_mut()
        .rev()
        .find(|r| r.get("round").is_some())
        .unwrap();
    last["target_output"] = json!([format!("{:#x}", target.0), format!("{:#x}", target.1)]);
    last["log2_target"] = json!(best.total);
    let data = records
        .iter()
        .map(serde_json::to_string)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .join("\n")
        + "\n";
    std::fs::write(output, data)?;
    Ok(())
}
