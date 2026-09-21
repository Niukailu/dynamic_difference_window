//! Exhaustive one-bit window swaps using exact endpoint posterior marginals.
//! Enlarging W_k by one bit exposes every one-bit replacement inside that cube.
//! F_k(t,l) H_k(t,l) counts each retained endpoint path once; summing a half cube
//! therefore scores a candidate without separately replaying its three transitions.
use crate::{
    engine::{Engine, Matrix},
    model::{Plan, Window},
    refine::{backward, endpoint, score},
    run::{self, Options},
};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{path::Path, time::Instant};

struct Local<'a> {
    prob: &'a Matrix,
    left: &'a Window,
    right: &'a Window,
    schedule: &'a [Window],
    position: usize,
    suffix: Option<(&'a Matrix, f64)>,
    point: (u64, u64),
}
impl Local<'_> {
    fn halves(&self, engine: &mut Engine, expanded: &Window) -> Result<(Vec<f64>, f64)> {
        let stop = (self.position + 3).min(self.schedule.len());
        let (forward, stats) = engine.step(self.prob, self.left, self.right, expanded, None)?;
        ensure!(stats.peak > 0., "expanded window lost original paths");
        engine.scale(&forward, 1. / stats.peak)?;
        let mut scale = stats.peak.log2();
        // Local states after the changed transition and its two successors.
        let mut states = vec![(expanded.clone(), self.left.clone())];
        for index in self.position + 1..stop {
            states.push((
                self.schedule[index].clone(),
                states.last().unwrap().0.clone(),
            ));
        }
        let mut owned;
        let mut h = if let Some((suffix, suffix_scale)) = self.suffix {
            scale += suffix_scale;
            suffix
        } else {
            let (left, right) = states.last().unwrap();
            let i = left
                .index(self.point.0)
                .ok_or_else(|| anyhow::anyhow!("missing terminal row"))?;
            let j = right
                .index(self.point.1)
                .ok_or_else(|| anyhow::anyhow!("missing terminal column"))?;
            owned = engine.point(left.size(), right.size(), i * right.size() + j)?;
            &owned
        };
        for j in (0..states.len() - 1).rev() {
            let (left, right) = &states[j];
            let next = engine.adjoint(h, left, right, &states[j + 1].0)?;
            let summary = engine.summary(next.ptr(), next.size())?;
            ensure!(summary.peak > 0., "expanded suffix lost original endpoint");
            engine.scale(&next, 1. / summary.peak)?;
            scale += summary.peak.log2();
            owned = next;
            h = &owned;
        }
        Ok((engine.posterior_halves(&forward, h)?, scale))
    }
    fn baseline(&self, engine: &mut Engine, target: &Window) -> Result<f64> {
        score(
            engine,
            self.prob,
            (self.left, self.right),
            (self.schedule, self.position),
            target,
            self.suffix,
            self.point,
        )
    }
}
fn log_weight(value: f64, scale: f64) -> f64 {
    if value > 0. {
        scale + value.log2()
    } else {
        f64::NEG_INFINITY
    }
}

pub fn execute(
    plan: &Plan,
    device: i32,
    memory: f64,
    host_memory: f64,
    passes: usize,
    output: &Path,
) -> Result<()> {
    ensure!(!output.exists(), "output exists");
    ensure!(
        plan.config.width < 20 && passes > 0 && host_memory.is_finite() && host_memory > 0.,
        "invalid posterior refinement parameters"
    );
    let initial = plan.config.initial();
    let point = plan.config.physical(plan.endpoint()?);
    let mut schedule = plan.windows.clone();
    // Later sweeps may grow initially narrow windows up to the configured width.
    let cache_limit = (schedule.len() + 1) as f64 * 8. * 2f64.powi((2 * plan.config.width) as i32);
    ensure!(
        cache_limit <= host_memory * 1073741824.,
        "host budget must cover full-width backward cache"
    );
    ensure!(
        12. * 8. * 2f64.powi((2 * plan.config.width) as i32) <= memory * 1073741824.,
        "expanded posterior workspace exceeds conservative GPU budget"
    );
    let mut engine = Engine::new(&plan.config, device, memory)?;
    let start = Instant::now();
    let mut history = vec![];
    for sweep in 0..passes {
        let messages = backward(&mut engine, &initial, &schedule, point)?;
        let before = messages[0].scale + messages[0].values[0].log2();
        let mut prob = engine.point(1, 1, 0)?;
        let (mut left, mut right) = initial.clone();
        let mut prefix_scale = 0.;
        let mut accepted = 0;
        let mut evaluated = 0;
        for position in 0..schedule.len() {
            let original = schedule[position].clone();
            let stop = (position + 3).min(schedule.len());
            let suffix = if stop < schedule.len() {
                let m = &messages[stop];
                Some((engine.from_host(m.rows, m.cols, &m.values)?, m.scale))
            } else {
                None
            };
            let local = Local {
                prob: &prob,
                left: &left,
                right: &right,
                schedule: &schedule,
                position,
                suffix: suffix.as_ref().map(|(m, s)| (m, *s)),
                point,
            };
            let original_score = local.baseline(&mut engine, &original)?;
            let mut best_score = original_score;
            let mut best = original.clone();
            for added in 0..plan.config.word_bits as i32 {
                if original.bits.contains(&added) {
                    continue;
                }
                let mut bits = original.bits.clone();
                bits.push(added);
                bits.sort();
                let expanded = Window::new(original.base, bits)?;
                let (halves, scale) = local.halves(&mut engine, &expanded)?;
                let added_index = expanded.bits.iter().position(|&bit| bit == added).unwrap();
                let original_half = ((original.base >> added) & 1) as usize;
                let checked = log_weight(halves[2 * added_index + original_half], scale);
                ensure!(
                    (checked - original_score).abs() < 1e-9,
                    "posterior/direct score mismatch at round {}: {checked} vs {original_score}",
                    position + 1
                );
                if expanded.bits.len() <= plan.config.width {
                    let current = log_weight(halves[0] + halves[1], scale);
                    evaluated += 1;
                    if current > best_score + 1e-12 {
                        best_score = current;
                        best = expanded.clone();
                    }
                }
                for (packed, &removed) in expanded.bits.iter().enumerate() {
                    for fixed in 0..2 {
                        let current = log_weight(halves[2 * packed + fixed], scale);
                        evaluated += 1;
                        if current > best_score + 1e-12 {
                            let bits = expanded
                                .bits
                                .iter()
                                .copied()
                                .filter(|&bit| bit != removed)
                                .collect();
                            best = Window::new(expanded.base | ((fixed as u64) << removed), bits)?;
                            best_score = current;
                        }
                    }
                }
            }
            if best != original {
                accepted += 1;
                let change = json!({"sweep":sweep+1,"round":position+1,"before":prefix_scale+original_score,"after":prefix_scale+best_score,"old":original,"new":best});
                println!("{change}");
                history.push(change);
            }
            schedule[position] = best.clone();
            let (next, stats) = engine.step(&prob, &left, &right, &best, None)?;
            ensure!(stats.peak > 0., "empty optimized distribution");
            engine.scale(&next, 1. / stats.peak)?;
            prefix_scale += stats.peak.log2();
            prob = next;
            right = left;
            left = best;
            println!(
                "{}",
                json!({"progress_round":position+1,"sweep":sweep+1,"endpoint_log2":prefix_scale-stats.peak.log2()+best_score})
            );
        }
        let after = prefix_scale + endpoint(&prob, &left, &right, point)?.log2();
        ensure!(
            after >= before - 1e-10,
            "posterior sweep decreased endpoint probability"
        );
        let summary = json!({"sweep":sweep+1,"before":before,"after":after,"accepted":accepted,"candidates_evaluated":evaluated});
        println!("{summary}");
        history.push(summary);
        if accepted == 0 {
            break;
        }
    }
    let seconds = start.elapsed().as_secs_f64();
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
    records[0]["optimization"] = json!({"method":"exact endpoint posterior one-bit swaps","endpoint":plan.endpoint()?,"seconds":seconds,"history":history});
    let data = records
        .iter()
        .map(|r| serde_json::to_string(r).unwrap())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    std::fs::write(output, data)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Config;
    #[test]
    #[ignore = "requires CUDA; set DDW_TEST_DEVICE"]
    fn every_posterior_half_matches_direct_replay() -> Result<()> {
        let device = std::env::var("DDW_TEST_DEVICE")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        for cipher in ["simon", "simeck"] {
            for mode in ["difference", "linear"] {
                let config = Config {
                    cipher: cipher.into(),
                    mode: mode.into(),
                    word_bits: 16,
                    left: 0,
                    right: 1,
                    width: 4,
                    rounds: 4,
                };
                let mut engine = Engine::new(&config, device, 2.)?;
                let left = Window::new(0, vec![0, 1, 2])?;
                let right = Window::new(0, vec![0, 1])?;
                let values: Vec<f64> = (0..left.size() * right.size())
                    .map(|i| (i % 7 + 1) as f64 / 16.)
                    .collect();
                let prob = engine.from_host(left.size(), right.size(), &values)?;
                let expanded = Window::new(0, vec![0, 1, 8, 14, 15])?;
                let full = [
                    Window::new(0, vec![0, 1, 8, 14])?,
                    Window::new(0, vec![0, 1, 2, 3, 8])?,
                    Window::new(0, vec![0, 1, 2, 3, 4, 8])?,
                    Window::point(0),
                ];
                let h: Vec<f64> = (0..full[2].size() * full[1].size())
                    .map(|i| (i % 11 + 1) as f64 / 8.)
                    .collect();
                let suffix = engine.from_host(full[2].size(), full[1].size(), &h)?;
                for count in 1..=4 {
                    let local = Local {
                        prob: &prob,
                        left: &left,
                        right: &right,
                        schedule: &full[..count],
                        position: 0,
                        suffix: if count == 4 {
                            Some((&suffix, 3.))
                        } else {
                            None
                        },
                        point: (0, 0),
                    };
                    let (halves, scale) = local.halves(&mut engine, &expanded)?;
                    for (bit, &physical) in expanded.bits.iter().enumerate() {
                        for fixed in 0..2 {
                            let target = Window::new(
                                (fixed as u64) << physical,
                                expanded
                                    .bits
                                    .iter()
                                    .copied()
                                    .filter(|&b| b != physical)
                                    .collect(),
                            )?;
                            let direct = local.baseline(&mut engine, &target)?;
                            let marginal = log_weight(halves[2 * bit + fixed], scale);
                            assert!(direct==marginal||(direct-marginal).abs()<1e-10,"{cipher}/{mode}, tail={count}, bit={physical}, fixed={fixed}: direct={direct}, posterior={marginal}");
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
