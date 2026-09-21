//! Nonnegative dynamic programming for paths outside one reference schedule.
use crate::{
    engine::{Engine, Matrix},
    model::{Config, Window},
    refine::{endpoint, windows},
};
use anyhow::{ensure, Result};

pub(crate) fn log_mass(value: f64, scale: f64) -> f64 {
    if value > 0. {
        scale + value.log2()
    } else {
        f64::NEG_INFINITY
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Score {
    pub total: f64,
    pub novel: f64,
}
pub(crate) struct Tagged {
    pub inside: Matrix,
    pub outside: Matrix,
    pub scale: f64,
}
impl Tagged {
    fn normalized(
        engine: &mut Engine,
        inside: Matrix,
        outside: Matrix,
        scale: f64,
    ) -> Result<Self> {
        let a = engine.summary(inside.ptr(), inside.size())?.peak;
        let b = engine.summary(outside.ptr(), outside.size())?.peak;
        let peak = a.max(b);
        if peak > 0. {
            engine.scale(&inside, 1. / peak)?;
            engine.scale(&outside, 1. / peak)?;
        }
        Ok(Self {
            inside,
            outside,
            scale: log_mass(peak, scale),
        })
    }
    pub fn seed(engine: &mut Engine, reference: bool) -> Result<Self> {
        let one = engine.point(1, 1, 0)?;
        let zero = engine.zeros(1, 1)?;
        let (inside, outside) = if reference { (one, zero) } else { (zero, one) };
        Ok(Self {
            inside,
            outside,
            scale: 0.,
        })
    }
    pub fn terminal(
        engine: &mut Engine,
        left: &Window,
        right: &Window,
        point: (u64, u64),
    ) -> Result<Self> {
        let i = left
            .index(point.0)
            .ok_or_else(|| anyhow::anyhow!("endpoint outside terminal row"))?;
        let j = right
            .index(point.1)
            .ok_or_else(|| anyhow::anyhow!("endpoint outside terminal column"))?;
        Ok(Self {
            inside: engine.zeros(left.size(), right.size())?,
            outside: engine.point(left.size(), right.size(), i * right.size() + j)?,
            scale: 0.,
        })
    }
    pub fn step(
        &self,
        engine: &mut Engine,
        left: &Window,
        right: &Window,
        target: &Window,
        allowed: Option<&Window>,
    ) -> Result<Self> {
        let (inside, _) = engine.step(&self.inside, left, right, target, None)?;
        let (outside, _) = engine.step(&self.outside, left, right, target, None)?;
        engine.novelty_route(&inside, &outside, target, allowed)?;
        Self::normalized(engine, inside, outside, self.scale)
    }
    pub fn backward(
        &self,
        engine: &mut Engine,
        left: &Window,
        right: &Window,
        target: &Window,
        allowed: Option<&Window>,
    ) -> Result<Self> {
        let mixed = engine.novelty_mix(&self.inside, &self.outside, target, allowed)?;
        let inside = engine.adjoint(&mixed, left, right, target)?;
        let outside = engine.adjoint(&self.outside, left, right, target)?;
        Self::normalized(engine, inside, outside, self.scale)
    }
    pub fn score(
        &self,
        engine: &mut Engine,
        suffix: Option<&Self>,
        left: &Window,
        right: &Window,
        point: (u64, u64),
    ) -> Result<Score> {
        if !self.scale.is_finite() {
            return Ok(Score {
                total: f64::NEG_INFINITY,
                novel: f64::NEG_INFINITY,
            });
        }
        let (total, novel, scale) = if let Some(h) = suffix {
            let outside = engine.dot(&self.outside, &h.outside)?;
            (
                engine.dot(&self.inside, &h.outside)? + outside,
                engine.dot(&self.inside, &h.inside)? + outside,
                self.scale + h.scale,
            )
        } else {
            let outside = endpoint(&self.outside, left, right, point)?;
            (
                endpoint(&self.inside, left, right, point)? + outside,
                outside,
                self.scale,
            )
        };
        Ok(Score {
            total: log_mass(total, scale),
            novel: log_mass(novel, scale),
        })
    }
}
pub(crate) fn suffix(
    engine: &mut Engine,
    config: &Config,
    schedule: &[Window],
    reference: Option<&[Window]>,
    point: (u64, u64),
    stop: usize,
) -> Result<Tagged> {
    let initial = config.initial();
    let (left, right) = windows(&initial, schedule, schedule.len());
    let mut h = Tagged::terminal(engine, &left, &right, point)?;
    for index in (stop..schedule.len()).rev() {
        let (left, right) = windows(&initial, schedule, index);
        h = h.backward(
            engine,
            &left,
            &right,
            &schedule[index],
            reference.map(|s| &s[index]),
        )?;
    }
    Ok(h)
}
pub(crate) fn replay(
    engine: &mut Engine,
    config: &Config,
    schedule: &[Window],
    reference: Option<&[Window]>,
    point: (u64, u64),
) -> Result<Score> {
    let (mut left, mut right) = config.initial();
    let mut p = Tagged::seed(engine, reference.is_some())?;
    for (index, target) in schedule.iter().enumerate() {
        p = p.step(engine, &left, &right, target, reference.map(|s| &s[index]))?;
        if !p.scale.is_finite() {
            break;
        }
        right = left;
        left = target.clone();
    }
    p.score(engine, None, &left, &right, point)
}
/// Posterior at one expanded window. All later local windows remain fixed.
pub(crate) struct Local<'a> {
    pub prob: &'a Tagged,
    pub left: &'a Window,
    pub right: &'a Window,
    pub schedule: &'a [Window],
    pub reference: Option<&'a [Window]>,
    pub position: usize,
    pub stop: usize,
    pub suffix: Option<&'a Tagged>,
    pub point: (u64, u64),
}
impl Local<'_> {
    pub fn halves(&self, engine: &mut Engine, expanded: &Window) -> Result<(Vec<f64>, f64)> {
        ensure!(
            self.position < self.stop && self.stop <= self.schedule.len(),
            "invalid local suffix"
        );
        let f = self.prob.step(
            engine,
            self.left,
            self.right,
            expanded,
            self.reference.map(|s| &s[self.position]),
        )?;
        ensure!(f.scale.is_finite(), "expanded prefix empty");
        let mut states = vec![(expanded.clone(), self.left.clone())];
        for index in self.position + 1..self.stop {
            states.push((
                self.schedule[index].clone(),
                states.last().unwrap().0.clone(),
            ));
        }
        let mut owned;
        let mut h = if let Some(h) = self.suffix {
            h
        } else {
            let (left, right) = states.last().unwrap();
            owned = Tagged::terminal(engine, left, right, self.point)?;
            &owned
        };
        for j in (0..states.len() - 1).rev() {
            let (left, right) = &states[j];
            let next = h.backward(
                engine,
                left,
                right,
                &states[j + 1].0,
                self.reference.map(|s| &s[self.position + j + 1]),
            )?;
            owned = next;
            h = &owned;
        }
        let masses =
            engine.posterior_tagged_halves((&f.inside, &f.outside), (&h.inside, &h.outside))?;
        Ok((masses, f.scale + h.scale))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires CUDA; set DDW_TEST_DEVICE"]
    fn tags_and_backward_match_projected_distribution() -> Result<()> {
        let device = std::env::var("DDW_TEST_DEVICE")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        let mut mixed = false;
        for cipher in ["simon", "simeck"] {
            for mode in ["difference", "linear"] {
                let config = Config {
                    cipher: cipher.into(),
                    mode: mode.into(),
                    word_bits: 16,
                    left: 1,
                    right: 1,
                    width: 5,
                    rounds: 4,
                };
                let mut engine = Engine::new(&config, device, 2.)?;
                // A tiny novel mass must survive even when total-overlap rounds
                // to zero in FP64. The production search never subtracts them.
                let rare = Tagged {
                    inside: engine.from_host(1, 1, &[0.5])?,
                    outside: engine.from_host(1, 1, &[2f64.powi(-80)])?,
                    scale: 0.,
                };
                let rare_score = rare.score(
                    &mut engine,
                    None,
                    &Window::point(0),
                    &Window::point(0),
                    (0, 0),
                )?;
                assert_eq!(rare_score.total, -1.);
                assert_eq!(rare_score.novel, -80.);
                let (mut left, mut right) = config.initial();
                let mut dense = engine.point(1, 1, 0)?;
                let mut common = engine.point(1, 1, 0)?;
                let mut prefixes = vec![Tagged::seed(&mut engine, true)?];
                let mut schedule = vec![];
                let mut reference = vec![];
                let mut point = (0, 0);
                for index in 0..4 {
                    let target = engine
                        .candidates(&dense, &left, &right, 5, None, 1)?
                        .remove(0);
                    let (next, stats) = engine.step(&dense, &left, &right, &target, None)?;
                    let (intersection, _) = engine.step(&common, &left, &right, &target, None)?;
                    let allowed = if index == 0 {
                        let bit = target.bits[0];
                        Window::new(
                            target.base,
                            target.bits.iter().copied().filter(|&b| b != bit).collect(),
                        )?
                    } else {
                        target.clone()
                    };
                    let mut expected_inside =
                        intersection.buffer.download::<f64>(intersection.size())?;
                    for row in 0..target.size() {
                        if allowed.index(target.value(row)).is_none() {
                            expected_inside[row * left.size()..(row + 1) * left.size()].fill(0.);
                        }
                    }
                    let tag = prefixes.last().unwrap().step(
                        &mut engine,
                        &left,
                        &right,
                        &target,
                        Some(&allowed),
                    )?;
                    let inside = tag.inside.buffer.download::<f64>(tag.inside.size())?;
                    let outside = tag.outside.buffer.download::<f64>(tag.outside.size())?;
                    let total = next.buffer.download::<f64>(next.size())?;
                    let scale = 2f64.powf(tag.scale);
                    for ((&a, &b), (&want, &all)) in inside
                        .iter()
                        .zip(&outside)
                        .zip(expected_inside.iter().zip(&total))
                    {
                        assert!((a * scale - want).abs() < 1e-11);
                        assert!(((a + b) * scale - all).abs() < 1e-11);
                        assert!(b >= 0.);
                    }
                    mixed |= inside.iter().any(|&p| p > 0.) && outside.iter().any(|&p| p > 0.);
                    common = engine.from_host(target.size(), left.size(), &expected_inside)?;
                    point = (
                        target.value(stats.index / left.size()),
                        left.value(stats.index % left.size()),
                    );
                    dense = next;
                    prefixes.push(tag);
                    schedule.push(target.clone());
                    reference.push(allowed);
                    right = left;
                    left = target;
                }
                let expected =
                    prefixes
                        .last()
                        .unwrap()
                        .score(&mut engine, None, &left, &right, point)?;
                for (stop, prefix) in prefixes.iter().enumerate() {
                    let h = suffix(
                        &mut engine,
                        &config,
                        &schedule,
                        Some(&reference),
                        point,
                        stop,
                    )?;
                    let (left, right) = windows(&config.initial(), &schedule, stop);
                    let actual = prefix.score(&mut engine, Some(&h), &left, &right, point)?;
                    assert!((actual.total - expected.total).abs() < 1e-10);
                    assert!(
                        actual.novel == expected.novel
                            || (actual.novel - expected.novel).abs() < 1e-10
                    );
                }
            }
        }
        assert!(mixed, "fixture must exercise both path tags");
        Ok(())
    }
}
