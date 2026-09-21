use crate::{
    engine::{select, Engine, Matrix, Stats},
    model::{Config, Window},
};
use anyhow::{ensure, Result};
use std::thread;
pub struct Cluster {
    pub engines: Vec<Engine>,
    outputs: Vec<Option<Matrix>>,
}
impl Cluster {
    pub fn new(config: &Config, devices: &[i32], memory: f64) -> Result<Self> {
        ensure!(
            !devices.is_empty() && devices.len().is_power_of_two(),
            "device count must be a power of two"
        );
        let mut unique = devices.to_vec();
        unique.sort();
        unique.dedup();
        ensure!(unique.len() == devices.len(), "duplicate devices");
        let engines = thread::scope(|scope| {
            let handles: Vec<_> = devices
                .iter()
                .map(|&device| scope.spawn(move || Engine::new(config, device, memory)))
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .map_err(|_| anyhow::anyhow!("GPU initialization thread panicked"))?
                })
                .collect::<Result<Vec<_>>>()
        })?;
        for engine in &engines {
            for peer in &engines {
                engine.context.peer(&peer.context)?;
            }
        }
        let outputs = (0..devices.len()).map(|_| None).collect();
        Ok(Self { engines, outputs })
    }
    fn each<T: Send, F: Fn(&mut Engine, usize) -> Result<T> + Sync>(
        &mut self,
        function: F,
    ) -> Result<Vec<T>> {
        thread::scope(|scope| {
            let function = &function;
            let handles: Vec<_> = self
                .engines
                .iter_mut()
                .enumerate()
                .map(|(rank, engine)| scope.spawn(move || function(engine, rank)))
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .map_err(|_| anyhow::anyhow!("GPU worker panicked"))?
                })
                .collect()
        })
    }
    pub fn split(&mut self, prob: &Matrix) -> Result<Vec<Matrix>> {
        let count = self.engines.len();
        ensure!(
            prob.rows >= count && prob.rows.is_multiple_of(count),
            "input rows not shardable"
        );
        let rows = prob.rows / count;
        self.each(|engine, rank| {
            let result = engine.allocate(rows, prob.cols)?;
            result.buffer.copy_from(
                &prob.buffer,
                rank * rows * prob.cols * 8,
                rows * prob.cols * 8,
            )?;
            Ok(result)
        })
    }
    pub fn candidates(
        &mut self,
        shards: &[Matrix],
        left: &Window,
        right: &Window,
        width: usize,
        include: Option<&Window>,
        count: usize,
    ) -> Result<Vec<Window>> {
        let devices = self.engines.len();
        let values = self.each(|engine, rank| {
            engine.moments(&shards[rank], &left.shard(rank, devices)?, right)
        })?;
        let mut totals = vec![0.; 5 * self.engines[0].config.word_bits as usize];
        for value in values {
            for (total, value) in totals.iter_mut().zip(value) {
                *total += value;
            }
        }
        select(
            &totals,
            self.engines[0].config.word_bits as usize,
            width,
            include,
            count,
        )
    }
    pub fn masses(
        &mut self,
        shards: &[Matrix],
        left: &Window,
        right: &Window,
        candidates: &[Window],
    ) -> Result<Vec<f64>> {
        let devices = self.engines.len();
        let scores = self.each(|engine, rank| {
            let left = left.shard(rank, devices)?;
            candidates
                .iter()
                .map(|target| engine.retained_mass(&shards[rank], &left, right, target))
                .collect::<Result<Vec<_>>>()
        })?;
        let mut total = vec![0.; candidates.len()];
        for row in scores {
            for (t, v) in total.iter_mut().zip(row) {
                *t += v;
            }
        }
        Ok(total)
    }
    pub fn compute(
        &mut self,
        shards: &[Matrix],
        left: &Window,
        right: &Window,
        target: &Window,
    ) -> Result<Stats> {
        let devices = self.engines.len();
        let results = thread::scope(|scope| {
            let mut handles = vec![];
            for (rank, ((engine, prob), spare)) in self
                .engines
                .iter_mut()
                .zip(shards)
                .zip(self.outputs.iter_mut())
                .enumerate()
            {
                let local = left.shard(rank, devices)?;
                let spare = spare.take();
                handles.push(scope.spawn(move || engine.step(prob, &local, right, target, spare)));
            }
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .map_err(|_| anyhow::anyhow!("GPU compute thread panicked"))?
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let peak = results.iter().map(|(_, s)| s.peak).fold(0., f64::max);
        let mut total = 0.;
        let mut index = usize::MAX;
        let cols = left.size() / devices;
        for (rank, (output, stats)) in results.into_iter().enumerate() {
            total += stats.total;
            if stats.peak == peak {
                index = index
                    .min((stats.index / cols) * left.size() + rank * cols + stats.index % cols);
            }
            self.outputs[rank] = Some(output);
        }
        Ok(Stats { peak, total, index })
    }
    pub fn exchange(
        &mut self,
        shards: Vec<Matrix>,
        left: &Window,
        target: &Window,
        peak: f64,
    ) -> Result<Vec<Matrix>> {
        ensure!(peak > 0., "empty retained distribution");
        let devices = self.engines.len();
        let rows = target.size() / devices;
        let nl = left.size();
        ensure!(rows > 0, "output must be collapsed to a single device");
        let pointers: Vec<_> = self
            .outputs
            .iter()
            .map(|m| m.as_ref().unwrap().ptr())
            .collect();
        thread::scope(|scope| {
            let mut handles = vec![];
            for (rank, (engine, old)) in self.engines.iter_mut().zip(shards).enumerate() {
                let pointers = &pointers;
                handles.push(scope.spawn(move || {
                    let output = if (old.rows, old.cols) == (rows, nl) {
                        old
                    } else {
                        drop(old);
                        engine.allocate(rows, nl)?
                    };
                    engine.unpack(pointers, &output, rows, nl / devices, rank, 1. / peak)?;
                    Ok(output)
                }));
            }
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .map_err(|_| anyhow::anyhow!("GPU exchange thread panicked"))?
                })
                .collect()
        })
    }
    pub fn collapse(&mut self, left: &Window, target: &Window, peak: f64) -> Result<Matrix> {
        let devices = self.engines.len();
        let pointers: Vec<_> = self
            .outputs
            .iter()
            .map(|m| m.as_ref().unwrap().ptr())
            .collect();
        let output = self.engines[0].allocate(target.size(), left.size())?;
        self.engines[0].unpack(
            &pointers,
            &output,
            target.size(),
            left.size() / devices,
            0,
            1. / peak,
        )?;
        Ok(output)
    }
}
