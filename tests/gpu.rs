//! Run explicitly on shared GPUs: cargo test --test gpu -- --ignored --test-threads=1.
use ddw::{
    engine::Engine,
    model::{Config, Window},
    multi::Cluster,
};

fn devices() -> Vec<i32> {
    std::env::var("DDW_TEST_DEVICES")
        .unwrap_or_else(|_| "0,1".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect()
}
fn close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!(
            (a - b).abs() <= 1e-11 * a.abs().max(b.abs()).max(1.),
            "index {i}: {a} != {b}"
        );
    }
}
#[test]
#[ignore = "requires CUDA and peer access on DDW_TEST_DEVICES"]
fn adjoint_projected_mass_and_sharding() -> anyhow::Result<()> {
    for cipher in ["simon", "simeck"] {
        for mode in ["difference", "linear"] {
            let config = Config {
                cipher: cipher.into(),
                mode: mode.into(),
                word_bits: 16,
                left: 0,
                right: 1,
                width: 10,
                rounds: 4,
            };
            let cards = devices();
            let mut engine = Engine::new(&config, cards[0], 2.)?;
            let left = Window::new(0, vec![0, 1, 2, 3])?;
            let right = Window::new(0, vec![0, 1, 2])?;
            let target = Window::new(0, vec![0, 1, 2, 3, 4, 5, 8, 9, 14, 15])?;
            let values: Vec<_> = (0..left.size() * right.size())
                .map(|i| (i % 13 + 1) as f64 / 64.)
                .collect();
            let prob = engine.from_host(left.size(), right.size(), &values)?;
            let (next, stats) = engine.step(&prob, &left, &right, &target, None)?;
            let mass = engine.retained_mass(&prob, &left, &right, &target)?;
            close(&[mass], &[stats.total]);
            let h: Vec<_> = (0..next.size())
                .map(|i| (i % 17 + 1) as f64 / 32.)
                .collect();
            let suffix = engine.from_host(next.rows, next.cols, &h)?;
            let adjoint = engine.adjoint(&suffix, &left, &right, &target)?;
            close(
                &[engine.dot(&next, &suffix)?],
                &[engine.dot(&prob, &adjoint)?],
            );
            let transpose = engine.transpose(&next)?;
            let roundtrip = engine.transpose(&transpose)?;
            close(
                &next.buffer.download::<f64>(next.size())?,
                &roundtrip.buffer.download::<f64>(roundtrip.size())?,
            );
            let mut cluster = Cluster::new(&config, &cards, 2.)?;
            let shards = cluster.split(&prob)?;
            let distributed = cluster.compute(&shards, &left, &right, &target)?;
            close(
                &[distributed.total, distributed.peak],
                &[stats.total, stats.peak],
            );
            assert_eq!(distributed.index, stats.index);
            let shards = cluster.exchange(shards, &left, &target, stats.peak)?;
            let actual: Vec<f64> = shards
                .iter()
                .map(|m| m.buffer.download(m.size()))
                .collect::<anyhow::Result<Vec<Vec<f64>>>>()?
                .concat();
            let expected: Vec<f64> = next
                .buffer
                .download::<f64>(next.size())?
                .iter()
                .map(|p| p / stats.peak)
                .collect();
            close(&actual, &expected);
            let small = Window::point(0);
            let normalized = engine.from_host(target.size(), left.size(), &expected)?;
            let (last, last_stats) = engine.step(&normalized, &target, &left, &small, None)?;
            let distributed = cluster.compute(&shards, &target, &left, &small)?;
            close(&[distributed.peak], &[last_stats.peak]);
            let collapsed = cluster.collapse(&target, &small, distributed.peak)?;
            let expected: Vec<f64> = last
                .buffer
                .download::<f64>(last.size())?
                .iter()
                .map(|p| p / last_stats.peak)
                .collect();
            close(
                &collapsed.buffer.download::<f64>(collapsed.size())?,
                &expected,
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA; validates virtual states against materialized FP64 states"]
fn implicit_virtual_state_matches_dense() -> anyhow::Result<()> {
    for n in [16, 24, 32, 48, 64] {
        for cipher in ["simon", "simeck"] {
            for mode in ["difference", "linear"] {
                let config = Config {
                    cipher: cipher.into(),
                    mode: mode.into(),
                    word_bits: n,
                    left: 0,
                    right: 1,
                    width: 6,
                    rounds: 5,
                };
                let mut engine = Engine::new(&config, devices()[0], 4.)?;
                let (mut left, mut right) = config.initial();
                let mut dense = engine.point(1, 1, 0)?;
                let mut virtual_state = engine.compressed_point()?;
                for _ in 0..5 {
                    let target = engine
                        .candidates(&dense, &left, &right, 6, None, 1)?
                        .remove(0);
                    let (next, stats) = engine.step(&dense, &left, &right, &target, None)?;
                    let (implicit, actual) =
                        engine.compressed_step(&virtual_state, &left, &right, &target)?;
                    close(&[stats.peak, stats.total], &[actual.peak, actual.total]);
                    assert_eq!(stats.index, actual.index);
                    engine.scale(&next, 1. / stats.peak)?;
                    let restored = engine.compressed_materialize(&implicit)?;
                    close(
                        &next.buffer.download::<f64>(next.size())?,
                        &restored.buffer.download::<f64>(restored.size())?,
                    );
                    let options = engine.compressed_candidates(
                        &implicit,
                        &target,
                        &left,
                        6,
                        Some(&target),
                    )?;
                    assert_eq!(options[0].mask() & target.mask(), target.mask());
                    close(
                        &[engine.compressed_value(&implicit, 0, 0)?],
                        &[next.value(0, 0)?],
                    );
                    dense = next;
                    virtual_state = implicit;
                    right = left;
                    left = target;
                }
            }
        }
    }
    Ok(())
}
