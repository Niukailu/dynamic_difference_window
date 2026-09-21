"""Opt-in peer GPU tests: DDW_TEST_DEVICES=1,2 python -m unittest discover -s tests."""

import os
import unittest

import numpy as np

from ddw.core import Window, cpu_step


@unittest.skipUnless(
    os.environ.get("DDW_TEST_DEVICES"), "set DDW_TEST_DEVICES for peer GPU tests"
)
class MultiGPU(unittest.TestCase):
    def test_full_distribution_and_reuse(self):
        import cupy as cp

        from ddw.multigpu import ShardedReplay

        devices = [int(x) for x in os.environ["DDW_TEST_DEVICES"].split(",")]
        cp.cuda.Device(devices[0]).use()
        rng = np.random.default_rng(2049)
        left = Window(0, (0, 1, 7, 15))
        right = Window(0, (0, 3, 8))
        target = Window(0, (0, 1, 2, 3, 7, 8, 9, 15))
        for mode, exchange in (
            ("difference", "peer"),
            ("linear", "peer"),
            ("difference", "staged"),
            ("linear", "staged"),
        ):
            cluster = ShardedReplay(devices, 16, "simon", mode, 2, exchange)
            try:
                probability = rng.integers(1, 32, (16, 8)).astype(float)
                probability /= probability.sum()
                source = cp.asarray(probability)
                cp.cuda.Stream.null.synchronize()
                shards = cluster.split(source)
                single = cluster.engines[0]
                cp.cuda.Device(devices[0]).use()
                expected_windows = single.candidates(
                    source, right, single.basis(left), 8, count=4
                )
                self.assertEqual(
                    cluster.candidates(shards, left, right, 8, count=4),
                    expected_windows,
                )
                current_left, current_right = left, right
                for t in (target, target, target):
                    expected = cpu_step(
                        probability, current_left, current_right, t, 16, "simon", mode
                    )
                    self.assertAlmostEqual(
                        cluster.masses(shards, current_left, current_right, [t])[0],
                        expected.sum(),
                        places=12,
                    )
                    _, trial_stats, _ = cluster.step(
                        shards, current_left, current_right, t, advance=False
                    )
                    self.assertAlmostEqual(trial_stats[1], expected.sum(), places=12)
                    # Candidate evaluation must leave the input distribution untouched.
                    np.testing.assert_allclose(
                        np.concatenate(cluster.map(lambda rank: shards[rank].get())),
                        probability,
                        rtol=2e-14,
                        atol=2e-15,
                    )
                    shards, (peak, total, index), _ = cluster.step(
                        shards, current_left, current_right, t
                    )
                    self.assertAlmostEqual(peak, expected.max(), places=14)
                    self.assertAlmostEqual(total, expected.sum(), places=12)
                    self.assertEqual(index, int(expected.argmax()))
                    probability = expected / expected.max()
                    actual = np.concatenate(
                        cluster.map(lambda rank: shards[rank].get())
                    )
                    np.testing.assert_allclose(
                        actual, probability, rtol=2e-14, atol=2e-15
                    )
                    current_left, current_right = t, current_left
            finally:
                cluster.close()
