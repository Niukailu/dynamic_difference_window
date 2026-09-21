"""Check transpose propagation and full-suffix candidate scoring independently."""

import os
import unittest

import numpy as np

from ddw.core import Window, cpu_step


class Refine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp

            cp.cuda.Device(int(os.environ.get("DDW_TEST_DEVICE", "0"))).use()
            cp.zeros(1)
        except Exception as exc:
            raise unittest.SkipTest(str(exc))
        cls.cp = cp

    def test_adjoint_inner_product(self):
        from ddw.gpu import Engine

        cp = self.cp
        rng = np.random.default_rng(428)
        left, right, target = (
            Window(0, (0, 7)),
            Window(0, (1, 3)),
            Window(0, (0, 1, 2, 3, 7, 8, 9)),
        )
        for cipher in ("simon", "simeck"):
            for mode in ("difference", "linear"):
                engine = Engine(16, cipher, mode)
                probability = rng.random((4, 4))
                value = rng.random((128, 4))
                expected = cpu_step(probability, left, right, target, 16, cipher, mode)
                backward = engine.adjoint(cp.asarray(value), left, right, target).get()
                self.assertAlmostEqual(
                    float((expected * value).sum()),
                    float((probability * backward).sum()),
                    places=12,
                )

    def test_suffix_score_equals_complete_replay(self):
        import math

        from ddw.gpu import Engine
        from ddw.refine import (
            backward_messages,
            candidate_score,
            endpoint_value,
            optimize,
        )

        cp = self.cp
        engine = Engine(16)
        initial = (Window(0), Window(1))
        left, right = initial
        probability = cp.ones((1, 1), dtype=cp.float64)
        schedule = []
        prefixes = []
        for _ in range(7):
            prefixes.append((probability, left, right))
            target = engine.choose(probability, right, engine.basis(left), 6)
            schedule.append(target)
            probability = engine.step(probability, left, right, target)
            left, right = target, left
        i, j = np.unravel_index(int(probability.argmax()), probability.shape)
        endpoint = (int(left.values()[i]), int(right.values()[j]))
        original = endpoint_value(probability, left, right, endpoint)
        messages = backward_messages(engine, initial, schedule, endpoint)
        for position, (prefix, left, right) in enumerate(prefixes):
            candidates = [schedule[position]] + engine.candidates(
                prefix, right, engine.basis(left), 6, count=3
            )
            for candidate in candidates:
                actual = candidate_score(
                    engine,
                    prefix,
                    left,
                    right,
                    schedule,
                    position,
                    candidate,
                    messages,
                    endpoint,
                )
                trial = list(schedule)
                trial[position] = candidate
                prob = np.ones((1, 1))
                current_left, current_right = initial
                for target in trial:
                    prob = cpu_step(prob, current_left, current_right, target, 16)
                    current_left, current_right = target, current_left
                ri, ci = (
                    current_left.index(endpoint[0]),
                    current_right.index(endpoint[1]),
                )
                expected = prob[ri, ci] if ri is not None and ci is not None else 0
                if expected:
                    self.assertAlmostEqual(actual, math.log2(expected), places=11)
                else:
                    self.assertEqual(actual, -math.inf)
        host_messages = backward_messages(
            engine, initial, schedule, endpoint, cache="host"
        )
        self.assertTrue(
            all(isinstance(message[0], np.ndarray) for message in host_messages)
        )
        for (host, host_scale), (device, device_scale) in zip(host_messages, messages):
            np.testing.assert_allclose(host, device.get(), rtol=2e-14, atol=1e-15)
            self.assertAlmostEqual(host_scale, device_scale, places=12)
        result, _ = optimize(engine, initial, schedule, endpoint, 6, 4, 2)
        host_result, _ = optimize(
            engine, initial, schedule, endpoint, 6, 4, 2, cache="host"
        )
        self.assertEqual(result, host_result)
        prob = np.ones((1, 1))
        left, right = initial
        for target in result:
            prob = cpu_step(prob, left, right, target, 16)
            left, right = target, left
        self.assertGreaterEqual(
            prob[left.index(endpoint[0]), right.index(endpoint[1])] + 1e-15, original
        )
