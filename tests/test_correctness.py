"""Independent truth tables, CPU oracle, CUDA equivalence, and nested-window checks.
Run: python -m unittest discover -s tests -v
Set DDW_TEST_DEVICE to choose a GPU; GPU tests skip if CUDA/CuPy unavailable.
"""

import os
import unittest

import numpy as np

from ddw.core import Window, cpu_step, fun, span, transition


class TruthTables(unittest.TestCase):
    def test_difference_truth_table(self):
        n = 16
        for cipher in ("simon", "simeck"):
            for delta in (0, 1, 0x101, 0xFFFF, 0xBEEF):
                counts = np.bincount(
                    [
                        fun(x, n, cipher) ^ fun(x ^ delta, n, cipher)
                        for x in range(1 << n)
                    ],
                    minlength=1 << n,
                )
                base, rows = transition(delta, n, cipher)
                expected = np.zeros(1 << n)
                expected[span(base, rows)] = 1 / (1 << len(rows))
                np.testing.assert_array_equal(counts / (1 << n), expected)

    def test_linear_walsh_truth_table(self):
        n = 16
        for cipher in ("simon", "simeck"):
            for beta in (0, 1, 0x101, 0xFFFF, 0xBEEF):
                walsh = np.array(
                    [
                        1 - 2 * ((fun(x, n, cipher) & beta).bit_count() % 2)
                        for x in range(1 << n)
                    ],
                    dtype=np.int64,
                )
                h = 1
                while h < len(walsh):
                    blocks = walsh.reshape(-1, h * 2)
                    a, b = blocks[:, :h].copy(), blocks[:, h:].copy()
                    blocks[:, :h], blocks[:, h:] = a + b, a - b
                    h *= 2
                base, rows = transition(beta, n, cipher, "linear")
                expected = np.zeros(1 << n)
                expected[span(base, rows)] = 1 / (1 << len(rows))
                np.testing.assert_array_equal((walsh / (1 << n)) ** 2, expected)

    def test_window_64(self):
        window = Window((1 << 63) | 3, (0, 31, 63))
        for i, x in enumerate(window.values()):
            self.assertEqual(window.index(int(x)), i)
        self.assertIsNone(window.index(4))


class GPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp

            cp.cuda.Device(int(os.environ.get("DDW_TEST_DEVICE", "0"))).use()
            cp.zeros(1)
        except Exception as exc:
            raise unittest.SkipTest(str(exc))
        cls.cp = cp

    def test_projected_retained_mass(self):
        from ddw.gpu import Engine

        cp = self.cp
        rng = np.random.default_rng(846)
        for n in (16, 32, 64):
            for cipher in ("simon", "simeck"):
                for mode in ("difference", "linear"):
                    engine = Engine(n, cipher, mode)
                    for width in (0, 3, 10, 13, min(17, n)):
                        left = Window(0, (0, n - 1))
                        right = Window(0, tuple(range(width)))
                        target = Window(0, tuple(sorted({0, 1, 2, 7, n - 1})))
                        prob = cp.asarray(rng.random((4, 1 << width)))
                        expected = float(engine.step(prob, left, right, target).sum())
                        self.assertAlmostEqual(
                            engine.retained_mass(prob, left, right, target),
                            expected,
                            delta=max(1e-12, expected * 2e-14),
                        )
                        target = Window(1 << (n - 2), target.bits)
                        expected = float(engine.step(prob, left, right, target).sum())
                        self.assertAlmostEqual(
                            engine.retained_mass(prob, left, right, target),
                            expected,
                            delta=max(1e-12, expected * 2e-14),
                        )

    def test_cpu_gpu_and_mass(self):
        from ddw.gpu import Engine

        cp = self.cp
        rng = np.random.default_rng(729)
        for n in (16, 24, 32, 48, 64):
            for cipher in ("simon", "simeck"):
                for mode in ("difference", "linear"):
                    with self.subTest(n=n, cipher=cipher, mode=mode):
                        engine = Engine(n, cipher, mode)
                        left, right = Window(0, (0, n - 1)), Window(2, (0, 3))
                        prob = rng.random((4, 4))
                        prob /= prob.sum()
                        target = Window(
                            2, tuple(sorted({0, 1, 2, 3, 4, 5, 7, 8, 9, n - 1}))
                        )
                        expected = cpu_step(prob, left, right, target, n, cipher, mode)
                        for kernel in ("gather", "scatter", "coset", "coset_lut"):
                            engine.kernel = kernel
                            actual = engine.step(
                                cp.asarray(prob), left, right, target
                            ).get()
                            np.testing.assert_allclose(
                                actual, expected, atol=2e-16, rtol=2e-14
                            )
                        self.assertLessEqual(actual.sum(), 1 + 1e-14)
                        basis = engine.basis(left)
                        bases, vectors, ranks, _ = [x.get() for x in basis]
                        for i, delta in enumerate(left.values()):
                            b, rows = transition(int(delta), n, cipher, mode)
                            self.assertEqual(int(bases[i]), b)
                            self.assertEqual(
                                list(map(int, vectors[i, : ranks[i]])), rows
                            )

    def test_outside_projection_regression(self):
        # Original dd ^= A brought active bits back into the next comparison,
        # causing a legal mass of 1/256 to be rejected entirely.
        from ddw.gpu import Engine

        cp = self.cp
        left, right = Window(0xE9AE), Window(0xF24A)
        target = Window(33368, (0, 1, 2, 5, 8, 11, 13, 14))
        expected = cpu_step(np.ones((1, 1)), left, right, target, 16)
        self.assertEqual(expected.sum(), 1 / 256)
        for kernel in ("scatter", "gather", "coset", "coset_lut"):
            engine = Engine(16, kernel=kernel)
            np.testing.assert_array_equal(
                engine.step(cp.ones((1, 1)), left, right, target).get(), expected
            )

    def test_dense_random_differences(self):
        from ddw.gpu import Engine

        cp = self.cp
        rng = np.random.default_rng(52)
        engine = Engine(16)
        for _ in range(12):
            left = Window(int(rng.integers(1, 65536)))
            right = Window(int(rng.integers(0, 65536)), (1, 7))
            target = Window(
                int(rng.integers(0, 65536)),
                tuple(sorted(map(int, rng.choice(16, 8, replace=False)))),
            )
            prob = rng.random((1, 4))
            prob /= prob.sum()
            expected = cpu_step(prob, left, right, target, 16)
            for kernel in ("scatter", "gather", "coset", "coset_lut"):
                engine.kernel = kernel
                np.testing.assert_allclose(
                    engine.step(cp.asarray(prob), left, right, target).get(),
                    expected,
                    atol=1e-16,
                    rtol=1e-13,
                )

    def test_coset_large_intersection_uniform_input(self):
        from ddw.gpu import Engine

        cp = self.cp
        engine = Engine(16, kernel="coset_lut")
        left = Window(0xFFFF)
        right = target = Window(0, tuple(range(16)))
        prob = cp.full((1, 1 << 16), 1 / (1 << 16), dtype=cp.float64)
        actual = engine.step(prob, left, right, target).get()
        # XOR convolution of any normalized transition with a uniform full word
        # must be uniform, including large-dimensional intersections.
        np.testing.assert_array_equal(actual, np.full((1 << 16, 1), 1 / (1 << 16)))

    def test_bit_marginals_against_exact_dyadic_sums(self):
        from ddw.gpu import Engine

        cp = self.cp
        engine = Engine(64)
        rng = np.random.default_rng(21)
        positions = (0, 2, 4, 7, 8, 11, 13, 16, 19, 23, 29, 33, 40, 50, 63)
        for width in (0, 2, 8, 9, 10, 11, 15):
            right = Window(0x123456, tuple(sorted(positions[:width])))
            prob = rng.integers(0, 16, size=(3, 1 << width)).astype(np.float64) / 4096
            bits = (
                (right.values()[:, None] >> np.arange(64, dtype=np.uint64)) & 1
            ).astype(np.float64)
            for method in ("gemm", "tiled", "bitwise", "hierarchical"):
                engine.statistics = method
                ones, edge = engine.marginals(cp.asarray(prob), right)
                np.testing.assert_array_equal(ones.get(), prob @ bits)
                np.testing.assert_array_equal(edge.get(), prob.sum(axis=1))

    def test_fused_summary_and_first_tie(self):
        from ddw.gpu import Engine

        cp = self.cp
        engine = Engine(32)
        rng = np.random.default_rng(24)
        for size in (1, 19, 257, 2049, 3000001):
            prob = rng.integers(0, 16, size=size).astype(np.float64) / 4096
            peak, total, index = engine.summary(cp.asarray(prob))
            self.assertEqual(peak, prob.max())
            self.assertEqual(total, prob.sum())
            self.assertEqual(index, int(prob.argmax()))
        self.assertEqual(engine.summary(cp.zeros(19)), (0.0, 0.0, 0))

    def test_fused_transition_and_reused_output(self):
        from ddw.gpu import Engine

        cp = self.cp
        engine = Engine(16, kernel="coset_lut", statistics="hierarchical")
        left, right = Window(0xE9AE), Window(0xF24A)
        target = Window(33368, (0, 1, 2, 5, 8, 11, 13, 14))
        prob = cp.ones((1, 1))
        expected = cpu_step(np.ones((1, 1)), left, right, target, 16)
        out = cp.full(expected.shape, 999.0)
        for _ in range(2):
            actual, (peak, total, index) = engine.step_and_summary(
                prob, left, right, target, out=out
            )
            self.assertIs(actual, out)
            np.testing.assert_array_equal(actual.get(), expected)
            self.assertEqual(peak, expected.max())
            self.assertEqual(total, expected.sum())
            self.assertEqual(index, int(expected.argmax()))
        with self.assertRaises(ValueError):
            engine.step(prob, left, right, Window(0), out=prob)

    def test_nested_windows(self):
        from ddw.gpu import Engine

        cp = self.cp
        engine = Engine(32)
        left, right = Window(1), Window(4)
        prob = cp.ones((1, 1))
        basis = engine.basis(left)
        small = engine.choose(prob, right, basis, 1)
        large = engine.choose(prob, right, basis, 4, small)
        a = engine.step(prob, left, right, small).get()
        b = engine.step(prob, left, right, large).get()
        for i, x in enumerate(small.values()):
            np.testing.assert_allclose(a[i], b[large.index(int(x))])


if __name__ == "__main__":
    unittest.main()
