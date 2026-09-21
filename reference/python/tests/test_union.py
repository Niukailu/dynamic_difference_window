"""Brute-force weighted path sets check reflection and inclusion-exclusion."""

import unittest

from ddw.core import Window, span, transition
from ddw.symmetric_union import intersect, reflected


def paths(initial, windows, cipher, mode):
    active = {(initial[1].base, initial[0].base): 1.0}
    for window in windows:
        result = {}
        for path, weight in active.items():
            base, basis = transition(path[-1], 4, cipher, mode)
            for target in span(base ^ path[-2], basis):
                if window.index(int(target)) is not None:
                    result[path + (int(target),)] = weight / (1 << len(basis))
        active = result
    return {
        p: w for p, w in active.items() if p[-2:] == (initial[0].base, initial[1].base)
    }


class SymmetricUnion(unittest.TestCase):
    def test_intersection_membership(self):
        for a in (Window(0, (0, 2)), Window(2, (0, 2))):
            for b in (Window(0, (1, 2)), Window(1, (1, 2)), Window(8, (0, 1))):
                common = intersect(a, b)
                expected = set(map(int, a.values())) & set(map(int, b.values()))
                self.assertEqual(
                    set(map(int, common.values())) if common else set(), expected
                )

    def test_weighted_path_reflection_and_union(self):
        full = Window(0, (0, 1, 2, 3))
        for cipher in ("simon", "simeck"):
            for mode in ("difference", "linear"):
                for x, y in ((1, 0), (1, 1), (1, 2), (2, 3)):
                    initial = (Window(x), Window(y))
                    whole = paths(initial, [full] * 4, cipher, mode)
                    if whole:
                        break
                self.assertTrue(whole)
                chosen = next(iter(whole))
                schedule = [Window(chosen[i + 2], (0, 2)) for i in range(4)]
                reverse = reflected(initial, schedule)
                a = paths(initial, schedule, cipher, mode)
                b = paths(initial, reverse, cipher, mode)
                self.assertEqual(
                    b, {tuple(reversed(path)): weight for path, weight in a.items()}
                )
                common = [intersect(x, y) for x, y in zip(schedule, reverse)]
                intersection = (
                    paths(initial, common, cipher, mode)
                    if all(w is not None for w in common)
                    else {}
                )
                self.assertEqual(set(intersection), set(a) & set(b))
                union = dict(a)
                union.update(b)
                self.assertAlmostEqual(
                    sum(union.values()),
                    2 * sum(a.values()) - sum(intersection.values()),
                    places=14,
                )
