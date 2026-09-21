"""Small CPU oracle and shared window representation (no CUDA dependency)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Window:
    base: int
    bits: tuple[int, ...] = ()

    def __post_init__(self):
        if tuple(sorted(set(self.bits))) != self.bits or any(
            b < 0 or b >= 64 for b in self.bits
        ):
            raise ValueError("window bits must be unique, sorted, and within [0,64)")
        object.__setattr__(self, "base", self.base & ~self.mask)

    @property
    def mask(self):
        return sum(1 << b for b in self.bits)

    def values(self):
        x = np.full(1 << len(self.bits), self.base, dtype=np.uint64)
        idx = np.arange(len(x), dtype=np.uint64)
        for i, bit in enumerate(self.bits):
            x |= ((idx >> np.uint64(i)) & np.uint64(1)) << np.uint64(bit)
        return x

    def index(self, x):
        if x & ~self.mask != self.base:
            return None
        return sum(((x >> b) & 1) << i for i, b in enumerate(self.bits))


def rot(x, r, n):
    r %= n
    return ((x << r) | (x >> ((n - r) % n))) & ((1 << n) - 1)


def fun(x, n, cipher="simon"):
    a, b, c = (8, 1, 2) if cipher == "simon" else (5, 0, 1)
    return (rot(x, a, n) & rot(x, b, n)) ^ rot(x, c, n)


def transition(delta, n, cipher="simon", mode="difference"):
    a, b, c = (8, 1, 2) if cipher == "simon" else (5, 0, 1)
    base = fun(delta, n, cipher) if mode == "difference" else rot(delta, -c, n)
    rows = []
    for j in range(n):
        e = 1 << j
        rows.append(
            fun(e, n, cipher) ^ fun(delta ^ e, n, cipher) ^ base
            if mode == "difference"
            else rot((delta & rot(e, a - b, n)) ^ rot(delta & e, b - a, n), -b, n)
        )
    rank = 0
    for p in range(n):
        k = next((k for k in range(rank, n) if rows[k] >> p & 1), None)
        if k is None:
            continue
        rows[rank], rows[k] = rows[k], rows[rank]
        for j in range(n):
            if j != rank and rows[j] >> p & 1:
                rows[j] ^= rows[rank]
        if base >> p & 1:
            base ^= rows[rank]
        rank += 1
    return base, rows[:rank]


def span(base, rows):
    values = [base]
    for a in rows:
        values += [x ^ a for x in values]
    return values


def cpu_step(prob, left, right, target, n, cipher="simon", mode="difference"):
    """Deliberately simple exhaustive oracle; for tiny windows only."""
    result = np.zeros((1 << len(target.bits), len(left.values())))
    for left_index, delta in enumerate(left.values()):
        base, rows = transition(int(delta), n, cipher, mode)
        outputs = span(base, rows)
        for r, gamma in enumerate(right.values()):
            if not prob[left_index, r]:
                continue
            for beta in outputs:
                j = target.index(beta ^ int(gamma))
                if j is not None:
                    result[j, left_index] += prob[left_index, r] / len(outputs)
    return result
