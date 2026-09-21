"""CUDA backend: affine basis construction, window scoring, exact restricted propagation."""

from pathlib import Path

import cupy as cp
import numpy as np

from .core import Window


class Engine:
    def __init__(
        self, n=32, cipher="simon", mode="difference", memory_gib=12, kernel="coset"
    ):
        if (
            not 2 <= n <= 64
            or cipher not in ("simon", "simeck")
            or mode not in ("difference", "linear")
        ):
            raise ValueError("invalid cipher parameters")
        if kernel not in ("gather", "scatter", "coset"):
            raise ValueError("unknown transition kernel")
        self.kernel = kernel
        self.n = n
        self.limit = int(memory_gib * 2**30)
        a, b, c = (8, 1, 2) if cipher == "simon" else (5, 0, 1)
        prefix = f"#define BITS {n}\n#define WORD_MASK {((1 << n) - 1)}ULL\n#define RA {a}\n#define RB {b}\n#define RC {c}\n#define LINEAR_MODE {int(mode == 'linear')}\n"
        self.module = cp.RawModule(
            code=prefix + Path(__file__).with_name("kernels.cu").read_text(),
            options=("--std=c++17",),
        )
        self.shifts = cp.arange(n, dtype=cp.uint64)

    def launch(self, name, count, args):
        self.module.get_function(name)(
            (min((count + 127) // 128, 65535),), (128,), args
        )

    def basis(self, left):
        values = cp.asarray(left.values())
        bases = cp.empty(len(values), dtype=cp.uint64)
        vectors = cp.empty((len(values), self.n), dtype=cp.uint64)
        ranks = cp.empty(len(values), dtype=cp.int32)
        supports = cp.empty_like(bases)
        self.launch(
            "basis",
            len(values),
            (values, np.uint64(len(values)), bases, vectors, ranks, supports),
        )
        return bases, vectors, ranks, supports

    def choose(self, prob, right, basis, width, include=None, strategy="legacy"):
        return self.candidates(prob, right, basis, width, include, strategy)[0]

    def candidates(
        self, prob, right, basis, width, include=None, strategy="legacy", count=1
    ):
        bases, _, ranks, supports = basis
        # Matrix multiplication computes per-row right-bit marginals once.
        right_bits = (
            (cp.asarray(right.values())[:, None] >> self.shifts) & cp.uint64(1)
        ).astype(cp.float64)
        ones = prob @ right_bits
        edge = prob.sum(axis=1)
        base_bits = ((bases[:, None] >> self.shifts) & cp.uint64(1)).astype(cp.bool_)
        ones = cp.where(base_bits, edge[:, None] - ones, ones)
        active = ((supports[:, None] >> self.shifts) & cp.uint64(1)).astype(cp.float64)
        if strategy == "marginal":
            # A coordinate of a uniform affine subspace is either fixed or balanced.
            ones = cp.where(active > 0, edge[:, None] * 0.5, ones)
        elif strategy != "legacy":
            raise ValueError("unknown window strategy")
        totals = ones.sum(axis=0)
        chosen = cp.asnumpy(totals > edge.sum() * 0.5)
        base = sum(1 << i for i, v in enumerate(chosen) if v)
        mismatch = cp.where(cp.asarray(chosen)[None, :], edge[:, None] - ones, ones)
        active_count = cp.maximum(active.sum(axis=1), 1)
        weights = cp.exp2(-ranks.astype(cp.float64))
        scores = cp.asnumpy(
            (
                (mismatch + active * (edge / active_count)[:, None]) * weights[:, None]
            ).sum(axis=0)
        )
        if strategy == "marginal":
            scores = cp.asnumpy(cp.minimum(totals, edge.sum() - totals))
        # Subtractions in marginals can create tiny negative round-off.
        scores = np.maximum(scores, 0)
        selected = set(include.bits if include is not None else ())
        if len(selected) > width:
            raise ValueError("included window exceeds requested width")
        for b in sorted(range(self.n), key=lambda b: (-scores[b], b)):
            if len(selected) >= width:
                break
            if scores[b] > 0:
                selected.add(b)
        # Keep the reference affine window as a subset: probabilities cannot decrease
        # in exact arithmetic along the same included schedule.
        if include is not None:
            base = include.base
        windows = [Window(base, tuple(sorted(selected)))]
        # Local one-bit exchanges, evaluated by actual propagation by the caller.
        mandatory = set(include.bits if include is not None else ())
        removable = sorted(selected - mandatory, key=lambda b: (scores[b], -b))
        outsiders = sorted(set(range(self.n)) - selected, key=lambda b: (-scores[b], b))
        pairs = sorted(
            ((old, new) for old in removable for new in outsiders if scores[new] > 0),
            key=lambda pair: (scores[pair[0]] - scores[pair[1]], pair),
        )
        for old, new in pairs[: max(0, count - 1)]:
            bits = (selected - {old}) | {new}
            windows.append(Window(base, tuple(sorted(bits))))
        return windows

    def step(self, prob, left, right, target, basis=None):
        if any(len(w.bits) > 20 for w in (left, right, target)):
            raise ValueError("CUDA windows support at most 20 active bits")
        if prob.shape != (1 << len(left.bits), 1 << len(right.bits)):
            raise ValueError("probability matrix does not match windows")
        nl, nr = prob.shape
        required = prob.nbytes + 8 * (1 << len(target.bits)) * nl
        if required > self.limit:
            raise MemoryError(
                f"two matrices need {required / 2**30:.2f} GiB; limit={self.limit / 2**30:.2f} GiB"
            )
        free, _ = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        if 8 * (1 << len(target.bits)) * nl > (free + pool.free_bytes()) * 0.85:
            raise MemoryError("insufficient free device memory with 15% headroom")
        bases, vectors, ranks, _ = self.basis(left) if basis is None else basis
        # Do not mutate a caller's basis: useful for evaluating multiple windows.
        bases, vectors = bases.copy(), vectors.copy()
        outside = cp.empty(nl, dtype=cp.int32)
        restricted = right if self.kernel in ("gather", "coset") else target
        self.launch(
            "restrict_basis",
            nl,
            (np.uint64(nl), np.uint64(restricted.mask), bases, vectors, ranks, outside),
        )
        packed = cp.empty_like(vectors)
        self.launch(
            "pack_basis",
            nl * self.n,
            (np.uint64(nl), np.uint64(restricted.mask), vectors, packed),
        )
        if self.kernel == "coset":
            return self.coset_step(
                prob, right, target, bases, vectors, packed, ranks, outside
            )
        allocate = cp.empty if self.kernel == "gather" else cp.zeros
        result = allocate((1 << len(target.bits), nl), dtype=cp.float64)
        if self.kernel == "gather":
            self.launch(
                "gather",
                result.size,
                (
                    prob,
                    np.uint64(nl),
                    np.uint64(nr),
                    np.uint64(result.shape[0]),
                    cp.asarray(target.values()),
                    prob.sum(axis=1),
                    bases,
                    vectors,
                    packed,
                    ranks,
                    outside,
                    np.uint64(right.mask),
                    np.uint64(right.base),
                    result,
                ),
            )
            return result
        self.launch(
            "propagate",
            nl * nr,
            (
                prob,
                np.uint64(nl),
                np.uint64(nr),
                cp.asarray(right.values()),
                bases,
                vectors,
                packed,
                ranks,
                outside,
                np.uint64(target.mask),
                np.uint64(target.base),
                result,
            ),
        )
        return result

    def coset_step(self, prob, right, target, bases, vectors, packed, ranks, outside):
        nl, nr = prob.shape
        mass = prob.sum(axis=1)
        reducers = cp.empty((nl, 20), dtype=cp.uint64)
        masks = cp.empty(nl, dtype=cp.uint64)
        sizes = cp.empty(nl, dtype=cp.uint64)
        self.launch(
            "coset_plan",
            nl,
            (
                np.uint64(nl),
                np.uint64(nr),
                packed,
                ranks,
                outside,
                mass,
                reducers,
                masks,
                sizes,
            ),
        )
        offsets = cp.empty(nl + 1, dtype=cp.uint64)
        offsets[0] = 0
        cp.cumsum(sizes, out=offsets[1:])
        buckets = int(offsets[-1])
        output_bytes = 8 * (1 << len(target.bits)) * nl
        required = prob.nbytes + 8 * buckets + output_bytes
        if required > self.limit:
            raise MemoryError(
                f"matrices plus coset sums need {required / 2**30:.2f} GiB"
            )
        free, _ = cp.cuda.runtime.memGetInfo()
        if (
            8 * buckets + output_bytes
            > (free + cp.get_default_memory_pool().free_bytes()) * 0.85
        ):
            raise MemoryError(
                "insufficient memory for coset sums and output with 15% headroom"
            )
        sums = cp.zeros(buckets, dtype=cp.float64)
        self.launch(
            "coset_sum",
            prob.size,
            (
                prob,
                np.uint64(nl),
                np.uint64(nr),
                ranks,
                outside,
                reducers,
                masks,
                offsets,
                sums,
            ),
        )
        result = cp.empty((1 << len(target.bits), nl), dtype=cp.float64)
        self.launch(
            "coset_lookup",
            result.size,
            (
                np.uint64(nl),
                np.uint64(result.shape[0]),
                cp.asarray(target.values()),
                mass,
                bases,
                vectors,
                ranks,
                outside,
                np.uint64(right.mask),
                np.uint64(right.base),
                reducers,
                masks,
                offsets,
                sums,
                result,
            ),
        )
        return result
