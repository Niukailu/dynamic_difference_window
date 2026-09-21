"""CUDA backend: affine basis construction, window scoring, exact restricted propagation."""

from pathlib import Path

import cupy as cp
import numpy as np

from .core import Window


class Engine:
    def __init__(
        self,
        n=32,
        cipher="simon",
        mode="difference",
        memory_gib=12,
        kernel="coset_lut",
        statistics="hierarchical",
    ):
        if (
            not 2 <= n <= 64
            or cipher not in ("simon", "simeck")
            or mode not in ("difference", "linear")
        ):
            raise ValueError("invalid cipher parameters")
        if kernel not in ("gather", "scatter", "coset", "coset_lut"):
            raise ValueError("unknown transition kernel")
        self.kernel = kernel
        if statistics not in ("bitwise", "tiled", "hierarchical", "gemm"):
            raise ValueError("unknown statistics algorithm")
        self.statistics = statistics
        self.n = n
        self.limit = int(memory_gib * 2**30)
        a, b, c = (8, 1, 2) if cipher == "simon" else (5, 0, 1)
        prefix = f"#define BITS {n}\n#define WORD_MASK {((1 << n) - 1)}ULL\n#define RA {a}\n#define RB {b}\n#define RC {c}\n#define LINEAR_MODE {int(mode == 'linear')}\n"
        self.module = cp.RawModule(
            code=prefix + (Path(__file__).resolve().parents[3] / "cuda/kernels.cu").read_text(),
            options=("--std=c++17",),
        )
        self.shifts = cp.arange(n, dtype=cp.uint64)

    def launch(self, name, count, args, threads=128):
        blocks = (count + threads - 1) // threads
        if name in (
            "propagate",
            "gather",
            "coset_sum",
            "coset_lookup",
            "pack_basis",
            "affine_tables",
            "coset_lookup_lut",
        ):
            blocks = min(blocks, 65535)
        self.module.get_function(name)((blocks,), (threads,), args)

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

    def marginals(self, prob, right):
        if self.statistics == "gemm":
            right_bits = (
                (cp.asarray(right.values())[:, None] >> self.shifts) & cp.uint64(1)
            ).astype(cp.float64)
            return prob @ right_bits, prob.sum(axis=1)
        nl, nr = prob.shape
        width = len(right.bits)
        marginal = cp.empty((nl, width + 1), dtype=cp.float64)
        if self.statistics == "hierarchical":
            tiles = (nr + 4095) // 4096
            partial = cp.empty((nl * tiles, 13), dtype=cp.float64)
            self.launch(
                "hierarchical_marginal_tiles",
                nl * tiles * 256,
                (prob, np.uint64(nr), np.uint64(tiles), partial),
                threads=256,
            )
            self.launch(
                "hierarchical_marginal_finish",
                marginal.size,
                (np.uint64(nl), np.uint64(tiles), np.int32(width), partial, marginal),
            )
        elif self.statistics == "bitwise":
            self.launch(
                "bit_marginal_rows",
                nl * 256,
                (prob, np.uint64(nr), np.int32(width), marginal),
                threads=256,
            )
        else:
            tiles = (nr + 1023) // 1024
            partial = cp.empty((nl * tiles, 11), dtype=cp.float64)
            self.launch(
                "bit_marginal_tiles",
                nl * tiles * 256,
                (prob, np.uint64(nr), np.uint64(tiles), partial),
                threads=256,
            )
            self.launch(
                "bit_marginal_finish",
                marginal.size,
                (np.uint64(nl), np.uint64(tiles), np.int32(width), partial, marginal),
            )
        edge = marginal[:, width].copy()
        fixed = ((cp.uint64(right.base) >> self.shifts) & cp.uint64(1)).astype(
            cp.float64
        )
        ones = edge[:, None] * fixed[None, :]
        if width:
            ones[:, cp.asarray(right.bits)] = marginal[:, :width]
        return ones, edge

    def summary(self, prob, partials=None):
        """Return peak, total, first argmax after one matrix read and one host sync."""
        if partials:
            maxima, indices, sums = partials
            blocks = maxima.size
        else:
            blocks = min((prob.size + 255) // 256, 8192)
            maxima = cp.empty(blocks, dtype=cp.float64)
            sums = cp.empty_like(maxima)
            indices = cp.empty(blocks, dtype=cp.uint64)
            self.launch(
                "distribution_stats",
                blocks * 256,
                (prob, np.uint64(prob.size), maxima, indices, sums),
                threads=256,
            )
        result = cp.empty(3, dtype=cp.float64)
        self.launch(
            "finish_stats",
            256,
            (np.uint64(blocks), maxima, indices, sums, result),
            threads=256,
        )
        peak, total, index = cp.asnumpy(result)
        return float(peak), float(total), int(index)

    def choose(self, prob, right, basis, width, include=None, strategy="legacy"):
        return self.candidates(prob, right, basis, width, include, strategy)[0]

    def candidates(
        self, prob, right, basis, width, include=None, strategy="legacy", count=1
    ):
        bases, _, ranks, supports = basis
        # Compute per-row right-bit marginals once.
        ones, edge = self.marginals(prob, right)
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
        return self.windows_from_scores(base, scores, width, include, count)

    def window_moments(self, prob, right, basis):
        """Additive sufficient statistics for global legacy window scoring."""
        bases, _, ranks, supports = basis
        ones, edge = self.marginals(prob, right)
        base_bits = ((bases[:, None] >> self.shifts) & cp.uint64(1)).astype(cp.bool_)
        ones = cp.where(base_bits, edge[:, None] - ones, ones)
        active = ((supports[:, None] >> self.shifts) & cp.uint64(1)).astype(cp.float64)
        weights = cp.exp2(-ranks.astype(cp.float64))
        weighted = edge * weights
        # Four vectors; the final vector stores scalar totals in its first two entries.
        result = cp.zeros((4, self.n), dtype=cp.float64)
        result[0] = ones.sum(axis=0)
        result[1] = (ones * weights[:, None]).sum(axis=0)
        result[2] = (
            active * (weighted / cp.maximum(active.sum(axis=1), 1))[:, None]
        ).sum(axis=0)
        result[3, 0] = edge.sum()
        result[3, 1] = weighted.sum()
        return cp.asnumpy(result)

    def windows_from_scores(self, base, scores, width, include=None, count=1):
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

    def adjoint(self, value, left, right, target):
        """Transpose of a fixed-window transition, without changing its weights.

        The forward weight depends on target XOR right and the unchanged left
        word, so exchange the target/right windows and transpose both matrices.
        """
        if value.shape != (1 << len(target.bits), 1 << len(left.bits)):
            raise ValueError("adjoint value does not match forward output windows")
        transposed = cp.ascontiguousarray(value.T)
        result = self.step(transposed, left, target, right)
        return cp.ascontiguousarray(result.T)

    def retained_mass(self, prob, left, right, target, basis=None):
        """Exact projected-affine mass; no output matrix or cross-device exchange."""
        if (
            prob.shape != (1 << len(left.bits), 1 << len(right.bits))
            or prob.dtype != cp.float64
            or not prob.flags.c_contiguous
        ):
            raise ValueError(
                "expected a contiguous FP64 probability matrix matching windows"
            )
        if max(len(left.bits), len(right.bits), len(target.bits)) > 20:
            raise ValueError("CUDA windows support at most 20 bits")
        nl, nr = prob.shape
        bases, vectors, ranks, _ = self.basis(left) if basis is None else basis
        bases, vectors = bases.copy(), vectors.copy()
        outside = cp.empty(nl, dtype=cp.int32)
        self.launch(
            "restrict_basis",
            nl,
            (np.uint64(nl), np.uint64(target.mask), bases, vectors, ranks, outside),
        )
        width = len(right.bits)
        chunks = max(1, (width + 7) // 8)
        columns = cp.empty((width + 1, nl), dtype=cp.uint64)
        tables = cp.empty((chunks * 256, nl), dtype=cp.uint64)
        self.launch(
            "mass_columns",
            columns.size,
            (
                np.uint64(nl),
                np.int32(width),
                cp.asarray(right.bits, dtype=cp.int32),
                np.uint64(target.mask),
                np.uint64(target.base),
                np.uint64(right.base),
                bases,
                vectors,
                outside,
                columns,
            ),
        )
        self.launch(
            "affine_tables",
            tables.size,
            (np.uint64(nl), np.int32(width), np.int32(chunks), columns, tables),
        )
        tiles = (nr + 4095) // 4096
        partial = cp.empty(nl * tiles, dtype=cp.float64)
        self.launch(
            "retained_mass_tiles",
            partial.size * 256,
            (
                prob,
                np.uint64(nl),
                np.uint64(nr),
                np.uint64(tiles),
                np.int32(chunks),
                tables,
                outside,
                partial,
            ),
            threads=256,
        )
        return float(partial.sum())

    def step_and_summary(self, prob, left, right, target, basis=None, out=None):
        """Fused output statistics when supported; sync once to return scalar results."""
        partials = []
        result = self.step(
            prob, left, right, target, basis, out, _summary_parts=partials
        )
        return result, self.summary(result, partials)

    def step(
        self, prob, left, right, target, basis=None, out=None, _summary_parts=None
    ):
        if any(len(w.bits) > 20 for w in (left, right, target)):
            raise ValueError("CUDA windows support at most 20 active bits")
        if prob.shape != (1 << len(left.bits), 1 << len(right.bits)):
            raise ValueError("probability matrix does not match windows")
        nl, nr = prob.shape
        expected_shape = (1 << len(target.bits), nl)
        if prob.dtype != cp.float64 or not prob.flags.c_contiguous:
            raise ValueError("probability matrix must be contiguous float64")
        if out is not None and (
            out.shape != expected_shape
            or out.dtype != cp.float64
            or not out.flags.c_contiguous
            or out.data.ptr == prob.data.ptr
        ):
            raise ValueError(
                "output must be a distinct contiguous float64 matrix of the expected shape"
            )
        required = prob.nbytes + 8 * (1 << len(target.bits)) * nl
        if required > self.limit:
            raise MemoryError(
                f"two matrices need {required / 2**30:.2f} GiB; limit={self.limit / 2**30:.2f} GiB"
            )
        free, _ = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        if (
            out is None
            and 8 * (1 << len(target.bits)) * nl > (free + pool.free_bytes()) * 0.85
        ):
            raise MemoryError("insufficient free device memory with 15% headroom")
        bases, vectors, ranks, _ = self.basis(left) if basis is None else basis
        # Do not mutate a caller's basis: useful for evaluating multiple windows.
        bases, vectors = bases.copy(), vectors.copy()
        outside = cp.empty(nl, dtype=cp.int32)
        restricted = (
            right if self.kernel in ("gather", "coset", "coset_lut") else target
        )
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
        if self.kernel in ("coset", "coset_lut"):
            return self.coset_step(
                prob,
                right,
                target,
                bases,
                vectors,
                packed,
                ranks,
                outside,
                out,
                _summary_parts,
            )
        allocate = cp.empty if self.kernel == "gather" else cp.zeros
        result = (
            out
            if out is not None
            else allocate((1 << len(target.bits), nl), dtype=cp.float64)
        )
        if out is not None and self.kernel == "scatter":
            result.fill(0)
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

    def coset_step(
        self,
        prob,
        right,
        target,
        bases,
        vectors,
        packed,
        ranks,
        outside,
        out=None,
        summary_parts=None,
    ):
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
        width = len(target.bits)
        chunks = max(1, (width + 7) // 8)
        lut_bytes = (
            nl * (256 * chunks + width + 1) * 8 if self.kernel == "coset_lut" else 0
        )
        required = prob.nbytes + 8 * buckets + output_bytes + lut_bytes
        if required > self.limit:
            raise MemoryError(
                f"matrices plus coset sums need {required / 2**30:.2f} GiB"
            )
        free, _ = cp.cuda.runtime.memGetInfo()
        if (
            8 * buckets + (output_bytes if out is None else 0) + lut_bytes
            > (free + cp.get_default_memory_pool().free_bytes()) * 0.85
        ):
            raise MemoryError(
                "insufficient memory for coset sums and output with 15% headroom"
            )
        if self.kernel == "coset_lut":
            sums = cp.empty(buckets, dtype=cp.float64)
            self.launch(
                "coset_reduce",
                nl * 256,
                (
                    prob,
                    np.uint64(nr),
                    ranks,
                    outside,
                    reducers,
                    masks,
                    offsets,
                    sizes,
                    sums,
                ),
                threads=256,
            )
        else:
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
        result = (
            out
            if out is not None
            else cp.empty((1 << len(target.bits), nl), dtype=cp.float64)
        )
        if self.kernel == "coset_lut":
            columns = cp.empty((width + 1, nl), dtype=cp.uint64)
            tables = cp.empty((chunks * 256, nl), dtype=cp.uint64)
            self.launch(
                "affine_columns",
                columns.size,
                (
                    np.uint64(nl),
                    np.int32(width),
                    cp.asarray(target.bits, dtype=cp.int32),
                    np.uint64(target.base),
                    np.uint64(right.mask),
                    np.uint64(right.base),
                    bases,
                    vectors,
                    ranks,
                    outside,
                    reducers,
                    masks,
                    columns,
                ),
            )
            self.launch(
                "affine_tables",
                tables.size,
                (np.uint64(nl), np.int32(width), np.int32(chunks), columns, tables),
            )
            args = (
                np.uint64(nl),
                np.uint64(result.shape[0]),
                np.int32(chunks),
                tables,
                sizes,
                offsets,
                ranks,
                sums,
                result,
            )
            if summary_parts is None:
                self.launch("coset_lookup_lut", result.size, args)
            else:
                blocks = min((result.size + 255) // 256, 8192)
                maxima = cp.empty(blocks, dtype=cp.float64)
                indices = cp.empty(blocks, dtype=cp.uint64)
                totals = cp.empty(blocks, dtype=cp.float64)
                self.launch(
                    "coset_lookup_lut_stats",
                    blocks * 256,
                    args + (maxima, indices, totals),
                    threads=256,
                )
                summary_parts.extend((maxima, indices, totals))
            return result
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
