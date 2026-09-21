"""Exact path-set union with the reflected schedule for swapped endpoints.
The stationary affine transition satisfies T[(l,r),(t,l)] = T[(l,t),(r,l)].
"""

import argparse
import json
import math
import time
from pathlib import Path

from .core import Window


def intersect(a, b):
    if (a.base ^ b.base) & ~(a.mask | b.mask):
        return None
    bits = tuple(sorted(set(a.bits) & set(b.bits)))
    return Window(a.base | b.base, bits)


def reflected(initial, schedule):
    if len(schedule) < 2:
        raise ValueError("at least two rounds required")
    return list(reversed([initial[1], initial[0], *schedule[:-2]]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("--rounds", type=int)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--memory-gib", type=float, default=28)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(x) for x in a.reference.read_text().splitlines()]
    config = rows[0]["config"].copy()
    records = [r for r in rows if "round" in r]
    rounds = a.rounds or len(records)
    if not 2 <= rounds <= len(records) or a.memory_gib <= 0:
        p.error("invalid rounds or budget")
    records = records[:rounds]
    endpoint = tuple(int(x, 0) for x in records[-1]["output"])
    if endpoint != (config["right"], config["left"]):
        p.error("endpoint must exactly swap the input branches")
    if a.output.exists():
        p.error("output exists")
    initial = tuple(
        map(
            Window,
            (config["right"], config["left"])
            if config["mode"] == "linear"
            else (config["left"], config["right"]),
        )
    )
    schedule = [
        Window(int(r["window_base"], 0), tuple(r["window_bits"])) for r in records
    ]
    reverse = reflected(initial, schedule)
    common = [intersect(x, y) for x, y in zip(schedule, reverse)]
    log_a = records[-1]["log2_max"]
    log_i = -math.inf
    started = time.perf_counter()
    if all(w is not None for w in common):
        import cupy as cp

        from .gpu import Engine

        cp.cuda.Device(a.device).use()
        cp.get_default_memory_pool().set_limit(size=int(a.memory_gib * 2**30))
        engine = Engine(
            config["word_bits"], config["cipher"], config["mode"], a.memory_gib
        )
        prob = cp.ones((1, 1), dtype=cp.float64)
        left, right = initial
        scale = 0.0
        for target in common:
            prob, (peak, _, _) = engine.step_and_summary(prob, left, right, target)
            if peak <= 0:
                break
            scale += math.log2(peak)
            prob *= 1 / peak
            left, right = target, left
        else:
            physical = endpoint[::-1] if config["mode"] == "linear" else endpoint
            ri, ci = left.index(physical[0]), right.index(physical[1])
            value = float(prob[ri, ci]) if ri is not None and ci is not None else 0.0
            if value:
                log_i = scale + math.log2(value)
    if log_i > log_a + 1e-10:
        raise AssertionError("intersection exceeds input path-set probability")
    log_union = log_a + math.log2(2 - 2 ** (log_i - log_a))
    result = dict(
        config=config,
        rounds=rounds,
        reference=str(a.reference),
        endpoint=[hex(x) for x in endpoint],
        log2_a=log_a,
        log2_reflection=log_a,
        log2_intersection=log_i if math.isfinite(log_i) else None,
        log2_union=log_union,
        threshold_log2=-2 * config["word_bits"],
        above_threshold=log_union > -2 * config["word_bits"],
        method="P(A union reflected(A)) = 2 P(A) - P(A intersection reflected(A)); swapped fixed endpoints",
        intersection_windows=[
            dict(base=hex(w.base), bits=list(w.bits)) if w else None for w in common
        ],
        seconds=time.perf_counter() - started,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("config", "intersection_windows")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
