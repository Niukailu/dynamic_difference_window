"""Inclusion-exclusion for at most four fixed-endpoint window path sets."""

import argparse
import itertools
import json
import math
import time
from pathlib import Path

from .core import Window
from .symmetric_union import intersect, reflected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("references", nargs="+", type=Path)
    p.add_argument("--rounds", type=int, required=True)
    p.add_argument("--reflect", action="store_true")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--memory-gib", type=float, default=80)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not 1 <= len(a.references) * (2 if a.reflect else 1) <= 4 or a.rounds < 2:
        p.error("requires 1..4 total sets and at least two rounds")
    if a.output.exists():
        p.error("output exists")
    config = None
    sets = []
    logs = []
    endpoint = None
    for path in a.references:
        data = [json.loads(x) for x in path.read_text().splitlines()]
        c = data[0]["config"]
        rows = [r for r in data if "round" in r][: a.rounds]
        if len(rows) != a.rounds:
            p.error("incomplete reference")
        current = tuple(int(v, 0) for v in rows[-1]["output"])
        if config is not None and (
            any(
                c[k] != config[k]
                for k in ("cipher", "mode", "word_bits", "left", "right")
            )
            or endpoint != current
        ):
            p.error("all sets must have the same input and final peak endpoint")
        config = c
        endpoint = current
        initial = tuple(
            map(
                Window,
                (c["right"], c["left"])
                if c["mode"] == "linear"
                else (c["left"], c["right"]),
            )
        )
        schedule = [
            Window(int(r["window_base"], 0), tuple(r["window_bits"])) for r in rows
        ]
        sets.append(schedule)
        logs.append(rows[-1]["log2_max"])
        if a.reflect:
            if endpoint != (c["right"], c["left"]):
                p.error("reflection requires swapped endpoints")
            sets.append(reflected(initial, schedule))
            logs.append(logs[-1])
    import cupy as cp

    from .gpu import Engine

    cp.cuda.Device(a.device).use()
    cp.get_default_memory_pool().set_limit(size=int(a.memory_gib * 2**30))
    engine = Engine(config["word_bits"], config["cipher"], config["mode"], a.memory_gib)
    physical = endpoint[::-1] if config["mode"] == "linear" else endpoint
    terms = []
    cache = {}
    started = time.perf_counter()

    def evaluate(windows):
        if any(w is None for w in windows):
            return -math.inf
        key = tuple(windows)
        if key in cache:
            return cache[key]
        prob = cp.ones((1, 1), dtype=cp.float64)
        left, right = initial
        scale = 0.0
        for target in windows:
            prob, (peak, _, _) = engine.step_and_summary(prob, left, right, target)
            if peak <= 0:
                return -math.inf
            scale += math.log2(peak)
            prob *= 1 / peak
            left, right = target, left
            cp.get_default_memory_pool().free_all_blocks()
        i, j = left.index(physical[0]), right.index(physical[1])
        value = float(prob[i, j]) if i is not None and j is not None else 0
        result = scale + math.log2(value) if value else -math.inf
        cache[key] = result
        return result

    for count in range(1, len(sets) + 1):
        for subset in itertools.combinations(range(len(sets)), count):
            if count == 1:
                logp = logs[subset[0]]
            else:
                windows = list(sets[subset[0]])
                for index in subset[1:]:
                    windows = [
                        intersect(x, y) if x is not None else None
                        for x, y in zip(windows, sets[index])
                    ]
                logp = evaluate(windows)
                if logp > min(logs[i] for i in subset) + 1e-10:
                    raise AssertionError("intersection exceeds a constituent")
            terms.append(
                dict(
                    subset=list(subset),
                    sign=1 if count % 2 else -1,
                    log2_probability=logp if math.isfinite(logp) else None,
                )
            )
            print(json.dumps(terms[-1]), flush=True)
    offset = max(logs)
    probability = math.fsum(
        t["sign"] * 2 ** (t["log2_probability"] - offset)
        for t in terms
        if t["log2_probability"] is not None
    )
    result = dict(
        config=config,
        references=list(map(str, a.references)),
        reflected=a.reflect,
        rounds=a.rounds,
        endpoint=list(map(hex, endpoint)),
        terms=terms,
        log2_union=offset + math.log2(probability),
        threshold_log2=-2 * config["word_bits"],
        seconds=time.perf_counter() - started,
    )
    if result["log2_union"] < max(logs) - 1e-10:
        raise AssertionError("union lost constituent mass")
    result["above_threshold"] = result["log2_union"] > result["threshold_log2"]
    with a.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("config", "terms")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
