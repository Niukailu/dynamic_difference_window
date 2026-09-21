"""Run with python -m ddw --help."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from .core import Window


def main():
    p = argparse.ArgumentParser(
        description="SIMON/SIMECK dynamic windows on one CUDA GPU"
    )
    p.add_argument("--cipher", choices=["simon", "simeck"], default="simon")
    p.add_argument("--mode", choices=["difference", "linear"], default="difference")
    p.add_argument("--word-bits", type=int, choices=[16, 24, 32, 48, 64], default=32)
    p.add_argument(
        "--left",
        type=lambda x: int(x, 0),
        default=0,
        help="physical input left word, also in linear mode",
    )
    p.add_argument("--right", type=lambda x: int(x, 0), default=1)
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--rounds", type=int, default=23)
    p.add_argument("--kernel", choices=["gather", "scatter", "coset"], default="coset")
    p.add_argument("--strategy", choices=["legacy", "marginal"], default="legacy")
    p.add_argument(
        "--lookahead-candidates",
        type=int,
        default=1,
        help="evaluate this many one-bit window exchanges (1 disables)",
    )
    p.add_argument("--lookahead-objective", choices=["peak", "mass"], default="peak")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--memory-gib", type=float, default=12)
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new JSONL file; refuses to overwrite",
    )
    schedules = p.add_mutually_exclusive_group()
    schedules.add_argument(
        "--include-windows",
        type=Path,
        help="enlarge a previous run while retaining all its windows",
    )
    schedules.add_argument(
        "--replay-windows",
        type=Path,
        help="use the exact recorded windows without rescoring",
    )
    args = p.parse_args()
    if (
        not 0 <= args.width <= min(args.word_bits, 20)
        or args.rounds < 1
        or args.memory_gib <= 0
        or not 1 <= args.lookahead_candidates <= 64
    ):
        p.error("invalid width, rounds, or memory budget")
    if (
        not 0 <= args.left < 1 << args.word_bits
        or not 0 <= args.right < 1 << args.word_bits
        or args.left == args.right == 0
    ):
        p.error("input must fit the word size and be nonzero")
    schedule = []
    schedule_path = args.replay_windows or args.include_windows
    if schedule_path:
        records = [json.loads(line) for line in schedule_path.read_text().splitlines()]
        config = records[0]["config"]
        for key in ("cipher", "mode", "word_bits", "left", "right"):
            if config[key] != getattr(args, key):
                p.error(f"included run differs in {key}")
        schedule = [r for r in records[1:] if "round" in r]
        if len(schedule) < args.rounds or any(
            r["round"] != i + 1 for i, r in enumerate(schedule)
        ):
            p.error(
                "included run must have a contiguous schedule covering every requested round"
            )
        if any(len(r["window_bits"]) > args.width for r in schedule[: args.rounds]):
            p.error("included windows exceed width")
    import cupy as cp

    from .gpu import Engine

    cp.cuda.Device(args.device).use()
    cp.get_default_memory_pool().set_limit(size=int(args.memory_gib * 2**30))
    engine = Engine(
        args.word_bits, args.cipher, args.mode, args.memory_gib, args.kernel
    )
    # Linear recurrence uses swapped branches internally; logs always use physical order.
    initial_left, initial_right = (
        (args.right, args.left) if args.mode == "linear" else (args.left, args.right)
    )
    left, right = Window(initial_left), Window(initial_right)
    prob = cp.ones((1, 1), dtype=cp.float64)
    log_scale = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        config = vars(args).copy()
        config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
        props = cp.cuda.runtime.getDeviceProperties(args.device)
        output.write(
            json.dumps(
                {
                    "config": config,
                    "gpu": props["name"].decode(),
                    "cupy": cp.__version__,
                    "numpy": np.__version__,
                    "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                    "precision": "float64, maximum rescaled each round",
                    "kernel_sha256": hashlib.sha256(
                        Path(__file__).with_name("kernels.cu").read_bytes()
                    ).hexdigest(),
                    "projection": "outside bits remain masked after every elimination",
                    "timing": "synchronized wall seconds; round 1 includes JIT",
                }
            )
            + "\n"
        )
        output.flush()
        search_start = time.perf_counter()
        completed = 0
        for round_no in range(1, args.rounds + 1):
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            basis = engine.basis(left)
            ref = schedule[round_no - 1] if schedule else None
            include = (
                Window(int(ref["window_base"], 0), tuple(ref["window_bits"]))
                if ref
                else None
            )
            candidates = (
                [include]
                if args.replay_windows
                else engine.candidates(
                    prob,
                    right,
                    basis,
                    args.width,
                    include,
                    args.strategy,
                    args.lookahead_candidates,
                )
            )
            target = candidates[0]
            if len(candidates) > 1:
                # Evaluate sequentially and release each full matrix before the next.
                # Recompute the winning candidate to keep the two-matrix memory bound.
                best_score = -1.0
                for candidate in candidates:
                    trial = engine.step(prob, left, right, candidate, basis)
                    score = float(
                        trial.max()
                        if args.lookahead_objective == "peak"
                        else trial.sum()
                    )
                    if score > best_score:
                        best_score, target = score, candidate
                    del trial
            next_prob = engine.step(prob, left, right, target, basis)
            peak = float(next_prob.max())
            if not math.isfinite(peak) or peak < 0:
                raise FloatingPointError("invalid probability")
            if peak == 0:
                output.write(
                    json.dumps(
                        {
                            "stopped_round": round_no,
                            "reason": "empty retained distribution",
                        }
                    )
                    + "\n"
                )
                print("No paths remain in window.", flush=True)
                break
            log_scale += math.log2(peak)
            next_prob /= peak
            index = int(cp.argmax(next_prob))
            row, col = divmod(index, next_prob.shape[1])
            endpoint = (int(target.values()[row]), int(left.values()[col]))
            if args.mode == "linear":
                endpoint = endpoint[::-1]
            record = {
                "round": round_no,
                "log2_max": log_scale,
                "output": [hex(v) for v in endpoint],
                "window_base": hex(target.base),
                "window_bits": list(target.bits),
                "shape": list(next_prob.shape),
                "log2_mass": log_scale + math.log2(float(next_prob.sum())),
                "candidates_evaluated": len(candidates),
            }
            if ref:
                physical = [int(v, 0) for v in ref["output"]]
                il, ir = physical[::-1] if args.mode == "linear" else physical
                ti, li = target.index(il), left.index(ir)
                value = (
                    float(next_prob[ti, li]) if ti is not None and li is not None else 0
                )
                record["reference_output"] = ref["output"]
                record["reference_log2_max"] = ref["log2_max"]
                record["log2_at_reference_output"] = (
                    log_scale + math.log2(value) if value > 0 else None
                )
            prob, right, left = next_prob, left, target
            del basis
            # Return free giant matrix blocks before small next-round allocations
            # can split them and prevent reuse under the configured pool limit.
            if args.width >= 15:
                cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Stream.null.synchronize()
            record["seconds"] = time.perf_counter() - start
            completed = round_no
            output.write(json.dumps(record, allow_nan=False) + "\n")
            output.flush()
            print(
                f"round={round_no:2d} log2_max={log_scale:.9f} mass={record['log2_mass']:.6f} shape={record['shape']} seconds={record['seconds']:.3f}",
                flush=True,
            )
        output.write(
            json.dumps(
                {
                    "completed_rounds": completed,
                    "search_seconds": time.perf_counter() - search_start,
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
