"""Beam search over window histories; beams are alternatives, never summed."""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cupy as cp

from .core import Window
from .gpu import Engine


@dataclass
class State:
    prob: object
    left: Window
    right: Window
    scale: float
    mass: float
    records: list


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--word-bits", type=int, choices=[16, 24, 32, 48, 64], default=32
    )
    parser.add_argument("--cipher", choices=["simon", "simeck"], default="simon")
    parser.add_argument(
        "--mode", choices=["difference", "linear"], default="difference"
    )
    parser.add_argument("--left", type=lambda x: int(x, 0), default=0)
    parser.add_argument("--right", type=lambda x: int(x, 0), default=1)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=23)
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--objective", choices=["peak", "mass"], default="peak")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--memory-gib", type=float, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not 1 <= args.width <= 12
        or not 1 <= args.beam_size <= 32
        or not 1 <= args.candidates <= 32
        or args.rounds < 1
    ):
        parser.error(
            "beam search supports width 1..12, beam/candidates 1..32, rounds >=1"
        )
    if (
        not 0 <= args.left < 1 << args.word_bits
        or not 0 <= args.right < 1 << args.word_bits
        or args.left == args.right == 0
    ):
        parser.error("nonzero input must fit word size")
    if args.memory_gib <= 0:
        parser.error("positive memory budget required")
    cp.cuda.Device(args.device).use()
    cp.get_default_memory_pool().set_limit(size=int(args.memory_gib * 2**30))
    engine = Engine(
        args.word_bits, args.cipher, args.mode, args.memory_gib, kernel="coset"
    )
    initial_left, initial_right = (
        (args.right, args.left) if args.mode == "linear" else (args.left, args.right)
    )
    states = [
        State(
            cp.ones((1, 1), dtype=cp.float64),
            Window(initial_left),
            Window(initial_right),
            0.0,
            0.0,
            [],
        )
    ]

    def score(state):
        return (
            (state.scale, state.mass)
            if args.objective == "peak"
            else (state.mass, state.scale)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        config = vars(args).copy()
        config["output"] = str(args.output)
        out.write(
            json.dumps(
                {
                    "config": config,
                    "search": "window-history beam; no cross-beam summation",
                    "kernel": "coset",
                    "gpu": cp.cuda.runtime.getDeviceProperties(args.device)[
                        "name"
                    ].decode(),
                    "cupy": cp.__version__,
                }
            )
            + "\n"
        )
        out.flush()
        start = time.perf_counter()
        for round_no in range(1, args.rounds + 1):
            best = []
            for state in states:
                basis = engine.basis(state.left)
                windows = engine.candidates(
                    state.prob, state.right, basis, args.width, count=args.candidates
                )
                for target in windows:
                    prob = engine.step(
                        state.prob, state.left, state.right, target, basis
                    )
                    peak = float(prob.max())
                    if peak == 0:
                        del prob
                        continue
                    scale = state.scale + math.log2(peak)
                    mass = state.scale + math.log2(float(prob.sum()))
                    i, j = divmod(int(cp.argmax(prob)), prob.shape[1])
                    endpoint = [
                        hex(int(target.values()[i])),
                        hex(int(state.left.values()[j])),
                    ]
                    if args.mode == "linear":
                        endpoint.reverse()
                    prob /= peak
                    record = dict(
                        round=round_no,
                        log2_max=scale,
                        log2_mass=mass,
                        output=endpoint,
                        window_base=hex(target.base),
                        window_bits=list(target.bits),
                        shape=list(prob.shape),
                    )
                    best.append(
                        State(
                            prob,
                            target,
                            state.left,
                            scale,
                            mass,
                            state.records + [record],
                        )
                    )
                    best.sort(key=score, reverse=True)
                    del best[args.beam_size :]
                    del prob
            if not best:
                raise RuntimeError(f"no surviving hypotheses at round {round_no}")
            states = best
            print(
                f"round={round_no} beam={len(states)} log2_max={max(s.scale for s in states):.9f} elapsed={time.perf_counter() - start:.2f}",
                flush=True,
            )
        # Report the best endpoint probability even if pruning used retained mass.
        winner = max(states, key=lambda s: s.scale)
        for record in winner.records:
            out.write(json.dumps(record, allow_nan=False) + "\n")
        out.write(
            json.dumps(
                {
                    "search_seconds": time.perf_counter() - start,
                    "completed_rounds": args.rounds,
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
