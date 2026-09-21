"""Endpoint-guided coordinate ascent over a fixed window schedule.
Backward messages encode the entire remaining schedule, not a short horizon.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cupy as cp
import numpy as np

from .core import Window
from .gpu import Engine


def windows_at(initial, schedule, completed):
    left = schedule[completed - 1] if completed else initial[0]
    right = schedule[completed - 2] if completed >= 2 else initial[1 - completed]
    return left, right


def endpoint_value(prob, left, right, endpoint):
    i, j = left.index(endpoint[0]), right.index(endpoint[1])
    return float(prob[i, j]) if i is not None and j is not None else 0.0


def backward_messages(engine, initial, schedule, endpoint, cache="device"):
    count = len(schedule)
    left, right = windows_at(initial, schedule, count)
    terminal = cp.zeros((1 << len(left.bits), 1 << len(right.bits)), dtype=cp.float64)
    i, j = left.index(endpoint[0]), right.index(endpoint[1])
    if i is None or j is None:
        raise ValueError("endpoint is outside terminal windows")
    terminal[i, j] = 1
    messages = [None] * (count + 1)
    messages[count] = (cp.asnumpy(terminal) if cache == "host" else terminal, 0.0)
    for completed in range(count - 1, -1, -1):
        left, right = windows_at(initial, schedule, completed)
        prob = engine.adjoint(
            cp.asarray(messages[completed + 1][0]), left, right, schedule[completed]
        )
        peak = float(prob.max())
        if peak <= 0:
            raise ValueError("endpoint has no retained paths")
        prob *= 1 / peak
        messages[completed] = (
            cp.asnumpy(prob) if cache == "host" else prob,
            messages[completed + 1][1] + math.log2(peak),
        )
    return messages


def candidate_score(
    engine, prob, left, right, schedule, position, candidate, messages, endpoint
):
    """Three transitions restore common state windows; suffix contraction is exact."""
    stop = min(position + 3, len(schedule))
    value = prob
    scale = 0.0
    for index in range(position, stop):
        target = candidate if index == position else schedule[index]
        value, (peak, _, _) = engine.step_and_summary(value, left, right, target)
        if peak <= 0:
            return -math.inf
        value *= 1 / peak
        scale += math.log2(peak)
        left, right = target, left
    if stop == len(schedule):
        mass = endpoint_value(value, left, right, endpoint)
    else:
        backward, backward_scale = messages[stop]
        mass = float(cp.sum(value * cp.asarray(backward)))
        scale += backward_scale
    return scale + math.log2(mass) if mass > 0 else -math.inf


def optimize(
    engine,
    initial,
    schedule,
    endpoint,
    width,
    candidates,
    passes,
    report=None,
    cache="device",
):
    schedule = list(schedule)
    history = []
    for sweep in range(passes):
        messages = backward_messages(engine, initial, schedule, endpoint, cache)
        before = endpoint_value(
            messages[0][0], *initial, (initial[0].base, initial[1].base)
        )
        before = messages[0][1] + math.log2(before)
        prob = cp.ones((1, 1), dtype=cp.float64)
        left, right = initial
        prefix_scale = 0.0
        accepted = 0
        for position, original in enumerate(schedule):
            proposed = engine.candidates(
                prob, right, engine.basis(left), width, count=candidates
            )
            choices = [original] + [window for window in proposed if window != original]
            stop = min(position + 3, len(schedule))
            saved_message = messages[stop]
            if stop < len(schedule) and isinstance(saved_message[0], np.ndarray):
                messages[stop] = (cp.asarray(saved_message[0]), saved_message[1])
            best, best_score = original, -math.inf
            original_score = None
            for candidate in choices:
                score = candidate_score(
                    engine,
                    prob,
                    left,
                    right,
                    schedule,
                    position,
                    candidate,
                    messages,
                    endpoint,
                )
                if original_score is None:
                    original_score = score
                if score > best_score + 1e-12:
                    best, best_score = candidate, score
            messages[stop] = saved_message
            if best != original:
                accepted += 1
                change = dict(
                    sweep=sweep + 1,
                    round=position + 1,
                    before=prefix_scale + original_score,
                    after=prefix_scale + best_score,
                    old_base=hex(original.base),
                    old_bits=list(original.bits),
                    new_base=hex(best.base),
                    new_bits=list(best.bits),
                )
                history.append(change)
                if report:
                    report(change)
            schedule[position] = best
            prob, (peak, _, _) = engine.step_and_summary(prob, left, right, best)
            prob *= 1 / peak
            prefix_scale += math.log2(peak)
            left, right = best, left
        after = prefix_scale + math.log2(endpoint_value(prob, left, right, endpoint))
        if after < before - 1e-10:
            raise AssertionError("coordinate sweep decreased the target probability")
        if report:
            report(dict(sweep=sweep + 1, before=before, after=after, accepted=accepted))
        del messages, prob
        if not accepted:
            break
    return schedule, history


def write_replay(
    engine, config, initial, schedule, endpoint, history, output, search_seconds
):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file:
        file.write(
            json.dumps(
                dict(
                    config=config,
                    algorithm="endpoint coordinate ascent with adjoint suffix",
                    endpoint=[
                        hex(v)
                        for v in (
                            endpoint[::-1] if config["mode"] == "linear" else endpoint
                        )
                    ],
                    changes=history,
                    optimization_seconds=search_seconds,
                    kernel_sha256=hashlib.sha256(
                        Path(__file__).with_name("kernels.cu").read_bytes()
                    ).hexdigest(),
                )
            )
            + "\n"
        )
        prob = cp.ones((1, 1), dtype=cp.float64)
        left, right = initial
        scale = 0.0
        for number, target in enumerate(schedule, 1):
            start = time.perf_counter()
            prob, (peak, total, index) = engine.step_and_summary(
                prob, left, right, target
            )
            prob *= 1 / peak
            scale += math.log2(peak)
            row, col = divmod(index, prob.shape[1])
            physical = [int(target.values()[row]), int(left.values()[col])]
            if config["mode"] == "linear":
                physical.reverse()
            cp.cuda.Stream.null.synchronize()
            record = dict(
                round=number,
                log2_max=scale,
                log2_mass=scale + math.log2(total / peak),
                output=[hex(v) for v in physical],
                window_base=hex(target.base),
                window_bits=list(target.bits),
                seconds=time.perf_counter() - start,
            )
            if number == len(schedule):
                record["log2_target"] = scale + math.log2(
                    endpoint_value(prob, target, left, endpoint)
                )
            file.write(json.dumps(record, allow_nan=False) + "\n")
            left, right = target, left
        file.write(json.dumps(dict(completed_rounds=len(schedule))) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cache", choices=("host", "device"), default="device")
    parser.add_argument("--host-memory-gib", type=float, default=128)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--memory-gib", type=float, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(row) for row in args.reference.read_text().splitlines()]
    config = records[0]["config"].copy()
    rows = [r for r in records if "round" in r]
    schedule = [Window(int(r["window_base"], 0), tuple(r["window_bits"])) for r in rows]
    if (
        not rows
        or config["width"] > 20
        or not 1 <= args.candidates <= 64
        or args.passes < 1
        or args.memory_gib <= 0
        or args.host_memory_gib <= 0
    ):
        parser.error(
            "requires a nonempty width <=20 schedule, 1..64 candidates, positive passes/budget"
        )
    if args.output.exists():
        parser.error("output already exists")
    cp.cuda.Device(args.device).use()
    cp.get_default_memory_pool().set_limit(size=int(args.memory_gib * 2**30))
    engine = Engine(
        config["word_bits"], config["cipher"], config["mode"], args.memory_gib
    )
    initial = tuple(
        map(
            Window,
            (config["right"], config["left"])
            if config["mode"] == "linear"
            else (config["left"], config["right"]),
        )
    )
    endpoint = tuple(int(v, 0) for v in rows[-1]["output"])
    if config["mode"] == "linear":
        endpoint = endpoint[::-1]
    needed = sum(
        (1 << len(windows_at(initial, schedule, i)[0].bits))
        * (1 << len(windows_at(initial, schedule, i)[1].bits))
        * 8
        for i in range(len(schedule) + 1)
    )
    device_cache = needed if args.cache == "device" else 0
    if (
        device_cache * 1.3 + 6 * 8 * 2 ** (2 * config["width"])
        > args.memory_gib * 2**30
    ):
        parser.error("backward cache plus workspace exceeds conservative GPU budget")
    if args.cache == "host" and needed > args.host_memory_gib * 2**30:
        parser.error("backward cache exceeds host memory budget")
    start = time.perf_counter()
    schedule, history = optimize(
        engine,
        initial,
        schedule,
        endpoint,
        config["width"],
        args.candidates,
        args.passes,
        report=lambda record: print(json.dumps(record), flush=True),
        cache=args.cache,
    )
    elapsed = time.perf_counter() - start
    config.update(
        device=args.device,
        rounds=len(schedule),
        output=str(args.output),
        reference=str(args.reference),
        candidates=args.candidates,
        passes=args.passes,
        memory_gib=args.memory_gib,
        cache=args.cache,
        host_memory_gib=args.host_memory_gib,
        kernel="coset_lut",
        statistics="hierarchical",
    )
    write_replay(
        engine, config, initial, schedule, endpoint, history, args.output, elapsed
    )


if __name__ == "__main__":
    main()
