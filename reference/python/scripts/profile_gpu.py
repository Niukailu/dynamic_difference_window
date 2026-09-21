"""CUDA-event stage/kernel timings on a fixed, reproducible round."""

import argparse
import json
import time
from pathlib import Path

import cupy as cp

from ddw.core import Window
from ddw.gpu import Engine

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("reference", type=Path)
p.add_argument("--round", type=int, default=12)
p.add_argument("--device", type=int, default=0)
p.add_argument("--repeats", type=int, default=5)
p.add_argument("--kernel", default="coset")
p.add_argument("--statistics", default="bitwise")
p.add_argument("--fused", action="store_true")
p.add_argument("--normalization", choices=["multiply", "divide"], default="multiply")
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
records = [json.loads(x) for x in a.reference.read_text().splitlines()]
c = records[0]["config"]
schedule = [r for r in records if "round" in r]
cp.cuda.Device(a.device).use()
cp.get_default_memory_pool().set_limit(size=28 * 2**30)
e = Engine(c["word_bits"], c["cipher"], c["mode"], 28, a.kernel, a.statistics)
initial_left, initial_right = (
    (c["right"], c["left"]) if c["mode"] == "linear" else (c["left"], c["right"])
)
left, right = Window(initial_left), Window(initial_right)
prob = cp.ones((1, 1), dtype=cp.float64)
for record in schedule[: a.round - 1]:
    target = Window(int(record["window_base"], 0), tuple(record["window_bits"]))
    new = e.step(prob, left, right, target)
    new /= new.max()
    prob, right, left = new, left, target
    cp.get_default_memory_pool().free_all_blocks()
density = float(cp.count_nonzero(prob)) / prob.size
print("input density", density, flush=True)
record = schedule[a.round - 1]
target = Window(int(record["window_base"], 0), tuple(record["window_bits"]))
launch = e.launch
kernel_events = []


def instrument(name, count, args, threads=128):
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    launch(name, count, args, threads=threads)
    end.record()
    kernel_events.append((name, start, end))


e.launch = instrument
runs = []
for repeat in range(a.repeats + 1):
    kernel_events.clear()
    stages = {}

    def measure(name, f):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        value = f()
        end.record()
        end.synchronize()
        stages[name] = cp.cuda.get_elapsed_time(start, end)
        return value

    start_wall = time.perf_counter()
    basis = measure("basis", lambda: e.basis(left))
    measure("choose", lambda: e.choose(prob, right, basis, c["width"]))
    result = measure("step", lambda: e.step(prob, left, right, target, basis))
    if a.fused:
        peak, total, index = measure("summary", lambda: e.summary(result))
        measure("normalize", lambda: cp.multiply(result, 1.0 / peak, out=result))
    else:
        peak = measure("max", lambda: result.max())
        measure("sum", lambda: result.sum())
        measure(
            "normalize",
            lambda: (
                cp.multiply(result, cp.reciprocal(peak), out=result)
                if a.normalization == "multiply"
                else cp.divide(result, peak, out=result)
            ),
        )
    kernels = {
        name: cp.cuda.get_elapsed_time(start, end) for name, start, end in kernel_events
    }
    entry = dict(
        stages_ms=stages,
        kernels_ms=kernels,
        wall_ms=1000 * (time.perf_counter() - start_wall),
    )
    if repeat:
        runs.append(entry)
    print(entry, flush=True)
    result = basis = peak = None
    cp.get_default_memory_pool().free_all_blocks()
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open("x") as f:
    json.dump(
        dict(
            reference=str(a.reference),
            round=a.round,
            device=a.device,
            kernel=a.kernel,
            statistics=a.statistics,
            normalization=a.normalization,
            fused=a.fused,
            density=density,
            matrix_bytes=prob.nbytes,
            output_bytes=(1 << len(target.bits)) * prob.shape[0] * 8,
            runs=runs,
        ),
        f,
        indent=2,
    )
