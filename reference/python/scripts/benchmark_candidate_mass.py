"""Alternate full-output and projected mass evaluation on the exact same input."""

import argparse
import hashlib
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
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
records = [json.loads(x) for x in a.reference.read_text().splitlines()]
c = records[0]["config"]
schedule = [r for r in records if "round" in r]
if not 1 <= a.round <= len(schedule) or a.repeats < 1:
    p.error("invalid round or repeats")
cp.cuda.Device(a.device).use()
cp.get_default_memory_pool().set_limit(size=28 * 2**30)
engine = Engine(c["word_bits"], c["cipher"], c["mode"], 28)
initial = (c["right"], c["left"]) if c["mode"] == "linear" else (c["left"], c["right"])
left, right = map(Window, initial)
prob = cp.ones((1, 1), dtype=cp.float64)
for row in schedule[: a.round - 1]:
    target = Window(int(row["window_base"], 0), tuple(row["window_bits"]))
    new, (peak, _, _) = engine.step_and_summary(prob, left, right, target)
    new *= 1 / peak
    prob, left, right = new, target, left
    cp.get_default_memory_pool().free_all_blocks()
row = schedule[a.round - 1]
target = Window(int(row["window_base"], 0), tuple(row["window_bits"]))
basis = engine.basis(left)
output = cp.empty((1 << len(target.bits), prob.shape[0]), dtype=cp.float64)
results = []
for repeat in range(a.repeats + 1):
    for method in (
        ("matrix", "projected") if repeat % 2 == 0 else ("projected", "matrix")
    ):
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        if method == "matrix":
            engine.step(prob, left, right, target, basis, out=output)
            mass = float(output.sum())
        else:
            mass = engine.retained_mass(prob, left, right, target, basis)
        elapsed = time.perf_counter() - start
        if repeat:
            results.append(
                dict(repeat=repeat, method=method, seconds=elapsed, mass=mass)
            )
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open("x") as file:
    json.dump(
        dict(
            reference=str(a.reference),
            round=a.round,
            device=a.device,
            input_bytes=prob.nbytes,
            output_bytes=output.nbytes,
            kernel_sha256=hashlib.sha256(
                (Path(__file__).resolve().parents[3] / "cuda/kernels.cu").read_bytes()
            ).hexdigest(),
            results=results,
        ),
        file,
        indent=2,
    )
print(json.dumps(results), flush=True)
