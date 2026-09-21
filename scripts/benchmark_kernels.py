"""Compare all CUDA algorithms sequentially on identical recorded windows."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("reference", type=Path)
p.add_argument("--device", type=int, default=0)
p.add_argument("--memory-gib", type=float, default=24)
p.add_argument("--output-dir", type=Path, required=True)
a = p.parse_args()
records = [json.loads(x) for x in a.reference.read_text().splitlines()]
c = records[0]["config"]
rounds = sum("round" in r for r in records)
a.output_dir.mkdir(parents=True, exist_ok=True)
for kernel in ("coset", "gather", "scatter"):
    command = [
        sys.executable,
        "-m",
        "ddw",
        "--device",
        str(a.device),
        "--memory-gib",
        str(a.memory_gib),
        "--kernel",
        kernel,
        "--replay-windows",
        str(a.reference),
        "--output",
        str(a.output_dir / (kernel + ".jsonl")),
    ]
    for key in ("word_bits", "cipher", "mode", "left", "right", "width"):
        command += ["--" + key.replace("_", "-"), str(c[key])]
    command += ["--rounds", str(rounds)]
    subprocess.run(command, check=True)
