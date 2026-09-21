"""Interleave old/new complete runs on the SAME GPU; keep every raw timing.
The baseline checkout must be the unchanged 0f62945 implementation.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--baseline-root", type=Path, required=True)
p.add_argument("--device", type=int, default=0)
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--output-dir", type=Path, required=True)
a = p.parse_args()
baseline_revision = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=a.baseline_root, text=True
).strip()
if (
    not baseline_revision.startswith("0f62945")
    or subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=a.baseline_root,
        text=True,
    ).strip()
):
    p.error("baseline must be an unchanged checkout of 0f62945")
root = Path(__file__).resolve().parents[1]
a.output_dir = a.output_dir.resolve()
a.output_dir.mkdir(parents=True, exist_ok=True)
reference = root / "results/simon64-diff-w15.jsonl"
for repeat in range(a.repeats):
    for variant in ("before", "after") if repeat % 2 == 0 else ("after", "before"):
        output = a.output_dir / f"{variant}-{repeat}.jsonl"
        command = [
            sys.executable,
            "-m",
            "ddw",
            "--device",
            str(a.device),
            "--width",
            "15",
            "--rounds",
            "23",
            "--memory-gib",
            "28",
            "--include-windows",
            str(reference),
            "--output",
            str(output),
        ]
        if variant == "after":
            command += ["--kernel", "coset_lut", "--statistics", "hierarchical"]
        with output.with_suffix(".stdout").open("x") as log:
            subprocess.run(
                command,
                cwd=a.baseline_root if variant == "before" else root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        rows = [json.loads(x) for x in output.read_text().splitlines()]
        print(
            variant,
            repeat,
            "seconds",
            sum(r.get("seconds", 0) for r in rows),
            flush=True,
        )
