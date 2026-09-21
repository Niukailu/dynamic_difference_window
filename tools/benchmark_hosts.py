"""Alternate Rust/Python hosts on one GPU using the same fixed window plan."""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--memory-gib", type=float, default=28)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.plan.read_text().splitlines()[0])["config"]
    runs = []
    for repeat in range(args.repeats):
        for host in ("rust", "python") if repeat % 2 == 0 else ("python", "rust"):
            output = args.output_dir / f"{host}-{repeat}.jsonl"
            common = ["--memory-gib", str(args.memory_gib), "--output", str(output)]
            if host == "rust":
                command = [
                    str(args.rust.resolve()),
                    "replay",
                    str(args.plan),
                    "--devices",
                    str(args.device),
                    *common,
                ]
            else:
                command = [
                    sys.executable,
                    "-m",
                    "ddw",
                    "--device",
                    str(args.device),
                    "--replay-windows",
                    str(args.plan),
                    *common,
                ]
                for key in (
                    "word_bits",
                    "cipher",
                    "mode",
                    "left",
                    "right",
                    "width",
                    "rounds",
                ):
                    command += ["--" + key.replace("_", "-"), str(config[key])]
            start = time.perf_counter()
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            wall = time.perf_counter() - start
            records = [json.loads(line) for line in output.read_text().splitlines()]
            rounds = [r for r in records if "round" in r]
            reference = [
                r
                for r in map(json.loads, args.plan.read_text().splitlines())
                if "round" in r
            ]
            if len(rounds) != len(reference):
                raise ValueError("incomplete replay")
            error = max(
                abs(a["log2_max"] - b["log2_max"]) for a, b in zip(rounds, reference)
            )
            if error > 1e-10:
                raise ValueError(f"replay disagreement: {error}")
            run = {
                "host": host,
                "repeat": repeat,
                "wall_seconds": wall,
                "round_seconds": sum(r["seconds"] for r in rounds),
                "max_log2_error": error,
            }
            runs.append(run)
            print(json.dumps(run), flush=True)
    medians = {
        host: {
            metric: statistics.median(r[metric] for r in runs if r["host"] == host)
            for metric in ("wall_seconds", "round_seconds")
        }
        for host in ("rust", "python")
    }
    summary = {
        "plan": str(args.plan),
        "device": args.device,
        "shared_gpu": True,
        "runs": runs,
        "medians": medians,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
