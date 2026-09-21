"""Greedy sweeps of native adjacent-window searches, one worker per GPU.

All blocks in an iteration share a source plan. Only the best complete replay
is accepted; independently optimized blocks are never merged without scoring.
With --reference, the objective is P(candidate outside reference), not its peak.
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def log_value(value):
    return float("-inf") if value is None else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--rounds", required=True, help="one-based starting rounds, e.g. 8,9,10,11"
    )
    parser.add_argument("--pool", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--memory-gib", type=float, default=192.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    devices = [int(x) for x in args.devices.split(",")]
    rounds = [int(x) for x in args.rounds.split(",")]
    if (
        len(set(devices)) != len(devices)
        or len(set(rounds)) != len(rounds)
        or args.iterations < 1
    ):
        parser.error("need distinct devices/rounds and positive iterations")
    args.output.mkdir(parents=True, exist_ok=False)
    current = args.plan
    history = []
    started = time.monotonic()
    for iteration in range(args.iterations):
        root = args.output / f"iteration-{iteration + 1}"
        root.mkdir()

        def worker(slot):
            finished = []
            for round_ in rounds[slot :: len(devices)]:
                path = root / f"round-{round_}.jsonl"
                cmd = [
                    str(args.binary),
                    "block-refine",
                    str(current),
                    "--round",
                    str(round_),
                    "--pool",
                    str(args.pool),
                    "--device",
                    str(devices[slot]),
                    "--memory-gib",
                    str(args.memory_gib),
                    "--output",
                    str(path),
                ]
                if args.reference is not None:
                    cmd += ["--reference", str(args.reference)]
                log_path = path.with_suffix(".log")
                with log_path.open("w") as log:
                    subprocess.run(
                        cmd, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
                metadata = json.loads(path.read_text().splitlines()[0])["optimization"]
                log_path.unlink()
                finished.append((path, metadata))
                print(
                    f"iteration={iteration + 1} device={devices[slot]} round={round_} novel={metadata['after_novel']}",
                    flush=True,
                )
            return finished

        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            results = [
                result
                for batch in pool.map(worker, range(len(devices)))
                for result in batch
            ]
        path, metadata = max(
            results, key=lambda item: log_value(item[1]["after_novel"])
        )
        before, after = (
            log_value(metadata["before_novel"]),
            log_value(metadata["after_novel"]),
        )
        improved = after > before + 1e-12
        entry = {
            "source": str(current),
            "selected": str(path),
            "improved": improved,
            "before": metadata["before_novel"],
            "after": metadata["after_novel"],
            "after_target": metadata["after_target"],
            "block_round": metadata["round"],
            "blocks_evaluated": len(results),
        }
        history.append(entry)
        if improved:
            current = path
        summary = {
            "reference": str(args.reference) if args.reference else None,
            "objective": "novel paths" if args.reference else "endpoint weight",
            "pool": args.pool,
            "rounds": rounds,
            "devices": devices,
            "history": history,
            "best_plan": str(current),
            "seconds": time.monotonic() - started,
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
        if not improved:
            break


if __name__ == "__main__":
    main()
