"""Allocate window bits by exact endpoint gain, using native Rust GPU scores.

Each iteration evaluates all single-bit additions at every eligible round,
replays each round's winner, and keeps the globally strongest endpoint gain.
This is a greedy state-budget allocation, not a global optimum or a claim
that gains measured on different plans can be added together.
"""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--memory-gib", type=float, default=192.0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    devices = [int(x) for x in args.devices.split(",")]
    if len(set(devices)) != len(devices) or not devices or args.iterations < 1:
        parser.error("need distinct devices and positive iterations")
    args.output.mkdir(parents=True, exist_ok=False)
    current = args.plan
    history = []
    for iteration in range(args.iterations):
        root = args.output / f"iteration-{iteration + 1}"
        root.mkdir()
        records = [json.loads(x) for x in current.read_text().splitlines()]
        rounds = [
            x["round"]
            for x in records
            if "round" in x and 0 < len(x["window_bits"]) < 20
        ]

        def worker(slot):
            finished = []
            for round_ in rounds[slot :: len(devices)]:
                path = root / f"round-{round_}.json"
                command = [
                    str(args.binary),
                    "affine-refine",
                    str(current),
                    "--round",
                    str(round_),
                    "--device",
                    str(devices[slot]),
                    "--memory-gib",
                    str(args.memory_gib),
                    "--output",
                    str(path),
                ]
                with path.with_suffix(".log").open("w") as log:
                    subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
                result = json.loads(path.read_text())
                # Successful runs are fully described by JSON/JSONL artifacts;
                # retain verbose logs only when a subprocess or parsing fails.
                path.with_suffix(".log").unlink()
                finished.append((path, result))
                print(
                    f"iteration={iteration + 1} device={devices[slot]} round={round_} done",
                    flush=True,
                )
            return finished

        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            results = [
                item
                for batch in pool.map(worker, range(len(devices)))
                for item in batch
            ]
        candidates = [(p, x) for p, x in results if x["best_expansion"] is not None]
        if not candidates:
            history.append({"source": str(current), "improved": False})
            break
        path, selected = max(
            candidates, key=lambda pair: pair[1]["best_expansion"]["log2_target"]
        )
        choice = selected["best_expansion"]
        # The next iteration's fixed endpoint must remain the same as the source;
        # don't silently change objectives if another endpoint becomes the peak.
        replay = [json.loads(x) for x in Path(choice["path"]).read_text().splitlines()]
        last = [x for x in replay if "round" in x][-1]
        if [int(x, 0) for x in last["output"]] != selected["endpoint"]:
            raise RuntimeError(
                "selected plan changed peak endpoint; explicit target continuation needed"
            )
        if abs(last["log2_max"] - choice["log2_target"]) > 1e-9:
            raise RuntimeError("selected full replay score mismatch")
        entry = {
            "source": str(current),
            "improved": True,
            "profile": str(path),
            "round": selected["round"],
            "added_bit": choice["added_bit"],
            "before": selected["before"],
            "after": choice["log2_target"],
            "plan": choice["path"],
            "rounds_evaluated": len(results),
        }
        history.append(entry)
        current = Path(choice["path"])
        print(json.dumps(entry), flush=True)
        (args.output / "summary.json").write_text(
            json.dumps({"history": history, "best_plan": str(current)}, indent=2) + "\n"
        )
    (args.output / "summary.json").write_text(
        json.dumps({"history": history, "best_plan": str(current)}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
