"""Explore rotationally distinct inputs drawn from recorded trajectory endpoints."""

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path


def canonical(left, right, n):
    mask = (1 << n) - 1

    def rotate(x, k):
        return ((x << k) | (x >> ((n - k) % n))) & mask

    return min((rotate(left, k), rotate(right, k)) for k in range(n))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("--rounds", type=int, required=True)
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(x) for x in a.reference.read_text().splitlines()]
    config = rows[0]["config"]
    n = config["word_bits"]
    seeds = {(config["left"], config["right"])}
    for row in rows:
        if "round" in row:
            x, y = map(lambda v: int(v, 0), row["output"])
            seeds.update(((x, y), (y, x)))
    seeds = sorted({canonical(x, y, n) for x, y in seeds if x or y})
    devices = [int(v) for v in a.devices.split(",")]

    def worker(slot):
        records = []
        for index in range(slot, len(seeds), len(devices)):
            x, y = seeds[index]
            out = a.output_dir / f"input-{index:03}.jsonl"
            command = [
                sys.executable,
                "-m",
                "ddw",
                "--device",
                str(devices[slot]),
                "--word-bits",
                str(n),
                "--cipher",
                config["cipher"],
                "--mode",
                config["mode"],
                "--left",
                str(x),
                "--right",
                str(y),
                "--width",
                str(a.width),
                "--rounds",
                str(a.rounds),
                "--memory-gib",
                "4",
                "--output",
                str(out),
            ]
            with out.with_suffix(".stdout").open("x") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            run = [json.loads(line) for line in out.read_text().splitlines()]
            last = next(r for r in reversed(run) if "round" in r)
            record = dict(
                left=hex(x),
                right=hex(y),
                round=last["round"],
                log2_max=last["log2_max"],
                file=str(out),
            )
            print(json.dumps(record), flush=True)
            records.append(record)
        return records

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        results = sum(executor.map(worker, range(len(devices))), [])
    results.sort(key=lambda r: (r["round"] != a.rounds, -r["log2_max"]))
    with (a.output_dir / "ranking.json").open("x") as out:
        json.dump(
            dict(
                reference=str(a.reference), rotation_deduplicated=True, results=results
            ),
            out,
            indent=2,
        )


if __name__ == "__main__":
    main()
