"""Search one-bit input neighbors using independent native Rust/CUDA jobs."""

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path


def canonical(pair, bits):
    mask = (1 << bits) - 1
    return min(
        tuple(((v << k) | (v >> ((bits - k) % bits))) & mask for v in pair)
        for k in range(bits)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("--rust", type=Path, required=True)
    p.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--rounds", type=int, default=25)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    devices = list(map(int, a.devices.split(",")))
    if len(devices) != len(set(devices)) or not devices:
        p.error("devices must be nonempty and unique")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(a.reference.read_text().splitlines()[0])["config"]
    seed = (config["left"], config["right"])
    inputs = {canonical(seed, config["word_bits"])}
    for side in range(2):
        for bit in range(config["word_bits"]):
            pair = list(seed)
            pair[side] ^= 1 << bit
            if any(pair):
                inputs.add(canonical(pair, config["word_bits"]))
    inputs = sorted(inputs)

    def worker(rank, device):
        found = []
        for index in range(rank, len(inputs), len(devices)):
            left, right = inputs[index]
            output = a.output_dir / f"input-{index:03d}.jsonl"
            command = [
                str(a.rust.resolve()),
                "search",
                "--word-bits",
                str(config["word_bits"]),
                "--cipher",
                config["cipher"],
                "--mode",
                config["mode"],
                "--left",
                str(left),
                "--right",
                str(right),
                "--rounds",
                str(a.rounds),
                "--width",
                str(a.width),
                "--devices",
                str(device),
                "--memory-gib",
                "4",
                "--output",
                str(output),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"input {index}, device {device}: {result.stderr}")
            rows = [
                r
                for r in map(json.loads, output.read_text().splitlines())
                if "round" in r
            ]
            if len(rows) != a.rounds:
                raise ValueError("incomplete result")
            item = {
                "input": [hex(left), hex(right)],
                "log2_max": rows[-1]["log2_max"],
                "output": rows[-1]["output"],
                "path": str(output),
            }
            found.append(item)
            print(json.dumps(item), flush=True)
        return found

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [
            pool.submit(worker, rank, device) for rank, device in enumerate(devices)
        ]
        ranking = [row for future in futures for row in future.result()]
    ranking.sort(key=lambda r: r["log2_max"], reverse=True)
    (a.output_dir / "ranking.json").write_text(
        json.dumps(
            {
                "config": config,
                "width": a.width,
                "rounds": a.rounds,
                "inputs": len(inputs),
                "ranking": ranking,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
