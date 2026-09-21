"""Reproduce and improve the 15 input points from the paper's result table.

One complete experiment per GPU. Commands and source/binary hashes are saved;
fixed replay, endpoint search, expansion and novel-path verification are distinct.
"""

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.check_results import read, validate
from tools.import_legacy_log import convert

CASES = [
    ("SIMON64", "DIFFERENCE", "0x4000000", "0x11000000"),
    ("SIMON64", "DIFFERENCE", "0x440", "0x1880"),
    ("SIMON64", "DIFFERENCE", "0x1", "0x40000004"),
    ("SIMON64", "DIFFERENCE", "0x80000", "0x222000"),
    ("SIMON64", "LINEAR", "0x40000004", "0x1"),
    ("SIMON64", "LINEAR", "0x44400", "0x1000"),
    ("SIMON96", "DIFFERENCE", "0x4000", "0x11101"),
    ("SIMON96", "LINEAR", "0x400000004044", "0x1"),
    ("SIMON96", "LINEAR", "0x400000000044", "0x1"),
    ("SIMON96", "LINEAR", "0x1", "0x0"),
    ("SIMON128", "DIFFERENCE", "0x1000", "0x4440"),
    ("SIMON128", "LINEAR", "0x4000000000000004", "0x1"),
    ("SIMON128", "LINEAR", "0x4000000000000044", "0x1"),
    ("SIMON128", "LINEAR", "0x1", "0x0"),
    ("SIMECK64", "DIFFERENCE", "0x0", "0x1"),
]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    devices = [int(x) for x in args.devices.split(",")]
    if not devices or len(devices) != len(set(devices)):
        parser.error("devices must be distinct")
    args.binary = args.binary.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    save(
        args.output / "manifest.json",
        {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "devices": devices,
            "cases": CASES,
            "scope": "All 15 inputs; last archived round; two local sweeps, width-16 expansion, one midpoint adjacent-window novel-path search. Not exhaustive search.",
            "memory_gib": 52,
            "host_memory_gib": 128,
            "block_pool": 16,
        },
    )

    def worker(slot):
        outcomes = []
        device = devices[slot]
        for index in range(slot, len(CASES), len(devices)):
            cipher, mode, left, right = CASES[index]
            root = args.output / f"{index + 1:02d}-{cipher.lower()}-{mode.lower()}"
            root.mkdir()
            source = (
                Path("experiments")
                / f"{cipher}_{mode}_FROM_{left}_{right}_PRECISION_14"
                / "info.log"
            )
            records = convert(source)
            records[0]["config"]["rounds"] = records[-1]["round"]
            original = root / "original.jsonl"
            original.write_text("".join(json.dumps(r) + "\n" for r in records))
            commands = []
            started = time.monotonic()

            def run(stage, arguments):
                path = root / f"{stage}.jsonl"
                command = [str(args.binary), *arguments, "--output", str(path)]
                entry = {"stage": stage, "command": command, "status": "running"}
                commands.append(entry)
                save(root / "commands.json", commands)
                print(f"case={index + 1} device={device} stage={stage}", flush=True)
                at = time.monotonic()
                with (root / f"{stage}.log").open("w") as log:
                    process = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT
                    )
                entry.update(
                    status="passed" if process.returncode == 0 else "failed",
                    seconds=time.monotonic() - at,
                )
                save(root / "commands.json", commands)
                if process.returncode:
                    raise RuntimeError(
                        f"{stage} exited {process.returncode}; see {root}"
                    )
                return path

            result = {
                "case": index + 1,
                "cipher": cipher,
                "mode": mode.lower(),
                "input": [left, right],
                "source": str(source),
                "rounds": records[-1]["round"],
                "before": records[-1]["log2_max"],
            }
            try:
                replay = run(
                    "replay",
                    [
                        "replay",
                        str(original),
                        "--devices",
                        str(device),
                        "--memory-gib",
                        "52",
                    ],
                )
                validate(replay)
                _, replay_rows = read(replay)
                result["historical_replay_max_error"] = max(
                    abs(a["log2_max"] - b["log2_max"])
                    for a, b in zip(records[1:], replay_rows)
                )
                refined = run(
                    "refined",
                    [
                        "refine",
                        str(replay),
                        "--device",
                        str(device),
                        "--memory-gib",
                        "52",
                        "--host-memory-gib",
                        "128",
                        "--passes",
                        "2",
                    ],
                )
                validate(refined)
                expanded = run(
                    "expanded",
                    [
                        "compressed",
                        str(refined),
                        "--width",
                        "16",
                        "--device",
                        str(device),
                        "--memory-gib",
                        "52",
                    ],
                )
                validate(expanded)
                novel = run(
                    "novel",
                    [
                        "block-refine",
                        str(refined),
                        "--reference",
                        str(expanded),
                        "--round",
                        str(result["rounds"] // 2),
                        "--pool",
                        "16",
                        "--device",
                        str(device),
                        "--memory-gib",
                        "52",
                    ],
                )
                validate(novel)
                meta, _ = read(novel)
                score = meta["optimization"]
                result.update(
                    status="passed",
                    after=score["log2_union"],
                    diff=score["log2_union"] - result["before"],
                    target=score["target"],
                    expanded=score["reference_log2"],
                    novel=score["after_novel"],
                    result_file=str(novel),
                    seconds=time.monotonic() - started,
                )
                threshold = -2 * records[0]["config"]["word_bits"]
                result["above_threshold"] = result["after"] > threshold
                result["old_above_threshold_rounds"] = max(
                    r["round"] for r in records[1:] if r["log2_max"] > threshold
                )
            except Exception as error:
                result.update(
                    status="failed",
                    error=str(error),
                    seconds=time.monotonic() - started,
                )
            save(root / "summary.json", result)
            outcomes.append(result)
            print(json.dumps(result), flush=True)
        return outcomes

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        outcomes = [
            row for batch in pool.map(worker, range(len(devices))) for row in batch
        ]
    outcomes.sort(key=lambda row: row["case"])
    save(args.output / "summary.json", outcomes)
    if any(row["status"] != "passed" for row in outcomes):
        raise SystemExit("Some experiments failed; see summaries")


if __name__ == "__main__":
    main()
