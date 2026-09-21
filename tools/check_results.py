"""Check saved runs without requiring a GPU; fail on numerical regressions."""

import argparse
import json
import math
from pathlib import Path


def read(path):
    records = [json.loads(x) for x in path.read_text().splitlines()]
    rounds = [x for x in records[1:] if "round" in x]
    if not rounds or [x["round"] for x in rounds] != list(range(1, len(rounds) + 1)):
        raise ValueError(f"{path}: missing or unordered rounds")
    return records[0], rounds


def validate(path, reference=None, tolerance=6e-7):
    metadata, records = read(path)
    requested = metadata.get("config", {}).get("rounds")
    if requested is not None and requested != len(records):
        raise ValueError(f"{path}: incomplete run ({len(records)}/{requested} rounds)")
    prior_mass = 0
    for record in records:
        peak = record["log2_max"]
        if not math.isfinite(peak) or peak > 1e-12:
            raise AssertionError(f"{path}: invalid peak at round {record['round']}")
        if "log2_mass" in record:
            mass = record["log2_mass"]
            if (
                not math.isfinite(mass)
                or mass > prior_mass + 1e-10
                or mass < peak - 1e-10
            ):
                raise AssertionError(f"{path}: probability mass violation")
            prior_mass = mass
        if "log2_at_reference_output" in record:
            v = record["log2_at_reference_output"]
            if v is None or v < record["reference_log2_max"] - tolerance:
                raise AssertionError(
                    f"{path}: nested window lost reference probability"
                )
    error = None
    if reference:
        meta2, expected = read(reference)
        for key in ("cipher", "mode", "word_bits", "left", "right"):
            if metadata["config"][key] != meta2["config"][key]:
                raise ValueError(f"incompatible {key}")
        if len(records) > len(expected):
            raise ValueError("reference does not cover all rounds")
        error = max(
            abs(a["log2_max"] - b["log2_max"]) for a, b in zip(records, expected)
        )
        if error > tolerance:
            raise AssertionError(f"{path}: max log2 error {error} exceeds {tolerance}")
    return {
        "file": str(path),
        "rounds": len(records),
        "max_log2_error": error,
        "total_seconds": sum(x.get("seconds", 0) for x in records),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--tolerance", type=float, default=6e-7)
    args = p.parse_args()
    print(json.dumps(validate(args.run, args.reference, args.tolerance), indent=2))
