"""Independently audit the important-input suite's logs and scalar identities."""

import argparse
import json
import math
from pathlib import Path

from tools.check_results import read, validate


def log_add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    largest = max(a, b)
    return largest + math.log2(math.fsum((2 ** (a - largest), 2 ** (b - largest))))


def audit(root):
    outcomes = json.loads((root / "summary.json").read_text())
    assert len(outcomes) == 15
    assert sorted(row["case"] for row in outcomes) == list(range(1, 16))
    checked = []
    for row in outcomes:
        assert row["status"] == "passed", row
        folder = Path(row["result_file"]).parent
        original_meta, original = read(folder / "original.jsonl")
        stages = {}
        for stage in ("replay", "refined", "expanded", "novel"):
            path = folder / f"{stage}.jsonl"
            # Replay compares against six-decimal historical records; later stages
            # reference full-precision native results.
            validation = validate(path, tolerance=6e-7 if stage == "replay" else 1e-9)
            meta, records = read(path)
            for key in ("cipher", "word_bits", "mode", "left", "right"):
                assert meta["config"][key] == original_meta["config"][key]
            assert len(records) == row["rounds"] == len(original)
            stages[stage] = (meta, records)
            checked.append(validation)
        source = stages["refined"][1][-1]
        expanded = stages["expanded"][1][-1]
        meta, candidate = stages["novel"]
        score = meta["optimization"]
        assert expanded["target_output"] == source["output"]
        assert candidate[-1]["target_output"] == source["output"]
        assert [int(x, 16) for x in source["output"]] == row["target"]
        assert abs(expanded["log2_target"] - score["reference_log2"]) < 1e-9
        assert abs(candidate[-1]["log2_target"] - score["after_target"]) < 1e-9
        assert (
            abs(
                log_add(score["log2_intersection"], score["after_novel"])
                - score["after_target"]
            )
            < 1e-9
        )
        union = log_add(score["reference_log2"], score["after_novel"])
        assert abs(union - row["after"]) < 1e-9
        assert union >= expanded["log2_target"] - 1e-9
        assert abs(row["before"] - original[-1]["log2_max"]) < 1e-9
        assert abs(row["diff"] - (union - row["before"])) < 1e-9
        assert row["above_threshold"] == (
            union > -2 * original_meta["config"]["word_bits"]
        )
        replay = stages["replay"][1]
        error = max(
            abs(a["log2_max"] - b["log2_max"]) for a, b in zip(original, replay)
        )
        assert abs(error - row["historical_replay_max_error"]) < 1e-12
    return {
        "cases": len(outcomes),
        "complete_runs_checked": len(checked),
        "checks": checked,
        "scope": "FP64 probability identities, fixed inputs/rounds/targets, historical deltas; not interval arithmetic proof",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        f"Checked {result['cases']} cases and {result['complete_runs_checked']} complete runs"
    )
