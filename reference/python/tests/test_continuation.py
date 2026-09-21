"""Continuing a saved schedule must preserve its prefix and evaluate new rounds."""

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ddw.core import Window, cpu_step


class Continuation(unittest.TestCase):
    def test_saved_prefix_and_new_tail(self):
        try:
            import cupy as cp

            device = int(os.environ.get("DDW_TEST_DEVICE", "0"))
            cp.cuda.Device(device).use()
            cp.zeros(1)
        except Exception as exc:
            raise unittest.SkipTest(str(exc))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prefix = root / "prefix.jsonl"
            base = [
                sys.executable,
                "-m",
                "ddw",
                "--device",
                str(device),
                "--word-bits",
                "16",
                "--width",
                "6",
            ]
            subprocess.run(
                [*base, "--rounds", "3", "--output", str(prefix)],
                stdout=subprocess.DEVNULL,
                check=True,
            )
            original = [
                json.loads(x)
                for x in prefix.read_text().splitlines()
                if '"round":' in x
            ]
            bad = subprocess.run(
                [
                    *base,
                    "--rounds",
                    "6",
                    "--replay-windows",
                    str(prefix),
                    "--output",
                    str(root / "bad.jsonl"),
                ],
                capture_output=True,
            )
            self.assertNotEqual(bad.returncode, 0)
            commands = [
                [
                    *base,
                    "--rounds",
                    "6",
                    "--replay-windows",
                    str(prefix),
                    "--continue-search",
                ]
            ]
            if os.environ.get("DDW_TEST_DEVICES"):
                commands.append(
                    [
                        sys.executable,
                        "-m",
                        "ddw.multigpu",
                        str(prefix),
                        "--rounds",
                        "6",
                        "--continue-search",
                        "--devices",
                        os.environ["DDW_TEST_DEVICES"],
                        "--memory-gib",
                        "2",
                    ]
                )
            for i, command in enumerate(commands):
                out = root / f"continued-{i}.jsonl"
                subprocess.run(
                    [*command, "--output", str(out)],
                    stdout=subprocess.DEVNULL,
                    check=True,
                )
                rows = [
                    json.loads(x)
                    for x in out.read_text().splitlines()
                    if '"round":' in x
                ]
                self.assertEqual(len(rows), 6)
                probability = np.ones((1, 1))
                left, right = Window(0), Window(1)
                for index, row in enumerate(rows):
                    target = Window(
                        int(row["window_base"], 0), tuple(row["window_bits"])
                    )
                    probability = cpu_step(probability, left, right, target, 16)
                    self.assertAlmostEqual(
                        row["log2_max"], math.log2(probability.max()), places=11
                    )
                    if index < 3:
                        self.assertEqual(
                            row["window_bits"], original[index]["window_bits"]
                        )
                    else:
                        self.assertNotIn("reference_output", row)
                    left, right = target, left
