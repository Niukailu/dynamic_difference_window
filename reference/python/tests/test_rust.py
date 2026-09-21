"""Independent CPU replay of Rust-generated schedules, including continuation."""

import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ddw.core import Window, cpu_step


@unittest.skipUnless(os.environ.get("DDW_RUST"), "set DDW_RUST to the Rust executable")
class Rust(unittest.TestCase):
    def check_cpu(self, path):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        config = records[0]["config"]
        left, right = config["left"], config["right"]
        if config["mode"] == "linear":
            left, right = right, left
        left, right = Window(left), Window(right)
        prob = np.ones((1, 1))
        rows = [row for row in records if "round" in row]
        self.assertEqual(len(rows), config["rounds"])
        for row in rows:
            target = Window(int(row["window_base"], 0), tuple(row["window_bits"]))
            prob = cpu_step(
                prob,
                left,
                right,
                target,
                config["word_bits"],
                config["cipher"],
                config["mode"],
            )
            self.assertAlmostEqual(row["log2_max"], math.log2(prob.max()), places=10)
            self.assertAlmostEqual(row["log2_mass"], math.log2(prob.sum()), places=10)
            left, right = target, left
        return rows

    def test_variants_and_continuation(self):
        binary = os.environ["DDW_RUST"]
        device = os.environ.get("DDW_TEST_DEVICE", "0")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for n in (16, 24, 32, 48, 64):
                for cipher in ("simon", "simeck"):
                    for mode in ("difference", "linear"):
                        with self.subTest(n=n, cipher=cipher, mode=mode):
                            path = root / f"{n}-{cipher}-{mode}.jsonl"
                            subprocess.run(
                                [
                                    binary,
                                    "search",
                                    "--word-bits",
                                    str(n),
                                    "--cipher",
                                    cipher,
                                    "--mode",
                                    mode,
                                    "--width",
                                    "6",
                                    "--rounds",
                                    "5",
                                    "--devices",
                                    device,
                                    "--candidates",
                                    "3",
                                    "--output",
                                    str(path),
                                ],
                                check=True,
                                stdout=subprocess.DEVNULL,
                            )
                            original = self.check_cpu(path)
                            if n == 16:
                                tail = root / (path.stem + "-tail.jsonl")
                                subprocess.run(
                                    [
                                        binary,
                                        "replay",
                                        str(path),
                                        "--rounds",
                                        "7",
                                        "--extend",
                                        "--devices",
                                        device,
                                        "--output",
                                        str(tail),
                                    ],
                                    check=True,
                                    stdout=subprocess.DEVNULL,
                                )
                                rows = self.check_cpu(tail)
                                self.assertEqual(
                                    [r["window_bits"] for r in original],
                                    [r["window_bits"] for r in rows[:5]],
                                )

    @unittest.skipUnless(os.environ.get("DDW_TEST_DEVICES"), "set DDW_TEST_DEVICES")
    def test_distributed_expansion_and_continuation(self):
        binary = os.environ["DDW_RUST"]
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("difference", "linear"):
                source = Path(directory) / (mode + "-source.jsonl")
                output = Path(directory) / (mode + "-expanded.jsonl")
                subprocess.run(
                    [
                        binary,
                        "search",
                        "--word-bits",
                        "16",
                        "--mode",
                        mode,
                        "--width",
                        "4",
                        "--rounds",
                        "5",
                        "--output",
                        str(source),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                subprocess.run(
                    [
                        binary,
                        "replay",
                        str(source),
                        "--width",
                        "6",
                        "--rounds",
                        "7",
                        "--extend",
                        "--candidates",
                        "3",
                        "--devices",
                        os.environ["DDW_TEST_DEVICES"],
                        "--output",
                        str(output),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                rows = self.check_cpu(output)
                self.assertTrue(any(r["active_devices"] > 1 for r in rows))
                subprocess.run(
                    [binary, "validate", str(output)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )

    def test_mirror_posterior_and_implicit_union(self):
        binary = os.environ["DDW_RUST"]
        device = os.environ.get("DDW_TEST_DEVICE", "0")
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("difference", "linear"):
                source = Path(directory) / (mode + "-source.jsonl")
                mirror = Path(directory) / (mode + "-mirror.jsonl")
                refined = Path(directory) / (mode + "-refined.jsonl")
                virtual = Path(directory) / (mode + "-virtual.jsonl")
                union = Path(directory) / (mode + "-union.json")
                union4 = Path(directory) / (mode + "-union4.json")
                # A nonzero self-loop with a known swapped endpoint. Short mirrors
                # of (0,1) can legitimately have no paths, so they are not fixtures.
                config = {
                    "cipher": "simon",
                    "mode": mode,
                    "word_bits": 16,
                    "left": 65535,
                    "right": 65535,
                    "width": 4,
                    "rounds": 5,
                }
                point = Window(65535)
                probability = np.ones((1, 1))
                records = [{"config": config}]
                for round_number in range(1, 6):
                    probability = cpu_step(
                        probability, point, point, point, 16, "simon", mode
                    )
                    records.append(
                        {
                            "round": round_number,
                            "window_base": "0xffff",
                            "window_bits": [],
                            "output": ["0xffff", "0xffff"],
                            "log2_max": math.log2(probability[0, 0]),
                            "log2_mass": math.log2(probability.sum()),
                        }
                    )
                source.write_text("\n".join(map(json.dumps, records)) + "\n")
                commands = [
                    [
                        "mirror",
                        str(source),
                        "--half-rounds",
                        "2",
                        "--devices",
                        device,
                        "--output",
                        str(mirror),
                    ],
                    [
                        "refine",
                        str(mirror),
                        "--strategy",
                        "posterior",
                        "--passes",
                        "1",
                        "--device",
                        device,
                        "--output",
                        str(refined),
                    ],
                    [
                        "compressed",
                        str(refined),
                        "--width",
                        "2",
                        "--device",
                        device,
                        "--output",
                        str(virtual),
                    ],
                    [
                        "symmetric-union",
                        str(mirror),
                        "--device",
                        device,
                        "--output",
                        str(union),
                    ],
                ]
                commands.append(
                    [
                        "path-union",
                        str(mirror),
                        str(mirror),
                        "--reflect",
                        "--device",
                        device,
                        "--output",
                        str(union4),
                    ]
                )
                for command in commands:
                    subprocess.run(
                        [binary, *command], check=True, stdout=subprocess.DEVNULL
                    )
                for path in (source, mirror, refined, virtual):
                    self.check_cpu(path)
                rows = self.check_cpu(mirror)
                self.assertEqual(rows[-1]["output"], ["0xffff", "0xffff"])
                summary4 = json.loads(union4.read_text())
                self.assertAlmostEqual(
                    summary4["log2_union"], rows[-1]["log2_max"], places=10
                )
                summary = json.loads(union.read_text())
                self.assertAlmostEqual(
                    summary["log2_union"], rows[-1]["log2_max"], places=10
                )
