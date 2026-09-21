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

    def test_joint_and_affine_windows(self):
        binary = os.environ["DDW_RUST"]
        device = os.environ.get("DDW_TEST_DEVICE", "0")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            subprocess.run(
                [
                    binary,
                    "search",
                    "--word-bits",
                    "16",
                    "--mode",
                    "linear",
                    "--width",
                    "3",
                    "--rounds",
                    "5",
                    "--devices",
                    device,
                    "--output",
                    str(source),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            source_rows = self.check_cpu(source)
            endpoint = tuple(int(x, 0) for x in source_rows[-1]["output"])

            def endpoint_weight(path, replacement=None):
                records = [json.loads(line) for line in path.read_text().splitlines()]
                config = records[0]["config"]
                left, right = Window(config["right"]), Window(config["left"])
                prob = np.ones((1, 1))
                for row in records[1:]:
                    if "round" not in row:
                        continue
                    target = Window(
                        int(row["window_base"], 0), tuple(row["window_bits"])
                    )
                    if replacement is not None and row["round"] == 3:
                        target = Window(replacement["base"], tuple(replacement["bits"]))
                    prob = cpu_step(prob, left, right, target, 16, "simon", "linear")
                    left, right = target, left

                # Physical linear output is swapped relative to the internal state.
                def packed(window, value):
                    if value & ~window.mask != window.base:
                        return None
                    return sum(
                        ((value >> bit) & 1) << i for i, bit in enumerate(window.bits)
                    )

                i, j = packed(left, endpoint[1]), packed(right, endpoint[0])
                return 0.0 if i is None or j is None else float(prob[i, j])

            for shard in range(2):
                path = root / f"joint-{shard}.jsonl"
                subprocess.run(
                    [
                        binary,
                        "joint-refine",
                        str(source),
                        "--round",
                        "3",
                        "--shard",
                        str(shard),
                        "--shards",
                        "2",
                        "--device",
                        device,
                        "--memory-gib",
                        "2",
                        "--output",
                        str(path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                self.check_cpu(path)
                metadata = json.loads(path.read_text().splitlines()[0])["optimization"]
                self.assertAlmostEqual(
                    metadata["after"], math.log2(endpoint_weight(path)), places=10
                )
                self.assertGreaterEqual(metadata["after"] + 1e-10, metadata["before"])
            path = root / "affine.json"
            subprocess.run(
                [
                    binary,
                    "affine-refine",
                    str(source),
                    "--round",
                    "3",
                    "--device",
                    device,
                    "--memory-gib",
                    "2",
                    "--output",
                    str(path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            summary = json.loads(path.read_text())
            for expanded in summary["expansion_profile"]:
                self.assertAlmostEqual(
                    expanded["log2_target"],
                    math.log2(endpoint_weight(source, expanded["window"])),
                    places=10,
                )
                self.assertGreaterEqual(expanded["gain_bits"] + 1e-10, 0.0)
            if summary["best_expansion"] is not None:
                expanded = summary["best_expansion"]
                expanded_path = Path(expanded["path"])
                self.check_cpu(expanded_path)
                self.assertAlmostEqual(
                    expanded["log2_target"],
                    math.log2(endpoint_weight(expanded_path)),
                    places=10,
                )
            if summary["improved"]:
                total = 0.0
                state_count = 0
                for branch in summary["branches"]:
                    state_count += 1 << len(branch["window"]["bits"])
                    if branch["path"] is not None:
                        branch_path = Path(branch["path"])
                        self.check_cpu(branch_path)
                        value = endpoint_weight(branch_path)
                        self.assertAlmostEqual(
                            math.log2(value), branch["log2_target"], places=10
                        )
                        total += value
                self.assertEqual(state_count, summary["state_count"])
                self.assertAlmostEqual(math.log2(total), summary["after"], places=10)
            else:
                self.assertAlmostEqual(
                    summary["after"], source_rows[-1]["log2_max"], places=10
                )

    def test_cross_round_and_novelty(self):
        binary = os.environ["DDW_RUST"]
        device = os.environ.get("DDW_TEST_DEVICE", "0")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("difference", "linear"):
                source = root / f"source-{mode}.jsonl"
                subprocess.run(
                    [
                        binary,
                        "search",
                        "--word-bits",
                        "16",
                        "--mode",
                        mode,
                        "--left",
                        "1",
                        "--right",
                        "1",
                        "--width",
                        "3",
                        "--rounds",
                        "5",
                        "--devices",
                        device,
                        "--output",
                        str(source),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                original = self.check_cpu(source)
                schedule = [
                    Window(int(x["window_base"], 0), tuple(x["window_bits"]))
                    for x in original
                ]
                physical = tuple(int(x, 0) for x in original[-1]["output"])
                point = physical[::-1] if mode == "linear" else physical

                def score(windows, novel):
                    left, right = Window(1), Window(1)
                    total = np.ones((1, 1))
                    common = np.ones((1, 1))
                    for index, target in enumerate(windows):
                        total = cpu_step(total, left, right, target, 16, "simon", mode)
                        common = cpu_step(
                            common, left, right, target, 16, "simon", mode
                        )
                        for row, value in enumerate(target.values()):
                            if schedule[index].index(int(value)) is None:
                                common[row, :] = 0.0
                        left, right = target, left
                    i, j = left.index(point[0]), right.index(point[1])
                    if i is None or j is None:
                        return 0.0, 0.0
                    weight = float(total[i, j])
                    return weight, weight - float(common[i, j]) if novel else weight

                for novel in (False, True):
                    for round_ in (1, 2, 4):
                        path = root / f"block-{mode}-{novel}-{round_}.jsonl"
                        command = [
                            binary,
                            "block-refine",
                            str(source),
                            "--round",
                            str(round_),
                            "--pool",
                            "4",
                            "--device",
                            device,
                            "--memory-gib",
                            "2",
                            "--output",
                            str(path),
                        ]
                        if novel:
                            command += ["--reference", str(source)]
                        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
                        self.check_cpu(path)
                        records = [json.loads(x) for x in path.read_text().splitlines()]
                        meta = records[0]["optimization"]
                        pools = [
                            [Window(x[0]["base"], tuple(x[0]["bits"])) for x in pool]
                            for pool in meta["candidate_pools"]
                        ]
                        best = 0.0
                        for a in pools[0]:
                            for b in pools[1]:
                                candidate = schedule.copy()
                                candidate[round_ - 1 : round_ + 1] = [a, b]
                                _, value = score(candidate, novel)
                                best = max(best, value)
                        actual = (
                            0.0
                            if meta["after_novel"] is None
                            else 2.0 ** meta["after_novel"]
                        )
                        self.assertTrue(
                            math.isclose(actual, best, rel_tol=1e-9, abs_tol=1e-30)
                        )
                        selected = schedule.copy()
                        selected[round_ - 1 : round_ + 1] = [
                            pools[k][i] for k, i in enumerate(meta["selected_indices"])
                        ]
                        total, value = score(selected, novel)
                        self.assertAlmostEqual(
                            math.log2(total), meta["after_target"], places=10
                        )
                        self.assertTrue(
                            math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-30)
                        )
                        final = [x for x in records if "round" in x][-1]
                        self.assertEqual(
                            [int(x, 0) for x in final["target_output"]], list(physical)
                        )
                        self.assertAlmostEqual(
                            final["log2_target"], math.log2(total), places=10
                        )
                        if novel:
                            ref_weight = 2.0 ** original[-1]["log2_max"]
                            self.assertAlmostEqual(
                                meta["log2_union"],
                                math.log2(ref_weight + value),
                                places=10,
                            )

    def test_union_uses_declared_target(self):
        binary = os.environ["DDW_RUST"]
        device = os.environ.get("DDW_TEST_DEVICE", "0")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "target.jsonl"
            config = {
                "cipher": "simon",
                "mode": "linear",
                "word_bits": 16,
                "left": 65535,
                "right": 65535,
                "width": 2,
                "rounds": 5,
            }
            left = right = Window(65535)
            prob = np.ones((1, 1))
            records = [{"config": config}]
            for round_ in range(1, 6):
                target = Window(65535, (0, 2)) if round_ == 5 else Window(65535)
                prob = cpu_step(prob, left, right, target, 16, "simon", "linear")
                row, col = np.unravel_index(prob.argmax(), prob.shape)
                record = {
                    "round": round_,
                    "window_base": hex(target.base),
                    "window_bits": target.bits,
                    "output": [
                        hex(int(left.values()[col])),
                        hex(int(target.values()[row])),
                    ],
                    "log2_max": math.log2(prob.max()),
                    "log2_mass": math.log2(prob.sum()),
                }
                records.append(record)
                right, left = left, target
            records[-1]["target_output"] = ["0xffff", "0xffff"]
            records[-1]["log2_target"] = math.log2(
                prob[left.index(65535), right.index(65535)]
            )
            self.assertNotEqual(records[-1]["output"], records[-1]["target_output"])
            source.write_text("\n".join(map(json.dumps, records)) + "\n")
            self.check_cpu(source)
            summary = root / "union.json"
            subprocess.run(
                [
                    binary,
                    "symmetric-union",
                    str(source),
                    "--device",
                    device,
                    "--memory-gib",
                    "2",
                    "--output",
                    str(summary),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            result = json.loads(summary.read_text())
            self.assertEqual(result["endpoint"], [65535, 65535])
            self.assertAlmostEqual(
                result["log2_a"], records[-1]["log2_target"], places=10
            )
            self.check_cpu(Path(result["intersection_file"]))
