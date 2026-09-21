import json
import tempfile
import unittest
from pathlib import Path

from tools.check_results import validate
from tools.import_legacy_log import convert

ROOT = Path(__file__).resolve().parents[3]


class HistoricalLogs(unittest.TestCase):
    def test_import_40_complete_logs_reject_truncated_log(self):
        paths = sorted((ROOT / "experiments").glob("*/info.log"))
        self.assertEqual(len(paths), 41)
        for path in paths:
            with self.subTest(path=path.parent.name):
                if path.parent.name == "SIMON48_LINEAR_FROM_0x400004_0x1_PRECISION_17":
                    with self.assertRaisesRegex(ValueError, "incomplete round 5"):
                        convert(path)
                    continue
                records = convert(path)
                self.assertGreater(len(records), 2)
                self.assertEqual(
                    [r["round"] for r in records[1:]], list(range(1, len(records)))
                )

    def test_swapped_linear_input(self):
        records = convert(
            ROOT
            / "experiments/SIMON64_LINEAR_FROM_0x44400_0x1000_PRECISION_14/info.log"
        )
        self.assertEqual(records[0]["config"]["left"], 0x44400)
        self.assertEqual(records[0]["config"]["right"], 0x1000)

    def test_validator_rejects_incomplete_run(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "partial.jsonl"
            records = [
                {"config": {"rounds": 2}},
                {"round": 1, "log2_max": -2, "log2_mass": 0},
            ]
            p.write_text("\n".join(json.dumps(r) for r in records))
            with self.assertRaisesRegex(ValueError, "incomplete run"):
                validate(p)

    def test_validator_detects_lost_mass(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.jsonl"
            records = [{"config": {}}, {"round": 1, "log2_max": -2, "log2_mass": 0.1}]
            p.write_text("\n".join(json.dumps(r) for r in records))
            with self.assertRaises(AssertionError):
                validate(p)
