"""Convert archived info.log to a replayable schedule; no checkpoint data needed."""

import argparse
import json
import re
from pathlib import Path


def convert(path):
    text = path.read_text()
    match = re.fullmatch(
        r"(SIMON|SIMECK)(\d+)_(DIFFERENCE|LINEAR)_FROM_.+_PRECISION_(\d+)",
        path.parent.name,
    )
    if not match:
        raise ValueError("expected original experiment directory name")
    cipher, block, mode, width = match.groups()
    linear = mode == "LINEAR"
    inp = re.search(r"Input: \((0x[0-9a-f]+|0),\s*(0x[0-9a-f]+|0)\)", text)
    if not inp:
        raise ValueError("log has no input")
    left, right = [int(x, 0) for x in inp.groups()]
    if linear:
        left, right = right, left
    config = dict(
        cipher=cipher.lower(),
        word_bits=int(block) // 2,
        mode=mode.lower(),
        width=int(width),
        left=left,
        right=right,
    )
    records = [
        {
            "config": config,
            "source": str(path),
            "note": "Archived log, probabilities rounded to six decimals; linear branches converted to physical order.",
        }
    ]
    for number, section in re.findall(
        r"Round (\d+):\s*(.*?)(?=Round \d+:|\Z)", text, re.DOTALL
    ):
        bits = re.search(r"window is \[(.*?)\]", section)
        peak = re.search(
            r"Max: (-?[\d.]+)\s+\((0x[0-9a-f]+|0),(0x[0-9a-f]+|0)\)", section
        )
        if not bits or not peak:
            raise ValueError(f"incomplete round {number}")
        active = [int(x.strip()) for x in bits[1].split(",") if x.strip()]
        logp, x, y = peak.groups()
        base = int(x, 0) & ~sum(1 << b for b in active)
        records.append(
            dict(
                round=int(number),
                window_bits=active,
                window_base=hex(base),
                log2_max=float(logp),
                output=[y, x] if linear else [x, y],
            )
        )
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    records = convert(args.log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
