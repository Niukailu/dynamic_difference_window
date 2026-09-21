"""Build an isolated original CPU implementation, without checkpoints or publishing.
Run from the repository root. This measures the original recurrence and scoring.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--width", type=int, default=10)
p.add_argument("--rounds", type=int, default=12)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if not 1 <= a.width <= 14 or not 1 <= a.rounds <= 30:
    p.error("benchmark supports width 1..14, rounds 1..30")
root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="ddw-cpu-") as directory:
    d = Path(directory)
    shutil.copytree(root / "legacy/src", d / "src")
    shutil.copy(root / "legacy/main.cpp", d / "original.cpp")
    header = d / "src/hyperparameter.hpp"
    header.write_text(f"""#pragma once
#include <cstring>
#include <string>
#include <cstdint>
#define SIMON64
#define SIMON
#define DIFFERENCE
#define PRECISION {a.width}
#define LOAD_ROUND 0
const int BITS = 32;
using dtype = uint32_t;
const dtype begin_left=0, begin_right=1;
const std::string name="isolated-benchmark";
inline dtype ROT(dtype x,int r) {{r=(r%32+32)%32; return r ? (x<<r)|(x>>(32-r)) : x;}}
inline dtype f(dtype x) {{return (ROT(x,8)&ROT(x,1))^ROT(x,2);}}
""")
    (d / "benchmark.cpp").write_text(f"""#define main original_main
#include "original.cpp"
#undef main
#include <chrono>
int main() {{
 log_fp=fopen("/dev/null","w");
 window_space left(0,{{}}), right(1,{{}});
 probability_matrix now(left,right); now(0,0)=1;
 for(int round=1;round<={a.rounds};round++) {{
  auto start=std::chrono::steady_clock::now();
  now=round_trans(now);
  probability peak=0;
  for(int i=0;i<now.left_size;i++) for(int j=0;j<now.right_size;j++)
   if(now(i,j)>peak) peak=now(i,j);
  double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  printf("RESULT %d %.12f %.9f\\n",round,peak.value,seconds); fflush(stdout);
 }}
 now.unalloc(); fclose(log_fp);
}}
""")
    subprocess.run(
        [
            "g++",
            "-O3",
            "-march=native",
            "-std=c++17",
            str(d / "benchmark.cpp"),
            "-o",
            str(d / "benchmark"),
        ],
        check=True,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as out:
        out.write(
            json.dumps(
                {
                    "config": dict(
                        width=a.width,
                        rounds=a.rounds,
                        cipher="simon",
                        mode="difference",
                        word_bits=32,
                        left=0,
                        right=1,
                    ),
                    "implementation": "original C++ -O3 -march=native, without save/print",
                    "timing": "wall time including window selection, transfer, max scan",
                }
            )
            + "\n"
        )
        with subprocess.Popen(
            [str(d / "benchmark")], stdout=subprocess.PIPE, text=True
        ) as proc:
            for line in proc.stdout:
                if line.startswith("RESULT"):
                    _, r, peak, seconds = line.split()
                    record = dict(
                        round=int(r), log2_max=float(peak), seconds=float(seconds)
                    )
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    print(record, flush=True)
            if proc.wait():
                raise RuntimeError("CPU benchmark failed")
