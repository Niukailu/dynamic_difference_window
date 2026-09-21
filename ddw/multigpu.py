"""NVLink row-sharded replay. Each round transposes the column shards all-to-all.
No probability contributions are duplicated or summed between devices.
"""

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cupy as cp
import numpy as np

from .core import Window
from .gpu import Engine

PACK = r"""
extern "C" __global__ void unpack(const double* staging, double* next,
    unsigned long long count, unsigned long long rows, unsigned long long cols,
    int devices, double inverse) {
    unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
    for(;i<count;i+=stride) {
        unsigned long long row=i/(cols*devices), col=i%(cols*devices);
        unsigned long long source=col/cols, local=col%cols;
        next[i]=staging[source*rows*cols+row*cols+local]*inverse;
    }
}
"""

PEER = r"""
extern "C" __global__ void peer_unpack(const unsigned long long* pointers, double* next,
    unsigned long long count, unsigned long long rows, unsigned long long cols,
    int devices, int destination, double inverse) {
    unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
    for(;i<count;i+=stride) {
        unsigned long long row=i/(cols*devices), col=i%(cols*devices);
        const double* source=(const double*)pointers[col/cols];
        next[i]=source[((unsigned long long)destination*rows+row)*cols+col%cols]*inverse;
    }
}
"""


def subset(window, rank, devices):
    count = devices.bit_length() - 1
    split = len(window.bits) - count
    if split < 0:
        raise ValueError("window is too small to shard")
    base = window.base
    for j, bit in enumerate(window.bits[split:]):
        base |= ((rank >> j) & 1) << bit
    return Window(base, window.bits[:split])


class ShardedReplay:
    def __init__(self, devices, n, cipher, mode, memory_gib, exchange="peer"):
        if exchange not in ("peer", "staged"):
            raise ValueError("unknown exchange method")
        self.exchange = exchange
        if (
            not devices
            or len(set(devices)) != len(devices)
            or len(devices) & (len(devices) - 1)
        ):
            raise ValueError("use a power-of-two number of distinct devices")
        self.devices = devices
        self.workers = ThreadPoolExecutor(max_workers=len(devices))

        def init(rank):
            cp.cuda.Device(devices[rank]).use()
            cp.get_default_memory_pool().set_limit(size=int(memory_gib * 2**30))
            for peer in devices:
                if peer == devices[rank]:
                    continue
                if not cp.cuda.runtime.deviceCanAccessPeer(devices[rank], peer):
                    raise RuntimeError("all selected GPUs must support peer access")
                try:
                    cp.cuda.runtime.deviceEnablePeerAccess(peer)
                except cp.cuda.runtime.CUDARuntimeError as error:
                    if error.status != 704:  # cudaErrorPeerAccessAlreadyEnabled
                        raise
            return (
                Engine(n, cipher, mode, memory_gib, "coset_lut", "hierarchical"),
                cp.RawKernel(PACK, "unpack"),
                cp.RawKernel(PEER, "peer_unpack"),
            )

        values = list(self.workers.map(init, range(len(devices))))
        self.engines = [v[0] for v in values]
        self.unpack = [v[1] for v in values]
        self.peer_unpack = [v[2] for v in values]
        self.outputs = [None] * len(devices)
        self.staging = [None] * len(devices)

    def map(self, function):
        def call(rank):
            cp.cuda.Device(self.devices[rank]).use()
            return function(rank)

        return list(self.workers.map(call, range(len(self.devices))))

    def split(self, prob):
        rows = prob.shape[0] // len(self.devices)
        source_device = prob.device.id

        def copy(rank):
            out = cp.empty((rows, prob.shape[1]), dtype=cp.float64)
            cp.cuda.runtime.memcpyPeerAsync(
                out.data.ptr,
                self.devices[rank],
                prob.data.ptr + rank * out.nbytes,
                source_device,
                out.nbytes,
                cp.cuda.Stream.null.ptr,
            )
            cp.cuda.Stream.null.synchronize()
            return out

        return self.map(copy)

    def step(self, shards, left, right, target):
        devices = len(self.devices)
        nt = 1 << len(target.bits)
        nl = 1 << len(left.bits)
        if nt < devices or nl < devices:
            raise ValueError("recorded windows shrink below device count")
        cols = nl // devices
        rows = nt // devices
        started = time.perf_counter()

        def compute(rank):
            desired = (nt, cols)
            if self.outputs[rank] is None or self.outputs[rank].shape != desired:
                self.outputs[rank] = cp.empty(desired, dtype=cp.float64)
            output, stats = self.engines[rank].step_and_summary(
                shards[rank],
                subset(left, rank, devices),
                right,
                target,
                out=self.outputs[rank],
            )
            return stats

        stats = self.map(compute)
        peak = max(s[0] for s in stats)
        if peak <= 0 or not math.isfinite(peak):
            raise RuntimeError("empty or invalid retained distribution")
        total = math.fsum(s[1] for s in stats)
        candidates = []
        for rank, (value, _, index) in enumerate(stats):
            if value == peak:
                row, col = divmod(index, cols)
                candidates.append(row * nl + rank * cols + col)
        index = min(candidates)
        compute_seconds = time.perf_counter() - started
        exchange_start = time.perf_counter()

        def exchange(rank):
            if self.exchange == "peer":
                pointers = cp.asarray(
                    [output.data.ptr for output in self.outputs], dtype=cp.uint64
                )
                out = (
                    shards[rank]
                    if shards[rank].shape == (rows, nl)
                    else cp.empty((rows, nl), dtype=cp.float64)
                )
                self.peer_unpack[rank](
                    (min((out.size + 255) // 256, 65535),),
                    (256,),
                    (
                        pointers,
                        out,
                        np.uint64(out.size),
                        np.uint64(rows),
                        np.uint64(cols),
                        np.int32(devices),
                        np.int32(rank),
                        np.float64(1 / peak),
                    ),
                )
                cp.cuda.Stream.null.synchronize()
                return out
            if self.staging[rank] is None or self.staging[rank].shape != (
                devices,
                rows,
                cols,
            ):
                self.staging[rank] = cp.empty((devices, rows, cols), dtype=cp.float64)
            staging = self.staging[rank]
            chunk_bytes = rows * cols * 8
            for source, device in enumerate(self.devices):
                cp.cuda.runtime.memcpyPeerAsync(
                    staging.data.ptr + source * chunk_bytes,
                    self.devices[rank],
                    self.outputs[source].data.ptr + rank * chunk_bytes,
                    device,
                    chunk_bytes,
                    cp.cuda.Stream.null.ptr,
                )
            # Reuse the old input after all devices have finished consuming it.
            out = (
                shards[rank]
                if shards[rank].shape == (rows, nl)
                else cp.empty((rows, nl), dtype=cp.float64)
            )
            self.unpack[rank](
                (min((out.size + 255) // 256, 65535),),
                (256,),
                (
                    staging,
                    out,
                    np.uint64(out.size),
                    np.uint64(rows),
                    np.uint64(cols),
                    np.int32(devices),
                    np.float64(1 / peak),
                ),
            )
            cp.cuda.Stream.null.synchronize()
            return out

        result = self.map(exchange)
        return (
            result,
            (peak, total, index),
            dict(
                compute_seconds=compute_seconds,
                exchange_seconds=time.perf_counter() - exchange_start,
            ),
        )

    def close(self):
        self.workers.shutdown(wait=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--rounds", type=int)
    p.add_argument("--exchange", choices=("peer", "staged"), default="peer")
    p.add_argument(
        "--memory-gib",
        type=float,
        default=64,
        help="per-device pool budget; peer: two matrix shards, staged: three, plus workspace",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    devices = [int(v) for v in args.devices.split(",")]
    records = [json.loads(x) for x in args.reference.read_text().splitlines()]
    config = records[0]["config"].copy()
    schedule = [r for r in records if "round" in r]
    rounds = len(schedule) if args.rounds is None else args.rounds
    if rounds < 1 or rounds > len(schedule) or args.memory_gib <= 0:
        p.error("invalid rounds or memory budget")
    cluster = ShardedReplay(
        devices,
        config["word_bits"],
        config["cipher"],
        config["mode"],
        args.memory_gib,
        args.exchange,
    )
    cp.cuda.Device(devices[0]).use()
    initial = (
        (config["right"], config["left"])
        if config["mode"] == "linear"
        else (config["left"], config["right"])
    )
    left, right = map(Window, initial)
    prob = cp.ones((1, 1), dtype=cp.float64)
    shards = None
    scale = 0.0
    config.update(
        devices=devices,
        rounds=rounds,
        memory_gib=args.memory_gib,
        output=str(args.output),
        reference=str(args.reference),
        exchange=args.exchange,
        kernel="coset_lut",
        statistics="hierarchical",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x") as out:
            out.write(
                json.dumps(
                    dict(
                        config=config,
                        algorithm="row shards, NVLink all-to-all, exact fixed window replay",
                        kernel_sha256=hashlib.sha256(
                            Path(__file__).with_name("kernels.cu").read_bytes()
                        ).hexdigest(),
                        exchange_sha256=hashlib.sha256(
                            (PACK + PEER).encode()
                        ).hexdigest(),
                    )
                )
                + "\n"
            )
            start_all = time.perf_counter()
            for number, reference in enumerate(schedule[:rounds], 1):
                cp.cuda.Device(devices[0]).use()
                start = time.perf_counter()
                target = Window(
                    int(reference["window_base"], 0), tuple(reference["window_bits"])
                )
                if shards is None and min(
                    1 << len(left.bits), 1 << len(target.bits)
                ) >= len(devices):
                    shards = cluster.split(prob)
                    prob = None
                if shards is None:
                    prob, (peak, total, index) = cluster.engines[0].step_and_summary(
                        prob, left, right, target
                    )
                    if peak <= 0 or not math.isfinite(peak):
                        raise RuntimeError("empty or invalid retained distribution")
                    prob *= 1 / peak
                    cp.cuda.Stream.null.synchronize()
                    phases = dict(
                        compute_seconds=time.perf_counter() - start,
                        exchange_seconds=0.0,
                    )
                else:
                    shards, (peak, total, index), phases = cluster.step(
                        shards, left, right, target
                    )
                scale += math.log2(peak)
                row, col = divmod(index, 1 << len(left.bits))
                endpoint = [int(target.values()[row]), int(left.values()[col])]
                if config["mode"] == "linear":
                    endpoint.reverse()
                physical = [int(v, 0) for v in reference["output"]]
                ref_left, ref_right = (
                    physical[::-1] if config["mode"] == "linear" else physical
                )
                ri, ci = target.index(ref_left), left.index(ref_right)
                if ri is None or ci is None:
                    value = 0
                elif shards is None:
                    value = float(prob[ri, ci])
                else:
                    rows = (1 << len(target.bits)) // len(devices)
                    owner = ri // rows
                    with cp.cuda.Device(devices[owner]):
                        value = float(shards[owner][ri % rows, ci])
                record = dict(
                    round=number,
                    log2_max=scale,
                    log2_mass=scale + math.log2(total / peak),
                    output=[hex(v) for v in endpoint],
                    window_base=hex(target.base),
                    window_bits=list(target.bits),
                    reference_output=reference["output"],
                    reference_log2_max=reference["log2_max"],
                    log2_at_reference_output=scale + math.log2(value)
                    if value
                    else None,
                    active_devices=len(devices) if shards else 1,
                    seconds=time.perf_counter() - start,
                    **phases,
                )
                out.write(json.dumps(record, allow_nan=False) + "\n")
                out.flush()
                print(
                    f"round={number} log2_max={scale:.9f} seconds={record['seconds']:.3f} exchange={phases['exchange_seconds']:.3f}",
                    flush=True,
                )
                left, right = target, left
            out.write(
                json.dumps(
                    dict(
                        completed_rounds=rounds,
                        search_seconds=time.perf_counter() - start_all,
                    )
                )
                + "\n"
            )
    finally:
        cluster.close()


if __name__ == "__main__":
    main()
