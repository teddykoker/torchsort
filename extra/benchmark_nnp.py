import sys
import argparse
import csv
from collections import defaultdict
from timeit import timeit
import time

import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import os
import gc

import torchsort

from functools import partial; print = partial(print, flush=True)

def fix_seed(seed=42):
    # 1. Python's built-in random module
    random.seed(seed)
    
    # 2. Numpy
    np.random.seed(seed)
    
    # 3. Torch (CPU)
    torch.manual_seed(seed)
    
    # 4. Torch (GPU/CUDA)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multi-GPU
    
    # 5. CUDNN (The critical part for determinism)
    # This ensures that the same algorithm is chosen for convolutions/ops every time
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 6. Python Hash Seed (Good for dictionary/set ordering)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 7. (Optional) Force PyTorch to use deterministic algorithms
    # Note: This might throw an error if an op doesn't have a deterministic version
    # torch.use_deterministic_algorithms(True)

    print(f"✅ Seeds fixed to: {seed}")

# try:
#     import fast_soft_sort.pytorch_ops as fss
# except ImportError:
#     print("install fast_soft_sort:")
#     print("pip install git+https://github.com/google-research/fast-soft-sort")
#     sys.exit()

init_len, final_len, step_len = 2500, 100000, 2500
N = list(range(init_len, final_len + step_len, step_len))
B = [2 ** i for i in range(9)]
B_CUDA = [2 ** i for i in range(13)]
SAMPLES = 100
WARMUP  = 10     # per-measurement warm-up calls
CONVERT = 1e-6   # seconds → microseconds

def time_fn(f):
    """CPU timing: timeit is accurate (CPU ops are synchronous)."""
    for _ in range(WARMUP):
        f()
    return timeit(f, number=SAMPLES) / SAMPLES / CONVERT


def time_fn_cuda(f):
    """GPU timing via CUDA events.

    CUDA kernels are asynchronous: without synchronization, timeit only
    measures kernel-launch overhead (a few μs), not actual execution time.
    CUDA events are recorded on the GPU stream and give the true elapsed
    time between them, independent of CPU-side scheduling jitter.

    A per-call warm-up ensures the first few runs (which may include
    CUDA-graph warm-up, cache fills, or JIT recompilation) are excluded
    from the timing window.
    """
    # warm up: let CUDA JIT, allocate caches, etc.
    for _ in range(WARMUP):
        f()
    torch.cuda.synchronize()   # flush warm-up work before starting the clock

    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(SAMPLES):
        f()
    end_event.record()

    # synchronize: block until the GPU has finished all recorded work
    torch.cuda.synchronize()

    # elapsed_time returns milliseconds; convert to microseconds
    return start_event.elapsed_time(end_event) / SAMPLES * 1e3


def backward(f, x):
    y = f(x)
    torch.autograd.grad(y.sum(), x)

plt.style.use("bmh")
palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
def style(name):
    if name == "torch.sort":
        return {"color": palette[0]}
    linestyle = "--" if "backward" in name else "-"
    if "fast_soft_sort" in name:
        return {"color": None, "linestyle": linestyle}
    elif "_parallel" in name:
        return {"color": palette[1], "linestyle": linestyle}
    else:
        return {"color": palette[2], "linestyle": linestyle}


def batch_size(ax):
    data = defaultdict(list)
    for b in B:
        x = torch.randn(b, 100)
        data["torch.sort"].append(time_fn(lambda: torch.sort(x)))
        data["torchsort"].append(time_fn(lambda: torchsort.soft_sort(x)))
        # data["fast_soft_sort"].append(time_fn(lambda: fss.soft_sort(x)))
        x = torch.randn(b, 100, requires_grad=True)
        data["torchsort (with backward)"].append(
            time_fn(lambda: backward(torchsort.soft_sort, x))
        )
        # data["fast_soft_sort (with backward)"].append(
        #     time_fn(lambda: backward(fss.soft_sort, x))
        # )

    for label in data.keys():
        ax.plot(B, data[label], label=label, **style(label))
    ax.set_xlabel("Batch Size")
    ax.set_ylim(0, 5000)
    ax.set_ylabel("Execution Time (μs)")
    ax.legend()


def sequence_length(ax):
    data = defaultdict(list)
    for n in N:
        x = torch.randn(1, n)
        data["torch.sort"].append(time_fn(lambda: torch.sort(x)))
        data["torchsort"].append(time_fn(lambda: torchsort.soft_sort(x)))
        # data["fast_soft_sort"].append(time_fn(lambda: fss.soft_sort(x)))
        x = torch.randn(1, n, requires_grad=True)
        data["torchsort (with backward)"].append(
            time_fn(lambda: backward(torchsort.soft_sort, x))
        )
        # data["fast_soft_sort (with backward)"].append(
        #     time_fn(lambda: backward(fss.soft_sort, x))
        # )

    for label in data.keys():
        ax.plot(N, data[label], label=label, **style(label))
    ax.set_xlabel("Sequence Length")
    ax.set_ylim(0, 1000)
    ax.set_ylabel("Execution Time (μs)")
    ax.legend()


def batch_size_cuda(ax, reg):
    data = defaultdict(list)
    print("Benchmarking batch sizes")
    L = 512
    sort_seq   = lambda x: torchsort.soft_sort(x, regularization=reg)
    sort_par   = lambda x: torchsort.soft_sort_parallel(x, regularization=reg)
    for b in B_CUDA:
        print(f"Batch size {b} running...")
        x = torch.randn(b, L).cuda()
        x_seq = x.clone()
        x_par = x.clone()

        data["torch.sort"].append(time_fn_cuda(lambda: torch.sort(x_seq)))
        data["torchsort_parallel"].append(time_fn_cuda(lambda: sort_par(x_par)))
        data["torchsort"].append(time_fn_cuda(lambda: sort_seq(x_seq)))

        x = torch.randn(b, L, requires_grad=True).cuda()
        x_seq = x.clone().detach().requires_grad_(True)
        x_par = x.clone().detach().requires_grad_(True)

        data["torchsort_parallel (with backward)"].append(
            time_fn_cuda(lambda: backward(sort_par, x_par))
        )
        data["torchsort (with backward)"].append(
            time_fn_cuda(lambda: backward(sort_seq, x_seq))
        )
        print(f"  Batch size {b} done")
    for label in data.keys():
        ax.plot(B_CUDA, data[label], label=label, **style(label))
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Execution Time (μs)")
    ax.legend()
    return data


def sequence_length_cuda(ax, reg):
    data = defaultdict(list)
    print("Benchmarking sequence lengths")
    B = 4
    sort_seq   = lambda x: torchsort.soft_sort(x, regularization=reg)
    sort_par   = lambda x: torchsort.soft_sort_parallel(x, regularization=reg)
    for n in N:
        print(f"Sequence length {n} running...")
        x_init = torch.randn(B, n).cuda()
        x_seq = x_init.clone()
        x_par = x_init.clone()

        data["torch.sort"].append(time_fn_cuda(lambda: torch.sort(x_seq)))
        data["torchsort_parallel"].append(time_fn_cuda(lambda: sort_par(x_par)))
        data["torchsort"].append(time_fn_cuda(lambda: sort_seq(x_seq)))

        x_init = torch.randn(B, n, requires_grad=True).cuda()
        x_seq = x_init.clone().detach().requires_grad_(True)
        x_par = x_init.clone().detach().requires_grad_(True)

        data["torchsort_parallel (with backward)"].append(
            time_fn_cuda(lambda: backward(sort_par, x_par))
        )
        data["torchsort (with backward)"].append(
            time_fn_cuda(lambda: backward(sort_seq, x_seq))
        )
        print(f"  Sequence length {n} done")
    for label in data.keys():
        ax.plot(N, data[label], label=label, **style(label))
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Execution Time (μs)")
    ax.legend()
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regularization", choices=["l2", "kl"], default="l2",
                        help="Regularization type: l2 or kl (default: l2)")
    args = parser.parse_args()
    reg = args.regularization

    fix_seed(42)

    if torch.cuda.is_available():
        # warmup
        x = torch.randn(4, 256, requires_grad=True).cuda()
        x_seq = x.clone().detach().requires_grad_(True)
        x_par = x.clone().detach().requires_grad_(True)
        backward(lambda v: torchsort.soft_sort_parallel(v, regularization=reg), x_par)
        backward(lambda v: torchsort.soft_sort(v, regularization=reg), x_seq)
        print("--- Warm up done ---")

        fig, (ax1, ax2) = plt.subplots(figsize=(10, 4), ncols=2)
        seq_data   = sequence_length_cuda(ax1, reg)   # fixed batch=4, varies N
        batch_data = batch_size_cuda(ax2, reg)         # fixed seq_len=512, varies B_CUDA
        fig.suptitle(f"Torchsort Benchmark: CUDA ({reg})")
        fig.tight_layout()
        png_path = f"extra/benchmark_cuda_nnp_{reg}.png"
        plt.savefig(png_path)
        print(f"PNG saved to {png_path}")

        # --- Save results to CSV ------------------------------------------
        csv_path = f"extra/benchmark_nnp_{reg}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sequence_length", "batch_size",
                "torch.sort", "torchsort", "torchsort_parallel",
                "torchsort (with backward)", "torchsort_parallel (with backward)",
            ])
            # Sequence-length sweep (batch_size fixed at 4)
            for i, n in enumerate(N):
                writer.writerow([
                    n, 4,
                    seq_data["torch.sort"][i],
                    seq_data["torchsort"][i],
                    seq_data["torchsort_parallel"][i],
                    seq_data["torchsort (with backward)"][i],
                    seq_data["torchsort_parallel (with backward)"][i],
                ])
            # Batch-size sweep (sequence_length fixed at 512)
            for i, b in enumerate(B_CUDA):
                writer.writerow([
                    512, b,
                    batch_data["torch.sort"][i],
                    batch_data["torchsort"][i],
                    batch_data["torchsort_parallel"][i],
                    batch_data["torchsort (with backward)"][i],
                    batch_data["torchsort_parallel (with backward)"][i],
                ])
        print(f"CSV saved to {csv_path}")
