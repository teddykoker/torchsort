import sys
from collections import defaultdict
from timeit import timeit

import matplotlib.pyplot as plt
import torch

import torchsort

try:
    import fast_soft_sort.pytorch_ops as fss
except ImportError:
    print("install fast_soft_sort:")
    print("pip install git+https://github.com/google-research/fast-soft-sort")
    sys.exit()


N = list(range(1, 5_000, 100))
B = [2 ** i for i in range(9)]
SAMPLES = 10
CONVERT = 1e-6  # convert seconds to micro-seconds


def time(f):
    return timeit(f, number=SAMPLES) / SAMPLES / CONVERT


def backward(f, x):
    y = f(x)
    torch.autograd.grad(y.sum(), x)


def batch_size(ax):
    data = defaultdict(list)
    jit = False
    for b in B:
        x = torch.randn(b, 100)
        data["torch.sort"].append(time(lambda: torch.sort(x)))
        data["torchsort (forward)"].append(time(lambda: torchsort.soft_sort(x)))
        data["fast_soft_sort (forward)"].append(time(lambda: fss.soft_sort(x)))
        x = torch.randn(b, 100, requires_grad=True)
        data["torchsort (forward + backward)"].append(
            time(lambda: backward(torchsort.soft_sort, x))
        )
        data["fast_soft_sort (forward + backward)"].append(
            time(lambda: backward(fss.soft_sort, x))
        )

    for label in data.keys():
        ax.plot(B, data[label], label=label)
    ax.set_xlabel("Batch Size")
    ax.set_ylim(0, 5000)
    ax.set_ylabel("Execution Time (μs)")
    ax.legend()


def sequence_length(ax):
    data = defaultdict(list)
    jit = False
    for n in N:
        x = torch.randn(1, n)
        data["torch.sort"].append(time(lambda: torch.sort(x)))
        data["torchsort (forward)"].append(time(lambda: torchsort.soft_sort(x)))
        data["fast_soft_sort (forward)"].append(time(lambda: fss.soft_sort(x)))
        x = torch.randn(1, n, requires_grad=True)
        data["torchsort (forward + backward)"].append(
            time(lambda: backward(torchsort.soft_sort, x))
        )
        data["fast_soft_sort (forward + backward)"].append(
            time(lambda: backward(fss.soft_sort, x))
        )

    for label in data.keys():
        ax.plot(N, data[label], label=label)
    ax.set_xlabel("Sequence Length")
    ax.set_ylim(0, 1000)
    ax.set_ylabel("Execution Time (μs)")
    ax.legend()


if __name__ == "__main__":
    # jit
    x = torch.randn(1, 10, requires_grad=True)
    backward(torchsort.soft_sort, x)
    backward(fss.soft_sort, x)
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 4), ncols=2)
    sequence_length(ax1)
    batch_size(ax2)
    fig.suptitle("Torchsort Benchmark: CPU")
    fig.tight_layout()
    plt.savefig("extra/benchmark.png")
