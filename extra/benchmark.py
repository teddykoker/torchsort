import sys
from collections import defaultdict
from timeit import timeit

import matplotlib.pyplot as plt
import torch

import torchsort

try:
    import fast_soft_sort.pytorch_ops as fast_soft_sort
except ImportError:
    print("install fast_soft_sort:")
    print("pip install git+https://github.com/google-research/fast-soft-sort")
    sys.exit()


N = list(range(1, 5_000, 100))
SAMPLES = 100


def forward_backward(f, x):
    y = f(x)
    torch.autograd.grad(y.sum(), x)


def main():
    data = defaultdict(list)
    jit = False
    for n in N:
        x = torch.randn(1, n)
        if not jit:
            x = torch.randn(1, n, requires_grad=True)
            forward_backward(torchsort.soft_sort, x)
            forward_backward(fast_soft_sort.soft_sort, x)
            jit = True

        data["torch.sort"].append(
            timeit(lambda: torch.sort(x), number=SAMPLES) / SAMPLES / 1e-6
        )
        data["torchsort (forward)"].append(
            timeit(lambda: torchsort.soft_sort(x), number=SAMPLES) / SAMPLES / 1e-6
        )
        data["fast_soft_sort (forward)"].append(
            timeit(lambda: fast_soft_sort.soft_sort(x), number=SAMPLES) / SAMPLES / 1e-6
        )

        x = torch.randn(1, n, requires_grad=True)
        data["torchsort (forward + backward)"].append(
            timeit(lambda: forward_backward(torchsort.soft_sort, x), number=SAMPLES)
            / SAMPLES
            / 1e-6
        )
        data["fast_soft_sort (forward + backward)"].append(
            timeit(
                lambda: forward_backward(fast_soft_sort.soft_sort, x), number=SAMPLES
            )
            / SAMPLES
            / 1e-6
        )

    plt.figure(figsize=(8, 6))
    for label in data.keys():
        plt.plot(N, data[label], label=label)
    plt.title("Torchsort Benchmark: CPU")
    plt.xlabel("Sequence Length")
    plt.ylim(0, 1000)
    plt.ylabel("Execution Time (μs)")
    plt.legend()
    plt.savefig("extra/benchmark.png")
    plt.show()


if __name__ == "__main__":
    main()
