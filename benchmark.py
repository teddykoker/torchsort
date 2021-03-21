from timeit import timeit

import fast_soft_sort.pytorch_ops as fast_soft_sort
import torch
import torch.autograd.profiler as profiler

import torchsort

if __name__ == "__main__":
    x = torch.randn(10, 1000)
    print("mine", timeit(lambda: torchsort.soft_rank(x), number=1000))
    print("theirs", timeit(lambda: fast_soft_sort.soft_rank(x), number=1000))

    x = torch.randn(1, 10)
    print(torchsort.soft_rank(x, regularization_strength=1.0))
    print(fast_soft_sort.soft_rank(x, regularization_strength=1.0))
    x = torch.randn(1, 10).double()
    print(torchsort.soft_rank(x, regularization_strength=1.0))
    print(fast_soft_sort.soft_rank(x, regularization_strength=1.0))

    # with profiler.profile() as prof:
    #     with profiler.record_function("soft_rank"):
    #         torchsort.soft_rank(x)

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    # with profiler.profile() as prof:
    #     with profiler.record_function("soft_rank"):
    #         fast_soft_sort.soft_rank(x)

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
