# Torchsort

![Tests](https://github.com/teddykoker/torchsort/workflows/Tests/badge.svg)

Fast, differentiable sorting and ranking in PyTorch.

Pure PyTorch implementation of [Fast Differentiable Sorting and
Ranking](https://arxiv.org/abs/2002.08871) (Blondel et al.). Much of the code is
copied from the original Numpy implementation at
[google-research/fast-soft-sort](https://github.com/google-research/fast-soft-sort),
with the isotonic regression solver rewritten as a PyTorch C++ Extension.

**NOTE**: I am actively working on this. The API should remain about the same;
but expect more optimizations and benchmarks soon. The C++ isotonic regression
solver is currently only implemented on CPU, so CUDA tensors will be copied over
to CPU to perform the operations. I am currently working on the CUDA kernel
implementation, which should be done soon.

## Install

```bash
pip install torchsort
```

## Usage

`torchsort` exposes two functions: `soft_rank` and `soft_sort`, each with
parameters `regularization` (`"l2"` or `"kl"`) and `regularization_strength` (a
scalar value). Each will rank/sort the last dimension of a 2-d tensor, with an
accuracy dependant upon the regularization strength:

```python
import torch
import torchsort

x = torch.tensor([[8, 0, 5, 3, 2, 1, 6, 7, 9]])

torchsort.soft_sort(x, regularization_strength=1.0)
# tensor([[0.5556, 1.5556, 2.5556, 3.5556, 4.5556, 5.5556, 6.5556, 7.5556, 8.5556]])
torchsort.soft_sort(x, regularization_strength=0.1)
# tensor([[-0., 1., 2., 3., 5., 6., 7., 8., 9.]])

torchsort.soft_rank(x)
# tensor([[8., 1., 5., 4., 3., 2., 6., 7., 9.]])
```

Both operations are fully differentiable, on CPU or GPU:

```python
x = torch.tensor([[8., 0., 5., 3., 2., 1., 6., 7., 9.]], requires_grad=True).cuda()
y = torchsort.soft_sort(x)

torch.autograd.grad(y[0, 0], x)
# (tensor([[0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111]],
#         device='cuda:0'),)
```

## Benchmark

![Benchmark](https://github.com/teddykoker/torchsort/raw/main/extra/benchmark.png)

`torchsort` and `fast_soft_sort` each operate with a time complexity of *O(n log
n)*, each with some additional overhead when compared to the built-in
`torch.sort`. With a batch size of 1 (see left), the Numba JIT'd forward pass of
`fast_soft_sort` performs about on-par with the `torchsort` CPU kernel, however
its backward pass still relies on some Python code, which greatly penalizes its
performance. 

Furthermore, the `torchsort` kernel supports batches, and yields much better
performance than `fast_soft_sort` as the batch size increases.

CUDA kernel is coming soon!

## Reference

Please site the original paper:

```
@inproceedings{blondel2020fast,
  title={Fast differentiable sorting and ranking},
  author={Blondel, Mathieu and Teboul, Olivier and Berthet, Quentin and Djolonga, Josip},
  booktitle={International Conference on Machine Learning},
  pages={950--959},
  year={2020},
  organization={PMLR}
}
```
