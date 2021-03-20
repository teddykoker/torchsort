# Torchsort

**NOTE**: The C++ isotonic regression solver is currently slower than the Numba
JIT version by the original authors. Please use
[google-research/fast-soft-sort](https://github.com/google-research/fast-soft-sort)
in the mean time.

A fast and differentiable sorting and ranking in PyTorch.

Pure PyTorch implementation of [Fast Differentiable Sorting and
Ranking](https://arxiv.org/abs/2002.08871) (Blondel et al.). Much of the code is
copied from the original Numpy implementation at
[google-research/fast-soft-sort](https://github.com/google-research/fast-soft-sort),
with the isotonic regression solver rewritten as a PyTorch C++ Extension.

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
