# Copyright 2020 Google LLC
# Copyright 2021 Teddy Koker

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from .isotonic_cpu import isotonic_kl as isotonic_kl_cpu
from .isotonic_cpu import isotonic_kl_backward as isotonic_kl_backward_cpu
from .isotonic_cpu import isotonic_l2 as isotonic_l2_cpu
from .isotonic_cpu import isotonic_l2_backward as isotonic_l2_backward_cpu

try:
    from .isotonic_cuda import isotonic_kl as isotonic_kl_cuda
    from .isotonic_cuda import isotonic_kl_backward as isotonic_kl_backward_cuda
    from .isotonic_cuda import isotonic_l2 as isotonic_l2_cuda
    from .isotonic_cuda import isotonic_l2_backward as isotonic_l2_backward_cuda
except ImportError:
    isotonic_l2_cuda = None
    isotonic_kl_cuda = None
    isotonic_l2_backward_cuda = None
    isotonic_kl_backward_cuda = None


def soft_rank(values, regularization="l2", regularization_strength=1.0):
    if len(values.shape) != 2:
        raise ValueError(f"'values' should be a 2d-tensor but got {values.shape}")
    if regularization not in ["l2", "kl"]:
        raise ValueError(f"'regularization' should be a 'l2' or 'kl'")
    return SoftRank.apply(values, regularization, regularization_strength)


def soft_sort(values, regularization="l2", regularization_strength=1.0):
    if len(values.shape) != 2:
        raise ValueError(f"'values' should be a 2d-tensor but got {values.shape}")
    if regularization not in ["l2", "kl"]:
        raise ValueError(f"'regularization' should be a 'l2' or 'kl'")
    return SoftSort.apply(values, regularization, regularization_strength)


isotonic_l2 = {"cpu": isotonic_l2_cpu, "cuda": isotonic_l2_cuda}
isotonic_kl = {"cpu": isotonic_kl_cpu, "cuda": isotonic_kl_cuda}
isotonic_l2_backward = {
    "cpu": isotonic_l2_backward_cpu,
    "cuda": isotonic_l2_backward_cuda,
}
isotonic_kl_backward = {
    "cpu": isotonic_kl_backward_cpu,
    "cuda": isotonic_kl_backward_cuda,
}


def _arange_like(x, reverse=False):
    # returns arange with len of x of the same dtype and device (assumes 2d, first dim batch)
    if reverse:
        ar = torch.arange(x.shape[1] - 1, -1, -1, dtype=x.dtype, device=x.device)
    else:
        ar = torch.arange(x.shape[1], dtype=x.dtype, device=x.device)
    return ar.expand(x.shape[0], -1)


def _inv_permutation(permutation):
    # returns inverse permutation of 'permutation'. (assumes 2d, first dim batch)
    inv_permutation = torch.zeros_like(permutation)
    inv_permutation.scatter_(1, permutation, _arange_like(permutation))
    return inv_permutation


# The following is from google-research/fast-soft-sort with the following modifications:
# - replace numpy functions with torch equivalent
# - remove uncessary operations
# - reimplement backward pass in C++


class SoftRank(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, regularization="l2", regularization_strength=1.0):
        ctx.scale = 1.0 / regularization_strength
        ctx.regularization = regularization
        w = _arange_like(tensor, reverse=True) + 1
        theta = tensor * ctx.scale
        s, permutation = torch.sort(theta, descending=True)
        inv_permutation = _inv_permutation(permutation)
        if ctx.regularization == "l2":
            dual_sol = isotonic_l2[s.device.type](s - w)
            ret = (s - dual_sol).gather(1, inv_permutation)
            factor = torch.tensor(1.0, device=s.device)
        else:
            dual_sol = isotonic_kl[s.device.type](s, torch.log(w))
            ret = torch.exp((s - dual_sol).gather(1, inv_permutation))
            factor = ret

        ctx.save_for_backward(factor, s, dual_sol, permutation, inv_permutation)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        factor, s, dual_sol, permutation, inv_permutation = ctx.saved_tensors
        grad = (grad_output * factor).clone()
        if ctx.regularization == "l2":
            grad -= isotonic_l2_backward[s.device.type](
                s, dual_sol, grad.gather(1, permutation)
            ).gather(1, inv_permutation)
        else:
            grad -= isotonic_kl_backward[s.device.type](
                s, dual_sol, grad.gather(1, permutation)
            ).gather(1, inv_permutation)
        return grad * ctx.scale, None, None


class SoftSort(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, regularization="l2", regularization_strength=1.0):
        ctx.sign = -1
        ctx.regularization = regularization
        w = (_arange_like(tensor, reverse=True) + 1) / regularization_strength
        tensor = ctx.sign * tensor  # for ascending
        s, permutation = torch.sort(tensor, descending=True)

        # note reverse order of args
        if ctx.regularization == "l2":
            sol = isotonic_l2[s.device.type](w - s)
        else:
            sol = isotonic_kl[s.device.type](w, s)
        ctx.save_for_backward(s, sol, permutation)
        return ctx.sign * (w - sol)

    @staticmethod
    def backward(ctx, grad_output):
        s, sol, permutation = ctx.saved_tensors
        inv_permutation = _inv_permutation(permutation)
        if ctx.regularization == "l2":
            grad = isotonic_l2_backward[s.device.type](s, sol, grad_output)
        else:
            grad = isotonic_kl_backward[s.device.type](s, sol, grad_output)
        return grad.gather(1, inv_permutation), None, None
