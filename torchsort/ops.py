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


def soft_rank(values, regularization="l2", regularization_strength=1.0):
    if len(values.shape) != 2:
        raise ValueError(f"'values' should be a 2d-tensor but got {values.shape}")
    device = values.device
    return _apply_batch(
        SoftRank.apply, values.cpu(), regularization, regularization_strength
    ).to(device)


def soft_sort(values, regularization="l2", regularization_strength=1.0):
    if len(values.shape) != 2:
        raise ValueError(f"'values' should be a 2d-tensor but got {values.shape}")
    device = values.device
    return _apply_batch(
        SoftSort.apply, values.cpu(), regularization, regularization_strength
    ).to(device)


def _apply_batch(f, tensor, *args, **kwargs):
    """Apply function f with args to the first dimension of tensor and then stack"""
    # TODO: this shouldn't be necessary, need to implement extra dimension in C++ code.
    return torch.stack([f(t, *args, **kwargs) for t in torch.unbind(tensor)])


def _arange_like(x, reverse=False):
    """Returns arange with len of x of the same dtype and device"""
    if reverse:
        return torch.arange(x.shape[0] - 1, -1, -1, dtype=x.dtype, device=x.device)
    return torch.arange(x.shape[0], dtype=x.dtype, device=x.device)


def _inv_permutation(permutation):
    """Returns inverse permutation of 'permutation'."""
    inv_permutation = torch.zeros_like(permutation)
    inv_permutation.scatter_(0, permutation, _arange_like(permutation))
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
            dual_sol = isotonic_l2_cpu(s - w)
            ret = (s - dual_sol)[inv_permutation]
            ctx.factor = 1.0
        else:
            dual_sol = isotonic_kl_cpu(s, torch.log(w))
            ret = torch.exp((s - dual_sol)[inv_permutation])
            ctx.factor = ret

        ctx.save_for_backward(s, dual_sol, permutation, inv_permutation)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        grad = (grad_output * ctx.factor).clone()
        s, dual_sol, permutation, inv_permutation = ctx.saved_tensors
        if ctx.regularization == "l2":
            grad -= isotonic_l2_backward_cpu(s, dual_sol, grad[permutation])[
                inv_permutation
            ]
        else:
            grad -= isotonic_kl_backward_cpu(s, dual_sol, grad[permutation])[
                inv_permutation
            ]
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
            sol = isotonic_l2_cpu(w - s)
        else:
            sol = isotonic_kl_cpu(w, s)
        ctx.save_for_backward(s, sol, permutation)
        return ctx.sign * (w - sol)

    @staticmethod
    def backward(ctx, grad_output):
        s, sol, permutation = ctx.saved_tensors
        inv_permutation = _inv_permutation(permutation)
        if ctx.regularization == "l2":
            grad = isotonic_l2_backward_cpu(s, sol, grad_output)
        else:
            grad = isotonic_kl_backward_cpu(s, sol, grad_output)
        return grad[inv_permutation], None, None
