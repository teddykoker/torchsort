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
from isotonic_cpu import isotonic_kl as isotonic_kl_cpu
from isotonic_cpu import isotonic_l2 as isotonic_l2_cpu


def soft_rank(values, regularization="l2", regularization_strength=1.0):
    device = values.device
    if len(values.shape) != 2:
        raise ValueError(f"'values' should be a 2d-tensor but got {values.shape}")
    return torch.stack(
        [
            SoftRank.apply(t, regularization, regularization_strength)
            for t in torch.unbind(values.cpu())
        ]
    ).to(device)


def soft_sort(values, regularization="l2", regularization_strength=1.0):
    device = values.device
    if len(values.shape) != 2:
        raise ValueError(f"'values' should be a 2d-tensor but got {values.shape}")
    return torch.stack(
        [
            SoftSort.apply(t, regularization, regularization_strength)
            for t in torch.unbind(values.cpu())
        ]
    ).to(device)


class SoftRank(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, regularization="l2", regularization_strength=1.0):
        ctx.scale = 1.0 / regularization_strength
        w = _arange_like(tensor, reverse=True) + 1
        if regularization == "l2":
            ctx.projection = Projection(tensor * ctx.scale, w, regularization)
            ctx.factor = 1.0
            return ctx.projection.compute()
        else:
            ctx.projection = Projection(
                tensor * ctx.scale, torch.log(w), regularization
            )
            ctx.factor = torch.exp(ctx.projection.compute())
            return ctx.factor

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.projection.vjp(ctx.factor * grad_output) * ctx.scale
        return out, None, None


def _arange_like(x, reverse=False):
    if reverse:
        return torch.arange(x.shape[0] - 1, -1, -1, dtype=x.dtype, device=x.device)
    return torch.arange(x.shape[0], dtype=x.dtype, device=x.device)


class SoftSort(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, regularization="l2", regularization_strength=1.0):
        ctx.sign = -1
        w = (_arange_like(tensor, reverse=True) + 1) / regularization_strength
        tensor = ctx.sign * tensor  # for ascending
        s, ctx.permutation = torch.sort(tensor, descending=True)
        ctx.isotonic = Isotonic(w, s, regularization=regularization)
        ret = ctx.isotonic.compute()
        ctx.isotonic.s = s
        return ctx.sign * (w - ret)

    @staticmethod
    def backward(ctx, grad_output):
        inv_permutation = _inv_permutation(ctx.permutation)
        return ctx.isotonic.vjp(grad_output)[inv_permutation], None, None


# Below is copied from google-research/fast-soft-sort with the following modifications:
# - replace numpy functions with torch equivalent


def isotonic_l2(s, w=None):
    if w is None:
        w = _arange_like(s, reverse=True) + 1
    return isotonic_l2_cpu(s - w)


def isotonic_kl(s, w=None):
    if w is None:
        w = _arange_like(s, reverse=True) + 1
    return isotonic_kl_cpu(s, w)


def _partition(solution, eps=1e-9):
    """Returns partition corresponding to solution."""
    if len(solution) == 0:
        return []

    sizes = [1]

    for i in range(1, len(solution)):
        if abs(solution[i] - solution[i - 1]) > eps:
            sizes.append(0)
        sizes[-1] += 1

    return sizes


class Isotonic:
    """Isotonic optimization."""

    def __init__(self, s, w, regularization="l2"):
        self.s = s
        self.w = w
        self.regularization = regularization
        self._solution = None

    def compute(self):
        if self.regularization == "l2":
            self._solution = isotonic_l2(self.s, self.w)
        else:
            self._solution = isotonic_kl(self.s, self.w)
        return self._solution

    def vjp(self, vector):
        start = 0
        ret = torch.zeros_like(self._solution)
        for size in _partition(self._solution):
            end = start + size
            if self.regularization == "l2":
                val = 1.0 / size
            else:
                val = torch.softmax(self.s[start:end], dim=0)
            ret[start:end] = val * torch.sum(vector[start:end])
            start = end
        return ret


def _inv_permutation(permutation):
    """Returns inverse permutation of 'permutation'."""
    inv_permutation = torch.zeros_like(permutation)
    inv_permutation.scatter_(0, permutation, _arange_like(permutation))
    return inv_permutation


class Projection:
    """Computes projection onto the permutahedron P(w)."""

    def __init__(self, theta, w=None, regularization="l2"):
        if w is None:
            w = _arange_like(theta) + 1
        self.theta = theta
        self.w = w
        self.regularization = regularization

    def compute(self):
        s, self.permutation = torch.sort(self.theta, descending=True)
        self._isotonic = Isotonic(s, self.w, self.regularization)
        dual_sol = self._isotonic.compute()
        primal_sol = s - dual_sol
        self.inv_permutation = _inv_permutation(self.permutation)
        return primal_sol[self.inv_permutation]

    def vjp(self, vector):
        ret = vector.clone()
        ret -= self._isotonic.vjp(vector[self.permutation])[self.inv_permutation]
        return ret
