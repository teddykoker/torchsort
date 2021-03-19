from functools import partial

import pytest
import torch

from torchsort import soft_rank, soft_sort

EPS = 1e-5
ATOL = 1e-3
RTOL = 1e-3

REGULARIZATION = ["l2", "kl"]
REGULARIZATION_STRENGTH = [1e-1, 1e0, 1e1]


@pytest.mark.parametrize("function", [soft_rank, soft_sort])
@pytest.mark.parametrize("regularization", REGULARIZATION)
@pytest.mark.parametrize("regularization_strength", REGULARIZATION_STRENGTH)
def test_soft_rank(function, regularization, regularization_strength):
    x = torch.randn(5, 10, dtype=torch.float64, requires_grad=True)
    f = partial(
        function,
        regularization=regularization,
        regularization_strength=regularization_strength,
    )
    torch.autograd.gradcheck(f, [x], eps=EPS, atol=ATOL, rtol=RTOL)
