from functools import partial

import pytest
import torch
from fast_soft_sort import pytorch_ops as fss

from torchsort import soft_rank, soft_sort

EPS = 1e-5
ATOL = 1e-3
RTOL = 1e-3
BATCH_SIZE = 8
SEQ_LEN = 10

REGULARIZATION = ["l2", "kl"]
REGULARIZATION_STRENGTH = [1e-1, 1e0, 1e1]

DEVICES = [torch.device("cpu")] + (
    [torch.device("cuda")] if torch.cuda.is_available() else []
)

torch.manual_seed(0)


@pytest.mark.parametrize("function", [soft_rank, soft_sort])
@pytest.mark.parametrize("regularization", REGULARIZATION)
@pytest.mark.parametrize("regularization_strength", REGULARIZATION_STRENGTH)
@pytest.mark.parametrize("device", DEVICES)
def test_gradcheck(function, regularization, regularization_strength, device):
    x = torch.randn(BATCH_SIZE, SEQ_LEN, dtype=torch.float64, requires_grad=True).to(
        device
    )
    f = partial(
        function,
        regularization=regularization,
        regularization_strength=regularization_strength,
    )
    torch.autograd.gradcheck(f, [x], eps=EPS, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "funcs",
    [(soft_rank, fss.soft_rank), (soft_sort, fss.soft_sort)],
)
@pytest.mark.parametrize("regularization", REGULARIZATION)
@pytest.mark.parametrize("regularization_strength", REGULARIZATION_STRENGTH)
@pytest.mark.parametrize("device", DEVICES)
def test_vs_original(funcs, regularization, regularization_strength, device):
    # test that torchsort outputs are consistent with the outputs of the code provided
    # from the original paper
    x = torch.randn(BATCH_SIZE, SEQ_LEN, dtype=torch.float64, requires_grad=True).to(
        device
    )
    kwargs = {
        "regularization": regularization,
        "regularization_strength": regularization_strength,
    }
    assert torch.allclose(
        funcs[0](x, **kwargs).cpu(),
        funcs[1](x.cpu(), **kwargs),
    )


@pytest.mark.parametrize("function", [soft_rank, soft_sort])
@pytest.mark.parametrize("regularization", REGULARIZATION)
@pytest.mark.parametrize("regularization_strength", REGULARIZATION_STRENGTH)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to test fp16")
def test_half(function, regularization, regularization_strength, device):
    x = torch.randn(BATCH_SIZE, SEQ_LEN, dtype=torch.float16, requires_grad=True).cuda()
    f = partial(
        function,
        regularization=regularization,
        regularization_strength=regularization_strength,
    )
    f(x)
