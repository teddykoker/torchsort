import torch
import torchsort
import numpy as np

from fast_soft_sort.third_party.isotonic import isotonic_l2, isotonic_kl

y = torch.randn(1, 10).cuda()
w = torch.arange(10).expand(1, -1).float().cuda()

print(torchsort.isotonic_cuda.isotonic_l2(y))
sol = np.zeros((10,))
isotonic_l2(y[0].cpu().numpy(), sol)
print(sol)

print("##############")

print(torchsort.isotonic_cuda.isotonic_kl(y, w))
sol = np.zeros((10,))
isotonic_kl(y[0].cpu().numpy(), w[0].cpu().numpy(), sol)
print(sol)
