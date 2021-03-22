import torch
import torchsort

x = torch.randn(1, 10).cuda()
y = torch.arange(10).expand(1, -1).float().cuda()
print(x)
print(torchsort.isotonic_cuda.isotonic_l2(x))
print(torchsort.isotonic_cuda.isotonic_kl(x, y))
print("did't fail")

