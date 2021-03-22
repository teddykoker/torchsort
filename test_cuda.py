import torch
import torchsort

x = torch.randn(1, 10).cuda()
print(x)
torchsort.isotonic_cuda.isotonic_l2(x)
print("did't fail")

