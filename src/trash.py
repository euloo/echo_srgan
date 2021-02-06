import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math


arch = vgg19(pretrained=True)
print(arch)
#print(arch.features)
#print(arch.children())
#print(list(arch.features.children())[2:36])

#feature_extractor = nn.Sequential(*list(arch.features.children())[:18])

arch = list(arch.children())
        #print(arch)
print(arch[0])
print(f'layer {arch[0][0]}')
w = arch[0][0].weight
print(f'weight {w.shape}')
arch[0][0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
print(f'new layer {arch[0][0]}')
arch[0][0].weight = nn.Parameter(torch.mean(w, dim=1, keepdim=True))
print(f'new weight {arch[0][0].weight.shape}')
