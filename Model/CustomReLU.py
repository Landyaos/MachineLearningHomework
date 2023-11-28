import torch.nn as nn
from Function.ReLUFunction import ReLUFunction


class CustomReLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return ReLUFunction.apply(input)

