import torch.nn as nn
from Function.LeakyReLUFunction import LeakyReLUFunction


class CustomLeakyReLU(nn.Module):

    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        return LeakyReLUFunction.apply(input, self.negative_slope)
