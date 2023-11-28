import torch
import torch.nn as nn
from Function.Conv2dFunction import Conv2dFunction


class CustomConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.empty(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.empty(self.out_channels))

        # ToDo: find a smart way to initialize weights and bias
        nn.init.uniform_(self.weight, -0.1, 0.1)
        nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input):
        return Conv2dFunction.apply(input, self.weight, self.bias, self.stride, self.padding)
