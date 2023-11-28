import torch.nn as nn
from Function.MaxPool2dFunction import MaxPool2dFunction


class CustomLinear(nn.Module):

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        return MaxPool2dFunction.apply(input, self.kernel_size, self.stride)
