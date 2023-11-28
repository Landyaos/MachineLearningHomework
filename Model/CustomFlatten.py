import torch
import torch.nn as nn

from Function.FlattenFunction import FlattenFunction


class CustomFlatten(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return FlattenFunction.apply(input)