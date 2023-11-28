import torch
import torch.nn as nn
from Function.CrossEntropyFunction import CrossEntropyFunction


class CustomCrossEntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        
        return CrossEntropyFunction.apply(input, target)
