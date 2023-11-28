import torch
import torch.nn as nn
from Function.LinearFunction import LinearFunction


class CustomLinear(nn.Module):

    def __init__(self, input_features, output_features, bias=True):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(
            torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            self.register_parameter('bias', None)

        # ToDo: find a smart way to initialize weights and bias
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input):
        return LinearFunction.apply(input, self.weight, self.bias)

