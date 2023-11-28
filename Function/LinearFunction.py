import torch
import torch.nn as nn
from torch.autograd import Function


class LinearFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        
        output = torch.matmul(input, weight.T)
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = torch.matmul(grad_output, weight)
        grad_weight = torch.matmul(grad_output.T, input)
        grad_bias = torch.sum(grad_output, 0)
        return grad_input, grad_weight, grad_bias
