import torch
from torch.autograd import Function

class LeakyReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, negative_slope=0.01):
        ctx.save_for_backward(input)
        ctx.negative_slope = negative_slope
        output = input.clone()
        output[input < 0] = output[input < 0] * negative_slope
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] *= ctx.negative_slope
        return grad_input, None
