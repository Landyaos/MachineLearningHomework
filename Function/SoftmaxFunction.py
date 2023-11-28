import torch
from torch.autograd import Function


class SoftmaxFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input_max, _ = torch.max(input=input, dim=1, keepdim=True)
        input_exp = torch.exp(input - input_max)
        output = \
            torch.div(input_exp, torch.sum(input_exp, dim=1, keepdim=True))
        ctx.output = output
        return output

    @staticmethod
    def backward(ctx, grad_output):
        pass
