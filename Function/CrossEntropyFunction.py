import torch
from torch.autograd import Function

class CrossEntropyFunction(Function):

    @staticmethod
    def forward(ctx, input, target):

        probs = torch.softmax(input, dim=1)

        # input_max, _ = torch.max(input=input, dim=1, keepdim=True)
        # input_exp = torch.exp(input - input_max)
        # probs = \
        #     torch.div(input_exp, torch.sum(input_exp, dim=1, keepdim=True))

        ctx.save_for_backward(probs, target)
        loss = torch.mean(torch.sum(-target * torch.log(probs), dim=1))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        probs, target, = ctx.saved_tensors
        grad_input = (probs - target) / target.shape[0]
        return grad_input, None
