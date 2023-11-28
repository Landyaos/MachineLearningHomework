import torch

from torch.autograd import Function




class FlattenFunction(Function):

    @staticmethod
    def forward(ctx,input):
        ctx.input_shape = input.shape
        return input.reshape(input.shape[0],-1)

    @staticmethod
    def backward(ctx,grad_output):
        return grad_output.reshape(ctx.input_shape)