import torch
from torch.autograd import Function


class BatchNorm2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_var, momentum, training=True, eps=1e-5):
        ctx.training = training
        ctx.eps = eps
        if ctx.training:
            mean = input.mean(dim=(0, 2, 3), keepdim=True)
            var = input.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            running_mean = (1 - momentum) * running_mean + momentum * mean
            running_var = (1 - momentum) * running_var + momentum * var
            input_normalized = (input - mean) / torch.sqrt(var + eps)
            ctx.save_for_backward(input, input_normalized, weight, mean, var)

        else:
            print(running_mean)
            input_normalized = (input - running_mean) / \
                torch.sqrt(running_var + eps)
        
        output = weight.view(1, -1, 1, 1) * \
            input_normalized + bias.view(1, -1, 1, 1)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, input_normalized, weight, mean, var = ctx.saved_tensors
        # 计算梯度
        grad_weight = (grad_output * input_normalized).sum(dim=(0, 2, 3))
        grad_bias = grad_output.sum(dim=(0, 2, 3))

        grad_input_normalized = grad_output * weight.view(1, -1, 1, 1)
        N = input.size(0) * input.size(2) * input.size(3)
        grad_var = (grad_input_normalized * (input - mean) * (-0.5)
                    * (var + ctx.eps).pow(-1.5)).sum(dim=(0, 2, 3))
        grad_mean = grad_input_normalized.sum(dim=(0, 2, 3)) * (-1.0 / torch.sqrt(var + ctx.eps)).sum(
            dim=(0, 2, 3)) + grad_var * (-2.0 / N) * (input - mean).sum(dim=(0, 2, 3))

        # 计算输入梯度
        grad_input = grad_input_normalized * 1.0 / torch.sqrt(var + ctx.eps) + grad_var.reshape(
            var.shape) * 2.0 / N * (input - mean) + grad_mean.reshape(mean.shape) / N

        return grad_input, grad_weight, grad_bias, None, None, None, None, None
