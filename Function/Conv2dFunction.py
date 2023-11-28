import torch
import torch.nn as nn
from torch.autograd import Function
Tensor = torch.Tensor


class Conv2dFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding

        b_N, in_C, in_H, in_W = input.shape
        kernel_size = weight.shape[2]
        out_C = weight.shape[0]
        out_H = (in_H + padding * 2 - kernel_size) // stride + 1
        out_W = (in_W + padding * 2 - kernel_size) // stride + 1

        input_pad = torch.zeros((b_N, in_C, in_H+padding*2, in_W+padding*2),
                                dtype=input.dtype, device=input.device)
        input_pad[:, :, padding:padding+in_H, padding:padding+in_W] = input

        input_col = torch.empty((b_N, out_H*out_W, in_C * kernel_size**2),
                                dtype=input.dtype, device=input.device)
        for idx_H in range(out_H):
            for idx_W in range(out_W):
                input_col[:, idx_H*out_W + idx_W, :] =\
                    input_pad[:, :,
                              idx_H*stride:idx_H*stride + kernel_size,
                              idx_W*stride:idx_W*stride+kernel_size].reshape(b_N, -1)
        input_col.transpose_(2, 1)

        ctx.input_col = input_col

        weight_col = weight.reshape(out_C, -1)

        output = (torch.matmul(weight_col, input_col)+bias.reshape(-1, 1))\
            .reshape(b_N, out_C, out_H, out_W)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        b_N, out_C, out_H, out_W = grad_output.shape
        _, in_C, in_H, in_W = input.shape
        kernel_size = weight.shape[2]
        in_H_pad, in_W_pad = in_H+padding*2, in_W+padding*2

        weight_invert_col = torch.flip(weight, [2, 3])\
            .transpose(1, 0).reshape(in_C, -1)
        grad_output_pad = \
            torch.zeros((b_N, out_C, out_H+(kernel_size-1) * 2, out_W+(kernel_size-1)*2),
                        dtype=input.dtype, device=input.device)
        grad_output_pad[:, :,
                        kernel_size-1:kernel_size - 1+out_H,
                        kernel_size-1:kernel_size - 1+out_W] = grad_output
        grad_output_pad_col =\
            torch.empty((b_N, in_H_pad * in_W_pad, out_C*kernel_size**2),
                        dtype=input.dtype, device=input.device)
        for idx_H in range(in_H_pad):
            for idx_W in range(in_W_pad):
                grad_output_pad_col[:, idx_H*in_W_pad+idx_W, :] =\
                    grad_output_pad[:, :,
                                    idx_H*stride:idx_H*stride + kernel_size,
                                    idx_W*stride:idx_W*stride + kernel_size].reshape(b_N, -1)
        grad_output_pad_col.transpose_(2, 1)
        grad_input = torch.matmul(weight_invert_col, grad_output_pad_col)\
            .reshape(b_N, in_C, in_H_pad, in_W_pad)[:, :, padding:padding+in_H, padding:padding+in_W]

        input_col = ctx.input_col
        output_col = grad_output.reshape(b_N, out_C, -1)
        grad_weight = torch.matmul(output_col, input_col.transpose(1, 2))\
            .sum(0).reshape(weight.shape)

        grad_bias = grad_output.sum((0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None
