import torch
from torch.autograd import Function
Tensor = torch.Tensor


class MaxPool2dFunction(Function):
    """
    Args:
        input: Tensor
        kernel_size: int
        stride: int
        padding: int
    """
    @staticmethod
    def forward(ctx, input, kernel_size, stride):
        ctx.input_shape = input.shape
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.dtype = input.dtype
        ctx.device = input.device

        batch_size, channels, input_height, input_width = input.shape

        output_height = (input_height - kernel_size) // stride + 1
        output_width = (input_width - kernel_size) // stride + 1

        output_col = torch.empty((batch_size, channels, output_height*output_width, kernel_size**2),
                                 dtype=input.dtype, device=input.device)

        for idx_h in range(output_height):
            for idx_w in range(output_width):
                output_col[:, :, idx_h*output_width+idx_w, :] = \
                    input[:, :, idx_h*stride:idx_h*stride+kernel_size, idx_w*stride:idx_w*stride+kernel_size]\
                    .reshape(batch_size, channels, -1)
        output = output_col.max(dim=3, keepdim=True)[0]
        ctx.output_indices = output_col == output
        return output.reshape(batch_size, channels, output_height, output_width)

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.input_shape
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        # indices = ctx.indices
        grad_input = \
            torch.zeros(input_shape, dtype=ctx.dtype, device=ctx.device)
        grad_input_col = \
            grad_output.reshape(grad_output.shape[0], grad_output.shape[1], -1)\
            .repeat_interleave(kernel_size**2, 2).reshape(ctx.output_indices.shape)*ctx.output_indices
        for idx_h in range(grad_output.shape[2]):
            for idx_w in range(grad_output.shape[3]):
                grad_input[:, :, idx_h*stride:idx_h*stride+kernel_size, idx_w*stride:idx_w*stride+kernel_size] =\
                    grad_input_col[:, :, idx_h*grad_output.shape[3] +idx_w].reshape(input_shape[0],input_shape[1],kernel_size,kernel_size)

        return grad_input, None, None