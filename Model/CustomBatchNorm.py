import torch
import torch.nn as nn
from Function.BatchNorm2d import BatchNorm2d


class CustomBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super(CustomBatchNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # 创建参数变量
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(1,num_features,1,1))
        self.register_buffer('running_var', torch.ones(1,num_features,1,1))

    def forward(self, input):
        output = BatchNorm2d().apply(input, self.weight,self.bias, self.running_mean, self.running_var, self.momentum, self.training, self.eps)
        return output
