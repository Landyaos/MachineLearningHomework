import torch
from torch.optim.optimizer import Optimizer

class Adagrad(Optimizer):
    def __init__(self, params, lr=1e-2, eps=1e-10):
        defaults = dict(lr=lr, eps=eps)
        super(Adagrad, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                state = self.state[param]

                if 'sum' not in state:
                    state['sum'] = torch.zeros_like(param.data)
                sum_ = state['sum']
                sum_.add_(grad ** 2)
                std = torch.sqrt(sum_ + eps)
                param.data.addcdiv_(-lr, grad, std)

    def zero_grad(self):
        for group in self.param_groups:
            for param in group['params']:
                param.grad = None
