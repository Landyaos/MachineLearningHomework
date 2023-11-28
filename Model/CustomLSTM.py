import torch

class CustomLSTM(torch.Module):
    def __init__(self) -> None:
        super().__init__()