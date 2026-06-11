import torch
import torch.nn as nn

class Softplus_jitter(nn.Softplus):
    """jitter(マージン)付きのsoftplus活性化関数

    Attributes:
        jitter (float): jitter値。
    """
    def __init__(self, beta:float = 1.0, threshold:float = 20.0, jitter:float = 1e-4):
        super().__init__(beta, threshold)
        self.jitter = jitter

    def forward(self, x:torch.Tensor):
        return super().forward(x) + self.jitter