import torch
import torch.nn as nn
import torch.nn.functional as F


class FastNormalizedFusion(nn.Module):
    def __init__(self, num_inputs: int, epsilon: float = 1e-4):
        super().__init__()
        self.num_inputs = num_inputs
        self.w = nn.Parameter(torch.ones(num_inputs))
        self.epsilon = epsilon

    def forward(self, *features: torch.Tensor):
        assert len(features) == self.num_inputs
        w = F.relu(self.w)
        fused = sum([w[i] * features[i] for i in range(self.num_inputs)]) / (w.sum() + self.epsilon)
        return fused
