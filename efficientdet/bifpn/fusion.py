"""Fast normalized feature fusion, the weighted-sum node used throughout the BiFPN."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastNormalizedFusion(nn.Module):
    """Fuse same-shaped feature maps with learnable, non-negative weights.

    Computes sum_i(w_i * f_i) / (sum_i(w_i) + epsilon) with w = ReLU(raw
    weights), the cheaper softmax-free fusion from the EfficientDet paper.

    Args:
        num_inputs : number of feature maps fused at this node
        epsilon    : keeps the division stable when all weights are near zero
    """

    def __init__(self, num_inputs: int, epsilon: float = 1e-4):
        super().__init__()
        self.num_inputs = num_inputs
        self.w = nn.Parameter(torch.ones(num_inputs))
        self.epsilon = epsilon

    def forward(self, *features: torch.Tensor):
        """features: num_inputs tensors of identical shape -> one fused tensor of that shape."""
        assert len(features) == self.num_inputs
        w = F.relu(self.w)
        fused = sum([w[i] * features[i] for i in range(self.num_inputs)]) / (w.sum() + self.epsilon)
        return fused
