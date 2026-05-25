import math
import torch
import torch.nn as nn


class AnchorGenerator(nn.Module):
    def __init__(
        self,
        scales=(1.0, 2**(1/3), 2**(2/3)),
        aspect_ratios=(0.5, 1.0, 2.0),
        anchor_scale=4.0,
        strides=(8, 16, 32, 64, 128),
    ):
        super().__init__()
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.anchor_scale = anchor_scale
        self.strides = strides

        # Precompute the 9 (w_mult, h_mult) pairs — shape (9, 2), stays on CPU
        self._base_anchors = self._make_base_anchors()

        # Cache: maps (H, W, device_str) → anchor tensor
        self._cache = {}

    def _make_base_anchors(self):
        """
        Returns tensor of shape (num_anchors, 2) = (9, 2).
        Each row is (w_multiplier, h_multiplier).
        Loop order: scales outermost, aspect_ratios innermost — must match head flattening.
        """
        anchors = []
        for scale in self.scales:
            for aspect_ratio in self.aspect_ratios:
                w = scale * math.sqrt(aspect_ratio)
                h = scale / math.sqrt(aspect_ratio)
                anchors.append((w, h))
        return torch.tensor(anchors, dtype=torch.float32)

    def _anchors_for_level(self, H, W, stride, device):
        """Returns (H * W * num_anchors, 4) in (cx, cy, w, h) format."""
        base_size = stride * self.anchor_scale

        shifts_x = (torch.arange(W, dtype=torch.float32, device=device) + 0.5) * stride
        shifts_y = (torch.arange(H, dtype=torch.float32, device=device) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)

        base = self._base_anchors.to(device)
        ws = base_size * base[:, 0]
        hs = base_size * base[:, 1]

        num_locs = H * W
        num_anchors = len(self.scales) * len(self.aspect_ratios)

        cx = grid_x.unsqueeze(1).expand(num_locs, num_anchors)
        cy = grid_y.unsqueeze(1).expand(num_locs, num_anchors)
        w = ws.unsqueeze(0).expand(num_locs, num_anchors)
        h = hs.unsqueeze(0).expand(num_locs, num_anchors)

        return torch.stack([cx, cy, w, h], dim=-1).reshape(-1, 4)

    @torch.no_grad()
    def forward(self, feature_map_sizes, device='cpu'):
        """
        Args:
            feature_map_sizes: list of (H_i, W_i) for each pyramid level
            device: 'cpu', 'cuda', or torch.device
        Returns:
            anchors: Tensor (N_total, 4) in (cx, cy, w, h) format
        """
        device_str = str(device)
        cache_key = (tuple(feature_map_sizes), device_str)

        if cache_key in self._cache:
            return self._cache[cache_key]

        all_anchors = [
            self._anchors_for_level(H, W, stride, device)
            for (H, W), stride in zip(feature_map_sizes, self.strides)
        ]
        anchors = torch.cat(all_anchors, dim=0)
        self._cache[cache_key] = anchors
        return anchors
