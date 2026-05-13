import math
import torch
import torch.nn as nn

class AnchorGenerator(nn.Module):
    def __init__(
        self,
        scales=(1.0, 2**(1/3), 2**(2/3)),   # octave sub-divisions
        aspect_ratios=(0.5, 1.0, 2.0),
        anchor_scale=4.0,                     # base_size = stride × anchor_scale
        strides=(8, 16, 32, 64, 128),         # one per pyramid level P3–P7
    ):
        super().__init__()
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.anchor_scale = anchor_scale
        self.strides = strides

        # Precompute the 9 (w_mult, h_mult) pairs
        # Shape: (9, 2) — stays on CPU, moved to device in forward()
        self._base_anchors = self._make_base_anchors()

        # Cache: maps (H, W, device_str) → anchor tensor
        # Avoids recomputing the same grid every training step
        self._cache = {}

    def _make_base_anchors(self):
        """
        Returns tensor of shape (num_anchors, 2) = (9, 2).
        Each row is (w_multiplier, h_multiplier).
        Multiply by (stride * anchor_scale) to get actual pixel sizes.

        The loop order is: scales outermost, aspect_ratios innermost.
        This determines anchor variant index 0..8.
        THIS ORDER MUST MATCH how the head outputs are flattened.
        """

        anchors = []
        for scale in self.scales:
            for aspect_ratio in self.aspect_ratios:
                w = scale * math.sqrt(aspect_ratio)
                h = scale / math.sqrt(aspect_ratio)
                anchors.append((w, h))

        return torch.tensor(anchors, dtype=torch.float32)

    def _anchors_for_level(self, H, W, stride, device):
        """
        Generates all anchors for one pyramid level.

        Args:
            H, W   : feature map spatial dimensions
            stride : how many input pixels per feature cell
            device : torch.device

        Returns:
            Tensor of shape (H * W * num_anchors, 4) in (cx, cy, w, h) format
        """
        base_size = stride * self.anchor_scale  # e.g. stride=8 → 32.0

        # ── Grid centers ────────────────────────────────────────────────
        # Each feature cell (i, j) covers input pixels from
        #   [j*stride, (j+1)*stride) horizontally
        #   [i*stride, (i+1)*stride) vertically
        # The anchor center is placed at the middle of this region.

        # shifts_x[j] = center x-coordinate of column j = (j + 0.5) * stride
        shifts_x = (torch.arange(W, dtype=torch.float32, device=device) + 0.5) * stride
        shifts_y = (torch.arange(H, dtype=torch.float32, device=device) + 0.5) * stride

        # meshgrid: grid_y[i, j] = y-coord of row i
        #           grid_x[i, j] = x-coord of column j
        # indexing='ij' → rows come from first argument (shifts_y)
        grid_y, grid_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        # Both are shape (H, W)

        # Flatten to 1D: visits all H*W cells in row-major order
        # (row 0 all columns, then row 1 all columns, ...)
        grid_x = grid_x.reshape(-1)  # shape: (H*W,)
        grid_y = grid_y.reshape(-1)  # shape: (H*W,)

        # ── Anchor dimensions ────────────────────────────────────────────
        base = self._base_anchors.to(device)   # (9, 2)
        ws = base_size * base[:, 0]             # (9,)  — actual widths in px
        hs = base_size * base[:, 1]             # (9,)  — actual heights in px

        # ── Combine: every (location, anchor_variant) pair ───────────────
        num_locs    = H * W   # e.g. 4096 for P3
        num_anchors = len(self.scales) * len(self.aspect_ratios)  # 9

        # Expand centers: (H*W,) → (H*W, 9)
        # unsqueeze(1) adds a column dimension → (H*W, 1)
        # expand(..., num_anchors) broadcasts along that dimension → (H*W, 9)
        cx = grid_x.unsqueeze(1).expand(num_locs, num_anchors)  # (H*W, 9)
        cy = grid_y.unsqueeze(1).expand(num_locs, num_anchors)  # (H*W, 9)

        # Expand dimensions: (9,) → (H*W, 9)
        # unsqueeze(0) adds a row dimension → (1, 9)
        # expand broadcasts along rows → (H*W, 9)
        w = ws.unsqueeze(0).expand(num_locs, num_anchors)  # (H*W, 9)
        h = hs.unsqueeze(0).expand(num_locs, num_anchors)  # (H*W, 9)

        # Stack: 4 tensors each (H*W, 9) → (H*W, 9, 4)
        level_anchors = torch.stack([cx, cy, w, h], dim=-1)

        # Flatten: (H*W, 9, 4) → (H*W*9, 4)
        # The reshape visits dim-1 (anchor variants) innermost,
        # which matches how the head output is flattened.
        return level_anchors.reshape(-1, 4)  # (H*W*9, 4)

    @torch.no_grad()
    def forward(self, feature_map_sizes, device='cpu'):
        """
        Args:
            feature_map_sizes : list of (H_i, W_i) for each pyramid level
                                e.g. [(64,64), (32,32), (16,16), (8,8), (4,4)]
            device            : 'cpu', 'cuda', or torch.device

        Returns:
            anchors : Tensor (N_total, 4) in (cx, cy, w, h) format
                      N_total = sum of H_i * W_i * 9 over all levels
                      e.g. 49,104 for a 512×512 input
        """
        # Build a hashable cache key
        device_str = str(device)
        cache_key = (tuple(feature_map_sizes), device_str)

        if cache_key in self._cache:
            return self._cache[cache_key]

        all_anchors = []
        for (H, W), stride in zip(feature_map_sizes, self.strides):
            level_anchors = self._anchors_for_level(H, W, stride, device)
            all_anchors.append(level_anchors)

        # Concatenate along dim=0: list of 5 tensors → (N_total, 4)
        anchors = torch.cat(all_anchors, dim=0)

        self._cache[cache_key] = anchors
        return anchors