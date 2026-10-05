"""
Compound-scaling configuration for EfficientDet D0-D7.

A single coefficient `phi` scales every part of the network together, following
the EfficientDet paper: backbone size, BiFPN width and depth, head depth, and
input resolution all grow with `phi`.
"""


class EfficientDetConfig:
    """Network dimensions derived from the compound-scaling coefficient `phi`.

    Attributes:
        phi              : scaling coefficient, 0-7 (D0-D7)
        out_channels     : BiFPN / head channel width (64 at phi 0)
        num_bifpn_layers : number of stacked BiFPN layers (3 + phi)
        num_head_layers  : conv layers in each class/box head (3 + phi // 3)
        input_resolution : square input size in pixels (512 + 128 * phi)
        backbone_name    : timm EfficientNet variant (D7 reuses b6)
    """

    def __init__(self, phi: int = 0):
        if not (0 <= phi <= 7):
            raise ValueError(f"phi must be 0–7, got {phi}")

        self.phi = phi
        self.out_channels = int(round(64 * (1.35 ** phi) / 8)) * 8
        self.num_bifpn_layers = 3 + phi
        self.num_head_layers = 3 + phi // 3
        self.input_resolution = 512 + phi * 128
        self.backbone_name = [
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "efficientnet_b3",
            "efficientnet_b4",
            "efficientnet_b5",
            "efficientnet_b6",
            "efficientnet_b6",
        ][phi]
