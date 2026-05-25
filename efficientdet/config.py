class EfficientDetConfig:
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
