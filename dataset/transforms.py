"""
Joint image + bounding-box transforms for detection.

All transforms accept and return:
    image  : PIL.Image (RGB)
    boxes  : FloatTensor[M, 4]  (x1, y1, x2, y2) absolute pixels
    labels : LongTensor[M]

`ToTensor` is always last; it converts the PIL image to a float tensor
and applies ImageNet normalisation.
"""
import random
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes, labels):
        for t in self.transforms:
            image, boxes, labels = t(image, boxes, labels)
        return image, boxes, labels


class Resize:
    """Scale-and-pad to a square canvas of `size` × `size` pixels.

    Preserves aspect ratio by scaling so the longer side equals `size`,
    then zero-padding the shorter side symmetrically on both sides.
    Boxes are scaled and shifted accordingly.
    """
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image, boxes, labels):
        orig_w, orig_h = image.size
        scale = self.size / max(orig_w, orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        image = TF.resize(image, [new_h, new_w])

        # Pad so the canvas is exactly size × size
        pad_left   = (self.size - new_w) // 2
        pad_top    = (self.size - new_h) // 2
        pad_right  = self.size - new_w - pad_left
        pad_bottom = self.size - new_h - pad_top
        image = TF.pad(image, [pad_left, pad_top, pad_right, pad_bottom])

        if boxes.numel() > 0:
            boxes = boxes * scale
            boxes[:, 0] += pad_left
            boxes[:, 1] += pad_top
            boxes[:, 2] += pad_left
            boxes[:, 3] += pad_top
            boxes = boxes.clamp(min=0, max=self.size)

        return image, boxes, labels


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, boxes, labels):
        if random.random() < self.p:
            w = image.width
            image = TF.hflip(image)
            if boxes.numel() > 0:
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = w - x2
                boxes[:, 2] = w - x1
        return image, boxes, labels


class ColorJitter:
    """Colour augmentation (image only — boxes and labels pass through)."""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self._jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, boxes, labels):
        return self._jitter(image), boxes, labels


class ToTensor:
    """PIL → float32 tensor + ImageNet normalisation."""
    _mean = [0.485, 0.456, 0.406]
    _std  = [0.229, 0.224, 0.225]

    def __call__(self, image, boxes, labels):
        image = TF.to_tensor(image)
        image = TF.normalize(image, self._mean, self._std)
        return image, boxes, labels
