"""
Lightweight HED (Holistically-Nested Edge Detection) inference helper.

This module provides a minimal HED-style network built on top of VGG16 features
and a helper `run_hed` function that returns a single-channel edge probability map
in range [0,255].

Notes:
- This file requires `torch` and `torchvision` for inference. Place a pretrained
  HED weight file at `models/hed_bsds.pth`. If missing, the module will print
  instructions and can optionally try to download if an URL is provided.
- The implementation focuses on inference simplicity; training/strict parity with
  original HED paper is out of scope here.
"""
from typing import Optional
import os
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision import transforms
except Exception:
    torch = None


class HEDNet(nn.Module):
    """Simple HED-style network using VGG16 features.
    Produces 5 side-outputs which are fused to form final edge map.
    This is a compact inference-oriented re-implementation.
    """
    def __init__(self, pretrained_backbone=True):
        super().__init__()
        vgg = torchvision.models.vgg16(pretrained=pretrained_backbone)
        features = list(vgg.features.children())

        # split layers into stages roughly corresponding to conv1..conv5
        # (indices chosen to match typical VGG16 layout)
        self.stage1 = nn.Sequential(*features[0:5])   # conv1_1..conv1_2 + relu
        self.stage2 = nn.Sequential(*features[5:10])  # conv2
        self.stage3 = nn.Sequential(*features[10:17]) # conv3
        self.stage4 = nn.Sequential(*features[17:24]) # conv4
        self.stage5 = nn.Sequential(*features[24:31]) # conv5

        # side-output convs: 1x1 conv to single channel
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(128, 1, kernel_size=1)
        self.side3 = nn.Conv2d(256, 1, kernel_size=1)
        self.side4 = nn.Conv2d(512, 1, kernel_size=1)
        self.side5 = nn.Conv2d(512, 1, kernel_size=1)

        # fuse
        self.fuse = nn.Conv2d(5, 1, kernel_size=1)

        # initialize side convs
        for m in [self.side1, self.side2, self.side3, self.side4, self.side5, self.fuse]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)

        o1 = self.side1(s1)
        o2 = self.side2(s2)
        o3 = self.side3(s3)
        o4 = self.side4(s4)
        o5 = self.side5(s5)

        # upsample to input size
        o1u = nn.functional.interpolate(o1, size=(h, w), mode='bilinear', align_corners=False)
        o2u = nn.functional.interpolate(o2, size=(h, w), mode='bilinear', align_corners=False)
        o3u = nn.functional.interpolate(o3, size=(h, w), mode='bilinear', align_corners=False)
        o4u = nn.functional.interpolate(o4, size=(h, w), mode='bilinear', align_corners=False)
        o5u = nn.functional.interpolate(o5, size=(h, w), mode='bilinear', align_corners=False)

        # concatenate and fuse
        cat = torch.cat([o1u, o2u, o3u, o4u, o5u], dim=1)
        fused = self.fuse(cat)

        # apply sigmoid to each side and fused
        out = torch.sigmoid(fused)
        return out


_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'hed_bsds.pth')


def load_hed_model(weights_path: Optional[str] = None, device: Optional[str] = None):
    """Load HEDNet and weights. If weights_path is None, will try default path.
    Returns the model on the specified device (cpu by default).
    """
    if torch is None:
        raise RuntimeError('torch/torchvision not available. Please install torch and torchvision to use HED.')
    if weights_path is None:
        weights_path = _DEFAULT_MODEL_PATH
    if device is None:
        device = 'cpu'

    model = HEDNet(pretrained_backbone=False)
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        # allow either full state_dict or model state inside dict
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        try:
            model.load_state_dict(state)
        except Exception:
            # try mapping keys without module.
            new_state = {}
            for k, v in state.items():
                nk = k.replace('module.', '')
                new_state[nk] = v
            model.load_state_dict(new_state)
    else:
        print(f"HED weights not found at {weights_path}.")
        print("Please download a pretrained HED weight file (e.g., trained on BSDS) and place it at the path above.")
        print("See README or HED project releases for pretrained weights.")

    model.eval()
    model.to(device)
    return model


def run_hed(np_bgr_image: np.ndarray, model: Optional[HEDNet] = None, device: Optional[str] = None):
    """Run HED inference on a single BGR image (numpy array). Returns uint8 single-channel map [0,255].

    If no model is provided, tries to load default model from models/hed_bsds.pth (may raise).
    """
    if torch is None:
        raise RuntimeError('torch not installed. Install torch to run HED inference.')
    if device is None:
        device = 'cpu'

    img = np_bgr_image.copy()
    # convert BGR -> RGB, to float32
    img = img[:, :, ::-1]
    img = img.astype(np.float32) / 255.0
    # HED often normalizes by ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    if model is None:
        model = load_hed_model(device=device)

    with torch.no_grad():
        out = model(tensor)
    # out: 1x1xHxW (sigmoid probabilities)
    out_np = out.squeeze().cpu().numpy()
    # convert to 0-255 uint8
    out_np = (out_np * 255.0).clip(0, 255).astype(np.uint8)
    return out_np


if __name__ == '__main__':
    # quick smoke test when run directly (no external image provided)
    print('hed.py: module loaded. Use load_hed_model() and run_hed(image_np).')
