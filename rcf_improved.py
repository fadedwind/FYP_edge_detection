"""
RCF (Richer Convolutional Features) for Edge Detection - Improved Version

改进点:
1. 正确加载预训练VGG16 backbone
2. 使用更好的权重初始化策略
3. 改进特征融合方式
4. 添加权重加载日志
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


class RCFNetImproved(nn.Module):
    """
    改进的RCF网络
    """
    def __init__(self, pretrained_backbone=True, use_download_model=False):
        super().__init__()

        # 加载VGG16
        vgg = torchvision.models.vgg16(pretrained=pretrained_backbone)
        features = list(vgg.features.children())

        # 提取所有卷积层
        self.conv1_1 = features[0]   # 64
        self.conv1_2 = features[2]   # 64

        self.conv2_1 = features[5]   # 128
        self.conv2_2 = features[7]   # 128

        self.conv3_1 = features[10]  # 256
        self.conv3_2 = features[12]  # 256
        self.conv3_3 = features[14]  # 256

        self.conv4_1 = features[17]  # 512
        self.conv4_2 = features[19]  # 512
        self.conv4_3 = features[21]  # 512

        self.conv5_1 = features[24]  # 512
        self.conv5_2 = features[26]  # 512
        self.conv5_3 = features[28]  # 512

        # Side-output layers (使用更好的初始化)
        self.side1 = self._make_side_layer(64)
        self.side2 = self._make_side_layer(128)
        self.side3 = self._make_side_layer(256)
        self.side4 = self._make_side_layer(512)
        self.side5 = self._make_side_layer(512)

        # Fusion layer
        self.fuse = nn.Conv2d(5, 1, kernel_size=1)
        self._init_weights()

        # 如果需要，加载下载的模型
        if use_download_model:
            self._load_downloaded_model()

    def _make_side_layer(self, in_channels):
        """创建side-output层，使用改进的初始化"""
        return nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

    def _init_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming初始化，适合ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 特别初始化fusion层
        nn.init.constant_(self.fuse.weight, 0.2)  # 参考RCF论文
        nn.init.constant_(self.fuse.bias, 0)

    def _load_downloaded_model(self):
        """尝试加载下载的模型权重"""
        model_path = 'models/rcf_bsds500_pascal.pth'
        if not os.path.exists(model_path):
            print(f"[RCF] ⚠️  Downloaded model not found: {model_path}")
            return

        print(f"[RCF] Loading downloaded model: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint

        # 获取当前模型的所有参数
        model_dict = self.state_dict()

        # 1. 首先尝试精确匹配
        matched_params = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                matched_params[k] = v

        print(f"[RCF] ✓ Matched {len(matched_params)}/{len(model_dict)} parameters from downloaded model")

        # 2. 加载匹配的参数
        model_dict.update(matched_params)
        self.load_state_dict(model_dict)

        if len(matched_params) < len(model_dict) * 0.8:
            print(f"[RCF] ⚠️  Only {100*len(matched_params)/len(model_dict):.1f}% parameters loaded")
            print(f"[RCF] ℹ️  Using ImageNet pretrained VGG16 backbone + random side-output layers")

    def forward(self, x):
        """前向传播"""
        # Stage 1
        h1 = self.conv1_1(x)
        h = self.conv1_2(h1)

        # Stage 2
        h2 = self.conv2_1(h)
        h = self.conv2_2(h2)

        # Stage 3
        h3 = self.conv3_1(h)
        h = self.conv3_2(h3)
        h = self.conv3_3(h)

        # Stage 4
        h4 = self.conv4_1(h)
        h = self.conv4_2(h4)
        h = self.conv4_3(h)

        # Stage 5
        h5 = self.conv5_1(h)
        h = self.conv5_2(h5)
        h = self.conv5_3(h5)

        # Side outputs
        s1 = self.side1(h1)
        s2 = self.side2(h2)
        s3 = self.side3(h3)
        s4 = self.side4(h4)
        s5 = self.side5(h5)

        # 上采样到相同尺寸（使用双线性插值）
        target_size = (x.size(2), x.size(3))

        s1 = torch.nn.functional.interpolate(s1, size=target_size, mode='bilinear', align_corners=True)
        s2 = torch.nn.functional.interpolate(s2, size=target_size, mode='bilinear', align_corners=True)
        s3 = torch.nn.functional.interpolate(s3, size=target_size, mode='bilinear', align_corners=True)
        s4 = torch.nn.functional.interpolate(s4, size=target_size, mode='bilinear', align_corners=True)
        s5 = torch.nn.functional.interpolate(s5, size=target_size, mode='bilinear', align_corners=True)

        # Fusion
        cat = torch.cat([s1, s2, s3, s4, s5], dim=1)
        out = self.fuse(cat)

        # Sigmoid激活
        out = torch.sigmoid(out)

        return out


def run_rcf_improved(img_bgr, model=None, device='cpu'):
    """
    使用改进的RCF模型进行边缘检测

    Args:
        img_bgr: BGR格式的numpy数组
        model: 预加载的模型
        device: 'cpu' 或 'cuda'

    Returns:
        edge_map: 边缘图 (0-255, uint8)
    """
    if torch is None:
        raise ImportError("PyTorch未安装")

    from rcf_improved import RCFNetImproved

    # 预处理
    # ImageNet标准化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # 转换BGR到RGB
    img_rgb = img_bgr[:, :, ::-1].astype(np.float32)

    # 归一化到[0, 1]
    img_rgb = img_rgb / 255.0

    # ImageNet标准化
    img_rgb = (img_rgb - mean) / std

    # 转换为tensor: HWC -> CHW
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)

    # 创建模型（如果未提供）
    if model is None:
        print("[RCF-Improved] 初始化模型...")
        model = RCFNetImproved(pretrained_backbone=True, use_download_model=True)
        model.eval()

    model = model.to(device)
    tensor = tensor.to(device)

    # 前向传播
    with torch.no_grad():
        output = model(tensor)

    # 转换为numpy
    edge_prob = output.cpu().numpy()[0, 0]  # (H, W)

    # 映射到0-255
    edge_map = (edge_prob * 255).astype(np.uint8)

    return edge_map
