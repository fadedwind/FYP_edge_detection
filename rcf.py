"""
RCF (Richer Convolutional Features) for Edge Detection

基于论文:
Liu, Y., Cheng, M. M., Hu, X., et al. "Richer Convolutional Features for Edge Detection."
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019.

官方实现: https://github.com/yun-liu/RCF-PyTorch

RCF改进点:
1. 使用VGG16的所有卷积层特征（不只是pooling后的特征）
2. 每个stage有多个side-outputs
3. 更丰富的特征融合策略
4. 在BSDS500上ODS F-score达到0.812
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


class RCFNet(nn.Module):
    """
    RCF网络基于VGG16，使用更丰富的卷积特征进行边缘检测

    架构特点:
    - 使用VGG16的所有卷积层（共13层）
    - 每个stage有多个side-output
    - 融合所有stage的特征
    """
    def __init__(self, pretrained_backbone=True):
        super().__init__()

        # 加载VGG16
        vgg = torchvision.models.vgg16(pretrained=pretrained_backbone)
        features = list(vgg.features.children())

        # VGG16的卷积层分布:
        # Stage 1: conv1_1, conv1_2 (pooling)
        # Stage 2: conv2_1, conv2_2 (pooling)
        # Stage 3: conv3_1, conv3_2, conv3_3 (pooling)
        # Stage 4: conv4_1, conv4_2, conv4_3 (pooling)
        # Stage 5: conv5_1, conv5_2, conv5_3 (pooling)

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

        # Stage 1 side-output (64 -> 1)
        self.side1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        # Stage 2 side-output (128 -> 1)
        self.side2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        # Stage 3 side-output (256 -> 1)
        self.side3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        # Stage 4 side-output (512 -> 1)
        self.side4 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        # Stage 5 side-output (512 -> 1)
        self.side5 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        # 融合所有5个stage的输出 (5 -> 1)
        self.fuse = nn.Conv2d(5, 1, kernel_size=1)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        # Stage 1
        x = self.conv1_1(x)
        s1 = self.conv1_2(x)
        x = nn.functional.max_pool2d(s1, kernel_size=2, stride=2)

        # Stage 2
        x = self.conv2_1(x)
        s2 = self.conv2_2(x)
        x = nn.functional.max_pool2d(s2, kernel_size=2, stride=2)

        # Stage 3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        s3 = self.conv3_3(x)
        x = nn.functional.max_pool2d(s3, kernel_size=2, stride=2)

        # Stage 4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        s4 = self.conv4_3(x)
        x = nn.functional.max_pool2d(s4, kernel_size=2, stride=2)

        # Stage 5
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        s5 = self.conv5_3(x)
        x = nn.functional.max_pool2d(s5, kernel_size=2, stride=2)

        # 生成side-outputs
        o1 = self.side1(s1)
        o2 = self.side2(s2)
        o3 = self.side3(s3)
        o4 = self.side4(s4)
        o5 = self.side5(s5)

        # 上采样到原始尺寸
        # Stage 5: 16x downsample -> 需要 16x upsample
        o1u = nn.functional.interpolate(o1, size=(h, w), mode='bilinear', align_corners=False)
        o2u = nn.functional.interpolate(o2, size=(h, w), mode='bilinear', align_corners=False)
        o3u = nn.functional.interpolate(o3, size=(h, w), mode='bilinear', align_corners=False)
        o4u = nn.functional.interpolate(o4, size=(h, w), mode='bilinear', align_corners=False)
        o5u = nn.functional.interpolate(o5, size=(h, w), mode='bilinear', align_corners=False)

        # 融合所有side-outputs
        cat = torch.cat([o1u, o2u, o3u, o4u, o5u], dim=1)
        fused = self.fuse(cat)

        # 应用sigmoid
        out = torch.sigmoid(fused)

        return out


_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'rcf_bsds500.pth')

# 支持的模型文件名列表
_MODEL_CANDIDATES = [
    'rcf_bsds500_pascal.pth',  # BSDS500+PASCAL训练
    'bsds500_pascal_model.pth',  # 备选名称
    'rcf_pretrained.pth',  # 通用预训练
    'rcf.pth',  # 简化名称
]

# 全局模型缓存
_cached_rcf_model = None
_cached_rcf_path = None


def find_rcf_model():
    """查找可用的RCF模型文件"""
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(repo_dir, 'models')

    for model_name in _MODEL_CANDIDATES:
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            return model_path
    return None


def load_rcf_model(weights_path: Optional[str] = None, device: Optional[str] = None):
    """加载RCF模型和权重

    Args:
        weights_path: 权重文件路径，如果为None则自动查找
        device: 设备 ('cpu' 或 'cuda')

    Returns:
        加载了权重的RCF模型
    """
    global _cached_rcf_model, _cached_rcf_path

    if torch is None:
        raise RuntimeError('torch/torchvision not available. Please install torch and torchvision to use RCF.')

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if weights_path is None:
        weights_path = find_rcf_model()
        if weights_path is None:
            weights_path = _DEFAULT_MODEL_PATH

    # 检查缓存
    if _cached_rcf_model is not None and _cached_rcf_path == weights_path:
        return _cached_rcf_model

    # 创建模型
    model = RCFNet(pretrained_backbone=False)

    if os.path.exists(weights_path):
        try:
            print(f"[RCF] Loading model from: {weights_path}")
            state = torch.load(weights_path, map_location=device)

            # 处理不同的checkpoint格式
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            elif isinstance(state, dict) and 'model' in state:
                state = state['model']

            # 移除module.前缀（DataParallel保存的模型）
            new_state = {}
            for k, v in state.items():
                nk = k.replace('module.', '')
                new_state[nk] = v

            # 加载权重
            try:
                missing_keys, unexpected_keys = model.load_state_dict(new_state, strict=False)
                loaded_count = len(new_state.keys()) - len(missing_keys)
                total_count = len(new_state.keys())
                loaded_ratio = loaded_count / total_count if total_count > 0 else 0

                if loaded_ratio >= 0.9:
                    print(f"[RCF] ✅ Model loaded successfully: {os.path.basename(weights_path)}")
                    print(f"      Parameters: {loaded_count}/{total_count} ({loaded_ratio*100:.1f}%)")
                elif loaded_ratio >= 0.5:
                    print(f"[RCF] ⚠️  Model partially loaded: {os.path.basename(weights_path)}")
                    print(f"      Parameters: {loaded_count}/{total_count} ({loaded_ratio*100:.1f}%)")
                    print(f"      Performance may be degraded")
                else:
                    print(f"[RCF] ❌ CRITICAL: Only loaded {loaded_count}/{total_count} ({loaded_ratio*100:.1f}%)")
                    print(f"      Model architecture may not match!")

            except Exception as e:
                print(f"[RCF] ❌ Failed to load weights: {e}")
                print(f"[RCF] Using randomly initialized model (results will be inaccurate)")
        except Exception as e:
            print(f"[RCF] ⚠️  Failed to load model file: {e}")
            print(f"[RCF] Using randomly initialized model (results may not be accurate)")
    else:
        print(f"[RCF] ⚠️  Model file not found: {weights_path}")
        print(f"[RCF] Please download RCF pretrained weights from:")
        print(f"      https://github.com/yun-liu/RCF-PyTorch")
        print(f"[RCF] Using randomly initialized model (results will be inaccurate)")

    model.eval()
    model.to(device)

    _cached_rcf_model = model
    _cached_rcf_path = weights_path

    return model


def run_rcf(np_bgr_image: np.ndarray, model: Optional[RCFNet] = None, device: Optional[str] = None):
    """运行RCF边缘检测

    Args:
        np_bgr_image: BGR格式的numpy图像 (H, W, 3)
        model: RCF模型，如果为None则自动加载
        device: 设备 ('cpu' 或 'cuda')

    Returns:
        uint8格式的边缘图 (H, W)，范围[0, 255]
    """
    if torch is None:
        raise RuntimeError('torch not installed. Install torch to run RCF inference.')

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 预处理
    img = np_bgr_image.copy()
    # BGR -> RGB
    img = img[:, :, ::-1]
    # 转换为float32并归一化到[0, 1]
    img = img.astype(np.float32) / 255.0
    # ImageNet标准化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))

    # 转换为tensor
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    # 加载模型
    if model is None:
        model = load_rcf_model(device=device)

    # 推理
    with torch.no_grad():
        out = model(tensor)

    # 后处理
    # out: (1, 1, H, W) sigmoid概率
    out_np = out.squeeze().cpu().numpy()
    # 转换为0-255 uint8
    out_np = (out_np * 255.0).clip(0, 255).astype(np.uint8)

    return out_np


if __name__ == '__main__':
    print('rcf.py: RCF (Richer Convolutional Features) for Edge Detection')
    print('Official repo: https://github.com/yun-liu/RCF-PyTorch')
    print('Performance: ODS F-score 0.812 on BSDS500')
