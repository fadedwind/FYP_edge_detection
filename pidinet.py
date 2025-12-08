"""
PiDiNet (Pixel Difference Networks) 实现
基于官方实现：https://github.com/hellozhuo/pidinet

参考官方实现修复架构，确保能正确加载预训练权重
"""
from typing import Optional
import os
import numpy as np
import math
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
except Exception:
    torch = None


# ============================================================================
# PDC (Pixel Difference Convolution) 操作
# ============================================================================

def createConvFunc(op_type):
    """创建PDC卷积函数"""
    assert op_type in ['cv', 'cd', 'ad', 'rd'], 'unknown op type: %s' % str(op_type)
    if op_type == 'cv':
        return F.conv2d

    if op_type == 'cd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'
            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc
        return func
    elif op_type == 'ad':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'
            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)  # clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    elif op_type == 'rd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation
            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
            buffer[:, :, 12] = 0
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    else:
        return None


class Conv2d(nn.Module):
    """PDC卷积层"""
    def __init__(self, pdc, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.pdc = pdc

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return self.pdc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


# ============================================================================
# 辅助模块
# ============================================================================

class CSAM(nn.Module):
    """Compact Spatial Attention Module"""
    def __init__(self, channels):
        super(CSAM, self).__init__()
        mid_channels = 4
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return x * y


class CDCM(nn.Module):
    """Compact Dilation Convolution based Module"""
    def __init__(self, in_channels, out_channels):
        super(CDCM, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
        nn.init.constant_(self.conv1.bias, 0)
        
    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        return x1 + x2 + x3 + x4


class MapReduce(nn.Module):
    """Reduce feature maps into a single edge map"""
    def __init__(self, channels):
        super(MapReduce, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


# ============================================================================
# PDC Block
# ============================================================================

class PDCBlock(nn.Module):
    """PDC Block - 根据官方实现"""
    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock, self).__init__()
        self.stride = stride
            
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        self.conv1 = Conv2d(pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y


# ============================================================================
# PiDiNet 主模型
# ============================================================================

class PiDiNet(nn.Module):
    """PiDiNet官方架构实现，匹配预训练权重"""
    def __init__(self, inplane=20, pdcs=None, dil=None, sa=False, rd=False):
        super(PiDiNet, self).__init__()
        self.sa = sa
        if dil is not None:
            assert isinstance(dil, int), 'dil should be an int'
        self.dil = dil

        self.fuseplanes = []

        self.inplane = inplane
        
        # 默认使用carv4配置（cd, ad, rd, cv的组合）
        if pdcs is None:
            # carv4配置：cd, ad, rd, cv的循环
            pdc_types = ['cd', 'ad', 'rd', 'cv']
            pdcs = [createConvFunc(pdc_types[i % 4]) for i in range(16)]
        
        # Init block: 使用第一个PDC类型
        self.init_block = Conv2d(pdcs[0], 3, self.inplane, kernel_size=3, padding=1)

        # Block 1: 3个块，保持inplane通道
        self.block1_1 = PDCBlock(pdcs[1], self.inplane, self.inplane)
        self.block1_2 = PDCBlock(pdcs[2], self.inplane, self.inplane)
        self.block1_3 = PDCBlock(pdcs[3], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # C

        # Block 2: 4个块，通道翻倍到inplane*2
        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block2_1 = PDCBlock(pdcs[4], inplane, self.inplane, stride=2)
        self.block2_2 = PDCBlock(pdcs[5], self.inplane, self.inplane)
        self.block2_3 = PDCBlock(pdcs[6], self.inplane, self.inplane)
        self.block2_4 = PDCBlock(pdcs[7], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 2C
        
        # Block 3: 4个块，通道翻倍到inplane*4
        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block3_1 = PDCBlock(pdcs[8], inplane, self.inplane, stride=2)
        self.block3_2 = PDCBlock(pdcs[9], self.inplane, self.inplane)
        self.block3_3 = PDCBlock(pdcs[10], self.inplane, self.inplane)
        self.block3_4 = PDCBlock(pdcs[11], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 4C

        # Block 4: 4个块，tiny版本保持inplane*4通道（不翻倍）
        self.block4_1 = PDCBlock(pdcs[12], self.inplane, self.inplane, stride=2)
        self.block4_2 = PDCBlock(pdcs[13], self.inplane, self.inplane)
        self.block4_3 = PDCBlock(pdcs[14], self.inplane, self.inplane)
        self.block4_4 = PDCBlock(pdcs[15], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 4C

        # conv_reduces: 必需的模块，将每个block的输出转换为1通道
        self.conv_reduces = nn.ModuleList()
        if self.sa and self.dil is not None:
            self.attentions = nn.ModuleList()
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.attentions.append(CSAM(self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        elif self.sa:
            self.attentions = nn.ModuleList()
            for i in range(4):
                self.attentions.append(CSAM(self.fuseplanes[i]))
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))
        elif self.dil is not None:
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        else:
            for i in range(4):
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))

        # classifier: 融合4个side output
        self.classifier = nn.Conv2d(4, 1, kernel_size=1)  # has bias
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        H, W = x.size()[2:]

        x = self.init_block(x)

        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2 = self.block2_4(x2)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)
        x4 = self.block4_4(x4)

        x_fuses = []
        if self.sa and self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](self.dilations[i](xi)))
        elif self.sa:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](xi))
        elif self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.dilations[i](xi))
        else:
            x_fuses = [x1, x2, x3, x4]

        e1 = self.conv_reduces[0](x_fuses[0])
        e1 = F.interpolate(e1, (H, W), mode="bilinear", align_corners=False)

        e2 = self.conv_reduces[1](x_fuses[1])
        e2 = F.interpolate(e2, (H, W), mode="bilinear", align_corners=False)

        e3 = self.conv_reduces[2](x_fuses[2])
        e3 = F.interpolate(e3, (H, W), mode="bilinear", align_corners=False)

        e4 = self.conv_reduces[3](x_fuses[3])
        e4 = F.interpolate(e4, (H, W), mode="bilinear", align_corners=False)

        outputs = [e1, e2, e3, e4]

        output = self.classifier(torch.cat(outputs, dim=1))
        outputs.append(output)
        outputs = [torch.sigmoid(r) for r in outputs]
        # 返回最终的融合输出（第5个，索引4）
        return outputs[-1]


_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'pidinet_bsds.pth')

# 支持的模型文件名列表（按优先级排序，优先选择轻量级模型以加快速度）
_MODEL_CANDIDATES = [
    # 轻量级模型（优先选择，速度快）
    'table5_pidinet-tiny.pth',  # 微型版本（最快）
    'table5_pidinet-tiny-l.pth',  # 微型-大型版本
    'table5_pidinet-small.pth',  # 小型版本
    'table5_pidinet-small-l.pth',  # 小型-大型版本
    # 标准模型（平衡速度和精度）
    'pidinet_bsds.pth',  # BSDS500训练的标准模型
    'table5_pidinet.pth',  # 标准 PiDiNet
    # 大型模型（精度高但速度慢）
    'table5_pidinet-l.pth',  # 大型版本
    # 其他变体
    'table6_pidinet.pth',
    'table7_pidinet.pth',
    'table5_baseline.pth'
]

# 全局模型缓存
_cached_pidinet_model = None
_cached_pidinet_path = None


def find_pidinet_model(prefer_fast=True):
    """查找可用的 PiDiNet 模型文件"""
    repo_dir = os.path.dirname(__file__)
    models_dir = os.path.join(repo_dir, 'models')
    
    if prefer_fast:
        fast_models = [
            'table5_pidinet-tiny.pth',
            'table5_pidinet-tiny-l.pth',
            'table5_pidinet-small.pth',
            'table5_pidinet-small-l.pth'
        ]
        for model_name in fast_models:
            model_path = os.path.join(models_dir, model_name)
            if os.path.exists(model_path):
                print(f"找到轻量级PiDiNet模型: {model_name}")
                return model_path
    
    for model_name in _MODEL_CANDIDATES:
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            return model_path
    return None


def load_pidinet_model(weights_path: Optional[str] = None, device: Optional[str] = None, prefer_fast: bool = True):
    """Load PiDiNet and weights"""
    global _cached_pidinet_model, _cached_pidinet_path
    
    if torch is None:
        raise RuntimeError('torch/torchvision not available. Please install torch and torchvision to use PiDiNet.')
    if device is None:
        device = 'cpu'
    
    if weights_path is None:
        weights_path = find_pidinet_model(prefer_fast=prefer_fast)
        if weights_path is None:
            weights_path = _DEFAULT_MODEL_PATH
    
    # 检查缓存
    if _cached_pidinet_model is not None and _cached_pidinet_path == weights_path:
        return _cached_pidinet_model
    
    # 根据模型文件名确定参数
    model_name = os.path.basename(weights_path)
    if 'tiny' in model_name.lower():
        inplane = 20
        dil = 8  # tiny版本使用dil=8
        sa = True  # 通常tiny版本使用sa
    elif 'small' in model_name.lower():
        inplane = 30
        dil = 12
        sa = True
    else:
        inplane = 60
        dil = 24
        sa = True
    
    # 检查模型文件是否包含sa和dil信息
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # 检查是否有attentions和dilations
            has_attentions = any('attentions' in k for k in state_dict.keys())
            has_dilations = any('dilations' in k for k in state_dict.keys())
            if not has_attentions:
                sa = False
            if not has_dilations:
                dil = None
    except:
        pass
    
    # 创建模型（使用carv4配置）
    pdc_types = ['cd', 'ad', 'rd', 'cv']
    pdcs = [createConvFunc(pdc_types[i % 4]) for i in range(16)]
    model = PiDiNet(inplane=inplane, pdcs=pdcs, dil=dil, sa=sa)
    
    if os.path.exists(weights_path):
        try:
            print(f"Loading PiDiNet model from: {weights_path}")
            state = torch.load(weights_path, map_location=device)
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            elif isinstance(state, dict) and 'model' in state:
                state = state['model']
            
            # 移除module.前缀
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
                    print(f"Successfully loaded PiDiNet model from {weights_path}")
                    print(f"Loaded {loaded_count}/{total_count} parameters ({loaded_ratio*100:.1f}%)")
                elif loaded_ratio >= 0.5:
                    print(f"Partially loaded PiDiNet model from {weights_path}")
                    print(f"Loaded {loaded_count}/{total_count} parameters ({loaded_ratio*100:.1f}%)")
                    print("⚠️  WARNING: Some weights may not match. Performance may be slightly degraded.")
                else:
                    print(f"⚠️  WARNING: Only loaded {loaded_count}/{total_count} parameters ({loaded_ratio*100:.1f}%)")
                    print("❌ CRITICAL: Model architecture may not match!")
                    if missing_keys:
                        print(f"Missing {len(missing_keys)} keys. Sample: {list(missing_keys)[:5]}")
            except Exception as e:
                print(f"ERROR: Failed to load PiDiNet weights: {e}")
                print("CRITICAL: Using randomly initialized model - results will be inaccurate!")
        except Exception as e:
            print(f"Warning: Failed to load PiDiNet weights from {weights_path}: {e}")
            print("Using randomly initialized model (results may not be accurate).")
    else:
        print(f"PiDiNet weights not found at {weights_path}.")
        print("Please download a pretrained PiDiNet weight file and place it at the path above.")
        print("You can download from: https://github.com/hellozhuo/pidinet")
        print("Using randomly initialized model (results may not be accurate).")
    
    model.eval()
    model.to(device)
    
    _cached_pidinet_model = model
    _cached_pidinet_path = weights_path
    
    return model


def run_pidinet(np_bgr_image: np.ndarray, model: Optional[PiDiNet] = None, device: Optional[str] = None, prefer_fast: bool = True):
    """Run PiDiNet inference on a single BGR image"""
    if torch is None:
        raise RuntimeError('torch not installed. Install torch to run PiDiNet inference.')
    if device is None:
        device = 'cpu'
    
    img = np_bgr_image.copy()
    # Convert BGR -> RGB, to float32
    img = img[:, :, ::-1]
    img = img.astype(np.float32) / 255.0
    # PiDiNet normalizes by ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    
    if model is None:
        model = load_pidinet_model(device=device, prefer_fast=prefer_fast)
    
    with torch.no_grad():
        out = model(tensor)
    # out: 1x1xHxW (sigmoid probabilities)
    out_np = out.squeeze().cpu().numpy()
    # 确保输出是2D数组（H, W）
    if len(out_np.shape) > 2:
        out_np = out_np.squeeze()
    if len(out_np.shape) != 2:
        if out_np.shape[0] == 1:
            out_np = out_np[0]
        elif len(out_np.shape) == 3:
            out_np = out_np[0] if out_np.shape[0] == 1 else out_np[:, :, 0]
    # Convert to 0-255 uint8
    out_np = (out_np * 255.0).clip(0, 255).astype(np.uint8)
    return out_np


if __name__ == '__main__':
    print('pidinet.py: module loaded. Use load_pidinet_model() and run_pidinet(image_np).')
