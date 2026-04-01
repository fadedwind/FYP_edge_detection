# 测试文件目录

本目录包含项目的所有测试脚本和验证工具。

---

## 📋 测试文件列表

### 模型检查脚本

| 文件 | 说明 | 用途 |
|------|------|------|
| `check_hed_model.py` | HED模型检查 | 验证HED模型加载是否正常 |
| `check_rcf_model.py` | RCF模型检查 | 检查RCF模型文件完整性 |
| `check_rcf_weights.py` | RCF权重检查 | 分析RCF权重匹配情况 |

### 算法测试脚本

| 文件 | 说明 | 状态 |
|------|------|------|
| `test_algorithms.py` | 算法测试（旧版） | ⚠️ 已过时 |
| `test_algorithms_fixed.py` | 算法测试（修复版） | ✅ 推荐使用 |
| `test_hed_only.py` | HED专项测试 | ✅ HED优化后 |
| `test_rcf.py` | RCF测试 | ⚠️ 模型不完整 |
| `test_rcf_improved.py` | RCF改进版测试 | ⚠️ 性能未达预期 |

### 集成测试脚本

| 文件 | 说明 | 状态 |
|------|------|------|
| `test_yolo_integration.py` | YOLOv8集成测试 | ✅ 可用 |
| `test_smart_removal.py` | 智能抠图测试 | ✅ 可用 |

---

## 🚀 使用方法

### 1. 算法性能测试

```bash
# 测试所有算法（修复版）
python test_algorithms_fixed.py

# 测试HED算法
python test_hed_only.py

# 测试RCF算法
python test_rcf.py
```

### 2. 集成功能测试

```bash
# 测试YOLOv8车辆检测
python test_yolo_integration.py test_image.jpg

# 测试智能抠图
python test_smart_removal.py photo.jpg
python test_smart_removal.py photo.jpg rembg  # 指定方法
```

### 3. 模型检查

```bash
# 检查HED模型
python check_hed_model.py

# 检查RCF模型
python check_rcf_model.py

# 分析RCF权重
python check_rcf_weights.py
```

---

## 📊 测试结果说明

### 算法性能测试结果

参考报告：`../docs/reports/FINAL_ALGORITHM_COMPARISON.md`

**性能排名**（BSDS500数据集）：
1. 🥇 PiDiNet - F1: 0.7138
2. 🥈 Canny - F1: 0.5137
3. 🥉 RCF - F1: 0.3970（模型不完整）
4. HED - F1: 0.3530

### 集成功能测试

- ✅ YOLOv8车辆检测 - 正常工作
- ✅ 智能抠图（rembg） - 效果最好
- ✅ 智能抠图（边缘辅助） - 使用PiDiNet
- ⚠️ RCF边缘检测 - 模型不完整，性能受限

---

## ⚠️ 注意事项

### 测试前提条件

1. **后端API必须运行**
```bash
python edge_detection_api.py
```

2. **依赖库已安装**
```bash
pip install -r ../requirements.txt
```

3. **模型文件已下载**
- HED: `models/hed_*.caffemodel`
- RCF: `models/rcf_bsds500_pascal.pth`
- PiDiNet: 自动下载
- YOLOv8: 自动下载

### 常见问题

**Q1: 测试脚本报错找不到模块**
```bash
A: 确保在项目根目录运行测试，或设置PYTHONPATH
   export PYTHONPATH=..
```

**Q2: YOLOv8测试失败**
```bash
A: 安装ultralytics库
   pip install ultralytics
```

**Q3: 智能抠图rembg方法失败**
```bash
A: 安装rembg库
   pip install rembg
```

---

## 📝 开发建议

### 添加新测试

1. 命名规范：`test_<功能名>.py`
2. 添加说明注释
3. 包含错误处理
4. 输出清晰的测试结果

### 测试模板

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
<测试名称>

用途：<测试目的>
依赖：<需要的库和模型>
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("="*70)
    print(" <测试名称>")
    print("="*70)

    # 测试代码
    pass

if __name__ == '__main__':
    main()
```

---

## 📚 相关文档

- [算法对比报告](../docs/reports/FINAL_ALGORITHM_COMPARISON.md)
- [RCF状态总结](../docs/reports/RCF_Status_Summary.md)
- [YOLOv8集成指南](../docs/YOLOv8_Integration_Guide.md)
- [智能抠图指南](../docs/Smart_Background_Removal_Guide.md)

---

**最后更新**: 2026-03-29
