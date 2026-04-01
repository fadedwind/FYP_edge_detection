# RCF (Richer Convolutional Features) 集成说明

## ✅ 已完成的工作

1. **创建RCF模型实现** (`rcf.py`)
   - 基于VGG16的RCF网络架构
   - 5个stage，每个stage有side-output
   - 融合所有特征的最终输出
   - 支持自动查找和加载预训练权重

2. **集成到边缘检测API** (`edge_detection_api.py`)
   - 添加了RCF算法的处理逻辑
   - 支持自动图像resize（最大1024x1024）
   - 使用Otsu自适应阈值

3. **更新前端界面** (`DeepLearningEdgeDetection.vue`)
   - 在算法下拉菜单中添加了"RCF"选项
   - 用户可以直接选择RCF进行边缘检测

4. **创建测试脚本** (`test_rcf.py`)
   - 专门的RCF性能测试脚本
   - 与Canny、HED、PiDiNet对比
   - 生成详细的评估报告

---

## 📊 RCF算法信息

### 论文
**Title**: Richer Convolutional Features for Edge Detection
**Authors**: Yun Liu, Ming-Ming Cheng, Xiaowei Hu, et al.
**Journal**: IEEE Transactions on Pattern Analysis and Machine Intelligence (2019)
**CVF**: [Open Access](https://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_Richer_Convolutional_Features_CVPR_2017_paper.pdf)

### 性能（BSDS500数据集）
- **ODS F-score**: 0.812
- **OIS F-score**: 0.830
- **vs Canny**: +58% improvement
- **vs HED**: +4% improvement

### 官方实现
- **PyTorch**: [yun-liu/RCF-PyTorch](https://github.com/yun-liu/RCF-PyTorch)
- **Caffe**: [yun-liu/RCF](https://github.com/yun-liu/RCF)
- **Jittor**: [yun-liu/RCF-Jittor](https://github.com/yun-liu/RCF-Jittor)

---

## ⚠️ 重要：需要下载预训练模型

### 当前状态
- ✅ RCF代码已实现并集成
- ❌ **预训练模型文件缺失**
- 如果没有预训练模型，RCF会使用随机初始化的权重，**性能会很差**

### 如何获取预训练模型

#### 方法1：从官方Google Drive下载（推荐）
1. 访问 [RCF-PyTorch GitHub](https://github.com/yun-liu/RCF-PyTorch)
2. 在README中找到预训练模型下载链接（Google Drive或百度云）
3. 下载 `bsds500_pascal_model.pth` 或 `rcf_bsds500_pascal.pth`
4. 将模型文件放到 `models/` 目录

#### 方法2：使用备用链接
如果官方链接无法访问，可以尝试：
- 搜索 "RCF pretrained model download"
- 查找其他开源实现的权重文件

### 模型文件命名
系统会按以下优先级查找模型：
1. `rcf_bsds500_pascal.pth` (最优)
2. `bsds500_pascal_model.pth` (官方名称)
3. `rcf_pretrained.pth` (通用)
4. `rcf.pth` (简化)

---

## 🧪 测试RCF性能

### 运行测试脚本
```bash
cd "F:\Coding\MUST\AAA--G4444444444444\FYP\FYP_edge_detection_git - v1"
python test_rcf.py
```

### 预期结果
如果有正确的预训练模型，RCF应该能达到：
- **F1 Score**: ~0.78-0.81
- **vs Canny**: +52-58% improvement
- **vs PiDiNet**: +9-14% improvement

### 如果模型缺失
- **F1 Score**: 可能低于0.2（随机权重）
- **错误信息**:
  ```
  [RCF] ⚠️  Model file not found: models/rcf_bsds500.pth
  [RCF] Using randomly initialized model (results will be inaccurate)
  ```

---

## 📊 算法对比表

| 算法 | ODS F1 | OIS F1 | vs Canny | 实现状态 | 预训练模型 |
|------|---------|---------|----------|----------|-----------|
| **Canny** | 0.60 | 0.64 | 基线 | ✅ 完整 | 不需要 |
| **HED** | 0.78 | 0.80 | +30% | ✅ 完整 | ✅ 有 |
| **PiDiNet** | 0.81 | 0.83 | +35% | ✅ 完整 | ✅ 有 |
| **RCF** | **0.81** | **0.83** | **+35%** | ✅ 完整 | ⚠️ 需下载 |

---

## 🎯 使用建议

### 1. 如果有预训练模型
- RCF可以作为主要的深度学习边缘检测算法
- 性能与PiDiNet相当或略优
- 训练数据集更大（BSDS500+PASCAL）

### 2. 如果暂时没有模型
- 优先使用 **PiDiNet**（已有模型，性能优秀 F1=0.71）
- HED作为备选（F1=0.35，性能不佳）
- RCF保留用于未来测试

### 3. 推荐算法优先级
1. **PiDiNet** - 最佳选择（有模型，性能好，速度快）
2. **RCF** - 如果下载了预训练模型
3. **Canny** - 传统基线
4. **HED** - 需要进一步优化

---

## 🔧 故障排查

### RCF性能差（F1 < 0.5）
**可能原因**：
1. ❌ 未使用预训练模型
2. ❌ 模型文件损坏或版本不匹配
3. ❌ 预处理参数不正确

**解决方法**：
1. 确认模型文件在 `models/` 目录
2. 检查启动日志中的模型加载信息
3. 确保下载了正确的预训练模型

### RCF加载失败
**错误信息**：
```
[RCF] ❌ Failed to load weights: ...
```

**解决方法**：
1. 检查模型文件路径
2. 确认模型文件完整性
3. 尝试重新下载模型

---

## 📝 后续改进建议

1. **下载RCF预训练模型**（必须）
2. **运行完整测试**验证性能
3. **对比所有算法**生成性能报告
4. **更新文档**说明各算法优缺点

---

**集成日期**: 2026-03-29
**实现者**: Claude Code
**论文**: Liu et al., IEEE TPAMI 2019
