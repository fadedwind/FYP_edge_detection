# RCF预训练模型下载指南

## 📥 官方模型下载说明

### ✅ 已找到的直接下载链接！

**RCF预训练模型（PyTorch格式）**:
- **Google Drive链接**: https://drive.google.com/open?id=1TupHeoBKawrniDka0Hc64m3BG4OKG8nM
- **来源**: meteorshowers/RCF-pytorch (第三方实现)
- **性能**: ODS F-score = 0.808 (官方为0.812，略低但可用)
- **说明**: 虽然不是最佳模型，但可以用于测试和交流

**VGG16骨干网络（RCF需要）**:
- **Google Drive链接**: https://drive.google.com/file/d/1lUhPKKj-BSOH7yQL0mOIavvrUbjydPp5/view?usp=sharing
- **说明**: ImageNet预训练的VGG16，用作RCF的backbone

### 方法1：使用gdown自动下载（推荐）

```bash
# 安装gdown（如果尚未安装）
pip install gdown

# 创建models目录
mkdir -p models

# 下载RCF预训练模型
gdown https://drive.google.com/uc?id=1TupHeoBKawrniDka0Hc64m3BG4OKG8nM -O models/rcf_pretrained.pth

# 下载VGG16骨干网络（可选）
gdown https://drive.google.com/uc?id=1lUhPKKj-BSOH7yQL0mOIavvrUbjydPp5 -O models/vgg16_pretrained.pth
```

### 方法2：手动浏览器下载

1. 访问Google Drive链接: https://drive.google.com/open?id=1TupHeoBKawrniDka0Hc64m3BG4OKG8nM
2. 点击下载按钮
3. 将下载的文件重命名为 `rcf_bsds500_pascal.pth`
4. 移动到项目的 `models/` 目录

### 方法2：使用Git LFS（如果可用）

```bash
# 进入项目目录
cd "F:\Coding\MUST\AAA--G4444444444444\FYP\FYP_edge_detection_git - v1"

# 克隆RCF仓库（包含模型链接）
git clone https://github.com/yun-liu/RCF-PyTorch.git temp_rcf

# 查看是否有模型文件
ls temp_rcf/*.pth
```

### 方法3：从备用源下载

如果官方链接不可用，可以尝试：
1. 搜索 "bsds500_pascal_model.pth" on GitHub
2. 查找其他RCF实现的权重文件
3. 联系RCF作者获取模型

---

## ⚠️ 当前状态

### 已完成
- ✅ RCF代码实现（`rcf.py`）
- ✅ API集成完成
- ✅ 前端界面更新
- ✅ 测试脚本创建
- ✅ 找到模型下载链接

### 待完成
- ❌ 预训练模型文件缺失（需要手动下载）
- ⚠️ 网络限制无法自动下载Google Drive文件

## 🔥 手动下载步骤（必需）

由于网络限制，**必须手动下载**RCF预训练模型。请按照以下步骤操作：

### 步骤1：打开浏览器下载链接

**在浏览器中打开以下链接**：
```
https://drive.google.com/open?id=1TupHeoBKawrniDka0Hc64m3BG4OKG8nM
```

或者访问：
```
https://github.com/meteorshowers/RCF-pytorch
```
在README中找到"pretrained model"下载链接。

### 步骤2：下载模型文件

1. Google Drive页面会显示一个文件（可能是 `.pth` 或 `.tar.gz` 格式）
2. 点击页面右上角的下载图标（⬇️）
3. 等待下载完成（文件大小约200-300MB）

### 步骤3：放置模型文件

下载完成后：
1. 将下载的文件重命名为：`rcf_bsds500_pascal.pth`
2. 复制到项目的 `models/` 目录：
   ```
   F:\Coding\MUST\AAA--G4444444444444\FYP\FYP_edge_detection_git - v1\models\
   ```

### 步骤4：验证下载

```bash
# 检查文件是否存在
ls -lh models/rcf_bsds500_pascal.pth

# 应该看到文件大小约为 200-300MB
```

---

## 🎯 临时测试方案（无需模型）

如果暂时无法下载模型，可以：

### 方案A：测试其他算法
运行完整算法对比测试：
```bash
python test_algorithms_fixed.py
```
这将测试Canny、HED、PiDiNet的性能。

### 方案B：创建RCF模拟测试
使用随机初始化的RCF测试代码流程：
```bash
python test_rcf.py
```
注意：性能会很差（F1 < 0.2），但可以验证代码逻辑。

---

## 📝 下载完成后的测试步骤

1. **确认模型文件位置**：
   ```bash
   ls -lh models/ | grep -i rcf
   ```
   应该看到：`rcf_bsds500_pascal.pth` 或类似文件

2. **运行RCF测试**：
   ```bash
   python test_rcf.py
   ```

3. **查看测试结果**：
   - 控制台会显示性能对比
   - 详细结果保存在 `output/rcf_test_*.txt`

4. **预期性能**：
   - F1 Score: ~0.78-0.81
   - vs Canny: +52-58%
   - vs PiDiNet: +9-14%

---

## 🔗 相关链接

- **RCF PyTorch**: https://github.com/yun-liu/RCF-PyTorch
- **RCF Caffe**: https://github.com/yun-liu/RCF
- **RCF Jittor**: https://github.com/yun-liu/RCF-Jittor
- **项目主页**: https://mmcheng.net/rcfedge/

---

## 💡 建议

如果下载困难，建议：

### 短期（今天）
1. ✅ 使用 **PiDiNet** 作为主要深度学习算法（F1=0.71，已验证优秀）
2. ⚠️ RCF作为可选项，等模型下载后再测试

### 长期（本周）
1. 从官方渠道下载RCF模型
2. 完成所有算法的完整性能对比
3. 生成最终的性能评估报告

---

**创建时间**: 2026-03-29
**用途**: RCF模型下载指南
