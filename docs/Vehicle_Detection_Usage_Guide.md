# 车辆检测模块使用指南

**文件**: `vehicle_detection.py`
**依赖**: ultralytics, opencv-python

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install ultralytics opencv-python
```

**注意**:
- 首次运行会自动下载YOLOv8模型（约6MB）
- 模型会缓存到 `~/.cache/ultralytics/` 目录

### 2. 基本使用

#### 方法1：简单检测（推荐）

```python
from vehicle_detection import detect_vehicles_simple

# 检测图像中的车辆
detections = detect_vehicles_simple(
    'input_image.jpg',
    'output_image.jpg'
)

# 结果会自动保存到output_image.jpg
```

#### 方法2：使用API

```python
from vehicle_detection import VehicleDetector
import cv2

# 读取图像
image = cv2.imread('test.jpg')

# 创建检测器
detector = VehicleDetector(model_size='n')  # 'n'=nano(快), 's'=small(精确)

# 检测车辆
detections = detector.detect(image, conf_threshold=0.5)

# 查看结果
for det in detections:
    print(f"类别: {det['class_name']}")
    print(f"置信度: {det['confidence']:.2f}")
    print(f"位置: {det['bbox']}")
    print()

# 绘制边界框
annotated, _ = detector.detect_and_draw(image)
cv2.imwrite('output.jpg', annotated)
```

---

## 📊 支持的车辆类型

| 类别ID | 类别名称 | 中文名称 |
|--------|----------|----------|
| 1 | bicycle | 自行车 |
| 2 | car | 汽车 |
| 3 | motorcycle | 摩托车 |
| 5 | bus | 公交车 |
| 7 | truck | 卡车 |
| 8 | boat | 船 |

**总计**: 6种车辆类型

---

## 🎯 参数说明

### VehicleDetector初始化参数

```python
detector = VehicleDetector(
    model_size='n',    # 模型大小: 'n', 's', 'm', 'l', 'x'
    device='cpu'       # 设备: 'cpu' 或 'cuda'
)
```

**模型大小选择**:

| 模型 | 参数量 | 速度 | 精度(mAP@0.5) | 推荐场景 |
|------|--------|------|---------------|----------|
| **yolov8n** | 3.2M | ⭐⭐⭐⭐⭐ | 37.3% | 实时、嵌入式 |
| **yolov8s** | 11.2M | ⭐⭐⭐⭐ | 44.9% | 通用、平衡 |
| yolov8m | 25.9M | ⭐⭐⭐ | 50.2% | 离线处理 |
| yolov8l | 43.7M | ⭐⭐ | 52.9% | 高精度 |
| yolov8x | 68.2M | ⭐ | 53.9% | 最高精度 |

**推荐**:
- **实时应用**: `model_size='n'` (nano)
- **通用场景**: `model_size='s'` (small)

### 检测参数

```python
detections = detector.detect(
    image,                # 输入图像 (numpy array)
    conf_threshold=0.5,   # 置信度阈值 (0-1)
    iou_threshold=0.45    # NMS IOU阈值 (0-1)
)
```

**参数说明**:
- `conf_threshold`: 只保留置信度高于此值的结果（默认0.5）
- `iou_threshold`: 非极大值抑制的IOU阈值（默认0.45）

---

## 💡 使用示例

### 示例1：单张图像检测

```python
import cv2
from vehicle_detection import VehicleDetector

# 读取图像
image = cv2.imread('traffic_scene.jpg')

# 创建检测器
detector = VehicleDetector(model_size='s')

# 检测车辆
detections = detector.detect(image, conf_threshold=0.6)

# 统计各类车辆数量
from vehicle_detection import count_vehicles
counts = count_vehicles(detections)

print("车辆统计:")
for vehicle_type, count in counts.items():
    print(f"  {vehicle_type}: {count}辆")

# 输出:
# 车辆统计:
#   car: 5辆
#   truck: 2辆
#   bus: 1辆
```

### 示例2：批量处理图像

```python
import os
import glob
from vehicle_detection import VehicleDetector

# 创建检测器
detector = VehicleDetector(model_size='n')

# 获取所有图像
image_files = glob.glob('images/*.jpg')

# 批量处理
for image_file in image_files:
    image = cv2.imread(image_file)
    detections = detector.detect(image)

    print(f"{os.path.basename(image_file)}: {len(detections)}个车辆")
```

### 示例3：视频检测

```python
from vehicle_detection import VehicleDetector
import cv2

# 创建检测器
detector = VehicleDetector(model_size='n')

# 打开视频
cap = cv2.VideoCapture('traffic_video.mp4')

# 获取视频信息
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# 逐帧处理
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 每5帧检测一次（提高速度）
    if frame_count % 5 == 0:
        detections = detector.detect(frame)
        print(f"Frame {frame_count}: {len(detections)} vehicles")

        # 绘制边界框
        annotated, _ = detector.detect_and_draw(frame)
    else:
        annotated = frame

    # 写入输出视频
    out.write(annotated)
    frame_count += 1

cap.release()
out.release()
print(f"处理完成，共 {frame_count} 帧")
```

### 示例4：只检测特定类型车辆

```python
import cv2
from vehicle_detection import VehicleDetector

image = cv2.imread('traffic.jpg')
detector = VehicleDetector()

# 检测所有车辆
detections = detector.detect(image)

# 过滤：只保留汽车和卡车
car_truck = [d for d in detections if d['class_name'] in ['car', 'truck']]

print(f"检测到汽车和卡车: {len(car_truck)}个")

# 或者使用ultralytics的classes参数直接过滤
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
# 只检测car(2)和truck(7)
results = model(image, classes=[2, 7])
```

---

## 🎨 可视化选项

### 绘制边界框

```python
# 方式1：使用内置方法（推荐）
annotated, detections = detector.detect_and_draw(image, show_labels=True)

# 方式2：自定义绘制
image = cv2.imread('test.jpg')
detections = detector.detect(image)

for det in detections:
    x1, y1, x2, y2 = det['bbox']

    # 绘制边界框
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 绘制标签
    label = f"{det['class_name']} {det['confidence']:.2f}"
    cv2.putText(image, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite('output.jpg', image)
```

### 自定义颜色

```python
import cv2

# 定义颜色字典 (BGR格式)
colors = {
    'bicycle': (0, 255, 0),    # 绿色
    'car': (255, 0, 0),        # 蓝色
    'motorcycle': (0, 0, 255), # 红色
    'bus': (255, 255, 0),      # 青色
    'truck': (255, 0, 255),    # 紫色
    'boat': (0, 255, 255)      # 黄色
}

for det in detections:
    x1, y1, x2, y2 = det['bbox']
    color = colors.get(det['class_name'], (128, 128, 128))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
```

---

## ⚡ 性能优化建议

### 1. 选择合适的模型

| 使用场景 | 推荐模型 | 原因 |
|----------|----------|------|
| 实时视频 | yolov8n | 速度快，精度足够 |
| 批量处理 | yolov8s | 平衡速度和精度 |
| 高精度要求 | yolov8m/l | 精度最高 |
| 树莓派/Jetson | yolov8n | 体积小，功耗低 |

### 2. 调整检测频率

```python
# 视频处理时，不必每帧都检测
if frame_count % 5 == 0:  # 每5帧检测一次
    detections = detector.detect(frame)
else:
    detections = []  # 使用前一帧的结果
```

### 3. 使用GPU加速

```python
# 如果有NVIDIA GPU
detector = VehicleDetector(model_size='s', device='cuda')  # 比CPU快5-10倍
```

### 4. 降低输入分辨率

```python
# 对于高分辨率图像，可以先缩小
height, width = image.shape[:2]
if max(height, width) > 1920:
    scale = 1920 / max(height, width)
    small_image = cv2.resize(image, None, fx=scale, fy=scale)
    detections = detector.detect(small_image)
    # 再将边界框坐标映射回原始尺寸
```

---

## 📚 API参考

### VehicleDetector类

#### 初始化
```python
detector = VehicleDetector(model_size='n', device='cpu')
```

#### detect()
```python
detections = detector.detect(image, conf_threshold=0.5, iou_threshold=0.45)
```

**返回值**: List[Dict]，每个Dict包含:
- `class_id`: int - 类别ID
- `class_name`: str - 类别名称
- `confidence`: float - 置信度 (0-1)
- `bbox`: [x1, y1, x2, y2] - 边界框坐标

#### detect_and_draw()
```python
annotated_image, detections = detector.detect_and_draw(
    image,
    conf_threshold=0.5,
    show_labels=True
)
```

**返回值**:
- `annotated_image`: 绘制了边界框的图像
- `detections`: 检测结果列表

---

## 🔧 故障排查

### 问题1：模型下载失败

```
错误: Cannot connect to Hugging Hub
```

**解决**:
```bash
# 设置镜像（中国用户）
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题2：检测不到车辆

**可能原因**:
1. 置信度阈值太高 → 降低 `conf_threshold`（如0.3）
2. 图像中确实没有车辆
3. 车辆类型不在支持的6类中

### 问题3：速度太慢

**解决方法**:
1. 使用更小的模型 (`model_size='n'`)
2. 使用GPU (`device='cuda'`)
3. 降低输入分辨率
4. 减少检测频率（视频场景）

---

## 📖 更多资源

- **Ultralytics文档**: https://docs.ultralytics.com/
- **YOLOv8 GitHub**: https://github.com/ultralytics/ultralytics
- **COCO数据集**: https://cocodataset.org/

---

**最后更新**: 2026-03-29
**版本**: 1.0
