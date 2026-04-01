# -*- coding: utf-8 -*-
"""
车辆检测模块 - 使用YOLOv8进行车辆检测

支持检测的车辆类别（COCO数据集）:
- Car (汽车)          - class id: 2
- Motorcycle (摩托车)  - class id: 3
- Bus (公交车)        - class id: 5
- Truck (卡车)        - class id: 7
- Boat (船)           - class id: 8
- Bicycle (自行车)    - class id: 1

依赖:
    pip install ultralytics opencv-python
"""
import os
import sys
from typing import List, Tuple, Dict, Optional
import numpy as np

# Windows兼容性
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# COCO数据集中的车辆类别
VEHICLE_CLASSES = {
    1: 'bicycle',   # 自行车
    2: 'car',       # 汽车
    3: 'motorcycle',# 摩托车
    5: 'bus',       # 公交车
    7: 'truck',     # 卡车
    8: 'boat'       # 船
}

# 类别ID列表
VEHICLE_CLASS_IDS = list(VEHICLE_CLASSES.keys())


class VehicleDetector:
    """车辆检测器类"""

    def __init__(self, model_size='n', device='cpu'):
        """
        初始化车辆检测器

        Args:
            model_size: 模型大小 ('n'=nano, 's'=small, 'm'=medium, 'l'=large, 'x'=extra large)
                      推荐: 'n' (速度快), 's' (精度高)
            device: 运行设备 ('cpu' 或 'cuda')
        """
        if not YOLO_AVAILABLE:
            raise ImportError("请先安装ultralytics: pip install ultralytics")

        self.model_size = model_size
        self.device = device
        self.model = None

        # 模型文件名映射
        model_names = {
            'n': 'yolov8n.pt',
            's': 'yolov8s.pt',
            'm': 'yolov8m.pt',
            'l': 'yolov8l.pt',
            'x': 'yolov8x.pt'
        }

        self.model_name = model_names.get(model_size, 'yolov8n.pt')

        print(f"[VehicleDetector] 初始化模型: {self.model_name}")
        print(f"[VehicleDetector] 设备: {device}")
        print(f"[VehicleDetector] 支持的车辆类别: {list(VEHICLE_CLASSES.values())}")

    def load_model(self):
        """加载YOLOv8模型"""
        if self.model is None:
            print(f"[VehicleDetector] 正在加载模型 {self.model_name}...")
            try:
                self.model = YOLO(self.model_name)
                print(f"[VehicleDetector] ✓ 模型加载成功")

                # 显示模型信息
                if hasattr(self.model, 'model'):
                    print(f"[VehicleDetector] 模型参数量: {self.model.model.parameters()['all']/1e6:.1f}M")

            except Exception as e:
                print(f"[VehicleDetector] ✗ 模型加载失败: {e}")
                raise

        return self.model

    def detect(self, image: np.ndarray, conf_threshold: float = 0.5,
              iou_threshold: float = 0.45) -> List[Dict]:
        """
        检测图像中的车辆

        Args:
            image: BGR格式的numpy数组 (H, W, 3)
            conf_threshold: 置信度阈值，默认0.5
            iou_threshold: NMS IOU阈值，默认0.45

        Returns:
            检测结果列表，每个元素包含:
            {
                'class_id': int,        # 类别ID
                'class_name': str,      # 类别名称 (car, truck等)
                'confidence': float,    # 置信度
                'bbox': [x1, y1, x2, y2] # 边界框坐标
            }
        """
        if not CV2_AVAILABLE:
            raise ImportError("请先安装opencv-python: pip install opencv-python")

        # 加载模型
        model = self.load_model()

        # 运行检测
        results = model(image, conf=conf_threshold, iou=iou_threshold,
                        classes=VEHICLE_CLASS_IDS, verbose=False)

        # 解析结果
        detections = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                # 获取类别ID
                cls_id = int(box.cls[0])

                # 获取置信度
                conf = float(box.conf[0])

                # 获取边界框坐标 [x1, y1, x2, y2]
                bbox = box.xyxy[0].tolist()

                # 添加到结果列表
                detections.append({
                    'class_id': cls_id,
                    'class_name': VEHICLE_CLASSES.get(cls_id, 'unknown'),
                    'confidence': conf,
                    'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                })

        return detections

    def detect_and_draw(self, image: np.ndarray,
                        conf_threshold: float = 0.5,
                        show_labels: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        检测车辆并在图像上绘制边界框

        Args:
            image: BGR格式的numpy数组
            conf_threshold: 置信度阈值
            show_labels: 是否显示标签

        Returns:
            (annotated_image, detections) - 标注后的图像和检测结果
        """
        # 检测车辆
        detections = self.detect(image, conf_threshold)

        # 复制图像用于绘制
        annotated = image.copy()

        # 定义颜色 (BGR格式)
        colors = {
            'bicycle': (0, 255, 0),    # 绿色
            'car': (255, 0, 0),        # 蓝色
            'motorcycle': (0, 0, 255), # 红色
            'bus': (255, 255, 0),      # 青色
            'truck': (255, 0, 255),    # 紫色
            'boat': (0, 255, 255)      # 黄色
        }

        # 绘制每个检测框
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            conf = det['confidence']

            # 获取颜色
            color = colors.get(class_name, (128, 128, 128))

            # 绘制边界框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            if show_labels:
                label = f"{class_name} {conf:.2f}"

                # 计算文本大小
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                         0.5, 1)

                # 绘制背景
                cv2.rectangle(annotated, (x1, y1 - text_h - 5),
                              (x1 + text_w, y1), color, -1)

                # 绘制文本
                cv2.putText(annotated, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated, detections


def detect_vehicles_simple(image_path: str, output_path: Optional[str] = None):
    """
    简单的车辆检测函数

    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径（可选）
    """
    if not CV2_AVAILABLE:
        print("[ERROR] opencv-python未安装")
        return

    if not YOLO_AVAILABLE:
        print("[ERROR] ultralytics未安装")
        print("请运行: pip install ultralytics opencv-python")
        return

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] 无法读取图像: {image_path}")
        return

    # 创建检测器
    detector = VehicleDetector(model_size='n', device='cpu')

    # 检测并绘制
    print(f"\n正在检测: {image_path}")
    annotated, detections = detector.detect_and_draw(image, conf_threshold=0.5)

    # 打印检测结果
    print(f"\n检测到 {len(detections)} 个车辆:")
    for i, det in enumerate(detections, 1):
        print(f"  {i}. {det['class_name']:12s} - 置信度: {det['confidence']:.2f} - "
              f"位置: ({det['bbox'][0]}, {det['bbox'][1]}) -> ({det['bbox'][2]}, {det['bbox'][3]})")

    # 保存结果
    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"\n结果已保存到: {output_path}")

    return detections


# 统计车辆数量
def count_vehicles(detections: List[Dict]) -> Dict[str, int]:
    """
    统计各类车辆的数量

    Args:
        detections: detect()返回的检测结果列表

    Returns:
        各类别的计数字典
    """
    counts = {}

    for det in detections:
        class_name = det['class_name']
        counts[class_name] = counts.get(class_name, 0) + 1

    return counts


if __name__ == '__main__':
    """测试代码"""
    # 测试图像检测
    test_image = 'test_car.jpg'

    if os.path.exists(test_image):
        print("="*70)
        print(" 车辆检测测试")
        print("="*70)

        detect_vehicles_simple(test_image, 'output_vehicles.jpg')

        print("\n" + "="*70)
        print(" 说明")
        print("="*70)
        print("如需使用自己的图像，请修改test_image变量")
        print("或调用: detect_vehicles_simple('your_image.jpg')")
        print("="*70)
    else:
        print(f"[INFO] 测试图像不存在: {test_image}")
        print("\n使用方法:")
        print("  detect_vehicles_simple('your_image.jpg', 'output.jpg')")
