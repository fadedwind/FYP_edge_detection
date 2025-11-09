"""image_recognition.py

将原 `image.recognition.py` 的内容改为下划线命名，便于作为正常 Python 模块导入。
包含一些简单的工具函数（detect_edges、extract_bounding_boxes、resize_for_model），
用于后续把识别/检测逻辑从主应用中抽离并作为可复用模块。
"""
from typing import Tuple
import numpy as np
import cv2


def detect_edges(img: np.ndarray, method: str = 'Canny', **kwargs) -> np.ndarray:
    """调用主算法（或独立实现）生成边缘二值图。返回 uint8 单通道图像。

    Args:
        img: BGR 图像（ndarray）
        method: 算法名，例如 'Canny','Sobel','Prewitt'
        kwargs: 算法参数（如 blur_ksize, low, high 等）
    Returns:
        edge_map: 单通道 uint8 图像（0/255）
    """
    h, w = img.shape[:2]
    if method == 'Canny':
        blur = kwargs.get('blur', 5)
        low = kwargs.get('low', 100)
        high = kwargs.get('high', 200)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = blur if blur % 2 == 1 else blur + 1
        gray = cv2.GaussianBlur(gray, (blur, blur), 1)
        edges = cv2.Canny(gray, low, high)
        return edges
    elif method == 'Sobel':
        k = kwargs.get('ksize', 3)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        mag = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
        _, th = cv2.threshold(mag, kwargs.get('thresh', 50), 255, cv2.THRESH_BINARY)
        return th
    else:
        # fallback: return empty image
        return np.zeros((h, w), dtype=np.uint8)


def extract_bounding_boxes(edge_map: np.ndarray, min_area: int = 100) -> list:
    """根据二值边缘图提取候选轮廓的边界框并按面积过滤。"""
    cnts, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h, area))
    return boxes


def resize_for_model(img: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """将输入图像缩放并归一化为模型输入格式（占位）。"""
    img_resized = cv2.resize(img, size)
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_resized = img_resized.astype('float32') / 255.0
    return img_resized


if __name__ == '__main__':
    print('这是 image_recognition.py 的占位实现。')
