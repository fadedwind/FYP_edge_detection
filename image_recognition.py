"""image_recognition.py

轻量工具：包含边缘检测工具与一个可选的猫检测+品种分类原型（prototype）。

说明：为了避免导入时强依赖（如 ultralytics / torch / timm），
涉及第三方深度学习库的函数在内部再导入对应包，便于按需使用。
"""
from typing import Tuple, List, Dict
import numpy as np
import cv2


def detect_edges(img: np.ndarray, method: str = 'Canny', **kwargs) -> np.ndarray:
    """生成单通道边缘图（0/255 uint8）。保持与原实现兼容。"""
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
        return np.zeros((h, w), dtype=np.uint8)


def extract_bounding_boxes(edge_map: np.ndarray, min_area: int = 100) -> List[Tuple[int, int, int, int, float]]:
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


# ---------------- Prototype: detection + breed classification helpers ----------------
def load_detector(weights: str = 'yolov8n.pt', device: str = 'cpu'):
    """按需加载 ultralytics YOLO 模型。

    Raises ImportError when ultralytics not installed. Returns model object.
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise ImportError('请安装 ultralytics: pip install ultralytics') from e
    model = YOLO(weights)
    return model


def load_breed_classifier(weights_path: str, device: str = 'cpu', num_classes: int = 37):
    """按需加载 timm 的 EfficientNet-B0 并载入权重（供分类猫品种使用）。"""
    try:
        import torch
        import timm
    except Exception as e:
        raise ImportError('请安装 torch 和 timm（pip install torch timm）') from e

    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
    state = torch.load(weights_path, map_location=device)
    # 兼容多种 checkpoint 格式
    if isinstance(state, dict) and 'state_dict' in state:
        sd = state['state_dict']
    elif isinstance(state, dict) and 'model_state_dict' in state:
        sd = state['model_state_dict']
    else:
        sd = state

    # 去掉可能的 module. 前缀
    try:
        model.load_state_dict(sd)
    except Exception:
        new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(new_sd)

    model.to(device)
    model.eval()
    return model


def preprocess_for_classifier(img_crop: np.ndarray, size: Tuple[int, int] = (224, 224), device: str = 'cpu'):
    """将 crop 转为 torch tensor（NCHW）并标准化。"""
    try:
        import torch
    except Exception:
        raise ImportError('请安装 torch 以运行分类器')

    img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype('float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))[None, ...]
    import torch
    tensor = torch.from_numpy(img).to(device)
    return tensor


def detect_and_classify_cats(image_bgr: np.ndarray,
                             detector,
                             classifier,
                             breed_labels: List[str],
                             device: str = 'cpu',
                             detect_conf: float = 0.25,
                             crop_pad: float = 0.05) -> List[Dict]:
    """检测图片中的猫并对每个实例进行品种分类。

    返回值：list of dict: {'bbox':(x1,y1,x2,y2),'breed':str,'score':float,'det_score':float}
    """
    results = detector(image_bgr, conf=detect_conf)
    out = []
    if len(results) == 0:
        return out
    r = results[0]
    boxes = []
    if hasattr(r, 'boxes'):
        for b in r.boxes:
            try:
                cls_val = int(b.cls.cpu().numpy()[0])
                conf_val = float(b.conf.cpu().numpy()[0])
                xyxy = b.xyxy.cpu().numpy()[0].astype(int)
            except Exception:
                continue
            # COCO 中 cat 的 class id 为 16（若 detector 使用 COCO 权重）
            if cls_val == 16:
                boxes.append((xyxy, conf_val))

    h, w = image_bgr.shape[:2]
    import torch
    for xyxy, det_conf in boxes:
        x1, y1, x2, y2 = xyxy
        pad_x = int((x2 - x1) * crop_pad)
        pad_y = int((y2 - y1) * crop_pad)
        x1p = max(0, x1 - pad_x)
        y1p = max(0, y1 - pad_y)
        x2p = min(w, x2 + pad_x)
        y2p = min(h, y2 + pad_y)
        crop = image_bgr[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue
        tensor = preprocess_for_classifier(crop, device=device)
        with torch.no_grad():
            logits = classifier(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
            best_idx = int(np.argmax(probs))
            breed = breed_labels[best_idx] if best_idx < len(breed_labels) else str(best_idx)
            conf_breed = float(probs[best_idx])
        out.append({'bbox': (int(x1p), int(y1p), int(x2p), int(y2p)),
                    'breed': breed,
                    'score': conf_breed,
                    'det_score': det_conf})
    return out


def count_unique_breeds(detections: List[Dict]) -> Dict[str, int]:
    """统计 detections 列表中每个品种的数量并返回字典"""
    counts = {}
    for d in detections:
        b = d.get('breed', 'unknown')
        counts[b] = counts.get(b, 0) + 1
    return counts


if __name__ == '__main__':
    print('image_recognition: basic utilities loaded.')
