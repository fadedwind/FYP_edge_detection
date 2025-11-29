"""
Flask API 后端：车辆识别系统
提供图片上传和边缘检测+车辆识别的 RESTful API
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
from datetime import datetime
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 输出目录
OUTPUT_FOLDER = 'output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 固定尺寸（与桌面版一致）
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ------------------------------------------------------------------------------
# 边缘检测算法（从 pro1.py 提取 + 通用边缘检测）
# ------------------------------------------------------------------------------
def canny_edge_detection(gray_img):
    """普通Canny：适合光线均匀的车辆图片"""
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 60, 180)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel_dilate)
    return edges

def color_canny_edge_detection(rgb_img):
    """彩色Canny：适合颜色鲜明的车辆"""
    r, g, b = cv2.split(rgb_img)
    r_edges = cv2.Canny(cv2.GaussianBlur(r, (5, 5), 1.5), 60, 180)
    g_edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 1.5), 60, 180)
    b_edges = cv2.Canny(cv2.GaussianBlur(b, (5, 5), 1.5), 60, 180)
    color_edges = cv2.bitwise_or(cv2.bitwise_or(r_edges, g_edges), b_edges)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    color_edges = cv2.morphologyEx(color_edges, cv2.MORPH_CLOSE, kernel_close)
    color_edges = cv2.morphologyEx(color_edges, cv2.MORPH_DILATE, kernel_dilate)
    return color_edges

def sobel_edge_detection(gray_img):
    """普通Sobel：适合车身边缘清晰的图片"""
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
    sobel_x = np.uint8(np.absolute(sobel_x))
    sobel_y = np.uint8(np.absolute(sobel_y))
    sobel_edges = cv2.bitwise_or(sobel_x, sobel_y)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    sobel_edges = cv2.morphologyEx(sobel_edges, cv2.MORPH_CLOSE, kernel_close)
    return sobel_edges

def color_sobel_edge_detection(rgb_img):
    """彩色Sobel：适合复杂背景下的彩色车辆"""
    r, g, b = cv2.split(rgb_img)
    def sobel_single_channel(channel):
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=5)
        return np.uint8(np.absolute(cv2.bitwise_or(sobel_x, sobel_y)))
    r_edges = sobel_single_channel(r)
    g_edges = sobel_single_channel(g)
    b_edges = sobel_single_channel(b)
    color_sobel_edges = cv2.bitwise_or(cv2.bitwise_or(r_edges, g_edges), b_edges)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    color_sobel_edges = cv2.morphologyEx(color_sobel_edges, cv2.MORPH_CLOSE, kernel_close)
    return color_sobel_edges

EDGE_ALGORITHMS = {
    "Canny边缘检测": canny_edge_detection,
    "彩色Canny边缘检测": color_canny_edge_detection,
    "Sobel边缘检测": sobel_edge_detection,
    "彩色Sobel边缘检测": color_sobel_edge_detection
}

# ------------------------------------------------------------------------------
# 特征提取和分类（从 pro1.py 提取）
# ------------------------------------------------------------------------------
def extract_vehicle_features(rgb_img, edge_img):
    """提取车辆关键特征"""
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0.0, 0.0, (0, 0, 0), (0, 0, 0, 0)
    
    max_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_contour)
    x, y, w, h = cv2.boundingRect(max_contour)
    aspect_ratio = w / h if h != 0 else 0.0
    rect_area = w * h
    rectangularity = area / rect_area if rect_area != 0 else 0.0
    
    mask = np.zeros_like(edge_img)
    cv2.drawContours(mask, [max_contour], -1, 255, -1)
    r_mean = np.mean(rgb_img[mask == 255, 2]) if np.sum(mask == 255) > 0 else 0
    g_mean = np.mean(rgb_img[mask == 255, 1]) if np.sum(mask == 255) > 0 else 0
    b_mean = np.mean(rgb_img[mask == 255, 0]) if np.sum(mask == 255) > 0 else 0
    main_color = (r_mean, g_mean, b_mean)
    
    return area, aspect_ratio, rectangularity, main_color, (x, y, w, h)

def classify_vehicle(area, aspect_ratio, rectangularity, main_color, min_rectangularity):
    """车辆分类规则"""
    r, g, b = main_color
    total_brightness = (r + g + b) / 3
    
    if area < 5000:
        return "未识别（轮廓过小，非车辆）"
    if total_brightness < 40:
        return "未识别（图片过暗，无法判断）"
    if rectangularity < min_rectangularity:
        return f"未识别（轮廓不规则，矩形度{rectangularity:.2f} < {min_rectangularity:.2f}）"
    
    if (aspect_ratio >= 2.0 and aspect_ratio <= 5.0) or (aspect_ratio >= 1.2 and aspect_ratio < 2.0):
        if (r > g + 30 and r > b + 30) or (g > r + 30 and g > b + 30) or (b > r + 30 and b > g + 30):
            return "识别为：车辆"
        else:
            if abs(r - g) < 20 and abs(g - b) < 20:
                return "识别为：车辆"
            else:
                return "未识别（颜色特征不符合车辆）"
    else:
        return f"未识别（长宽比{aspect_ratio:.1f}，不符合车辆范围）"

# ------------------------------------------------------------------------------
# 工具函数：图片编码/解码
# ------------------------------------------------------------------------------
def image_to_base64(img_array):
    """将 numpy 数组转换为 base64 字符串"""
    if len(img_array.shape) == 2:  # 灰度图
        pil_img = Image.fromarray(img_array, mode='L')
    else:  # 彩色图
        pil_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def base64_to_image(base64_str):
    """将 base64 字符串转换为 numpy 数组"""
    # 移除 data:image/...;base64, 前缀
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    pil_img = Image.open(BytesIO(img_data))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ------------------------------------------------------------------------------
# API 路由
# ------------------------------------------------------------------------------
@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({"status": "ok", "message": "Vehicle Detection API is running"})

@app.route('/api/detect', methods=['POST'])
def detect_vehicle():
    """车辆识别主接口"""
    try:
        data = request.get_json()
        
        # 验证必需参数
        if 'image' not in data:
            return jsonify({"error": "缺少图片数据"}), 400
        if 'algorithm' not in data:
            return jsonify({"error": "缺少算法选择"}), 400
        
        algorithm = data['algorithm']
        min_rectangularity = float(data.get('min_rectangularity', 0.2))
        
        # 解码图片
        img_bgr = base64_to_image(data['image'])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 执行边缘检测
        if algorithm not in EDGE_ALGORITHMS:
            return jsonify({"error": f"不支持的算法: {algorithm}"}), 400
        
        algo_func = EDGE_ALGORITHMS[algorithm]
        if "彩色" in algorithm:
            edge_img = algo_func(img_bgr)
        else:
            edge_img = algo_func(gray_img)
        
        # 提取特征和分类
        area, aspect_ratio, rectangularity, main_color, vehicle_rect = extract_vehicle_features(img_bgr, edge_img)
        classification_result = classify_vehicle(area, aspect_ratio, rectangularity, main_color, min_rectangularity)
        
        # 绘制标记矩形（如果识别成功）
        marked_img = img_bgr.copy()
        is_vehicle = "识别为：车辆" in classification_result
        if is_vehicle:
            x, y, w, h = vehicle_rect
            cv2.rectangle(marked_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(marked_img, "车辆", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 编码返回图片
        marked_img_base64 = image_to_base64(marked_img)
        edge_img_base64 = image_to_base64(edge_img)
        
        # 返回结果
        return jsonify({
            "success": True,
            "result": classification_result,
            "is_vehicle": is_vehicle,
            "features": {
                "area": int(area),
                "aspect_ratio": round(aspect_ratio, 2),
                "rectangularity": round(rectangularity, 2),
                "main_color": [int(c) for c in main_color],
                "vehicle_rect": vehicle_rect
            },
            "images": {
                "marked": marked_img_base64,
                "edge": edge_img_base64
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    """获取可用的边缘检测算法列表"""
    return jsonify({
        "algorithms": list(EDGE_ALGORITHMS.keys())
    })

@app.route('/api/edge-detect', methods=['POST'])
def edge_detect():
    """通用边缘检测接口（主页使用），返回单通道边缘图和简单指标"""
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "缺少图片数据"}), 400
        algorithm = data.get('algorithm', 'Canny')
        blur = int(data.get('blur', 7))
        canny_low = int(data.get('canny_low', 100))
        canny_high = int(data.get('canny_high', 220))
        sobel_ksize = int(data.get('sobel_ksize', 3))
        dilate_ksize = int(data.get('dilate_ksize', 1))

        # 解码图片
        img_bgr = base64_to_image(data['image'])
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 预处理：核大小规范化（奇数且>=1）
        if blur % 2 == 0:
            blur += 1
        if blur < 1:
            blur = 1
        if sobel_ksize % 2 == 0 or sobel_ksize < 1:
            sobel_ksize = 3
        if dilate_ksize % 2 == 0 or dilate_ksize < 1:
            dilate_ksize = 1

        edge = None
        if algorithm == 'Sobel':
            gray_blur = cv2.GaussianBlur(img_gray, (blur, blur), 1)
            gx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
            gy = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
            edge = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
        elif algorithm == '彩色Sobel':
            img_blur = cv2.GaussianBlur(img_bgr, (blur, blur), 1)
            sobel_edges = []
            for i in range(3):
                gx = cv2.Sobel(img_blur[:, :, i], cv2.CV_64F, 1, 0, ksize=sobel_ksize)
                gy = cv2.Sobel(img_blur[:, :, i], cv2.CV_64F, 0, 1, ksize=sobel_ksize)
                sobel_edges.append(cv2.convertScaleAbs(cv2.magnitude(gx, gy)))
            edge = cv2.bitwise_or(sobel_edges[0], sobel_edges[1])
            edge = cv2.bitwise_or(edge, sobel_edges[2])
        elif algorithm == 'Canny':
            gray_blur = cv2.GaussianBlur(img_gray, (blur, blur), 1)
            edge = cv2.Canny(gray_blur, canny_low, canny_high)
        elif algorithm == '彩色Canny':
            img_blur = cv2.GaussianBlur(img_bgr, (blur, blur), 1)
            canny_edges = [cv2.Canny(img_blur[:, :, i], canny_low, canny_high) for i in range(3)]
            edge = cv2.bitwise_or(canny_edges[0], canny_edges[1])
            edge = cv2.bitwise_or(edge, canny_edges[2])
        elif algorithm == 'Prewitt':
            gray_blur = cv2.GaussianBlur(img_gray, (blur, blur), 1)
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
            gx = cv2.filter2D(gray_blur, cv2.CV_64F, kernel_x)
            gy = cv2.filter2D(gray_blur, cv2.CV_64F, kernel_y)
            edge = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
        else:
            # 兜底使用Canny
            gray_blur = cv2.GaussianBlur(img_gray, (blur, blur), 1)
            edge = cv2.Canny(gray_blur, canny_low, canny_high)

        if edge is None:
            edge = np.zeros_like(img_gray)

        # 膨胀（与桌面版统一：可通过 dilate_ksize 调整粗细/连通性）
        if dilate_ksize > 1:
            kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
            edge = cv2.dilate(edge, kernel, iterations=1)

        # 计算指标：非零边缘像素数、Precision、Recall、F1
        edge_pixels = int((edge > 0).sum())
        
        # 计算 Precision/Recall/F1（使用 Canny 作为参考）
        reference = get_reference_edge(img_bgr)
        _, detected = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)
        _, ref_binary = cv2.threshold(reference, 127, 255, cv2.THRESH_BINARY)
        
        TP = cv2.bitwise_and(detected, ref_binary).sum() // 255
        FP = cv2.bitwise_and(detected, cv2.bitwise_not(ref_binary)).sum() // 255
        FN = cv2.bitwise_and(cv2.bitwise_not(detected), ref_binary).sum() // 255
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        edge_b64 = image_to_base64(edge)
        return jsonify({
            "success": True,
            "images": {
                "edge": edge_b64
            },
            "metrics": {
                "edge_pixels": edge_pixels,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_reference_edge(img):
    """生成参考边缘（使用 Canny）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reference_edge = cv2.Canny(gray, 150, 255)
    return reference_edge

def compute_edge_strength(img, algorithm, blur_ksize, sobel_ksize, canny_low=100, canny_high=220):
    """生成单通道的边强度图（0-255 uint8），供阈值化用于 ODS/OIS 计算"""
    img_resized = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
    edge = None
    
    if algorithm == "Sobel":
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
        grad_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        grad_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        edge = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
    elif algorithm == "彩色Sobel":
        img_blur = cv2.GaussianBlur(img_resized, (blur_ksize, blur_ksize), 1)
        sobel_edges = []
        for i in range(3):
            grad_x = cv2.Sobel(img_blur[:, :, i], cv2.CV_64F, 1, 0, ksize=sobel_ksize)
            grad_y = cv2.Sobel(img_blur[:, :, i], cv2.CV_64F, 0, 1, ksize=sobel_ksize)
            sobel_edges.append(cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y)))
        edge = cv2.bitwise_or(sobel_edges[0], sobel_edges[1])
        edge = cv2.bitwise_or(edge, sobel_edges[2])
    elif algorithm == "Canny":
        # 对于批量处理和PR曲线生成，使用真正的Canny算法
        # Canny返回二值图，但我们需要强度图来生成PR曲线
        # 解决方案：使用Canny算法内部的梯度幅值（在非极大值抑制和双阈值之前）
        # 这样可以获得强度图，同时保持Canny算法的特性
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
        
        # 使用真正的Canny算法计算梯度（Canny内部使用3x3 Sobel）
        # 计算梯度幅值和方向
        gx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = cv2.magnitude(gx, gy)
        gradient_direction = np.arctan2(gy, gx) * 180 / np.pi
        
        # 实现非极大值抑制（NMS）- Canny算法的关键步骤
        # 这是真正的Canny算法的一部分，与edge_detection.py中的实现一致
        h, w = gradient_magnitude.shape
        suppressed = gradient_magnitude.copy()
        
        # 将角度归一化到0-180度
        gradient_direction = np.abs(gradient_direction)
        
        # 向量化实现NMS（更高效）
        # 创建4个方向的掩码
        mask_0 = ((gradient_direction >= 0) & (gradient_direction < 22.5)) | \
                 ((gradient_direction >= 157.5) & (gradient_direction <= 180))
        mask_45 = (gradient_direction >= 22.5) & (gradient_direction < 67.5)
        mask_90 = (gradient_direction >= 67.5) & (gradient_direction < 112.5)
        mask_135 = (gradient_direction >= 112.5) & (gradient_direction < 157.5)
        
        # 对每个方向进行NMS（只处理内部像素，边界设为0）
        # 0°方向：比较左右
        mag_center = suppressed[1:-1, 1:-1]
        mag_left = suppressed[1:-1, :-2]
        mag_right = suppressed[1:-1, 2:]
        mask_0_inner = mask_0[1:-1, 1:-1]
        suppressed[1:-1, 1:-1] = np.where(
            mask_0_inner,
            np.where((mag_center >= mag_left) & (mag_center >= mag_right), mag_center, 0),
            suppressed[1:-1, 1:-1]
        )
        
        # 45°方向：比较对角
        mag_diag1_45 = suppressed[:-2, 2:]
        mag_diag2_45 = suppressed[2:, :-2]
        mask_45_inner = mask_45[1:-1, 1:-1]
        suppressed[1:-1, 1:-1] = np.where(
            mask_45_inner,
            np.where((mag_center >= mag_diag1_45) & (mag_center >= mag_diag2_45), mag_center, 0),
            suppressed[1:-1, 1:-1]
        )
        
        # 90°方向：比较上下
        mag_up = suppressed[:-2, 1:-1]
        mag_down = suppressed[2:, 1:-1]
        mask_90_inner = mask_90[1:-1, 1:-1]
        suppressed[1:-1, 1:-1] = np.where(
            mask_90_inner,
            np.where((mag_center >= mag_up) & (mag_center >= mag_down), mag_center, 0),
            suppressed[1:-1, 1:-1]
        )
        
        # 135°方向：比较对角
        mag_diag1_135 = suppressed[:-2, :-2]
        mag_diag2_135 = suppressed[2:, 2:]
        mask_135_inner = mask_135[1:-1, 1:-1]
        suppressed[1:-1, 1:-1] = np.where(
            mask_135_inner,
            np.where((mag_center >= mag_diag1_135) & (mag_center >= mag_diag2_135), mag_center, 0),
            suppressed[1:-1, 1:-1]
        )
        
        # 边界处理：边界像素设为0
        suppressed[0, :] = 0
        suppressed[-1, :] = 0
        suppressed[:, 0] = 0
        suppressed[:, -1] = 0
        
        # 将NMS后的结果转换为uint8，作为强度图
        # 这样既保持了Canny的NMS特性（真正的Canny算法步骤），又能用于阈值扫描生成PR曲线
        edge = cv2.convertScaleAbs(suppressed)
    elif algorithm == "彩色Canny":
        # 对于批量处理，使用真正的Canny算法（每个通道应用NMS）
        img_blur = cv2.GaussianBlur(img_resized, (blur_ksize, blur_ksize), 1)
        canny_edges = []
        for i in range(3):
            # 计算梯度
            gx = cv2.Sobel(img_blur[:, :, i], cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(img_blur[:, :, i], cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = cv2.magnitude(gx, gy)
            gradient_direction = np.arctan2(gy, gx) * 180 / np.pi
            
            # 应用非极大值抑制（NMS）
            h, w = gradient_magnitude.shape
            suppressed = gradient_magnitude.copy()
            gradient_direction = np.abs(gradient_direction)
            
            # 创建方向掩码
            mask_0 = ((gradient_direction >= 0) & (gradient_direction < 22.5)) | \
                     ((gradient_direction >= 157.5) & (gradient_direction <= 180))
            mask_45 = (gradient_direction >= 22.5) & (gradient_direction < 67.5)
            mask_90 = (gradient_direction >= 67.5) & (gradient_direction < 112.5)
            mask_135 = (gradient_direction >= 112.5) & (gradient_direction < 157.5)
            
            # 对每个方向进行NMS
            mag_center = suppressed[1:-1, 1:-1]
            
            # 0°方向
            if np.any(mask_0[1:-1, 1:-1]):
                mag_left = suppressed[1:-1, :-2]
                mag_right = suppressed[1:-1, 2:]
                mask_0_inner = mask_0[1:-1, 1:-1]
                suppressed[1:-1, 1:-1] = np.where(
                    mask_0_inner,
                    np.where((mag_center >= mag_left) & (mag_center >= mag_right), mag_center, 0),
                    suppressed[1:-1, 1:-1]
                )
            
            # 45°方向
            if np.any(mask_45[1:-1, 1:-1]):
                mag_diag1 = suppressed[:-2, 2:]
                mag_diag2 = suppressed[2:, :-2]
                mask_45_inner = mask_45[1:-1, 1:-1]
                suppressed[1:-1, 1:-1] = np.where(
                    mask_45_inner,
                    np.where((mag_center >= mag_diag1) & (mag_center >= mag_diag2), mag_center, 0),
                    suppressed[1:-1, 1:-1]
                )
            
            # 90°方向
            if np.any(mask_90[1:-1, 1:-1]):
                mag_up = suppressed[:-2, 1:-1]
                mag_down = suppressed[2:, 1:-1]
                mask_90_inner = mask_90[1:-1, 1:-1]
                suppressed[1:-1, 1:-1] = np.where(
                    mask_90_inner,
                    np.where((mag_center >= mag_up) & (mag_center >= mag_down), mag_center, 0),
                    suppressed[1:-1, 1:-1]
                )
            
            # 135°方向
            if np.any(mask_135[1:-1, 1:-1]):
                mag_diag1 = suppressed[:-2, :-2]
                mag_diag2 = suppressed[2:, 2:]
                mask_135_inner = mask_135[1:-1, 1:-1]
                suppressed[1:-1, 1:-1] = np.where(
                    mask_135_inner,
                    np.where((mag_center >= mag_diag1) & (mag_center >= mag_diag2), mag_center, 0),
                    suppressed[1:-1, 1:-1]
                )
            
            # 边界处理
            suppressed[0, :] = 0
            suppressed[-1, :] = 0
            suppressed[:, 0] = 0
            suppressed[:, -1] = 0
            
            canny_edges.append(cv2.convertScaleAbs(suppressed))
        
        # 合并三个通道的结果
        edge = cv2.bitwise_or(canny_edges[0], canny_edges[1])
        edge = cv2.bitwise_or(edge, canny_edges[2])
    elif algorithm == "Prewitt":
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        grad_x = cv2.filter2D(gray_blur, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(gray_blur, cv2.CV_64F, kernel_y)
        edge = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
    else:
        # 默认使用梯度强度图（类似Canny）
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
        gx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
        edge = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
    
    if edge is None:
        edge = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
    return edge

@app.route('/api/batch-process', methods=['POST'])
def batch_process():
    """批量处理图片，计算 ODS/OIS 指标和 PR 曲线"""
    try:
        data = request.get_json()
        if 'images' not in data or not isinstance(data['images'], list):
            return jsonify({"error": "缺少图片数据列表"}), 400
        
        algorithm = data.get('algorithm', 'Canny')
        blur = int(data.get('blur', 7))
        sobel_ksize = int(data.get('sobel_ksize', 3))
        canny_low = int(data.get('canny_low', 100))
        canny_high = int(data.get('canny_high', 220))
        
        # 参数规范化
        if blur % 2 == 0:
            blur += 1
        if blur < 1:
            blur = 1
        if sobel_ksize % 2 == 0 or sobel_ksize < 1:
            sobel_ksize = 3
        
        # 准备输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_output_dir = os.path.join(OUTPUT_FOLDER, timestamp)
        os.makedirs(run_output_dir, exist_ok=True)
        
        # 阈值列表
        thresholds = list(range(0, 256, 1))
        thr_TP_sum = np.zeros(len(thresholds), dtype=np.float64)
        thr_FP_sum = np.zeros(len(thresholds), dtype=np.float64)
        thr_FN_sum = np.zeros(len(thresholds), dtype=np.float64)
        
        per_image_best = []
        processed_images = []
        
        # 处理每张图片
        for idx, img_base64 in enumerate(data['images']):
            try:
                img_bgr = base64_to_image(img_base64)
                img_resized = cv2.resize(img_bgr, (FRAME_WIDTH, FRAME_HEIGHT))
                reference = get_reference_edge(img_resized)
                edge_strength = compute_edge_strength(img_bgr, algorithm, blur, sobel_ksize, canny_low, canny_high)
                
                best_f1 = -1.0
                best_prec = best_rec = best_thr = 0
                
                # 对每个阈值计算 TP/FP/FN
                for i, t in enumerate(thresholds):
                    _, detected = cv2.threshold(edge_strength, t, 255, cv2.THRESH_BINARY)
                    det_mask = (detected > 0)
                    ref_mask = (reference > 0)
                    TP = int(np.logical_and(det_mask, ref_mask).sum())
                    FP = int(np.logical_and(det_mask, np.logical_not(ref_mask)).sum())
                    FN = int(np.logical_and(np.logical_not(det_mask), ref_mask).sum())
                    
                    thr_TP_sum[i] += TP
                    thr_FP_sum[i] += FP
                    thr_FN_sum[i] += FN
                    
                    # 计算单图指标
                    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                    if f1 > best_f1:
                        best_f1 = f1
                        best_prec = prec
                        best_rec = rec
                        best_thr = t
                
                per_image_best.append({
                    "index": idx,
                    "best_threshold": int(best_thr),
                    "best_precision": round(best_prec, 3),
                    "best_recall": round(best_rec, 3),
                    "best_f1": round(best_f1, 3)
                })
                
                # 保存最佳阈值下的边缘图
                # 对于Canny算法，如果使用真正的Canny会更好，但为了PR曲线我们使用阈值化的强度图
                # 这里我们保存阈值化后的结果
                _, detected_final = cv2.threshold(edge_strength, best_thr, 255, cv2.THRESH_BINARY)
                
                # 对于Canny，可以额外使用真正的Canny算法生成一个对比图（可选）
                # 但为了保持一致性，我们使用阈值化的结果
                edge_save_name = f"image_{idx}_edge.png"
                edge_save_path = os.path.join(run_output_dir, edge_save_name)
                cv2.imwrite(edge_save_path, detected_final)
                
                processed_images.append({
                    "index": idx,
                    "edge_image": image_to_base64(detected_final),
                    "filename": edge_save_name
                })
            except Exception as e:
                print(f'处理图片 {idx} 失败：', e)
                continue
        
        if len(per_image_best) == 0:
            return jsonify({"error": "未成功处理任何图片"}), 400
        
        # 计算 ODS（数据集级）
        denom_prec = thr_TP_sum + thr_FP_sum
        precision_mean = np.zeros_like(thr_TP_sum)
        nonzero = denom_prec > 0
        precision_mean[nonzero] = thr_TP_sum[nonzero] / denom_prec[nonzero]
        
        denom_rec = thr_TP_sum + thr_FN_sum
        recall_mean = np.zeros_like(thr_TP_sum)
        nonzero_r = denom_rec > 0
        recall_mean[nonzero_r] = thr_TP_sum[nonzero_r] / denom_rec[nonzero_r]
        
        mean_f1_per_thr = np.zeros_like(precision_mean)
        nonzero_f = (precision_mean + recall_mean) > 0
        mean_f1_per_thr[nonzero_f] = 2 * precision_mean[nonzero_f] * recall_mean[nonzero_f] / (
            precision_mean[nonzero_f] + recall_mean[nonzero_f]
        )
        
        best_idx = int(np.argmax(mean_f1_per_thr))
        ods_thr = int(thresholds[best_idx])
        ods_prec = float(precision_mean[best_idx])
        ods_rec = float(recall_mean[best_idx])
        ods_f1 = float(mean_f1_per_thr[best_idx])
        
        # 计算 OIS
        ois_prec = np.mean([x["best_precision"] for x in per_image_best])
        ois_rec = np.mean([x["best_recall"] for x in per_image_best])
        ois_f1 = np.mean([x["best_f1"] for x in per_image_best])
        
        # 生成 PR 曲线
        precision_mean = np.clip(precision_mean, 0.0, 1.0)
        recall_mean = np.clip(recall_mean, 0.0, 1.0)
        
        idxs = np.argsort(recall_mean)
        recall_sorted = recall_mean[idxs]
        precision_sorted = precision_mean[idxs]
        
        # 去重并取最大值（对于每个recall值，取最大的precision）
        unique_recalls = []
        max_precisions = []
        for r, p in zip(recall_sorted, precision_sorted):
            if len(unique_recalls) == 0 or r != unique_recalls[-1]:
                unique_recalls.append(r)
                max_precisions.append(p)
            else:
                if p > max_precisions[-1]:
                    max_precisions[-1] = p
        
        recall_sorted = np.array(unique_recalls)
        precision_sorted = np.array(max_precisions)
        
        # 过滤掉recall很小且precision很高的点（去掉左侧蓝色段）
        # 只过滤recall < 0.05 且 precision > 0.95 的点（这些是蓝色段）
        # 保留其他所有点
        valid_mask = ~((recall_sorted < 0.05) & (precision_sorted > 0.95))
        if np.any(valid_mask):
            recall_sorted = recall_sorted[valid_mask]
            precision_sorted = precision_sorted[valid_mask]
        
        # 确保PR曲线以(1, precision_at_recall_1)结束
        # 如果recall_sorted的最大值不是1，添加最后一个点
        if len(recall_sorted) > 0 and recall_sorted[-1] < 1.0 - 1e-6:
            recall_sorted = np.concatenate([recall_sorted, [1.0]])
            precision_sorted = np.concatenate([precision_sorted, [precision_sorted[-1]]])
        
        # 插值生成平滑曲线
        if len(recall_sorted) == 0 or recall_sorted.max() - recall_sorted.min() < 1e-6:
            recall_fine = recall_sorted
            precision_fine = precision_sorted
        else:
            # 使用更多的插值点以获得更平滑的曲线
            recall_fine = np.linspace(recall_sorted.min(), recall_sorted.max(), 512)
            precision_fine = np.interp(recall_fine, recall_sorted, precision_sorted)
            # 对于PR曲线，应该使用单调递减的precision（随着recall增加，precision应该递减或保持不变）
            # 使用累积最大值来确保单调性
            for i in range(len(precision_fine) - 2, -1, -1):
                precision_fine[i] = max(precision_fine[i], precision_fine[i + 1])
        
        # 绘制 PR 曲线
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(recall_fine, precision_fine, '-', linewidth=2, label='PR Curve')
        ax.plot(recall_sorted, precision_sorted, 'o', markersize=3, alpha=0.6)
        
        # 绘制F1-score等值线
        # F1 = 2*P*R/(P+R)，对于固定的F1值，可以解出P = F1*R/(2*R - F1)
        f1_values = [0.3, 0.5, 0.7]
        recall_f1 = np.linspace(0.01, 0.99, 100)
        for f1 in f1_values:
            # 计算对应的precision值
            # 从 F1 = 2*P*R/(P+R) 解出 P = F1*R/(2*R - F1)
            precision_f1 = f1 * recall_f1 / (2 * recall_f1 - f1)
            # 只保留有效的precision值（0到1之间）
            valid_f1 = (precision_f1 >= 0) & (precision_f1 <= 1) & (recall_f1 > f1 / 2)
            if np.any(valid_f1):
                ax.plot(recall_f1[valid_f1], precision_f1[valid_f1], '--', 
                       linewidth=1, alpha=0.5, color='gray', label=f'F1={f1}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'PR Curve ({algorithm})')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend()
        
        pr_path = os.path.join(run_output_dir, f"PR_curve_{algorithm}.png")
        fig.tight_layout()
        fig.savefig(pr_path, dpi=100)
        plt.close(fig)
        
        # 读取 PR 曲线图片并转换为 base64
        with open(pr_path, 'rb') as f:
            pr_img_data = f.read()
        pr_base64 = base64.b64encode(pr_img_data).decode()
        pr_image_base64 = f"data:image/png;base64,{pr_base64}"
        
        # 保存 CSV
        csv_path = os.path.join(run_output_dir, f"batch_metrics_{algorithm}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.writer(cf)
            writer.writerow(["index", "best_threshold", "best_precision", "best_recall", "best_f1"])
            for row in per_image_best:
                writer.writerow([row["index"], row["best_threshold"], row["best_precision"], 
                               row["best_recall"], row["best_f1"]])
            writer.writerow([])
            writer.writerow(["ODS_threshold", ods_thr])
            writer.writerow(["ODS_precision", round(ods_prec, 3)])
            writer.writerow(["ODS_recall", round(ods_rec, 3)])
            writer.writerow(["ODS_f1", round(ods_f1, 3)])
            writer.writerow(["OIS_precision", round(ois_prec, 3)])
            writer.writerow(["OIS_recall", round(ois_rec, 3)])
            writer.writerow(["OIS_f1", round(ois_f1, 3)])
        
        return jsonify({
            "success": True,
            "metrics": {
                "ods": {
                    "threshold": ods_thr,
                    "precision": round(ods_prec, 3),
                    "recall": round(ods_rec, 3),
                    "f1": round(ods_f1, 3)
                },
                "ois": {
                    "precision": round(ois_prec, 3),
                    "recall": round(ois_rec, 3),
                    "f1": round(ois_f1, 3)
                }
            },
            "per_image_results": per_image_best,
            "pr_curve": pr_image_base64,
            "output_dir": run_output_dir,
            "processed_count": len(per_image_best),
            "processed_images": processed_images
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("启动车辆识别 API 服务器...")
    print("API 地址: http://localhost:5000")
    print("可用接口:")
    print("  GET  /api/health      - 健康检查")
    print("  GET  /api/algorithms  - 获取车辆识别算法列表")
    print("  POST /api/detect      - 车辆识别")
    print("  POST /api/edge-detect - 通用边缘检测（主页）")
    print("  POST /api/batch-process - 批量处理图片（计算 ODS/OIS 和 PR 曲线）")
    app.run(host='0.0.0.0', port=5000, debug=True)

