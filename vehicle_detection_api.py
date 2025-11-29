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

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 上传目录
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

        # 简单指标：非零边缘像素数
        edge_pixels = int((edge > 0).sum())

        edge_b64 = image_to_base64(edge)
        return jsonify({
            "success": True,
            "images": {
                "edge": edge_b64
            },
            "metrics": {
                "edge_pixels": edge_pixels
            }
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
    app.run(host='0.0.0.0', port=5000, debug=True)

