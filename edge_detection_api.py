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

# HED模型缓存（避免每次重新加载）
_cached_hed_net = None
_cached_hed_paths = None  # (prototxt_path, caffemodel_path)

def get_hed_net():
    """返回缓存的HED OpenCV DNN网络，首次调用时加载并缓存
    如果找不到Caffe模型/prototxt或加载失败，返回None
    """
    global _cached_hed_net, _cached_hed_paths
    try:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        repo_dir = os.getcwd()
    prototxt_path = os.path.join(repo_dir, 'models', 'hed_deploy.prototxt')
    cand1 = os.path.join(repo_dir, 'models', 'hed_pretrained_bsds.caffemodel')
    cand2 = os.path.join(repo_dir, 'models', 'hed_bsds.caffemodel')
    caffemodel_path = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else None)
    
    paths = (prototxt_path, caffemodel_path)
    # 如果已缓存且路径相同，直接返回
    if _cached_hed_net is not None and _cached_hed_paths == paths:
        return _cached_hed_net
    
    # 检查文件是否存在
    if not (os.path.exists(prototxt_path) and caffemodel_path and os.path.exists(caffemodel_path)):
        print(f'HED模型文件不存在: prototxt={os.path.exists(prototxt_path)}, caffemodel={caffemodel_path}')
        return None
    
    # 加载并缓存模型
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        _cached_hed_net = net
        _cached_hed_paths = paths
        print(f'HED模型加载成功: {caffemodel_path}')
        return _cached_hed_net
    except Exception as e:
        print(f'HED模型加载失败: {e}')
        _cached_hed_net = None
        _cached_hed_paths = None
        return None

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
    """Sobel边缘检测"""
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.bitwise_or(sobel_x, sobel_y)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    sobel_edges = cv2.morphologyEx(sobel_edges, cv2.MORPH_CLOSE, kernel_close)
    return sobel_edges

def color_sobel_edge_detection(rgb_img):
    """彩色Sobel边缘检测"""
    def sobel_single_channel(channel):
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        return np.uint8(np.absolute(cv2.bitwise_or(sobel_x, sobel_y)))
    r_edges = sobel_single_channel(r)
    g_edges = sobel_single_channel(g)
    b_edges = sobel_single_channel(b)
    color_sobel_edges = cv2.bitwise_or(cv2.bitwise_or(r_edges, g_edges), b_edges)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    color_sobel_edges = cv2.morphologyEx(color_sobel_edges, cv2.MORPH_CLOSE, kernel_close)
    return color_sobel_edges

# 算法映射字典
EDGE_ALGORITHMS = {
    "Canny": canny_edge_detection,
    "color-canny": color_canny_edge_detection,
    "Sobel": sobel_edge_detection,
    "color-sobel": color_sobel_edge_detection,
}

# ------------------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------------------
def base64_to_image(base64_string):
    """将base64字符串转换为OpenCV图像（BGR格式）"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    img_np = np.array(img)
    if len(img_np.shape) == 2:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_bgr

def image_to_base64(img):
    """将OpenCV图像（BGR格式）转换为base64字符串"""
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def get_reference_edge(img):
    """生成参考边缘（使用 Canny）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reference_edge = cv2.Canny(gray, 150, 255)
    return reference_edge

def generate_output_dir_name(algorithm, blur, sobel_ksize=None, canny_low=None, canny_high=None, prefix=""):
    """根据算法和参数生成输出文件夹名称
    
    注意：
    - Canny算法不使用sobel_ksize参数
    - Sobel算法不使用canny_low和canny_high参数
    """
    # 清理算法名称（移除特殊字符，用于文件夹名）
    algo_clean = algorithm.replace(' ', '_').replace('/', '_').replace('彩色', 'color')
    
    # 构建参数部分
    params = []
    params.append(f"blur{blur}")
    
    # 根据算法类型添加相关参数
    # Canny算法（包括color-canny）：只使用canny参数，不使用sobel_ksize
    if "canny" in algorithm.lower():
        if canny_low is not None and canny_high is not None:
            params.append(f"canny{canny_low}_{canny_high}")
    # Sobel算法（包括color-sobel）：只使用sobel_ksize参数，不使用canny参数
    elif "sobel" in algorithm.lower():
        if sobel_ksize is not None:
            params.append(f"sobelK{sobel_ksize}")
    # Prewitt、HED、PiDiNet等算法：只使用blur参数
    # （深度学习算法如HED和PiDiNet不需要传统参数，但保留blur用于兼容性）
    
    # 组合文件夹名
    param_str = "_".join(params)
    if prefix:
        dir_name = f"{prefix}_{algo_clean}_{param_str}"
    else:
        dir_name = f"{algo_clean}_{param_str}"
    
    return dir_name

def load_bsds_ground_truth(image_name, H, W):
    """加载BSDS500数据集的ground truth标注"""
    try:
        from scipy import io
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        bsds_root = os.path.join(repo_dir, 'BSR_bsds500', 'BSDS500', 'data', 'groundTruth')
        
        for subset in ['val', 'test', 'train']:
            gt_dir = os.path.join(bsds_root, subset)
            if not os.path.exists(gt_dir):
                continue
            gt_path = os.path.join(gt_dir, f"{image_name}.mat")
            if os.path.exists(gt_path):
                try:
                    mat = io.loadmat(gt_path)
                    union = np.zeros((H, W), dtype=np.float32)
                    
                    # BSDS格式：groundTruth是一个cell数组，每个元素是一个struct
                    gt = mat.get('groundTruth', None)
                    if gt is None:
                        continue
                    
                    # 遍历所有标注
                    num_annotations = gt.shape[1] if len(gt.shape) > 1 else 1
                    for i in range(num_annotations):
                        entry = gt[0, i] if len(gt.shape) > 1 else gt
                        if isinstance(entry, np.ndarray) and entry.size > 0:
                            # 尝试不同的字段名
                            if 'Boundaries' in entry.dtype.names:
                                bmap = entry['Boundaries'][0, 0]
                            elif 'boundaries' in entry.dtype.names:
                                bmap = entry['boundaries'][0, 0]
                            else:
                                # 尝试直接访问
                                bmap = entry[0, 0]['Boundaries'][0, 0]
                            
                            if bmap.dtype != np.float32:
                                bmap = bmap.astype(np.float32)
                            
                            # 缩放到目标尺寸
                            if bmap.shape[0] != H or bmap.shape[1] != W:
                                bmap = cv2.resize(bmap, (W, H), interpolation=cv2.INTER_NEAREST)
                            
                            union = np.maximum(union, bmap)
                    
                    # 转换为0/255的uint8格式
                    ref = (union > 0.5).astype(np.uint8) * 255
                    return ref
                except Exception as e:
                    print(f'加载BSDS GT失败 {gt_path}: {e}')
                    continue
        return None
    except ImportError:
        print('scipy未安装，无法加载BSDS ground truth')
        return None
    except Exception as e:
        print(f'加载BSDS ground truth异常: {e}')
        return None

# ------------------------------------------------------------------------------
# 批量处理核心函数（模块化）
# ------------------------------------------------------------------------------
def compute_metrics_from_thresholds(thr_TP_sum, thr_FP_sum, thr_FN_sum, thresholds):
    """从阈值统计计算ODS和OIS指标"""
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
    
    return {
        'ods': {'threshold': ods_thr, 'precision': ods_prec, 'recall': ods_rec, 'f1': ods_f1},
        'precision_mean': precision_mean,
        'recall_mean': recall_mean
    }

def compute_ois_from_per_image(per_image_best):
    """从每张图片的最佳结果计算OIS指标"""
    ois_prec = np.mean([x["best_precision"] for x in per_image_best])
    ois_rec = np.mean([x["best_recall"] for x in per_image_best])
    ois_f1 = np.mean([x["best_f1"] for x in per_image_best])
    return {'precision': ois_prec, 'recall': ois_rec, 'f1': ois_f1}

def generate_pr_curve(precision_mean, recall_mean, algorithm, output_dir):
    """生成PR曲线并保存，返回base64编码的图片"""
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
    valid_mask = ~((recall_sorted < 0.05) & (precision_sorted > 0.95))
    if np.any(valid_mask):
        recall_sorted = recall_sorted[valid_mask]
        precision_sorted = precision_sorted[valid_mask]
    
    # 确保PR曲线以(1, precision_at_recall_1)结束
    if len(recall_sorted) > 0 and recall_sorted[-1] < 1.0 - 1e-6:
        recall_sorted = np.concatenate([recall_sorted, [1.0]])
        precision_sorted = np.concatenate([precision_sorted, [precision_sorted[-1]]])
    
    # 插值生成平滑曲线
    if len(recall_sorted) == 0 or recall_sorted.max() - recall_sorted.min() < 1e-6:
        recall_fine = recall_sorted
        precision_fine = precision_sorted
    else:
        recall_fine = np.linspace(recall_sorted.min(), recall_sorted.max(), 512)
        precision_fine = np.interp(recall_fine, recall_sorted, precision_sorted)
        # 对于PR曲线，应该使用单调递减的precision
        for i in range(len(precision_fine) - 2, -1, -1):
            precision_fine[i] = max(precision_fine[i], precision_fine[i + 1])
    
    # 绘制 PR 曲线
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall_fine, precision_fine, '-', linewidth=2, label='PR Curve')
    ax.plot(recall_sorted, precision_sorted, 'o', markersize=3, alpha=0.6)
    
    # 绘制F1-score等值线
    f1_values = [0.3, 0.5, 0.7]
    recall_f1 = np.linspace(0.01, 0.99, 100)
    for f1 in f1_values:
        precision_f1 = f1 * recall_f1 / (2 * recall_f1 - f1)
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
    
    pr_path = os.path.join(output_dir, f"PR_curve_{algorithm}.png")
    fig.tight_layout()
    fig.savefig(pr_path, dpi=100)
    plt.close(fig)
    
    # 读取 PR 曲线图片并转换为 base64
    with open(pr_path, 'rb') as f:
        pr_img_data = f.read()
    pr_base64 = base64.b64encode(pr_img_data).decode()
    pr_image_base64 = f"data:image/png;base64,{pr_base64}"
    
    return pr_image_base64

def save_batch_metrics_csv(per_image_best, ods_metrics, ois_metrics, algorithm, output_dir):
    """保存批量处理指标到CSV文件"""
    csv_path = os.path.join(output_dir, f"batch_metrics_{algorithm}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(["index", "best_threshold", "best_precision", "best_recall", "best_f1"])
        for row in per_image_best:
            writer.writerow([row["index"], row["best_threshold"], row["best_precision"], 
                           row["best_recall"], row["best_f1"]])
        writer.writerow([])
        writer.writerow(["ODS_threshold", ods_metrics['threshold']])
        writer.writerow(["ODS_precision", round(ods_metrics['precision'], 3)])
        writer.writerow(["ODS_recall", round(ods_metrics['recall'], 3)])
        writer.writerow(["ODS_f1", round(ods_metrics['f1'], 3)])
        writer.writerow(["OIS_precision", round(ois_metrics['precision'], 3)])
        writer.writerow(["OIS_recall", round(ois_metrics['recall'], 3)])
        writer.writerow(["OIS_f1", round(ois_metrics['f1'], 3)])
    return csv_path

def process_single_image_for_batch(img_bgr, algorithm, blur, sobel_ksize, canny_low, canny_high, 
                                   thresholds, reference=None, use_tolerance=True, max_dist=0.0075, use_thinning=True):
    """处理单张图片用于批量处理，返回最佳阈值和统计信息
    所有算法都使用原始图像尺寸处理，以获得最佳精度
    
    Args:
        use_tolerance: 是否使用边界匹配容差（类似BSDS500官方评估）
        max_dist: 最大匹配距离（图像对角线比例，默认0.0075）
        use_thinning: 是否对边界进行细化（默认True）
    """
    # 导入边界评估工具
    try:
        from boundary_evaluation import evaluate_boundary_with_tolerance
    except ImportError:
        # 如果导入失败，使用简单评估
        use_tolerance = False
    
    # 所有算法都使用原始图像尺寸
    img_for_processing = img_bgr.copy()
    original_h, original_w = img_for_processing.shape[:2]
    
    # reference也需要匹配原始尺寸
    if reference is None:
        reference = get_reference_edge(img_for_processing)
    else:
        if reference.shape != (original_h, original_w):
            reference = cv2.resize(reference, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
    edge_strength = compute_edge_strength(img_for_processing, algorithm, blur, sobel_ksize, canny_low, canny_high)
    
    best_f1 = -1.0
    best_prec = best_rec = best_thr = 0
    thr_TP = np.zeros(len(thresholds), dtype=np.float64)
    thr_FP = np.zeros(len(thresholds), dtype=np.float64)
    thr_FN = np.zeros(len(thresholds), dtype=np.float64)
    
    # 对每个阈值计算 TP/FP/FN
    for i, t in enumerate(thresholds):
        _, detected = cv2.threshold(edge_strength, t, 255, cv2.THRESH_BINARY)
        
        if use_tolerance:
            # 使用容差评估（类似BSDS500官方评估）
            TP, FP, FN = evaluate_boundary_with_tolerance(detected, reference, max_dist=max_dist, thin=use_thinning)
        else:
            # 使用严格的像素级匹配
            det_mask = (detected > 0)
            ref_mask = (reference > 0)
            TP = int(np.logical_and(det_mask, ref_mask).sum())
            FP = int(np.logical_and(det_mask, np.logical_not(ref_mask)).sum())
            FN = int(np.logical_and(np.logical_not(det_mask), ref_mask).sum())
        
        thr_TP[i] = TP
        thr_FP[i] = FP
        thr_FN[i] = FN
        
        # 计算单图指标
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_prec = prec
            best_rec = rec
            best_thr = t
    
    # 生成最佳阈值下的边缘图（保持原始尺寸）
    _, detected_final = cv2.threshold(edge_strength, best_thr, 255, cv2.THRESH_BINARY)
    
    # 保存时resize到标准尺寸（用于显示和保存）
    if detected_final.shape != (FRAME_HEIGHT, FRAME_WIDTH):
        detected_final_display = cv2.resize(detected_final, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_NEAREST)
    else:
        detected_final_display = detected_final
    
    return {
        'best_threshold': int(best_thr),
        'best_precision': round(best_prec, 3),
        'best_recall': round(best_rec, 3),
        'best_f1': round(best_f1, 3),
        'edge_image': detected_final_display,  # 返回resize后的用于显示
        'thr_TP': thr_TP,
        'thr_FP': thr_FP,
        'thr_FN': thr_FN
    }

# ------------------------------------------------------------------------------
# 边缘强度计算函数（支持所有算法，包括深度学习）
# ------------------------------------------------------------------------------
def compute_edge_strength(img, algorithm, blur_ksize, sobel_ksize, canny_low=100, canny_high=220):
    """生成单通道的边强度图（0-255 uint8），供阈值化用于 ODS/OIS 计算
    支持所有算法，包括经典算法和深度学习算法（HED、PiDiNet）
    
    所有算法都使用原始图像尺寸处理，以获得最佳精度
    """
    # 所有算法都使用原始图像尺寸
    img_for_processing = img.copy()
    edge = None
    
    if algorithm == "Sobel":
        gray = cv2.cvtColor(img_for_processing, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
        grad_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        grad_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        edge = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
    elif algorithm == "color-sobel":
        img_blur = cv2.GaussianBlur(img_for_processing, (blur_ksize, blur_ksize), 1)
        sobel_edges = []
        for i in range(3):
            grad_x = cv2.Sobel(img_blur[:, :, i], cv2.CV_64F, 1, 0, ksize=sobel_ksize)
            grad_y = cv2.Sobel(img_blur[:, :, i], cv2.CV_64F, 0, 1, ksize=sobel_ksize)
            sobel_edges.append(cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y)))
        edge = cv2.bitwise_or(sobel_edges[0], sobel_edges[1])
        edge = cv2.bitwise_or(edge, sobel_edges[2])
    elif algorithm == "Canny":
        # 对于批量处理和PR曲线生成，使用真正的Canny算法
        gray = cv2.cvtColor(img_for_processing, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
        
        # 使用真正的Canny算法计算梯度（Canny内部使用3x3 Sobel）
        gx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = cv2.magnitude(gx, gy)
        gradient_direction = np.arctan2(gy, gx) * 180 / np.pi
        
        # 实现非极大值抑制（NMS）
        h, w = gradient_magnitude.shape
        suppressed = gradient_magnitude.copy()
        gradient_direction = np.abs(gradient_direction)
        
        # 创建4个方向的掩码
        mask_0 = ((gradient_direction >= 0) & (gradient_direction < 22.5)) | \
                 ((gradient_direction >= 157.5) & (gradient_direction <= 180))
        mask_45 = (gradient_direction >= 22.5) & (gradient_direction < 67.5)
        mask_90 = (gradient_direction >= 67.5) & (gradient_direction < 112.5)
        mask_135 = (gradient_direction >= 112.5) & (gradient_direction < 157.5)
        
        # 对每个方向进行NMS
        mag_center = suppressed[1:-1, 1:-1]
        mag_left = suppressed[1:-1, :-2]
        mag_right = suppressed[1:-1, 2:]
        mask_0_inner = mask_0[1:-1, 1:-1]
        suppressed[1:-1, 1:-1] = np.where(
            mask_0_inner,
            np.where((mag_center >= mag_left) & (mag_center >= mag_right), mag_center, 0),
            suppressed[1:-1, 1:-1]
        )
        
        mag_diag1_45 = suppressed[:-2, 2:]
        mag_diag2_45 = suppressed[2:, :-2]
        mask_45_inner = mask_45[1:-1, 1:-1]
        suppressed[1:-1, 1:-1] = np.where(
            mask_45_inner,
            np.where((mag_center >= mag_diag1_45) & (mag_center >= mag_diag2_45), mag_center, 0),
            suppressed[1:-1, 1:-1]
        )
        
        mag_up = suppressed[:-2, 1:-1]
        mag_down = suppressed[2:, 1:-1]
        mask_90_inner = mask_90[1:-1, 1:-1]
        suppressed[1:-1, 1:-1] = np.where(
            mask_90_inner,
            np.where((mag_center >= mag_up) & (mag_center >= mag_down), mag_center, 0),
            suppressed[1:-1, 1:-1]
        )
        
        mag_diag1_135 = suppressed[:-2, :-2]
        mag_diag2_135 = suppressed[2:, 2:]
        mask_135_inner = mask_135[1:-1, 1:-1]
        suppressed[1:-1, 1:-1] = np.where(
            mask_135_inner,
            np.where((mag_center >= mag_diag1_135) & (mag_center >= mag_diag2_135), mag_center, 0),
            suppressed[1:-1, 1:-1]
        )
        
        suppressed[0, :] = 0
        suppressed[-1, :] = 0
        suppressed[:, 0] = 0
        suppressed[:, -1] = 0
        
        edge = cv2.convertScaleAbs(suppressed)
    elif algorithm == "color-canny":
        img_blur = cv2.GaussianBlur(img_for_processing, (blur_ksize, blur_ksize), 1)
        canny_edges = []
        for i in range(3):
            gx = cv2.Sobel(img_blur[:, :, i], cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(img_blur[:, :, i], cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = cv2.magnitude(gx, gy)
            gradient_direction = np.arctan2(gy, gx) * 180 / np.pi
            
            h, w = gradient_magnitude.shape
            suppressed = gradient_magnitude.copy()
            gradient_direction = np.abs(gradient_direction)
            
            mask_0 = ((gradient_direction >= 0) & (gradient_direction < 22.5)) | \
                     ((gradient_direction >= 157.5) & (gradient_direction <= 180))
            mask_45 = (gradient_direction >= 22.5) & (gradient_direction < 67.5)
            mask_90 = (gradient_direction >= 67.5) & (gradient_direction < 112.5)
            mask_135 = (gradient_direction >= 112.5) & (gradient_direction < 157.5)
            
            mag_center = suppressed[1:-1, 1:-1]
            
            if np.any(mask_0[1:-1, 1:-1]):
                mag_left = suppressed[1:-1, :-2]
                mag_right = suppressed[1:-1, 2:]
                mask_0_inner = mask_0[1:-1, 1:-1]
                suppressed[1:-1, 1:-1] = np.where(
                    mask_0_inner,
                    np.where((mag_center >= mag_left) & (mag_center >= mag_right), mag_center, 0),
                    suppressed[1:-1, 1:-1]
                )
            
            if np.any(mask_45[1:-1, 1:-1]):
                mag_diag1 = suppressed[:-2, 2:]
                mag_diag2 = suppressed[2:, :-2]
                mask_45_inner = mask_45[1:-1, 1:-1]
                suppressed[1:-1, 1:-1] = np.where(
                    mask_45_inner,
                    np.where((mag_center >= mag_diag1) & (mag_center >= mag_diag2), mag_center, 0),
                    suppressed[1:-1, 1:-1]
                )
            
            if np.any(mask_90[1:-1, 1:-1]):
                mag_up = suppressed[:-2, 1:-1]
                mag_down = suppressed[2:, 1:-1]
                mask_90_inner = mask_90[1:-1, 1:-1]
                suppressed[1:-1, 1:-1] = np.where(
                    mask_90_inner,
                    np.where((mag_center >= mag_up) & (mag_center >= mag_down), mag_center, 0),
                    suppressed[1:-1, 1:-1]
                )
            
            if np.any(mask_135[1:-1, 1:-1]):
                mag_diag1 = suppressed[:-2, :-2]
                mag_diag2 = suppressed[2:, 2:]
                mask_135_inner = mask_135[1:-1, 1:-1]
                suppressed[1:-1, 1:-1] = np.where(
                    mask_135_inner,
                    np.where((mag_center >= mag_diag1) & (mag_center >= mag_diag2), mag_center, 0),
                    suppressed[1:-1, 1:-1]
                )
            
            suppressed[0, :] = 0
            suppressed[-1, :] = 0
            suppressed[:, 0] = 0
            suppressed[:, -1] = 0
            
            canny_edges.append(cv2.convertScaleAbs(suppressed))
        
        edge = cv2.bitwise_or(canny_edges[0], canny_edges[1])
        edge = cv2.bitwise_or(edge, canny_edges[2])
    elif algorithm == "Prewitt":
        gray = cv2.cvtColor(img_for_processing, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        grad_x = cv2.filter2D(gray_blur, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(gray_blur, cv2.CV_64F, kernel_y)
        edge = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
    elif algorithm == "HED":
        # HED算法：使用深度学习模型
        try:
            repo_dir = os.path.dirname(os.path.abspath(__file__))
        except Exception:
            repo_dir = os.getcwd()
        prototxt_path = os.path.join(repo_dir, 'models', 'hed_deploy.prototxt')
        cand1 = os.path.join(repo_dir, 'models', 'hed_pretrained_bsds.caffemodel')
        cand2 = os.path.join(repo_dir, 'models', 'hed_bsds.caffemodel')
        caffemodel_path = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else None)

        # 使用缓存的HED模型（避免每次重新加载）
        net = get_hed_net()
        if net is not None:
            try:
                blob = cv2.dnn.blobFromImage(img_for_processing, scalefactor=1.0, size=(500, 500),
                                            mean=(104.00698793, 116.66876762, 122.67891434),
                                            swapRB=False, crop=False)
                net.setInput(blob)
                out = net.forward()
                out_map = out[0, 0, :, :]
                out_map = (out_map * 255.0).clip(0, 255).astype('uint8')
                # HED Caffe输出是500x500，需要resize回原始尺寸
                original_h, original_w = img_for_processing.shape[:2]
                edge = cv2.resize(out_map, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                    print(f'HED (Caffe/OpenCV) 推理失败：{e}')
                    import traceback
                    traceback.print_exc()
                    # 回退到PyTorch实现（但PyTorch模型可能不存在）
                    try:
                        from hed import run_hed
                        print('警告: 回退到PyTorch HED实现（可能使用随机初始化的模型）')
                        hed_map = run_hed(img_for_processing, model=None, device='cpu')
                        # PyTorch HED输出保持原始尺寸
                        edge = hed_map.astype(np.uint8)
                    except Exception as e2:
                        print(f'HED (PyTorch) 不可用或失败：{e2}')
                        original_h, original_w = img_for_processing.shape[:2]
                        edge = np.zeros((original_h, original_w), dtype=np.uint8)
        else:
            # Caffe模型未找到，尝试PyTorch实现（但可能没有权重）
            print('警告: HED Caffe模型未找到，尝试PyTorch实现（可能使用随机初始化的模型）')
            try:
                from hed import run_hed
                hed_map = run_hed(img_for_processing, model=None, device='cpu')
                # HED输出保持原始尺寸
                edge = hed_map.astype(np.uint8)
            except Exception as e:
                print(f'HED (PyTorch) 不可用或失败：{e}')
                original_h, original_w = img_for_processing.shape[:2]
                edge = np.zeros((original_h, original_w), dtype=np.uint8)
    elif algorithm == "PiDiNet":
        # PiDiNet算法：使用深度学习模型
        # 注意：PiDiNet应该使用原始图像尺寸或更大的尺寸，而不是先resize到640x480
        # 为了保持一致性，我们先用原始图像运行PiDiNet，然后resize结果
        try:
            from pidinet import run_pidinet
            # 使用原始图像尺寸运行PiDiNet（保持更好的精度）
            # PiDiNet在BSDS500上训练时通常使用原始图像尺寸或较大的尺寸
            original_h, original_w = img_for_processing.shape[:2]
            # PiDiNet可以处理任意尺寸，但为了性能，如果图像太大（>1024），可以适当缩小
            max_size = 1024
            if original_h > max_size or original_w > max_size:
                scale = min(max_size / original_h, max_size / original_w)
                new_h, new_w = int(original_h * scale), int(original_w * scale)
                img_for_pidinet = cv2.resize(img_for_processing, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                # 图像尺寸合适，直接使用
                img_for_pidinet = img_for_processing.copy()
            
            pidinet_map = run_pidinet(img_for_pidinet, model=None, device='cpu')
            # 确保输出是2D数组
            if len(pidinet_map.shape) > 2:
                pidinet_map = pidinet_map.squeeze()
            if len(pidinet_map.shape) != 2:
                raise ValueError(f"PiDiNet output shape is invalid: {pidinet_map.shape}")
            
            # 如果输入图像被resize了，需要将输出resize回原始尺寸
            if img_for_pidinet.shape[:2] != (original_h, original_w):
                pidinet_map = cv2.resize(pidinet_map, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            
            # 保持原始尺寸
            edge = pidinet_map
            # 确保是uint8类型和2D数组
            edge = edge.astype(np.uint8)
            if len(edge.shape) != 2:
                edge = edge.squeeze()
        except Exception as e:
            print(f'PiDiNet 不可用或失败：{e}')
            import traceback
            traceback.print_exc()
            original_h, original_w = img_for_processing.shape[:2]
            edge = np.zeros((original_h, original_w), dtype=np.uint8)
    else:
        # 默认使用梯度强度图（类似Canny）
        gray = cv2.cvtColor(img_for_processing, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
        gx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
        edge = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
    
    if edge is None:
        original_h, original_w = img_for_processing.shape[:2]
        edge = np.zeros((original_h, original_w), dtype=np.uint8)
    return edge

# ------------------------------------------------------------------------------
# API 路由
# ------------------------------------------------------------------------------
@app.route('/api/health', methods=['GET'])
def health_check():
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
        if "color" in algorithm.lower() or "彩色" in algorithm:
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
            "classification": classification_result,
            "images": {
                "marked": marked_img_base64,
                "edge": edge_img_base64
            },
            "features": {
                "area": area,
                "aspect_ratio": round(aspect_ratio, 2),
                "rectangularity": round(rectangularity, 2),
                "main_color": main_color
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/edge-detect', methods=['POST'])
def edge_detect():
    """
    边缘检测接口：支持多种算法
    支持经典算法：Sobel, Canny, Prewitt, color-sobel, color-canny
    支持深度学习算法：HED, PiDiNet
    """
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "缺少图片数据"}), 400
        
        algorithm = data.get('algorithm', 'Canny')
        blur = int(data.get('blur', 7))
        sobel_ksize = int(data.get('sobel_ksize', 3))
        canny_low = int(data.get('canny_low', 100))
        canny_high = int(data.get('canny_high', 220))
        dilate_ksize = int(data.get('dilate_ksize', 1))
        
        # 参数规范化
        if blur % 2 == 0:
            blur += 1
        if blur < 1:
            blur = 1
        if sobel_ksize % 2 == 0 or sobel_ksize < 1:
            sobel_ksize = 3
        
        img_bgr = base64_to_image(data['image'])
        # 所有算法都使用原始图像尺寸处理
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        edge = None
        
        if algorithm == 'Sobel':
            gray_blur = cv2.GaussianBlur(img_gray, (blur, blur), 1)
            grad_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
            grad_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
            edge = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
            _, edge = cv2.threshold(edge, 50, 255, cv2.THRESH_BINARY)
        elif algorithm == 'color-sobel':
            img_blur = cv2.GaussianBlur(img_bgr, (blur, blur), 1)
            sobel_edges = []
            for i in range(3):
                grad_x = cv2.Sobel(img_blur[:, :, i], cv2.CV_64F, 1, 0, ksize=sobel_ksize)
                grad_y = cv2.Sobel(img_blur[:, :, i], cv2.CV_64F, 0, 1, ksize=sobel_ksize)
                sobel_edges.append(cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y)))
            edge = cv2.bitwise_or(sobel_edges[0], sobel_edges[1])
            edge = cv2.bitwise_or(edge, sobel_edges[2])
            _, edge = cv2.threshold(edge, 50, 255, cv2.THRESH_BINARY)
        elif algorithm == 'Canny':
            gray_blur = cv2.GaussianBlur(img_gray, (blur, blur), 1)
            edge = cv2.Canny(gray_blur, canny_low, canny_high)
        elif algorithm == 'color-canny':
            img_blur = cv2.GaussianBlur(img_bgr, (blur, blur), 1)
            canny_edges = []
            for i in range(3):
                canny_edges.append(cv2.Canny(img_blur[:, :, i], canny_low, canny_high))
            edge = cv2.bitwise_or(canny_edges[0], canny_edges[1])
            edge = cv2.bitwise_or(edge, canny_edges[2])
        elif algorithm == 'Prewitt':
            gray_blur = cv2.GaussianBlur(img_gray, (blur, blur), 1)
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
            grad_x = cv2.filter2D(gray_blur, cv2.CV_64F, kernel_x)
            grad_y = cv2.filter2D(gray_blur, cv2.CV_64F, kernel_y)
            edge = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
            _, edge = cv2.threshold(edge, 50, 255, cv2.THRESH_BINARY)
        elif algorithm == 'HED':
            # HED算法：使用深度学习模型
            try:
                repo_dir = os.path.dirname(os.path.abspath(__file__))
            except Exception:
                repo_dir = os.getcwd()
            prototxt_path = os.path.join(repo_dir, 'models', 'hed_deploy.prototxt')
            cand1 = os.path.join(repo_dir, 'models', 'hed_pretrained_bsds.caffemodel')
            cand2 = os.path.join(repo_dir, 'models', 'hed_bsds.caffemodel')
            caffemodel_path = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else None)

            if os.path.exists(prototxt_path) and caffemodel_path and os.path.exists(caffemodel_path):
                try:
                    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
                    blob = cv2.dnn.blobFromImage(img_bgr, scalefactor=1.0, size=(500, 500),
                                                mean=(104.00698793, 116.66876762, 122.67891434),
                                                swapRB=False, crop=False)
                    net.setInput(blob)
                    out = net.forward()
                    out_map = out[0, 0, :, :]
                    out_map = (out_map * 255.0).clip(0, 255).astype('uint8')
                    # HED Caffe输出是500x500，需要resize回原始尺寸
                    original_h, original_w = img_bgr.shape[:2]
                    edge = cv2.resize(out_map, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
                except Exception as e:
                    print(f'HED (Caffe/OpenCV) 推理失败：{e}')
                    # 回退到PyTorch实现
                    try:
                        from hed import run_hed
                        hed_map = run_hed(img_bgr, model=None, device='cpu')
                        edge = hed_map.astype(np.uint8)
                    except Exception as e2:
                        print(f'HED (PyTorch) 不可用或失败：{e2}')
                        original_h, original_w = img_bgr.shape[:2]
                        edge = np.zeros((original_h, original_w), dtype=np.uint8)
            else:
                # 尝试PyTorch实现
                try:
                    from hed import run_hed
                    hed_map = run_hed(img_bgr, model=None, device='cpu')
                    edge = hed_map.astype(np.uint8)
                except Exception as e:
                    print(f'HED (PyTorch) 不可用或失败：{e}')
                    original_h, original_w = img_bgr.shape[:2]
                    edge = np.zeros((original_h, original_w), dtype=np.uint8)
        elif algorithm == 'PiDiNet':
            # PiDiNet算法：使用深度学习模型
            try:
                from pidinet import run_pidinet
                pidinet_map = run_pidinet(img_bgr, model=None, device='cpu')
                # 确保输出是2D数组
                if len(pidinet_map.shape) > 2:
                    pidinet_map = pidinet_map.squeeze()
                if len(pidinet_map.shape) != 2:
                    raise ValueError(f"PiDiNet output shape is invalid: {pidinet_map.shape}")
                # PiDiNet输出保持原始尺寸
                edge = pidinet_map.astype(np.uint8)
                if len(edge.shape) != 2:
                    edge = edge.squeeze()
                # PiDiNet输出是概率值映射到0-255，需要进行阈值化得到黑白边缘图
                # 使用固定阈值（127，相当于概率0.5）来二值化，将边缘概率转换为黑白图像
                # 白色(255)表示边缘，黑色(0)表示非边缘
                _, edge = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)
            except Exception as e:
                print(f'PiDiNet 不可用或失败：{e}')
                import traceback
                traceback.print_exc()
                original_h, original_w = img_bgr.shape[:2]
                edge = np.zeros((original_h, original_w), dtype=np.uint8)
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
        # 确保 edge 和 reference 大小一致（应该已经是原始尺寸）
        if edge.shape != reference.shape:
            reference = cv2.resize(reference, (edge.shape[1], edge.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # 保存原始尺寸的edge用于指标计算
        edge_for_metrics = edge.copy()
        
        # 为了前端显示，resize到标准尺寸
        if edge.shape != (FRAME_HEIGHT, FRAME_WIDTH):
            edge = cv2.resize(edge, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_NEAREST)
        # 使用原始尺寸计算指标
        _, detected = cv2.threshold(edge_for_metrics, 127, 255, cv2.THRESH_BINARY)
        _, ref_binary = cv2.threshold(reference, 127, 255, cv2.THRESH_BINARY)
        
        # 确保两个数组都是2D且大小一致
        if len(detected.shape) != 2 or len(ref_binary.shape) != 2:
            detected = detected.squeeze()
            ref_binary = ref_binary.squeeze()
        if detected.shape != ref_binary.shape:
            ref_binary = cv2.resize(ref_binary, (detected.shape[1], detected.shape[0]), interpolation=cv2.INTER_NEAREST)
        
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

@app.route('/api/batch-process', methods=['POST'])
def batch_process():
    """批量处理图片，计算 ODS/OIS 指标和 PR 曲线
    支持所有算法，包括经典算法和深度学习算法（HED、PiDiNet）
    """
    try:
        data = request.get_json()
        if 'images' not in data or not isinstance(data['images'], list):
            return jsonify({"error": "缺少图片数据列表"}), 400
        
        algorithm = data.get('algorithm', 'Canny')
        blur = int(data.get('blur', 7))
        sobel_ksize = int(data.get('sobel_ksize', 3))
        canny_low = int(data.get('canny_low', 100))
        canny_high = int(data.get('canny_high', 220))
        
        # 评估参数（BSDS500标准）
        use_tolerance = data.get('use_tolerance', True)
        max_dist = float(data.get('max_dist', 0.0075))
        use_thinning = data.get('use_thinning', True)
        
        # 参数规范化
        if blur % 2 == 0:
            blur += 1
        if blur < 1:
            blur = 1
        if sobel_ksize % 2 == 0 or sobel_ksize < 1:
            sobel_ksize = 3
        
        # 准备输出目录（使用算法名称+参数）
        dir_name = generate_output_dir_name(algorithm, blur, sobel_ksize, canny_low, canny_high)
        run_output_dir = os.path.join(OUTPUT_FOLDER, dir_name)
        # 如果目录已存在，添加序号
        counter = 1
        original_dir = run_output_dir
        while os.path.exists(run_output_dir):
            run_output_dir = f"{original_dir}_{counter}"
            counter += 1
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
                result = process_single_image_for_batch(
                    img_bgr, algorithm, blur, sobel_ksize, canny_low, canny_high,
                    thresholds, reference=None, 
                    use_tolerance=use_tolerance, max_dist=max_dist, use_thinning=use_thinning
                )
                
                # 累加阈值统计
                thr_TP_sum += result['thr_TP']
                thr_FP_sum += result['thr_FP']
                thr_FN_sum += result['thr_FN']
                
                # 记录单图最佳结果
                per_image_best.append({
                    "index": idx,
                    "best_threshold": result['best_threshold'],
                    "best_precision": result['best_precision'],
                    "best_recall": result['best_recall'],
                    "best_f1": result['best_f1']
                })
                
                # 保存最佳阈值下的边缘图
                edge_save_name = f"image_{idx}_edge.png"
                edge_save_path = os.path.join(run_output_dir, edge_save_name)
                cv2.imwrite(edge_save_path, result['edge_image'])
                
                processed_images.append({
                    "index": idx,
                    "edge_image": image_to_base64(result['edge_image']),
                    "filename": edge_save_name
                })
            except Exception as e:
                print(f'处理图片 {idx} 失败：', e)
                continue
        
        if len(per_image_best) == 0:
            return jsonify({"error": "未成功处理任何图片"}), 400
        
        # 计算指标
        metrics_result = compute_metrics_from_thresholds(thr_TP_sum, thr_FP_sum, thr_FN_sum, thresholds)
        ois_metrics = compute_ois_from_per_image(per_image_best)
        
        # 生成PR曲线
        pr_image_base64 = generate_pr_curve(
            metrics_result['precision_mean'], 
            metrics_result['recall_mean'], 
            algorithm, 
            run_output_dir
        )
        
        # 保存CSV
        save_batch_metrics_csv(
            per_image_best, 
            metrics_result['ods'], 
            ois_metrics, 
            algorithm, 
            run_output_dir
        )
        
        return jsonify({
            "success": True,
            "metrics": {
                "ods": {
                    "threshold": metrics_result['ods']['threshold'],
                    "precision": round(metrics_result['ods']['precision'], 3),
                    "recall": round(metrics_result['ods']['recall'], 3),
                    "f1": round(metrics_result['ods']['f1'], 3)
                },
                "ois": {
                    "precision": round(ois_metrics['precision'], 3),
                    "recall": round(ois_metrics['recall'], 3),
                    "f1": round(ois_metrics['f1'], 3)
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

# ------------------------------------------------------------------------------
# 车辆识别相关函数
# ------------------------------------------------------------------------------
def extract_vehicle_features(rgb_img, edge_img):
    """提取车辆关键特征：轮廓面积、长宽比、矩形度、主色调 + 外接矩形坐标（用于标记）"""
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
    """车辆分类规则：多特征组合判断（使用可调节矩形度阈值）"""
    r, g, b = main_color
    total_brightness = (r + g + b) / 3
    
    if area < 5000:
        return "未识别（轮廓过小，非车辆）"
    if total_brightness < 40:
        return "未识别（图片过暗，无法判断）"
    if rectangularity < min_rectangularity:
        return "未识别（矩形度不足，形状不匹配）"
    if aspect_ratio < 0.8 or aspect_ratio > 3.5:
        return "未识别（长宽比异常，非车辆比例）"
    
    return "识别为：车辆"

# ------------------------------------------------------------------------------
# 添加 HED 支持到 compute_edge_strength
# ------------------------------------------------------------------------------
# （已在 compute_edge_strength 函数中添加）

@app.route('/api/bsds500-quick-test', methods=['POST'])
def bsds500_quick_test():
    """BSDS500快速测试：自动从BSDS500数据集读取图片并使用ground truth进行评估"""
    try:
        data = request.get_json()
        algorithm = data.get('algorithm', 'Canny')
        blur = int(data.get('blur', 7))
        sobel_ksize = int(data.get('sobel_ksize', 3))
        canny_low = int(data.get('canny_low', 100))
        canny_high = int(data.get('canny_high', 220))
        subset = data.get('subset', 'val')
        
        # 评估参数（BSDS500标准）
        use_tolerance = data.get('use_tolerance', True)
        max_dist = float(data.get('max_dist', 0.0075))
        use_thinning = data.get('use_thinning', True)
        
        # 参数规范化
        if blur % 2 == 0:
            blur += 1
        if blur < 1:
            blur = 1
        if sobel_ksize % 2 == 0 or sobel_ksize < 1:
            sobel_ksize = 3
        
        # BSDS500数据集路径
        bsds_root = os.path.join('BSR_bsds500', 'BSDS500', 'data')
        images_dir = os.path.join(bsds_root, 'images', subset)
        
        if not os.path.exists(images_dir):
            return jsonify({"error": f"BSDS500数据集路径不存在: {images_dir}"}), 400
        
        # 获取所有图片文件
        import glob
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        
        if len(image_files) == 0:
            return jsonify({"error": f"在{images_dir}中未找到图片文件"}), 400
        
        # 准备输出目录
        dir_name = generate_output_dir_name(algorithm, blur, sobel_ksize, canny_low, canny_high, prefix=f"BSDS500_{subset}")
        run_output_dir = os.path.join(OUTPUT_FOLDER, dir_name)
        counter = 1
        original_dir = run_output_dir
        while os.path.exists(run_output_dir):
            run_output_dir = f"{original_dir}_{counter}"
            counter += 1
        os.makedirs(run_output_dir, exist_ok=True)
        
        # 阈值列表
        thresholds = list(range(0, 256, 1))
        thr_TP_sum = np.zeros(len(thresholds), dtype=np.float64)
        thr_FP_sum = np.zeros(len(thresholds), dtype=np.float64)
        thr_FN_sum = np.zeros(len(thresholds), dtype=np.float64)
        
        per_image_best = []
        processed_images = []
        
        # 处理每张图片
        for idx, img_path in enumerate(image_files):
            try:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    continue
                
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                reference = load_bsds_ground_truth(img_name, FRAME_HEIGHT, FRAME_WIDTH)
                if reference is None:
                    reference = get_reference_edge(img_bgr)
                
                # 获取评估参数
                use_tolerance = data.get('use_tolerance', True)
                max_dist = float(data.get('max_dist', 0.0075))
                use_thinning = data.get('use_thinning', True)
                
                result = process_single_image_for_batch(
                    img_bgr, algorithm, blur, sobel_ksize, canny_low, canny_high,
                    thresholds, reference=reference,
                    use_tolerance=use_tolerance, max_dist=max_dist, use_thinning=use_thinning
                )
                
                thr_TP_sum += result['thr_TP']
                thr_FP_sum += result['thr_FP']
                thr_FN_sum += result['thr_FN']
                
                per_image_best.append({
                    "index": idx,
                    "filename": os.path.basename(img_path),
                    "best_threshold": result['best_threshold'],
                    "best_precision": result['best_precision'],
                    "best_recall": result['best_recall'],
                    "best_f1": result['best_f1']
                })
                
                edge_save_name = f"{img_name}_edge.png"
                edge_save_path = os.path.join(run_output_dir, edge_save_name)
                cv2.imwrite(edge_save_path, result['edge_image'])
                
                processed_images.append({
                    "index": idx,
                    "edge_image": image_to_base64(result['edge_image']),
                    "filename": edge_save_name
                })
            except Exception as e:
                import traceback
                error_msg = f'处理图片 {img_path} 失败：{str(e)}'
                print(error_msg)
                traceback.print_exc()  # 打印完整堆栈跟踪以便调试
                continue
        
        if len(per_image_best) == 0:
            return jsonify({"error": "未成功处理任何图片"}), 400
        
        # 计算指标
        metrics_result = compute_metrics_from_thresholds(thr_TP_sum, thr_FP_sum, thr_FN_sum, thresholds)
        ois_metrics = compute_ois_from_per_image(per_image_best)
        
        # 生成PR曲线
        pr_image_base64 = generate_pr_curve(
            metrics_result['precision_mean'], 
            metrics_result['recall_mean'], 
            algorithm, 
            run_output_dir
        )
        
        # 保存CSV
        save_batch_metrics_csv(
            per_image_best, 
            metrics_result['ods'], 
            ois_metrics, 
            algorithm, 
            run_output_dir
        )
        
        return jsonify({
            "success": True,
            "metrics": {
                "ods": {
                    "threshold": metrics_result['ods']['threshold'],
                    "precision": round(metrics_result['ods']['precision'], 3),
                    "recall": round(metrics_result['ods']['recall'], 3),
                    "f1": round(metrics_result['ods']['f1'], 3)
                },
                "ois": {
                    "precision": round(ois_metrics['precision'], 3),
                    "recall": round(ois_metrics['recall'], 3),
                    "f1": round(ois_metrics['f1'], 3)
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

@app.route('/api/git-log', methods=['GET'])
def git_log():
    """获取Git提交日志"""
    try:
        import subprocess
        limit = int(request.args.get('limit', 50))
        
        # 查找Git仓库根目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        git_root = None
        depth = 0
        max_depth = 10
        current_dir = script_dir
        
        while depth < max_depth:
            if os.path.exists(os.path.join(current_dir, '.git')):
                git_root = current_dir
                break
            parent = os.path.dirname(current_dir)
            if parent == current_dir:
                break
            current_dir = parent
            depth += 1
        
        if not git_root:
            return jsonify({
                "success": False,
                "error": f"未找到Git仓库（从 {script_dir} 向上查找了 {depth} 层）"
            }), 400
        
        # 检查是否为空仓库
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--verify', 'HEAD'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=git_root,
                timeout=5
            )
            if result.returncode != 0:
                return jsonify({
                    "success": False,
                    "error": "Git仓库为空（没有提交记录）"
                }), 400
        except Exception:
            pass
        
        # 获取Git日志
        try:
            cmd = [
                'git', 'log',
                f'--max-count={limit}',
                '--pretty=format:%H|||%h|||%an|||%ai|||%s|||%b',
                '--name-only',
                '--'
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=git_root,
                timeout=10
            )
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else '未知错误'
                return jsonify({
                    "success": False,
                    "error": f"Git命令执行失败 (返回码: {result.returncode}): {error_msg}"
                }), 500
            
            output = result.stdout
            if output is None or not output.strip():
                return jsonify({
                    "success": False,
                    "error": f"Git命令未返回任何输出（工作目录: {git_root}，返回码: {result.returncode}，stderr: {result.stderr[:200] if result.stderr else '无'}）"
                }), 500
            
            # 解析输出
            commits = []
            lines = output.strip().split('\n')
            i = 0
            while i < len(lines):
                if '|||' in lines[i]:
                    parts = lines[i].split('|||')
                    if len(parts) >= 5:
                        commit = {
                            'hash': parts[0],
                            'short_hash': parts[1],
                            'author': parts[2],
                            'date': parts[3],
                            'message': parts[4],
                            'body': parts[5] if len(parts) > 5 else '',
                            'files': []
                        }
                        i += 1
                        while i < len(lines) and lines[i] and '|||' not in lines[i]:
                            if lines[i].strip():
                                commit['files'].append(lines[i].strip())
                            i += 1
                        commits.append(commit)
                    else:
                        i += 1
                else:
                    i += 1
            
            return jsonify({
                "success": True,
                "commits": commits,
                "count": len(commits)
            })
        except FileNotFoundError:
            return jsonify({
                "success": False,
                "error": "Git未安装或不在系统PATH中"
            }), 500
        except subprocess.TimeoutExpired:
            return jsonify({
                "success": False,
                "error": "Git命令执行超时"
            }), 500
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"获取Git日志失败: {str(e)}"
            }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
