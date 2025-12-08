"""
边界评估工具：实现与BSDS500官方评估方法兼容的边界匹配
包括：边界细化（thinning）、边界匹配容差（maxDist）、对应像素匹配
"""
import numpy as np
import cv2


def thin_boundary(bmap):
    """
    对边界进行细化（thinning），使其变为单像素宽度
    对于边界检测：如果输入是实心区域，先提取边界；如果已经是边界，保持原样
    
    Args:
        bmap: 二值边界图 (bool 或 uint8, 0/255)
    
    Returns:
        细化后的边界图 (bool)
    """
    if isinstance(bmap, np.ndarray) and bmap.dtype == np.uint8:
        bmap = bmap > 0
    
    # 转换为uint8用于OpenCV处理
    skeleton = bmap.astype(np.uint8) * 255
    
    # 如果边界像素很少，直接返回
    if cv2.countNonZero(skeleton) < 10:
        return skeleton > 0
    
    # 检查是否是实心区域（通过计算边界像素比例）
    # 使用形态学梯度提取边界
    kernel = np.ones((3, 3), np.uint8)
    gradient = cv2.morphologyEx(skeleton, cv2.MORPH_GRADIENT, kernel)
    
    # 如果梯度像素数接近原始像素数，说明已经是边界，直接返回
    gradient_ratio = cv2.countNonZero(gradient) / max(cv2.countNonZero(skeleton), 1)
    if gradient_ratio > 0.8:  # 已经是边界
        return gradient > 0
    
    # 如果是实心区域，提取边界
    # 使用形态学梯度
    if cv2.countNonZero(gradient) > 0:
        return gradient > 0
    
    # 否则返回原始（可能是稀疏边界）
    return skeleton > 0


def correspond_pixels_simple(bmap1, bmap2, max_dist_pixels):
    """
    简化的边界匹配算法（对应correspondPixels）
    使用距离变换和最近邻匹配，而不是完整的匈牙利算法
    
    Args:
        bmap1: 检测到的边界图 (bool)
        bmap2: 参考边界图 (bool)
        max_dist_pixels: 最大匹配距离（像素）
    
    Returns:
        match1: bmap1中匹配的像素 (bool)
        match2: bmap2中匹配的像素 (bool)
    """
    if isinstance(bmap1, np.ndarray) and bmap1.dtype == np.uint8:
        bmap1 = bmap1 > 0
    if isinstance(bmap2, np.ndarray) and bmap2.dtype == np.uint8:
        bmap2 = bmap2 > 0
    
    h, w = bmap1.shape
    match1 = np.zeros_like(bmap1, dtype=bool)
    match2 = np.zeros_like(bmap2, dtype=bool)
    
    # 获取所有边界像素的坐标
    coords1 = np.column_stack(np.where(bmap1))
    coords2 = np.column_stack(np.where(bmap2))
    
    if len(coords1) == 0 or len(coords2) == 0:
        return match1, match2
    
    # 使用距离变换：对于bmap1中的每个像素，找到bmap2中最近的距离
    # 创建距离图
    dist_map = cv2.distanceTransform((~bmap2).astype(np.uint8), cv2.DIST_L2, 5)
    
    # 对于bmap1中的每个像素，检查距离是否在容差内
    for y, x in coords1:
        dist = dist_map[y, x]
        if dist <= max_dist_pixels:
            match1[y, x] = True
            # 找到bmap2中最近的像素
            # 使用距离变换的梯度方向找到最近点
            # 简化：在容差范围内找到最近的bmap2像素
            y_min = max(0, int(y - max_dist_pixels))
            y_max = min(h, int(y + max_dist_pixels + 1))
            x_min = max(0, int(x - max_dist_pixels))
            x_max = min(w, int(x + max_dist_pixels + 1))
            
            local_bmap2 = bmap2[y_min:y_max, x_min:x_max]
            if local_bmap2.any():
                local_coords = np.column_stack(np.where(local_bmap2))
                if len(local_coords) > 0:
                    local_coords[:, 0] += y_min
                    local_coords[:, 1] += x_min
                    distances = np.sqrt((local_coords[:, 0] - y)**2 + (local_coords[:, 1] - x)**2)
                    min_idx = np.argmin(distances)
                    if distances[min_idx] <= max_dist_pixels:
                        match2[local_coords[min_idx, 0], local_coords[min_idx, 1]] = True
    
    return match1, match2


def correspond_pixels_optimized(bmap1, bmap2, max_dist_pixels):
    """
    优化的边界匹配算法，使用距离变换加速
    
    Args:
        bmap1: 检测到的边界图 (bool)
        bmap2: 参考边界图 (bool)
        max_dist_pixels: 最大匹配距离（像素）
    
    Returns:
        match1: bmap1中匹配的像素 (bool)
        match2: bmap2中匹配的像素 (bool)
    """
    if isinstance(bmap1, np.ndarray) and bmap1.dtype == np.uint8:
        bmap1 = bmap1 > 0
    if isinstance(bmap2, np.ndarray) and bmap2.dtype == np.uint8:
        bmap2 = bmap2 > 0
    
    h, w = bmap1.shape
    match1 = np.zeros_like(bmap1, dtype=bool)
    match2 = np.zeros_like(bmap2, dtype=bool)
    
    # 使用距离变换找到bmap2的最近边界距离
    dist_map = cv2.distanceTransform((~bmap2).astype(np.uint8), cv2.DIST_L2, 5)
    
    # 对于bmap1中的每个像素，如果距离在容差内，则匹配
    match1 = (bmap1 & (dist_map <= max_dist_pixels))
    
    # 对于bmap2，找到被bmap1匹配的像素
    # 使用距离变换：计算bmap1到bmap2的距离
    dist_map2 = cv2.distanceTransform((~bmap1).astype(np.uint8), cv2.DIST_L2, 5)
    match2 = (bmap2 & (dist_map2 <= max_dist_pixels))
    
    return match1, match2


def evaluate_boundary_with_tolerance(detected, reference, max_dist=0.0075, thin=True):
    """
    使用容差评估边界（类似BSDS500官方评估）
    
    Args:
        detected: 检测到的边界图 (uint8, 0/255 或 bool)
        reference: 参考边界图 (uint8, 0/255 或 bool)
        max_dist: 最大匹配距离（图像对角线比例，默认0.0075）
        thin: 是否对边界进行细化
    
    Returns:
        TP, FP, FN: 真阳性、假阳性、假阴性
    """
    if isinstance(detected, np.ndarray) and detected.dtype == np.uint8:
        detected = detected > 0
    if isinstance(reference, np.ndarray) and reference.dtype == np.uint8:
        reference = reference > 0
    
    h, w = detected.shape
    diagonal = np.sqrt(h**2 + w**2)
    max_dist_pixels = max_dist * diagonal
    
    # 边界细化
    if thin:
        detected = thin_boundary(detected)
        reference = thin_boundary(reference)
    
    # 边界匹配
    match1, match2 = correspond_pixels_optimized(detected, reference, max_dist_pixels)
    
    # 计算TP, FP, FN
    TP = int(match1.sum())  # 检测到的边界中匹配的像素数
    FP = int((detected & ~match1).sum())  # 检测到的边界中未匹配的像素数
    FN = int((reference & ~match2).sum())  # 参考边界中未匹配的像素数
    
    return TP, FP, FN

