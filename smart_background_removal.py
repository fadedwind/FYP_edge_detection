# -*- coding: utf-8 -*-
"""
智能抠图模块 - 移除图像背景

支持的方法：
1. rembg (推荐) - 基于U²-Net深度学习模型，自动抠图
2. GrabCut - OpenCV传统方法，需要用户交互
3. 边缘辅助方法 - 结合边缘检测和颜色特征

依赖:
    pip install rembg  # 推荐使用
    # 或 pip install rembg[gpu]  # GPU加速版本

使用示例:
    from smart_background_removal import remove_background_auto, remove_background_grabcut

    # 自动抠图（推荐）
    result = remove_background_auto('input.jpg', 'output.png')

    # GrabCut方法（需要指定ROI）
    result = remove_background_grabcut('input.jpg', 'output.png')
"""
import os
import sys
import cv2
import numpy as np
from typing import Tuple, Optional

# Windows兼容性
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# 方法1: rembg - 自动抠图（推荐）
# ============================================================================

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print('[INFO] rembg未安装，自动抠图功能不可用')
    print('       安装: pip install rembg')


def remove_background_auto(
    input_path: str,
    output_path: Optional[str] = None,
    return_mask: bool = False,
    session_model: str = 'u2net'
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    使用rembg自动移除背景（推荐方法）

    Args:
        input_path: 输入图像路径
        output_path: 输出图像路径（可选）
        return_mask: 是否返回mask
        session_model: 模型类型
            - 'u2net': 高质量，较慢（推荐）
            - 'u2netp': 快速版本
            - 'u2net_human_seg': 人像专用
            - 'isnet-general-use': 通用场景
            - 'silueta': 最快

    Returns:
        (result_image, mask) - 如果return_mask=True
        (result_image, None) - 如果return_mask=False
        result_image是BGRA格式的numpy数组
    """
    if not REMBG_AVAILABLE:
        raise ImportError('rembg未安装，请运行: pip install rembg')

    # 读取图像
    with open(input_path, 'rb') as f:
        input_data = f.read()

    # 创建session（只创建一次，提高性能）
    session = new_session(model_name=session_model)

    # 移除背景
    output_data = remove(input_data, session=session, alpha_matting=True)

    # 转换为numpy数组
    result_array = np.frombuffer(output_data, dtype=np.uint8)
    result_img = cv2.imdecode(result_array, cv2.IMREAD_UNCHANGED)

    # 提取mask
    mask = None
    if return_mask and result_img.shape[2] == 4:
        mask = result_img[:, :, 3]  # Alpha通道作为mask

    # 保存结果
    if output_path:
        cv2.imwrite(output_path, result_img)

    return result_img, mask


# ============================================================================
# 方法2: GrabCut - OpenCV传统方法
# ============================================================================

def remove_background_grabcut(
    image_path: str,
    output_path: Optional[str] = None,
    rect: Optional[Tuple[int, int, int, int]] = None,
    iter_count: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用GrabCut算法移除背景（需要指定ROI）

    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径
        rect: 前景矩形区域 (x, y, w, h)，如果为None则自动检测
        iter_count: 迭代次数（越大越好，但越慢）

    Returns:
        (result_image, mask) - BGRA格式图像和mask
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f'无法读取图像: {image_path}')

    height, width = img.shape[:2]

    # 自动检测前景区域（使用边缘检测）
    if rect is None:
        print('[INFO] 自动检测前景区域...')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 找到最大轮廓
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)

            # 扩大矩形（包含完整前景）
            margin = 20
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(width - x, w + 2 * margin)
            h = min(height - y, h + 2 * margin)

            rect = (x, y, w, h)
            print(f'[INFO] 检测到前景区域: {rect}')
        else:
            # 使用整个图像中心区域
            margin = min(width, height) // 4
            rect = (margin, margin, width - 2*margin, height - 2*margin)
            print(f'[INFO] 使用默认前景区域: {rect}')

    # 初始化GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 矩形区域：肯定是前景
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_RECT)

    # 修改mask：0和2转为0（背景），1和3转为1（前景）
    mask2 = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')

    # 应用mask
    result = cv2.bitwise_and(img, img, mask=mask2)

    # 转换为BGRA（添加透明通道）
    result_bgra = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result_bgra[:, :, 3] = mask2  # Alpha通道

    # 保存结果
    if output_path:
        cv2.imwrite(output_path, result_bgra)

    return result_bgra, mask2


# ============================================================================
# 方法3: 边缘辅助的智能抠图
# ============================================================================

def remove_background_edge_assisted(
    image_path: str,
    output_path: Optional[str] = None,
    edge_algorithm: str = 'auto',
    threshold: int = 127
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于边缘检测的智能抠图（结合边缘和颜色特征）

    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径
        edge_algorithm: 边缘检测算法 ('canny', 'sobel', 'pidinet', 'auto')
        threshold: 边缘阈值

    Returns:
        (result_image, mask) - BGRA格式图像和mask
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f'无法读取图像: {image_path}')

    height, width = img.shape[:2]

    # 使用PiDiNet进行边缘检测（如果可用）
    if edge_algorithm == 'auto' or edge_algorithm == 'pidinet':
        try:
            from pidinet import run_pidinet
            print('[INFO] 使用PiDiNet进行边缘检测...')
            edge_map = run_pidinet(img, model=None, device='cpu')
            _, edge_binary = cv2.threshold(edge_map, threshold, 255, cv2.THRESH_BINARY)
        except Exception as e:
            print(f'[INFO] PiDiNet不可用，回退到Canny: {e}')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edge_binary = cv2.Canny(gray, 50, 150)
    elif edge_algorithm == 'canny':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge_binary = cv2.Canny(gray, 50, 150)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge_binary = cv2.Canny(gray, 50, 150)

    # 形态学操作：填充边缘内部区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_dilated = cv2.dilate(edge_binary, kernel, iterations=3)

    # 查找轮廓并填充
    contours, _ = cv2.findContours(edge_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # 填充所有轮廓（假设最大的轮廓是前景）
    if contours:
        # 按面积排序，保留最大的几个轮廓
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

        # 填充最大轮廓（前景）
        cv2.fillPoly(mask, [contours_sorted[0]], 255)

        # 可选：填充第二大的轮廓（如果有多个物体）
        if len(contours_sorted) > 1:
            area_ratio = cv2.contourArea(contours_sorted[1]) / cv2.contourArea(contours_sorted[0])
            if area_ratio > 0.1:  # 如果第二大的轮廓面积大于最大的10%
                cv2.fillPoly(mask, [contours_sorted[1]], 255)

    # 优化mask边缘（形态学操作）
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 高斯模糊平滑边缘
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # 应用mask
    result = cv2.bitwise_and(img, img, mask=mask)

    # 转换为BGRA
    result_bgra = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result_bgra[:, :, 3] = mask  # Alpha通道

    # 保存结果
    if output_path:
        cv2.imwrite(output_path, result_bgra)

    return result_bgra, mask


# ============================================================================
# 便捷函数
# ============================================================================

def smart_remove_background(
    image_path: str,
    output_path: Optional[str] = None,
    method: str = 'auto'
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    智能抠图（自动选择最佳方法）

    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径
        method: 方法选择
            - 'auto': 自动选择（优先rembg，回退到边缘辅助）
            - 'rembg': 使用rembg（推荐，需要安装）
            - 'grabcut': 使用GrabCut
            - 'edge': 使用边缘辅助方法

    Returns:
        (result_image, mask)
    """
    if method == 'auto':
        # 优先使用rembg
        if REMBG_AVAILABLE:
            print('[INFO] 使用rembg进行自动抠图...')
            return remove_background_auto(image_path, output_path, return_mask=True)
        else:
            print('[INFO] rembg不可用，使用边缘辅助方法...')
            return remove_background_edge_assisted(image_path, output_path)

    elif method == 'rembg':
        if not REMBG_AVAILABLE:
            raise ImportError('rembg未安装，请运行: pip install rembg')
        return remove_background_auto(image_path, output_path, return_mask=True)

    elif method == 'grabcut':
        return remove_background_grabcut(image_path, output_path)

    elif method == 'edge':
        return remove_background_edge_assisted(image_path, output_path)

    else:
        raise ValueError(f'未知的方法: {method}')


# ============================================================================
# 命令行使用
# ============================================================================

if __name__ == '__main__':
    """命令行使用示例"""
    import glob

    print('='*70)
    print(' 智能抠图工具')
    print('='*70)

    # 检查命令行参数
    if len(sys.argv) < 2:
        print('\n使用方法:')
        print('  python smart_background_removal.py <input_image> [output_image] [method]')
        print('\n方法选项:')
        print('  - auto: 自动选择（默认）')
        print('  - rembg: 使用rembg（推荐，需要安装）')
        print('  - grabcut: 使用GrabCut')
        print('  - edge: 使用边缘辅助方法')
        print('\n示例:')
        print('  python smart_background_removal.py photo.jpg result.png')
        print('  python smart_background_removal.py photo.jpg result.png rembg')
        print('\n如需批量处理，可将多张图片放在文件夹并运行脚本。')

        # 尝试查找测试图像
        test_images = glob.glob('*.jpg') + glob.glob('*.png')
        if test_images:
            print(f'\n💡 发现 {len(test_images)} 张图像，是否测试？')
            for img in test_images[:3]:
                print(f'   - {img}')
        sys.exit(0)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    method = sys.argv[3] if len(sys.argv) > 3 else 'auto'

    # 自动生成输出路径
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f'{base_name}_no_bg.png'

    # 检查输入文件
    if not os.path.exists(input_path):
        print(f'[ERROR] 输入文件不存在: {input_path}')
        sys.exit(1)

    # 执行抠图
    try:
        print(f'\n输入: {input_path}')
        print(f'输出: {output_path}')
        print(f'方法: {method}')
        print(f'\n处理中...')

        result_img, mask = smart_remove_background(input_path, output_path, method=method)

        print(f'✅ 抠图完成！')
        print(f'   输出尺寸: {result_img.shape[1]}x{result_img.shape[0]}')
        print(f'   通道数: {result_img.shape[2]} (BGRA)')

        if mask is not None:
            fg_ratio = np.sum(mask > 0) / mask.size * 100
            print(f'   前景占比: {fg_ratio:.1f}%')

        print(f'\n结果已保存到: {output_path}')

    except Exception as e:
        print(f'[ERROR] 抠图失败: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
