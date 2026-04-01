# -*- coding: utf-8 -*-
"""
检查HED模型加载状态
"""
import os
import cv2
import numpy as np

print("="*70)
print(" HED Model Loading Check")
print("="*70)

# 检查模型文件
repo_dir = os.path.dirname(os.path.abspath(__file__))
prototxt_path = os.path.join(repo_dir, 'models', 'hed_deploy.prototxt')
cand1 = os.path.join(repo_dir, 'models', 'hed_pretrained_bsds.caffemodel')
cand2 = os.path.join(repo_dir, 'models', 'hed_bsds.caffemodel')

print(f"\n[1] Model Files Check:")
print(f"    prototxt: {os.path.exists(prototxt_path)} - {prototxt_path}")
print(f"    caffemodel 1: {os.path.exists(cand1)} - {cand1}")
print(f"    caffemodel 2: {os.path.exists(cand2)} - {cand2}")

caffemodel_path = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else None)

if caffemodel_path is None:
    print("\n[ERROR] No Caffe model found!")
    exit(1)

print(f"\n[2] Loading HED Model...")

try:
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    print(f"[SUCCESS] HED Caffe model loaded successfully!")
    print(f"          Model: {caffemodel_path}")

    print(f"\n[3] Model Information:")
    # print(f"    Backend: {net.getPreferableBackend()}")
    # print(f"    Target: {net.getPreferableTarget()}")
    print(f"    Model loaded successfully")

    # 测试推理
    print(f"\n[4] Testing Inference...")

    # 创建测试图像
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    blob = cv2.dnn.blobFromImage(test_img, scalefactor=1.0, size=(500, 500),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()
    out_map = out[0, 0, :, :]

    print(f"[SUCCESS] Inference successful!")
    print(f"          Output shape: {out_map.shape}")
    print(f"          Output range: [{out_map.min():.3f}, {out_map.max():.3f}]")
    print(f"          Output mean: {out_map.mean():.3f}")
    print(f"          Output dtype: {out_map.dtype}")

    # 测试不同阈值
    print(f"\n[5] Testing Different Thresholds:")
    out_map_uint8 = (out_map * 255.0).clip(0, 255).astype('uint8')

    for threshold in [50, 100, 127, 150, 200]:
        _, binary = cv2.threshold(out_map_uint8, threshold, 255, cv2.THRESH_BINARY)
        edge_pixels = np.sum(binary > 0)
        total_pixels = binary.shape[0] * binary.shape[1]
        edge_ratio = edge_pixels / total_pixels
        print(f"    Threshold={threshold:3d}: Edge pixels={edge_pixels:6d} ({edge_ratio*100:.2f}%)")

    print(f"\n[RECOMMENDATION]")
    print(f"    Based on the output distribution, the optimal threshold")
    print(f"    might be different from 127. Consider using:")
    print(f"    - Otsu's method: cv2.threshold(out_map_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)")
    print(f"    - Adaptive threshold based on image statistics")

except Exception as e:
    print(f"[ERROR] Failed to load HED model: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
