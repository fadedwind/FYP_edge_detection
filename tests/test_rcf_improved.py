# -*- coding: utf-8 -*-
"""
RCF改进版测试脚本
"""
import os
import sys
import cv2
import numpy as np
from edge_detection_api import compute_edge_strength
from boundary_evaluation import evaluate_boundary_with_tolerance

# Windows兼容性
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def test_rcf_improved():
    """测试改进的RCF算法"""
    print("\n" + "="*70)
    print(" RCF改进版性能测试")
    print("="*70)

    # 检查BSDS500数据集
    bsds_images_dir = 'BSR_bsds500/BSDS500/data/images/val'

    if not os.path.exists(bsds_images_dir):
        print(f"[ERROR] BSDS500 dataset not found: {bsds_images_dir}")
        return

    # 获取测试图片
    import glob
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(bsds_images_dir, ext)))

    if len(image_files) == 0:
        print(f"[ERROR] No images found in {bsds_images_dir}")
        return

    # 测试前10张图片
    test_files = image_files[:10]

    results = {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'count': 0,
        'f1_list': []
    }

    print(f"\nTesting {len(test_files)} images with RCF-Improved...\n")

    for idx, img_path in enumerate(test_files):
        print(f"[Image {idx+1}/{len(test_files)}] {os.path.basename(img_path)}")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [ERROR] Cannot read image")
            continue

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        h, w = img_bgr.shape[:2]

        # 加载ground truth
        from edge_detection_api import load_bsds_ground_truth
        reference = load_bsds_ground_truth(img_name, h, w)

        if reference is None:
            print(f"  [WARNING] Ground truth not found, skipping")
            continue

        try:
            print(f"  Testing RCF-Improved...", end=' ')

            # 计算RCF边缘
            edge_map = compute_edge_strength(
                img_bgr,
                "RCF",
                blur_ksize=7,
                sobel_ksize=3,
                canny_low=100,
                canny_high=200
            )

            # edge_map已经是二值化的了（Otsu阈值在内部应用）
            detected = edge_map

            # 评估
            TP, FP, FN = evaluate_boundary_with_tolerance(
                detected,
                reference,
                max_dist=0.0075,
                thin=True
            )

            # 计算F1
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results['TP'] += TP
            results['FP'] += FP
            results['FN'] += FN
            results['count'] += 1
            results['f1_list'].append(f1)

            print(f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

            # 高亮显示优秀结果
            if f1 >= 0.70:
                print(f"         ✓ Excellent!")
            elif f1 >= 0.60:
                print(f"         ↑ Very Good")

        except Exception as e:
            print(f"  [ERROR] RCF-Improved failed: {e}")
            import traceback
            traceback.print_exc()

    # 打印结果
    print("\n" + "="*70)
    print(" RCF-Improved Performance Summary")
    print("="*70)

    if results['count'] > 0:
        TP = results['TP']
        FP = results['FP']
        FN = results['FN']

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nOverall Metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Images:    {results['count']}")

        # 统计信息
        if len(results['f1_list']) > 0:
            f1_array = np.array(results['f1_list'])
            f1_mean = np.mean(f1_array)
            f1_std = np.std(f1_array)
            f1_min = np.min(f1_array)
            f1_max = np.max(f1_array)

            print(f"\nF1 Score Statistics:")
            print(f"  Mean:  {f1_mean:.4f}")
            print(f"  Std:   {f1_std:.4f}")
            print(f"  Min:   {f1_min:.4f}")
            print(f"  Max:   {f1_max:.4f}")

        # 与原版RCF对比
        print(f"\n" + "="*70)
        print(" Comparison with Original RCF")
        print("="*70)

        rcf_original_f1 = 0.3970
        improvement = (f1 - rcf_original_f1) / rcf_original_f1 * 100 if rcf_original_f1 > 0 else 0

        print(f"\n[RCF Original]")
        print(f"  F1 Score: {rcf_original_f1:.4f}")
        print(f"  RCF-Improved vs Original: {improvement:+.1f}%")

        # 与其他算法对比
        pidinet_f1 = 0.7138
        canny_f1 = 0.5137

        print(f"\n[Other Algorithms]")
        print(f"  PiDiNet: {pidinet_f1:.4f}")
        print(f"  Canny:   {canny_f1:.4f}")

        print(f"\n" + "="*70)
        print(" Evaluation")
        print("="*70)

        if f1 >= 0.70:
            print(f"  [EXCELLENT] F1={f1:.4f} >= 0.70")
            print(f"  RCF-Improved性能优秀！")
        elif f1 >= 0.60:
            print(f"  [VERY GOOD] F1={f1:.4f} >= 0.60")
            print(f"  RCF-Improved性能很好")
        elif f1 >= 0.50:
            print(f"  [GOOD] F1={f1:.4f} >= 0.50")
            print(f"  RCF-Improved性能良好")
        elif f1 >= rcf_original_f1:
            print(f"  [IMPROVED] F1={f1:.4f} > {rcf_original_f1:.4f} (original)")
            print(f"  比原版RCF有提升")
        else:
            print(f"  [NEEDS WORK] F1={f1:.4f}")
            print(f"  需要进一步优化")

        print(f"\n" + "="*70)

    else:
        print("\n[ERROR] No valid test results")

    return f1 if results['count'] > 0 else 0

if __name__ == '__main__':
    test_rcf_improved()
