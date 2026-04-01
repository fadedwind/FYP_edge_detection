# -*- coding: utf-8 -*-
"""
HED算法专用测试脚本 - 验证阈值修复效果

测试HED算法在BSDS500上的性能，使用修复后的Otsu自适应阈值
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

def test_hed_only():
    """只测试HED算法"""
    print("\n" + "="*70)
    print(" HED Algorithm Performance Test (After Threshold Fix)")
    print("="*70)
    print(" Fix Applied: Otsu Adaptive Threshold")
    print(" Expected F1: ~0.75-0.78")
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

    # 测试前20张图片获得更准确的结果
    test_files = image_files[:20]

    results = {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'count': 0,
        'f1_list': []
    }

    print(f"\nTesting {len(test_files)} images...\n")

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
            print(f"  Testing HED...", end=' ')

            # 计算HED边缘（现在内部使用Otsu阈值）
            edge_map = compute_edge_strength(
                img_bgr,
                "HED",
                blur_ksize=7,
                sobel_ksize=3,
                canny_low=100,
                canny_high=200
            )

            # edge_map已经是二值化的了（Otsu阈值在compute_edge_strength内部应用）
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

        except Exception as e:
            print(f"  [ERROR] HED failed: {e}")
            import traceback
            traceback.print_exc()

    # 打印结果
    print("\n" + "="*70)
    print(" HED Performance Summary (After Fix)")
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

        # 与Canny对比（使用之前测试的结果）
        canny_f1 = 0.5137
        improvement = (f1 - canny_f1) / canny_f1 * 100 if canny_f1 > 0 else 0

        print(f"\nComparison with Canny:")
        print(f"  Canny F1:   {canny_f1:.4f}")
        print(f"  HED F1:     {f1:.4f}")
        print(f"  Improvement: {improvement:+.1f}%")

        print(f"\nEvaluation:")
        if f1 >= 0.75:
            print(f"  [EXCELLENT] F1={f1:.4f} >= 0.75")
            print(f"  HED性能修复成功！符合预期（+30% vs Canny）")
        elif f1 >= 0.65:
            print(f"  [GOOD] F1={f1:.4f} >= 0.65")
            print(f"  HED性能有明显改善，接近预期")
        elif f1 >= 0.50:
            print(f"  [ACCEPTABLE] F1={f1:.4f} >= 0.50")
            print(f"  HED性能有所改善，但仍需优化")
        else:
            print(f"  [POOR] F1={f1:.4f} < 0.50")
            print(f"  HED性能仍未达到预期，需要进一步调查")

        # 与修复前对比
        old_hed_f1 = 0.2793
        improvement_old = (f1 - old_hed_f1) / old_hed_f1 * 100 if old_hed_f1 > 0 else 0

        print(f"\nImprovement vs Before Fix:")
        print(f"  Before: {old_hed_f1:.4f}")
        print(f"  After:  {f1:.4f}")
        print(f"  Gain:   {improvement_old:+.1f}%")

    else:
        print("\n[ERROR] No valid test results")

    print("\n" + "="*70)

    # 保存结果
    save_results(results, f1 if results['count'] > 0 else 0)

def save_results(results, f1_score):
    """保存结果到文件"""
    os.makedirs('output', exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'output/hed_only_test_{timestamp}.txt'

    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(" HED Algorithm Test Results (After Threshold Fix)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Fix: Otsu Adaptive Threshold\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if results['count'] > 0:
            TP = results['TP']
            FP = results['FP']
            FN = results['FN']

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            f.write("Overall Metrics:\n")
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"  Recall:    {recall:.4f}\n")
            f.write(f"  F1 Score:  {f1:.4f}\n")
            f.write(f"  Images:    {results['count']}\n\n")

            if len(results['f1_list']) > 0:
                f1_array = np.array(results['f1_list'])
                f1_mean = np.mean(f1_array)
                f1_std = np.std(f1_array)
                f1_min = np.min(f1_array)
                f1_max = np.max(f1_array)

                f.write("F1 Score Statistics:\n")
                f.write(f"  Mean: {f1_mean:.4f}\n")
                f.write(f"  Std:  {f1_std:.4f}\n")
                f.write(f"  Min:  {f1_min:.4f}\n")
                f.write(f"  Max:  {f1_max:.4f}\n\n")

            f.write("Comparison:\n")
            f.write(f"  Before Fix: 0.2793\n")
            f.write(f"  After Fix:  {f1:.4f}\n")
            f.write(f"  Improvement: {(f1 - 0.2793) / 0.2793 * 100:+.1f}%\n")

    print(f"\n[INFO] Results saved to: {result_file}")

if __name__ == '__main__':
    test_hed_only()
