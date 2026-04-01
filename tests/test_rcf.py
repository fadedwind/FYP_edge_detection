# -*- coding: utf-8 -*-
"""
RCF算法测试脚本 - 验证RCF在BSDS500上的性能

RCF (Richer Convolutional Features) 是一个优秀的深度学习边缘检测算法
在BSDS500上 ODS F-score 达到 0.812

预期性能:
- RCF: ODS F1 ≈ 0.81
- vs Canny: +58% improvement
- vs HED: +4% improvement
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

def test_rcf_only():
    """测试RCF算法"""
    print("\n" + "="*70)
    print(" RCF (Richer Convolutional Features) Performance Test")
    print("="*70)
    print(" Paper: Liu et al., IEEE TPAMI 2019")
    print(" Expected ODS F1: ~0.812 on BSDS500")
    print(" GitHub: https://github.com/yun-liu/RCF-PyTorch")
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

    # 测试前20张图片
    test_files = image_files[:20]

    results = {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'count': 0,
        'f1_list': []
    }

    print(f"\nTesting {len(test_files)} images with RCF...\n")

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
            print(f"  Testing RCF...", end=' ')

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
            if f1 >= 0.75:
                print(f"         ✓ Excellent!")
            elif f1 >= 0.65:
                print(f"         ↑ Good")

        except Exception as e:
            print(f"  [ERROR] RCF failed: {e}")
            import traceback
            traceback.print_exc()

    # 打印结果
    print("\n" + "="*70)
    print(" RCF Performance Summary")
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

        # 与其他算法对比
        print(f"\n" + "="*70)
        print(" Algorithm Comparison")
        print("="*70)

        # Canny基线
        canny_f1 = 0.5137
        improvement_canny = (f1 - canny_f1) / canny_f1 * 100 if canny_f1 > 0 else 0

        print(f"\n[Canny Baseline]")
        print(f"  F1 Score: {canny_f1:.4f}")
        print(f"  RCF vs Canny: {improvement_canny:+.1f}%")

        # HED
        hed_f1 = 0.3530  # 修复后的HED
        improvement_hed = (f1 - hed_f1) / hed_f1 * 100 if hed_f1 > 0 else 0

        print(f"\n[HED (Fixed)]")
        print(f"  F1 Score: {hed_f1:.4f}")
        print(f"  RCF vs HED: {improvement_hed:+.1f}%")

        # PiDiNet
        pidinet_f1 = 0.7138
        improvement_pidinet = (f1 - pidinet_f1) / pidinet_f1 * 100 if pidinet_f1 > 0 else 0

        print(f"\n[PiDiNet]")
        print(f"  F1 Score: {pidinet_f1:.4f}")
        print(f"  RCF vs PiDiNet: {improvement_pidinet:+.1f}%")

        # RCF预期
        rcf_expected = 0.812
        diff_expected = f1 - rcf_expected

        print(f"\n[RCF Expected]")
        print(f"  Expected F1: {rcf_expected:.4f}")
        print(f"  Actual F1:   {f1:.4f}")
        print(f"  Difference:  {diff_expected:+.4f}")

        print(f"\n" + "="*70)
        print(" Evaluation")
        print("="*70)

        if f1 >= 0.78:
            print(f"  [EXCELLENT] F1={f1:.4f} >= 0.78")
            print(f"  RCF性能符合或超过预期！")
        elif f1 >= 0.70:
            print(f"  [VERY GOOD] F1={f1:.4f} >= 0.70")
            print(f"  RCF性能优秀，接近预期")
        elif f1 >= 0.60:
            print(f"  [GOOD] F1={f1:.4f} >= 0.60")
            print(f"  RCF性能良好")
        elif f1 >= 0.50:
            print(f"  [ACCEPTABLE] F1={f1:.4f} >= 0.50")
            print(f"  RCF性能可接受，但未达到预期")
        else:
            print(f"  [POOR] F1={f1:.4f} < 0.50")
            print(f"  RCF性能不达标，需要检查模型和实现")

        # 排名
        print(f"\n" + "="*70)
        print(" Algorithm Ranking (by F1 Score)")
        print("="*70)

        algorithms = [
            ('Canny', 0.5137),
            ('HED (Fixed)', 0.3530),
            ('PiDiNet', 0.7138),
            ('RCF', f1)
        ]

        algorithms.sort(key=lambda x: x[1], reverse=True)

        for rank, (name, score) in enumerate(algorithms, 1):
            marker = '🏆' if name == 'RCF' else '  '
            print(f"  {rank}. {marker} {name:20s}: {score:.4f}")

        print(f"\n" + "="*70)

    else:
        print("\n[ERROR] No valid test results")

    # 保存结果
    save_results(results, f1 if results['count'] > 0 else 0)

def save_results(results, f1_score):
    """保存结果到文件"""
    os.makedirs('output', exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'output/rcf_test_{timestamp}.txt'

    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(" RCF Algorithm Test Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Paper: Liu et al., IEEE TPAMI 2019\n")
        f.write(f"GitHub: https://github.com/yun-liu/RCF-PyTorch\n\n")

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
            f.write(f"  Canny:  0.5137\n")
            f.write(f"  HED:    0.3530\n")
            f.write(f"  PiDiNet: 0.7138\n")
            f.write(f"  RCF:    {f1:.4f}\n")

    print(f"\n[INFO] Results saved to: {result_file}")

if __name__ == '__main__':
    test_rcf_only()
