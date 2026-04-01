# -*- coding: utf-8 -*-
"""
测试脚本：验证HED和PiDiNet在BSDS500上的性能是否优于Canny

预期结果（基于BSDS500数据集）：
- Canny: ODS F1 ≈ 0.60
- HED: ODS F1 ≈ 0.78 (+30% vs Canny)
- PiDiNet: ODS F1 ≈ 0.81 (+35% vs Canny)
"""
import os
import sys
import cv2
import numpy as np
from edge_detection_api import compute_edge_strength, process_single_image_for_batch
from boundary_evaluation import evaluate_boundary_with_tolerance

# Windows兼容性：设置UTF-8编码输出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def test_single_image():
    """测试单张图片的边缘检测性能"""
    # 检查BSDS500数据集
    bsds_images_dir = 'BSR_bsds500/BSDS500/data/images/val'

    if not os.path.exists(bsds_images_dir):
        print(f"[ERROR] BSDS500 dataset not found: {bsds_images_dir}")
        print("Please ensure BSDS500 dataset is properly placed")
        return

    # 获取测试图片
    import glob
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(bsds_images_dir, ext)))

    if len(image_files) == 0:
        print(f"[ERROR] No images found in {bsds_images_dir}")
        return

    # 测试前10张图片（获得更准确的结果）
    test_files = image_files[:10]

    print("\n" + "="*70)
    print(" Edge Detection Algorithm Performance Comparison (BSDS500)")
    print("="*70)
    print(" Expected Performance (based on papers):")
    print("   - Canny:    F1 = 0.60 (baseline)")
    print("   - HED:      F1 = 0.78 (+30%)")
    print("   - PiDiNet:  F1 = 0.81 (+35%)")
    print("="*70)

    algorithms = ['Canny', 'HED', 'PiDiNet']
    results = {algo: {'TP': 0, 'FP': 0, 'FN': 0, 'count': 0, 'f1_list': []} for algo in algorithms}

    for idx, img_path in enumerate(test_files):
        print(f"\n[Image {idx+1}/{len(test_files)}] {os.path.basename(img_path)}")

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

        # 测试每个算法
        for algo in algorithms:
            try:
                print(f"  Testing {algo}...", end=' ')

                # 计算边缘强度
                edge_strength = compute_edge_strength(
                    img_bgr,
                    algo,
                    blur_ksize=7,
                    sobel_ksize=3,
                    canny_low=100,
                    canny_high=200
                )

                # 使用固定阈值127
                _, detected = cv2.threshold(edge_strength, 127, 255, cv2.THRESH_BINARY)

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

                results[algo]['TP'] += TP
                results[algo]['FP'] += FP
                results[algo]['FN'] += FN
                results[algo]['count'] += 1
                results[algo]['f1_list'].append(f1)

                print(f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

            except Exception as e:
                print(f"  [ERROR] {algo} failed: {e}")
                import traceback
                traceback.print_exc()

    # 打印总体结果
    print("\n" + "="*70)
    print(" Overall Performance Comparison")
    print("="*70)

    for algo in algorithms:
        if results[algo]['count'] > 0:
            TP = results[algo]['TP']
            FP = results[algo]['FP']
            FN = results[algo]['FN']

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"\n{algo}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"  Images:    {results[algo]['count']}")

    # 对比分析
    print("\n" + "="*70)
    print(" Performance Improvement Analysis")
    print("="*70)

    if results['Canny']['count'] > 0:
        # 计算Canny的F1
        canny_tp, canny_fp, canny_fn = results['Canny']['TP'], results['Canny']['FP'], results['Canny']['FN']
        canny_prec = canny_tp / (canny_tp + canny_fp) if (canny_tp + canny_fp) > 0 else 0
        canny_rec = canny_tp / (canny_tp + canny_fn) if (canny_tp + canny_fn) > 0 else 0
        canny_f1 = 2 * canny_prec * canny_rec / (canny_prec + canny_rec) if (canny_prec + canny_rec) > 0 else 0

        print(f"\n[Baseline] Canny:")
        print(f"   F1 Score: {canny_f1:.4f}")

        # 测试HED
        if results['HED']['count'] > 0:
            hed_tp, hed_fp, hed_fn = results['HED']['TP'], results['HED']['FP'], results['HED']['FN']
            hed_prec = hed_tp / (hed_tp + hed_fp) if (hed_tp + hed_fp) > 0 else 0
            hed_rec = hed_tp / (hed_tp + hed_fn) if (hed_tp + hed_fn) > 0 else 0
            hed_f1 = 2 * hed_prec * hed_rec / (hed_prec + hed_rec) if (hed_prec + hed_rec) > 0 else 0

            improvement_hed = (hed_f1 - canny_f1) / canny_f1 * 100 if canny_f1 > 0 else 0

            print(f"\n[HED]")
            print(f"   F1 Score: {hed_f1:.4f}")
            print(f"   Improvement: {improvement_hed:+.1f}%")

            if improvement_hed > 20:
                print(f"   [EXCELLENT] Performance meets expectation (+30%)")
            elif improvement_hed > 10:
                print(f"   [GOOD] Performance is slightly below expectation")
            elif improvement_hed > 0:
                print(f"   [WARNING] Minor improvement but below expectation")
            else:
                print(f"   [ERROR] No improvement detected. Possible issues:")
                print(f"      1. Check if HED model is loaded correctly (check startup logs)")
                print(f"      2. Verify Caffe model is used, not PyTorch random init")
                print(f"      3. Check preprocessing parameters (mean, scalefactor)")

        # 测试PiDiNet
        if results['PiDiNet']['count'] > 0:
            pidinet_tp, pidinet_fp, pidinet_fn = results['PiDiNet']['TP'], results['PiDiNet']['FP'], results['PiDiNet']['FN']
            pidinet_prec = pidinet_tp / (pidinet_tp + pidinet_fp) if (pidinet_tp + pidinet_fp) > 0 else 0
            pidinet_rec = pidinet_tp / (pidinet_tp + pidinet_fn) if (pidinet_tp + pidinet_fn) > 0 else 0
            pidinet_f1 = 2 * pidinet_prec * pidinet_rec / (pidinet_prec + pidinet_rec) if (pidinet_prec + pidinet_rec) > 0 else 0

            improvement_pidinet = (pidinet_f1 - canny_f1) / canny_f1 * 100 if canny_f1 > 0 else 0

            print(f"\n[PiDiNet]")
            print(f"   F1 Score: {pidinet_f1:.4f}")
            print(f"   Improvement: {improvement_pidinet:+.1f}%")

            if improvement_pidinet > 25:
                print(f"   [EXCELLENT] Performance meets expectation (+35%)")
            elif improvement_pidinet > 15:
                print(f"   [GOOD] Performance is slightly below expectation")
            elif improvement_pidinet > 0:
                print(f"   [WARNING] Minor improvement but below expectation")
            else:
                print(f"   [ERROR] No improvement detected. Possible issues:")
                print(f"      1. Check if PiDiNet model is loaded correctly (check startup logs)")
                print(f"      2. Verify model file integrity")
                print(f"      3. Check configuration parameters (inplane, dil, sa)")

    print("\n" + "="*70)
    print(" Tips:")
    print("   - If performance is below expectation, restart backend and check logs")
    print("   - Deep learning algorithms require proper pretrained weights")
    print("   - Use full BSDS500 dataset for more accurate results")
    print("="*70)

    # 保存结果到文件
    save_results_to_file(results, algorithms)

def save_results_to_file(results, algorithms):
    """保存结果到output目录"""
    os.makedirs('output', exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'output/benchmark_results_{timestamp}.txt'

    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(" Edge Detection Algorithm Benchmark Results\n")
        f.write("="*70 + "\n\n")

        for algo in algorithms:
            if results[algo]['count'] > 0:
                TP = results[algo]['TP']
                FP = results[algo]['FP']
                FN = results[algo]['FN']

                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                f.write(f"\n{algo}:\n")
                f.write(f"  Precision: {precision:.4f}\n")
                f.write(f"  Recall:    {recall:.4f}\n")
                f.write(f"  F1 Score:  {f1:.4f}\n")
                f.write(f"  Images:    {results[algo]['count']}\n")

                # 计算标准差
                if len(results[algo]['f1_list']) > 0:
                    f1_array = np.array(results[algo]['f1_list'])
                    f1_std = np.std(f1_array)
                    f1_mean = np.mean(f1_array)
                    f.write(f"  F1 Mean:   {f1_mean:.4f}\n")
                    f.write(f"  F1 Std:    {f1_std:.4f}\n")

        f.write("\n" + "="*70 + "\n")
        f.write(" Summary:\n")
        f.write("="*70 + "\n")

        if results['Canny']['count'] > 0 and results['HED']['count'] > 0:
            canny_f1 = 2 * (results['Canny']['TP'] / (results['Canny']['TP'] + results['Canny']['FP']) *
                          results['Canny']['TP'] / (results['Canny']['TP'] + results['Canny']['FN'])) / \
                         ((results['Canny']['TP'] / (results['Canny']['TP'] + results['Canny']['FP'])) +
                          (results['Canny']['TP'] / (results['Canny']['TP'] + results['Canny']['FN']))) \
                         if (results['Canny']['TP'] + results['Canny']['FP']) > 0 and (results['Canny']['TP'] + results['Canny']['FN']) > 0 else 0

            hed_f1 = 2 * (results['HED']['TP'] / (results['HED']['TP'] + results['HED']['FP']) *
                        results['HED']['TP'] / (results['HED']['TP'] + results['HED']['FN'])) / \
                       ((results['HED']['TP'] / (results['HED']['TP'] + results['HED']['FP'])) +
                        (results['HED']['TP'] / (results['HED']['TP'] + results['HED']['FN']))) \
                       if (results['HED']['TP'] + results['HED']['FP']) > 0 and (results['HED']['TP'] + results['HED']['FN']) > 0 else 0

            if canny_f1 > 0:
                improvement = (hed_f1 - canny_f1) / canny_f1 * 100
                f.write(f"HED vs Canny: {improvement:+.1f}%\n")

        if results['Canny']['count'] > 0 and results['PiDiNet']['count'] > 0:
            pidinet_f1 = 2 * (results['PiDiNet']['TP'] / (results['PiDiNet']['TP'] + results['PiDiNet']['FP']) *
                            results['PiDiNet']['TP'] / (results['PiDiNet']['TP'] + results['PiDiNet']['FN'])) / \
                           ((results['PiDiNet']['TP'] / (results['PiDiNet']['TP'] + results['PiDiNet']['FP'])) +
                            (results['PiDiNet']['TP'] / (results['PiDiNet']['TP'] + results['PiDiNet']['FN']))) \
                           if (results['PiDiNet']['TP'] + results['PiDiNet']['FP']) > 0 and (results['PiDiNet']['TP'] + results['PiDiNet']['FN']) > 0 else 0

            if canny_f1 > 0:
                improvement = (pidinet_f1 - canny_f1) / canny_f1 * 100
                f.write(f"PiDiNet vs Canny: {improvement:+.1f}%\n")

    print(f"\n[INFO] Results saved to: {result_file}")

if __name__ == '__main__':
    from datetime import datetime
    test_single_image()
