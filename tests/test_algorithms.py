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

def test_single_image():
    """测试单张图片的边缘检测性能"""
    # 检查BSDS500数据集
    bsds_images_dir = 'BSR_bsds500/BSDS500/data/images/val'

    if not os.path.exists(bsds_images_dir):
        print(f"❌ BSDS500数据集不存在: {bsds_images_dir}")
        print("请确保BSDS500数据集已正确放置")
        return

    # 获取测试图片
    import glob
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(bsds_images_dir, ext)))

    if len(image_files) == 0:
        print(f"❌ 在 {bsds_images_dir} 中未找到图片")
        return

    # 测试前5张图片
    test_files = image_files[:5]

    print("\n" + "="*70)
    print("🔍 边缘检测算法性能对比测试 (BSDS500)")
    print("="*70)
    print("📊 预期性能（基于论文）:")
    print("   - Canny:    F1 ≈ 0.60 (基线)")
    print("   - HED:      F1 ≈ 0.78 (+30%)")
    print("   - PiDiNet:  F1 ≈ 0.81 (+35%)")
    print("="*70)

    algorithms = ['Canny', 'HED', 'PiDiNet']
    results = {algo: {'TP': 0, 'FP': 0, 'FN': 0, 'count': 0} for algo in algorithms}

    for idx, img_path in enumerate(test_files):
        print(f"\n📷 测试图片 {idx+1}/{len(test_files)}: {os.path.basename(img_path)}")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  ❌ 无法读取图片")
            continue

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        h, w = img_bgr.shape[:2]

        # 加载ground truth
        from edge_detection_api import load_bsds_ground_truth
        reference = load_bsds_ground_truth(img_name, h, w)

        if reference is None:
            print(f"  ⚠️  未找到ground truth，跳过")
            continue

        # 测试每个算法
        for algo in algorithms:
            try:
                print(f"  🔹 测试 {algo}...")

                # 计算边缘强度
                edge_strength = compute_edge_strength(
                    img_bgr,
                    algo,
                    blur=7,
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

                print(f"     Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

            except Exception as e:
                print(f"     ❌ {algo} 失败: {e}")

    # 打印总体结果
    print("\n" + "="*70)
    print("📊 总体性能对比")
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
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall:    {recall:.3f}")
            print(f"  F1 Score:  {f1:.3f}")

    # 对比分析
    print("\n" + "="*70)
    print("📈 性能提升分析")
    print("="*70)

    if results['Canny']['count'] > 0:
        # 计算Canny的F1
        canny_tp, canny_fp, canny_fn = results['Canny']['TP'], results['Canny']['FP'], results['Canny']['FN']
        canny_prec = canny_tp / (canny_tp + canny_fp) if (canny_tp + canny_fp) > 0 else 0
        canny_rec = canny_tp / (canny_tp + canny_fn) if (canny_tp + canny_fn) > 0 else 0
        canny_f1 = 2 * canny_prec * canny_rec / (canny_prec + canny_rec) if (canny_prec + canny_rec) > 0 else 0

        print(f"\n🎯 基线 (Canny):")
        print(f"   F1 Score: {canny_f1:.3f}")

        # 测试HED
        if results['HED']['count'] > 0:
            hed_tp, hed_fp, hed_fn = results['HED']['TP'], results['HED']['FP'], results['HED']['FN']
            hed_prec = hed_tp / (hed_tp + hed_fp) if (hed_tp + hed_fp) > 0 else 0
            hed_rec = hed_tp / (hed_tp + hed_fn) if (hed_tp + hed_fn) > 0 else 0
            hed_f1 = 2 * hed_prec * hed_rec / (hed_prec + hed_rec) if (hed_prec + hed_rec) > 0 else 0

            improvement_hed = (hed_f1 - canny_f1) / canny_f1 * 100 if canny_f1 > 0 else 0

            print(f"\n🧠 HED:")
            print(f"   F1 Score: {hed_f1:.3f}")
            print(f"   提升: {improvement_hed:+.1f}%")

            if improvement_hed > 20:
                print(f"   ✅ 性能优秀！符合预期（+30%）")
            elif improvement_hed > 10:
                print(f"   ⚠️  性能良好，略低于预期")
            elif improvement_hed > 0:
                print(f"   ⚠️  有轻微提升，但未达预期")
            else:
                print(f"   ❌ 性能未提升，可能存在问题：")
                print(f"      1. 检查HED模型是否正确加载（查看启动日志）")
                print(f"      2. 检查是否使用了Caffe模型而非PyTorch随机初始化")
                print(f"      3. 检查预处理参数（mean、scalefactor）")

        # 测试PiDiNet
        if results['PiDiNet']['count'] > 0:
            pidinet_tp, pidinet_fp, pidinet_fn = results['PiDiNet']['TP'], results['PiDiNet']['FP'], results['PiDiNet']['FN']
            pidinet_prec = pidinet_tp / (pidinet_tp + pidinet_fp) if (pidinet_tp + pidinet_fp) > 0 else 0
            pidinet_rec = pidinet_tp / (pidinet_tp + pidinet_fn) if (pidinet_tp + pidinet_fn) > 0 else 0
            pidinet_f1 = 2 * pidinet_prec * pidinet_rec / (pidinet_prec + pidinet_rec) if (pidinet_prec + pidinet_rec) > 0 else 0

            improvement_pidinet = (pidinet_f1 - canny_f1) / canny_f1 * 100 if canny_f1 > 0 else 0

            print(f"\n🚀 PiDiNet:")
            print(f"   F1 Score: {pidinet_f1:.3f}")
            print(f"   提升: {improvement_pidinet:+.1f}%")

            if improvement_pidinet > 25:
                print(f"   ✅ 性能优秀！符合预期（+35%）")
            elif improvement_pidinet > 15:
                print(f"   ⚠️  性能良好，略低于预期")
            elif improvement_pidinet > 0:
                print(f"   ⚠️  有轻微提升，但未达预期")
            else:
                print(f"   ❌ 性能未提升，可能存在问题：")
                print(f"      1. 检查PiDiNet模型是否正确加载（查看启动日志）")
                print(f"      2. 检查模型文件是否完整")
                print(f"      3. 检查配置参数（inplane、dil、sa）")

    print("\n" + "="*70)
    print("💡 提示：")
    print("   - 如果性能未达预期，请重启后端并查看模型加载日志")
    print("   - 深度学习算法需要正确加载预训练权重")
    print("   - 使用BSDS500完整数据集测试可获得更准确的结果")
    print("="*70)

if __name__ == '__main__':
    test_single_image()
