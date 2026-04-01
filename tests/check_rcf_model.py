# -*- coding: utf-8 -*-
"""
RCF模型检查工具
检查RCF预训练模型是否存在，并提供下载指引
"""
import os
import sys

# Windows兼容性
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def check_rcf_model():
    """检查RCF模型文件"""
    print("\n" + "="*70)
    print(" RCF模型文件检查工具")
    print("="*70)

    # 可能的模型文件名
    model_names = [
        'rcf_bsds500_pascal.pth',
        'bsds500_pascal_model.pth',
        'rcf_pretrained.pth',
        'rcf.pth'
    ]

    models_dir = 'models'
    found_models = []

    print(f"\n正在检查 {models_dir}/ 目录...")

    # 检查models目录是否存在
    if not os.path.exists(models_dir):
        print(f"  [警告] models/ 目录不存在")
        print(f"  正在创建 {models_dir}/ 目录...")
        os.makedirs(models_dir)
        print(f"  ✓ 目录已创建")
    else:
        print(f"  ✓ {models_dir}/ 目录存在")

    # 检查每个可能的模型文件
    print(f"\n正在搜索RCF模型文件...")
    for model_name in model_names:
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            found_models.append((model_name, file_size))
            print(f"  ✓ 找到: {model_name} ({file_size:.1f} MB)")

    # 输出结果
    print("\n" + "="*70)
    print(" 检查结果")
    print("="*70)

    if found_models:
        print(f"\n✅ 成功！找到 {len(found_models)} 个RCF模型文件：\n")
        for model_name, file_size in found_models:
            print(f"  - {model_name} ({file_size:.1f} MB)")

        print(f"\n✓ RCF可以正常使用！")
        print(f"\n运行RCF测试：")
        print(f"  python test_rcf.py")
    else:
        print(f"\n❌ 未找到RCF预训练模型文件\n")
        print(f"您需要手动下载RCF预训练模型。\n")
        print(f"📥 下载步骤：\n")
        print(f"1. 在浏览器中打开以下链接：")
        print(f"   https://drive.google.com/open?id=1TupHeoBKawrniDka0Hc64m3BG4OKG8nM\n")
        print(f"2. 点击下载按钮（⬇️）下载模型文件")
        print(f"3. 将下载的文件重命名为：rcf_bsds500_pascal.pth")
        print(f"4. 将文件移动到项目的 models/ 目录\n")
        print(f"详细下载指南请查看：docs/RCF_Model_Download_Guide.md\n")
        print(f"⚠️ 注意：")
        print(f"  - 没有预训练模型，RCF性能会很差（F1 < 0.2）")
        print(f"  - 模型文件大小约200-300MB")
        print(f"  - 如果无法下载Google Drive，可以使用PiDiNet（F1=0.71）\n")

    print("="*70 + "\n")

    return len(found_models) > 0

if __name__ == '__main__':
    model_exists = check_rcf_model()
    sys.exit(0 if model_exists else 1)
