# -*- coding: utf-8 -*-
"""
检查RCF模型权重加载情况
"""
import torch
import sys

# Windows兼容性
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def check_rcf_weights():
    """检查RCF模型权重"""
    print("\n" + "="*70)
    print(" RCF模型权重检查工具")
    print("="*70)

    # 加载下载的模型
    model_path = 'models/rcf_bsds500_pascal.pth'
    print(f"\n正在加载模型: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✓ 模型文件加载成功")

        # 检查模型结构
        if isinstance(checkpoint, dict):
            print(f"\n模型包含 {len(checkpoint)} 个键:")
            for key in checkpoint.keys():
                if isinstance(checkpoint[key], dict):
                    print(f"  - {key}: dict ({len(checkpoint[key])} 参数)")
                    # 显示前几个参数
                    count = 0
                    for param_name, param_value in checkpoint[key].items():
                        if count < 10:
                            shape = param_value.shape if hasattr(param_value, 'shape') else type(param_value).__name__
                            print(f"      {param_name}: {shape}")
                            count += 1
                        elif count == 10:
                            print(f"      ... (还有 {len(checkpoint[key]) - 10} 个参数)")
                            break
                else:
                    print(f"  - {key}: {type(checkpoint[key])}")

        # 创建我们的RCF模型
        print(f"\n创建RCF模型...")
        from rcf import RCFNet
        model = RCFNet(pretrained_backbone=False)
        model_dict = model.state_dict()

        print(f"我们的RCF模型有 {len(model_dict)} 个参数\n")

        # 尝试匹配参数
        if isinstance(checkpoint, dict) and len(checkpoint) > 0:
            # 获取第一个字典（通常是模型权重）
            pretrained_dict = None
            for key in checkpoint.keys():
                if isinstance(checkpoint[key], dict):
                    pretrained_dict = checkpoint[key]
                    break

            if pretrained_dict is None:
                pretrained_dict = checkpoint

            print("="*70)
            print(" 参数匹配情况")
            print("="*70)

            # 找到匹配的参数
            matched = []
            unmatched_pretrained = []
            unmatched_model = []

            for k, v in pretrained_dict.items():
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        matched.append(k)
                    else:
                        print(f"⚠️  形状不匹配: {k}")
                        print(f"     预训练: {v.shape}")
                        print(f"     模型:    {model_dict[k].shape}")
                        unmatched_pretrained.append(k)
                else:
                    unmatched_pretrained.append(k)

            for k in model_dict.keys():
                if k not in pretrained_dict:
                    unmatched_model.append(k)

            print(f"\n✓ 匹配的参数: {len(matched)}/{len(model_dict)} ({100*len(matched)/len(model_dict):.1f}%)")

            if len(matched) > 0:
                print(f"\n匹配的参数列表（前20个）:")
                for i, k in enumerate(matched[:20]):
                    print(f"  {i+1}. {k}")

                if len(matched) > 20:
                    print(f"  ... (还有 {len(matched)-20} 个)")

            if len(unmatched_pretrained) > 0:
                print(f"\n⚠️  预训练模型中未匹配的参数: {len(unmatched_pretrained)}")
                for i, k in enumerate(unmatched_pretrained[:10]):
                    print(f"  {i+1}. {k}")

                if len(unmatched_pretrained) > 10:
                    print(f"  ... (还有 {len(unmatched_pretrained)-10} 个)")

            if len(unmatched_model) > 0:
                print(f"\n⚠️  我们模型中未匹配的参数: {len(unmatched_model)}")
                for i, k in enumerate(unmatched_model[:10]):
                    print(f"  {i+1}. {k}")

                if len(unmatched_model) > 10:
                    print(f"  ... (还有 {len(unmatched_model)-10} 个)")

        print("\n" + "="*70)

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    check_rcf_weights()
