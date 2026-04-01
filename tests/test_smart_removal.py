# -*- coding: utf-8 -*-
"""
测试智能抠图功能
"""
import requests
import base64
import cv2
import sys
import os

# Windows兼容性
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def image_to_base64(image_path):
    """将图像转换为base64编码"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图像: {image_path}")
        return None
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def save_base64_image(base64_data, output_path):
    """保存base64图像到文件"""
    if ',' in base64_data:
        base64_data = base64_data.split(',')[1]
    img_data = base64.b64decode(base64_data)
    with open(output_path, 'wb') as f:
        f.write(img_data)

def test_smart_removal(image_path, method='auto', api_url="http://localhost:5000"):
    """测试智能抠图API"""
    print(f"\n{'='*70}")
    print(f" 测试智能抠图功能")
    print(f"{'='*70}\n")

    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        print(f"\n💡 请提供一个测试图像路径")
        return False

    # 转换图像
    print(f"📁 读取图像: {image_path}")
    img_base64 = image_to_base64(image_path)
    if img_base64 is None:
        return False

    # 测试API健康状态
    print(f"\n🔍 测试1: 检查API健康状态")
    try:
        response = requests.get(f"{api_url}/api/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ API运行正常")
        else:
            print(f"❌ API状态异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到API: {e}")
        print(f"💡 请确保后端正在运行: python edge_detection_api.py")
        return False

    # 测试智能抠图
    print(f"\n🔍 测试2: 智能抠图 (方法: {method})")
    try:
        payload = {
            "image": img_base64,
            "method": method,
            "edge_threshold": 127
        }

        print(f"⏳ 发送抠图请求...")
        response = requests.post(f"{api_url}/api/remove-background", json=payload, timeout=60)

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ 抠图成功！")
                print(f"\n📊 结果:")
                print(f"   - 方法: {data.get('method')}")
                print(f"   - 前景占比: {data.get('foreground_ratio', 0):.1f}%")

                # 保存结果图像
                if 'images' in data and 'result' in data['images']:
                    result_b64 = data['images']['result']
                    output_path = f"output_removal_{method}_{os.path.basename(image_path)}"
                    save_base64_image(result_b64, output_path)
                    print(f"\n💾 结果图像已保存: {output_path}")

                # 保存mask
                if 'images' in data and 'mask' in data['images']:
                    mask_b64 = data['images']['mask']
                    mask_path = f"output_mask_{method}_{os.path.basename(image_path)}"
                    save_base64_image(mask_b64, mask_path)
                    print(f"💾 Mask已保存: {mask_path}")

                print(f"\n✅ 智能抠图测试通过！")
                return True
            else:
                print(f"❌ 抠图失败: {data.get('error')}")
                if 'rembg' in data.get('error', ''):
                    print(f"\n💡 提示: rembg方法需要安装依赖库")
                    print(f"   安装命令: pip install rembg")
                return False
        else:
            print(f"❌ API返回错误: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False

    except Exception as e:
        print(f"❌ 抠图请求失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("="*70)
    print(" 智能抠图功能测试")
    print("="*70)

    # 检查命令行参数
    if len(sys.argv) < 2:
        print(f"\n使用方法: python test_smart_removal.py <image_path> [method]")
        print(f"\n方法选项:")
        print(f"  - auto: 自动选择（默认）")
        print(f"  - rembg: 使用rembg（推荐，需要安装）")
        print(f"  - edge: 使用边缘辅助方法")
        print(f"  - grabcut: 使用GrabCut")
        print(f"\n示例:")
        print(f"  python test_smart_removal.py photo.jpg")
        print(f"  python test_smart_removal.py photo.jpg rembg")
        print(f"  python test_smart_removal.py photo.jpg edge")

        # 尝试查找测试图像
        test_images = [f for f in os.listdir('.') if f.endswith(('.jpg', '.jpeg', '.png'))]
        if test_images:
            print(f"\n💡 发现 {len(test_images)} 张图像:")
            for img in test_images[:5]:
                size_kb = os.path.getsize(img) / 1024
                print(f"   - {img} ({size_kb:.1f} KB)")
            print(f"\n可以使用: python test_smart_removal.py <image_name>")
        sys.exit(0)

    image_path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'auto'

    test_smart_removal(image_path, method=method)

    print(f"\n{'='*70}\n")
