# -*- coding: utf-8 -*-
"""
测试YOLOv8车辆检测集成
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

def test_yolo_detection(image_path, api_url="http://localhost:5000"):
    """测试YOLOv8车辆检测API"""
    print(f"\n{'='*70}")
    print(f" 测试YOLOv8车辆检测集成")
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

    # 测试API端点
    print(f"\n🔍 测试1: 检查API健康状态")
    try:
        response = requests.get(f"{api_url}/api/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ API运行正常: {response.json()['message']}")
        else:
            print(f"❌ API状态异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到API: {e}")
        print(f"💡 请确保后端正在运行: python edge_detection_api.py")
        return False

    # 测试算法列表
    print(f"\n🔍 测试2: 获取算法列表")
    try:
        response = requests.get(f"{api_url}/api/algorithms", timeout=5)
        if response.status_code == 200:
            algorithms = response.json()['algorithms']
            print(f"✅ 可用算法: {algorithms}")
            if 'YOLOv8车辆检测' in algorithms:
                print(f"✅ YOLOv8车辆检测已添加到算法列表")
            else:
                print(f"⚠️ 警告: YOLOv8车辆检测未在算法列表中")
        else:
            print(f"❌ 获取算法列表失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")

    # 测试YOLOv8检测
    print(f"\n🔍 测试3: YOLOv8车辆检测")
    try:
        payload = {
            "image": img_base64,
            "algorithm": "YOLOv8车辆检测",
            "min_rectangularity": 0.2
        }

        print(f"⏳ 发送检测请求...")
        response = requests.post(f"{api_url}/api/detect", json=payload, timeout=60)

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ 检测成功！")
                print(f"\n📊 检测结果:")
                print(f"   - 算法: {data.get('algorithm')}")
                print(f"   - 分类结果: {data.get('classification')}")

                if 'total_vehicles' in data:
                    total = data['total_vehicles']
                    print(f"   - 检测到车辆: {total} 辆")

                if 'vehicle_counts' in data:
                    counts = data['vehicle_counts']
                    print(f"\n   车辆类型统计:")
                    for vehicle_type, count in counts.items():
                        print(f"      • {vehicle_type}: {count} 辆")

                if 'detections' in data and len(data['detections']) > 0:
                    detections = data['detections']
                    print(f"\n   检测详情 (显示前5个):")
                    for i, det in enumerate(detections[:5], 1):
                        print(f"      {i}. {det['class_name']} - 置信度: {det['confidence']:.2f} - 位置: {det['bbox']}")

                    if len(detections) > 5:
                        print(f"      ... 共 {len(detections)} 个检测")

                # 保存结果图像
                if 'images' in data and 'marked' in data['images']:
                    marked_b64 = data['images']['marked']
                    marked_data = base64.b64decode(marked_b64.split(',')[1])
                    output_path = "output_yolo_detection.png"
                    with open(output_path, 'wb') as f:
                        f.write(marked_data)
                    print(f"\n💾 结果图像已保存: {output_path}")

                print(f"\n✅ YOLOv8集成测试通过！")
                return True
            else:
                print(f"❌ 检测失败: {data.get('error')}")
                return False
        else:
            print(f"❌ API返回错误: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False

    except Exception as e:
        print(f"❌ 检测请求失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("="*70)
    print(" YOLOv8车辆检测集成测试")
    print("="*70)

    # 检查命令行参数
    if len(sys.argv) < 2:
        print(f"\n使用方法: python test_yolo_integration.py <image_path>")
        print(f"\n示例: python test_yolo_integration.py test_car.jpg")
        print(f"\n如果没有测试图像，请准备一张包含车辆的图像进行测试。")

        # 尝试查找测试图像
        test_images = ['test_car.jpg', 'car.jpg', 'test.jpg', 'images/car.png']
        found = False
        for test_img in test_images:
            if os.path.exists(test_img):
                print(f"\n💡 发现测试图像: {test_img}")
                choice = input("是否使用此图像进行测试？(y/n): ")
                if choice.lower() == 'y':
                    test_yolo_detection(test_img)
                    found = True
                    break

        if not found:
            print(f"\n❌ 未找到测试图像")
    else:
        image_path = sys.argv[1]
        test_yolo_detection(image_path)

    print(f"\n{'='*70}\n")
