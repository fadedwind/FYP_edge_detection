import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import tkinter.scrolledtext as scrolledtext
import json
import webbrowser
from datetime import datetime
import html

# -------------------------- 全局参数配置 --------------------------
frameWidth = 640
frameHeight = 480
file_path = ""  # 选择的文件路径
file_type = ""  # 文件类型：image/video
process_result = {"img_original": None, "img_edge": None, "img_contour": None, "metrics": None}

# 全局声明需要跨函数访问的UI组件
algo_combobox = None
file_label = None
original_label = None
edge_label = None
contour_label = None
metrics_label = None
pr_label = None

# 设置文件路径
SETTINGS_PATH = os.path.join(os.path.dirname(__file__), 'settings.json')

# 默认的高-precision 参数
DEFAULT_SETTINGS = {
    "algorithm": "Canny",
    "Blur": 7,
    "Sobel_Ksize": 3,
    "Dilate": 1,
    "Canny_Low": 100,
    "Canny_High": 220,
    "Area": 8000
}

# 预设集合（可扩展）
PRESETS = {
    "高精度": {
        "algorithm": "Canny",
        "Blur": 7,
        "Sobel_Ksize": 3,
        "Dilate": 1,
        "Canny_Low": 100,
        "Canny_High": 220,
        "Area": 8000
    },
    "快速": {
        "algorithm": "Sobel",
        "Blur": 3,
        "Sobel_Ksize": 3,
        "Dilate": 1,
        "Canny_Low": 50,
        "Canny_High": 150,
        "Area": 2000
    }
}

# 算法原理文档（简要版，UI 点击后显示）
ALGO_DOCS = {
    "Sobel": (
        "Sobel 算子用于近似图像的一阶导数（梯度），检测强度变化处的边缘。\n"
        "对灰度图像应用两个卷积核分别计算 x、y 方向的梯度：\n"
            "G_x = I * K_x, \\quad G_y = I * K_y\n"
        "其中常用的 3x3 Sobel 核为：\n"
            "K_x = [[-1,0,1],[-2,0,2],[-1,0,1]],\\quad K_y = K_x^T\n"
            "然后计算梯度幅值作为边强度：\n"
            "M = \\sqrt{G_x^2 + G_y^2}，通常使用近似 \\|G_x\\| + \\|G_y\\| 或 cv2.convertScaleAbs。\n"
        "Sobel 对噪声敏感，因此通常先做高斯平滑再计算梯度。\n"
    ),
    "彩色Sobel": (
        "对彩色图像在每个通道上分别计算 Sobel 梯度，然后将通道间的边响应取并集（bitwise_or）或加权合并，\n"
        "以保留那些仅在某一颜色通道上明显的边缘。优点是能检测颜色边界，缺点是可能增加噪声与假阳。\n"
    ),
    "Prewitt": (
        "Prewitt 算子与 Sobel 类似，也是通过卷积核计算 3x3 的水平和垂直梯度：\n"
    "K_x = [[-1,0,1],[-1,0,1],[-1,0,1]],\\quad K_y = K_x^T\n"
        "与 Sobel 不同的是核权重更均匀（Sobel 在中行有更高权重），对噪声更敏感但实现更简单。\n"
    ),
    "Canny": (
        "Canny 边缘检测是一种多阶段算法，步骤为：\n"
    "1) 高斯平滑以去噪：I_s = G_\\sigma * I。\n"
    "2) 计算梯度幅值与方向（通常用 Sobel）：G_x,G_y，幅值 M=\\sqrt{G_x^2+G_y^2}。\n"
        "3) 非极大值抑制（NMS）：沿梯度方向保留局部最大值，细化到 1 像素宽的边。\n"
        "4) 双阈值连接（hysteresis）：使用低阈值 t_l 和高阈值 t_h，将幅值>t_h 的像素标记为强边缘，介于 t_l 和 t_h 之间且与强边相连的像素标记为边缘，其他丢弃。\n"
        "Canny 的两个阈值控制了精确率/召回的 trade-off：提高 t_h 可以减少假阳（提高 precision），但可能降低 recall。\n"
    ),
    "彩色Canny": (
        "对每个颜色通道单独做 Canny（或对通道梯度幅值阈值化），再合并结果。保留颜色边的优点和噪声增加的缺点同样适用。\n"
    )
}

# 扩展说明（详细版），包含数学公式、评估指标与实践建议
EXTENDED_DOC = (
    "\n通用预处理：高斯平滑（去噪）\n"
    "在任何基于梯度的边缘检测前，通常先做高斯模糊：I_s(x,y)=G_\\sigma(x,y)*I(x,y)。\n"
    "作用：抑制高频噪声，减少假阳（FP），但过强模糊会抹去弱边（降低 TP → 降低 recall）。\n\n"
    "Sobel 算子（灰度）\n"
    "Sobel 近似图像的一阶导数。常见 3x3 卷积核：\n"
    "K_x=[[-1,0,1],[-2,0,2],[-1,0,1]], K_y=K_x^T。对平滑后的灰度图 I_s：G_x = I_s * K_x, G_y = I_s * K_y，梯度幅值 M = sqrt(G_x^2+G_y^2)。\n"
    "大 M 表示强边；常接二值阈值 M>t 判为边。Sobel 无 NMS，输出 '厚' 边带。\n\n"
    "Prewitt 算子\n"
    "与 Sobel 类似，核 K_x = [[-1,0,1],[-1,0,1],[-1,0,1]]。权重不同，Sobel 中心权更大，对噪声更鲁棒。\n\n"
    "彩色变体\n"
    "对每个通道单独计算响应 M_c，然后合并（例如按位或或取最大）。可检测颜色边，但容易增加噪声与假阳。\n\n"
    "Canny 边缘检测（多阶段）\n"
    "1) 高斯平滑：I_s = G_\\sigma * I。\n"
    "2) 梯度计算（Sobel）：G_x,G_y；幅值 M, 方向 θ。\n"
    "3) 非极大值抑制（NMS）：沿梯度方向保留局部最大值，将宽边变细。\n"
    "4) 双阈值 + 连接：使用低阈 t_l 和高阈 t_h，实现 hysteresis，既能连接边段也能滤掉孤立噪点。\n\n"
    "形态学（膨胀/腐蚀）\n"
    "膨胀（dilation）可扩展边缘并连接邻近碎片，但可能把噪点合并成假轮廓。腐蚀用于移除孤点。开/闭运算常用于去噪或填缝。\n\n"
    "轮廓检测与 Area 筛选\n"
    "通过 findContours 提取连通边界并按面积过滤（area < A_min 则丢弃），这是提高 precision 的常用手段。\n\n"
    "评估指标（像素级）\n"
    "TP = sum(D & R), FP = sum(D & ~R), FN = sum(~D & R)。Precision = TP/(TP+FP)，Recall = TP/(TP+FN)，F1 = 2PR/(P+R)。\n\n"
    "ODS / OIS\n"
    "ODS：选择全局阈值使得在数据集上平均 F1 最大；OIS：对每张图选择其最优阈值再平均。\n\n"
    "参数对 precision 的影响（总结）\n"
    "- Blur ↑ → 噪声↓ → FP↓ → precision↑；但弱边也被抹去 → recall↓。\n"
    "- Canny_High ↑ → 更严格 → FP↓、precision↑、recall↓。Low 与 High 的比例常取 0.3–0.5。\n"
    "- Area ↑ → 去除小噪点 → precision↑（但可能去掉真实小目标）。\n"
    "- Dilate ↑ → 连接断裂（可能 recall↑），也可能合并噪点（precision↓）。\n\n"
    "实践建议（可用于批量调参）\n"
    "使用批量 ODS/OIS 做小范围网格搜索（例如 Blur ∈ {3,5,7}, High ∈ {150,180,210}, Area ∈ {1000,5000}），选出在你任务上 trade-off 最佳的参数。"
)


def show_algorithm_docs():
    """弹出窗口显示当前选中算法的数学/实现原理说明。"""
    alg = None
    try:
        alg = algo_combobox.get()
    except Exception:
        pass
    if not alg:
        messagebox.showwarning("提示", "请先在界面选择一个算法（Algorithm）！")
        return

    doc = ALGO_DOCS.get(alg, "暂无该算法的说明。")
    win = tk.Toplevel(root)
    win.title(f"{alg} — 算法原理说明")
    win.geometry('700x500')
    txt = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=("Consolas", 11))
    txt.insert(tk.END, f"算法：{alg}\n\n")
    txt.insert(tk.END, doc)
    txt.config(state=tk.DISABLED)
    txt.pack(fill=tk.BOTH, expand=True)

    def open_docs_in_browser(a):
        try:
            repo_dir = os.path.dirname(__file__)
        except Exception:
            repo_dir = os.getcwd()
        docs_path = os.path.join(repo_dir, 'docs', 'edge_detection_docs.html')
        if not os.path.exists(docs_path):
            messagebox.showwarning('文档不存在', f'未找到静态文档：{docs_path}')
            return
        # 打开文档并跳转到对应锚点（HTML 里使用 id="Sobel" 等）
        anchor = a.replace(' ', '')
        webbrowser.open('file://' + docs_path + f'#{anchor}')

    # 在文档窗口添加打开浏览器按钮
    btn_frame = ttk.Frame(win)
    btn_frame.pack(fill=tk.X, padx=6, pady=6)
    open_btn = ttk.Button(btn_frame, text='在浏览器中打开完整文档', command=lambda: open_docs_in_browser(alg))
    open_btn.pack(side=tk.RIGHT)


def apply_preset(name: str):
    """应用预设：设置 Trackbar、算法选择，触发一次处理并保存设置。"""
    preset = PRESETS.get(name)
    if not preset:
        messagebox.showwarning("提示", f"未找到预设：{name}")
        return

    # 设置算法选择
    try:
        algo_combobox.set(preset.get('algorithm', DEFAULT_SETTINGS['algorithm']))
    except Exception:
        pass

    # 设置 Trackbar 值（如果参数窗口存在）
    try:
        cv2.setTrackbarPos('Sobel_Ksize', 'Parameters', int(preset.get('Sobel_Ksize', DEFAULT_SETTINGS['Sobel_Ksize'])))
        cv2.setTrackbarPos('Blur', 'Parameters', int(preset.get('Blur', DEFAULT_SETTINGS['Blur'])))
        cv2.setTrackbarPos('Dilate', 'Parameters', int(preset.get('Dilate', DEFAULT_SETTINGS['Dilate'])))
        cv2.setTrackbarPos('Canny_Low', 'Parameters', int(preset.get('Canny_Low', DEFAULT_SETTINGS['Canny_Low'])))
        cv2.setTrackbarPos('Canny_High', 'Parameters', int(preset.get('Canny_High', DEFAULT_SETTINGS['Canny_High'])))
        cv2.setTrackbarPos('Area', 'Parameters', int(preset.get('Area', DEFAULT_SETTINGS['Area'])))
    except Exception:
        # 如果参数窗口不存在或设置失败，则忽略（窗口通常已创建）
        pass

    # 触发处理并保存设置
    try:
        alg = preset.get('algorithm', DEFAULT_SETTINGS['algorithm'])
        process_image_realtime(alg)
        save_current_params()
    except Exception as e:
        print('应用预设失败：', e)


def load_settings():
    """从 settings.json 加载参数（若不存在则返回默认值）"""
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 兼容缺少字段的旧文件
                for k, v in DEFAULT_SETTINGS.items():
                    if k not in data:
                        data[k] = v
                return data
    except Exception:
        pass
    return DEFAULT_SETTINGS.copy()


def save_settings(data: dict):
    """将参数保存到 settings.json（覆盖）"""
    try:
        with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("保存设置失败：", e)


def save_current_params():
    """读取当前 Trackbar/算法设置并保存（写入 settings.json）"""
    try:
        settings = {}
        settings['algorithm'] = algo_combobox.get() if algo_combobox else DEFAULT_SETTINGS['algorithm']
        # 当参数窗口不存在时，getTrackbarPos 会抛错；捕获并跳过
        try:
            settings['Blur'] = cv2.getTrackbarPos('Blur', 'Parameters')
            settings['Sobel_Ksize'] = cv2.getTrackbarPos('Sobel_Ksize', 'Parameters')
            settings['Dilate'] = cv2.getTrackbarPos('Dilate', 'Parameters')
            settings['Canny_Low'] = cv2.getTrackbarPos('Canny_Low', 'Parameters')
            settings['Canny_High'] = cv2.getTrackbarPos('Canny_High', 'Parameters')
            settings['Area'] = cv2.getTrackbarPos('Area', 'Parameters')
        except Exception:
            # 若获取失败则使用默认/已保存的值
            loaded = load_settings()
            for k in ['Blur', 'Sobel_Ksize', 'Dilate', 'Canny_Low', 'Canny_High', 'Area']:
                settings[k] = loaded.get(k, DEFAULT_SETTINGS[k])

        save_settings(settings)
    except Exception as e:
        print("保存当前参数失败：", e)


def process_and_save(algorithm):
    """用于防抖延迟调用：处理图像并保存当前参数"""
    process_image_realtime(algorithm)
    save_current_params()

# 防抖相关：避免Trackbar滑动时频繁触发处理
debounce_id = None  # 延迟任务ID
debounce_delay = 300  # 防抖延迟（毫秒）：滑动停止后300ms再更新

# -------------------------- 算法核心函数 --------------------------
def empty(a):
    pass


def getContours(img, imgContour):
    """轮廓检测（保留原功能）"""
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgContour, f"Points: {len(approx)}", (x + w + 10, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(imgContour, f"Area: {int(area)}", (x + w + 10, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def get_reference_edge(img):
    """生成参考边缘（作为Ground Truth，用于计算评估指标）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reference_edge = cv2.Canny(gray, 150, 255)  # 高阈值保证参考边缘可靠性
    return reference_edge


def calculate_metrics(detected_edge, reference_edge):
    """计算评估指标：Precision、Recall、F1-Score"""
    _, detected = cv2.threshold(detected_edge, 127, 255, cv2.THRESH_BINARY)
    _, reference = cv2.threshold(reference_edge, 127, 255, cv2.THRESH_BINARY)

    TP = cv2.bitwise_and(detected, reference).sum() // 255
    FP = cv2.bitwise_and(detected, cv2.bitwise_not(reference)).sum() // 255
    FN = cv2.bitwise_and(cv2.bitwise_not(detected), reference).sum() // 255

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return round(precision, 3), round(recall, 3), round(f1, 3)


def read_params():
    """读取Trackbar参数（独立提取，方便复用）"""
    # 模糊核（确保奇数）
    blur_ksize = cv2.getTrackbarPos("Blur", "Parameters")
    blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    if blur_ksize < 1:
        blur_ksize = 1

    # Sobel核（确保奇数且≥1）
    sobel_ksize = cv2.getTrackbarPos("Sobel_Ksize", "Parameters")
    sobel_ksize = sobel_ksize if (sobel_ksize % 2 == 1 and sobel_ksize >= 1) else 3

    # Canny阈值
    canny_low = cv2.getTrackbarPos("Canny_Low", "Parameters")
    canny_high = cv2.getTrackbarPos("Canny_High", "Parameters")
    if canny_low > canny_high:  # 避免低阈值>高阈值
        canny_low, canny_high = canny_high, canny_low

    # 膨胀核（确保奇数且≥1）
    dilate_ksize = cv2.getTrackbarPos("Dilate", "Parameters")
    dilate_ksize = dilate_ksize if (dilate_ksize % 2 == 1 and dilate_ksize >= 1) else 5

    return blur_ksize, sobel_ksize, canny_low, canny_high, dilate_ksize


def process_image_realtime(algorithm):
    """实时处理图片（供Trackbar回调和开始按钮调用）"""
    global process_result
    if file_type != "image" or not file_path:
        return  # 仅处理图片文件

    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("错误", "无法读取图片文件！")
        return

    img = cv2.resize(img, (frameWidth, frameHeight))
    reference_edge = get_reference_edge(img)
    blur_ksize, sobel_ksize, canny_low, canny_high, dilate_ksize = read_params()
    # 算法执行（与视频处理相同的分支，但在静态图片上）
    img_edge = None
    if algorithm == "Sobel":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
        grad_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        grad_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        img_edge = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
    elif algorithm == "彩色Sobel":
        imgBlur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 1)
        sobel_edges = []
        for i in range(3):
            grad_x = cv2.Sobel(imgBlur[:, :, i], cv2.CV_64F, 1, 0, ksize=sobel_ksize)
            grad_y = cv2.Sobel(imgBlur[:, :, i], cv2.CV_64F, 0, 1, ksize=sobel_ksize)
            sobel_edges.append(cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y)))
        img_edge = cv2.bitwise_or(sobel_edges[0], sobel_edges[1])
        img_edge = cv2.bitwise_or(img_edge, sobel_edges[2])
    elif algorithm == "Canny":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
        img_edge = cv2.Canny(gray_blur, canny_low, canny_high)
    elif algorithm == "彩色Canny":
        imgBlur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 1)
        canny_edges = [cv2.Canny(imgBlur[:, :, i], canny_low, canny_high) for i in range(3)]
        img_edge = cv2.bitwise_or(canny_edges[0], canny_edges[1])
        img_edge = cv2.bitwise_or(img_edge, canny_edges[2])
    elif algorithm == "Prewitt":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        grad_x = cv2.filter2D(gray_blur, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(gray_blur, cv2.CV_64F, kernel_y)
        img_edge = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))

    # 膨胀边缘
    kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
    img_edge = cv2.dilate(img_edge, kernel, iterations=1)

    # 轮廓检测+指标计算
    img_contour = img.copy()
    getContours(img_edge, img_contour)
    precision, recall, f1 = calculate_metrics(img_edge, reference_edge)

    process_result = {
        "img_original": img,
        "img_edge": img_edge,
        "img_contour": img_contour,
        "metrics": (precision, recall, f1)
    }

    update_result_display()


def process_video(algorithm):
    """处理视频（仍需手动触发，不支持实时参数调整）"""
    global process_result
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        messagebox.showerror("错误", "无法读取视频文件！")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 保存视频
    output_dir = "video_output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"processed_{algorithm}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 读取一次参数（视频处理时参数固定，不实时更新）
    blur_ksize, sobel_ksize, canny_low, canny_high, dilate_ksize = read_params()
    metrics_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (frameWidth, frameHeight))
        reference_edge = get_reference_edge(frame_resized)

        # 算法执行（参数固定）
        img_edge = None
        if algorithm == "Sobel":
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
            grad_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
            grad_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
            img_edge = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
        elif algorithm == "彩色Sobel":
            imgBlur = cv2.GaussianBlur(frame_resized, (blur_ksize, blur_ksize), 1)
            sobel_edges = []
            for i in range(3):
                grad_x = cv2.Sobel(imgBlur[:, :, i], cv2.CV_64F, 1, 0, ksize=sobel_ksize)
                grad_y = cv2.Sobel(imgBlur[:, :, i], cv2.CV_64F, 0, 1, ksize=sobel_ksize)
                sobel_edges.append(cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y)))
            img_edge = cv2.bitwise_or(sobel_edges[0], sobel_edges[1])
            img_edge = cv2.bitwise_or(img_edge, sobel_edges[2])
        elif algorithm == "Canny":
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
            img_edge = cv2.Canny(gray_blur, canny_low, canny_high)
        elif algorithm == "彩色Canny":
            imgBlur = cv2.GaussianBlur(frame_resized, (blur_ksize, blur_ksize), 1)
            canny_edges = [cv2.Canny(imgBlur[:, :, i], canny_low, canny_high) for i in range(3)]
            img_edge = cv2.bitwise_or(canny_edges[0], canny_edges[1])
            img_edge = cv2.bitwise_or(img_edge, canny_edges[2])
        elif algorithm == "Prewitt":
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
            grad_x = cv2.filter2D(gray_blur, cv2.CV_64F, kernel_x)
            grad_y = cv2.filter2D(gray_blur, cv2.CV_64F, kernel_y)
            img_edge = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))

        # 膨胀+轮廓检测
        kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
        img_edge = cv2.dilate(img_edge, kernel, iterations=1)
        frame_contour = frame_resized.copy()
        getContours(img_edge, frame_contour)

        # 计算指标+写入视频
        precision, recall, f1 = calculate_metrics(img_edge, reference_edge)
        metrics_list.append((precision, recall, f1))
        # 缩放回原视频尺寸写入（保持视频比例）
        frame_contour = cv2.resize(frame_contour, (width, height))
        out.write(frame_contour)

        cv2.imshow("Video Processing (Press 'q' to stop)", frame_contour)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyWindow("Video Processing (Press 'q' to stop)")

    # 显示平均指标
    avg_precision = round(np.mean([m[0] for m in metrics_list]), 3)
    avg_recall = round(np.mean([m[1] for m in metrics_list]), 3)
    avg_f1 = round(np.mean([m[2] for m in metrics_list]), 3)
    process_result["metrics"] = (avg_precision, avg_recall, avg_f1)
    metrics_label.config(text=f"视频平均指标：\nPrecision: {avg_precision}\nRecall: {avg_recall}\nF1-Score: {avg_f1}")
    messagebox.showinfo("成功",
                        f"视频处理完成！\n保存路径：{output_path}\n平均指标：P={avg_precision}, R={avg_recall}, F1={avg_f1}")


def compute_edge_strength(img, algorithm, blur_ksize, sobel_ksize):
    """生成单通道的边强度图（0-255 uint8），供阈值化用于 ODS/OIS 计算"""
    # 接受输入为 BGR 彩色图或灰度图（ndarray）
    edge = None
    if algorithm == "Sobel":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
        grad_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        grad_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        edge = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))

    elif algorithm == "彩色Sobel":
        imgBlur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 1)
        sobel_edges = []
        for i in range(3):
            grad_x = cv2.Sobel(imgBlur[:, :, i], cv2.CV_64F, 1, 0, ksize=sobel_ksize)
            grad_y = cv2.Sobel(imgBlur[:, :, i], cv2.CV_64F, 0, 1, ksize=sobel_ksize)
            sobel_edges.append(cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y)))
        edge = cv2.bitwise_or(sobel_edges[0], sobel_edges[1])
        edge = cv2.bitwise_or(edge, sobel_edges[2])

    elif algorithm == "Canny":
        # Canny 本身是二值，但作为强度我们用高斯平滑后的灰度梯度幅值
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
        gx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
        edge = cv2.convertScaleAbs(cv2.magnitude(gx, gy))

    elif algorithm == "彩色Canny":
        imgBlur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 1)
        canny_edges = []
        for i in range(3):
            gx = cv2.Sobel(imgBlur[:, :, i], cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(imgBlur[:, :, i], cv2.CV_64F, 0, 1, ksize=3)
            canny_edges.append(cv2.convertScaleAbs(cv2.magnitude(gx, gy)))
        edge = cv2.bitwise_or(canny_edges[0], canny_edges[1])
        edge = cv2.bitwise_or(edge, canny_edges[2])

    elif algorithm == "Prewitt":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        grad_x = cv2.filter2D(gray_blur, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(gray_blur, cv2.CV_64F, kernel_y)
        edge = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))

    # 保证返回 uint8 单通道
    if edge is None:
        edge = np.zeros((frameHeight, frameWidth), dtype=np.uint8)
    return edge


def batch_process_directory():
    """对指定目录下的所有图片运行检测并计算 ODS/OIS 指标，结果写入 CSV 文件"""
    import csv
    from datetime import datetime
    folder = filedialog.askdirectory(title="选择包含图片的目录")
    if not folder:
        return

    # 收集图片文件
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    if not files:
        messagebox.showwarning("警告", "该目录下没有支持的图片文件！")
        return

    algorithm = algo_combobox.get()
    if not algorithm:
        messagebox.showwarning("警告", "请先在界面选择算法（Algorithm）！")
        return

    # 准备输出目录：repository/output/<timestamp>/ （提前创建，供循环中保存每张图）
    try:
        repo_dir = os.path.dirname(__file__)
    except Exception:
        repo_dir = os.getcwd()
    output_root = os.path.join(repo_dir, 'output')
    os.makedirs(output_root, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_dir = os.path.join(output_root, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)

    # 读取参数（固定）
    blur_ksize, sobel_ksize, canny_low, canny_high, dilate_ksize = read_params()

    # 阈值列表（用于搜索 ODS/OIS）
    # 将步长设为 1 以获得更细致的 PR 曲线（注意：阈值步长越小，计算时间越长）
    thresholds = list(range(0, 256, 1))

    per_image_best = []  # 存放每张图像的 OIS（best per-image）数据
    # 用于 ODS：记录每个阈值在所有图像上的累积 precision/recall/f1
    thr_precision_sum = np.zeros(len(thresholds), dtype=np.float64)
    thr_recall_sum = np.zeros(len(thresholds), dtype=np.float64)
    thr_f1_sum = np.zeros(len(thresholds), dtype=np.float64)

    for fp in files:
        img = cv2.imread(fp)
        if img is None:
            continue
        img = cv2.resize(img, (frameWidth, frameHeight))
        reference = get_reference_edge(img)
        edge_strength = compute_edge_strength(img, algorithm, blur_ksize, sobel_ksize)

        best_f1 = -1.0
        best_prec = best_rec = best_thr = 0

        # 对每个阈值计算指标
        for i, t in enumerate(thresholds):
            _, detected = cv2.threshold(edge_strength, t, 255, cv2.THRESH_BINARY)
            p, r, f = calculate_metrics(detected, reference)
            thr_precision_sum[i] += p
            thr_recall_sum[i] += r
            thr_f1_sum[i] += f
            if f > best_f1:
                best_f1 = f
                best_prec = p
                best_rec = r
                best_thr = t

        per_image_best.append((os.path.basename(fp), best_thr, best_prec, best_rec, best_f1))

        # 保存该图像在最佳阈值下的边缘图与轮廓图到 output/<timestamp>/
        try:
            # 生成二值边缘和轮廓图
            _, detected_final = cv2.threshold(edge_strength, best_thr, 255, cv2.THRESH_BINARY)
            contour_img = img.copy()
            getContours(detected_final, contour_img)
            # 保存文件名
            base_name = os.path.splitext(os.path.basename(fp))[0]
            edge_save_name = f"{base_name}_edge.png"
            contour_save_name = f"{base_name}_contour.png"
            cv2.imwrite(os.path.join(run_output_dir, edge_save_name), detected_final)
            cv2.imwrite(os.path.join(run_output_dir, contour_save_name), contour_img)
        except Exception:
            # 若保存失败，不影响整体处理
            pass

    n = len(per_image_best)
    if n == 0:
        messagebox.showwarning("警告", "未处理到任何图片（可能格式不支持或读取失败）！")
        return

    # 计算 ODS：在所有阈值上取平均 F1，选择平均F1最大的阈值
    mean_f1_per_thr = thr_f1_sum / n
    best_idx = int(np.argmax(mean_f1_per_thr))
    ods_thr = thresholds[best_idx]
    ods_prec = thr_precision_sum[best_idx] / n
    ods_rec = thr_recall_sum[best_idx] / n
    ods_f1 = thr_f1_sum[best_idx] / n

    # 计算 OIS：对每张图的最佳值取平均
    ois_prec = np.mean([x[2] for x in per_image_best])
    ois_rec = np.mean([x[3] for x in per_image_best])
    ois_f1 = np.mean([x[4] for x in per_image_best])


    # 生成并显示 PR 曲线（平均 precision-recall across thresholds）
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        precision_mean = thr_precision_sum / n
        recall_mean = thr_recall_sum / n
        # 为绘图按 recall 升序排序并做简单插值以获得更平滑的曲线
        idxs = np.argsort(recall_mean)
        recall_sorted = recall_mean[idxs]
        precision_sorted = precision_mean[idxs]

        # 创建更细致的 x 轴点并插值 precision
        try:
            recall_fine = np.linspace(recall_sorted.min(), recall_sorted.max(), 512)
            precision_fine = np.interp(recall_fine, recall_sorted, precision_sorted)
        except Exception:
            recall_fine = recall_sorted
            precision_fine = precision_sorted

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(recall_fine, precision_fine, '-', linewidth=1.5)
        ax.plot(recall_sorted, precision_sorted, 'o', markersize=3, alpha=0.6)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'PR Curve ({algorithm})')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.4)

        pr_path = os.path.join(run_output_dir, f"PR_curve_{algorithm}.png")
        fig.tight_layout()
        fig.savefig(pr_path, dpi=100)
        plt.close(fig)

        # 在 GUI 中显示 PR 曲线
        try:
            pr_img = Image.open(pr_path)
            pr_img = pr_img.resize((280, 220), Image.LANCZOS)
            pr_tk = ImageTk.PhotoImage(pr_img)
            pr_label.config(image=pr_tk)
            pr_label.image = pr_tk
        except Exception:
            # 如果 GUI 无法加载图片，则忽略（但文件已保存）
            pass
    except Exception as e:
        # matplotlib 不可用则提示用户但仍继续（保留 CSV）
        print("matplotlib not available or plotting failed:", e)
        messagebox.showwarning("绘图失败", "未能生成 PR 曲线 (需要 matplotlib)。请安装 matplotlib 或检查错误。")

    # 保存 CSV 报告到 run_output_dir
    out_csv = os.path.join(run_output_dir, f"batch_metrics_{algorithm}.csv")
    try:
        with open(out_csv, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.writer(cf)
            writer.writerow(["filename", "best_threshold", "best_precision", "best_recall", "best_f1"])
            for row in per_image_best:
                writer.writerow(row)
            writer.writerow([])
            writer.writerow(["ODS_threshold", ods_thr])
            writer.writerow(["ODS_precision", round(ods_prec, 3)])
            writer.writerow(["ODS_recall", round(ods_rec, 3)])
            writer.writerow(["ODS_f1", round(ods_f1, 3)])
            writer.writerow(["OIS_precision", round(ois_prec, 3)])
            writer.writerow(["OIS_recall", round(ois_rec, 3)])
            writer.writerow(["OIS_f1", round(ois_f1, 3)])
            if 'pr_path' in locals():
                writer.writerow(["PR_curve", pr_path])
    except Exception as e:
        print('保存 CSV 失败：', e)
    # 在 GUI 上显示简短结果并弹窗
    metrics_label.config(text=f"Batch ODS F1: {round(ods_f1,3)} | OIS F1: {round(ois_f1,3)}")
    messagebox.showinfo("批量处理完成", f"已处理 {n} 张图片\nCSV结果：{out_csv}\nODS_F1={round(ods_f1,3)}, OIS_F1={round(ois_f1,3)}")


# -------------------------- 实时参数回调+防抖 --------------------------
def on_param_change(val):
    """Trackbar参数变化时的回调函数（带防抖）"""
    global debounce_id
    algorithm = algo_combobox.get()
    if not algorithm or file_type != "image":
        return

    # 取消之前的延迟任务（防抖核心）
    if debounce_id is not None:
        root.after_cancel(debounce_id)

    # 延迟执行处理（滑动停止后300ms再更新并保存当前参数）
    debounce_id = root.after(debounce_delay, process_and_save, algorithm)


# -------------------------- 前端UI函数 --------------------------
def create_parameter_window():
    """创建参数调节窗口（绑定实时回调）"""
    cv2.namedWindow("Parameters", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Parameters", 640, 300)
    # 给每个Trackbar绑定回调函数 on_param_change
    # 尝试加载上次保存的参数作为初始值
    settings = load_settings()
    cv2.createTrackbar("Sobel_Ksize", "Parameters", int(settings.get('Sobel_Ksize', 3)), 7, on_param_change)
    cv2.createTrackbar("Blur", "Parameters", int(settings.get('Blur', 7)), 31, on_param_change)
    cv2.createTrackbar("Dilate", "Parameters", int(settings.get('Dilate', 1)), 15, on_param_change)
    cv2.createTrackbar("Canny_Low", "Parameters", int(settings.get('Canny_Low', 100)), 300, on_param_change)
    cv2.createTrackbar("Canny_High", "Parameters", int(settings.get('Canny_High', 220)), 300, on_param_change)
    cv2.createTrackbar("Area", "Parameters", int(settings.get('Area', 8000)), 30000, on_param_change)


def select_file():
    """选择文件（图片选中后自动触发一次实时处理）"""
    global file_path, file_type
    file_types = [
        ("所有支持文件", "*.jpg *.png *.bmp *.mp4 *.avi *.mov"),
        ("图片文件", "*.jpg *.png *.bmp"),
        ("视频文件", "*.mp4 *.avi *.mov")
    ]
    file_path = filedialog.askopenfilename(title="选择文件", filetypes=file_types)
    if file_path:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".jpg", ".png", ".bmp"]:
            file_type = "image"
            # 图片选中后自动触发一次处理（初始化显示）
            algorithm = algo_combobox.get()
            if algorithm:
                process_image_realtime(algorithm)
        elif ext in [".mp4", ".avi", ".mov"]:
            file_type = "video"
            # 切换到视频时清空图片结果显示
            original_label.config(image='')
            edge_label.config(image='')
            contour_label.config(image='')
            metrics_label.config(text="Precision: --\nRecall: --\nF1-Score: --")
        else:
            file_type = ""
            messagebox.showwarning("警告", "不支持的文件格式！")
        # 更新文件路径显示
        file_label.config(text=f"已选择：{os.path.basename(file_path)}")
    else:
        file_type = ""
        file_label.config(text="未选择文件")


def start_process():
    """开始处理（图片：触发实时处理；视频：触发视频处理）"""
    if not file_path:
        messagebox.showwarning("警告", "请先选择文件！")
        return
    algorithm = algo_combobox.get()
    if not algorithm:
        messagebox.showwarning("警告", "请选择算法！")
        return

    # 确保参数窗口已创建
    if not cv2.getWindowProperty("Parameters", cv2.WND_PROP_VISIBLE):
        create_parameter_window()

    # 区分文件类型处理
    if file_type == "image":
        process_image_realtime(algorithm)  # 图片直接调用实时处理函数
    elif file_type == "video":
        process_video(algorithm)  # 视频调用专门的处理函数


def update_result_display():
    """更新圖片處理結果顯示"""
    if process_result["img_original"] is None:
        return

    # 根据 result_frame 动态计算显示尺寸，保持长宽比，避免畸变
    try:
        rw = result_frame.winfo_width()
        rh = result_frame.winfo_height()
    except Exception:
        rw, rh = 900, 400

    # 目标每张图像的显示区域：三列布局，顶部为图片，占据大部分高度
    target_w = max(160, int((rw - 60) / 3))
    target_h = max(120, int((rh - 40) / 2))

    def cv2_to_tk(img, tw=target_w, th=target_h):
        if img is None:
            return None
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        # 保持长宽比缩放到 fit 区域
        img_pil.thumbnail((tw, th), Image.LANCZOS)
        # 如果需要填充以保持布局一致，可以创建背景，但这里直接返回缩放图
        return ImageTk.PhotoImage(image=img_pil)

    # 显示圖片
    img_original_tk = cv2_to_tk(process_result["img_original"]) 
    img_edge_tk = cv2_to_tk(process_result["img_edge"]) 
    img_contour_tk = cv2_to_tk(process_result["img_contour"]) 

    original_label.config(image=img_original_tk)
    original_label.image = img_original_tk
    edge_label.config(image=img_edge_tk)
    edge_label.image = img_edge_tk
    contour_label.config(image=img_contour_tk)
    contour_label.image = img_contour_tk

    # 显示指标
    if process_result["metrics"] is not None:
        precision, recall, f1 = process_result["metrics"]
        metrics_label.config(text=f"Precision: {precision}\nRecall: {recall}\nF1-Score: {f1}")
    else:
        metrics_label.config(text="Precision: --\nRecall: --\nF1-Score: --")


def save_result():
    """保存圖片處理結果"""
    if process_result["img_edge"] is None:
        messagebox.showwarning("警告", "暂无处理结果可保存！")
        return

    save_path = filedialog.asksaveasfilename(
        title="保存边缘检测结果",
        defaultextension=".png",
        filetypes=[("PNG图片", "*.png"), ("JPG图片", "*.jpg"), ("所有文件", "*.*")]
    )
    if save_path:
        cv2.imwrite(save_path, process_result["img_edge"])
        contour_save_path = os.path.splitext(save_path)[0] + "_contour.png"
        cv2.imwrite(contour_save_path, process_result["img_contour"])
        messagebox.showinfo("成功", f"结果已保存！\n边缘图：{save_path}\n轮廓图：{contour_save_path}")


# -------------------------- 初始化前端界面 --------------------------
def init_gui():
    global root, algo_combobox, file_label, original_label, edge_label, contour_label, metrics_label, result_frame
    root = tk.Tk()  # 全局root，供防抖回调使用
    root.title("边缘检测工具（实时参数调整版）")
    root.geometry("900x600")

    # 1. 顶部控制区（自动折行布局，窗口宽度不足时会换行而不是隐藏按钮）
    control_frame = ttk.Frame(root, padding="6")
    control_frame.pack(fill=tk.X)

    # 自动折行布局函数（将 control_frame 的子控件以流式布局排列）
    def flow_layout(event=None):
        try:
            width = control_frame.winfo_width()
        except Exception:
            width = root.winfo_width()
        if width <= 10:
            return
        padding_x = 5
        padding_y = 5
        x = padding_x
        y = padding_y
        row_h = 0
        for child in control_frame.winfo_children():
            # 使用请求宽度来布局
            child.update_idletasks()
            w = child.winfo_reqwidth()
            h = child.winfo_reqheight()
            if x + w + padding_x > width:
                # 换行
                x = padding_x
                y += row_h + padding_y
                row_h = h
            child.place(x=x, y=y)
            x += w + padding_x
            row_h = max(row_h, h)
        # 调整 control_frame 的高度以包下子控件
        control_frame.config(height=y + row_h + padding_y)

    # 在 root 和 control_frame 大小变化时重新布局
    control_frame.bind('<Configure>', flow_layout)
    root.bind('<Configure>', flow_layout)

    # 算法选择
    algo_label = ttk.Label(control_frame, text="选择算法：")
    algo_label.grid(row=0, column=0, padx=5, pady=5)
    algo_options = ["Sobel", "彩色Sobel", "Canny", "彩色Canny", "Prewitt"]
    algo_combobox = ttk.Combobox(control_frame, values=algo_options, state="readonly")
    algo_combobox.grid(row=0, column=1, padx=5, pady=5)
    # 算法切换时自动触发图片处理并保存当前设置
    def on_algo_change(event):
        alg = algo_combobox.get()
        if alg:
            process_image_realtime(alg)
            save_current_params()

    algo_combobox.bind("<<ComboboxSelected>>", on_algo_change)

    # 加载上次保存的算法选择
    loaded_settings = load_settings()
    try:
        algo_combobox.set(loaded_settings.get('algorithm', DEFAULT_SETTINGS['algorithm']))
    except Exception:
        pass

    # 按钮组
    select_btn = ttk.Button(control_frame, text="选择文件", command=select_file)
    select_btn.grid(row=0, column=2, padx=5, pady=5)
    process_btn = ttk.Button(control_frame, text="开始处理", command=start_process)
    process_btn.grid(row=0, column=3, padx=5, pady=5)
    save_btn = ttk.Button(control_frame, text="保存结果", command=save_result)
    save_btn.grid(row=0, column=4, padx=5, pady=5)
    batch_btn = ttk.Button(control_frame, text="批量处理目录", command=batch_process_directory)
    batch_btn.grid(row=0, column=6, padx=5, pady=5)

    # 预设选择与应用
    preset_label = ttk.Label(control_frame, text="预设：")
    preset_label.grid(row=0, column=7, padx=5, pady=5)
    preset_names = list(PRESETS.keys())
    preset_combobox = ttk.Combobox(control_frame, values=preset_names, state="readonly")
    preset_combobox.grid(row=0, column=8, padx=5, pady=5)
    def apply_preset_ui():
        name = preset_combobox.get()
        if not name:
            messagebox.showwarning("提示", "请选择一个预设后再应用！")
            return
        apply_preset(name)

    apply_preset_btn = ttk.Button(control_frame, text="应用预设", command=apply_preset_ui)
    apply_preset_btn.grid(row=0, column=9, padx=5, pady=5)

    reset_btn = ttk.Button(control_frame, text="恢复默认", command=lambda: apply_preset('高精度'))
    reset_btn.grid(row=0, column=10, padx=5, pady=5)
    docs_btn = ttk.Button(control_frame, text="算法原理", command=show_algorithm_docs)
    docs_btn.grid(row=0, column=11, padx=5, pady=5)

    # 文件路径显示
    file_label = ttk.Label(control_frame, text="未选择文件", wraplength=300)
    file_label.grid(row=0, column=5, padx=5, pady=5)

    # 2. 结果显示区
    result_frame = ttk.Frame(root, padding="10")
    result_frame.pack(fill=tk.BOTH, expand=True)
    # 使 result_frame 内的三列在窗口拉伸时均分宽度
    try:
        result_frame.columnconfigure(0, weight=1)
        result_frame.columnconfigure(1, weight=1)
        result_frame.columnconfigure(2, weight=1)
    except Exception:
        pass

    # 图片显示标签
    original_label = ttk.Label(result_frame, text="原图")
    original_label.grid(row=0, column=0, padx=10, pady=10)
    edge_label = ttk.Label(result_frame, text="边缘检测结果")
    edge_label.grid(row=0, column=1, padx=10, pady=10)
    contour_label = ttk.Label(result_frame, text="轮廓检测结果")
    contour_label.grid(row=0, column=2, padx=10, pady=10)

    # PR 曲线显示（批量处理结果）
    global pr_label
    pr_label = ttk.Label(result_frame, text="PR Curve")
    pr_label.grid(row=1, column=2, padx=10, pady=10)

    # 当窗口尺寸改变时，防抖更新显示，避免图片在快速拖动窗口时频繁重绘
    global resize_id
    resize_id = None
    def on_window_resize(event):
        global resize_id
        try:
            if resize_id is not None:
                root.after_cancel(resize_id)
        except Exception:
            pass
        resize_id = root.after(200, update_result_display)

    root.bind('<Configure>', on_window_resize)

    # 3. 评估指标显示区
    metrics_frame = ttk.Frame(root, padding="10")
    metrics_frame.pack(fill=tk.X)
    metrics_label = ttk.Label(metrics_frame, text="Precision: --\nRecall: --\nF1-Score: --", font=("Arial", 12))
    metrics_label.pack(padx=10, pady=5, anchor=tk.W)

    # 初始化参数窗口
    create_parameter_window()

    root.mainloop()


# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    init_gui()
