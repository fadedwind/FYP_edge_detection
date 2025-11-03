import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

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

    # 算法执行
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
        for i in range(3):  # B/G/R通道
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

    # 延迟执行处理（滑动停止后300ms再更新）
    debounce_id = root.after(debounce_delay, process_image_realtime, algorithm)


# -------------------------- 前端UI函数 --------------------------
def create_parameter_window():
    """创建参数调节窗口（绑定实时回调）"""
    cv2.namedWindow("Parameters", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Parameters", 640, 300)
    # 给每个Trackbar绑定回调函数 on_param_change
    cv2.createTrackbar("Sobel_Ksize", "Parameters", 3, 7, on_param_change)
    cv2.createTrackbar("Blur", "Parameters", 7, 31, on_param_change)
    cv2.createTrackbar("Dilate", "Parameters", 5, 15, on_param_change)
    cv2.createTrackbar("Canny_Low", "Parameters", 50, 200, on_param_change)
    cv2.createTrackbar("Canny_High", "Parameters", 150, 300, on_param_change)
    cv2.createTrackbar("Area", "Parameters", 5000, 30000, on_param_change)


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
            messagebox.warning("警告", "不支持的文件格式！")
        # 更新文件路径显示
        file_label.config(text=f"已选择：{os.path.basename(file_path)}")
    else:
        file_type = ""
        file_label.config(text="未选择文件")


def start_process():
    """开始处理（图片：触发实时处理；视频：触发视频处理）"""
    if not file_path:
        messagebox.warning("警告", "请先选择文件！")
        return
    algorithm = algo_combobox.get()
    if not algorithm:
        messagebox.warning("警告", "请选择算法！")
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
    """更新图片处理结果显示"""
    if process_result["img_original"] is None:
        return

    # 转换OpenCV图像为Tkinter格式
    def cv2_to_tk(img):
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((280, 220), Image.LANCZOS)
        return ImageTk.PhotoImage(image=img_pil)

    # 显示图片
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
    """保存图片处理结果"""
    if process_result["img_edge"] is None:
        messagebox.warning("警告", "暂无处理结果可保存！")
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
    global root, algo_combobox, file_label, original_label, edge_label, contour_label, metrics_label
    root = tk.Tk()  # 全局root，供防抖回调使用
    root.title("边缘检测工具（实时参数调整版）")
    root.geometry("900x600")

    # 1. 顶部控制区
    control_frame = ttk.Frame(root, padding="10")
    control_frame.pack(fill=tk.X)

    # 算法选择
    algo_label = ttk.Label(control_frame, text="选择算法：")
    algo_label.grid(row=0, column=0, padx=5, pady=5)
    algo_options = ["Sobel", "彩色Sobel", "Canny", "彩色Canny", "Prewitt"]
    algo_combobox = ttk.Combobox(control_frame, values=algo_options, state="readonly")
    algo_combobox.grid(row=0, column=1, padx=5, pady=5)
    # 算法切换时自动触发图片处理
    algo_combobox.bind("<<ComboboxSelected>>", lambda e: process_image_realtime(algo_combobox.get()))

    # 按钮组
    select_btn = ttk.Button(control_frame, text="选择文件", command=select_file)
    select_btn.grid(row=0, column=2, padx=5, pady=5)
    process_btn = ttk.Button(control_frame, text="开始处理", command=start_process)
    process_btn.grid(row=0, column=3, padx=5, pady=5)
    save_btn = ttk.Button(control_frame, text="保存结果", command=save_result)
    save_btn.grid(row=0, column=4, padx=5, pady=5)

    # 文件路径显示
    file_label = ttk.Label(control_frame, text="未选择文件", wraplength=300)
    file_label.grid(row=0, column=5, padx=5, pady=5)

    # 2. 结果显示区
    result_frame = ttk.Frame(root, padding="10")
    result_frame.pack(fill=tk.BOTH, expand=True)

    # 图片显示标签
    original_label = ttk.Label(result_frame, text="原图")
    original_label.grid(row=0, column=0, padx=10, pady=10)
    edge_label = ttk.Label(result_frame, text="边缘检测结果")
    edge_label.grid(row=0, column=1, padx=10, pady=10)
    contour_label = ttk.Label(result_frame, text="轮廓检测结果")
    contour_label.grid(row=0, column=2, padx=10, pady=10)

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