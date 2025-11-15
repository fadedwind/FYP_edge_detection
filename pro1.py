import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk


# ------------------------------------------------------------------------------
# 1. 边缘检测算法（保持4种可选，优化边缘连贯性）
# ------------------------------------------------------------------------------
def canny_edge_detection(gray_img):
    """普通Canny：适合光线均匀的车辆图片"""
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 1.5)  # 增强降噪，适配车辆复杂背景
    edges = cv2.Canny(blurred, 60, 180)
    # 形态学闭运算+膨胀：连接断裂边缘，让车身轮廓更完整
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel_dilate)
    return edges


def color_canny_edge_detection(rgb_img):
    """彩色Canny：适合颜色鲜明的车辆（如红色、蓝色车身）"""
    r, g, b = cv2.split(rgb_img)
    r_edges = cv2.Canny(cv2.GaussianBlur(r, (5, 5), 1.5), 60, 180)
    g_edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 1.5), 60, 180)
    b_edges = cv2.Canny(cv2.GaussianBlur(b, (5, 5), 1.5), 60, 180)
    color_edges = cv2.bitwise_or(cv2.bitwise_or(r_edges, g_edges), b_edges)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    color_edges = cv2.morphologyEx(color_edges, cv2.MORPH_CLOSE, kernel_close)
    color_edges = cv2.morphologyEx(color_edges, cv2.MORPH_DILATE, kernel_dilate)
    return color_edges


def sobel_edge_detection(gray_img):
    """普通Sobel：适合车身边缘清晰的图片"""
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)  # 重点检测水平边缘（车身轮廓）
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
    sobel_x = np.uint8(np.absolute(sobel_x))
    sobel_y = np.uint8(np.absolute(sobel_y))
    sobel_edges = cv2.bitwise_or(sobel_x, sobel_y)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    sobel_edges = cv2.morphologyEx(sobel_edges, cv2.MORPH_CLOSE, kernel_close)
    return sobel_edges


def color_sobel_edge_detection(rgb_img):
    """彩色Sobel：适合复杂背景下的彩色车辆"""
    r, g, b = cv2.split(rgb_img)

    def sobel_single_channel(channel):
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=5)
        return np.uint8(np.absolute(cv2.bitwise_or(sobel_x, sobel_y)))

    r_edges = sobel_single_channel(r)
    g_edges = sobel_single_channel(g)
    b_edges = sobel_single_channel(b)
    color_sobel_edges = cv2.bitwise_or(cv2.bitwise_or(r_edges, g_edges), b_edges)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    color_sobel_edges = cv2.morphologyEx(color_sobel_edges, cv2.MORPH_CLOSE, kernel_close)
    return color_sobel_edges


# 算法映射（推荐优先用彩色Canny/Sobel）
EDGE_ALGORITHMS = {
    "Canny边缘检测": canny_edge_detection,
    "彩色Canny边缘检测": color_canny_edge_detection,
    "Sobel边缘检测": sobel_edge_detection,
    "彩色Sobel边缘检测": color_sobel_edge_detection
}


# ------------------------------------------------------------------------------
# 2. 车辆特征提取+分类逻辑（核心：新增矩形坐标返回+可调节矩形度阈值）
# ------------------------------------------------------------------------------
def extract_vehicle_features(rgb_img, edge_img):
    """提取车辆关键特征：轮廓面积、长宽比、矩形度、主色调 + 外接矩形坐标（用于标记）"""
    # 1. 提取轮廓（只保留外部轮廓，过滤内部小轮廓）
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0.0, 0.0, (0, 0, 0), (0, 0, 0, 0)  # 最后一个返回值：(x,y,w,h) 矩形坐标

    # 2. 筛选最大轮廓（车辆主体）
    max_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_contour)

    # 3. 计算形状特征（外接矩形+矩形度）
    x, y, w, h = cv2.boundingRect(max_contour)  # 外接矩形（x,y：左上角坐标；w,h：宽高）
    aspect_ratio = w / h if h != 0 else 0.0  # 长宽比（宽/高）
    rect_area = w * h
    rectangularity = area / rect_area if rect_area != 0 else 0.0  # 矩形度（越接近1越像矩形）

    # 4. 计算主色调（仅轮廓内像素，避免背景干扰）
    mask = np.zeros_like(edge_img)
    cv2.drawContours(mask, [max_contour], -1, 255, -1)  # 轮廓掩码
    # 只取轮廓内的RGB像素均值（适配OpenCV BGR格式）
    r_mean = np.mean(rgb_img[mask == 255, 2]) if np.sum(mask == 255) > 0 else 0
    g_mean = np.mean(rgb_img[mask == 255, 1]) if np.sum(mask == 255) > 0 else 0
    b_mean = np.mean(rgb_img[mask == 255, 0]) if np.sum(mask == 255) > 0 else 0
    main_color = (r_mean, g_mean, b_mean)

    return area, aspect_ratio, rectangularity, main_color, (x, y, w, h)  # 新增矩形坐标返回


def classify_vehicle(area, aspect_ratio, rectangularity, main_color, min_rectangularity):
    """车辆分类规则：多特征组合判断（使用可调节矩形度阈值）"""
    r, g, b = main_color
    total_brightness = (r + g + b) / 3  # 亮度过滤（避免过暗图片）

    # 过滤条件（必须满足以下所有基础条件）
    if area < 5000:  # 最小轮廓面积（车辆尺寸远大于小物品）
        return "未识别（轮廓过小，非车辆）"
    if total_brightness < 40:  # 亮度阈值（避免暗图噪声）
        return "未识别（图片过暗，无法判断）"
    if rectangularity < min_rectangularity:  # 矩形度（使用可调节阈值）
        return f"未识别（轮廓不规则，矩形度{rectangularity:.2f} < {min_rectangularity:.2f}）"

    # 车辆长宽比判断（适配侧面/正面视角）
    if (aspect_ratio >= 2.0 and aspect_ratio <= 5.0) or (aspect_ratio >= 1.2 and aspect_ratio < 2.0):
        # 进一步过滤：车辆主色调通常单一（R/G/B中有一个通道显著高于其他两个）
        if (r > g + 30 and r > b + 30) or (g > r + 30 and g > b + 30) or (b > r + 30 and b > g + 30):
            return "识别为：车辆"
        else:
            # 允许部分颜色均匀的车辆（如白色、银色）
            if abs(r - g) < 20 and abs(g - b) < 20:  # 白/银/灰（三通道接近）
                return "识别为：车辆"
            else:
                return "未识别（颜色特征不符合车辆）"
    else:
        return f"未识别（长宽比{aspect_ratio:.1f}，不符合车辆范围）"


# ------------------------------------------------------------------------------
# 3. 中文UI界面（添加矩形度调节滑块+车辆矩形框标记）
# ------------------------------------------------------------------------------
class VehicleClassificationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("车辆识别系统（边缘检测+形状特征）")
        self.root.geometry("1100x1000")  # 增加高度容纳矩形度调节滑块
        self.root.resizable(False, False)

        # 存储变量
        self.image_path = None
        self.rgb_img = None
        self.gray_img = None
        self.edge_img = None
        self.classification_result = None
        self.selected_algorithm = tk.StringVar(value="彩色Canny边缘检测")  # 推荐算法
        self.vehicle_rect = (0, 0, 0, 0)  # 车辆外接矩形坐标(x,y,w,h)
        # 矩形度可调节参数（默认0.2，范围0.1-0.9）
        self.min_rectangularity = tk.DoubleVar(value=0.2)

        # ---------------------- 顶部控制区 ----------------------
        control_frame = ttk.Frame(root)
        control_frame.grid(row=0, column=0, columnspan=2, padx=20, pady=20, sticky="w")

        # 算法选择+使用提示
        ttk.Label(
            control_frame,
            text="选择边缘检测算法（推荐彩色Canny/Sobel）：",
            font=("微软雅黑", 11)
        ).grid(row=0, column=0, padx=10)

        algorithm_combobox = ttk.Combobox(
            control_frame, textvariable=self.selected_algorithm,
            values=list(EDGE_ALGORITHMS.keys()), font=("微软雅黑", 10), width=20
        )
        algorithm_combobox.grid(row=0, column=1, padx=10)
        algorithm_combobox.current(1)  # 默认选中彩色Canny

        # 功能按钮
        self.select_btn = ttk.Button(control_frame, text="选择车辆图片", command=self.select_image)
        self.select_btn.grid(row=0, column=2, padx=10)

        self.detect_btn = ttk.Button(control_frame, text="开始识别", command=self.start_detection, state="disabled")
        self.detect_btn.grid(row=0, column=3, padx=10)

        # 图片路径显示
        self.path_label = ttk.Label(control_frame, text="未选择图片", font=("微软雅黑", 10), width=40)
        self.path_label.grid(row=0, column=4, padx=10)

        # ---------------------- 图片展示区 ----------------------
        # 原始图片（带车辆标记）
        ttk.Label(root, text="原始车辆图片（红色矩形标记识别结果）", font=("微软雅黑", 12)).grid(row=1, column=0, padx=20,
                                                                                               pady=10)
        self.raw_canvas = tk.Canvas(root, width=400, height=350, borderwidth=1, relief="solid")
        self.raw_canvas.grid(row=2, column=0, padx=20, pady=10)

        # 边缘检测结果
        ttk.Label(root, text="边缘检测结果（车身轮廓需完整）", font=("微软雅黑", 12)).grid(row=1, column=1, padx=20,
                                                                                         pady=10)
        self.edge_canvas = tk.Canvas(root, width=400, height=350, borderwidth=1, relief="solid")
        self.edge_canvas.grid(row=2, column=1, padx=20, pady=10)

        # ---------------------- 矩形度调节区（新增）----------------------
        rect_frame = ttk.Frame(root)
        rect_frame.grid(row=3, column=0, columnspan=2, padx=20, pady=10, sticky="w")
        ttk.Label(
            rect_frame,
            text="矩形度阈值调节（默认0.2，值越小越宽松）：",
            font=("微软雅黑", 11)
        ).grid(row=0, column=0, padx=10)
        # 矩形度滑块（0.1-0.9范围）
        rect_slider = ttk.Scale(
            rect_frame, from_=0.1, to=0.9, variable=self.min_rectangularity,
            orient=tk.HORIZONTAL, command=self.update_rect_display,
            length=300
        )
        rect_slider.grid(row=0, column=1, padx=10)
        # 矩形度数值显示
        self.rect_value_label = ttk.Label(
            rect_frame, text=f"{self.min_rectangularity.get():.2f}",
            font=("微软雅黑", 11), width=10
        )
        self.rect_value_label.grid(row=0, column=2, padx=10)

        # ---------------------- 识别结果区 ----------------------
        result_frame = ttk.Frame(root)
        result_frame.grid(row=4, column=0, columnspan=2, padx=20, pady=20, sticky="w")

        ttk.Label(result_frame, text="识别结果：", font=("微软雅黑", 14, "bold")).grid(row=0, column=0, padx=10)
        self.result_label = ttk.Label(
            result_frame, text="--", font=("微软雅黑", 16, "bold"),
            foreground="darkgreen" if "识别为" in str("--") else "red"
        )
        self.result_label.grid(row=0, column=1, padx=20)

        # 关键特征值（帮助调试）
        ttk.Label(
            result_frame,
            text="车辆特征（参考范围）：",
            font=("微软雅黑", 12)
        ).grid(row=1, column=0, padx=10, pady=15)

        self.feat_label = ttk.Label(
            result_frame,
            text="轮廓面积：--（>5000） | 长宽比：--（1.2-5.0） | 矩形度：--（≥0.2）",
            font=("微软雅黑", 11)
        )
        self.feat_label.grid(row=1, column=1, padx=20, pady=15)

        # 使用提示
        ttk.Label(
            result_frame,
            text="使用提示：1. 选择侧面/正面视角、背景简单的车辆图片；2. 光线充足避免暗图；3. 车身无严重遮挡；4. 矩形度阈值越小，越容易识别不规则轮廓",
            font=("微软雅黑", 10), foreground="gray"
        ).grid(row=2, column=0, columnspan=2, padx=10, pady=5)

    # ---------------------- 新增：矩形度数值更新函数 ----------------------
    def update_rect_display(self, value):
        """实时更新矩形度阈值显示"""
        self.rect_value_label.config(text=f"{float(value):.2f}")

    # ---------------------- UI回调函数（修改：添加矩形框标记）----------------------
    def select_image(self):
        """选择车辆图片并预览"""
        file_types = [("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        self.image_path = filedialog.askopenfilename(title="选择车辆图片（推荐侧面视角）", filetypes=file_types)
        if not self.image_path:
            return

        try:
            # 读取图片（RGB+灰度图）
            self.rgb_img = cv2.imread(self.image_path)
            if self.rgb_img is None:
                raise ValueError("图片损坏或格式不支持")
            self.gray_img = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2GRAY)

            # 预览原始图片（无矩形框）
            pil_img = Image.fromarray(cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2RGB))
            pil_img = pil_img.resize((400, 350), Image.Resampling.LANCZOS)
            self.raw_img_tk = ImageTk.PhotoImage(pil_img)
            self.raw_canvas.delete("all")
            self.raw_canvas.create_image(0, 0, anchor="nw", image=self.raw_img_tk)

            # 更新UI状态
            display_path = self.image_path[:50] + "..." if len(self.image_path) > 50 else self.image_path
            self.path_label.config(text=display_path)
            self.detect_btn.config(state="normal")
            self.result_label.config(text="--", foreground="red")
            self.feat_label.config(text="轮廓面积：--（>5000） | 长宽比：--（1.2-5.0） | 矩形度：--（≥0.2）")
            self.vehicle_rect = (0, 0, 0, 0)  # 重置矩形框

        except Exception as e:
            messagebox.showerror("读取失败", f"无法读取图片：{str(e)}\n建议选择清晰、背景简单的车辆图片")

    def start_detection(self):
        """执行边缘检测+车辆识别（添加矩形框标记）"""
        if not self.image_path:
            messagebox.showwarning("警告", "请先选择图片！")
            return

        try:
            # 1. 执行选中的边缘检测算法
            algorithm_name = self.selected_algorithm.get()
            algorithm_func = EDGE_ALGORITHMS[algorithm_name]
            if "彩色" in algorithm_name:
                self.edge_img = algorithm_func(self.rgb_img)
            else:
                self.edge_img = algorithm_func(self.gray_img)

            # 2. 预览边缘检测结果
            edge_pil = Image.fromarray(self.edge_img)
            edge_pil = edge_pil.resize((400, 350), Image.Resampling.LANCZOS)
            self.edge_img_tk = ImageTk.PhotoImage(edge_pil)
            self.edge_canvas.delete("all")
            self.edge_canvas.create_image(0, 0, anchor="nw", image=self.edge_img_tk)

            # 3. 提取特征（包含矩形坐标）+ 分类识别（传入可调节矩形度阈值）
            area, aspect_ratio, rectangularity, main_color, self.vehicle_rect = extract_vehicle_features(self.rgb_img,
                                                                                                         self.edge_img)
            self.classification_result = classify_vehicle(
                area, aspect_ratio, rectangularity, main_color,
                self.min_rectangularity.get()  # 传入可调节矩形度阈值
            )

            # 4. 更新结果显示（绿色=识别成功，红色=未识别）
            if "识别为：车辆" in self.classification_result:
                self.result_label.config(text=self.classification_result, foreground="darkgreen")
                # 绘制红色矩形框标记车辆
                self.draw_vehicle_rect()
            else:
                self.result_label.config(text=self.classification_result, foreground="red")
                # 未识别时显示原图（无矩形框）
                pil_img = Image.fromarray(cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2RGB))
                pil_img = pil_img.resize((400, 350), Image.Resampling.LANCZOS)
                self.raw_img_tk = ImageTk.PhotoImage(pil_img)
                self.raw_canvas.delete("all")
                self.raw_canvas.create_image(0, 0, anchor="nw", image=self.raw_img_tk)

            # 5. 更新特征值显示（保留1位小数）
            self.feat_label.config(
                text=f"轮廓面积：{int(area)}（>5000） | 长宽比：{aspect_ratio:.1f}（1.2-5.0） | 矩形度：{rectangularity:.2f}（≥{self.min_rectangularity.get():.2f}）"
            )

        except Exception as e:
            messagebox.showerror("识别失败", f"错误原因：{str(e)}")

    # ---------------------- 新增：绘制车辆矩形框函数 ----------------------
    def draw_vehicle_rect(self):
        """在原始图片上绘制红色矩形框标记车辆"""
        x, y, w, h = self.vehicle_rect
        # 复制原图避免修改原始数据
        marked_img = self.rgb_img.copy()
        # 绘制红色矩形框（线宽3，醒目且不遮挡细节）
        cv2.rectangle(marked_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        # 添加"车辆"文字标注
        cv2.putText(marked_img, "车辆", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 缩放后显示到画布
        pil_img = Image.fromarray(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB))
        pil_img = pil_img.resize((400, 350), Image.Resampling.LANCZOS)
        self.raw_img_tk = ImageTk.PhotoImage(pil_img)
        self.raw_canvas.delete("all")
        self.raw_canvas.create_image(0, 0, anchor="nw", image=self.raw_img_tk)


# ------------------------------------------------------------------------------
# 4. 运行程序
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # 检查依赖库
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageTk
    except ImportError as e:
        print(f"缺少依赖库：{str(e)}")
        print("请运行以下命令安装：")
        print("pip install opencv-python numpy pillow")
        exit(1)

    # 启动UI
    root = tk.Tk()
    app = VehicleClassificationUI(root)
    root.mainloop()