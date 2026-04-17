import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

# -------------------------- 全局参数配置 --------------------------
frameWidth = 640
frameHeight = 480
file_path = ""
file_type = ""
process_result = {"img_original": None, "img_edge": None, "img_contour": None}

algo_combobox = None
file_label = None
original_label = None
edge_label = None
contour_label = None
root = None

debounce_id = None
debounce_delay = 300


def empty(a):
    pass


def getContours(img, imgContour):
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


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0]) if isinstance(imgArray[0], list) else rows
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1] if rowsAvailable else imgArray[0].shape[1]
    height = imgArray[0][0].shape[0] if rowsAvailable else imgArray[0].shape[0]

    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y],
                                                (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        hor = [np.zeros((height, width, 3), np.uint8)] * rows
        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x],
                                         (imgArray[0].shape[1], imgArray[0].shape[0]),
                                         None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        ver = np.hstack(imgArray)
    return ver


def read_params():
    blur_ksize = cv2.getTrackbarPos("Blur", "Parameters")
    blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    if blur_ksize < 1:
        blur_ksize = 1

    sobel_ksize = cv2.getTrackbarPos("Sobel_Ksize", "Parameters")
    sobel_ksize = sobel_ksize if (sobel_ksize % 2 == 1 and sobel_ksize >= 1) else 3

    canny_low = cv2.getTrackbarPos("Canny_Low", "Parameters")
    canny_high = cv2.getTrackbarPos("Canny_High", "Parameters")
    if canny_low > canny_high:
        canny_low, canny_high = canny_high, canny_low

    dilate_ksize = cv2.getTrackbarPos("Dilate", "Parameters")
    dilate_ksize = dilate_ksize if (dilate_ksize % 2 == 1 and dilate_ksize >= 1) else 5

    return blur_ksize, sobel_ksize, canny_low, canny_high, dilate_ksize


# ====================== 角度Canny · 最终完整修复 ======================
def calc_vector_length_gradient(p1, p2):
    return abs(int(p1[0]) - int(p2[0])) + abs(int(p1[1]) - int(p2[1])) + abs(int(p1[2]) - int(p2[2]))

def calc_vector_angle_gradient(p1, p2):
    r1, g1, b1 = int(p1[0]), int(p1[1]), int(p1[2])
    r2, g2, b2 = int(p2[0]), int(p2[1]), int(p2[2])

    dot = r1 * r2 + g1 * g2 + b1 * b2
    norm1 = np.sqrt(r1 ** 2 + g1 ** 2 + b1 ** 2)
    norm2 = np.sqrt(r2 ** 2 + g2 ** 2 + b2 ** 2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0

    cos_theta = dot / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -0.99999, 0.99999)
    rad = np.arccos(cos_theta)
    return rad * 100  # 放大到有效梯度范围

def non_max_suppression(grad_mag, grad_dir):
    h, w = grad_mag.shape
    result = np.zeros_like(grad_mag, dtype=np.float32)
    angle = grad_dir * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                n1, n2 = grad_mag[i, j + 1], grad_mag[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                n1, n2 = grad_mag[i - 1, j + 1], grad_mag[i + 1, j - 1]
            elif 67.5 <= angle[i, j] < 112.5:
                n1, n2 = grad_mag[i - 1, j], grad_mag[i + 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                n1, n2 = grad_mag[i - 1, j - 1], grad_mag[i + 1, j + 1]
            else:
                n1, n2 = 0, 0

            if grad_mag[i, j] >= n1 and grad_mag[i, j] >= n2:
                result[i, j] = grad_mag[i, j]
    return result

def length_canny(img, blur_ksize, low, high):
    img_blur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 1)
    h, w = img_blur.shape[:2]
    grad_x = np.zeros((h, w), np.float32)
    grad_y = np.zeros((h, w), np.float32)

    for i in range(h):
        for j in range(w - 1):
            grad_x[i, j] = calc_vector_length_gradient(img_blur[i, j], img_blur[i, j + 1])
    for i in range(h - 1):
        for j in range(w):
            grad_y[i, j] = calc_vector_length_gradient(img_blur[i, j], img_blur[i + 1, j])

    mag = np.hypot(grad_x, grad_y)
    dir = np.arctan2(grad_y, grad_x)
    nms = non_max_suppression(mag, dir)
    nms = cv2.normalize(nms, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    edge = cv2.threshold(nms, low, 255, cv2.THRESH_BINARY)[1]
    return edge

def angle_canny(img, blur_ksize, low, high):
    img_blur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 1)
    h, w = img_blur.shape[:2]
    grad_x = np.zeros((h, w), np.float32)
    grad_y = np.zeros((h, w), np.float32)

    for i in range(h):
        for j in range(w - 1):
            grad_x[i, j] = calc_vector_angle_gradient(img_blur[i, j], img_blur[i, j + 1])
    for i in range(h - 1):
        for j in range(w):
            grad_y[i, j] = calc_vector_angle_gradient(img_blur[i, j], img_blur[i + 1, j])

    mag = np.hypot(grad_x, grad_y)
    dir = np.arctan2(grad_y, grad_x)
    nms = non_max_suppression(mag, dir)
    nms = cv2.normalize(nms, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    edge = cv2.threshold(nms, low, 255, cv2.THRESH_BINARY)[1]
    return edge
# ========================================================================


def process_image_realtime(algorithm):
    global process_result
    if file_type != "image" or not file_path:
        return

    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("错误", "无法读取图片文件！")
        return

    img = cv2.resize(img, (frameWidth, frameHeight))
    blur_ksize, sobel_ksize, canny_low, canny_high, dilate_ksize = read_params()

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

    elif algorithm == "长度Canny":
        img_edge = length_canny(img, blur_ksize, canny_low, canny_high)

    elif algorithm == "角度Canny":
        img_edge = angle_canny(img, blur_ksize, canny_low, canny_high)

    kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
    img_edge = cv2.dilate(img_edge, kernel, iterations=1)
    img_contour = img.copy()
    getContours(img_edge, img_contour)

    process_result = {
        "img_original": img,
        "img_edge": img_edge,
        "img_contour": img_contour
    }
    update_result_display()


def process_video(algorithm):
    global process_result
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        messagebox.showerror("错误", "无法读取视频文件！")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = "video_output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"processed_{algorithm}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    blur_ksize, sobel_ksize, canny_low, canny_high, dilate_ksize = read_params()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (frameWidth, frameHeight))
        img_contour = frame_resized.copy()
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

        elif algorithm == "长度Canny":
            img_edge = length_canny(frame_resized, blur_ksize, canny_low, canny_high)

        elif algorithm == "角度Canny":
            img_edge = angle_canny(frame_resized, blur_ksize, canny_low, canny_high)

        kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
        img_edge = cv2.dilate(img_edge, kernel, iterations=1)
        getContours(img_edge, img_contour)

        img_stack = stackImages(0.8, ([frame_resized, img_edge], [np.zeros_like(img_edge), img_contour]))
        cv2.imshow("Video Processing (Press 'q' to stop)", img_stack)
        final_frame = cv2.resize(img_contour, (width, height))
        out.write(final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("成功", f"视频处理完成！\n保存路径：{output_path}")


def on_param_change(val):
    global debounce_id
    algorithm = algo_combobox.get()
    if not algorithm or file_type != "image":
        return
    if debounce_id is not None:
        root.after_cancel(debounce_id)
    debounce_id = root.after(debounce_delay, process_image_realtime, algorithm)


def create_parameter_window():
    cv2.namedWindow("Parameters", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Parameters", 640, 300)
    cv2.createTrackbar("Sobel_Ksize", "Parameters", 3, 7, on_param_change)
    cv2.createTrackbar("Blur", "Parameters", 7, 31, on_param_change)
    cv2.createTrackbar("Dilate", "Parameters", 5, 15, on_param_change)
    cv2.createTrackbar("Canny_Low", "Parameters", 30, 200, on_param_change)
    cv2.createTrackbar("Canny_High", "Parameters", 100, 300, on_param_change)
    cv2.createTrackbar("Area", "Parameters", 500, 30000, on_param_change)


def select_file():
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
            algorithm = algo_combobox.get()
            if algorithm:
                process_image_realtime(algorithm)
        elif ext in [".mp4", ".avi", ".mov"]:
            file_type = "video"
            original_label.config(image='')
            edge_label.config(image='')
            contour_label.config(image='')
        else:
            file_type = ""
            messagebox.showwarning("警告", "不支持的文件格式！")
        file_label.config(text=f"已选择：{os.path.basename(file_path)}")
    else:
        file_type = ""
        file_label.config(text="未选择文件")


def start_process():
    if not file_path:
        messagebox.showwarning("警告", "请先选择文件！")
        return
    algorithm = algo_combobox.get()
    if not algorithm:
        messagebox.showwarning("警告", "请选择算法！")
        return

    if not cv2.getWindowProperty("Parameters", cv2.WND_PROP_VISIBLE):
        create_parameter_window()

    if file_type == "image":
        process_image_realtime(algorithm)
    elif file_type == "video":
        process_video(algorithm)


def update_result_display():
    if process_result["img_original"] is None:
        return

    def cv2_to_tk(img):
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((280, 220), Image.LANCZOS)
        return ImageTk.PhotoImage(image=img_pil)

    img_original_tk = cv2_to_tk(process_result["img_original"])
    img_edge_tk = cv2_to_tk(process_result["img_edge"])
    img_contour_tk = cv2_to_tk(process_result["img_contour"])

    original_label.config(image=img_original_tk)
    original_label.image = img_original_tk
    edge_label.config(image=img_edge_tk)
    edge_label.image = img_edge_tk
    contour_label.config(image=img_contour_tk)
    contour_label.image = img_contour_tk


def save_result():
    if process_result["img_edge"] is None:
        messagebox.showwarning("警告", "暂无处理结果可保存！")
        return

    save_path = filedialog.asksaveasfilename(
        title="保存处理结果",
        defaultextension=".png",
        filetypes=[("PNG图片", "*.png"), ("JPG图片", "*.jpg")]
    )
    if save_path:
        cv2.imwrite(save_path, process_result["img_edge"])
        contour_save_path = os.path.splitext(save_path)[0] + "_contour.png"
        cv2.imwrite(contour_save_path, process_result["img_contour"])
        messagebox.showinfo("成功", f"结果已保存！\n处理图：{save_path}\n轮廓图：{contour_save_path}")


def init_gui():
    global root, algo_combobox, file_label, original_label, edge_label, contour_label
    root = tk.Tk()
    root.title("边缘检测工具（Sobel/Canny/向量Canny）")
    root.geometry("900x550")

    control_frame = ttk.Frame(root, padding="10")
    control_frame.pack(fill=tk.X)

    algo_label = ttk.Label(control_frame, text="选择算法：")
    algo_label.grid(row=0, column=0, padx=5, pady=5)
    algo_options = ["Sobel", "彩色Sobel", "Canny", "长度Canny", "角度Canny"]
    algo_combobox = ttk.Combobox(control_frame, values=algo_options, state="readonly")
    algo_combobox.grid(row=0, column=1, padx=5, pady=5)
    algo_combobox.bind("<<ComboboxSelected>>", lambda e: process_image_realtime(algo_combobox.get()))

    select_btn = ttk.Button(control_frame, text="选择文件", command=select_file)
    select_btn.grid(row=0, column=2, padx=5, pady=5)
    process_btn = ttk.Button(control_frame, text="开始处理", command=start_process)
    process_btn.grid(row=0, column=3, padx=5, pady=5)
    save_btn = ttk.Button(control_frame, text="保存结果", command=save_result)
    save_btn.grid(row=0, column=4, padx=5, pady=5)

    file_label = ttk.Label(control_frame, text="未选择文件", wraplength=300)
    file_label.grid(row=0, column=5, padx=5, pady=5)

    result_frame = ttk.Frame(root, padding="10")
    result_frame.pack(fill=tk.BOTH, expand=True)

    original_label = ttk.Label(result_frame, text="原图")
    original_label.grid(row=0, column=0, padx=10, pady=10)
    edge_label = ttk.Label(result_frame, text="处理结果")
    edge_label.grid(row=0, column=1, padx=10, pady=10)
    contour_label = ttk.Label(result_frame, text="轮廓结果")
    contour_label.grid(row=0, column=2, padx=10, pady=10)

    create_parameter_window()
    root.mainloop()


if __name__ == "__main__":
    print("请确保已安装依赖：pip install opencv-python numpy pillow")
    init_gui()