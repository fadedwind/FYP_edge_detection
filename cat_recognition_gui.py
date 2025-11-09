import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import io
import contextlib

# 尝试导入检测器加载函数（仅检测，不做分类）
try:
    from image_recognition import load_detector
except Exception:
    load_detector = None


class CatRecognitionGUI:
    def __init__(self, root):
        self.root = root
        root.title('猫识别 Demo')
        root.geometry('900x600')

        # 模型句柄
        self.detector = None
    # 分类相关已移除

        # 选中图片路径
        self.img_path = None
        self.img_bgr = None
        self.display_img_tk = None

        # UI 布局
        ctrl = ttk.Frame(root, padding=8)
        ctrl.pack(fill=tk.X)

        ttk.Button(ctrl, text='选择图片', command=self.select_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text='加载检测器 (YOLOv8)', command=self.load_detector_action).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text='运行识别', command=self.run_recognition).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text='退出', command=root.quit).pack(side=tk.RIGHT, padx=4)

        # 图片显示与结果区
        body = ttk.Frame(root, padding=8)
        body.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, bg='black')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # 右侧结果面板已移除（简化为仅在画面上绘制 bbox 并在命令行输出摘要/弹窗提示）

    def select_image(self):
        path = filedialog.askopenfilename(title='选择图片', filetypes=[('Image', '*.jpg;*.png;*.bmp')])
        if not path:
            return
        self.img_path = path
        self.img_bgr = cv2.imread(path)
        if self.img_bgr is None:
            messagebox.showerror('错误', '无法读取图片文件')
            return
        self.show_image(self.img_bgr)

    def show_image(self, img_bgr, boxes=None, labels=None):
        # 在副本上绘制框
        img = img_bgr.copy()
        if boxes:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = labels[i] if labels and i < len(labels) else ''
                if text:
                    cv2.putText(img, text, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # 转换成 Tk 图片并展示
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        # 缩放到 canvas 大小
        canvas_w = max(200, int(self.canvas.winfo_width()))
        canvas_h = max(200, int(self.canvas.winfo_height()))
        scale = min(canvas_w / w, canvas_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        img_pil = Image.fromarray(img_rgb).resize((new_w, new_h), Image.LANCZOS)
        self.display_img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_img_tk)

    def load_detector_action(self):
        if load_detector is None:
            messagebox.showwarning('缺少依赖', '请先安装 ultralytics（pip install ultralytics）以使用检测功能')
            return
        path = filedialog.askopenfilename(title='选择检测器权重（yolov8n.pt，或取消使用本地 models/yolov8n.pt）', filetypes=[('pt', '*.pt'), ('all','*.*')])
        # 如果用户没有选择文件且本地 models/yolov8n.pt 存在，则使用该本地权重
        if not path:
            local = os.path.join(os.getcwd(), 'models', 'yolov8n.pt')
            if os.path.exists(local):
                path = local
            else:
                return
        try:
            self.detector = load_detector(path)
            messagebox.showinfo('成功', '检测器加载完成')
        except Exception as e:
            messagebox.showerror('加载失败', str(e))



    def run_recognition(self):
        if self.img_bgr is None:
            messagebox.showwarning('提示', '请先选择图片')
            return
        # 仅使用检测器（不进行分类）

        # 备用：仅检测（如果 detector 可用）
        if self.detector is not None:
            try:
                # capture any console output from the detector so UI matches cmd
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    results = self.detector(self.img_bgr)
                console_output = buf.getvalue()
                detections = []
                if len(results) > 0:
                    r = results[0]
                    names = getattr(results, 'names', None) or getattr(r, 'names', None)

                    # Attempt vectorized extraction first
                    try:
                        if hasattr(r.boxes, 'xyxy') and hasattr(r.boxes, 'cls'):
                            xyxy_arr = r.boxes.xyxy.cpu().numpy()
                            conf_arr = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else np.zeros(len(xyxy_arr))
                            cls_arr = r.boxes.cls.cpu().numpy()
                            for xyxy, conf_val, cls_val in zip(xyxy_arr, conf_arr, cls_arr):
                                try:
                                    cls_i = int(cls_val)
                                    label = None
                                    if names is not None and cls_i < len(names):
                                        label = str(names[cls_i]).lower()
                                    if (label and 'cat' in label) or cls_i == 16:
                                        detections.append({'bbox': tuple(xyxy.astype(int).tolist()), 'det_score': float(conf_val), 'breed': 'cat'})
                                except Exception:
                                    continue
                        else:
                            # Fallback: iterate over box objects and extract attributes robustly
                            for b in r.boxes:
                                try:
                                    # xyxy
                                    xyxy = None
                                    if hasattr(b, 'xyxy'):
                                        val = b.xyxy
                                        if hasattr(val, 'cpu'):
                                            arr = val.cpu().numpy()
                                        else:
                                            arr = np.array(val)
                                        # handle both (N,4) and (4,) shapes
                                        if arr.ndim == 2:
                                            arr0 = arr[0]
                                        else:
                                            arr0 = arr
                                        xyxy = arr0.astype(int)

                                    # conf
                                    conf_val = 0.0
                                    if hasattr(b, 'conf'):
                                        val = b.conf
                                        try:
                                            conf_val = float(val.cpu().numpy()[0]) if hasattr(val, 'cpu') else float(val)
                                        except Exception:
                                            conf_val = 0.0

                                    # cls
                                    cls_val = None
                                    if hasattr(b, 'cls'):
                                        val = b.cls
                                        try:
                                            cls_val = int(val.cpu().numpy()[0]) if hasattr(val, 'cpu') else int(val)
                                        except Exception:
                                            cls_val = None

                                    if cls_val is None or xyxy is None:
                                        continue
                                    label = None
                                    if names is not None and cls_val < len(names):
                                        label = str(names[cls_val]).lower()
                                    if (label and 'cat' in label) or cls_val == 16:
                                        detections.append({'bbox': tuple(xyxy.tolist()), 'det_score': conf_val, 'breed': 'cat'})
                                except Exception:
                                    continue
                    except Exception as e:
                        print('Warning: failed to parse detection results:', e)
                boxes = [d['bbox'] for d in detections]
                labels = [f"cat({d['det_score']:.2f})" for d in detections]
                self.show_image(self.img_bgr, boxes=boxes, labels=labels)
                if console_output and console_output.strip():
                    print(console_output)
                messagebox.showinfo('检测完成', f'检测到 {len(detections)} 只猫（详见命令行输出）')
                return
            except Exception as e:
                messagebox.showerror('运行失败', f'检测出错：{e}')
                return

        messagebox.showwarning('缺少模型', '当前未加载检测器。请先加载检测器或安装 ultralytics。')

    # right-side UI removed; results are printed to console and a brief popup shown


def main():
    root = tk.Tk()
    app = CatRecognitionGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
