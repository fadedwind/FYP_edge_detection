````markdown
# 项目快速归档（summary）

目标：为新的对话或协作者提供一份能让人快速上手本仓库的清晰摘要，包含关键文件、最近改动、运行和调试要点、以及下一步建议。

---

## 项目概览
- 名称：FYP_edge_detection
- 主要语言/库：Python，OpenCV（cv2），NumPy，Pillow，matplotlib，Tkinter（GUI）
- 主要用途：桌面端边缘检测工具（支持 Sobel/Prewitt/Canny 等），包含实时参数调节、视频/批处理、ODS/OIS 指标评估。

## 关键文件
- `edge_detection.py` — 主程序（Tkinter GUI + 边缘检测算法实现 + 批量评估/ODS-OIS 逻辑）。
- `image_recognition.py` — 新增的占位模块，供将来抽离图像识别/检测逻辑使用（已改为下划线命名，便于 import）。
- `requirements.txt` — 依赖清单（opencv-python, numpy, Pillow, matplotlib）。
- `docs/edge_detection_docs.html` — 算法与调参文档（已配置 MathJax 支持单美元 `$...$` 行内公式）。
- `docs/metrics.html` — 评估指标（ODS/OIS/PR/F1 等）独立页面（从主文档中拆出，便于维护）。
- `AIchat/` — 聊天记录与内部笔记目录（包含本次 summary 与前端思路草案等）。

## 最近重要改动（在 `gyc` 分支）
- 将主脚本 `pro.py` 重命名为 `edge_detection.py`（并保留原实现功能）。
- 新增：`image.recognition.py`（占位实现，提供 detect_edges、extract_bounding_boxes 等工具函数）。
- 修复：将 Tkinter 中错误使用的 `messagebox.warning(...)` 替换为 `messagebox.showwarning(...)`（避免 AttributeError）。
- 文档：扩展并改进 `docs/edge_detection_docs.html` 的算法说明，添加 MathJax 配置以支持 `$...$` 行内公式。
- 新增：`docs/metrics.html`（评估指标独立页）。

## 如何快速在本地运行和验证（开发者步骤）
1. 切到仓库目录（已在项目根）：

```powershell
# 激活虚拟环境（如果存在）
& ".\.venv\Scripts\Activate.ps1"

# 安装依赖
pip install -r requirements.txt

# 运行主程序
python edge_detection.py
```

2. 打开文档：在浏览器中打开 `docs/edge_detection_docs.html` 及 `docs/metrics.html`。如果页面公式没有渲染，确认能访问 MathJax CDN（https://cdn.jsdelivr.net）。如需离线查看，可请求把 MathJax 捆绑到仓库。

3. 快速 smoke-test：试用 GUI 打开一张图片，切换到 Canny，调整 Blur/High/Low Trackbars，观察输出并保存结果。

## 开发/维护提示与已知事项
- 在新环境中若出现 `import cv2 / numpy / PIL / matplotlib` 错误，先执行 `pip install -r requirements.txt`。
- 文档中使用 `$...$` 作为行内公式（已为 MathJax 配置），若有特殊字符仍不渲染，检查是否被 HTML 转义（例如 `\\` 被多次转义）。
- 文件 `image.recognition.py` 使用了点号（.）作为文件名的一部分，建议将来改为 `image_recognition.py` 以便 Python 可以直接 import。

## 下一步建议（短期优先级）
1. 在干净的虚拟环境中运行一次完整 smoke 测试（激活 venv → pip install → python edge_detection.py）。记录任何 RuntimeError 并修复。
2. （可选）为关键模块添加单元测试（处理函数如 `compute_edge_strength`, `calculate_metrics` 的输出可加小样例测试）。
3. 若计划发布或展示：考虑把 docs 页面放到 GitHub Pages（或简单静态服务器）以便导师/同学在线查看。

---

作者注：本摘要由自动化助手生成并存入仓库，目的是让从新建聊天开始的协作者在最短时间内了解项目状态与如何上手。

````
