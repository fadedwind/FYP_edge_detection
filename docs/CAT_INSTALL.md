# Cat detection & breed-classifier setup

This document explains how the project ships a local YOLOv8 tiny weight and how to install requirements to run the cat recognition GUI.

1) Local model
- A copy of `yolov8n.pt` has been downloaded to `models/yolov8n.pt` in this repository root for convenience.
- To avoid accidentally committing this binary into git the `models/` directory is added to `.gitignore`.

2) Install dependencies (PowerShell)
```powershell
& ".\.venv\Scripts\Activate.ps1"
pip install -r requirements.txt
# Then install torch according to your platform (CPU or CUDA). See https://pytorch.org/ for the correct command.
```

3) Run the GUI
- Start the GUI that lets you choose an image and run detection:
```powershell
& ".\.venv\Scripts\Activate.ps1"
python cat_recognition_gui.py
```
- In the GUI: click "加载检测器 (YOLOv8)" and either choose a local weight or cancel; if you cancel the GUI will use the bundled `models/yolov8n.pt` automatically.
- Optionally click "加载分类器 (可选)" and choose a `breed_weights.pth` trained for your breed classes and an optional labels `.txt` file (one label per line).

4) Notes
- `ultralytics` will auto-download weights if you pass a model name (e.g. `yolov8n`) to `YOLO(...)`. We supplied a local weight to allow offline testing.
- The classification prototype expects a PyTorch `.pth` state_dict compatible with the timm EfficientNet-B0 head (see `image_recognition.py`).
- If you want me to download different weights or place weights into a different path, tell me and I will update the GUI defaults.
