# 项目快速归档（summary_v0.1.3）

版本：v0.1.3

## 变更要点
- 新增 `image_recognition.py` 的检测+品种分类 prototype：包含 `load_detector`, `load_breed_classifier`, `detect_and_classify_cats`, `count_unique_breeds` 等函数，便于基于 YOLO (ultralytics) + timm/torch 的检测+分类流水线原型开发。
- 删除重复的 `image.recognition.py`（避免与下划线版冲突）。
- 已将改动提交并推送：commit message = `v0.1.3`，并创建 tag `v0.1.3`（已 push 到 `origin/gyc`）。

## 运行/依赖提示
- 若要运行检测+分类 prototype，需要安装第三方库：
  - ultralytics（检测器 YOLOv8）：`pip install ultralytics`
  - torch, timm（分类器）：按平台安装合适的 torch wheel，例如使用 CPU 版本或匹配 CUDA 的版本，随后 `pip install timm`。

## 使用建议
- 训练/微调：在 Oxford-IIIT Pet 或更贴合任务的数据上微调分类器权重，然后将 `weights_path` 指向微调后的 `.pth`。
- 若没有 GPU，可使用更小的检测器（`yolov8n`）和分类器（`efficientnet_b0`），并在 CPU 上测试性能。 

## 文件状态（本地工作区）
- 主脚本：`edge_detection.py`（GUI 主体）
- 图像识别模块：`image_recognition.py`（已包含 prototype）
- 文档：`docs/edge_detection_docs.html`, `docs/metrics.html`

---

请参考 `image_recognition.py` 中的示例函数来接入检测器和分类器，或让我把示例封装成单独的 `cat_breed_pipeline.py` 并做示例调用脚本。