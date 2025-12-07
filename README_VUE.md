# 车辆识别系统 - Vue 前端改造

已将 `pro1.py` 的 Tkinter 桌面应用改造为 **Vue 前端 + Flask 后端** 的 Web 应用。

## 📁 项目结构

```
项目根目录/
├── vehicle_detection_api.py    # Flask 后端 API
├── vue-frontend/                # Vue 前端项目
│   ├── src/
│   │   ├── main.js
│   │   └── App.vue
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   └── README.md
└── README_VUE.md               # 本文件
```

## 🚀 快速开始

### 1. 安装后端依赖

```bash
pip install flask flask-cors opencv-python numpy pillow
```

### 2. 启动后端 API

```bash
python vehicle_detection_api.py
```

后端将在 `http://localhost:5000` 运行。

### 3. 安装前端依赖

```bash
cd vue-frontend

```

### 4. 启动前端开发服务器

```bash
npm run dev
```

前端将在 `http://localhost:3000` 运行。

## 🎯 功能对比

| 功能 | Tkinter 版本 (pro1.py) | Vue 版本 |
|------|----------------------|---------|
| 图片选择 | 文件对话框 | 拖拽/点击上传 |
| 算法选择 | 下拉框 | 下拉选择 |
| 矩形度调节 | 滑块 | 滑块（实时显示数值）|
| 结果显示 | 标签+特征文本 | 卡片式展示 |
| 图片展示 | Canvas | `<img>` 标签 |
| 跨平台 | 需要 GUI 库 | 浏览器即可 |

## 🔧 API 接口说明

### POST /api/detect

**请求体：**
```json
{
  "image": "data:image/png;base64,...",
  "algorithm": "彩色Canny边缘检测",
  "min_rectangularity": 0.2
}
```

**响应：**
```json
{
  "success": true,
  "result": "识别为：车辆",
  "is_vehicle": true,
  "features": {
    "area": 12345,
    "aspect_ratio": 2.5,
    "rectangularity": 0.65,
    "main_color": [120, 80, 60],
    "vehicle_rect": [100, 50, 200, 80]
  },
  "images": {
    "marked": "data:image/png;base64,...",
    "edge": "data:image/png;base64,..."
  }
}
```

## 📝 主要改动

1. **后端 API 化**：将 Tkinter UI 逻辑分离，保留核心算法为 API
2. **图片传输**：使用 base64 编码在前后端传递图片
3. **响应式设计**：Vue 组件化，支持移动端
4. **现代化 UI**：使用 CSS Grid/Flexbox，渐变背景，卡片式布局

## 🎨 UI 特性

- 渐变背景和圆角设计
- 响应式布局（支持移动端）
- 实时特征值显示
- 成功/失败状态的颜色区分
- 使用提示和帮助信息

## 🔄 后续优化建议

1. **图片压缩**：上传前压缩大图，减少传输时间
2. **进度条**：显示处理进度
3. **历史记录**：保存识别历史
4. **批量处理**：支持多张图片同时识别
5. **结果导出**：导出识别结果和特征数据

## 📦 部署

### 开发环境
- 前端：`npm run dev` (Vite dev server)
- 后端：`python vehicle_detection_api.py` (Flask debug mode)

### 生产环境
- 前端：`npm run build` → 部署 `dist/` 到 Nginx
- 后端：使用 Gunicorn 或 uWSGI 部署 Flask

## 🐛 故障排查

1. **CORS 错误**：确保后端已启用 `CORS(app)`
2. **图片加载失败**：检查 base64 编码格式
3. **API 连接失败**：确认后端在 5000 端口运行
4. **算法列表为空**：检查 `/api/algorithms` 接口是否正常

## 📄 许可证

与原项目保持一致。




