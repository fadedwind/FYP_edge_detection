# 车辆识别系统 - Vue 前端

基于 Vue 3 + Vite 的车辆识别系统前端界面。

## 功能特性

- 🖼️ 图片上传和预览
- 🔍 4种边缘检测算法可选（Canny、彩色Canny、Sobel、彩色Sobel）
- 🎚️ 矩形度阈值实时调节
- 📊 识别结果和特征值展示
- 🎨 现代化 UI 设计

## 安装和运行

### 1. 安装依赖

```bash
cd vue-frontend
npm install
```

### 2. 启动开发服务器

```bash
npm run dev
```

前端将在 `http://localhost:3000` 启动。

### 3. 确保后端 API 运行

确保 Flask 后端 API (`vehicle_detection_api.py`) 在 `http://localhost:5000` 运行。

## 项目结构

```
vue-frontend/
├── index.html          # HTML 入口
├── vite.config.js      # Vite 配置
├── package.json        # 依赖配置
└── src/
    ├── main.js         # Vue 入口
    └── App.vue         # 主组件
```

## 构建生产版本

```bash
npm run build
```

构建产物在 `dist/` 目录。

## API 接口

前端通过以下接口与后端通信：

- `GET /api/health` - 健康检查
- `GET /api/algorithms` - 获取算法列表
- `POST /api/detect` - 车辆识别

详见 `vehicle_detection_api.py`。





