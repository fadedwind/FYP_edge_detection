# 项目总结 / Project Summary

## 前端架构 / Frontend Architecture

### 技术栈 / Technology Stack
- **框架**: Vue 3 (Composition API)
- **构建工具**: Vite
- **路由**: Vue Router
- **HTTP客户端**: Axios
- **样式**: CSS3 (响应式设计)

### 核心组件 / Core Components

#### 1. 应用主结构 / Application Structure
- **App.vue**: 主应用容器，管理侧边栏状态、主题设置、语言切换
- **Sidebar.vue**: 可折叠侧边栏导航组件，支持图标和文字显示
- **ThemeSettingsPanel.vue**: 主题设置面板，支持亮色/暗色主题切换

#### 2. 页面视图 / Page Views

**经典算法边缘检测页面 (EdgeDetection.vue)**
- 支持图片和视频两种输入模式
- 实时边缘检测处理
- 高级参数调节面板（滑块控制）
- 批量处理功能：支持多图片上传和处理
- BSDS500快速测试功能
- 结果保存功能（图片/视频帧）
- PR曲线可视化展示
- 单图指标显示

**深度学习边缘检测页面 (DeepLearningEdgeDetection.vue)**
- 与经典算法页面功能一致
- 支持HED和PiDiNet两种深度学习算法
- 批量处理和BSDS500快速测试
- 评估参数配置面板（边界匹配容差、边界细化等）

**车辆识别页面 (VehicleDetection.vue)**
- 车辆识别功能
- 矩形度阈值调节

**文档页面 (Documentation.vue)**
- 算法原理文档链接展示
- 直接跳转到外部文档页面

**开发日志页面 (Changelog.vue)**
- Git提交历史可视化
- 从后端API获取Git日志
- 提交信息格式化显示

#### 3. 国际化支持 / Internationalization
- **i18n.js**: 完整的中英文双语支持
- 支持动态语言切换（中文/English）
- 所有UI文本均已国际化
- 语言偏好本地存储

#### 4. 路由配置 / Routing
- `/` - 经典算法边缘检测（首页）
- `/deep-learning` - 深度学习边缘检测
- `/vehicle` - 车辆识别
- `/docs` - 算法原理文档
- `/changelog` - 开发日志

#### 5. UI特性 / UI Features
- **响应式设计**: 支持桌面和移动端
- **暗色主题**: 默认暗色主题，支持切换
- **侧边栏折叠**: 可折叠侧边栏节省空间
- **实时处理**: 图片和视频实时边缘检测
- **结果保存**: 支持保存处理后的图片和视频帧
- **批量处理面板**: 可展开/收起的批量处理界面
- **评估参数配置**: 可配置的评估参数面板

---

## 后端架构 / Backend Architecture

### 技术栈 / Technology Stack
- **框架**: Flask (Python)
- **跨域支持**: Flask-CORS
- **图像处理**: OpenCV (cv2), NumPy, PIL
- **数据可视化**: Matplotlib
- **数据格式**: JSON, Base64, CSV

### API端点 / API Endpoints

#### 1. 健康检查 / Health Check
- `GET /api/health`: 后端服务健康状态检查

#### 2. 单图边缘检测 / Single Image Edge Detection
- `POST /api/edge-detect`: 
  - 接收Base64编码的图片
  - 支持多种算法（经典算法和深度学习）
  - 返回边缘检测结果和基础指标
  - 支持图片和视频帧处理

#### 3. 批量处理 / Batch Processing
- `POST /api/batch-process`:
  - 接收多张图片（Base64数组）
  - 批量处理并计算ODS/OIS指标
  - 生成PR曲线图
  - 保存处理结果到output目录
  - 生成CSV格式的批量指标文件
  - 支持评估参数配置（容差、细化等）

#### 4. BSDS500快速测试 / BSDS500 Quick Test
- `POST /api/bsds500-quick-test`:
  - 自动从BSDS500数据集读取图片
  - 使用ground truth进行自动评估
  - 支持验证集/测试集/训练集选择
  - 自动生成评估报告和PR曲线

#### 5. Git日志 / Git Log
- `GET /api/git-log`:
  - 获取Git提交历史
  - 解析提交信息（作者、日期、消息）
  - 支持分页和限制数量
  - 用于前端开发日志页面展示

#### 6. 车辆识别 / Vehicle Detection
- `POST /api/detect`:
  - 车辆识别功能
  - 基于边缘检测和形状特征

### 核心功能模块 / Core Modules

#### 1. 图像处理工具 / Image Processing Utilities
- `base64_to_image()`: Base64转OpenCV图像
- `image_to_base64()`: OpenCV图像转Base64
- `get_reference_edge()`: 生成参考边缘（用于评估）

#### 2. 批量处理工具 / Batch Processing Utilities
- `process_single_image_for_batch()`: 单图批量处理
- `compute_metrics_from_thresholds()`: 从阈值列表计算指标
- `compute_ois_from_per_image()`: 计算OIS指标
- `save_batch_metrics_csv()`: 保存批量指标到CSV
- `generate_pr_curve()`: 生成PR曲线图

#### 3. 模型管理 / Model Management
- `get_hed_net()`: HED模型缓存管理（避免重复加载）
- 模型文件路径自动检测
- 模型加载错误处理和回退机制

#### 4. 输出管理 / Output Management
- `generate_output_dir_name()`: 根据算法和参数生成输出目录名
- 自动创建输出目录
- 结果文件命名和组织

#### 5. BSDS500数据集支持 / BSDS500 Dataset Support
- `load_bsds_ground_truth()`: 加载BSDS500 ground truth标注
- 支持.mat格式的标注文件
- 多标注融合处理

### 配置和常量 / Configuration & Constants
- **输出目录**: `output/` (自动创建)
- **固定尺寸**: 640x480 (用于显示)
- **CORS**: 启用跨域请求支持
- **模型缓存**: 全局模型缓存机制

### 错误处理 / Error Handling
- 统一的异常捕获和错误响应
- 详细的错误信息返回
- 模型加载失败时的回退机制

---

## 前后端通信 / Frontend-Backend Communication

### 数据格式 / Data Format
- **请求**: JSON格式，图片使用Base64编码
- **响应**: JSON格式，包含success标志、数据、错误信息
- **文件传输**: Base64编码的图片数据

### 工作流程 / Workflow
1. 前端用户上传图片/选择视频
2. 前端将图片转换为Base64
3. 通过Axios发送POST请求到后端API
4. 后端处理图像并返回结果（Base64编码）
5. 前端显示处理结果
6. 支持结果保存到本地

### 状态管理 / State Management
- 前端使用Vue的响应式数据管理状态
- 处理状态（processing）控制UI禁用
- 结果数据缓存在组件中

---

## 项目特色 / Project Features

### 用户体验 / User Experience
- **实时处理**: 图片和视频实时边缘检测
- **批量处理**: 支持多图片批量处理
- **结果可视化**: PR曲线、指标展示
- **参数可调**: 丰富的参数调节选项
- **多语言**: 完整的中英文支持
- **主题切换**: 亮色/暗色主题
- **响应式**: 适配不同屏幕尺寸

### 开发体验 / Developer Experience
- **模块化设计**: 清晰的代码结构
- **组件复用**: 可复用的Vue组件
- **API标准化**: RESTful API设计
- **错误处理**: 完善的错误处理机制
- **Git集成**: 开发日志可视化

### 技术亮点 / Technical Highlights
- **模型缓存**: HED模型全局缓存，提升性能
- **批量处理优化**: 支持大规模图片处理
- **评估参数可配置**: 灵活的评估方法配置
- **Base64编码**: 无需文件上传，直接数据传输
- **实时视频处理**: 视频帧实时边缘检测

---

## 文件结构 / File Structure

### 前端 / Frontend
```
vue-frontend/
├── src/
│   ├── App.vue                    # 主应用组件
│   ├── main.js                    # 入口文件
│   ├── router.js                  # 路由配置
│   ├── i18n.js                    # 国际化配置
│   ├── components/                # 组件
│   │   ├── Sidebar.vue           # 侧边栏
│   │   └── ThemeSettingsPanel.vue # 主题设置
│   ├── views/                     # 页面视图
│   │   ├── EdgeDetection.vue     # 经典算法边缘检测
│   │   ├── DeepLearningEdgeDetection.vue # 深度学习边缘检测
│   │   ├── VehicleDetection.vue   # 车辆识别
│   │   ├── Documentation.vue      # 文档
│   │   └── Changelog.vue          # 开发日志
│   ├── styles/                    # 样式文件
│   │   └── common.css             # 公共样式
│   └── utils/                      # 工具函数
│       └── i18n.js                # 国际化工具
├── index.html                      # HTML入口
├── vite.config.js                  # Vite配置
└── package.json                    # 依赖配置
```

### 后端 / Backend
```
.
├── edge_detection_api.py          # Flask API主文件
├── boundary_evaluation.py          # 边界评估模块
├── pidinet.py                      # PiDiNet模型实现
├── hed.py                          # HED模型实现
├── start_backend.bat               # 后端启动脚本
├── requirements.txt                # Python依赖
└── output/                         # 输出目录（自动生成）
```

---

## 总结 / Summary

本项目是一个功能完整的边缘检测和车辆识别Web应用，采用前后端分离架构。前端使用Vue 3构建现代化的用户界面，支持多种输入模式、实时处理、批量操作和结果可视化。后端使用Flask提供RESTful API服务，支持多种算法、批量处理、数据集评估和Git日志查询。整个系统具有良好的用户体验、完善的错误处理和灵活的配置选项。

This project is a comprehensive edge detection and vehicle recognition web application using a frontend-backend separation architecture. The frontend is built with Vue 3, providing a modern user interface supporting multiple input modes, real-time processing, batch operations, and result visualization. The backend uses Flask to provide RESTful API services, supporting multiple algorithms, batch processing, dataset evaluation, and Git log queries. The entire system features excellent user experience, comprehensive error handling, and flexible configuration options.

