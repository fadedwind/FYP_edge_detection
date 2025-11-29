// 国际化配置
export const i18n = {
  zh: {
    // 侧边栏
    sidebar: {
      title: '导航',
      edgeDetection: '边缘检测',
      vehicleDetection: '汽车识别',
      documentation: '算法原理',
      appearanceSettings: '外观设置'
    },
    // 边缘检测页面
    edgeDetection: {
      title: '图像边缘检测主页',
      subtitle: '支持 Sobel / Canny / Prewitt / HED 等多种边缘检测算法',
      algorithm: '边缘检测算法：',
      moreParams: '更多参数调整',
      collapseParams: '收起更多参数',
      blur: '模糊核大小：',
      cannyThreshold: 'Canny 阈值：',
      selectImage: '📁 选择图片',
      startDetection: '✨ 开始边缘检测',
      processing: '处理中...',
      originalImage: '原始图片',
      edgeResult: '边缘检测结果',
      selectImagePlaceholder: '请选择一张图片',
      edgeResultPlaceholder: '边缘结果将显示在这里',
      metrics: '评估指标',
      edgePixels: '非零边缘像素数：',
      // 高级参数
      advancedParams: '高级参数（与桌面版 Trackbar 对应）：',
      sobelKsize: 'Sobel_Ksize（奇数）：',
      blurKernel: 'Blur（模糊核）：',
      dilateKernel: 'Dilate（膨胀核）：',
      cannyLow: 'Canny_Low：',
      cannyHigh: 'Canny_High：',
      areaMin: 'Area（轮廓最小面积，示意参数）：',
      // 批量处理
      batchProcessing: '📁 批量处理',
      expandBatch: '展开批量处理',
      collapseBatch: '收起批量处理',
      selectMultipleImages: '📁 选择多张图片（可多选）',
      selectedCount: '已选择 {count} 张图片',
      startBatch: '🚀 开始批量处理',
      batchProcessingStatus: '处理中...',
      batchResults: '批量处理结果',
      ods: 'ODS (Optimal Dataset Scale)',
      ois: 'OIS (Optimal Image Scale)',
      threshold: '阈值',
      prCurve: 'PR 曲线',
      processedResults: '处理结果（共 {count} 张）',
      // 指标标签（保持英文）
      precision: 'Precision',
      recall: 'Recall',
      f1Score: 'F1-Score'
    },
    // 车辆识别页面
    vehicleDetection: {
      title: '车辆识别系统',
      subtitle: '基于边缘检测 + 形状特征的车辆识别',
      algorithm: '边缘检测算法：',
      rectangularity: '矩形度阈值：',
      selectImage: '📁 选择车辆图片',
      startDetection: '🔍 开始识别',
      processing: '处理中...',
      originalImage: '原始图片（红色矩形标记识别结果）',
      edgeResult: '边缘检测结果',
      selectImagePlaceholder: '请选择图片并开始识别',
      edgeResultPlaceholder: '边缘检测结果将显示在这里',
      result: '识别结果',
      features: '车辆特征：',
      area: '轮廓面积',
      aspectRatio: '长宽比',
      rectangularity: '矩形度',
      mainColor: '主色调 RGB',
      requirement: '要求',
      tips: '💡 使用提示：',
      tip1: '选择侧面/正面视角、背景简单的车辆图片',
      tip2: '光线充足，避免暗图',
      tip3: '车身无严重遮挡',
      tip4: '矩形度阈值越小，越容易识别不规则轮廓'
    },
    // 文档页面
    documentation: {
      title: '文档目录',
      edgeDetectionDocs: '边缘检测算法详解',
      metricsDocs: '评估指标与 ODS/OIS',
      loading: '加载中...',
      loadError: '加载文档失败'
    },
    // 外观设置
    appearanceSettings: {
      title: '外观设置',
      backgroundColor: '背景颜色：',
      containerColor: '容器颜色：',
      textColor: '文字颜色：',
      resetDefault: '恢复默认（黑金）'
    },
    // 通用
    common: {
      close: '×',
      back: '返回',
      home: '主页'
    }
  },
  en: {
    // Sidebar
    sidebar: {
      title: 'Navigation',
      edgeDetection: 'Edge Detection',
      vehicleDetection: 'Vehicle Detection',
      documentation: 'Documentation',
      appearanceSettings: 'Appearance Settings'
    },
    // Edge Detection Page
    edgeDetection: {
      title: 'Image Edge Detection',
      subtitle: 'Support Sobel / Canny / Prewitt / HED and other edge detection algorithms',
      algorithm: 'Edge Detection Algorithm:',
      moreParams: 'More Parameters',
      collapseParams: 'Collapse Parameters',
      blur: 'Blur Kernel Size:',
      cannyThreshold: 'Canny Threshold:',
      selectImage: '📁 Select Image',
      startDetection: '✨ Start Edge Detection',
      processing: 'Processing...',
      originalImage: 'Original Image',
      edgeResult: 'Edge Detection Result',
      selectImagePlaceholder: 'Please select an image',
      edgeResultPlaceholder: 'Edge result will be displayed here',
      metrics: 'Evaluation Metrics',
      edgePixels: 'Non-zero Edge Pixels:',
      // Advanced Parameters
      advancedParams: 'Advanced Parameters (corresponding to desktop Trackbar):',
      sobelKsize: 'Sobel_Ksize (odd):',
      blurKernel: 'Blur (blur kernel):',
      dilateKernel: 'Dilate (dilation kernel):',
      cannyLow: 'Canny_Low:',
      cannyHigh: 'Canny_High:',
      areaMin: 'Area (minimum contour area, reference parameter):',
      // Batch Processing
      batchProcessing: '📁 Batch Processing',
      expandBatch: 'Expand Batch Processing',
      collapseBatch: 'Collapse Batch Processing',
      selectMultipleImages: '📁 Select Multiple Images (multiple selection)',
      selectedCount: '{count} images selected',
      startBatch: '🚀 Start Batch Processing',
      batchProcessingStatus: 'Processing...',
      batchResults: 'Batch Processing Results',
      ods: 'ODS (Optimal Dataset Scale)',
      ois: 'OIS (Optimal Image Scale)',
      threshold: 'Threshold',
      prCurve: 'PR Curve',
      processedResults: 'Processed Results ({count} images)',
      // Metrics labels (keep English)
      precision: 'Precision',
      recall: 'Recall',
      f1Score: 'F1-Score'
    },
    // Vehicle Detection Page
    vehicleDetection: {
      title: 'Vehicle Detection System',
      subtitle: 'Based on edge detection + shape features',
      algorithm: 'Edge Detection Algorithm:',
      rectangularity: 'Rectangularity Threshold:',
      selectImage: '📁 Select Vehicle Image',
      startDetection: '🔍 Start Detection',
      processing: 'Processing...',
      originalImage: 'Original Image (red rectangle marks detection result)',
      edgeResult: 'Edge Detection Result',
      selectImagePlaceholder: 'Please select an image and start detection',
      edgeResultPlaceholder: 'Edge detection result will be displayed here',
      result: 'Detection Result',
      features: 'Vehicle Features:',
      area: 'Contour Area',
      aspectRatio: 'Aspect Ratio',
      rectangularity: 'Rectangularity',
      mainColor: 'Main Color RGB',
      requirement: 'Requirement',
      tips: '💡 Usage Tips:',
      tip1: 'Select vehicle images with side/front view and simple background',
      tip2: 'Ensure sufficient lighting, avoid dark images',
      tip3: 'No severe occlusion on vehicle body',
      tip4: 'Lower rectangularity threshold makes it easier to detect irregular contours'
    },
    // Documentation Page
    documentation: {
      title: 'Documentation',
      edgeDetectionDocs: 'Edge Detection Algorithm Details',
      metricsDocs: 'Evaluation Metrics & ODS/OIS',
      loading: 'Loading...',
      loadError: 'Failed to load document'
    },
    // Appearance Settings
    appearanceSettings: {
      title: 'Appearance Settings',
      backgroundColor: 'Background Color:',
      containerColor: 'Container Color:',
      textColor: 'Text Color:',
      resetDefault: 'Reset to Default (Black & Gold)'
    },
    // Common
    common: {
      close: '×',
      back: 'Back',
      home: 'Home'
    }
  }
}

// 获取当前语言
export function getCurrentLanguage() {
  return localStorage.getItem('language') || 'zh'
}

// 设置语言
export function setLanguage(lang) {
  localStorage.setItem('language', lang)
}

// 获取翻译文本
export function t(key, lang = null) {
  const currentLang = lang || getCurrentLanguage()
  const keys = key.split('.')
  let value = i18n[currentLang]
  
  for (const k of keys) {
    if (value && typeof value === 'object') {
      value = value[k]
    } else {
      return key // 如果找不到，返回原始key
    }
  }
  
  return value || key
}

