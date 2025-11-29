// 翻译文本对象
const translations = {
  zh: {
    // Sidebar
    nav: '导航',
    edgeDetection: '边缘检测',
    vehicleDetection: '汽车识别',
    documentation: '算法原理',
    appearanceSettings: '外观设置',
    language: '语言',
    chinese: '中文',
    english: 'English',
    
    // EdgeDetection
    edgeDetectionTitle: '图像边缘检测主页',
    edgeDetectionSubtitle: '支持 Sobel / Canny / Prewitt / HED 等多种边缘检测算法',
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
    pleaseSelectImage: '请选择一张图片',
    edgeResultPlaceholder: '边缘结果将显示在这里',
    advancedParams: '高级参数（与桌面版 Trackbar 对应）：',
    sobelKsize: 'Sobel_Ksize（奇数）：',
    blurKernel: 'Blur（模糊核）：',
    dilateKernel: 'Dilate（膨胀核）：',
    cannyLow: 'Canny_Low：',
    cannyHigh: 'Canny_High：',
    areaMin: 'Area（轮廓最小面积，示意参数）：',
    metrics: '评估指标',
    edgePixels: '非零边缘像素数：',
    precision: 'Precision：',
    recall: 'Recall：',
    f1Score: 'F1-Score：',
    batchProcessing: '📁 批量处理',
    expandBatch: '展开批量处理',
    collapseBatch: '收起批量处理',
    selectMultipleImages: '📁 选择多张图片（可多选）',
    selectedImages: '已选择 {count} 张图片',
    startBatch: '🚀 开始批量处理',
    batchResults: '批量处理结果',
    processedCount: '处理结果（共 {count} 张）',
    goToVehicle: '🚗 前往车辆识别',
    pleaseSelectImageFirst: '请先选择图片！',
    detectionFailed: '边缘检测失败：',
    unknownError: '未知错误',
    batchFailed: '批量处理失败：',
    
    // VehicleDetection
    vehicleDetectionTitle: '车辆识别系统',
    vehicleDetectionSubtitle: '基于边缘检测 + 形状特征的车辆识别',
    rectangularityThreshold: '矩形度阈值：',
    selectVehicleImage: '📁 选择车辆图片',
    startRecognition: '🔍 开始识别',
    markedImage: '原始图片（红色矩形标记识别结果）',
    pleaseSelectAndStart: '请选择图片并开始识别',
    recognitionResult: '识别结果',
    vehicleFeatures: '车辆特征：',
    contourArea: '轮廓面积：',
    areaRequirement: '（要求 >5000）',
    aspectRatio: '长宽比：',
    aspectRatioRequirement: '（要求 1.2-5.0）',
    rectangularity: '矩形度：',
    rectangularityRequirement: '（要求 ≥{value}）',
    mainColor: '主色调 RGB：',
    tips: '💡 使用提示：',
    tip1: '选择侧面/正面视角、背景简单的车辆图片',
    tip2: '光线充足，避免暗图',
    tip3: '车身无严重遮挡',
    tip4: '矩形度阈值越小，越容易识别不规则轮廓',
    goToEdgeDetection: '🖼️ 前往边缘检测',
    recognitionFailed: '识别失败：',
    
    // ThemeSettingsPanel
    appearanceSettingsTitle: '外观设置',
    backgroundColor: '背景颜色：',
    containerColor: '容器颜色：',
    textColor: '文字颜色：',
    resetDefault: '恢复默认（黑金）'
  },
  en: {
    // Sidebar
    nav: 'Navigation',
    edgeDetection: 'Edge Detection',
    vehicleDetection: 'Vehicle Detection',
    documentation: 'Documentation',
    appearanceSettings: 'Appearance',
    language: 'Language',
    chinese: '中文',
    english: 'English',
    
    // EdgeDetection
    edgeDetectionTitle: 'Image Edge Detection',
    edgeDetectionSubtitle: 'Support Sobel / Canny / Prewitt / HED and other edge detection algorithms',
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
    pleaseSelectImage: 'Please select an image',
    edgeResultPlaceholder: 'Edge result will be displayed here',
    advancedParams: 'Advanced Parameters (corresponding to desktop Trackbar):',
    sobelKsize: 'Sobel_Ksize (odd):',
    blurKernel: 'Blur (kernel):',
    dilateKernel: 'Dilate (kernel):',
    cannyLow: 'Canny_Low:',
    cannyHigh: 'Canny_High:',
    areaMin: 'Area (min contour area, reference):',
    metrics: 'Evaluation Metrics',
    edgePixels: 'Non-zero Edge Pixels:',
    precision: 'Precision:',
    recall: 'Recall:',
    f1Score: 'F1-Score:',
    batchProcessing: '📁 Batch Processing',
    expandBatch: 'Expand Batch Processing',
    collapseBatch: 'Collapse Batch Processing',
    selectMultipleImages: '📁 Select Multiple Images',
    selectedImages: '{count} images selected',
    startBatch: '🚀 Start Batch Processing',
    batchResults: 'Batch Processing Results',
    processedCount: 'Processed Results ({count} images)',
    goToVehicle: '🚗 Go to Vehicle Detection',
    pleaseSelectImageFirst: 'Please select an image first!',
    detectionFailed: 'Edge detection failed:',
    unknownError: 'Unknown error',
    batchFailed: 'Batch processing failed:',
    
    // VehicleDetection
    vehicleDetectionTitle: 'Vehicle Detection System',
    vehicleDetectionSubtitle: 'Vehicle detection based on edge detection + shape features',
    rectangularityThreshold: 'Rectangularity Threshold:',
    selectVehicleImage: '📁 Select Vehicle Image',
    startRecognition: '🔍 Start Recognition',
    markedImage: 'Original Image (red rectangles mark detection results)',
    pleaseSelectAndStart: 'Please select an image and start recognition',
    recognitionResult: 'Recognition Result',
    vehicleFeatures: 'Vehicle Features:',
    contourArea: 'Contour Area:',
    areaRequirement: '(requirement >5000)',
    aspectRatio: 'Aspect Ratio:',
    aspectRatioRequirement: '(requirement 1.2-5.0)',
    rectangularity: 'Rectangularity:',
    rectangularityRequirement: '(requirement ≥{value})',
    mainColor: 'Main Color RGB:',
    tips: '💡 Usage Tips:',
    tip1: 'Select vehicle images with side/front view and simple background',
    tip2: 'Ensure sufficient lighting, avoid dark images',
    tip3: 'No severe occlusion on vehicle body',
    tip4: 'Lower rectangularity threshold makes it easier to recognize irregular contours',
    goToEdgeDetection: '🖼️ Go to Edge Detection',
    recognitionFailed: 'Recognition failed:',
    
    // ThemeSettingsPanel
    appearanceSettingsTitle: 'Appearance Settings',
    backgroundColor: 'Background Color:',
    containerColor: 'Container Color:',
    textColor: 'Text Color:',
    resetDefault: 'Reset to Default (Black & Gold)'
  }
}

// 当前语言，默认中文
let currentLang = localStorage.getItem('appLanguage') || 'zh'

// 获取翻译函数
export function t(key, params = {}) {
  const translation = translations[currentLang]?.[key] || translations.zh[key] || key
  
  // 简单的参数替换
  if (params && Object.keys(params).length > 0) {
    return translation.replace(/\{(\w+)\}/g, (match, paramKey) => {
      return params[paramKey] !== undefined ? params[paramKey] : match
    })
  }
  
  return translation
}

// 设置语言
export function setLanguage(lang) {
  if (translations[lang]) {
    currentLang = lang
    localStorage.setItem('appLanguage', lang)
    // 触发自定义事件，通知所有组件更新
    window.dispatchEvent(new CustomEvent('languageChanged', { detail: lang }))
  }
}

// 获取当前语言
export function getCurrentLanguage() {
  return currentLang
}

// 导出翻译对象（用于组件中直接访问）
export function getTranslations() {
  return translations[currentLang]
}

