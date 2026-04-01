// 国际化配置
export const i18n = {
  zh: {
    // 侧边栏
    sidebar: {
      title: '导航',
      classicalEdgeDetection: '经典算法边缘检测',
      deepLearningEdgeDetection: '深度学习边缘检测',
      vehicleDetection: '应用：车辆识别',
      smartRemoval: '智能抠图',
      documentation: '算法原理',
      appearanceSettings: '外观设置',
      changelog: '开发日志',
      github: 'GitHub 仓库'
    },
    // 边缘检测页面
    edgeDetection: {
      title: '经典算法边缘检测',
      subtitleClassical: '支持 Sobel / Canny / Prewitt 等经典边缘检测算法',
      algorithm: '边缘检测算法：',
      moreParams: '更多参数调整',
      collapseParams: '收起更多参数',
      blur: '模糊核大小：',
      cannyThreshold: 'Canny 阈值：',
      selectImage: '📁 选择图片',
      selectVideo: '📹 选择视频',
      startDetection: '✨ 开始边缘检测',
      processing: '处理中...',
      saveImage: '💾 保存边缘检测结果',
      saveSuccess: '保存成功',
      saveError: '保存失败',
      videoMode: '视频模式',
      imageMode: '图片模式',
      videoPlaying: '视频播放中',
      videoPaused: '视频已暂停',
      selectVideoPlaceholder: '请选择一个视频文件',
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
      quickTestBSDS500: '⚡ 快速测试:BSDS500',
      quickTestProcessing: 'BSDS500测试处理中...',
      batchResults: '批量处理结果',
      ods: 'ODS (Optimal Dataset Scale)',
      ois: 'OIS (Optimal Image Scale)',
      threshold: '阈值',
      prCurve: 'PR 曲线',
      processedResults: '处理结果（共 {count} 张）',
      // 指标标签（保持英文）
      precision: 'Precision',
      recall: 'Recall',
      f1Score: 'F1-Score',
      // 评估参数（BSDS500标准）
      evaluationParams: '评估参数（BSDS500标准）',
      useTolerance: '使用边界匹配容差',
      maxDist: '最大匹配距离（图像对角线比例）',
      maxDistDesc: 'BSDS500标准：0.0075，NYUD：0.011',
      useThinning: '边界细化（Thinning）',
      useThinningDesc: '将边界细化为单像素宽度'
    },
    // 深度学习边缘检测页面
    deepLearningEdgeDetection: {
      title: '深度学习边缘检测',
      subtitle: '支持 HED、PiDiNet 等基于深度学习的边缘检测算法',
      algorithm: '边缘检测算法：',
      selectImage: '📁 选择图片',
      selectVideo: '📹 选择视频',
      startDetection: '✨ 开始边缘检测',
      processing: '处理中...',
      saveImage: '💾 保存边缘检测结果',
      saveSuccess: '保存成功',
      saveError: '保存失败',
      videoMode: '视频模式',
      imageMode: '图片模式',
      videoPlaying: '视频播放中',
      videoPaused: '视频已暂停',
      selectVideoPlaceholder: '请选择一个视频文件',
      originalImage: '原始图片',
      edgeResult: '边缘检测结果',
      selectImagePlaceholder: '请选择一张图片',
      edgeResultPlaceholder: '边缘结果将显示在这里',
      metrics: '评估指标',
      edgePixels: '非零边缘像素数：',
      precision: 'Precision',
      recall: 'Recall',
      f1Score: 'F1-Score',
      hedDescription: 'HED算法说明',
      hedIntro: 'HED (Holistically-Nested Edge Detection) 是一种基于深度学习的边缘检测算法：',
      hedFeature1: '使用VGG16作为骨干网络',
      hedFeature2: '通过多个侧输出层融合不同尺度的特征',
      hedFeature3: '能够检测更精细和连续的边缘',
      hedFeature4: '需要预训练模型文件（Caffe或PyTorch格式）',
      pidinetDescription: 'PiDiNet算法说明',
      pidinetIntro: 'PiDiNet (Pixel Difference Networks) 是一种高效的边缘检测深度学习算法：',
      pidinetFeature1: '使用像素差异卷积（PDC）进行边缘检测',
      pidinetFeature2: '相比传统方法更加高效和准确',
      pidinetFeature3: '能够检测细粒度的边缘特征',
      pidinetFeature4: '需要预训练模型文件（PyTorch格式）',
      hedModelStatus: 'HED模型状态',
      pidinetModelStatus: 'PiDiNet模型状态',
      hedModelLoaded: 'HED模型已加载',
      hedModelFailed: 'HED模型加载失败',
      pidinetModelLoaded: 'PiDiNet模型已加载',
      pidinetModelFailed: 'PiDiNet模型加载失败',
      // 批量处理
      batchProcessing: '📁 批量处理',
      expandBatch: '展开批量处理',
      collapseBatch: '收起批量处理',
      selectMultipleImages: '📁 选择多张图片（可多选）',
      selectedCount: '已选择 {count} 张图片',
      startBatch: '🚀 开始批量处理',
      batchProcessingStatus: '处理中...',
      quickTestBSDS500: '⚡ 快速测试:BSDS500',
      quickTestProcessing: 'BSDS500测试处理中...',
      batchResults: '批量处理结果',
      ods: 'ODS (Optimal Dataset Scale)',
      ois: 'OIS (Optimal Image Scale)',
      threshold: '阈值',
      prCurve: 'PR 曲线',
      processedResults: '处理结果（共 {count} 张）',
      // 评估参数（BSDS500标准）
      evaluationParams: '评估参数（BSDS500标准）',
      useTolerance: '使用边界匹配容差',
      maxDist: '最大匹配距离（图像对角线比例）',
      maxDistDesc: 'BSDS500标准：0.0075，NYUD：0.011',
      useThinning: '边界细化（Thinning）',
      useThinningDesc: '将边界细化为单像素宽度'
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
    // 智能抠图页面
    smartRemoval: {
      vehicleDetection: '车辆识别',
      smartRemoval: '智能抠图',
      title: '智能背景移除',
      subtitle: '提取前景对象，去除背景（支持游戏素材、人物、车辆等）',
      method: '抠图方法：',
      auto: '自动选择',
      recommended: '（推荐）',
      edgeAssisted: '边缘辅助',
      edgeThreshold: '边缘阈值：',
      selectImage: '📁 选择图片',
      startRemoval: '✂️ 开始抠图',
      processing: '处理中...',
      originalImage: '原始图片',
      result: '抠图结果（透明背景）',
      mask: 'Mask掩码',
      download: '下载结果',
      autoDescription: '自动选择最佳抠图方法：优先使用rembg，如果不可用则使用边缘辅助方法',
      rembgDescription: '基于U²-Net深度学习模型的高质量自动抠图',
      rembgAdvantage1: '自动识别前景和背景',
      rembgAdvantage2: '适用于人物、车辆、游戏角色等各种对象',
      rembgAdvantage3: '边缘平滑自然',
      edgeDescription: '结合边缘检测（PiDiNet/Canny）和颜色特征进行抠图',
      edgeAdvantage1: '与边缘检测算法紧密结合',
      edgeAdvantage2: '无需额外模型，使用现有边缘检测功能',
      grabcutDescription: 'OpenCV传统图割算法',
      grabcutAdvantage1: 'OpenCV内置，无需额外依赖',
      grabcutAdvantage2: '可能需要多次迭代，复杂场景效果一般',
      tips: '💡 使用提示：',
      tip1: 'rembg方法效果最好，推荐优先使用',
      tip2: '边缘辅助方法适合背景简单的图片',
      tip3: '抠图结果为PNG格式，背景透明',
      tip4: '可用于游戏素材提取、照片抠图、图标制作等'
    },
    // 文档页面
    documentation: {
      title: '算法原理文档',
      subtitle: '点击下方链接查看详细的算法原理说明',
      edgeDetectionDocs: '边缘检测算法详解',
      edgeDetectionDesc: '详细介绍 Sobel、Canny、Prewitt 等经典边缘检测算法的原理和实现',
      metricsDocs: '评估指标与 ODS/OIS',
      metricsDesc: '介绍边缘检测的评估指标，包括 Precision、Recall、F1-Score 以及 ODS/OIS 的计算方法'
    },
    // 外观设置
    appearanceSettings: {
      title: '外观设置',
      backgroundColor: '背景颜色：',
      containerColor: '容器颜色：',
      textColor: '文字颜色：',
      resetDefault: '恢复默认（黑金）'
    },
    // 开发日志
    changelog: {
      title: '开发日志',
      subtitle: 'Git 提交记录',
      refresh: '🔄 刷新',
      loading: '加载中',
      searchPlaceholder: '搜索提交记录...',
      loadError: '加载失败',
      noCommits: '暂无提交记录',
      changedFiles: '修改的文件',
      justNow: '刚刚',
      minutesAgo: '分钟前',
      hoursAgo: '小时前',
      daysAgo: '天前'
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
      classicalEdgeDetection: 'Classical Edge Detection',
      deepLearningEdgeDetection: 'Deep Learning Edge Detection',
      vehicleDetection: 'application:Vehicle Detection',
      smartRemoval: 'Smart Background Removal',
      documentation: 'Documentation',
      appearanceSettings: 'Appearance Settings',
      changelog: 'Changelog',
      github: 'GitHub Repository'
    },
    // Edge Detection Page
    edgeDetection: {
      title: 'Classical Edge Detection',
      subtitleClassical: 'Support Sobel / Canny / Prewitt and other classical edge detection algorithms',
      algorithm: 'Edge Detection Algorithm:',
      moreParams: 'More Parameters',
      collapseParams: 'Collapse Parameters',
      blur: 'Blur Kernel Size:',
      cannyThreshold: 'Canny Threshold:',
      selectImage: '📁 Select Image',
      selectVideo: '📹 Select Video',
      startDetection: '✨ Start Edge Detection',
      processing: 'Processing...',
      saveImage: '💾 Save Edge Detection Result',
      saveSuccess: 'Save successful',
      saveError: 'Save failed',
      videoMode: 'Video Mode',
      imageMode: 'Image Mode',
      videoPlaying: 'Video Playing',
      videoPaused: 'Video Paused',
      selectVideoPlaceholder: 'Please select a video file',
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
      quickTestBSDS500: '⚡ Quick Test:BSDS500',
      quickTestProcessing: 'BSDS500 test processing...',
      batchResults: 'Batch Processing Results',
      ods: 'ODS (Optimal Dataset Scale)',
      ois: 'OIS (Optimal Image Scale)',
      threshold: 'Threshold',
      prCurve: 'PR Curve',
      processedResults: 'Processed Results ({count} images)',
      // Metrics labels (keep English)
      precision: 'Precision',
      recall: 'Recall',
      f1Score: 'F1-Score',
      // Evaluation Parameters (BSDS500 Standard)
      evaluationParams: 'Evaluation Parameters (BSDS500 Standard)',
      useTolerance: 'Use Boundary Matching Tolerance',
      maxDist: 'Max Matching Distance (Image Diagonal Ratio)',
      maxDistDesc: 'BSDS500 Standard: 0.0075, NYUD: 0.011',
      useThinning: 'Boundary Thinning',
      useThinningDesc: 'Thin boundaries to single-pixel width'
    },
    // Deep Learning Edge Detection Page
    deepLearningEdgeDetection: {
      title: 'Deep Learning Edge Detection',
      subtitle: 'Support HED, PiDiNet and other deep learning-based edge detection algorithms',
      algorithm: 'Edge Detection Algorithm:',
      selectImage: '📁 Select Image',
      selectVideo: '📹 Select Video',
      startDetection: '✨ Start Edge Detection',
      processing: 'Processing...',
      saveImage: '💾 Save Edge Detection Result',
      saveSuccess: 'Save successful',
      saveError: 'Save failed',
      videoMode: 'Video Mode',
      imageMode: 'Image Mode',
      videoPlaying: 'Video Playing',
      videoPaused: 'Video Paused',
      selectVideoPlaceholder: 'Please select a video file',
      originalImage: 'Original Image',
      edgeResult: 'Edge Detection Result',
      selectImagePlaceholder: 'Please select an image',
      edgeResultPlaceholder: 'Edge result will be displayed here',
      metrics: 'Evaluation Metrics',
      edgePixels: 'Non-zero Edge Pixels:',
      precision: 'Precision',
      recall: 'Recall',
      f1Score: 'F1-Score',
      hedDescription: 'HED Algorithm Description',
      hedIntro: 'HED (Holistically-Nested Edge Detection) is a deep learning-based edge detection algorithm:',
      hedFeature1: 'Uses VGG16 as the backbone network',
      hedFeature2: 'Fuses multi-scale features through multiple side output layers',
      hedFeature3: 'Can detect finer and more continuous edges',
      hedFeature4: 'Requires pre-trained model files (Caffe or PyTorch format)',
      pidinetDescription: 'PiDiNet Algorithm Description',
      pidinetIntro: 'PiDiNet (Pixel Difference Networks) is an efficient deep learning edge detection algorithm:',
      pidinetFeature1: 'Uses Pixel Difference Convolution (PDC) for edge detection',
      pidinetFeature2: 'More efficient and accurate compared to traditional methods',
      pidinetFeature3: 'Can detect fine-grained edge features',
      pidinetFeature4: 'Requires pre-trained model files (PyTorch format)',
      hedModelStatus: 'HED Model Status',
      pidinetModelStatus: 'PiDiNet Model Status',
      hedModelLoaded: 'HED Model Loaded',
      hedModelFailed: 'HED Model Failed',
      pidinetModelLoaded: 'PiDiNet Model Loaded',
      pidinetModelFailed: 'PiDiNet Model Failed',
      // Batch Processing
      batchProcessing: '📁 Batch Processing',
      expandBatch: 'Expand Batch Processing',
      collapseBatch: 'Collapse Batch Processing',
      selectMultipleImages: '📁 Select Multiple Images (Multiple Selection)',
      selectedCount: '{count} images selected',
      startBatch: '🚀 Start Batch Processing',
      batchProcessingStatus: 'Processing...',
      quickTestBSDS500: '⚡ Quick Test: BSDS500',
      quickTestProcessing: 'BSDS500 test processing...',
      batchResults: 'Batch Processing Results',
      ods: 'ODS (Optimal Dataset Scale)',
      ois: 'OIS (Optimal Image Scale)',
      threshold: 'Threshold',
      prCurve: 'PR Curve',
      processedResults: 'Processed Results ({count} images)',
      // Evaluation Parameters (BSDS500 Standard)
      evaluationParams: 'Evaluation Parameters (BSDS500 Standard)',
      useTolerance: 'Use Boundary Matching Tolerance',
      maxDist: 'Max Matching Distance (Image Diagonal Ratio)',
      maxDistDesc: 'BSDS500 Standard: 0.0075, NYUD: 0.011',
      useThinning: 'Boundary Thinning',
      useThinningDesc: 'Thin boundaries to single-pixel width'
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
    // Smart Background Removal
    smartRemoval: {
      vehicleDetection: 'Vehicle Detection',
      smartRemoval: 'Smart Removal',
      title: 'Smart Background Removal',
      subtitle: 'Extract foreground objects and remove background (supports game assets, portraits, vehicles, etc.)',
      method: 'Removal Method:',
      auto: 'Auto Select',
      recommended: '(Recommended)',
      edgeAssisted: 'Edge Assisted',
      edgeThreshold: 'Edge Threshold:',
      selectImage: '📁 Select Image',
      startRemoval: '✂️ Start Removal',
      processing: 'Processing...',
      originalImage: 'Original Image',
      result: 'Result (Transparent Background)',
      mask: 'Mask',
      download: 'Download Result',
      autoDescription: 'Automatically select the best method: prioritize rembg, fallback to edge-assisted if unavailable',
      rembgDescription: 'High-quality automatic background removal based on U²-Net deep learning model',
      rembgAdvantage1: 'Automatically identify foreground and background',
      rembgAdvantage2: 'Suitable for various objects like portraits, vehicles, game characters',
      rembgAdvantage3: 'Smooth and natural edges',
      edgeDescription: 'Background removal combining edge detection (PiDiNet/Canny) and color features',
      edgeAdvantage1: 'Tightly integrated with edge detection algorithms',
      edgeAdvantage2: 'No additional models needed, uses existing edge detection features',
      grabcutDescription: 'OpenCV traditional graph cut algorithm',
      grabcutAdvantage1: 'Built-in OpenCV, no extra dependencies',
      grabcutAdvantage2: 'May require multiple iterations, average results for complex scenes',
      tips: '💡 Usage Tips:',
      tip1: 'rembg method works best, recommended as first choice',
      tip2: 'Edge-assisted method works well for simple backgrounds',
      tip3: 'Result is in PNG format with transparent background',
      tip4: 'Suitable for game asset extraction, photo cropping, icon creation, etc.'
    },
    // Documentation Page
    documentation: {
      title: 'Algorithm Documentation',
      subtitle: 'Click the links below to view detailed algorithm explanations',
      edgeDetectionDocs: 'Edge Detection Algorithm Details',
      edgeDetectionDesc: 'Detailed introduction to classical edge detection algorithms including Sobel, Canny, Prewitt and their implementations',
      metricsDocs: 'Evaluation Metrics & ODS/OIS',
      metricsDesc: 'Introduction to edge detection evaluation metrics including Precision, Recall, F1-Score, and ODS/OIS calculation methods'
    },
    // Appearance Settings
    appearanceSettings: {
      title: 'Appearance Settings',
      backgroundColor: 'Background Color:',
      containerColor: 'Container Color:',
      textColor: 'Text Color:',
      resetDefault: 'Reset to Default (Black & Gold)'
    },
    // Changelog
    changelog: {
      title: 'Changelog',
      subtitle: 'Git Commit History',
      refresh: '🔄 Refresh',
      loading: 'Loading',
      searchPlaceholder: 'Search commits...',
      loadError: 'Failed to load',
      noCommits: 'No commits found',
      changedFiles: 'Changed files',
      justNow: 'Just now',
      minutesAgo: ' minutes ago',
      hoursAgo: ' hours ago',
      daysAgo: ' days ago'
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





