<template>
  <div class="page-container">
    <header class="header">
      <h1>🖼️ 图像边缘检测主页</h1>
      <p class="subtitle">支持 Sobel / Canny / Prewitt / HED 等多种边缘检测算法</p>
    </header>

    <main class="main-content">

      <!-- 控制面板 -->
      <div class="control-panel">
        <div class="control-group">
          <label for="ed-algorithm">边缘检测算法：</label>
          <select id="ed-algorithm" v-model="algorithm" class="select-input">
            <option v-for="algo in algoOptions" :key="algo" :value="algo">{{ algo }}</option>
          </select>
        </div>

        <!-- 更多参数按钮 -->
        <button class="nav-btn" @click="showAdvanced = !showAdvanced">
          {{ showAdvanced ? '收起更多参数' : '更多参数调整' }}
        </button>

        <div class="control-group">
          <label for="blur">模糊核大小：</label>
          <input
            id="blur"
            type="range"
            min="1"
            max="31"
            step="2"
            v-model.number="blur"
            class="slider"
          />
          <span class="value-display">{{ blur }}</span>
        </div>

        <div class="control-group" v-if="algorithm.includes('Canny')">
          <label> Canny 阈值：</label>
          <input
            type="number"
            v-model.number="cannyLow"
            min="0"
            max="255"
            class="num-input"
          />
          <span>~</span>
          <input
            type="number"
            v-model.number="cannyHigh"
            min="0"
            max="255"
            class="num-input"
          />
        </div>

        <div class="control-group">
          <input
            type="file"
            id="edge-file-input"
            ref="fileInput"
            @change="handleFileSelect"
            accept="image/*"
            class="file-input"
          />
          <label for="edge-file-input" class="file-label">📁 选择图片</label>
        </div>

        <button
          @click="runEdgeDetection"
          :disabled="!selectedImage || processing"
          class="detect-btn"
        >
          {{ processing ? '处理中...' : '✨ 开始边缘检测' }}
        </button>
      </div>

      <!-- 高级参数面板（与桌面版参数对应，无预设） -->
      <div v-if="showAdvanced" class="advanced-panel">
        <div class="advanced-header">
          <span>高级参数（与桌面版 Trackbar 对应）：</span>
        </div>

        <div class="advanced-grid">
          <div class="advanced-item">
            <label>Sobel_Ksize（奇数）：{{ sobelKsize }}</label>
            <input
              type="range"
              min="1"
              max="7"
              step="2"
              v-model.number="sobelKsize"
              class="slider"
            />
          </div>

          <div class="advanced-item">
            <label>Blur（模糊核）：{{ blur }}</label>
            <input
              type="range"
              min="1"
              max="31"
              step="2"
              v-model.number="blur"
              class="slider"
            />
          </div>

          <div class="advanced-item">
            <label>Dilate（膨胀核）：{{ dilateKsize }}</label>
            <input
              type="range"
              min="1"
              max="15"
              step="2"
              v-model.number="dilateKsize"
              class="slider"
            />
          </div>

          <div class="advanced-item">
            <label>Canny_Low：{{ cannyLow }}</label>
            <input
              type="range"
              min="0"
              max="300"
              step="1"
              v-model.number="cannyLow"
              class="slider"
            />
          </div>

          <div class="advanced-item">
            <label>Canny_High：{{ cannyHigh }}</label>
            <input
              type="range"
              min="0"
              max="300"
              step="1"
              v-model.number="cannyHigh"
              class="slider"
            />
          </div>

          <div class="advanced-item">
            <label>Area（轮廓最小面积，示意参数）：{{ areaMin }}</label>
            <input
              type="range"
              min="0"
              max="30000"
              step="500"
              v-model.number="areaMin"
              class="slider"
            />
          </div>
        </div>
      </div>

      <!-- 图片展示区 -->
      <div class="image-section">
        <div class="image-box">
          <h3>原始图片</h3>
          <div class="image-container">
            <img v-if="selectedImage" :src="selectedImage" alt="原图" class="result-image" />
            <div v-else class="placeholder">请选择一张图片</div>
          </div>
        </div>

        <div class="image-box">
          <h3>边缘检测结果</h3>
          <div class="image-container">
            <img v-if="edgeImage" :src="edgeImage" alt="边缘图" class="result-image" />
            <div v-else class="placeholder">边缘结果将显示在这里</div>
          </div>
        </div>
      </div>

      <!-- 指标展示（可选） -->
      <div class="result-section" v-if="metrics && !batchResults">
        <div class="result-card">
          <h3>评估指标</h3>
          <div class="metrics-display">
            <p>非零边缘像素数：<strong>{{ metrics.edge_pixels }}</strong></p>
            <p v-if="metrics.precision !== undefined">Precision：<strong>{{ metrics.precision }}</strong></p>
            <p v-if="metrics.recall !== undefined">Recall：<strong>{{ metrics.recall }}</strong></p>
            <p v-if="metrics.f1 !== undefined">F1-Score：<strong>{{ metrics.f1 }}</strong></p>
          </div>
        </div>
      </div>

      <!-- 批量处理区域 -->
      <div class="batch-section">
        <div class="batch-header">
          <h3>📁 批量处理</h3>
          <button class="nav-btn" @click="showBatchPanel = !showBatchPanel">
            {{ showBatchPanel ? '收起批量处理' : '展开批量处理' }}
          </button>
        </div>

        <div v-if="showBatchPanel" class="batch-panel">
          <div class="control-group">
            <input
              type="file"
              id="batch-file-input"
              ref="batchFileInput"
              @change="handleBatchFileSelect"
              accept="image/*"
              multiple
              class="file-input"
            />
            <label for="batch-file-input" class="file-label">📁 选择多张图片（可多选）</label>
            <span v-if="batchFiles.length > 0" class="file-count">已选择 {{ batchFiles.length }} 张图片</span>
          </div>

          <button
            @click="startBatchProcess"
            :disabled="batchFiles.length === 0 || batchProcessing"
            class="detect-btn"
          >
            {{ batchProcessing ? '处理中...' : '🚀 开始批量处理' }}
          </button>

          <!-- 批量处理结果 -->
          <div v-if="batchResults" class="batch-results">
            <div class="result-card">
              <h3>批量处理结果</h3>
              <div class="metrics-grid">
                <div class="metric-item">
                  <h4>ODS (Optimal Dataset Scale)</h4>
                  <p>阈值: <strong>{{ batchResults.metrics.ods.threshold }}</strong></p>
                  <p>Precision: <strong>{{ batchResults.metrics.ods.precision }}</strong></p>
                  <p>Recall: <strong>{{ batchResults.metrics.ods.recall }}</strong></p>
                  <p>F1-Score: <strong>{{ batchResults.metrics.ods.f1 }}</strong></p>
                </div>
                <div class="metric-item">
                  <h4>OIS (Optimal Image Scale)</h4>
                  <p>Precision: <strong>{{ batchResults.metrics.ois.precision }}</strong></p>
                  <p>Recall: <strong>{{ batchResults.metrics.ois.recall }}</strong></p>
                  <p>F1-Score: <strong>{{ batchResults.metrics.ois.f1 }}</strong></p>
                </div>
              </div>

              <!-- PR 曲线 -->
              <div v-if="batchResults.pr_curve" class="pr-curve-container">
                <h4>PR 曲线</h4>
                <img :src="batchResults.pr_curve" alt="PR Curve" class="pr-curve-image" />
              </div>

              <!-- 处理结果列表 -->
              <div class="processed-images-list">
                <h4>处理结果（共 {{ batchResults.processed_count }} 张）</h4>
                <div class="images-grid">
                  <div v-for="img in batchResults.processed_images" :key="img.index" class="processed-image-item">
                    <img :src="img.edge_image" :alt="img.filename" class="processed-thumbnail" />
                    <p class="image-metrics">
                      P: {{ getImageMetrics(img.index).precision }} | 
                      R: {{ getImageMetrics(img.index).recall }} | 
                      F1: {{ getImageMetrics(img.index).f1 }}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'EdgeDetection',
  data() {
    return {
      algorithm: 'Canny',
      algoOptions: ['Sobel', '彩色Sobel', 'Canny', '彩色Canny', 'Prewitt', 'HED'],
      blur: 7,
      cannyLow: 100,
      cannyHigh: 220,
      sobelKsize: 3,
      dilateKsize: 1,
      areaMin: 8000,
      selectedImage: null,
      edgeImage: null,
      metrics: null,
      processing: false,
      showAdvanced: false,
      showBatchPanel: false,
      batchFiles: [],
      batchProcessing: false,
      batchResults: null
    }
  },
  mounted() {
    // 加载保存的主题设置
    const savedBg = localStorage.getItem('customBgColor')
    const savedContainer = localStorage.getItem('customContainerColor')
    const savedText = localStorage.getItem('customTextColor')
    if (savedBg) document.body.style.backgroundColor = savedBg
    if (savedContainer) document.documentElement.style.setProperty('--container-color', savedContainer)
    if (savedText) document.documentElement.style.setProperty('--text-color', savedText)
  },
  methods: {
    handleFileSelect(event) {
      const file = event.target.files[0]
      if (!file) return
      const reader = new FileReader()
      reader.onload = (e) => {
        this.selectedImage = e.target.result
        this.edgeImage = null
        this.metrics = null
      }
      reader.readAsDataURL(file)
    },
    async runEdgeDetection() {
      if (!this.selectedImage) {
        alert('请先选择图片！')
        return
      }
      this.processing = true
      try {
        const response = await axios.post('/api/edge-detect', {
          image: this.selectedImage,
          algorithm: this.algorithm,
          blur: this.blur,
          canny_low: this.cannyLow,
          canny_high: this.cannyHigh,
          sobel_ksize: this.sobelKsize,
          dilate_ksize: this.dilateKsize,
          area_min: this.areaMin
        })
        if (response.data.success) {
          this.edgeImage = response.data.images.edge
          this.metrics = response.data.metrics
        } else {
          alert('边缘检测失败：' + (response.data.error || '未知错误'))
        }
      } catch (error) {
        console.error('边缘检测错误:', error)
        alert('边缘检测失败：' + (error.response?.data?.error || error.message))
      } finally {
        this.processing = false
      }
    },
    handleBatchFileSelect(event) {
      const files = Array.from(event.target.files)
      this.batchFiles = files
    },
    async startBatchProcess() {
      if (this.batchFiles.length === 0) {
        alert('请先选择图片！')
        return
      }

      this.batchProcessing = true
      this.batchResults = null

      try {
        // 读取所有图片并转换为 base64
        const imagePromises = this.batchFiles.map(file => {
          return new Promise((resolve, reject) => {
            const reader = new FileReader()
            reader.onload = (e) => resolve(e.target.result)
            reader.onerror = reject
            reader.readAsDataURL(file)
          })
        })

        const images = await Promise.all(imagePromises)

        // 调用批量处理 API
        const response = await axios.post('/api/batch-process', {
          images: images,
          algorithm: this.algorithm,
          blur: this.blur,
          sobel_ksize: this.sobelKsize,
          canny_low: this.cannyLow,
          canny_high: this.cannyHigh
        })

        if (response.data.success) {
          this.batchResults = response.data
        } else {
          alert('批量处理失败：' + (response.data.error || '未知错误'))
        }
      } catch (error) {
        console.error('批量处理错误:', error)
        alert('批量处理失败：' + (error.response?.data?.error || error.message))
      } finally {
        this.batchProcessing = false
      }
    },
    getImageMetrics(index) {
      if (!this.batchResults || !this.batchResults.per_image_results) {
        return { precision: 0, recall: 0, f1: 0 }
      }
      const result = this.batchResults.per_image_results.find(r => r.index === index)
      return result || { precision: 0, recall: 0, f1: 0 }
    }
  }
}
</script>

<style scoped>
@import '../styles/common.css';

.advanced-panel {
  background: #1f1f1f;
  border: 1px solid rgba(212, 175, 55, 0.2);
  border-radius: 0;
  padding: 20px;
  margin-bottom: 24px;
}

.advanced-header {
  color: var(--text-color);
  font-weight: 600;
  margin-bottom: 15px;
  padding-bottom: 10px;
  border-bottom: 1px solid rgba(212, 175, 55, 0.2);
  transition: color 0.3s ease;
}

.advanced-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
}

.advanced-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.advanced-item label {
  color: var(--text-color);
  font-size: 13px;
  font-weight: 600;
  transition: color 0.3s ease;
}

.batch-section {
  margin-top: 30px;
  background: #1f1f1f;
  border: 1px solid rgba(212, 175, 55, 0.2);
  border-radius: 0;
  padding: 20px;
}

.batch-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
  border-bottom: 1px solid rgba(212, 175, 55, 0.2);
  padding-bottom: 10px;
}

.batch-header h3 {
  color: var(--text-color);
  margin: 0;
  font-size: 1.2em;
}

.batch-panel {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.file-count {
  color: var(--text-color);
  margin-left: 10px;
  font-size: 14px;
  opacity: 0.8;
}

.batch-results {
  margin-top: 20px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin: 20px 0;
}

.metric-item {
  background: rgba(0, 0, 0, 0.3);
  padding: 15px;
  border: 1px solid rgba(212, 175, 55, 0.2);
  border-radius: 0;
}

.metric-item h4 {
  color: var(--text-color);
  margin-bottom: 10px;
  font-size: 1.1em;
}

.metric-item p {
  color: var(--text-color);
  margin: 5px 0;
  opacity: 0.9;
}

.metrics-display {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.metrics-display p {
  color: var(--text-color);
  margin: 0;
  opacity: 0.9;
}

.pr-curve-container {
  margin: 20px 0;
  text-align: center;
}

.pr-curve-container h4 {
  color: var(--text-color);
  margin-bottom: 15px;
}

.pr-curve-image {
  max-width: 100%;
  height: auto;
  border: 1px solid rgba(212, 175, 55, 0.3);
  border-radius: 0;
}

.processed-images-list {
  margin-top: 20px;
}

.processed-images-list h4 {
  color: var(--text-color);
  margin-bottom: 15px;
}

.images-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 15px;
}

.processed-image-item {
  background: rgba(0, 0, 0, 0.3);
  padding: 10px;
  border: 1px solid rgba(212, 175, 55, 0.2);
  border-radius: 0;
  text-align: center;
}

.processed-thumbnail {
  width: 100%;
  height: auto;
  max-height: 150px;
  object-fit: contain;
  margin-bottom: 8px;
}

.image-metrics {
  color: var(--text-color);
  font-size: 12px;
  margin: 0;
  opacity: 0.8;
}

@media (max-width: 768px) {
  .metrics-grid {
    grid-template-columns: 1fr;
  }
  
  .images-grid {
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  }
}
</style>


