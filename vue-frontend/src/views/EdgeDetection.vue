<template>
  <div class="page-container">
    <header class="header">
      <h1>🖼️ 图像边缘检测主页</h1>
      <p class="subtitle">支持 Sobel / Canny / Prewitt / HED 等多种边缘检测算法</p>
    </header>

    <main class="main-content">
      <div class="top-bar">
        <button class="nav-btn primary" @click="$router.push('/vehicle')">🚗 应用：汽车识别</button>
      </div>

      <!-- 控制面板 -->
      <div class="control-panel">
        <div class="control-group">
          <label for="ed-algorithm">边缘检测算法：</label>
          <select id="ed-algorithm" v-model="algorithm" class="select-input">
            <option v-for="algo in algoOptions" :key="algo" :value="algo">{{ algo }}</option>
          </select>
        </div>

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
      <div class="result-section" v-if="metrics">
        <div class="result-card">
          <h3>简单指标（仅供参考）</h3>
          <p>非零边缘像素数：<strong>{{ metrics.edge_pixels }}</strong></p>
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
      selectedImage: null,
      edgeImage: null,
      metrics: null,
      processing: false
    }
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
          canny_high: this.cannyHigh
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
    }
  }
}
</script>

<style scoped>
@import '../styles/common.css';
</style>


