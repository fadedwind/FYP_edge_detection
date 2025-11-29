<template>
  <div class="page-container">
    <header class="header">
      <h1>🚗 车辆识别系统</h1>
      <p class="subtitle">基于边缘检测 + 形状特征的车辆识别</p>
    </header>

    <main class="main-content">
      <div class="top-bar">
        <button class="nav-btn" @click="$router.push('/')">🏠 返回主页（边缘检测）</button>
      </div>

      <!-- 控制面板 -->
      <div class="control-panel">
        <div class="control-group">
          <label for="algorithm">边缘检测算法：</label>
          <select id="algorithm" v-model="selectedAlgorithm" class="select-input">
            <option v-for="algo in algorithms" :key="algo" :value="algo">{{ algo }}</option>
          </select>
        </div>

        <div class="control-group">
          <label for="rectangularity">矩形度阈值：</label>
          <input 
            type="range" 
            id="rectangularity" 
            v-model.number="minRectangularity" 
            min="0.1" 
            max="0.9" 
            step="0.05"
            class="slider"
          />
          <span class="value-display">{{ minRectangularity.toFixed(2) }}</span>
        </div>

        <div class="control-group">
          <input 
            type="file" 
            id="file-input" 
            ref="fileInput" 
            @change="handleFileSelect" 
            accept="image/*"
            class="file-input"
          />
          <label for="file-input" class="file-label">📁 选择车辆图片</label>
        </div>

        <button 
          @click="startDetection" 
          :disabled="!selectedImage || processing"
          class="detect-btn"
        >
          {{ processing ? '处理中...' : '🔍 开始识别' }}
        </button>
      </div>

      <!-- 图片展示区 -->
      <div class="image-section">
        <div class="image-box">
          <h3>原始图片（红色矩形标记识别结果）</h3>
          <div class="image-container">
            <img v-if="markedImage" :src="markedImage" alt="标记后的图片" class="result-image" />
            <div v-else class="placeholder">请选择图片并开始识别</div>
          </div>
        </div>

        <div class="image-box">
          <h3>边缘检测结果</h3>
          <div class="image-container">
            <img v-if="edgeImage" :src="edgeImage" alt="边缘检测结果" class="result-image" />
            <div v-else class="placeholder">边缘检测结果将显示在这里</div>
          </div>
        </div>
      </div>

      <!-- 识别结果 -->
      <div class="result-section" v-if="result">
        <div class="result-card" :class="{ 'success': isVehicle, 'failed': !isVehicle }">
          <h3>识别结果</h3>
          <p class="result-text">{{ result }}</p>
          
          <div class="features" v-if="features">
            <h4>车辆特征：</h4>
            <ul>
              <li>轮廓面积：<strong>{{ features.area }}</strong>（要求 >5000）</li>
              <li>长宽比：<strong>{{ features.aspect_ratio }}</strong>（要求 1.2-5.0）</li>
              <li>矩形度：<strong>{{ features.rectangularity }}</strong>（要求 ≥{{ minRectangularity.toFixed(2) }}）</li>
              <li>主色调 RGB：<strong>[{{ features.main_color.join(', ') }}]</strong></li>
            </ul>
          </div>
        </div>
      </div>

      <!-- 使用提示 -->
      <div class="tips">
        <h4>💡 使用提示：</h4>
        <ul>
          <li>选择侧面/正面视角、背景简单的车辆图片</li>
          <li>光线充足，避免暗图</li>
          <li>车身无严重遮挡</li>
          <li>矩形度阈值越小，越容易识别不规则轮廓</li>
        </ul>
      </div>
    </main>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'VehicleDetection',
  data() {
    return {
      selectedAlgorithm: '彩色Canny边缘检测',
      minRectangularity: 0.2,
      selectedImage: null,
      markedImage: null,
      edgeImage: null,
      result: null,
      isVehicle: false,
      features: null,
      algorithms: [],
      processing: false
    }
  },
  mounted() {
    this.loadAlgorithms()
  },
  methods: {
    async loadAlgorithms() {
      try {
        const response = await axios.get('/api/algorithms')
        this.algorithms = response.data.algorithms
      } catch (error) {
        console.error('加载算法列表失败:', error)
        this.algorithms = [
          'Canny边缘检测',
          '彩色Canny边缘检测',
          'Sobel边缘检测',
          '彩色Sobel边缘检测'
        ]
      }
    },
    handleFileSelect(event) {
      const file = event.target.files[0]
      if (!file) return
      
      const reader = new FileReader()
      reader.onload = (e) => {
        this.selectedImage = e.target.result
        this.markedImage = null
        this.edgeImage = null
        this.result = null
        this.features = null
      }
      reader.readAsDataURL(file)
    },
    async startDetection() {
      if (!this.selectedImage) {
        alert('请先选择图片！')
        return
      }
      
      this.processing = true
      try {
        const response = await axios.post('/api/detect', {
          image: this.selectedImage,
          algorithm: this.selectedAlgorithm,
          min_rectangularity: this.minRectangularity
        })
        
        if (response.data.success) {
          this.markedImage = response.data.images.marked
          this.edgeImage = response.data.images.edge
          this.result = response.data.result
          this.isVehicle = response.data.is_vehicle
          this.features = response.data.features
        } else {
          alert('识别失败：' + (response.data.error || '未知错误'))
        }
      } catch (error) {
        console.error('识别错误:', error)
        alert('识别失败：' + (error.response?.data?.error || error.message))
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


