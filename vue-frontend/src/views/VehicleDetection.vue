<template>
  <div class="page-container">
    <header class="header">
      <h1>🚗 {{ t('vehicleDetection.title') }}</h1>
      <p class="subtitle">{{ t('vehicleDetection.subtitle') }}</p>
    </header>

    <main class="main-content">

      <!-- 控制面板 -->
      <div class="control-panel">
        <div class="control-group">
          <label for="algorithm">{{ t('vehicleDetection.algorithm') }}</label>
          <select id="algorithm" v-model="selectedAlgorithm" class="select-input">
            <option v-for="algo in algorithms" :key="algo" :value="algo">{{ algo }}</option>
          </select>
        </div>

        <div class="control-group">
          <label for="rectangularity">{{ t('vehicleDetection.rectangularity') }}</label>
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
          <label for="file-input" class="file-label">{{ t('vehicleDetection.selectImage') }}</label>
        </div>

        <button 
          @click="startDetection" 
          :disabled="!selectedImage || processing"
          class="detect-btn"
        >
          {{ processing ? t('vehicleDetection.processing') : t('vehicleDetection.startDetection') }}
        </button>
      </div>

      <!-- 图片展示区 -->
      <div class="image-section">
        <div class="image-box">
          <h3>{{ t('vehicleDetection.originalImage') }}</h3>
          <div class="image-container">
            <img v-if="markedImage" :src="markedImage" :alt="t('vehicleDetection.originalImage')" class="result-image" />
            <div v-else class="placeholder">{{ t('vehicleDetection.selectImagePlaceholder') }}</div>
          </div>
        </div>

        <div class="image-box">
          <h3>{{ t('vehicleDetection.edgeResult') }}</h3>
          <div class="image-container">
            <img v-if="edgeImage" :src="edgeImage" :alt="t('vehicleDetection.edgeResult')" class="result-image" />
            <div v-else class="placeholder">{{ t('vehicleDetection.edgeResultPlaceholder') }}</div>
          </div>
        </div>
      </div>

      <!-- 识别结果 -->
      <div class="result-section" v-if="result">
        <div class="result-card" :class="{ 'success': isVehicle, 'failed': !isVehicle }">
          <h3>{{ t('vehicleDetection.result') }}</h3>
          <p class="result-text">{{ result }}</p>
          
          <div class="features" v-if="features">
            <h4>{{ t('vehicleDetection.features') }}</h4>
            <ul>
              <li>{{ t('vehicleDetection.area') }}：<strong>{{ features.area }}</strong>（{{ t('vehicleDetection.requirement') }} >5000）</li>
              <li>{{ t('vehicleDetection.aspectRatio') }}：<strong>{{ features.aspect_ratio }}</strong>（{{ t('vehicleDetection.requirement') }} 1.2-5.0）</li>
              <li>{{ t('vehicleDetection.rectangularity') }}：<strong>{{ features.rectangularity }}</strong>（{{ t('vehicleDetection.requirement') }} ≥{{ minRectangularity.toFixed(2) }}）</li>
              <li>{{ t('vehicleDetection.mainColor') }}：<strong>[{{ features.main_color.join(', ') }}]</strong></li>
            </ul>
          </div>
        </div>
      </div>

      <!-- 使用提示 -->
      <div class="tips">
        <h4>{{ t('vehicleDetection.tips') }}</h4>
        <ul>
          <li>{{ t('vehicleDetection.tip1') }}</li>
          <li>{{ t('vehicleDetection.tip2') }}</li>
          <li>{{ t('vehicleDetection.tip3') }}</li>
          <li>{{ t('vehicleDetection.tip4') }}</li>
        </ul>
      </div>
    </main>
  </div>
</template>

<script>
import axios from 'axios'
import { getCurrentLanguage, t as translate } from '../i18n'

export default {
  name: 'VehicleDetection',
  inject: ['currentLanguage'],
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
  computed: {
    currentLang() {
      try {
        return this.currentLanguage && typeof this.currentLanguage === 'function' 
          ? this.currentLanguage() 
          : getCurrentLanguage()
      } catch (e) {
        return getCurrentLanguage()
      }
    }
  },
  watch: {
    currentLang() {
      this.$forceUpdate()
    }
  },
  methods: {
    t(key) {
      return translate(key, this.currentLang)
    },
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
  },
  mounted() {
    this.loadAlgorithms()
    // 加载保存的主题设置
    const savedBg = localStorage.getItem('customBgColor')
    const savedContainer = localStorage.getItem('customContainerColor')
    const savedText = localStorage.getItem('customTextColor')
    if (savedBg) document.body.style.backgroundColor = savedBg
    if (savedContainer) document.documentElement.style.setProperty('--container-color', savedContainer)
    if (savedText) document.documentElement.style.setProperty('--text-color', savedText)
  }
}
</script>

<style scoped>
@import '../styles/common.css';
</style>


