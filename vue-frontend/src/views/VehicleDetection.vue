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

        <!-- YOLOv8不显示边缘图 -->
        <div class="image-box" v-if="selectedAlgorithm !== 'YOLOv8车辆检测'">
          <h3>{{ t('vehicleDetection.edgeResult') }}</h3>
          <div class="image-container">
            <img v-if="edgeImage" :src="edgeImage" :alt="t('vehicleDetection.edgeResult')" class="result-image" />
            <div v-else class="placeholder">{{ t('vehicleDetection.edgeResultPlaceholder') }}</div>
          </div>
        </div>
      </div>

      <!-- 识别结果 -->
      <div class="result-section" v-if="result">
        <!-- YOLOv8检测结果 -->
        <div class="result-card success" v-if="selectedAlgorithm === 'YOLOv8车辆检测'">
          <h3>{{ t('vehicleDetection.result') }}</h3>
          <p class="result-text">{{ result }}</p>

          <!-- 车辆统计 -->
          <div class="features" v-if="vehicleCounts && Object.keys(vehicleCounts).length > 0">
            <h4>车辆类型统计</h4>
            <ul>
              <li v-for="(count, type) in vehicleCounts" :key="type">
                <strong>{{ type }}</strong>: {{ count }} 辆
              </li>
            </ul>
          </div>

          <!-- 详细检测列表 -->
          <div class="detection-list" v-if="detections && detections.length > 0">
            <h4>检测详情</h4>
            <div class="detection-table">
              <table>
                <thead>
                  <tr>
                    <th>序号</th>
                    <th>类型</th>
                    <th>置信度</th>
                    <th>位置</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(det, index) in detections" :key="index">
                    <td>{{ index + 1 }}</td>
                    <td><strong>{{ det.class_name }}</strong></td>
                    <td>{{ (det.confidence * 100).toFixed(1) }}%</td>
                    <td>[{{ det.bbox.join(', ') }}]</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <!-- 边缘检测结果 -->
        <div class="result-card" :class="{ 'success': isVehicle, 'failed': !isVehicle }" v-else>
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
      selectedAlgorithm: 'color-canny',
      minRectangularity: 0.2,
      selectedImage: null,
      markedImage: null,
      edgeImage: null,
      result: null,
      isVehicle: false,
      features: null,
      algorithms: [],
      processing: false,
      // YOLOv8检测结果
      detections: null,
      vehicleCounts: null,
      totalVehicles: 0
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
          'color-canny',
          'Sobel边缘检测',
          'color-sobel',
          'YOLOv8车辆检测'
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

          // YOLOv8特殊处理
          if (this.selectedAlgorithm === 'YOLOv8车辆检测') {
            this.edgeImage = null  // YOLOv8不生成边缘图
            this.result = response.data.classification
            this.detections = response.data.detections || []
            this.vehicleCounts = response.data.vehicle_counts || {}
            this.totalVehicles = response.data.total_vehicles || 0
            this.features = null
            this.isVehicle = this.totalVehicles > 0
          } else {
            // 边缘检测算法
            this.edgeImage = response.data.images.edge
            this.result = response.data.classification
            this.isVehicle = response.data.is_vehicle
            this.features = response.data.features
            this.detections = null
            this.vehicleCounts = null
            this.totalVehicles = 0
          }
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

.detection-list {
  margin-top: 1rem;
}

.detection-table {
  overflow-x: auto;
  margin-top: 0.5rem;
}

.detection-table table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}

.detection-table thead {
  background-color: rgba(212, 175, 55, 0.2);
}

.detection-table th,
.detection-table td {
  padding: 0.75rem;
  text-align: left;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.detection-table th {
  font-weight: 600;
  color: #D4AF37;
}

.detection-table tbody tr:hover {
  background-color: rgba(212, 175, 55, 0.1);
}

.detection-table tbody tr:last-child td {
  border-bottom: none;
}
</style>


