<template>
  <div class="page-container">
    <header class="header">
      <h1>🖼️ 图像边缘检测主页</h1>
      <p class="subtitle">支持 Sobel / Canny / Prewitt / HED 等多种边缘检测算法</p>
    </header>

    <main class="main-content">
      <div class="top-bar">
        <button class="nav-btn" @click="showThemeSettings = !showThemeSettings">🎨 外观设置</button>
        <button class="nav-btn primary" @click="$router.push('/vehicle')">🚗 应用：汽车识别</button>
      </div>

      <!-- 外观设置面板 -->
      <div v-if="showThemeSettings" class="theme-panel">
        <div class="theme-header">
          <h3>外观设置</h3>
          <button class="close-btn" @click="showThemeSettings = false">×</button>
        </div>
        <div class="theme-content">
          <div class="theme-item">
            <label>背景颜色：</label>
            <input type="color" v-model="customBgColor" @change="applyTheme" class="color-picker" />
          </div>
          <div class="theme-item">
            <label>容器颜色：</label>
            <input type="color" v-model="customContainerColor" @change="applyTheme" class="color-picker" />
          </div>
          <div class="theme-item">
            <label>文字颜色：</label>
            <input type="color" v-model="customTextColor" @change="applyTheme" class="color-picker" />
          </div>
          <button class="nav-btn" @click="resetTheme">恢复默认（黑金）</button>
        </div>
      </div>

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
      sobelKsize: 3,
      dilateKsize: 1,
      areaMin: 8000,
      selectedImage: null,
      edgeImage: null,
      metrics: null,
      processing: false,
      showAdvanced: false,
      showThemeSettings: false,
      customBgColor: '#0a0a0a',
      customContainerColor: '#1a1a1a',
      customTextColor: '#d4af37'
    }
  },
  mounted() {
    // 加载保存的主题设置
    const savedBg = localStorage.getItem('customBgColor')
    const savedContainer = localStorage.getItem('customContainerColor')
    const savedText = localStorage.getItem('customTextColor')
    if (savedBg) this.customBgColor = savedBg
    if (savedContainer) this.customContainerColor = savedContainer
    if (savedText) this.customTextColor = savedText
    this.applyTheme()
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
    applyTheme() {
      document.body.style.backgroundColor = this.customBgColor
      document.documentElement.style.setProperty('--container-color', this.customContainerColor)
      document.documentElement.style.setProperty('--text-color', this.customTextColor)
      localStorage.setItem('customBgColor', this.customBgColor)
      localStorage.setItem('customContainerColor', this.customContainerColor)
      localStorage.setItem('customTextColor', this.customTextColor)
    },
    resetTheme() {
      this.customBgColor = '#0a0a0a'
      this.customContainerColor = '#1a1a1a'
      this.customTextColor = '#d4af37'
      this.applyTheme()
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
    }
  }
}
</script>

<style scoped>
@import '../styles/common.css';

.theme-panel {
  background: #1f1f1f;
  border: 1px solid rgba(212, 175, 55, 0.3);
  border-radius: 0;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
}

.theme-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
  border-bottom: 1px solid rgba(212, 175, 55, 0.2);
  padding-bottom: 10px;
}

.theme-header h3 {
  color: var(--text-color);
  margin: 0;
  font-size: 1.2em;
  transition: color 0.3s ease;
}

.close-btn {
  background: transparent;
  border: 1px solid rgba(212, 175, 55, 0.3);
  color: var(--text-color);
  width: 30px;
  height: 30px;
  border-radius: 0;
  cursor: pointer;
  font-size: 20px;
  line-height: 1;
  transition: all 0.2s;
}

.close-btn:hover {
  border-color: var(--text-color);
  background: rgba(212, 175, 55, 0.1);
}

.theme-content {
  display: flex;
  align-items: center;
  gap: 15px;
  flex-wrap: wrap;
}

.theme-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.theme-content label {
  color: var(--text-color);
  font-weight: 600;
  transition: color 0.3s ease;
}

.color-picker {
  width: 60px;
  height: 40px;
  border: 1px solid rgba(212, 175, 55, 0.3);
  border-radius: 0;
  cursor: pointer;
  background: #1a1a1a;
}

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
</style>


