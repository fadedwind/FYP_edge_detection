<template>
  <div class="page-container">
    <header class="header">
      <h1>🧠 {{ t('deepLearningEdgeDetection.title') }}</h1>
      <p class="subtitle">{{ t('deepLearningEdgeDetection.subtitle') }}</p>
    </header>

    <main class="main-content">
      <!-- 控制面板 -->
      <div class="control-panel">
        <div class="control-group">
          <label for="dl-algorithm">{{ t('deepLearningEdgeDetection.algorithm') }}</label>
          <select id="dl-algorithm" v-model="algorithm" class="select-input">
            <option v-for="algo in algoOptions" :key="algo" :value="algo">{{ algo }}</option>
          </select>
        </div>

        <!-- 模式切换 -->
        <div class="control-group">
          <label>{{ t('deepLearningEdgeDetection.imageMode') }} / {{ t('deepLearningEdgeDetection.videoMode') }}</label>
          <div class="mode-toggle">
            <button 
              @click="inputMode = 'image'" 
              :class="['mode-btn', { active: inputMode === 'image' }]"
            >
              {{ t('deepLearningEdgeDetection.imageMode') }}
            </button>
            <button 
              @click="inputMode = 'video'" 
              :class="['mode-btn', { active: inputMode === 'video' }]"
            >
              {{ t('deepLearningEdgeDetection.videoMode') }}
            </button>
          </div>
        </div>

        <!-- 图片输入 -->
        <div v-if="inputMode === 'image'" class="control-group">
          <input
            type="file"
            id="dl-file-input"
            ref="fileInput"
            @change="handleFileSelect"
            accept="image/*"
            class="file-input"
          />
          <label for="dl-file-input" class="file-label">{{ t('deepLearningEdgeDetection.selectImage') }}</label>
        </div>

        <!-- 视频输入 -->
        <div v-if="inputMode === 'video'" class="control-group">
          <input
            type="file"
            id="dl-video-input"
            ref="videoInput"
            @change="handleVideoSelect"
            accept="video/*"
            class="file-input"
          />
          <label for="dl-video-input" class="file-label">{{ t('deepLearningEdgeDetection.selectVideo') }}</label>
        </div>

        <button
          v-if="inputMode === 'image'"
          @click="runEdgeDetection"
          :disabled="!selectedImage || processing"
          class="detect-btn"
        >
          {{ processing ? t('deepLearningEdgeDetection.processing') : t('deepLearningEdgeDetection.startDetection') }}
        </button>
      </div>

      <!-- 图片展示区 -->
      <div v-if="inputMode === 'image'" class="image-section">
        <div class="image-box">
          <div class="image-box-header">
            <h3>{{ t('deepLearningEdgeDetection.originalImage') }}</h3>
          </div>
          <div class="image-container">
            <img v-if="selectedImage" :src="selectedImage" :alt="t('deepLearningEdgeDetection.originalImage')" class="result-image" />
            <div v-else class="placeholder">{{ t('deepLearningEdgeDetection.selectImagePlaceholder') }}</div>
          </div>
        </div>

        <div class="image-box">
          <div class="image-box-header">
            <h3>{{ t('deepLearningEdgeDetection.edgeResult') }}</h3>
            <button
              v-if="edgeImage"
              @click="saveEdgeImage"
              class="save-btn"
              :title="t('deepLearningEdgeDetection.saveImage')"
            >
              {{ t('deepLearningEdgeDetection.saveImage') }}
            </button>
          </div>
          <div class="image-container">
            <img v-if="edgeImage" :src="edgeImage" :alt="t('deepLearningEdgeDetection.edgeResult')" class="result-image" />
            <div v-else class="placeholder">{{ t('deepLearningEdgeDetection.edgeResultPlaceholder') }}</div>
          </div>
        </div>
      </div>

      <!-- 视频展示区 -->
      <div v-if="inputMode === 'video'" class="video-section">
        <div class="video-box">
          <div class="image-box-header">
            <h3>{{ t('deepLearningEdgeDetection.originalImage') }}</h3>
          </div>
          <div class="video-container">
            <video
              v-if="selectedVideo"
              ref="videoPlayer"
              :src="selectedVideo"
              @loadedmetadata="onVideoLoaded"
              @timeupdate="onVideoTimeUpdate"
              @play="onVideoPlay"
              @pause="onVideoPause"
              class="video-player"
              controls
            ></video>
            <div v-else class="placeholder">{{ t('deepLearningEdgeDetection.selectVideoPlaceholder') }}</div>
          </div>
        </div>

        <div class="video-box">
          <div class="image-box-header">
            <h3>{{ t('deepLearningEdgeDetection.edgeResult') }}</h3>
            <button
              v-if="selectedVideo && hasVideoEdgeResult"
              @click="saveVideoFrame"
              class="save-btn"
              :title="t('deepLearningEdgeDetection.saveImage')"
            >
              {{ t('deepLearningEdgeDetection.saveImage') }}
            </button>
          </div>
          <div class="video-container">
            <canvas
              ref="edgeCanvas"
              class="edge-canvas"
            ></canvas>
            <div v-if="!selectedVideo" class="placeholder">{{ t('deepLearningEdgeDetection.edgeResultPlaceholder') }}</div>
          </div>
        </div>
      </div>

      <!-- 指标展示 -->
      <div class="result-section" v-if="metrics">
        <div class="result-card">
          <h3>{{ t('deepLearningEdgeDetection.metrics') }}</h3>
          <div class="metrics-display">
            <p>{{ t('deepLearningEdgeDetection.edgePixels') }}<strong>{{ metrics.edge_pixels }}</strong></p>
            <p v-if="metrics.precision !== undefined">{{ t('deepLearningEdgeDetection.precision') }}：<strong>{{ metrics.precision }}</strong></p>
            <p v-if="metrics.recall !== undefined">{{ t('deepLearningEdgeDetection.recall') }}：<strong>{{ metrics.recall }}</strong></p>
            <p v-if="metrics.f1 !== undefined">{{ t('deepLearningEdgeDetection.f1Score') }}：<strong>{{ metrics.f1 }}</strong></p>
          </div>
        </div>
      </div>

      <!-- HED算法说明 -->
      <div v-if="algorithm === 'HED'" class="info-section">
        <div class="info-card">
          <h3>💡 HED算法说明</h3>
          <p>HED (Holistically-Nested Edge Detection) 是一种基于深度学习的边缘检测算法：</p>
          <ul>
            <li>使用VGG16作为骨干网络</li>
            <li>通过多个侧输出层融合不同尺度的特征</li>
            <li>能够检测更精细和连续的边缘</li>
            <li>需要预训练模型文件（Caffe或PyTorch格式）</li>
          </ul>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import axios from 'axios'
import { getCurrentLanguage, t as translate } from '../i18n'

export default {
  name: 'DeepLearningEdgeDetection',
  inject: ['currentLanguage'],
  data() {
    return {
      algorithm: 'HED',
      algoOptions: ['HED'],
      inputMode: 'image', // 'image' or 'video'
      selectedImage: null,
      selectedVideo: null,
      edgeImage: null,
      metrics: null,
      processing: false,
      videoProcessing: false,
      videoFrameInterval: null,
      lastProcessedFrame: 0,
      hasVideoEdgeResult: false
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
    handleVideoSelect(event) {
      const file = event.target.files[0]
      if (!file) return
      
      this.stopVideoProcessing()
      this.hasVideoEdgeResult = false
      
      const reader = new FileReader()
      reader.onload = (e) => {
        this.selectedVideo = e.target.result
        this.$nextTick(() => {
          if (this.$refs.videoPlayer) {
            this.$refs.videoPlayer.load()
          }
        })
      }
      reader.readAsDataURL(file)
    },
    async runEdgeDetection() {
      if (!this.selectedImage) {
        alert(this.t('deepLearningEdgeDetection.selectImagePlaceholder'))
        return
      }
      this.processing = true
      try {
        const response = await axios.post('/api/edge-detect', {
          image: this.selectedImage,
          algorithm: this.algorithm
        })
        if (response.data.success) {
          this.edgeImage = response.data.images.edge
          this.metrics = response.data.metrics
        } else {
          alert('Edge detection failed: ' + (response.data.error || 'Unknown error'))
        }
      } catch (error) {
        console.error('Edge detection error:', error)
        alert('Edge detection failed: ' + (error.response?.data?.error || error.message))
      } finally {
        this.processing = false
      }
    },
    onVideoLoaded() {
      if (this.$refs.videoPlayer && this.$refs.edgeCanvas) {
        const video = this.$refs.videoPlayer
        const canvas = this.$refs.edgeCanvas
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
      }
    },
    onVideoTimeUpdate() {
      if (this.videoProcessing && this.$refs.videoPlayer) {
        const video = this.$refs.videoPlayer
        if (!video.paused && !video.ended) {
          if (!this.videoFrameInterval) {
            this.processVideoFrameLoop()
          }
        }
      }
    },
    onVideoPlay() {
      this.startVideoProcessing()
    },
    onVideoPause() {
      this.stopVideoProcessing()
    },
    async startVideoProcessing() {
      if (!this.selectedVideo || !this.$refs.videoPlayer) return
      
      this.videoProcessing = true
      this.lastProcessedFrame = -1
      await this.processVideoFrame()
    },
    stopVideoProcessing() {
      this.videoProcessing = false
      if (this.videoFrameInterval) {
        cancelAnimationFrame(this.videoFrameInterval)
        this.videoFrameInterval = null
      }
    },
    async processVideoFrameLoop() {
      if (!this.videoProcessing || !this.$refs.videoPlayer) {
        this.videoFrameInterval = null
        return
      }
      
      const video = this.$refs.videoPlayer
      if (video.paused || video.ended) {
        this.videoFrameInterval = null
        return
      }
      
      await this.processVideoFrame()
      
      this.videoFrameInterval = requestAnimationFrame(() => {
        this.processVideoFrameLoop()
      })
    },
    async processVideoFrame() {
      if (!this.$refs.videoPlayer || !this.$refs.edgeCanvas) return
      
      const video = this.$refs.videoPlayer
      const canvas = this.$refs.edgeCanvas
      const ctx = canvas.getContext('2d')
      
      if (video.readyState < 2) return
      
      const tempCanvas = document.createElement('canvas')
      tempCanvas.width = video.videoWidth || 640
      tempCanvas.height = video.videoHeight || 480
      const tempCtx = tempCanvas.getContext('2d')
      tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height)
      
      const frameData = tempCanvas.toDataURL('image/jpeg', 0.8)
      
      try {
        const response = await axios.post('/api/edge-detect', {
          image: frameData,
          algorithm: this.algorithm
        })
        
        if (response.data.success && response.data.images.edge) {
          const img = new Image()
          img.onload = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height)
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
            this.hasVideoEdgeResult = true
          }
          img.onerror = () => {
            console.error('Failed to load edge image')
            this.hasVideoEdgeResult = false
          }
          img.src = response.data.images.edge
        }
      } catch (error) {
        console.error('Video frame processing error:', error)
      }
    },
    saveEdgeImage() {
      if (!this.edgeImage) {
        alert(this.t('deepLearningEdgeDetection.saveError'))
        return
      }
      
      try {
        const link = document.createElement('a')
        link.href = this.edgeImage
        link.download = `edge_detection_${this.algorithm}_${new Date().getTime()}.png`
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        alert(this.t('deepLearningEdgeDetection.saveSuccess'))
      } catch (error) {
        console.error('Save image error:', error)
        alert(this.t('deepLearningEdgeDetection.saveError') + ': ' + error.message)
      }
    },
    saveVideoFrame() {
      if (!this.$refs.edgeCanvas) {
        alert(this.t('deepLearningEdgeDetection.saveError'))
        return
      }
      
      try {
        const canvas = this.$refs.edgeCanvas
        if (canvas.width === 0 || canvas.height === 0) {
          alert(this.t('deepLearningEdgeDetection.saveError') + ': ' + 'No edge detection result available')
          return
        }
        
        canvas.toBlob((blob) => {
          if (!blob) {
            alert(this.t('deepLearningEdgeDetection.saveError'))
            return
          }
          
          const url = URL.createObjectURL(blob)
          const link = document.createElement('a')
          link.href = url
          link.download = `edge_detection_video_${this.algorithm}_${new Date().getTime()}.png`
          document.body.appendChild(link)
          link.click()
          document.body.removeChild(link)
          URL.revokeObjectURL(url)
          alert(this.t('deepLearningEdgeDetection.saveSuccess'))
        }, 'image/png')
      } catch (error) {
        console.error('Save video frame error:', error)
        alert(this.t('deepLearningEdgeDetection.saveError') + ': ' + error.message)
      }
    }
  },
  beforeUnmount() {
    this.stopVideoProcessing()
  }
}
</script>

<style scoped>
@import '../styles/common.css';

.mode-toggle {
  display: flex;
  gap: 10px;
}

.mode-btn {
  flex: 1;
  padding: 8px 16px;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(212, 175, 55, 0.3);
  color: var(--text-color);
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 14px;
}

.mode-btn:hover {
  background: rgba(212, 175, 55, 0.1);
  border-color: rgba(212, 175, 55, 0.5);
}

.mode-btn.active {
  background: rgba(212, 175, 55, 0.2);
  border-color: var(--text-color);
  font-weight: 600;
}

.image-box-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.image-box-header h3 {
  color: var(--text-color);
  margin: 0;
  font-size: 1.2em;
  transition: color 0.3s ease;
}

.save-btn {
  padding: 8px 16px;
  background: rgba(212, 175, 55, 0.2);
  border: 1px solid rgba(212, 175, 55, 0.5);
  color: var(--text-color);
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 14px;
  white-space: nowrap;
}

.save-btn:hover {
  background: rgba(212, 175, 55, 0.3);
  border-color: var(--text-color);
  transform: translateY(-1px);
}

.save-btn:active {
  transform: translateY(0);
}

.video-section {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-top: 24px;
}

.video-box {
  background: var(--container-color);
  border: 1px solid rgba(212, 175, 55, 0.2);
  border-radius: 0;
  padding: 20px;
  transition: border-color 0.3s ease, background-color 0.3s ease;
}

.video-container {
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #000;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}

.video-player {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.edge-canvas {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #000;
}

.info-section {
  margin-top: 30px;
}

.info-card {
  background: var(--container-color);
  border: 1px solid rgba(212, 175, 55, 0.2);
  border-radius: 0;
  padding: 20px;
}

.info-card h3 {
  color: var(--text-color);
  margin-bottom: 15px;
}

.info-card p {
  color: var(--text-color);
  opacity: 0.9;
  margin-bottom: 10px;
}

.info-card ul {
  color: var(--text-color);
  opacity: 0.9;
  margin-left: 20px;
}

.info-card li {
  margin-bottom: 8px;
}

@media (max-width: 768px) {
  .video-section {
    grid-template-columns: 1fr;
  }
}
</style>

