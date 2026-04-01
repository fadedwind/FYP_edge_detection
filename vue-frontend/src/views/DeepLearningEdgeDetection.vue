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

      <!-- 指标展示已移除：单张图片的指标没有意义（因为没有真实边缘参考） -->

      <!-- HED算法说明 -->
      <div v-if="algorithm === 'HED'" class="info-section">
        <div class="info-card">
          <h3>💡 {{ t('deepLearningEdgeDetection.hedDescription') }}</h3>
          <p>{{ t('deepLearningEdgeDetection.hedIntro') }}</p>
          <ul>
            <li>{{ t('deepLearningEdgeDetection.hedFeature1') }}</li>
            <li>{{ t('deepLearningEdgeDetection.hedFeature2') }}</li>
            <li>{{ t('deepLearningEdgeDetection.hedFeature3') }}</li>
            <li>{{ t('deepLearningEdgeDetection.hedFeature4') }}</li>
          </ul>
        </div>
      </div>

      <!-- PiDiNet算法说明 -->
      <div v-if="algorithm === 'PiDiNet'" class="info-section">
        <div class="info-card">
          <h3>💡 {{ t('deepLearningEdgeDetection.pidinetDescription') }}</h3>
          <p>{{ t('deepLearningEdgeDetection.pidinetIntro') }}</p>
          <ul>
            <li>{{ t('deepLearningEdgeDetection.pidinetFeature1') }}</li>
            <li>{{ t('deepLearningEdgeDetection.pidinetFeature2') }}</li>
            <li>{{ t('deepLearningEdgeDetection.pidinetFeature3') }}</li>
            <li>{{ t('deepLearningEdgeDetection.pidinetFeature4') }}</li>
          </ul>
        </div>
      </div>

      <!-- 批量处理区域 -->
      <div class="batch-section">
        <div class="batch-header">
          <h3>{{ t('deepLearningEdgeDetection.batchProcessing') }}</h3>
          <button class="nav-btn" @click="showBatchPanel = !showBatchPanel">
            {{ showBatchPanel ? t('deepLearningEdgeDetection.collapseBatch') : t('deepLearningEdgeDetection.expandBatch') }}
          </button>
        </div>

        <div v-if="showBatchPanel" class="batch-panel">
          <div class="control-group">
            <input
              type="file"
              id="dl-batch-file-input"
              ref="batchFileInput"
              @change="handleBatchFileSelect"
              accept="image/*"
              multiple
              class="file-input"
            />
            <label for="dl-batch-file-input" class="file-label">{{ t('deepLearningEdgeDetection.selectMultipleImages') }}</label>
            <span v-if="batchFiles.length > 0" class="file-count">{{ t('deepLearningEdgeDetection.selectedCount').replace('{count}', batchFiles.length) }}</span>
          </div>

          <!-- 评估参数设置 -->
          <div class="evaluation-params-panel">
            <div class="evaluation-params-header">
              <h4>{{ t('deepLearningEdgeDetection.evaluationParams') }}</h4>
            </div>
            <div class="evaluation-params-grid">
              <div class="evaluation-param-item">
                <label>
                  <input type="checkbox" v-model="useTolerance" />
                  {{ t('deepLearningEdgeDetection.useTolerance') }}
                </label>
                <p class="param-desc">{{ t('deepLearningEdgeDetection.maxDistDesc') }}</p>
              </div>
              <div v-if="useTolerance" class="evaluation-param-item">
                <label>{{ t('deepLearningEdgeDetection.maxDist') }}: {{ maxDist }}</label>
                <input
                  type="range"
                  min="0.001"
                  max="0.02"
                  step="0.0005"
                  v-model.number="maxDist"
                  class="slider"
                />
                <span class="param-value">{{ maxDist.toFixed(4) }}</span>
              </div>
              <div class="evaluation-param-item">
                <label>
                  <input type="checkbox" v-model="useThinning" />
                  {{ t('deepLearningEdgeDetection.useThinning') }}
                </label>
                <p class="param-desc">{{ t('deepLearningEdgeDetection.useThinningDesc') }}</p>
              </div>
            </div>
          </div>

          <div class="batch-buttons">
            <button
              @click="startBatchProcess"
              :disabled="batchFiles.length === 0 || batchProcessing || bsds500Processing"
              class="detect-btn"
            >
              {{ batchProcessing ? t('deepLearningEdgeDetection.batchProcessingStatus') : t('deepLearningEdgeDetection.startBatch') }}
            </button>
            
            <button
              @click="startBSDS500QuickTest"
              :disabled="batchProcessing || bsds500Processing"
              class="detect-btn bsds500-btn"
            >
              {{ bsds500Processing ? t('deepLearningEdgeDetection.quickTestProcessing') : t('deepLearningEdgeDetection.quickTestBSDS500') }}
            </button>
          </div>

          <!-- 批量处理结果 -->
          <div v-if="batchResults" class="batch-results">
            <!-- 指标显示（仅在有metrics时显示，即BSDS500快速测试） -->
            <div v-if="batchResults.metrics" class="metrics-grid">
              <div class="metric-item">
                <h4>{{ t('deepLearningEdgeDetection.ods') }}</h4>
                <p>{{ t('deepLearningEdgeDetection.threshold') }}: <strong>{{ batchResults.metrics.ods.threshold }}</strong></p>
                <p>{{ t('deepLearningEdgeDetection.precision') }}: <strong>{{ batchResults.metrics.ods.precision }}</strong></p>
                <p>{{ t('deepLearningEdgeDetection.recall') }}: <strong>{{ batchResults.metrics.ods.recall }}</strong></p>
                <p>{{ t('deepLearningEdgeDetection.f1Score') }}: <strong>{{ batchResults.metrics.ods.f1 }}</strong></p>
              </div>
              <div class="metric-item">
                <h4>{{ t('deepLearningEdgeDetection.ois') }}</h4>
                <p>{{ t('deepLearningEdgeDetection.precision') }}: <strong>{{ batchResults.metrics.ois.precision }}</strong></p>
                <p>{{ t('deepLearningEdgeDetection.recall') }}: <strong>{{ batchResults.metrics.ois.recall }}</strong></p>
                <p>{{ t('deepLearningEdgeDetection.f1Score') }}: <strong>{{ batchResults.metrics.ois.f1 }}</strong></p>
              </div>
            </div>

            <!-- PR 曲线（仅在有metrics时显示，即BSDS500快速测试） -->
            <div v-if="batchResults.metrics && batchResults.pr_curve" class="pr-curve-container">
              <h4>{{ t('deepLearningEdgeDetection.prCurve') }}</h4>
              <img :src="batchResults.pr_curve" :alt="t('deepLearningEdgeDetection.prCurve')" class="pr-curve-image" />
            </div>

            <!-- 处理结果列表 -->
            <div class="processed-images-list">
              <h4>{{ t('deepLearningEdgeDetection.processedResults').replace('{count}', batchResults.processed_count) }}</h4>
              <div class="images-grid">
                <div v-for="img in batchResults.processed_images" :key="img.index" class="processed-image-item">
                  <img :src="img.edge_image" :alt="`Processed image ${img.index}`" class="processed-image" />
                  <p class="image-info">{{ img.filename }}</p>
                  <!-- 单图指标（仅在有per_image_results时显示，即BSDS500快速测试） -->
                  <div v-if="batchResults.metrics && getImageMetrics(img.index)" class="image-metrics">
                    <p>{{ t('deepLearningEdgeDetection.precision') }}: {{ getImageMetrics(img.index).precision }}</p>
                    <p>{{ t('deepLearningEdgeDetection.recall') }}: {{ getImageMetrics(img.index).recall }}</p>
                    <p>{{ t('deepLearningEdgeDetection.f1Score') }}: {{ getImageMetrics(img.index).f1 }}</p>
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
import { getCurrentLanguage, t as translate } from '../i18n'

export default {
  name: 'DeepLearningEdgeDetection',
  inject: ['currentLanguage'],
  data() {
    return {
      algorithm: 'HED',
      algoOptions: ['HED', 'PiDiNet', 'RCF'],
      inputMode: 'image', // 'image' or 'video'
      selectedImage: null,
      selectedVideo: null,
      edgeImage: null,
      metrics: null,
      processing: false,
      videoProcessing: false,
      videoFrameInterval: null,
      lastProcessedFrame: 0,
      hasVideoEdgeResult: false,
      showBatchPanel: false,
      batchFiles: [],
      batchProcessing: false,
      batchResults: null,
      bsds500Processing: false,
      useTolerance: true,
      maxDist: 0.0075,
      useThinning: true
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
    },
    handleBatchFileSelect(event) {
      const files = Array.from(event.target.files || [])
      this.batchFiles = files
    },
    async startBatchProcess() {
      if (this.batchFiles.length === 0) {
        alert(this.t('deepLearningEdgeDetection.selectMultipleImages'))
        return
      }
      
      this.batchProcessing = true
      this.batchResults = null
      
      try {
        // 读取所有图片文件并转换为base64
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
          blur: 7, // 深度学习算法不需要这些参数，但为了兼容性保留
          sobel_ksize: 3,
          canny_low: 100,
          canny_high: 220,
          use_tolerance: this.useTolerance,
          max_dist: this.maxDist,
          use_thinning: this.useThinning
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
        return null
      }
      const result = this.batchResults.per_image_results.find(r => r.index === index)
      if (!result) return null
      return {
        precision: result.best_precision || result.precision || 0,
        recall: result.best_recall || result.recall || 0,
        f1: result.best_f1 || result.f1 || 0
      }
    },
    async startBSDS500QuickTest() {
      this.bsds500Processing = true
      this.batchResults = null
      
      try {
        // 调用BSDS500快速测试API
        const response = await axios.post('/api/bsds500-quick-test', {
          algorithm: this.algorithm,
          blur: 7, // 深度学习算法不需要这些参数，但为了兼容性保留
          sobel_ksize: 3,
          canny_low: 100,
          canny_high: 220,
          subset: 'val', // 默认使用验证集
          use_tolerance: this.useTolerance,
          max_dist: this.maxDist,
          use_thinning: this.useThinning
        })
        
        if (response.data.success) {
          this.batchResults = response.data
          // 调试：检查返回的数据
          console.log('BSDS500快速测试返回数据:', {
            hasMetrics: !!this.batchResults.metrics,
            hasPrCurve: !!this.batchResults.pr_curve,
            hasPerImageResults: !!this.batchResults.per_image_results,
            processedCount: this.batchResults.processed_count,
            processedImagesCount: this.batchResults.processed_images?.length
          })
        } else {
          alert('BSDS500快速测试失败：' + (response.data.error || '未知错误'))
        }
      } catch (error) {
        console.error('BSDS500快速测试错误:', error)
        alert('BSDS500快速测试失败：' + (error.response?.data?.error || error.message))
      } finally {
        this.bsds500Processing = false
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

.batch-section {
  margin-top: 30px;
  background: var(--container-color);
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

.evaluation-params-panel {
  margin: 15px 0;
  padding: 15px;
  background: rgba(0, 0, 0, 0.2);
  border: 1px solid rgba(212, 175, 55, 0.2);
  border-radius: 0;
}

.evaluation-params-header {
  margin-bottom: 15px;
}

.evaluation-params-header h4 {
  color: var(--text-color);
  margin: 0;
  font-size: 1.1em;
}

.evaluation-params-grid {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.evaluation-param-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.evaluation-param-item label {
  color: var(--text-color);
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.evaluation-param-item input[type="checkbox"] {
  width: 18px;
  height: 18px;
  cursor: pointer;
}

.evaluation-param-item input[type="range"] {
  width: 100%;
  margin: 5px 0;
}

.param-desc {
  color: var(--text-color);
  font-size: 0.85em;
  opacity: 0.7;
  margin: 0;
  font-style: italic;
}

.param-value {
  color: var(--text-color);
  font-weight: bold;
  margin-left: 10px;
}

.batch-panel {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.batch-buttons {
  display: flex;
  gap: 15px;
  flex-wrap: wrap;
}

.bsds500-btn {
  background: rgba(212, 175, 55, 0.3);
  border-color: rgba(212, 175, 55, 0.7);
}

.bsds500-btn:hover:not(:disabled) {
  background: rgba(212, 175, 55, 0.4);
  border-color: var(--text-color);
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
}

.processed-image {
  width: 100%;
  height: auto;
  display: block;
  margin-bottom: 10px;
}

.image-info {
  color: var(--text-color);
  font-size: 12px;
  margin: 5px 0;
  opacity: 0.8;
}

.image-metrics {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid rgba(212, 175, 55, 0.2);
}

.image-metrics p {
  color: var(--text-color);
  font-size: 11px;
  margin: 3px 0;
  opacity: 0.9;
}

@media (max-width: 768px) {
  .video-section {
    grid-template-columns: 1fr;
  }
  .metrics-grid {
    grid-template-columns: 1fr;
  }
  .images-grid {
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  }
}
</style>

