<template>
  <div class="page-container">
    <header class="header">
      <h1>✂️ {{ t('smartRemoval.title') }}</h1>
      <p class="subtitle">{{ t('smartRemoval.subtitle') }}</p>
    </header>

    <main class="main-content">
      <!-- 控制面板 -->
      <div class="control-panel">
        <div class="control-group">
          <label for="method">{{ t('smartRemoval.method') }}</label>
          <select id="method" v-model="selectedMethod" class="select-input">
            <option value="auto">🤖 {{ t('smartRemoval.auto') }}</option>
            <option value="rembg">⭐ rembg {{ t('smartRemoval.recommended') }}</option>
            <option value="edge">🔍 {{ t('smartRemoval.edgeAssisted') }}</option>
            <option value="grabcut">✂️ GrabCut</option>
          </select>
        </div>

        <div class="control-group" v-if="selectedMethod === 'edge'">
          <label for="threshold">{{ t('smartRemoval.edgeThreshold') }}</label>
          <input
            type="range"
            id="threshold"
            v-model.number="edgeThreshold"
            min="0"
            max="255"
            step="1"
            class="slider"
          />
          <span class="value-display">{{ edgeThreshold }}</span>
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
          <label for="file-input" class="file-label">
            {{ t('smartRemoval.selectImage') }}
          </label>
        </div>

        <button
          @click="startRemoval"
          :disabled="!selectedImage || processing"
          class="detect-btn"
        >
          {{ processing ? t('smartRemoval.processing') : t('smartRemoval.startRemoval') }}
        </button>
      </div>

      <!-- 方法说明 -->
      <div class="method-info">
        <div v-if="selectedMethod === 'auto'" class="info-card">
          <h4>🤖 {{ t('smartRemoval.auto') }}</h4>
          <p>{{ t('smartRemoval.autoDescription') }}</p>
        </div>
        <div v-else-if="selectedMethod === 'rembg'" class="info-card">
          <h4>⭐ rembg ({{ t('smartRemoval.recommended') }})</h4>
          <p>{{ t('smartRemoval.rembgDescription') }}</p>
          <ul>
            <li>✅ {{ t('smartRemoval.rembgAdvantage1') }}</li>
            <li>✅ {{ t('smartRemoval.rembgAdvantage2') }}</li>
            <li>✅ {{ t('smartRemoval.rembgAdvantage3') }}</li>
          </ul>
        </div>
        <div v-else-if="selectedMethod === 'edge'" class="info-card">
          <h4>🔍 {{ t('smartRemoval.edgeAssisted') }}</h4>
          <p>{{ t('smartRemoval.edgeDescription') }}</p>
          <ul>
            <li>🔗 {{ t('smartRemoval.edgeAdvantage1') }}</li>
            <li>⚡ {{ t('smartRemoval.edgeAdvantage2') }}</li>
          </ul>
        </div>
        <div v-else-if="selectedMethod === 'grabcut'" class="info-card">
          <h4>✂️ GrabCut</h4>
          <p>{{ t('smartRemoval.grabcutDescription') }}</p>
          <ul>
            <li>⚙️ {{ t('smartRemoval.grabcutAdvantage1') }}</li>
            <li>⚠️ {{ t('smartRemoval.grabcutAdvantage2') }}</li>
          </ul>
        </div>
      </div>

      <!-- 结果展示 -->
      <div class="result-section" v-if="resultImage || maskImage">
        <div class="image-box">
          <h3>{{ t('smartRemoval.originalImage') }}</h3>
          <div class="image-container">
            <img v-if="selectedImage" :src="selectedImage" class="result-image" />
          </div>
        </div>

        <div class="image-box">
          <h3>{{ t('smartRemoval.result') }}</h3>
          <div class="image-container checkerboard">
            <img v-if="resultImage" :src="resultImage" class="result-image" />
          </div>
        </div>

        <div class="image-box" v-if="maskImage">
          <h3>{{ t('smartRemoval.mask') }}</h3>
          <div class="image-container">
            <img v-if="maskImage" :src="maskImage" class="result-image" />
          </div>
        </div>
      </div>

      <!-- 下载按钮 -->
      <div class="download-section" v-if="resultImage">
        <button @click="downloadResult" class="download-btn">
          💾 {{ t('smartRemoval.download') }}
        </button>
      </div>

      <!-- 使用提示 -->
      <div class="tips">
        <h4>{{ t('smartRemoval.tips') }}</h4>
        <ul>
          <li>{{ t('smartRemoval.tip1') }}</li>
          <li>{{ t('smartRemoval.tip2') }}</li>
          <li>{{ t('smartRemoval.tip3') }}</li>
          <li>{{ t('smartRemoval.tip4') }}</li>
        </ul>
      </div>
    </main>
  </div>
</template>

<script>
import axios from 'axios'
import { getCurrentLanguage, t as translate } from '../i18n'

export default {
  name: 'SmartRemoval',
  inject: ['currentLanguage'],
  data() {
    return {
      selectedMethod: 'auto',
      edgeThreshold: 127,
      selectedImage: null,
      resultImage: null,
      maskImage: null,
      processing: false,
      foregroundRatio: 0
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
        this.resultImage = null
        this.maskImage = null
      }
      reader.readAsDataURL(file)
    },
    async startRemoval() {
      if (!this.selectedImage) {
        alert('请先选择图片！')
        return
      }

      this.processing = true
      try {
        const response = await axios.post('/api/remove-background', {
          image: this.selectedImage,
          method: this.selectedMethod,
          edge_threshold: this.edgeThreshold
        })

        if (response.data.success) {
          this.resultImage = response.data.images.result
          this.maskImage = response.data.images.mask || null
          this.foregroundRatio = response.data.foreground_ratio || 0
        } else {
          alert('抠图失败：' + (response.data.error || '未知错误'))
        }
      } catch (error) {
        console.error('抠图错误:', error)
        alert('抠图失败：' + (error.response?.data?.error || error.message))
      } finally {
        this.processing = false
      }
    },
    downloadResult() {
      if (!this.resultImage) return

      const link = document.createElement('a')
      link.href = this.resultImage
      link.download = `removed_background_${Date.now()}.png`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
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
  }
}
</script>

<style scoped>
@import '../styles/common.css';

.method-info {
  margin: 1.5rem 0;
}

.info-card {
  background: rgba(212, 175, 55, 0.05);
  border: 1px solid rgba(212, 175, 55, 0.2);
  border-radius: 8px;
  padding: 1rem 1.5rem;
}

.info-card h4 {
  color: #D4AF37;
  margin-bottom: 0.5rem;
  font-size: 1.1rem;
}

.info-card p {
  color: #b0b0b0;
  margin-bottom: 0.75rem;
  line-height: 1.6;
}

.info-card ul {
  list-style: none;
  padding-left: 0;
}

.info-card li {
  color: #a0a0a0;
  padding: 0.25rem 0;
  font-size: 0.95rem;
}

.result-section {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}

.image-box {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(212, 175, 55, 0.2);
  border-radius: 8px;
  padding: 1rem;
}

.image-box h3 {
  color: #D4AF37;
  margin-bottom: 1rem;
  font-size: 1rem;
}

.image-container {
  background: rgba(0, 0, 0, 0.3);
  border-radius: 6px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 250px;
}

.image-container.checkerboard {
  background-image:
    linear-gradient(45deg, #2a2a2a 25%, transparent 25%),
    linear-gradient(-45deg, #2a2a2a 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #2a2a2a 75%),
    linear-gradient(-45deg, transparent 75%, #2a2a2a 75%);
  background-size: 20px 20px;
  background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
}

.result-image {
  max-width: 100%;
  max-height: 400px;
  object-fit: contain;
}

.download-section {
  display: flex;
  justify-content: center;
  margin: 2rem 0;
}

.download-btn {
  padding: 0.75rem 2rem;
  background: linear-gradient(135deg, #D4AF37, #B8860B);
  border: none;
  border-radius: 8px;
  color: #1a1a1a;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.download-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(212, 175, 55, 0.4);
}

.tips {
  background: rgba(212, 175, 55, 0.05);
  border-left: 4px solid #D4AF37;
  border-radius: 6px;
  padding: 1rem 1.5rem;
  margin: 2rem 0;
}

.tips h4 {
  color: #D4AF37;
  margin-bottom: 0.75rem;
}

.tips ul {
  list-style: none;
  padding-left: 0;
}

.tips li {
  color: #b0b0b0;
  padding: 0.4rem 0;
  padding-left: 1.5rem;
  position: relative;
}

.tips li::before {
  content: '💡';
  position: absolute;
  left: 0;
  top: 0.4rem;
  font-size: 0.9rem;
}
</style>
