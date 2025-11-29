<template>
  <div class="theme-settings-overlay" @click.self="close">
    <div class="theme-settings-panel">
      <div class="theme-header">
        <h3>外观设置</h3>
        <button class="close-btn" @click="close">×</button>
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
  </div>
</template>

<script>
export default {
  name: 'ThemeSettingsPanel',
  data() {
    return {
      customBgColor: '#0a0a0a',
      customContainerColor: '#1a1a1a',
      customTextColor: '#d4af37'
    }
  },
  mounted() {
    const savedBg = localStorage.getItem('customBgColor')
    const savedContainer = localStorage.getItem('customContainerColor')
    const savedText = localStorage.getItem('customTextColor')
    if (savedBg) this.customBgColor = savedBg
    if (savedContainer) this.customContainerColor = savedContainer
    if (savedText) this.customTextColor = savedText
    this.applyTheme()
  },
  methods: {
    close() {
      this.$emit('close')
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
    }
  }
}
</script>

<style scoped>
@import '../styles/common.css';

.theme-settings-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
  backdrop-filter: blur(4px);
}

.theme-settings-panel {
  background: var(--container-color, #1a1a1a);
  border: 1px solid rgba(212, 175, 55, 0.3);
  border-radius: 0;
  padding: 30px;
  max-width: 500px;
  width: 90%;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
}

.theme-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  border-bottom: 1px solid rgba(212, 175, 55, 0.2);
  padding-bottom: 15px;
}

.theme-header h3 {
  color: var(--text-color, #d4af37);
  margin: 0;
  font-size: 1.3em;
  transition: color 0.3s ease;
}

.close-btn {
  background: transparent;
  border: 1px solid rgba(212, 175, 55, 0.3);
  color: var(--text-color, #d4af37);
  width: 32px;
  height: 32px;
  border-radius: 0;
  cursor: pointer;
  font-size: 22px;
  line-height: 1;
  transition: all 0.2s;
}

.close-btn:hover {
  border-color: var(--text-color, #d4af37);
  background: rgba(212, 175, 55, 0.1);
}

.theme-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.theme-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 15px;
}

.theme-item label {
  color: var(--text-color, #d4af37);
  font-weight: 600;
  transition: color 0.3s ease;
  flex: 1;
}

.color-picker {
  width: 80px;
  height: 40px;
  border: 1px solid rgba(212, 175, 55, 0.3);
  border-radius: 0;
  cursor: pointer;
  background: #1a1a1a;
}

.nav-btn {
  padding: 10px 20px;
  border-radius: 0;
  border: 1px solid var(--text-color, #d4af37);
  cursor: pointer;
  font-size: 14px;
  font-weight: 600;
  background: var(--container-color, #1a1a1a);
  color: var(--text-color, #d4af37);
  transition: all 0.2s;
  margin-top: 10px;
}

.nav-btn:hover {
  background: #2a2a2a;
  border-color: var(--text-color, #d4af37);
  color: var(--text-color, #d4af37);
  box-shadow: 0 0 10px rgba(212, 175, 55, 0.3);
}
</style>

