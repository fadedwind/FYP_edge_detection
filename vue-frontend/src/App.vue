<template>
  <div class="app-container">
    <Sidebar 
      @toggle-theme-settings="toggleThemeSettings"
      @sidebar-toggle="handleSidebarToggle"
      @language-changed="setLanguage"
      :collapsed="sidebarCollapsed"
    />
    <div class="main-wrapper" :class="{ 'sidebar-collapsed': sidebarCollapsed }">
      <ThemeSettingsPanel 
        v-if="showThemeSettings" 
        @close="showThemeSettings = false"
      />
      <router-view />
    </div>
  </div>
</template>

<script>
import Sidebar from './components/Sidebar.vue'
import ThemeSettingsPanel from './components/ThemeSettingsPanel.vue'
import { getCurrentLanguage, setLanguage } from './i18n'

export default {
  name: 'App',
  components: {
    Sidebar,
    ThemeSettingsPanel
  },
  data() {
    return {
      showThemeSettings: false,
      sidebarCollapsed: false,
      currentLanguage: getCurrentLanguage()
    }
  },
  provide() {
    const self = this
    return {
      currentLanguage() {
        return self.currentLanguage
      },
      setLanguage(lang) {
        self.setLanguage(lang)
      }
    }
  },
  methods: {
    toggleThemeSettings() {
      this.showThemeSettings = !this.showThemeSettings
    },
    handleSidebarToggle(collapsed) {
      this.sidebarCollapsed = collapsed
    },
    setLanguage(lang) {
      this.currentLanguage = lang
      setLanguage(lang)
      // 触发全局更新
      this.$forceUpdate()
    }
  },
  mounted() {
    const saved = localStorage.getItem('sidebarCollapsed')
    if (saved !== null) {
      this.sidebarCollapsed = saved === 'true'
    }
    this.currentLanguage = getCurrentLanguage()
  }
}
</script>

<style>
@import './styles/common.css';

.app-container {
  display: flex;
  min-height: 100vh;
  background: #0a0a0a;
}

.main-wrapper {
  flex: 1;
  margin-left: 220px;
  transition: margin-left 0.3s ease;
  background: #0a0a0a;
  min-height: 100vh;
}

.main-wrapper.sidebar-collapsed {
  margin-left: 60px;
}

@media (max-width: 768px) {
  .main-wrapper {
    margin-left: 0;
  }
  
  .main-wrapper.sidebar-collapsed {
    margin-left: 0;
  }
}
</style>

