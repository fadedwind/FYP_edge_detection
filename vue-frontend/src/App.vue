<template>
  <div class="app-container">
    <Sidebar 
      @toggle-theme-settings="toggleThemeSettings"
      @sidebar-toggle="handleSidebarToggle"
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

export default {
  name: 'App',
  components: {
    Sidebar,
    ThemeSettingsPanel
  },
  data() {
    return {
      showThemeSettings: false,
      sidebarCollapsed: false
    }
  },
  methods: {
    toggleThemeSettings() {
      this.showThemeSettings = !this.showThemeSettings
    },
    handleSidebarToggle(collapsed) {
      this.sidebarCollapsed = collapsed
    }
  },
  mounted() {
    const saved = localStorage.getItem('sidebarCollapsed')
    if (saved !== null) {
      this.sidebarCollapsed = saved === 'true'
    }
  }
}
</script>

<style>
.app-container {
  display: flex;
  min-height: 100vh;
}

.main-wrapper {
  flex: 1;
  margin-left: 220px;
  transition: margin-left 0.3s ease;
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

