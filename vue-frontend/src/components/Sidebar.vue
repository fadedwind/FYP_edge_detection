<template>
  <div class="sidebar" :class="{ collapsed: isCollapsed }">
    <div class="sidebar-header">
      <button class="toggle-btn" @click="toggleSidebar">
        {{ isCollapsed ? '☰' : '✕' }}
      </button>
      <h3 v-if="!isCollapsed" class="sidebar-title">导航</h3>
    </div>
    
    <nav class="sidebar-nav">
      <button 
        class="nav-item" 
        :class="{ active: $route.path === '/' }"
        @click="navigate('/')"
        :title="isCollapsed ? '边缘检测' : ''"
      >
        <span class="icon">🖼️</span>
        <span v-if="!isCollapsed" class="label">边缘检测</span>
      </button>
      
      <button 
        class="nav-item" 
        :class="{ active: $route.path === '/vehicle' }"
        @click="navigate('/vehicle')"
        :title="isCollapsed ? '汽车识别' : ''"
      >
        <span class="icon">🚗</span>
        <span v-if="!isCollapsed" class="label">汽车识别</span>
      </button>
      
      <button 
        class="nav-item" 
        :class="{ active: $route.path === '/docs' }"
        @click="navigate('/docs')"
        :title="isCollapsed ? '算法原理' : ''"
      >
        <span class="icon">📚</span>
        <span v-if="!isCollapsed" class="label">算法原理</span>
      </button>
      
      <div class="divider" v-if="!isCollapsed"></div>
      
      <button 
        class="nav-item settings-btn" 
        @click="toggleThemeSettings"
        :title="isCollapsed ? '外观设置' : ''"
      >
        <span class="icon">🎨</span>
        <span v-if="!isCollapsed" class="label">外观设置</span>
      </button>
    </nav>
  </div>
</template>

<script>
export default {
  name: 'Sidebar',
  props: {
    collapsed: {
      type: Boolean,
      default: false
    }
  },
  data() {
    return {
      isCollapsed: false
    }
  },
  watch: {
    collapsed(newVal) {
      this.isCollapsed = newVal
    }
  },
  methods: {
    toggleSidebar() {
      this.isCollapsed = !this.isCollapsed
      localStorage.setItem('sidebarCollapsed', this.isCollapsed)
      this.$emit('sidebar-toggle', this.isCollapsed)
    },
    navigate(path) {
      if (this.$route.path !== path) {
        this.$router.push(path)
      }
    },
    toggleThemeSettings() {
      this.$emit('toggle-theme-settings')
    }
  },
  mounted() {
    const saved = localStorage.getItem('sidebarCollapsed')
    if (saved !== null) {
      this.isCollapsed = saved === 'true'
    } else {
      this.isCollapsed = this.collapsed
    }
    this.$emit('sidebar-toggle', this.isCollapsed)
  }
}
</script>

<style scoped>
.sidebar {
  position: fixed;
  left: 0;
  top: 0;
  height: 100vh;
  width: 220px;
  background: var(--container-color, #1a1a1a);
  border-right: 1px solid rgba(212, 175, 55, 0.2);
  transition: width 0.3s ease, transform 0.3s ease;
  z-index: 1000;
  display: flex;
  flex-direction: column;
  box-shadow: 2px 0 10px rgba(0, 0, 0, 0.3);
}

.sidebar.collapsed {
  width: 60px;
}

.sidebar-header {
  padding: 20px;
  border-bottom: 1px solid rgba(212, 175, 55, 0.2);
  display: flex;
  align-items: center;
  gap: 10px;
}

.toggle-btn {
  background: transparent;
  border: 1px solid rgba(212, 175, 55, 0.3);
  color: var(--text-color, #d4af37);
  width: 32px;
  height: 32px;
  border-radius: 0;
  cursor: pointer;
  font-size: 18px;
  transition: all 0.2s;
  flex-shrink: 0;
}

.toggle-btn:hover {
  border-color: var(--text-color, #d4af37);
  background: rgba(212, 175, 55, 0.1);
}

.sidebar-title {
  color: var(--text-color, #d4af37);
  margin: 0;
  font-size: 1.1em;
  font-weight: 600;
  white-space: nowrap;
  transition: opacity 0.3s ease;
}

.sidebar.collapsed .sidebar-title {
  opacity: 0;
  width: 0;
  overflow: hidden;
}

.sidebar-nav {
  flex: 1;
  padding: 10px;
  overflow-y: auto;
}

.nav-item {
  width: 100%;
  padding: 12px 16px;
  margin-bottom: 8px;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 0;
  color: var(--text-color, #d4af37);
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 12px;
  transition: all 0.2s;
  text-align: left;
  font-size: 14px;
  font-weight: 500;
}

.nav-item:hover {
  background: rgba(212, 175, 55, 0.1);
  border-color: rgba(212, 175, 55, 0.3);
}

.nav-item.active {
  background: rgba(212, 175, 55, 0.15);
  border-color: var(--text-color, #d4af37);
  box-shadow: 0 0 8px rgba(212, 175, 55, 0.2);
}

.icon {
  font-size: 18px;
  flex-shrink: 0;
  width: 24px;
  text-align: center;
}

.label {
  white-space: nowrap;
  transition: opacity 0.3s ease;
}

.sidebar.collapsed .label {
  opacity: 0;
  width: 0;
  overflow: hidden;
}

.divider {
  height: 1px;
  background: rgba(212, 175, 55, 0.2);
  margin: 12px 0;
}

.settings-btn {
  margin-top: auto;
}

@media (max-width: 768px) {
  .sidebar {
    transform: translateX(-100%);
  }
  
  .sidebar.collapsed {
    transform: translateX(0);
  }
}
</style>

