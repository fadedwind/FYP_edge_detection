<template>
  <div class="documentation-page">
    <header class="header">
      <h1>{{ t('documentation.title') }}</h1>
      <p class="subtitle">{{ t('documentation.subtitle') }}</p>
    </header>
    
    <main class="main-content">
      <div class="doc-links">
        <a
          v-for="doc in documents"
          :key="doc.id"
          :href="doc.url"
          target="_blank"
          rel="noopener noreferrer"
          class="doc-link"
        >
          <span class="doc-icon">{{ doc.icon }}</span>
          <div class="doc-info">
            <h3>{{ t(doc.titleKey) }}</h3>
            <p v-if="doc.description">{{ t(doc.descriptionKey) }}</p>
          </div>
          <span class="doc-arrow">→</span>
        </a>
      </div>
    </main>
  </div>
</template>

<script>
import { getCurrentLanguage, t as translate } from '../i18n'

export default {
  name: 'Documentation',
  inject: ['currentLanguage'],
  data() {
    return {
      documents: [
        { 
          id: 'edge_detection', 
          titleKey: 'documentation.edgeDetectionDocs', 
          descriptionKey: 'documentation.edgeDetectionDesc',
          icon: '📖',
          url: 'http://127.0.0.1:5500/vue-frontend/docs/edge_detection_docs.html'
        },
        { 
          id: 'metrics', 
          titleKey: 'documentation.metricsDocs',
          descriptionKey: 'documentation.metricsDesc',
          icon: '📊',
          url: 'http://127.0.0.1:5500/vue-frontend/docs/metrics.html'
        }
      ]
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
    }
  }
}
</script>

<style scoped>
@import '../styles/common.css';

.documentation-page {
  min-height: calc(100vh - 40px);
  margin: 20px;
}

.header {
  margin-bottom: 30px;
}

.header h1 {
  color: var(--text-color, #d4af37);
  font-size: 2em;
  margin-bottom: 10px;
}

.subtitle {
  color: var(--text-color, #d4af37);
  opacity: 0.8;
  font-size: 1.1em;
}

.main-content {
  max-width: 800px;
  margin: 0 auto;
}

.doc-links {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.doc-link {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 24px;
  background: var(--container-color, #1a1a1a);
  border: 1px solid rgba(212, 175, 55, 0.2);
  border-radius: 0;
  text-decoration: none;
  color: var(--text-color, #d4af37);
  transition: all 0.3s ease;
  cursor: pointer;
}

.doc-link:hover {
  background: rgba(212, 175, 55, 0.1);
  border-color: rgba(212, 175, 55, 0.5);
  transform: translateX(5px);
}

.doc-icon {
  font-size: 2.5em;
  flex-shrink: 0;
}

.doc-info {
  flex: 1;
}

.doc-info h3 {
  color: var(--text-color, #d4af37);
  font-size: 1.3em;
  margin: 0 0 8px 0;
  font-weight: 600;
}

.doc-info p {
  color: var(--text-color, #d4af37);
  opacity: 0.7;
  margin: 0;
  font-size: 0.95em;
}

.doc-arrow {
  font-size: 1.5em;
  color: var(--text-color, #d4af37);
  opacity: 0.6;
  transition: all 0.3s ease;
  flex-shrink: 0;
}

.doc-link:hover .doc-arrow {
  opacity: 1;
  transform: translateX(5px);
}

@media (max-width: 768px) {
  .documentation-page {
    margin: 10px;
  }
  
  .header h1 {
    font-size: 1.5em;
  }
  
  .doc-link {
    padding: 20px;
    gap: 15px;
  }
  
  .doc-icon {
    font-size: 2em;
  }
  
  .doc-info h3 {
    font-size: 1.1em;
  }
}
</style>
