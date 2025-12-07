<template>
  <div class="page-container">
    <header class="header">
      <h1>📝 {{ t('changelog.title') }}</h1>
      <p class="subtitle">{{ t('changelog.subtitle') }}</p>
    </header>

    <main class="main-content">
      <div class="changelog-controls">
        <button 
          @click="loadCommits" 
          :disabled="loading"
          class="refresh-btn"
        >
          {{ loading ? t('changelog.loading') : t('changelog.refresh') }}
        </button>
        <input
          v-model="searchQuery"
          :placeholder="t('changelog.searchPlaceholder')"
          class="search-input"
        />
      </div>

      <div v-if="error" class="error-message">
        {{ error }}
      </div>

      <div v-if="loading && commits.length === 0" class="loading-message">
        {{ t('changelog.loading') }}...
      </div>

      <div v-else class="commits-list">
        <div 
          v-for="commit in filteredCommits" 
          :key="commit.hash"
          class="commit-item"
        >
          <div class="commit-header">
            <div class="commit-hash">{{ commit.hash.substring(0, 7) }}</div>
            <div class="commit-date">{{ formatDate(commit.date) }}</div>
          </div>
          <div class="commit-author">
            <span class="author-icon">👤</span>
            <span>{{ commit.author }}</span>
          </div>
          <div class="commit-message">{{ commit.message }}</div>
          <div v-if="commit.files && commit.files.length > 0" class="commit-files">
            <div class="files-header">{{ t('changelog.changedFiles') }}:</div>
            <div class="files-list">
              <span 
                v-for="(file, idx) in commit.files" 
                :key="idx"
                class="file-item"
              >
                {{ file }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div v-if="!loading && commits.length === 0" class="empty-message">
        {{ t('changelog.noCommits') }}
      </div>
    </main>
  </div>
</template>

<script>
import axios from 'axios'
import { getCurrentLanguage, t as translate } from '../i18n'

export default {
  name: 'Changelog',
  inject: ['currentLanguage'],
  data() {
    return {
      commits: [],
      loading: false,
      error: null,
      searchQuery: ''
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
    },
    filteredCommits() {
      if (!this.searchQuery) {
        return this.commits
      }
      const query = this.searchQuery.toLowerCase()
      return this.commits.filter(commit => 
        commit.message.toLowerCase().includes(query) ||
        commit.author.toLowerCase().includes(query) ||
        commit.hash.toLowerCase().includes(query) ||
        (commit.files && commit.files.some(file => file.toLowerCase().includes(query)))
      )
    }
  },
  mounted() {
    this.loadCommits()
  },
  methods: {
    t(key) {
      return translate(key, this.currentLang)
    },
    async loadCommits() {
      this.loading = true
      this.error = null
      try {
        const response = await axios.get('/api/git-log')
        if (response.data.success) {
          this.commits = response.data.commits
        } else {
          this.error = response.data.error || this.t('changelog.loadError')
        }
      } catch (error) {
        console.error('加载git日志失败:', error)
        this.error = this.t('changelog.loadError') + ': ' + (error.response?.data?.error || error.message)
      } finally {
        this.loading = false
      }
    },
    formatDate(dateString) {
      if (!dateString) return ''
      const date = new Date(dateString)
      const now = new Date()
      const diffMs = now - date
      const diffMins = Math.floor(diffMs / 60000)
      const diffHours = Math.floor(diffMs / 3600000)
      const diffDays = Math.floor(diffMs / 86400000)
      
      if (diffMins < 1) return this.t('changelog.justNow')
      if (diffMins < 60) return `${diffMins}${this.t('changelog.minutesAgo')}`
      if (diffHours < 24) return `${diffHours}${this.t('changelog.hoursAgo')}`
      if (diffDays < 7) return `${diffDays}${this.t('changelog.daysAgo')}`
      
      return date.toLocaleDateString(this.currentLang === 'zh' ? 'zh-CN' : 'en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
    }
  }
}
</script>

<style scoped>
@import '../styles/common.css';

.changelog-controls {
  display: flex;
  gap: 15px;
  margin-bottom: 20px;
  align-items: center;
}

.refresh-btn {
  padding: 10px 20px;
  background: rgba(212, 175, 55, 0.2);
  border: 1px solid rgba(212, 175, 55, 0.5);
  color: var(--text-color);
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 14px;
  white-space: nowrap;
}

.refresh-btn:hover:not(:disabled) {
  background: rgba(212, 175, 55, 0.3);
  border-color: var(--text-color);
}

.refresh-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.search-input {
  flex: 1;
  padding: 10px 15px;
  background: var(--container-color);
  border: 1px solid rgba(212, 175, 55, 0.3);
  color: var(--text-color);
  font-size: 14px;
  transition: border-color 0.3s ease;
}

.search-input:focus {
  outline: none;
  border-color: var(--text-color);
}

.error-message {
  padding: 15px;
  background: rgba(255, 0, 0, 0.1);
  border: 1px solid rgba(255, 0, 0, 0.3);
  color: #ff6b6b;
  border-radius: 0;
  margin-bottom: 20px;
}

.loading-message,
.empty-message {
  text-align: center;
  padding: 40px;
  color: var(--text-color);
  opacity: 0.7;
}

.commits-list {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.commit-item {
  background: var(--container-color);
  border: 1px solid rgba(212, 175, 55, 0.2);
  border-radius: 0;
  padding: 20px;
  transition: border-color 0.3s ease, background-color 0.3s ease;
}

.commit-item:hover {
  border-color: rgba(212, 175, 55, 0.4);
  background: rgba(212, 175, 55, 0.05);
}

.commit-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.commit-hash {
  font-family: 'Courier New', monospace;
  color: var(--text-color);
  font-weight: 600;
  font-size: 14px;
  background: rgba(212, 175, 55, 0.2);
  padding: 4px 8px;
  border-radius: 0;
}

.commit-date {
  color: var(--text-color);
  opacity: 0.7;
  font-size: 13px;
}

.commit-author {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
  color: var(--text-color);
  opacity: 0.9;
  font-size: 14px;
}

.author-icon {
  font-size: 16px;
}

.commit-message {
  color: var(--text-color);
  font-size: 15px;
  line-height: 1.6;
  margin-bottom: 12px;
  white-space: pre-wrap;
  word-break: break-word;
}

.commit-files {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid rgba(212, 175, 55, 0.1);
}

.files-header {
  color: var(--text-color);
  opacity: 0.8;
  font-size: 13px;
  margin-bottom: 8px;
  font-weight: 600;
}

.files-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.file-item {
  background: rgba(212, 175, 55, 0.15);
  border: 1px solid rgba(212, 175, 55, 0.3);
  color: var(--text-color);
  padding: 4px 10px;
  font-size: 12px;
  font-family: 'Courier New', monospace;
  border-radius: 0;
  opacity: 0.9;
}

@media (max-width: 768px) {
  .changelog-controls {
    flex-direction: column;
    align-items: stretch;
  }
  
  .commit-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
}
</style>

