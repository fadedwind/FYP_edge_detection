<template>
  <div class="documentation-page">
    <div class="doc-sidebar">
      <h3>文档目录</h3>
      <ul class="doc-menu">
        <li 
          v-for="doc in documents" 
          :key="doc.id"
          :class="{ active: currentDoc === doc.id }"
          @click="loadDocument(doc.id)"
        >
          {{ doc.title }}
        </li>
      </ul>
    </div>
    
    <div class="doc-content">
      <div v-if="loading" class="loading">加载中...</div>
      <div v-else-if="error" class="error">{{ error }}</div>
      <div v-else v-html="docContent" class="doc-html"></div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'Documentation',
  data() {
    return {
      documents: [
        { id: 'edge_detection', title: '边缘检测算法详解', file: 'edge_detection_docs.html' },
        { id: 'metrics', title: '评估指标与 ODS/OIS', file: 'metrics.html' }
      ],
      currentDoc: 'edge_detection',
      docContent: '',
      loading: false,
      error: null
    }
  },
  mounted() {
    this.loadDocument('edge_detection')
  },
  methods: {
    async loadDocument(docId) {
      const doc = this.documents.find(d => d.id === docId)
      if (!doc) return
      
      this.currentDoc = docId
      this.loading = true
      this.error = null
      
      try {
        const response = await axios.get(`/docs/${doc.file}`, {
          responseType: 'text'
        })
        
        const parser = new DOMParser()
        const docHtml = parser.parseFromString(response.data, 'text/html')
        let bodyContent = docHtml.body.innerHTML
        const headContent = docHtml.head.innerHTML
        const scripts = Array.from(docHtml.querySelectorAll('script'))
        
        // 移除 HED 文档中的 max-width 限制和其他可能影响布局的样式
        bodyContent = bodyContent.replace(/max-width:\s*\d+px/gi, 'max-width: 100%')
        bodyContent = bodyContent.replace(/margin:\s*\d+px\s+auto/gi, 'margin: 0')
        
        // 提取并保留所有 script 标签（包括配置和加载脚本）
        let filteredHead = headContent.replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
        
        // 查找 MathJax 配置脚本
        const mathJaxConfig = scripts.find(s => s.textContent && s.textContent.includes('MathJax'))
        if (mathJaxConfig) {
          const scriptOpen = '<' + 'script' + '>'
          const scriptClose = '</' + 'script' + '>'
          const scriptTag = scriptOpen + mathJaxConfig.textContent + scriptClose
          filteredHead += scriptTag
        } else {
          // 如果没有找到配置，添加默认配置
          const defaultConfig = '<' + 'script' + '>window.MathJax={tex:{inlineMath:[[\'$\',\'$\'],[\'\\\\(\',\'\\\\)\']],displayMath:[[\'$$\',\'$$\'],[\'\\\\[\',\'\\\\]\']]}};</' + 'script' + '>'
          filteredHead += defaultConfig
        }
        
        // 添加 MathJax 加载脚本（如果还没有）
        if (!filteredHead.includes('mathjax') && !filteredHead.includes('MathJax')) {
          const scriptOpen = '<' + 'script'
          const scriptSrc = ' src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"'
          const scriptClose = '></' + 'script' + '>'
          const mathJaxScript = scriptOpen + scriptSrc + scriptClose
          filteredHead += mathJaxScript
        }
        
        // 移除 water.css 链接（如果存在），避免样式冲突
        filteredHead = filteredHead.replace(/<link[^>]*water\.css[^>]*>/gi, '')
        
        this.docContent = filteredHead + bodyContent
        
        // 等待 DOM 更新后初始化并渲染 MathJax
        this.$nextTick(() => {
          this.renderMathJax()
        })
      } catch (err) {
        console.error('Failed to load document:', err)
        this.error = `加载文档失败: ${err.message}`
      } finally {
        this.loading = false
      }
    },
    renderMathJax() {
      // 检查 MathJax 是否已加载
      if (window.MathJax && window.MathJax.typesetPromise) {
        // MathJax 3.x
        window.MathJax.typesetPromise().catch((err) => {
          console.warn('MathJax rendering error:', err)
        })
      } else if (window.MathJax && window.MathJax.Hub) {
        // MathJax 2.x (兼容旧版本)
        window.MathJax.Hub.Queue(['Typeset', window.MathJax.Hub])
      } else {
        // 等待 MathJax 加载完成
        const checkMathJax = setInterval(() => {
          if (window.MathJax) {
            clearInterval(checkMathJax)
            if (window.MathJax.typesetPromise) {
              window.MathJax.typesetPromise().catch((err) => {
                console.warn('MathJax rendering error:', err)
              })
            } else if (window.MathJax.Hub) {
              window.MathJax.Hub.Queue(['Typeset', window.MathJax.Hub])
            } else if (window.MathJax.typeset) {
              window.MathJax.typeset()
            }
          }
        }, 100)
        
        // 10秒后停止检查
        setTimeout(() => clearInterval(checkMathJax), 10000)
      }
    }
  }
}
</script>

<style scoped>
@import '../styles/common.css';

.documentation-page {
  display: flex;
  min-height: calc(100vh - 40px);
  margin: 20px;
  gap: 20px;
}

.doc-sidebar {
  width: 250px;
  background: var(--container-color, #1a1a1a);
  border: 1px solid rgba(212, 175, 55, 0.2);
  border-radius: 0;
  padding: 20px;
  height: fit-content;
  max-height: calc(100vh - 80px);
  overflow-y: auto;
  position: sticky;
  top: 20px;
}

.doc-sidebar h3 {
  color: var(--text-color, #d4af37);
  margin-bottom: 15px;
  font-size: 1.1em;
  border-bottom: 1px solid rgba(212, 175, 55, 0.2);
  padding-bottom: 10px;
}

.doc-menu {
  list-style: none;
  padding: 0;
  margin: 0;
}

.doc-menu li {
  padding: 12px 15px;
  margin-bottom: 5px;
  color: var(--text-color, #d4af37);
  cursor: pointer;
  border: 1px solid transparent;
  border-radius: 0;
  transition: all 0.2s;
}

.doc-menu li:hover {
  background: rgba(212, 175, 55, 0.1);
  border-color: rgba(212, 175, 55, 0.3);
}

.doc-menu li.active {
  background: rgba(212, 175, 55, 0.15);
  border-color: var(--text-color, #d4af37);
  font-weight: 600;
}

.doc-content {
  flex: 1;
  background: var(--container-color, #1a1a1a);
  border: 1px solid rgba(212, 175, 55, 0.2);
  border-radius: 0;
  padding: 30px;
  overflow-y: auto;
  max-height: calc(100vh - 80px);
  min-width: 0; /* 允许内容收缩 */
}

/* 覆盖文档中的 max-width 限制和样式 */
.doc-html {
  width: 100% !important;
  max-width: 100% !important;
}

.doc-html :deep(body),
.doc-html :deep(div),
.doc-html :deep(section),
.doc-html :deep(article) {
  max-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  width: 100% !important;
  box-sizing: border-box !important;
}

/* 覆盖 HED 文档中的特定样式 */
.doc-html :deep(style) {
  display: none !important;
}

/* 确保所有文档内容宽度一致 */
.doc-html :deep(*) {
  max-width: 100% !important;
  box-sizing: border-box !important;
}

/* 特别处理 HED 文档的 body 样式 */
.doc-html :deep(body[style*="max-width"]) {
  max-width: 100% !important;
}

/* 确保 MathJax 渲染的公式正确显示 */
.doc-html :deep(.MathJax),
.doc-html :deep(mjx-container) {
  display: inline-block !important;
  margin: 0 0.2em !important;
  vertical-align: middle !important;
}

.doc-html :deep(mjx-math) {
  display: inline-block !important;
}

/* 确保行内公式正确显示 */
.doc-html :deep(mjx-container[display="true"]) {
  display: block !important;
  margin: 1em 0 !important;
  text-align: center !important;
}

.loading, .error {
  color: var(--text-color, #d4af37);
  text-align: center;
  padding: 40px;
  font-size: 1.1em;
}

.error {
  color: #ff6b6b;
}

.doc-html {
  color: var(--text-color, #d4af37);
}

.doc-html :deep(h1),
.doc-html :deep(h2),
.doc-html :deep(h3) {
  color: var(--text-color, #d4af37);
  margin-top: 1.5em;
  margin-bottom: 0.8em;
}

.doc-html :deep(p) {
  color: var(--text-color, #d4af37);
  opacity: 0.9;
  line-height: 1.6;
  margin-bottom: 1em;
}

.doc-html :deep(ul),
.doc-html :deep(ol) {
  color: var(--text-color, #d4af37);
  opacity: 0.9;
  margin-left: 20px;
  margin-bottom: 1em;
}

.doc-html :deep(li) {
  margin-bottom: 0.5em;
}

.doc-html :deep(code) {
  background: rgba(212, 175, 55, 0.1);
  padding: 2px 6px;
  border-radius: 0;
  color: var(--text-color, #d4af37);
  font-family: 'Courier New', monospace;
}

.doc-html :deep(pre) {
  background: rgba(0, 0, 0, 0.3);
  padding: 15px;
  border-radius: 0;
  border: 1px solid rgba(212, 175, 55, 0.2);
  overflow-x: auto;
  margin-bottom: 1em;
}

.doc-html :deep(pre code) {
  background: transparent;
  padding: 0;
}

.doc-html :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 1em;
}

.doc-html :deep(table th),
.doc-html :deep(table td) {
  border: 1px solid rgba(212, 175, 55, 0.3);
  padding: 10px;
  text-align: left;
  color: var(--text-color, #d4af37);
}

.doc-html :deep(table th) {
  background: rgba(212, 175, 55, 0.1);
  font-weight: 600;
}

.doc-html :deep(a) {
  color: var(--text-color, #d4af37);
  text-decoration: underline;
  opacity: 0.8;
}

.doc-html :deep(a:hover) {
  opacity: 1;
}

.doc-html :deep(section) {
  margin-bottom: 2em;
}

@media (max-width: 1024px) {
  .documentation-page {
    flex-direction: column;
  }
  
  .doc-sidebar {
    width: 100%;
    position: relative;
    top: 0;
    max-height: 200px;
  }
}
</style>
