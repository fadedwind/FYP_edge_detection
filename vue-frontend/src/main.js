import { createApp } from 'vue'
import App from './App.vue'
import { router } from './router'
import './styles/common.css'

const app = createApp(App)
app.use(router)
app.mount('#app')

// 确保 body 背景色为黑金色
document.body.style.backgroundColor = '#0a0a0a'

