import { createRouter, createWebHistory } from 'vue-router'
import EdgeDetection from './views/EdgeDetection.vue'
import DeepLearningEdgeDetection from './views/DeepLearningEdgeDetection.vue'
import VehicleDetection from './views/VehicleDetection.vue'
import SmartRemoval from './views/SmartRemoval.vue'
import Documentation from './views/Documentation.vue'
import Changelog from './views/Changelog.vue'

const routes = [
  { path: '/', name: 'home', component: EdgeDetection },
  { path: '/deep-learning', name: 'deepLearning', component: DeepLearningEdgeDetection },
  { path: '/vehicle', name: 'vehicle', component: VehicleDetection },
  { path: '/smart-removal', name: 'smartRemoval', component: SmartRemoval },
  { path: '/docs', name: 'docs', component: Documentation },
  { path: '/changelog', name: 'changelog', component: Changelog }
]

export const router = createRouter({
  history: createWebHistory(),
  routes
})


