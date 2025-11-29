import { createRouter, createWebHistory } from 'vue-router'
import EdgeDetection from './views/EdgeDetection.vue'
import VehicleDetection from './views/VehicleDetection.vue'
import Documentation from './views/Documentation.vue'

const routes = [
  { path: '/', name: 'home', component: EdgeDetection },
  { path: '/vehicle', name: 'vehicle', component: VehicleDetection },
  { path: '/docs', name: 'docs', component: Documentation }
]

export const router = createRouter({
  history: createWebHistory(),
  routes
})


