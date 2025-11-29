import { createRouter, createWebHistory } from 'vue-router'
import EdgeDetection from './views/EdgeDetection.vue'
import VehicleDetection from './views/VehicleDetection.vue'

const routes = [
  { path: '/', name: 'home', component: EdgeDetection },
  { path: '/vehicle', name: 'vehicle', component: VehicleDetection }
]

export const router = createRouter({
  history: createWebHistory(),
  routes
})


