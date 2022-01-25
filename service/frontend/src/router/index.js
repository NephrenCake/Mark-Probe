import Vue from 'vue';
import VueRouter from 'vue-router';

Vue.use(VueRouter);

const routes = [
  {
    path: '/',
    name: 'Main',
    component: () => import('@/views/Main'),
  },
  {
    path: '/encoder',
    name: 'Frame',
    component: () => import('@/views/Frame'),
    children: [
      {
        path: '/encoder',
        name: 'Encoder',
        component: () => import('@/views/Encoder')
      },
      {
        path: '/decoder',
        name: 'Decoder',
        component: () => import('@/views/Decoder')
      }
    ]
  },
  {
    path: '/404',
    name: '404',
    component: () => import('@/views/404')
  },
  {
    path: '*',
    name: 'NotFound',
    component: () => import('@/views/404')
  }  
];

const router = new VueRouter({
  mode: 'history',
  routes
});

export default router;