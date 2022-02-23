import Vue from 'vue';
import VueRouter from 'vue-router';

Vue.use(VueRouter);

const routes = [
  {
    path: '/',
    // name: 'Frame',
    component: () => import('@/views/Frame'),
    children: [
      {
        path: '/',
        name: 'Main',
        component: () => import('@/views/Main')
      },
      {
        path: '/encoder/pic',
        name: 'EncoderPic',
        component: () => import('@/views/EncoderPic')
      },
      {
        path: '/encoder/stream',
        name: 'EncoderStream',
        component: () => import('@/views/EncoderStream')
      },
      {
        path: '/decoder/pic',
        name: 'DecoderPic',
        component: () => import('@/views/DecoderPic')
      },
      // {
      //   path: '/decoder/stream',
      //   name: 'DecoderStream',
      //   component: () => import('@/views/DecoderStream')
      // },
      {
        path: '/ps',
        name: 'PS',
        component: () => import('@/views/PS')
      },
      {
        path: '*',
        name: 'NotFound',
        component: () => import('@/views/404')
      } 
    ]
  }
];

const router = new VueRouter({
  mode: 'history',
  routes
});

export default router;