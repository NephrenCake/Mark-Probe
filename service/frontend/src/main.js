import Vue from 'vue';
import App from './App.vue';
import './registerServiceWorker';
import router from './router';
import store from './store';
import ElementUI from 'element-ui';
import flvjs from 'flv.js';

import 'element-ui/lib/theme-chalk/index.css';
import 'normalize.css/normalize.css';
import '@/styles/init.css';

import '@/styles/fonts/regular.css';
import '@/styles/fonts/medium.css';

import http from '@/utils/http';
import Func from '@/utils/others';

// 静态资源
import res from '@/assets/res';

Vue.config.productionTip = false;

Vue.use(ElementUI);

Vue.prototype.$http = http;
Vue.prototype.$func = Func;

Vue.prototype.$flv = flvjs;

Vue.prototype.$res = res;

new Vue({
  router,
  store,
  render: h => h(App)
}).$mount('#app');