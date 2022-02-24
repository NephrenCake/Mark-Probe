import Vue from 'vue';
import Vuex from 'vuex';

Vue.use(Vuex);

const store = new Vuex.Store({
  state: {
    // 要送去解码的失真图片
    // Vuex 的 state 使用方式（不推荐）: let value = $store.state.psPicToDecode;
    psPicToDecode: null
  },
  mutations: {
    setPsPicToDecode(state, pic) {
      // Vuex 的 mutations 使用方式（推荐）: (是一个方法) $store.commit('setPsPicToDecode', 第二个形参 pic);
      state.psPicToDecode = pic;
    },
    rmPsPicToDecode(state) {
      state.psPicToDecode = null;
    }
  },
  getters: {
    getPsPicToDecode(state) {
      // Vuex 的 getters 使用方式: let value = $store.getters.getPsPicToDecode;
      return state.psPicToDecode;
    }
  },
  actions: {
  },
  modules: {
  }
});

export default store;