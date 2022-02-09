import axios from 'axios';

const BASE_URL = 'http://127.0.0.1:5000';

class Http {
  constructor() {
    this.http = axios.create({
      baseURL: BASE_URL,
      timeout: 1000
    });
  }

  // 设置 ID
  sendID(params) {
    const url = '/custom';
    return this.http.post(url, params);
  }

  // 开始推拉流
  pullStream() {
    const url = '/pull';
    return this.http.get(url);
  }

  // 结束推拉流
  stopStream() {
    const url = '/stop';
    return this.http.get(url);
  }

  // 上传图片以及校正参数
  uploadPic(params) {
    const url = "/upload";
    return this.http.post(url, params)
  }

  // 上传图片以及校正参数
  uploadPicEn(params) {
    const url = "/uploaden";
    return this.http.post(url, params)
  }
} 

export default new Http();