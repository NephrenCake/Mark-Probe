// 其他工具类

class Func {
  getDate() {
    const date = new Date();
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const strDate = date.getDate().toString().padStart(2, '0');
    return `${date.getFullYear()} 年 ${month} 月 ${strDate} 日`;
  }

  getDateTime() {
    const date = new Date();
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const strDate = date.getDate().toString().padStart(2, '0');
    const starHours = date.getHours().toString().padStart(2, '0');
    const starMinutes = date.getMinutes().toString().padStart(2, '0');
    const starSeconds = date.getSeconds().toString().padStart(2, '0');
    return `${date.getFullYear()}-${month}-${strDate} ${starHours}:${starMinutes}:${starSeconds}`;
  }
  
  // 清除对象
  clearObj(obj) {
    for (let item in obj) {
      if (Array.isArray(obj[item])) {
        obj[item] = [];
      } else {
        obj[item] = null;
      }
    }
  }

  // 文件转 Base64：此为异步函数，必须异步调用
  file2Base64(file) {
    return new Promise((resolve, reject) => {
        // FileReader 类就是专门用来读文件的
        const reader = new FileReader()
        // 开始读文件
        // readAsDataURL: dataurl 它的本质就是图片的二进制数据， 进行 base64 加密后形成的一个字符串，
        reader.readAsDataURL(file)
        // 成功和失败返回对应的信息，reader.result 一个 base64，可以直接使用
        reader.onload = () => resolve(reader.result)
        // 失败返回失败的信息
        reader.onerror = error => reject(error)
    })
  }

  // 图片转 blob(url)
  img2Blob(event) {
    // 获取图片的本地 URL，用于本地预览
    let URL = null;
    if (window.createObjectURL != undefined) {
      // basic
      URL = window.createObjectURL(event.raw);
    } else if (window.URL != undefined) {
      // mozilla (firefox)
      URL = window.URL.createObjectURL(event.raw);
    } else if (window.webkitURL != undefined) {
      // webkit or chrome
      URL = window.webkitURL.createObjectURL(event.raw);
    }
    // 转换后的地址为 blob:http://xxx/7bf54338-74bb-47b9-9a7f-7a7093c716b5

    return URL;
  }

  // base64(无文件信息部分) 转 Blob 对象
  base64ToBlob(dataurl, filename) {
    let arr = dataurl.split(',');
    let mime = arr[0].match(/:(.*?);/)[1];
    let bstr = atob(arr[1]);
    let n = bstr.length;
    let u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], { type: mime });
  }

  // 生成 Blob 对象下载链接
  createDownloadFileUrl(fileName, file) {
    const blob = this.base64ToBlob(`data:image/jpeg;base64,${file}`, fileName);
    blob.lastModifiedDate = new Date();
    blob.name = fileName;
    return URL.createObjectURL(blob);
  }

  // blob 对象转 base64
  blob2Base64(blob) {
    return new Promise((resolve, reject) => {
      const fileReader = new FileReader();
      fileReader.onload = (e) => {
        resolve(e.target.result);
      };
      // readAsDataURL
      fileReader.readAsDataURL(blob);
      fileReader.onerror = () => {
        reject(new Error('blobToBase64 error'));
      };
    });
  }
}

export default new Func();