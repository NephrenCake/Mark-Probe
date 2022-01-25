// 其他工具类

class Func {
  getDateTime() {
    const date = new Date();
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const strDate = date.getDate().toString().padStart(2, '0');
    const starHours = date.getHours().toString().padStart(2, '0');
    const starMinutes = date.getMinutes().toString().padStart(2, '0');
    const starSeconds = date.getSeconds().toString().padStart(2, '0');
    return `${date.getFullYear()}-${month}-${strDate}T${starHours}:${starMinutes}:${starSeconds}`;
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

  // 文件转Base64：此为异步函数，必须异步调用
  file2Base64(file) {
    return new Promise((resolve, reject) => {
        ///FileReader类就是专门用来读文件的
        const reader = new FileReader()
        //开始读文件
        //readAsDataURL: dataurl它的本质就是图片的二进制数据， 进行base64加密后形成的一个字符串，
        reader.readAsDataURL(file)
        // 成功和失败返回对应的信息，reader.result一个base64，可以直接使用
        reader.onload = () => resolve(reader.result)
        // 失败返回失败的信息
        reader.onerror = error => reject(error)
    })
  }
}

export default new Func();