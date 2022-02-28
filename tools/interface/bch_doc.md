
## 使用方法

给 bchlib 做了一层包装，现在可以更加像人一样使用这个库。同时，针对本项目应用场景，加入了可以直接使用的接口：

- BCHHelper()

  > 首先需要创建一个对象来调用所有包装的方法。

- bch.convert_uid_to_data(Union[int, str]) -> (bytearray, str, str)

  > 可以输入整数或者字符串类型，会对大小进行检查。
  >
  > 分别取用户 id 转换成二进制字符串、分钟级时间戳转换成二进制字符串，两者向左填充 0 并拼接。
  >
  > 转换成二进制流返回（注意此时长度将 / 8 ），同时返回写入信息的当地格式化时间，以及用于保存数据库附加信息的主键。

- bch.convert_msg_to_data(str) -> bytearray

  > 用于原 StegaStamp 的接口，将字符串信息转换成二进制流。

- bch.encode_data(bytearray) -> List[int]

  > 将上两条的返回值，即数据段的二进制流，编码并拼接纠错段。
  >
  > 返回用于模型的二进制数组。

- bch.decode_data(Union[List[int], np.ndarray]) -> (int, bytearray)

  > 将模型的输出解码，返回**纠错位数**和**纠错后的数据段**。
  >
  > 当纠错位数为 -1 时，表示检错成功但纠错失败。

- bch.convert_data_to_uid(int, bytearray) -> (int, str, str)

  > 解析上述纠错结果，并返回写入时的用户 id 、当地格式化时间、以及用于查询数据库附加信息的主键

- bch.convert_data_to_msg(int, bytearray) -> str

  > 用于原 StegaStamp 的接口，将结果转换成写入时的信息。

## 代码示例

```python
bch = BCHHelper()

i = 114514
dat, now, key = bch.convert_uid_to_data(i)  # uid -> msg数据段
packet = bch.encode_data(dat)  # 数据段+校验段

# make BCH_BITS errors
for _ in range(5):
    byte_num = random.randint(0, len(packet))
    packet[byte_num] = 1 - packet[byte_num]

bf, dat = bch.decode_data(packet)
i_, now_, key_ = bch.convert_data_to_uid(bf, dat)

print(f"now:  {now}", f"uid: {i}", f"key: {key}")
print(f"time: {now_}", f"uid: {i_}", f"key: {key_}")
```

