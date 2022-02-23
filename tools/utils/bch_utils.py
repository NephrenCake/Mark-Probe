# -- coding: utf-8 --
"""
@Date: 2021/12/15 22:51
@Author: NephrenCake
@File: bch_utils.py
"""
import random
import time
import bchlib
import numpy as np
from typing import List, Union


class BCHHelper:
    def __init__(self,
                 polynomial=137,
                 bits=5,
                 uid_size=30,
                 time_size=26,
                 ecc_size=40,
                 format_spec="%Y-%m-%d %H:%M:%S",
                 start_time_str="2022-01-01 00:00:00",
                 ):
        self.format_spec = format_spec
        self.start_time = int(time.mktime(time.strptime(start_time_str, format_spec)))
        self.ecc_size = ecc_size
        self.time_size = time_size
        self.uid_size = uid_size
        self.uid_limits = 2 ** uid_size
        self.bch = bchlib.BCH(polynomial, bits)

    def convert_uid_to_data(self, uid: Union[int, str]) -> (bytearray, str, str):
        """
        给定十进制 uid，返回 msg 数据段的二进制流
        """
        if isinstance(uid, str):
            uid = int(uid)
        assert uid < self.uid_limits, "超过设计可用用户总数"

        msg_uid = self.fill_bin(bin(uid)[2:], self.uid_size)

        current_time = int(time.mktime(time.localtime()))  # 获取当前时间
        msg_time = int((current_time - self.start_time) / 60)  # 计算当前时间与设置的开始时间差

        msg = msg_uid + self.fill_bin(bin(msg_time)[2:], self.time_size)
        data = bytes(int(msg[i: i + 8], 2) for i in range(0, len(msg), 8))

        # print(msg, len(msg))
        return bytearray(data), time.strftime(self.format_spec, time.gmtime(msg_time * 60 + self.start_time)), msg

    @staticmethod
    def convert_msg_to_data(msg: str) -> bytearray:
        """
        将给定的小于7个char的英文字符转换为二进制数组
        """
        if isinstance(msg, str):
            assert len(msg) <= 7, 'Error: Can only encode 56bits (7 characters) with ECC'

        # 补齐到7个字符，utf8编码
        return bytearray(msg + ' ' * (7 - len(msg)), 'utf-8')

    def encode_data(self, data: bytearray) -> List[int]:
        ecc = self.bch.encode(data)
        packet = data + ecc

        packet_binary = ''.join(format(x, '08b') for x in packet)  # bytearray转二进制str

        return [int(x) for x in packet_binary]  # str 转 [int]

    def decode_data(self, msg_pred: Union[List[int], np.ndarray]) -> (int, bytearray):
        """
        将 数据段+校验段 解码返回 纠正位数,纠正后数据段
        """
        packet_binary = "".join([str(int(bit)) for bit in msg_pred])

        packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)

        data, ecc = packet[:-self.bch.ecc_bytes], packet[-self.bch.ecc_bytes:]

        bitflips = self.bch.decode_inplace(data, ecc)

        return bitflips, data

    def convert_data_to_uid(self, bit_flips: int, data: bytearray) -> (int, str, str):
        if bit_flips == -1:
            print("Failed to decode. Can't correct data!")

        data = ''.join(format(x, '08b') for x in data)
        msg_uid, msg_time = data[:self.uid_size], data[-self.time_size:]

        t = time.strftime(self.format_spec, time.gmtime(int(msg_time, 2) * 60 + self.start_time))

        return int(msg_uid, 2), t, data

    @staticmethod
    def convert_data_to_msg(bit_flips: int, data: bytearray) -> str:
        if bit_flips != -1:
            try:
                code = data.decode("utf-8")
                print(code)
                return code
            except:
                print('Failed to decode. Not encoded in utf-8!')
                return ""
        print("Failed to decode. Can't correct data!")
        return ""

    @staticmethod
    def fill_bin(msg: str, size: int, pre: bool = True, fill: str = "0") -> str:
        """
        使用 fill 填充大小为 size 的二进制字符串 str
        """
        return fill * (size - len(msg)) + msg if pre else msg + fill * (size - len(msg))


if __name__ == '__main__':
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
