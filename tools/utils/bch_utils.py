# -- coding: utf-8 --
"""
@Date: 2021/12/15 22:51
@Author: NephrenCake
@File: bch_utils.py
"""
import bchlib

BCH_POLYNOMIAL = 137
BCH_BITS = 5


def get_byte_msg(msg):
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    if len(msg) > 7:
        print('Error: Can only encode 56bits (7 characters) with ECC')
        return
    # 补齐到7个字符，utf8编码
    data = bytearray(msg + ' ' * (7 - len(msg)), 'utf-8')
    ecc = bch.encode(data)  # bytearray(b'\x88\xa9\xfbN@')
    packet = data + ecc  # bytearray(b'Stega!!\x88\xa9\xfbN@')  12 = 7 + 5 字节
    # 校验码，两者加起来最多96bits
    packet_binary = ''.join(format(x, '08b') for x in packet)  # 转二进制
    # '010100110111010001100101011001110110000100100001001000011000100010101001111110110100111001000000'
    byte_msg = [int(x) for x in packet_binary]  # 转数组，len=96
    byte_msg.extend([0, 0, 0, 0])  # 补到len=100
    return byte_msg


def get_row_msg(msg_pred):
    packet_binary = "".join([str(int(bit)) for bit in msg_pred[:96]])
    packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
    packet = bytearray(packet)

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

    bitflips = bch.decode_inplace(data, ecc)
    if bitflips != -1:
        try:
            code = data.decode("utf-8")
            print(code)
            return
        except:
            return
    print('Failed to decode')
