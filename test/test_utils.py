import torch

from steganography.utils.train_utils import get_msg_acc,get_msg_acc_o


# 生成随机的10 个 torch
size = [2, 96]
true_msg = torch.zeros(size[0], size[1])
pred_msg = torch.zeros(size[0], size[1])
true_msg[torch.randn(size[0], size[1]) > 0.5] = 1
pred_msg[torch.randn(size[0], size[1]) > 0.5] = 1


def print_msg():
    print("true_msg: ",true_msg)
    print("dtype: ", type(true_msg))
    print("pred_msg: ",true_msg)

def test_get_msg_acc():
    print_msg()
    # ================= 测试 区 1================================
    a,b,c = get_msg_acc(msg_true=true_msg, msg_pred=pred_msg)

    print(a)
    print(b)
    print(c)

    print("=======================================")
    a, b, c = get_msg_acc_o(msg_true=true_msg, msg_pred=pred_msg)

    print(a)
    print(b)
    print(c)



    # ================= 测试 区 2================================
    _m = true_msg-1

    k = torch.count_nonzero(_m)


    print(_m)
    print(k)

    pass

if __name__=="__main__":
    test_get_msg_acc()