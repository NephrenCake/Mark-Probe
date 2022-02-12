import torch
import copy
from steganography.utils import train_utils as tu


# =====================关于添加right_str_acc 相关测试 start====================
def test1():
    # 测试troch数组的遍历：
    size = [2,5]
    x = torch.zeros(size[0],size[1])
    y = torch.zeros(size[0],size[1])
    x[torch.randn(size[0],size[1]) > 0.5] = 1
    y[torch.randn(size[0],size[1]) > 0.5] = 1
    print(y)
    print(x)
    a,b = tu.get_msg_acc(x, y)
    print(a)
    print(b)

    pass

def test2():
    # 深拷贝
    a = torch.randn(3)
    b = copy.deepcopy(a)
    print(a)

    for i in range(len(b)):
        print(b[i])
        if b[i]>0:
            b[i] = 1
        else:
            b[i] = 0
    print(b)
    print(a)
    pass

def test3():
    size = [2, 96]
    x = torch.zeros(size[0], size[1])
    y = torch.zeros(size[0], size[1])
    x[torch.randn(size[0], size[1]) > 0.5] = 1
    y[torch.randn(size[0], size[1]) > 0.5] = 1
    print(y)
    print(x)
    a, b, c = tu.get_msg_acc(x, y)
    print(a)
    print(b)
    print(c)
    pass
# =====================关于添加right_str_acc 相关测试 end  ======================
def test4():
    # 如何将torch.Tensor 的引用传入

    # tensor 是静态的还是动态的?
    a = torch.randn(1,3)
    print(type(a))
    print(a[0][0].item())
    a[0][0] += 1
    print(a[0][0].item())
    loop(a)
    print(a)
    pass

def loop(a):

    if a[0][0].item() >= 10.0:
        return
    a[0][0]+=1
    loop(a)
    loop(a)



if __name__=="__main__":
    test4()