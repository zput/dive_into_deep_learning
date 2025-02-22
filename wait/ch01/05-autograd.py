import torch

# https://zhuanlan.zhihu.com/p/168748668
# https://blog.csdn.net/Weary_PJ/article/details/105706318

print('1.自动梯度计算')
x = torch.arange(4.0, requires_grad=True)  # 1.将梯度附加到想要对其计算偏导数的变量
print('x:', x)
print('x.grad:', x.grad)
print("=====================================================>", torch.dot(x, x))
# y = 2 * torch.dot(x, x)  # 2.记录目标值的计算 torch.dot()类似于mul()，它是向量(即只能是一维的张量)的对应位相乘再求和，返回一个tensor数值 0+1*1+2*2+3*3 = 1+4+9 = 14
                           # 2*(x1^2+x2^2+x3^2+x4^2)
y = torch.sum(x)  # 2.记录目标值的计算 torch.dot()类似于mul()，它是向量(即只能是一维的张量)的对应位相乘再求和，返回一个tensor数值 0+1*1+2*2+3*3 = 1+4+9 = 14
                  # x1+x2+x3+x4
print('y:', y)

print("==============================================================================================================>")
y.backward()  # 3.执行它的反向传播函数
print("===反向传播===")
print('x.grad:', x.grad)  # 4.访问得到的梯度
print('x.grad == 4*x:', x.grad == 4 * x)


print("<=============================================================================================================>")

## 计算另一个函数
x.grad.zero_()
y = x.sum()
print('y:', y)
y.backward()
print('x.grad:', x.grad)

# 非标量变量的反向传播
x.grad.zero_()
print('x:', x)
y = x * x
y.sum().backward()
print('x.grad:', x.grad)


def f(a):
    b = a * 2
    print(b.norm())
    while b.norm() < 1000:  # 求L2范数：元素平方和的平方根
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


print('2.Python控制流的梯度计算')
a = torch.tensor(2.0)  # 初始化变量
a.requires_grad_(True)  # 1.将梯度赋给想要对其求偏导数的变量
print('a:', a)
d = f(a)  # 2.记录目标函数
print('d:', d)
d.backward()  # 3.执行目标函数的反向传播函数
print('a.grad:', a.grad)  # 4.获取梯度
