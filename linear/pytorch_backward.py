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