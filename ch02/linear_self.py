import torch
from IPython import display


from matplotlib import pyplot as plt
import numpy as np
import random
import matplotlib_inline

from matplotlib_inline.backend_inline import set_matplotlib_formats

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)

print(features)
print("==================>", len(features), type(features))
print("第一列:", features[:, 0]) # 第一列
print("第一行:", features[0, :]) # 第一行
print("=================================")
print("第一维度:", features.index_select(0, index=torch.tensor([0, 1, 2, 3, 4])))
print("=================================")
#print("第二维度:", features.index_select(1, index=torch.tensor([0, 1])))
print("第二维度,的第一列:", features.index_select(1, index=torch.tensor([0])))
print("第二维度,的第二列:", features.index_select(1, index=torch.tensor([1])))
print("=================================")

labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)


print(features[0], labels[0])
#tensor([0.8557, 0.4793])
#tensor(4.2887)

def use_svg_display():
    # display.set_matplotlib_formats('svg')
    # 用矢量图显示
    #matplotlib_inline.backend_inline.set_matplotlib_formats()
    set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

# # 在../d2lzh_pytorch里面添加上面两个函数后就可以这样导入
# import sys
# sys.path.append("..")
# from d2lzh_pytorch import *

set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
plt.show()


# 本函数已保存在d2lzh包中方便以后使用
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j)


batch_size = 2

count=0
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    count+=1
    if count == 2:
        break



w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

print(b)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

def linreg(X, w, b):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    return torch.mm(X, w) + b

#把真实值y变形成预测值y_hat的形状
def squared_loss(y_hat, y):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    # print("y_hat:{}, size:{}".format(y_hat, y_hat.size()))
    # print("y.view()::", y.view(y_hat.size()))
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data


lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels): # 提取特征，标签(结果), 准备训练.
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        print("l:{}; type:{}, {}".format(l, type(l), l.grad_fn))
        print("==1==>w:{}; b:{}".format(w.grad,b.grad))
        l.backward()  # 小批量的损失对模型参数求梯度
        print("==2==>w:{}; b:{}".format(w.grad,b.grad))
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))


print(true_w, '\n', w)
print(true_b, '\n', b)

