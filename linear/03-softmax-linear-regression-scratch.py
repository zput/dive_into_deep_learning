import sys
import os
import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l
from d2lutil import common
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_fashion_mnist_labels(labels):  # @save
    """返回Fashion-MNIST数据集的文本标签。"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

# 定义模型
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


def net(X):
    # kprint("===>X.lenght:{}".format(len(X)))
    # kprint(X)
    # kprint("===>", X.reshape(-1, num_inputs))
    # kprint("===>", len(torch.mm(X.reshape(-1, num_inputs), W)))
    return softmax(torch.matmul(X.reshape(-1, num_inputs), W) + b) # 触发广播

def cross_entropy(y_hat, y):
    # print("=================> \n y_hat:{}; \n y:{} \n =======================< \n".format(y_hat, y))
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# 计算分类准确率
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


# 计算这个训练集的准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def train_linear(net, train_iter, test_iter, loss, num_epochs, batch_size,
                 params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            # 执行优化方法
            if optimizer is not None:
                optimizer.step()
            else:
                d2l.sgd(params, lr, batch_size)

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

######################################################################################
# > 获取并读取数据
# > 定义模型
# > 损失函数并使用优化算法训练模型
#####################################################################################


## 读取小批量数据
batch_size = 256   # batch_size = 2
train_iter, test_iter = common.load_fashion_mnist(batch_size)
print(len(train_iter))  # train_iter的长度是235；说明数据被分成了234组大小为256的数据加上最后一组大小不足256的数据
print('数据读取完成!!!')

#展示部分训练数据
train_data, train_targets = iter(train_iter).next()
show_fashion_mnist(train_data[0:10], get_fashion_mnist_labels(train_targets[0:10]))
# print(train_data, train_targets, len(train_data), len(train_targets))
# show_fashion_mnist(train_data[:], get_fashion_mnist_labels(train_targets[:]))

# 初始化模型参数
num_inputs = 784
num_outputs = 10

## W, b为全局变量
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
# print("w.length:{}".format(len(W)))

# 定义损失函数
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
y_hat.gather(1, y.view(-1, 1))

# 训练模型
num_epochs, lr = 10, 0.1 # 设置迭代次数, 学习率.
train_linear(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

# 使用测试集来跑一下我们生成的预测模型.
X, y = iter(test_iter).next()
true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
print("titles.length:{}; title0:{}; type:{}".format(len(titles),titles[0], type(titles)))
show_fashion_mnist(X[0:9], titles[0:9])
