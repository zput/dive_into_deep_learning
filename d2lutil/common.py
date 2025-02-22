import torch
import torchvision
import torchvision.transforms as transforms
from d2l import torch as d2l
from torch.utils import data
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.datasets.utils import extract_archive


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
    d2l.plt.show()


def hello():
    print("semilogy_HELLO")


def get_path() -> str:
    import os
    dotenv_path = os.path.dirname(os.path.abspath(__file__)) + '/data_source'
    print(dotenv_path)
    return dotenv_path

# 采用本地下载的方式先将数据集下载到本地
# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
def load_fashion_mnist(batch_size):
    path_head = get_path()
    extract_archive(path_head + '/t10k-images-idx3-ubyte.gz',  path_head + '/FashionMNIST/raw', False)
    extract_archive(path_head + '/train-images-idx3-ubyte.gz', path_head + '/FashionMNIST/raw', False)
    extract_archive(path_head + '/t10k-labels-idx1-ubyte.gz',  path_head + '/FashionMNIST/raw', False)
    extract_archive(path_head + '/train-labels-idx1-ubyte.gz', path_head + '/FashionMNIST/raw', False)

    training_set = (
        read_image_file(path_head + '/FashionMNIST/raw/train-images-idx3-ubyte'),
        read_label_file(path_head + '/FashionMNIST/raw/train-labels-idx1-ubyte')
    )
    test_set = (
        read_image_file(path_head + '/FashionMNIST/raw/t10k-images-idx3-ubyte'),
        read_label_file(path_head + '/FashionMNIST/raw/t10k-labels-idx1-ubyte')
    )
    with open(path_head + '/FashionMNIST/processed/training.pt', 'wb') as f:
        torch.save(training_set, f)
    with open(path_head + '/FashionMNIST/processed/test.pt', 'wb') as f:
        torch.save(test_set, f)
    print('Done!')

    # train_data, train_targets = torch.load('D://d2l-data//FashionMNIST//processed//training.pt')
    # test_data, test_targets = torch.load('D://d2l-data//FashionMNIST//processed//test.pt')

    mnist_train = torchvision.datasets.FashionMNIST(root=path_head + "/", train=True, transform=transforms.ToTensor(),
                                                    download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root=path_head + "/", train=False, transform=transforms.ToTensor(),
                                                   download=False)
    num_workers = 0
    import sys
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 0

    train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return (train_iter, test_iter)
