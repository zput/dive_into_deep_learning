import torch

y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat)
y = torch.LongTensor([0, 2])
print(y.view(-1,1))  # 把一个[0, 2] 变为了 [[0],[2]]       ## 有两种方法: >使用view; >使用t();
ret = y_hat.gather(1, y.view(-1, 1))
print(ret)

a = torch.arange(0,16).view(4,4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11],
#         [12, 13, 14, 15]])
print(a)
# tensor([[ 0, X,  X,  X],
#         [ 4, X,  X,  X],
#         [ X, X,  X, 11],
#         [ X, X,  X, 15]])

# 如果想要得到上面的值，row index 是不能的
## row = torch.LongTensor([[0, 1, 2, 3]])
## print(a.gather(0,row))
print("================================")
column = torch.LongTensor([[0, 0, 3, 3]])
print("column:{} \n column.t:{}".format(column, column.t()))
print(a.gather(1,column.t()))
print(a.gather(0,column))
