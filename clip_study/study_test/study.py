
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.utils.data

torch.manual_seed(0)
torch.cuda.manual_seed(0)

output = torch.randn(2, 3)
print("output=\n", output)
print("softmax=\n", F.softmax(output, dim=1))  # dim = 1, 每一行的相加和=1
print("log_softmax=\n", F.log_softmax(output, dim=1))
print("torch_log_softmax=\n", torch.log(F.softmax(output,  dim=1)))  # same with log_softmax
print("nll_loss=\n", F.nll_loss(torch.tensor([[-1.2, -2, -3]]), torch.tensor([2])))

# log softmax 和 nll loss相结合，log softmax本身就代表的loss的负数
# 在分类问题中，CrossEntropy 等价于 log softmax 结合 nll loss
output1 = torch.tensor([[1.2, 2, 3]])
log_sm_output = F.log_softmax(output1, dim=1)
target0 = torch.tensor([0])
target1 = torch.tensor([1])
target2 = torch.tensor([2])
print("log_softmax=\n", log_sm_output)
print("target0:\n", F.nll_loss(log_sm_output, target0))
print("target2:\n", F.nll_loss(log_sm_output, target1))
print("target2:\n", F.nll_loss(log_sm_output, target2))




print("Hello", 1)
print('hello world')
print(1+2)


