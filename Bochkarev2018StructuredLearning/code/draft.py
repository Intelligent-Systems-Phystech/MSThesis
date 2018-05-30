import torch
from torch.autograd import Variable

a = torch.Tensor([1, 2])
a = Variable(a, requires_grad=True)
y = torch.abs(a)
y = torch.sum(y)
y.backward()
