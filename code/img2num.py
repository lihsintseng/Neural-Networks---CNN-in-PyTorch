# Credit: Li-Hsin Tseng
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.functional as F

'''
Update your code in order to create your network, perform forward and back-prop using Pytorch’s nn package.
In order to update parameters use optim package.
Compare speed and training error vs epochs charts.
'''

class Img2Num(nn.Module):
	def __init__(self):
		# https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
		super(Img2Num, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, (5,5), padding=2)
		self.conv2 = nn.Conv2d(6, 16, (5,5))
		self.fc1   = nn.Linear(16*5*5, 120)
		self.fc2   = nn.Linear(120, 84)
		self.fc3   = nn.Linear(84, 10)
		self.criterion = nn.CrossEntropyLoss()

		root, download, self.batch_size = './data', False, 64
		self.train_loader = torch.utils.data.DataLoader(
		    datasets.MNIST(root, train=True, download=download,
		                   transform=transforms.Compose([
		                       transforms.ToTensor(),
		                       transforms.Normalize((0.1307,), (0.3081,))
		                   ])),
		    batch_size=self.batch_size, shuffle=True)

	# [nil] train()
	def train(self):
		optimizer = optim.SGD(self.parameters(), lr=0.01)
		for batch_idx, (data, target) in enumerate(self.train_loader):
			optimizer.zero_grad()
			data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
			x = F.max_pool2d(F.relu(self.conv1(data)), (2,2))
			x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
			size, res = x.size()[1:], 1
			for s in size: res *= s
			x = x.view(-1, res)
			x = F.relu(self.fc1(x))
			x = F.relu(self.fc2(x))
			x = self.fc3(x)
			loss = self.criterion(x, target)
			loss.backward()
			optimizer.step()

	# [int] forward([28x28 ByteTensor] img)
	def forward(self, img):
		x = F.max_pool2d(F.relu(self.conv1(img)), (2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
		size, res = x.size()[1:], 1
		for s in size: res *= s
		x = x.view(-1, res)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		_, pred_label = torch.max(x.data, 1)
		return pred_label