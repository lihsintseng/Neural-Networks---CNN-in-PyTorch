# Credit: Li-Hsin Tseng

import cv2, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class Img2obj(nn.Module):
	def __init__(self):
		super(Img2obj, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, (5,5), padding=2)
		self.conv2 = nn.Conv2d(6, 16, (5,5))
		self.fc1   = nn.Linear(16*6*6, 120)
		self.fc2   = nn.Linear(120, 84)
		self.fc3   = nn.Linear(84, 100)
		self.criterion = nn.CrossEntropyLoss()
		self.CIFAR100_LABELS_LIST = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]
		root, download, self.batch_size = './data', False, 64
		#mean = [x / 255 for x in [125.3, 123.0, 113.9]]
		#std = [x / 255 for x in [63.0, 62.1, 66.7]]
		train_transform = transforms.Compose(
			[transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor()]) # ,transforms.Normalize(mean, std)
    
		train_data = datasets.CIFAR100(root, train=True, transform=train_transform, download=download)
		self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
    
	# [nil] train()
	def train(self):
		optimizer = optim.SGD(self.parameters(), lr=0.01)
		for batch_idx, (data, target) in enumerate(self.train_loader):
			optimizer.zero_grad()
			data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
			x = F.max_pool2d(F.relu(self.conv1(data)), (2,2))
			x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
			x = x.view(x.size(0), -1)
			x = F.relu(self.fc1(x))
			x = F.relu(self.fc2(x))
			x = self.fc3(x)
			loss = self.criterion(x, target)
			loss.backward()
			optimizer.step()
	
	# [str] forward([3x32x32 ByteTensor] img)
	def forward(self, img):
		# https://www.kaggle.com/usingtc/lenet-with-pytorch
		# https://gist.github.com/xmfbit/b27cdbff68870418bdb8cefa86a2d558
		x = F.max_pool2d(F.relu(self.conv1(img)), (2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		_, pred_label = torch.max(x.data, 1)
		return self.CIFAR100_LABELS_LIST[pred_label.numpy()[0]]	

	# [nil] view([3x32x32 ByteTensor] img) -- view the image and prediction
	def view(self, img):
		label = self.forward(img)
		f = plt.figure()
		ax = f.add_subplot(111)
		f.text(0.1, 0.9,label, ha='center', va='center', transform=ax.transAxes, fontsize=14)
		img = img.data.numpy()
		new = img[0].transpose(1, 2, 0)
		plt.xticks([]), plt.yticks([])
		plt.imshow(new) 
		plt.show()
		

	# [nil] cam([int] /idx/) -- fetch images from the camera
	def cam(self, idx = 0):
		# https://softwarerecs.stackexchange.com/questions/18134/python-library-for-taking-camera-images
		camera = cv2.VideoCapture(idx)
		time.sleep(0.1) 
		font = cv2.FONT_HERSHEY_SIMPLEX
		while(True):
			_, image = camera.read()
			tmp = image.shape
			img = [[]]
			img[0] = image.transpose(2, 0, 1)
			new = np.zeros((1, tmp[2], 32, 32))
			for i in range(tmp[2]):
				tmp = img[0][i]
				new[0][i] = np.resize(tmp, (32, 32))
				maximum, minimum = np.max(new[0][i]), np.min(new[0][i])
				new[0][i] = (new[0][i] - minimum)/(maximum-minimum)
			new = torch.from_numpy(new).type(torch.FloatTensor)
			label = self.forward(torch.autograd.Variable(new))
			cv2.putText(image,label,(10,500), font, 4,(1,1,1),2,cv2.LINE_AA)
			cv2.imshow('preview', image)
			if cv2.waitKey(1) & 0xFF == ord('q'):
			    break

		camera.release()
		cv2.destroyAllWindows() 
		'''
		t_end = time.time() + 20 * 1
		while time.time() < t_end:
			_, image = camera.read()
			tmp = image.shape
			img = [[]]
			img[0] = image.transpose(2, 0, 1)
			#img = image.reshape(1, tmp[2], tmp[0], tmp[1])
			new = np.zeros((1, tmp[2], 32, 32))
			for i in range(tmp[2]):
				tmp = img[0][i]
				new[0][i] = np.resize(tmp, (32, 32))
				maximum, minimum = np.max(new[0][i]), np.min(new[0][i])
				new[0][i] = (new[0][i] - minimum)/(maximum-minimum)
			new = torch.from_numpy(new).type(torch.FloatTensor)
			label = self.forward(torch.autograd.Variable(new))
			f = plt.figure()
			ax = f.add_subplot(111)
			f.text(0.1, 0.9,label, ha='center', va='center', transform=ax.transAxes, fontsize=14)
			plt.imshow(image) 
			plt.show()
		del(camera)
		'''