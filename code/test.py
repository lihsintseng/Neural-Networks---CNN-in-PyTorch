# Credit: Li-Hsin Tseng

import torch
import img2obj
import img2num
import time
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

def test_img2num(num_model, epoch_num):
	root, download, test_batch_size = './data', False, 1
	test_loader = torch.utils.data.DataLoader(
    	datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
					])),
    batch_size=test_batch_size, shuffle=True)
	time_spent, conv, forward = [], [], []
	train_start = time.time()
	for i in range(epoch_num):
		num_model.train()
		train_end = time.time()
		num_time = train_end - train_start
		cnt = 0
		forward_start = time.time()
		for batch_idx, (data, target) in enumerate(test_loader):
			data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
			res = num_model.forward(data)
			if (torch.eq(res, target).data).numpy(): cnt += 1
		forward_end = time.time()
		print(str(i+1) + ' epoch')
		print('img2num time (sec):', end = '')
		print(num_time)
		time_spent.append(num_time)
		print('img2num accuracy:', end = '')
		print(cnt/len(test_loader))
		conv.append(1-cnt/len(test_loader))
		forward.append(forward_end - forward_start)
	return num_model, time_spent, conv, forward

def test_img2obj(obj_model, epoch_num):
	CIFAR100_LABELS_LIST = [
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
	root, download, test_batch_size = './data', False, 1
	time_spent, conv, forward = [], [], []
	#mean = [x / 255 for x in [125.3, 123.0, 113.9]]
	#std = [x / 255 for x in [63.0, 62.1, 66.7]]
	test_transform = transforms.Compose(
	        [transforms.ToTensor()]) # , transforms.Normalize(mean, std)
	test_data = datasets.CIFAR100(root, train=False, transform=test_transform, download=download)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

	train_start = time.time()
	for i in range(epoch_num):
		obj_model.train()
		train_end = time.time()
		cnt = 0
		forward_start = time.time()
		for batch_idx, (data, target) in enumerate(test_loader):
			data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
			res = obj_model.forward(data)
			if res == CIFAR100_LABELS_LIST[target.data.numpy()[0]]: cnt += 1
			if batch_idx % 1000 == 0: obj_model.view(data)
		forward_end = time.time()
		print(str(i+1) + ' epoch')
		obj_time = train_end - train_start
		print('img2obj time (sec):', end = '')
		print(obj_time)
		time_spent.append(obj_time)
		print('img2obj accuracy:', end = '')
		print(cnt/len(test_loader))	
		conv.append(1-cnt/len(test_loader))
		forward.append(forward_end - forward_start)
	
	return obj_model, time_spent, conv, forward


def test_cam(obj_model):
	obj_model.cam()

def plot_compare(nn, num, topic):
	x = [i+1 for i in range(len(nn))]
	plt.plot(x, nn, 'r', label = 'nn')
	plt.plot(x, num, 'b', label = 'cnn')
	plt.legend()
	plt.xlabel('epoch number')
	plt.ylabel(topic)

	plt.title(topic + ' comparison between nn and cnn')
	plt.tight_layout()
	plt.show()
	

def plot_time(time_spent, time_forward):
	x = [i+1 for i in range(len(time_spent))]
	plt.plot(x, time_spent, 'r', label = 'time on training')
	plt.plot(x, time_forward, 'b', label = 'time on forwarding')
	plt.legend()
	plt.xlabel('epoch number')
	plt.ylabel('time (s)')

	plt.title('img2obj training/forwarding time vs. epoch number')
	plt.tight_layout()
	plt.show()

def plot_single(conv):
	x = [i+1 for i in range(len(conv))]
	plt.plot(x, conv, 'b')
	plt.legend()
	plt.xlabel('epoch number')
	plt.ylabel('error rate')

	plt.title('img2obj error rate vs. epoch number')
	plt.tight_layout()
	plt.show()
'''
num_model = img2num.Img2Num()
obj_model = img2obj.Img2obj()
#print('img2num')
num_model, num_time, num_conv, num_forward = test_img2num(num_model, 50)
#print('img2obj')
obj_model, obj_time, obj_conv, obj_forward = test_img2obj(obj_model, 50)
torch.save(num_model, './model/num_model.pt')
torch.save(obj_model, './model/obj_model.pt')
np.save('./eval/num_time_spent.npy', num_time)
np.save('./eval/num_conv.npy', num_conv)
np.save('./eval/num_forward.npy', num_forward)
np.save('./eval/obj_time_spent.npy', obj_time)
np.save('./eval/obj_conv.npy', obj_conv)
np.save('./eval/obj_forward.npy', obj_forward)
'''

#obj_model = torch.load('./model/obj_model.pt')
#obj_model, obj_time, obj_conv, obj_forward = test_img2obj(obj_model, 1)
'''
num_time = np.load('./eval/num_time_spent.npy')
num_conv = np.load('./eval/num_conv.npy')
num_forward = np.load('./eval/num_forward.npy')
obj_time = np.load('./eval/obj_time_spent.npy')
obj_conv = np.load('./eval/obj_conv.npy')
obj_forward = np.load('./eval/obj_forward.npy')
#===============================================
nn_time = np.load('/Users/lihsintseng/Desktop/2017Fall/BME595/BME595HW4/tseng24_HW04/for_hw5/time_spent.npy')
nn_conv = np.load('/Users/lihsintseng/Desktop/2017Fall/BME595/BME595HW4/tseng24_HW04/for_hw5/conv.npy')
nn_forward = np.load('/Users/lihsintseng/Desktop/2017Fall/BME595/BME595HW4/tseng24_HW04/for_hw5/forward.npy')
plt.figure(1)
plot_time(obj_time, obj_forward)
plt.figure(2)
plot_single(obj_conv)
plt.figure(3)
plot_compare(nn_time, num_time, 'time spent')
plt.figure(4)
plot_compare(nn_conv, num_conv, 'error rate')
plt.figure(5)
plot_compare(nn_forward, num_forward, 'time spent')
'''
#obj_model = torch.load('./model/obj_model.pt')
#test_cam(obj_model)

