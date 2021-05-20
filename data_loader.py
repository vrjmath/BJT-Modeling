import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import scipy.io

class Data(Dataset):

	def __init__(self, data_list):
		self.data_list = data_list
		self.x = np.load('/data/shared/rosenbaum/viraj2/weiman/equalizer/BJT model v4/x_normalizedv88.npy')
		self.y = np.load('/data/shared/rosenbaum/viraj2/weiman/equalizer/BJT model v4/y_normalizedv88.npy')
		self.x = torch.from_numpy(self.x)
		self.y = torch.from_numpy(self.y)
		self.x = torch.transpose(self.x, 0, 1)
		self.x = torch.transpose(self.x, 1, 2)
		self.y = torch.transpose(self.y, 0, 1)
		self.y = torch.transpose(self.y, 1, 2)
		mid = int(len(self))
		high = int(10*len(self)/9)
		self.x_train = self.x[0:mid, 0:501, 0:2]
		self.y_train = self.y[0:mid, 0:501, 0:2]
		self.x_test = self.x[mid:high, 0:501, 0:2]
		self.y_test = self.y[mid:high, 0:501, 0:2]
		print(self.x_train.shape)
	
	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, item):
		input_tensor = self.x_train[item]
		output_tensor = self.y_train[item]
		return input_tensor, output_tensor

	def getTest(self, item):
		return self.x_test[item], self.y_test[item]
