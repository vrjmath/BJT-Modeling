import torch
import matplotlib.pyplot as plt
import numpy as np

def nrmse_loss(predicted, true):
	loss_num = predicted - true
	loss_num = torch.square(loss_num)
	loss_num = torch.sum(loss_num)
	loss_denom = torch.square(true)
	loss_denom = torch.sum(loss_denom)
	nrmse = torch.sqrt(loss_num)/torch.sqrt(loss_denom)
	return nrmse.item()

class Metric: 
	def __init__(self, name, num_samples):
		self.name = name
		self.num_samples = num_samples
		self.true = [0] * self.num_samples
		self.predicted= [0] * self.num_samples
		self.diff = [0] * self.num_samples

	def dc_absolute_error(self, predicted, true, index):
		self.true[index] = 0.5*(torch.max(true) + torch.min(true))
		self.predicted[index] = 0.5*(torch.max(predicted) + torch.min(predicted))
		self.diff[index] = 0.5*(torch.max(predicted) + torch.min(predicted)) - 0.5*(torch.max(true) + torch.min(true))

	def ac_relative_error(self, predicted, true, index):
		self.true[index] = 0.5*(torch.max(true) - torch.min(true))
		self.predicted[index] = 0.5*(torch.max(predicted) - torch.min(predicted))
		self.diff[index] = (0.5*(torch.max(predicted) - torch.min(predicted)) - 0.5*(torch.max(true) - torch.min(true)))/0.5*(torch.max(true) - torch.min(true))
	
	def plot_vs(self):
		plt.scatter(self.true, self.predicted)
		plt.xlabel("true")
		plt.ylabel("predicted")
		axes = plt.gca()
		x_vals = np.array(axes.get_xlim())
		y_vals = 0 + 1 * x_vals
		plt.plot(x_vals, y_vals, '--')
		plt.title(self.name)
		plt.show()

	def plot_diff(self):
		sample_index = [0] * self.num_samples
		for i in range(self.num_samples):
			sample_index[i] = i
		plt.scatter(sample_index, self.diff)
		plt.xlabel("sample")
		plt.ylabel("error")
		axes = plt.gca()
		x_vals = np.array(axes.get_xlim())
		y_vals = 0 * x_vals
		plt.plot(x_vals, y_vals, '--')
		plt.title(self.name)
		plt.show()
		


