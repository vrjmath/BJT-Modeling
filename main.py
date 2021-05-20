import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

from data_loader import Data
from RNN import RNN
from metrics import Metric, nrmse_loss
from plot import plot_wf

import matplotlib.pyplot as plt
import scipy.io


batch_size = 45
num_epochs = 500
num_samples = 50
split = int(0.9*num_epochs)

train_list = []
for i in range(split):
	train_list.append(i)

train_set = Data(train_list)
train_loader = DataLoader(train_set, batch_size, shuffle=True)
torch.set_default_tensor_type('torch.DoubleTensor')

model = RNN(2, 10, 1)
loss = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

train_loss = [0] * num_epochs
test_loss = [0] * num_epochs
train_loss_nrmse = [0] * num_epochs
test_loss_nrmse = [0] * num_epochs
lr_value = [0] * num_epochs

base_dc_metric = Metric("Absolute DC Error (Base Current)", num_samples)
collector_dc_metric = Metric("Absolute DC Error (Collector Current)", num_samples)
base_ac_metric = Metric("Relative AC Error (Base Current)", num_samples)
collector_ac_metric = Metric("Relative AC Error (Collector Current)", num_samples) 

for epoch in range(num_epochs):
	print("Epoch " + repr(epoch))
	for i, (data, label) in enumerate(train_loader):
		data = torch.transpose(data, 1, 0)
		label = torch.transpose(label, 1, 0)
		optimizer.zero_grad()
		output, hidden = model(data)
		error = loss(output, label)
		error.backward()
		optimizer.step()
		#print("Batch " + repr(i) + ": " + repr(error.item()))
		train_loss[epoch] = train_loss[epoch] + error.item()
		train_loss_nrmse[epoch] = train_loss_nrmse[epoch] + nrmse_loss(output, label)
	train_loss[epoch] = train_loss[epoch]/(split/batch_size)
	train_loss_nrmse[epoch] = train_loss_nrmse[epoch]/(split/batch_size)

	scheduler.step(train_loss[epoch])
	print(optimizer.param_groups[0]['lr'])
	lr_value.append(optimizer.param_groups[0]['lr'])


	for i in range(num_samples):
		voltage, current = train_set.getTest(i)
		voltage = voltage.unsqueeze(1)
		current = current.unsqueeze(1)
		output, hn = model(voltage)

		#print("Test sample " + repr(i) + ": " + repr(test_error.item()))
		#print("Test sample " + repr(i) + ": " + repr(test_error_nrmse.item()))

		test_loss[epoch] = test_loss[epoch] + loss(output, current).item()
		test_loss_nrmse[epoch] = test_loss_nrmse[epoch] + nrmse_loss(output, current)
		if epoch == num_epochs - 1:
			true_base = torch.transpose(current, 0, 2)[1][0]
			predicted_base = torch.transpose(output.detach(), 0, 2)[1][0]
			true_collector = torch.transpose(current, 0, 2)[0][0]
			predicted_collector = torch.transpose(output.detach(), 0, 2)[0][0]

			base_dc_metric.dc_absolute_error(predicted_base, true_base, i)
			base_ac_metric.ac_relative_error(predicted_base, true_base, i)
			collector_dc_metric.dc_absolute_error(predicted_collector, true_collector, i)
			collector_ac_metric.ac_relative_error(predicted_collector, true_collector, i)
			
			if i%10==0:
				plot_wf(true_base, predicted_base, true_collector, predicted_collector)
		

	test_loss[epoch] = test_loss[epoch]/num_samples
	test_loss_nrmse[epoch] = test_loss_nrmse[epoch]/num_samples
	print(test_loss_nrmse[epoch])

collector_dc_metric.plot_diff()
base_dc_metric.plot_diff()
collector_ac_metric.plot_diff()
base_ac_metric.plot_diff()

plt.subplot(1, 1, 1)
plt.semilogy(train_loss, label="train loss")
plt.semilogy(test_loss, label="test loss")
plt.semilogy(test_loss_nrmse, label="nrmse test loss")
plt.semilogy(lr_value, label="lr")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend(loc='best')
plt.show()

