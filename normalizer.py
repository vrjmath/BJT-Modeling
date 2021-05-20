import numpy as np
import torch
import scipy.io

import matplotlib.pyplot as plt

sample = 5

def normalize(data):
	for i in range(data.shape[0]):
			print(torch.min(data[i]))
			print(torch.max(data[i]))
			data[i] = (data[i] - torch.min(data[i]))/(torch.max(data[i]) - torch.min(data[i]))
	return data

voltage = scipy.io.loadmat('/data/shared/rosenbaum/viraj2/ash_ctle/datav2/fivehundredinsamplesv12.mat')

current = scipy.io.loadmat('/data/shared/rosenbaum/viraj2/ash_ctle/datav2/fivehundreddoutsamplesv12.mat')

x = np.swapaxes(voltage['datain'], 0, 1)
y = np.swapaxes(current['dataout'], 0, 1)

x = x[0:2, 0:500, 0:1000]
y = y[0:3, 0:500, 0:1000]

print(x.shape)
print(y.shape)

x = torch.from_numpy(x)
y = torch.from_numpy(y)
x_normalized = x#normalize(x)
y_normalized = normalize(y)

torch.save(x_normalized, "x_normalized.pt")
torch.save(y_normalized, "y_normalized.pt")

np.save('x_normalizedv88.npy', x_normalized)
np.save('y_normalizedv88.npy', y_normalized)

plt.figure(1)
plt.plot(x_normalized[0][sample])	
plt.title("Vcc")
plt.xlabel("Time [ns]")
plt.ylabel("Normalized Voltage")
plt.legend(loc='best')
plt.show()

plt.figure(2)
plt.plot(x_normalized[1][sample])	
plt.title("Base Voltage")
plt.xlabel("Time [ns]")
plt.ylabel("Normalized Voltage")
plt.legend(loc='best')
plt.show()

plt.figure(3)
plt.plot(y_normalized[0][sample])	
plt.title("Collector Current")
plt.xlabel("Time [ns]")
plt.ylabel("Normalized Current")
plt.legend(loc='best')
plt.show()

plt.figure(4)
plt.plot(y_normalized[1][sample])	
plt.title("Base Current ")
plt.xlabel("Time [ns]")
plt.ylabel("Normalized Current")
plt.legend(loc='best')
plt.show()



	
