import torch
import torch.nn as nn
import numpy as np

from rootfind import compute_ic

class RNN(nn.Module):
	def __init__(self, input_size, output_size, seq_len):
		super(RNN, self).__init__()
		self.output_size = output_size
		self.seq_len = seq_len
		self.rnn = nn.RNN(input_size, output_size, 1)
		self.linear = nn.Linear(output_size, 2) 
		self.batch_size = 0
	
	def forward(self, x):
		self.batch_size = x.size(1)
		x_0 = torch.unsqueeze(x[0], 0)
		guess = self.initHidden()
		g = lambda hidden: self.solve(x_0, hidden)
		hn = compute_ic(guess, g, self.batch_size, self.output_size, torch.device('cpu'))
		hn = hn.unsqueeze(0)
		output, hn = self.rnn(x, hn)
		output = self.linear(output)	
		return output, hn

	def solve(self, x, hidden):
		output, hn = self.rnn(x, hidden)
		result = -hidden + output
		return result

	def initHidden(self):
		return torch.zeros(self.seq_len, self.batch_size, self.output_size)
