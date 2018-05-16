import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F

import numpy as np

dtype = torch.float32
device = torch.device('cpu')
learning_rate = 1e-2

X_train = torch.from_numpy(np.load('X_train.npy'))
X_val = torch.from_numpy(np.load('X_val.npy'))
y_train = torch.from_numpy(np.load('y_train.npy'))
y_val = torch.from_numpy(np.load('y_val.npy'))

X_train = X_train.type('torch.FloatTensor')
X_val = X_val.type('torch.FloatTensor')
y_train = y_train.type('torch.LongTensor')
y_val = y_val.type('torch.LongTensor')

def flatten(x):
	N = x.shape[0]
	return x.view(N, -1)

class Flatten(nn.Module):
	def forward(self, x):
		return flatten(x)


def check_accuracy_part34(model):
	num_correct = 0
	num_samples = 0
	model.eval()
	with torch.no_grad():
		scores = model(X_val)
		_, preds = scores.max(1)
		num_correct += (preds == y_val).sum()
		num_samples += preds.size(0)
		acc = float(num_correct) / num_samples
		print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
		return acc


def train_part34(model, optimizer, epochs = 20):
	model = model.to(device = device)
	best_acc = 0
	best_model = None
	weights = torch.from_numpy(np.array([1.0, 1.5])).type('torch.FloatTensor')
	for e in range(epochs):
		model.train()
		scores = model(X_train)
		loss = F.cross_entropy(scores, y_train,weight=weights)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		print('Iteration %d, loss = %.4f' % (e, loss.item()))
		acc = check_accuracy_part34(model)
		if acc > best_acc:
			best_acc = acc
			best_model = model
		print()
	return best_acc, best_model

def main():
	model = nn.Sequential(
		nn.Conv2d(3, 32, 5, padding = 2, bias = True),
		nn.BatchNorm2d(32, momentum = 0.9),
		nn.ReLU(),
		nn.Conv2d(32, 32, 3, padding = 1, bias = True),
		nn.BatchNorm2d(32,momentum = 0.9),
		nn.ReLU(),
		nn.MaxPool2d(2, stride = 2),
		Flatten(),
		nn.Linear(32*14*16, 500, bias = True),
		nn.BatchNorm1d(500, momentum = 0.9),
		nn.Dropout(0.5),
		nn.ReLU(),
		nn.Linear(500, 2, bias = True),
	)
	optimizer = optim.SGD(model.parameters(), lr = learning_rate, 
		momentum = 0.95, nesterov = True)

	best_acc, best_model = train_part34(model, optimizer, epochs = 20)
	print("Best Accuracy: {}".format(best_acc))

if __name__ == '__main__':
	main()