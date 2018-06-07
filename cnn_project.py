import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import argparse

parser = argparse.ArgumentParser(description = "CS231N")
parser.add_argument('--baseline', action = "store_true", help = "Include if baseline, not GAN")
parser.add_argument('--train', action = "store_true", help = "Include if using cycle loss rather than color gradient loss")
parser.add_argument('--b', dest = "batch_size", type = int, nargs = '?', default = -1, help = "Batch size")
parser.add_argument('--lr', dest = "learning_rate", type = float, nargs = '?', default = 5e-3, help = "learning_rate")
args = parser.parse_args()

baseline = True if args.baseline else False
train = True if args.train else False
batch_size = args.batch_size
learning_rate = args.learning_rate

dtype = torch.float32
device = torch.device('cpu')

if baseline:
	X_train = torch.from_numpy(np.load('X_train.npy'))
	X_val = torch.from_numpy(np.load('X_val.npy'))
	y_train = torch.from_numpy(np.load('y_train.npy'))
	y_val = torch.from_numpy(np.load('y_val.npy'))
else:
	X_train = torch.from_numpy(np.load('X_train_gan.npy'))
	X_val = torch.from_numpy(np.load('X_val_gan.npy'))
	y_train = torch.from_numpy(np.load('y_train_gan.npy'))
	y_val = torch.from_numpy(np.load('y_val_gan.npy'))

X_train = X_train.type('torch.FloatTensor')
X_val = X_val.type('torch.FloatTensor')
y_train = y_train.type('torch.LongTensor')
y_val = y_val.type('torch.LongTensor')


if batch_size == -1:
	batch_size = len(X_train)
	
if train:
	X_check = X_train
	y_check = y_train
else:
	X_check = X_val
	y_check = y_val


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
		scores = model(X_check)
		_, preds = scores.max(1)
		num_correct += (preds == y_check).sum()
		num_samples += preds.size(0)
		acc = float(num_correct) / num_samples
		print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
		return acc

def train_part34(model, optimizer, num_epochs=10):
	model = model.to(device = device)
	best_acc = 0
	best_model = None
	weights = torch.from_numpy(np.array([4.0, 1.0])).type('torch.FloatTensor')
	length = len(X_train)
	indices = np.random.permutation(length)
	num_batches = int(length / batch_size + 1) if length % batch_size != 0 else int(length / batch_size)
	for epoch in range(num_epochs):
		for i in range(num_batches):
			start = i * batch_size
			end = (i+1) * batch_size if (i+1)*batch_size < length else length
			index = indices[start:end]
			X_train_b = X_train[index]
			y_train_b = y_train[index]
			model.train()
			scores = model(X_train_b)
			loss = F.cross_entropy(scores, y_train_b, weight=weights) # weights = weights
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print('Iteration %d, loss = %.4f' % (epoch, loss.item()))
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
		nn.Linear(32*16*16, 2000, bias = True),
		nn.BatchNorm1d(2000, momentum = 0.9),
		nn.Dropout(0.5),
		nn.ReLU(),
		nn.Linear(2000, 2, bias = True),
	)
	optimizer = optim.SGD(model.parameters(), lr = learning_rate, 
		momentum = 0.9, nesterov = True)

	best_acc, best_model = train_part34(model, optimizer, num_epochs=200)
	print("Best Accuracy: {}".format(best_acc))

if __name__ == '__main__':
	main()

