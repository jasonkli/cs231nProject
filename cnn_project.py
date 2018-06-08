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
import os

parser = argparse.ArgumentParser(description = "CS231N")
parser.add_argument('--baseline', action = "store_true", help = "Include if baseline, not GAN")
parser.add_argument('--b', dest = "batch_size", type = int, nargs = '?', default = -1, help = "Batch size")
parser.add_argument('--e', dest = "epochs", type = int, nargs = '?', default = 50, help = "Number of epochs")
parser.add_argument('--lr', dest = "learning_rate", type = float, nargs = '?', default = 5e-3, help = "learning_rate")
parser.add_argument('--f', dest = "f", type = str, nargs = '?', required=True, help = "filename")
parser.add_argument('--d', dest = "d", type = str, nargs = '?', required=True, help = "description")
args = parser.parse_args()

baseline = True if args.baseline else False
batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
filename = args.f
assert not os.path.exists(filename)
description = args.d

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


def flatten(x):
    N = x.shape[0]
    return x.view(N, -1)

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

def check_accuracy_part34(model,f):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        scores = model(X_train)
        _, preds = scores.max(1)
        num_correct += (preds == y_train).sum()
        num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Training: %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        f.write("{}, ".format(100 * acc))
        scores = model(X_val)
        _, preds = scores.max(1)
        num_correct += (preds == y_val).sum()
        num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Validation: %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        f.write("{}\n".format(100 * acc))
        return acc

def train_part34(model, optimizer, num_epochs=10):
    model = model.to(device = device)
    best_acc = 0
    best_model = None
    weights = torch.from_numpy(np.array([1.0, 1.0])).type('torch.FloatTensor')
    length = len(X_train)
    indices = np.random.permutation(length)
    num_batches = int(length / batch_size + 1) if length % batch_size != 0 else int(length / batch_size)
    f = open(filename, 'w')
    f.write(description + "\n")
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
        f.write("{}, {}, ".format(epoch, loss.item()))
        acc = check_accuracy_part34(model,f)
        if acc > best_acc:
            best_acc = acc
            best_model = model
        print()
    f.close()
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

    best_acc, best_model = train_part34(model, optimizer, num_epochs=epochs)
    print("Best Accuracy: {}".format(best_acc))
    scores = best_model(X_test)
    _, preds = scores.max(1)
    num_correct += (preds == y_test).sum()
    num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Test: %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

if __name__ == '__main__':
    main()

