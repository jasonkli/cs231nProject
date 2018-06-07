from os import listdir
from os.path import isdir, join
from PIL import Image
import numpy as np
import os
import csv

FRAMES_PATH = "/frames/x20/"
ATYPIA_PATH = "/atypia/x20/"
#KEYWORD = "decision"
SIZE = 36,36

def imageToVector(directory):
	frames = directory + FRAMES_PATH
	atypia = directory + ATYPIA_PATH
	frames_files = [join(frames,f) for f in listdir(frames)]
	atypia_files = [join(atypia, f) for f in listdir(atypia) if "decision" in f]
	assert len(frames_files) == len(atypia_files)
	num_files = len(frames_files)
	data = []
	scores = []

	for i in range(num_files):
		with open(atypia_files[i], 'r') as f:
			lines = list(csv.reader(f, delimiter=','))
			if len(lines) > 0 and int(lines[0][0]) < 3:
				scores.append(int(lines[0][0]))
				img = Image.open(frames_files[i])
				img.thumbnail(SIZE, Image.ANTIALIAS)
				arr = np.array(img)
				arr = arr[:, 0:arr.shape[0],:]
				data.append(arr)

	return np.array(data), np.array(scores)

def main():
	directories = [f for f in listdir(os.getcwd()) if isdir(f) and 'git' not in f]
	X_a = []
	y_a = []
	X_h = []
	y_h = []
	X = []
	y = []

	# Extract data from files into X and y
	for directory in directories:
		data, scores = imageToVector(directory)
		a,b,c,d = data.shape
		data = data.reshape(a,d,b,c)
		X.append(data)
		y.append(scores)
		if "A" in directory:
			X_a.append(data)
			y_a.append(scores)
		else:
			X_h.append(data)
			y_h.append(scores)
	X = np.concatenate(np.array(X), axis=0) / 255.0
	y = np.concatenate(np.array(y), axis=0) - 1
	X_a = np.concatenate(np.array(X_a), axis=0) / 255.0
	y_a = np.concatenate(np.array(y_a), axis=0) - 1
	X_h = np.concatenate(np.array(X_h), axis=0) / 255.0
	y_h = np.concatenate(np.array(y_h), axis=0) - 1
	print(np.count_nonzero(y_a == 0))
	print(np.count_nonzero(y_a == 1))
	#print(np.count_nonzero(y_a == 2))
	length = X_a.shape[0]
	print(X_a[0].shape)

	# Randomize
	index = np.random.permutation(length)
	index_full = np.random.permutation(2 * length)
	X_a= X_a[index]
	y_a = y_a[index]
	X_h = X_h[index]
	y_h = y_h[index]
	X = X[index_full]
	y = y[index_full]

	#Split into train and val sets
	cutoff = int(length * .7)
	X_a_train = X_a[0:cutoff,:,:,:]
	y_a_train = y_a[0:cutoff]
	X_a_val = X_a[cutoff:length,:,:,:]
	y_a_val = y_a[cutoff:length]
	X_h_train = X_h[0:cutoff,:,:,:]
	y_h_train = y_h[0:cutoff]
	X_h_val = X_h[cutoff:length,:,:,:]
	y_h_val = y_h[cutoff:length]
	X_train = X[0:2*cutoff,:,:,:]
	y_train = y[0:2*cutoff]
	X_val = X[2*cutoff:2*length,:,:,:]
	y_val = y[2*cutoff:2*length]


	#Save files
	np.save('X_a.npy', X_a)
	np.save('y_a.npy', y_a)
	np.save('X_h.npy', X_h)
	np.save('y_h.npy', y_h)
	np.save('X_a_train.npy', X_a_train)
	np.save('y_a_train.npy', y_a_train)
	np.save('X_a_val.npy', X_a_val)
	np.save('y_a_val.npy', y_a_val)
	np.save('X_h_train.npy', X_h_train)
	np.save('y_h_train.npy', y_h_train)
	np.save('X_h_val.npy', X_h_val)
	np.save('y_h_val.npy', y_h_val)
	np.save('X_train.npy', X_train)
	np.save('y_train.npy', y_train)
	np.save('X_val.npy', X_val)
	np.save('y_val.npy', y_val)

if __name__ == "__main__":
	main()

