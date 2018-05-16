from os import listdir
from os.path import isdir, join
from PIL import Image
import numpy as np
import os
import csv

FRAMES_PATH = "/frames/x40/"
ATYPIA_PATH = "/atypia/x40/"
#KEYWORD = "decision"
SIZE = 32, 32

def imageToVector(directory):
	frames = directory + FRAMES_PATH
	atypia = directory + ATYPIA_PATH
	frames_files = [join(frames,f) for f in listdir(frames)]
	atypia_files = [join(atypia, f) for f in listdir(atypia)]
	assert len(frames_files) == len(atypia_files)
	num_files = len(frames_files)
	data = []
	scores = []

	for i in range(num_files):
		with open(atypia_files[i], 'r') as f:
			lines = list(csv.reader(f, delimiter=','))
		if len(lines[0]) > 1:
			score = int(max(set(lines[1][1:4]), key=lines[1][1:4].count))
			if score < 3:
				scores.append(int(max(set(lines[1][1:4]), key=lines[1][1:4].count)))
				img = Image.open(frames_files[i])
				img.thumbnail(SIZE, Image.ANTIALIAS)
				arr = np.array(img)
				data.append(arr)

	return np.array(data), np.array(scores)

def main():
	directories = [f for f in listdir(os.getcwd()) if isdir(f) and 'git' not in f]
	X = []
	y = []

	# Extract data from files into X and y
	for directory in directories:
		data, scores = imageToVector(directory)
		a,b,c,d = data.shape
		data = data.reshape(a,d,b,c)
		X.append(data)
		y.append(scores)
	X = np.concatenate(np.array(X), axis=0)
	y = np.concatenate(np.array(y), axis=0) - 1
	print(y)
	length = X.shape[0]

	# Randomize
	index = np.random.permutation(length)
	X = X[index]
	print(X.shape)
	y = y[index]

	#Split into train and val sets
	cutoff = int(length * .75)
	X_train = X[0:cutoff,:,:,:]
	y_train = y[0:cutoff]
	print y[y == 0].shape
	print y[y == 1].shape
	print y[y == 2].shape
	X_val = X[cutoff:length,:,:,:]
	y_val = y[cutoff:length]

	#Save files
	np.save('X_train.npy', X_train)
	np.save('y_train.npy', y_train)
	np.save('X_val.npy', X_val)
	np.save('y_val.npy', y_val)

if __name__ == "__main__":
	main()





