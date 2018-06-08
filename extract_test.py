from os import listdir
from os.path import isdir, join
from PIL import Image
import numpy as np
import os
import csv

FRAMES_PATH = "/frames/x40/"
ATYPIA_PATH = "/atypia/x40/"
#KEYWORD = "decision"
SIZE = 36,36

counts = [[0 for j in range(3)] for i in range(6)]

def imageToVector(directory):
	frames = directory + FRAMES_PATH
	atypia = directory + ATYPIA_PATH
	frames_files = [join(frames,f) for f in listdir(frames)]
	atypia_files = [join(atypia, f) for f in listdir(atypia)]
	assert len(frames_files) == len(atypia_files)
	num_files = len(frames_files)
	data = []
	scores = []
	scores_list = [[] for i in range(6)]

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
	directories = [f for f in listdir(os.getcwd()) if isdir(f) and 'git' not in f and 'test' in f]
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
	length = X_a.shape[0]
	print(X_a[0].shape)

	#Save files
	np.save('X_a_test.npy', X_a_train)
	np.save('y_a_test.npy', y_a_train)
	np.save('X_h_test.npy', X_h_train)
	np.save('y_h_test.npy', y_h_train)
	np.save('X_test.npy', X_train)
	np.save('y_test.npy', y_train)


if __name__ == "__main__":
	main()

