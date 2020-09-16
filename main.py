# add necessary libraries
import matplotlib.pyplot as plt  # plotting
import pandas as pd
import numpy as np

import os

# Import required libraries
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.utils.data

from mpl_toolkits import mplot3d

# Import files
import autoencoder
import pca


def linear_case():
	encoding_dim = [100, 50, 1]
	samples = 1000
	matrix = np.empty((samples, 2))
	matrix[:, 0] = np.linspace(0, 1000, samples)
	matrix[:, 1] = 2*matrix[:, 0] + 20  # y = mx + c
	# matrix = matrix + 10 * np.random.normal(size=matrix.shape)
	input_dim = matrix.shape[1]
	for i in range(matrix.shape[1]):
		matrix[:, i] = (matrix[:, i] - matrix[:, i].min()) / (matrix[:, i].max() - matrix[:, i].min())
	#plt.figure(0)
	#plt.plot(matrix[:, 0], matrix[:, 1])
	#plt.show()

	#pca_mean = pca.testPCAFit(matrix, 1)
	#autoencoder_mean = autoencoder.testAutoEncoderFit(matrix, encoding_dim)


def nonLinear_case():
	samples = 1000
	matrix = np.empty((samples, 2))
	matrix[:, 0] = np.linspace(0, 1000, samples)
	matrix[:, 1] = matrix[:, 0] ** 2 + 20  # y = mx^2 + c
	# matrix = matrix + 10 * np.random.normal(size=matrix.shape)
	for i in range(matrix.shape[1]):
		matrix[:, i] = (matrix[:, i] - matrix[:, i].min()) / (matrix[:, i].max() - matrix[:, i].min())
	plt.plot(matrix[:, 0], matrix[:, 1])
	plt.show()

	pca_mean = pca.testPCAFit(matrix, 1)


def linear_3d():
	samples = 30
	x = np.linspace(0, 1, samples)
	y = x
	X, Y = np.meshgrid(x, y)
	Z = Y + X + 20

	#fig = plt.figure()
	#ax = plt.axes(projection='3d')
	#ax.plot_wireframe(X, Y, Z)
	# ax.scatter3D(X, Y, Z)
	#plt.show()

	matrix = np.empty((samples * samples, 3))
	matrix[:, 0] = X.reshape(samples * samples)
	matrix[:, 1] = Y.reshape(samples * samples)
	matrix[:, 2] = Z.reshape(samples * samples)
	for i in range(matrix.shape[1]):
		matrix[:, i] = (matrix[:, i] - matrix[:, i].min()) / (matrix[:, i].max() - matrix[:, i].min())

	pca_mean = pca.testPCAFit(matrix, 2, True)


def curve_3d():
	return -1


def main():
	np.random.seed(112)
	# linear_case()
	# nonLinear_case()
	linear_3d()


if __name__ == "__main__":
	main()
