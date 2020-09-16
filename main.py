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
	learning_rate = 0.0001
	batch_size = 100
	num_epochs = 1000
	samples = 1000
	matrix = np.empty((samples, 2))
	matrix[:, 0] = np.linspace(0, 1000, samples)
	matrix[:, 1] = 2*matrix[:, 0] + 20
	# matrix = matrix + 10 * np.random.normal(size=matrix.shape)
	input_dim = matrix.shape[1]
	for i in range(matrix.shape[1]):
		matrix[:, i] = (matrix[:, i] - matrix[:, i].min()) / (matrix[:, i].max() - matrix[:, i].min())
	#plt.figure(0)
	#plt.plot(matrix[:, 0], matrix[:, 1])
	#plt.show()

	#pca_mean = pca.testPCAFit(matrix, 1)
	autoencoder_mean = autoencoder.testAutoEncoderFit(matrix, encoding_dim)


def main():
	np.random.seed(112)
	linear_case()


if __name__ == "__main__":
	main()
