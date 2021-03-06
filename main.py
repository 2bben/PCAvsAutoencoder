# add necessary libraries
import matplotlib.pyplot as plt  # plotting
import numpy as np
import sys

# Import files
import autoencoder
import pca


def linear_case():
	print("===== Running Linear case =====")
	encoding_dim = [100, 50, 1]
	samples = 1000
	matrix = np.empty((samples, 2))
	matrix[:, 0] = np.linspace(0, 1000, samples)
	matrix[:, 1] = 2*matrix[:, 0] + 10  # y = mx + c
	matrix = matrix + 10 * np.random.normal(size=matrix.shape)
	for i in range(matrix.shape[1]):
		matrix[:, i] = (matrix[:, i] - matrix[:, i].min()) / (matrix[:, i].max() - matrix[:, i].min())
	plt.figure(0)
	plt.plot(matrix[:, 0], matrix[:, 1])
	plt.savefig("figures/lin_true.png")
	print("\t-> Finished making true plot")

	pca_mean = pca.testPCAFit(matrix, 1, "figures/lin_pca.png")
	print("\t-> Finished making PCA plot")
	autoencoder_mean = autoencoder.testAutoEncoderFit(matrix, encoding_dim, "figures/lin_autoencoder.png")
	print("\t-> Finished making autoencoder plot\n")
	printMSE(pca_mean, autoencoder_mean)


def nonLinear_case():
	print("===== Running Non Linear case =====")
	samples = 1000
	matrix = np.empty((samples, 2))
	encoding_dim = [100, 50, 1]
	matrix[:, 0] = np.linspace(0, 1000, samples)
	matrix[:, 1] = matrix[:, 0] ** 3 + 10  # y = mx^3 + c
	matrix = matrix + 10 * np.random.normal(size=matrix.shape)
	for i in range(matrix.shape[1]):
		matrix[:, i] = (matrix[:, i] - matrix[:, i].min()) / (matrix[:, i].max() - matrix[:, i].min())
	plt.figure(0)
	plt.plot(matrix[:, 0], matrix[:, 1])
	plt.savefig("figures/nonlin_true.png")
	print("\t-> Finished making true plot")

	pca_mean = pca.testPCAFit(matrix, 1, "figures/nonlin_pca.png")
	print("\t-> Finished making PCA plot")
	autoencoder_mean = autoencoder.testAutoEncoderFit(matrix, encoding_dim, "figures/nonlin_autoencoder.png")
	print("\t-> Finished making autoencoder plot\n")
	printMSE(pca_mean, autoencoder_mean)


def linear_3d():
	print("===== Linear 3D case =====")
	samples = 30
	x = np.linspace(0, 1, samples)
	encoding_dim = [100, 50, 1]
	y = x
	X, Y = np.meshgrid(x, y)
	Z = Y + X + 20

	matrix = np.empty((samples * samples, 3))
	matrix[:, 0] = X.reshape(samples * samples)
	matrix[:, 1] = Y.reshape(samples * samples)
	matrix[:, 2] = Z.reshape(samples * samples)
	for i in range(matrix.shape[1]):
		matrix[:, i] = (matrix[:, i] - matrix[:, i].min()) / (matrix[:, i].max() - matrix[:, i].min())

	plt.figure(0)
	ax = plt.axes(projection='3d')
	ax.plot_wireframe(X, Y, Z)
	ax.scatter3D(X, Y, Z)
	plt.savefig("figures/lin3D_true.png")
	print("\t-> Finished making true plot")

	pca_mean = pca.testPCAFit(matrix, 2, "figures/lin3D_pca.png", True)
	print("\t-> Finished making PCA plot")
	autoencoder_mean = autoencoder.testAutoEncoderFit(matrix, encoding_dim, "figures/lin3D_autoencoder.png", True)
	print("\t-> Finished making autoencoder plot\n")
	printMSE(pca_mean, autoencoder_mean)


def curve_3d():
	print("===== Running Curve 3D case =====")
	encoding_dim = [100, 50, 1]
	samples = 30
	x = np.linspace(0, 1, samples)
	y = x
	X, Y = np.meshgrid(x, y)
	Z = Y**4 + X**4 + 20

	plt.figure(0)
	ax = plt.axes(projection='3d')
	ax.plot_wireframe(X, Y, Z)
	#ax.scatter3D(X, Y, Z)
	plt.savefig("figures/curve3D_true.png")
	print("\t-> Finished making true plot")

	matrix = np.empty((samples*samples, 3))
	matrix[:,0] = X.reshape(samples*samples)
	matrix[:,1] = Y.reshape(samples*samples)
	matrix[:,2] = Z.reshape(samples*samples)
	
	for i in range(matrix.shape[1]):
		matrix[:, i] = (matrix[:, i] - matrix[:, i].min()) / (matrix[:, i].max() - matrix[:, i].min())
	
	pca_mean = pca.testPCAFit(matrix, 2, "figures/curve3D_pca.png", True)
	print("\t-> Finished making PCA plot")
	autoencoder_mean = autoencoder.testAutoEncoderFit(matrix, encoding_dim, "figures/curve3D_autoencoder.png", True)
	print("\t-> Finished making autoencoder plot\n")
	printMSE(pca_mean, autoencoder_mean)


def printMSE(pca, ae):
	print("Reconstruction MSE for PCA:\t\t{} \t\t = {:.3}".format(pca, pca))
	print("Reconstruction MSE for Autoencoder:\t{} \t\t = {:.3}".format(ae, ae))
	print("-----------------------------------------------------------\n")


def main():
	np.random.seed(112)
	linear_case() if int(sys.argv[1]) == 1 else None
	nonLinear_case() if int(sys.argv[2]) == 1 else None
	linear_3d() if int(sys.argv[3]) == 1 else None
	curve_3d() if int(sys.argv[4]) == 1 else None


if __name__ == "__main__":
	main()
