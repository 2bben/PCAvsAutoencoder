# add necessary libraries
import matplotlib.pyplot as plt  # plotting
import numpy as np

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
	for i in range(matrix.shape[1]):
		matrix[:, i] = (matrix[:, i] - matrix[:, i].min()) / (matrix[:, i].max() - matrix[:, i].min())
	plt.figure(0)
	plt.plot(matrix[:, 0], matrix[:, 1])
	plt.savefig("lin_true.png")

	pca_mean = pca.testPCAFit(matrix, 1, "lin_pca.png")
	autoencoder_mean = autoencoder.testAutoEncoderFit(matrix, encoding_dim, "lin_autoencoder.png")
	printMSE(pca_mean, autoencoder_mean)


def nonLinear_case():
	samples = 1000
	matrix = np.empty((samples, 2))
	encoding_dim = [100, 50, 1]
	matrix[:, 0] = np.linspace(0, 1000, samples)
	matrix[:, 1] = matrix[:, 0] ** 2 + 20  # y = mx^2 + c
	# matrix = matrix + 10 * np.random.normal(size=matrix.shape)
	for i in range(matrix.shape[1]):
		matrix[:, i] = (matrix[:, i] - matrix[:, i].min()) / (matrix[:, i].max() - matrix[:, i].min())
	plt.plot(matrix[:, 0], matrix[:, 1])
	plt.savefig("nonlin_true.png")

	pca_mean = pca.testPCAFit(matrix, 1, "nonlin_pca.png")
	autoencoder_mean = autoencoder.testAutoEncoderFit(matrix, encoding_dim, "nonlin_autoencoder.png")
	printMSE(pca_mean, autoencoder_mean)


def linear_3d():
	samples = 30
	x = np.linspace(0, 1, samples)
	encoding_dim = [100, 50, 1]
	y = x
	X, Y = np.meshgrid(x, y)
	Z = Y + X + 20

	plt.figure(0)
	ax = plt.axes(projection='3d')
	ax.plot_wireframe(X, Y, Z)
	ax.scatter3D(X, Y, Z)
	plt.savefig("lin3D_true.png")

	matrix = np.empty((samples * samples, 3))
	matrix[:, 0] = X.reshape(samples * samples)
	matrix[:, 1] = Y.reshape(samples * samples)
	matrix[:, 2] = Z.reshape(samples * samples)
	for i in range(matrix.shape[1]):
		matrix[:, i] = (matrix[:, i] - matrix[:, i].min()) / (matrix[:, i].max() - matrix[:, i].min())

	pca_mean = pca.testPCAFit(matrix, 2, "lin3D_pca.png", True)
	autoencoder_mean = autoencoder.testAutoEncoderFit(matrix, encoding_dim, "lin3D_autoencoder.png", True)
	printMSE(pca_mean, autoencoder_mean)


def curve_3d():
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
	plt.savefig("curve3D_true.png")

	matrix = np.empty((samples*samples, 3))
	matrix[:,0] = X.reshape(samples*samples)
	matrix[:,1] = Y.reshape(samples*samples)
	matrix[:,2] = Z.reshape(samples*samples)
	
	for i in range(matrix.shape[1]):
		matrix[:, i] = (matrix[:, i] - matrix[:, i].min()) / (matrix[:, i].max() - matrix[:, i].min())
	
	pca_mean = pca.testPCAFit(matrix, 2, "curve3D_pca.png", True)
	autoencoder_mean = autoencoder.testAutoEncoderFit(matrix, encoding_dim, "curve3D_autoencoder.png", True)
	printMSE(pca_mean, autoencoder_mean)


def printMSE(pca, ae):
	print("Reconstruction MSE for PCA:\t\t{} \t\t = {:.3}".format(pca, pca))
	print("Reconstruction MSE for Autoencoder:\t\t{} \t\t = {:.3}".format(ae, ae))
	print("-----------------------------------------------------------\n")


def main():
	np.random.seed(112)
	# inear_case()
	# nonLinear_case()
	# linear_3d()
	curve_3d()


if __name__ == "__main__":
	main()
