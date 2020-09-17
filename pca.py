from sklearn.decomposition import PCA
import matplotlib.pyplot as plt  # plotting
import numpy as np


def testPCAFit(matrix, n, name,three_D=False, scatter=False):
    pca = PCA(n_components=n)
    pca.fit(matrix)
    samples = 30

    reducedMatrixPCA = pca.transform(matrix)

    reconMatrixPCA = pca.inverse_transform(reducedMatrixPCA)
    reconCostPCA = np.mean(np.power(reconMatrixPCA - matrix, 2), axis=1)
    reconCostPCA = reconCostPCA.reshape(-1, 1)

    plt.figure(1)
    if three_D:
        if scatter:
            ax = plt.axes(projection='3d')
            # ax.plot_wireframe(matrix[:,0],matrix[:,1],matrix[:,2])
            ax.scatter3D(reconMatrixPCA[:, 0], reconMatrixPCA[:, 1], reconMatrixPCA[:, 2])
        else:
            X = reconMatrixPCA[:, 0].reshape(samples, samples)
            Y = reconMatrixPCA[:, 1].reshape(samples, samples)
            Z = reconMatrixPCA[:, 2].reshape(samples, samples)

            ax = plt.axes(projection='3d')
            ax.plot_wireframe(X, Y, Z)
    else:
        plt.plot(reconMatrixPCA[:, 0], reconMatrixPCA[:, 1])

    plt.savefig(name)
    return np.mean(reconCostPCA)
