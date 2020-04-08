import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch


def PlotClustering(X, labels, title=None):
    """
    """
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    if(X.shape[1] == 2):
        plt.figure()
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], str(labels[i]),
                     color=plt.cm.nipy_spectral(labels[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})

    elif(X.shape[1] == 3):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for i in range(X.shape[0]):
            ax.text(X[i, 0], X[i, 1], X[i, 2], str(labels[i]),
                     color=plt.cm.nipy_spectral(labels[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
    else:
        print("Data not 2D or 3D vector tuples")
        return

    if title is not None:
        plt.title(title, size=17)
    plt.tight_layout()

def PlotDendrogram(data, cut=None, range=None, domain=None, linkage='average', title=None):
    """
    """
    plt.figure()
    if cut is not None:
        plt.axhline(y=cut)
    dendrogram = sch.dendrogram(sch.linkage(data, method=linkage), truncate_mode='lastp')
    if range is not None:
        plt.ylim(range)
    if domain is not None:
        plt.xlim(domain)
    if title is not None:
        plt.title(title, size=17)
    plt.tight_layout()

def PlotPerformanceFit(data, ):






























#end
