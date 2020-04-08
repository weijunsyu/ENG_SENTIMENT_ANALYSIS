import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import random as rand
import scipy.cluster.hierarchy as sch
from sklearn.metrics import mean_squared_error
from sklearn.cluster import AgglomerativeClustering

import DataPreprocessing
import Out

#CONSTANTS
RAW_DATA = "../Data/Binary/Twitter_Sentiment.csv"
NUM_TRAINING = 1000 #total dataset has 1599998 elements
NUM_TESTING = 0




def AggCluster(data, k=None, distance=None, affinity='euclidean', linkage='average'):
    """
    """
    if k is None and distance is not None:
        return AgglomerativeClustering(n_clusters=None, compute_full_tree=True,
                                       affinity=affinity, linkage=linkage,
                                       distance_threshold=distance).fit(data)
    if distance is None and k is not None:
        return AgglomerativeClustering(n_clusters=k, affinity=affinity, linkage=linkage).fit(data)

    else:
        print("k and distance cannot both be None in AggCluster")
        return


Xy = pd.read_csv(RAW_DATA)
y = Xy.iloc[:, 0]
X = Xy.iloc[:, -1]

#get random subsets:
X_train, y_train, X_test, y_test = DataPreprocessing.selectRandSubSet(X, y, NUM_TRAINING, NUM_TESTING)

#preprocessing (string to float)
X_train, X_test = DataPreprocessing.inputStrToFloat(X_train, X_test)

X_train = X_train.toarray()
print(X_train.shape)
#X_train = DataPreprocessing.matrixCollapsing(X_train, 2)

#aggCluster2D_single = AggCluster(X_train, k=3, linkage='single')
#X-default, Y-1.395-1.415 CUT at 1.41
Out.PlotDendrogram(X_train, linkage='average')

#Out.PlotClustering(X_train, aggCluster2D_single.labels_)




plt.show()





















#end
