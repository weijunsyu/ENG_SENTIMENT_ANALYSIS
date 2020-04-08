import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rand

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.manifold import SpectralEmbedding


def selectRandSubSet(npArrayX, npArrayY, sizeOfTrain, sizeOfTest):
    """
    Returns tuple (X_train, y_train, X_test, y_test) consisting of some random subset of original data with the specified size.
    If sizeOfTest OR sizeOfTrain == 'MAX' then select random subset of whichever is train .
    """
    if (len(npArrayX) != len(npArrayY)):
        print("size of X and y do not match!")
        return


    index = 0
    indices = []
    i = rand.randrange(len(npArrayX))
    for c in range(sizeOfTrain):
        while(i in indices) and (len(indices) < sizeOfTrain):
            i = rand.randrange(len(npArrayX))
        indices.append(i)

    X_train = npArrayX.iloc[indices]
    y_train = npArrayY.iloc[indices]

    index = 0
    i = rand.randrange(len(npArrayX))
    for c in range(sizeOfTest):
        while(i in indices) and (len(indices) < (sizeOfTest + sizeOfTrain)):
            i = rand.randrange(len(npArrayX))
        indices.append(i)

    X_test = npArrayX.iloc[indices]
    y_test = npArrayY.iloc[indices]

    return (X_train, y_train, X_test, y_test)


def inputStrToFloat(stringX_train, stringX_test):
    """
    Returns cleaned tuple (X_train, X_test) of type float from type str
    """
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(stringX_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    X_test_counts = count_vect.transform(stringX_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    return (X_train_tfidf, X_test_tfidf)

def matrixCollapsing(matrix, vector_dim, affinity='nearest_neighbors'):
    """
    """
    X_vectors = SpectralEmbedding(n_components=vector_dim, affinity=affinity)
    return X_vectors
