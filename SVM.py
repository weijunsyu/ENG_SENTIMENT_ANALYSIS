import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import DataPreprocessing

from sklearn import svm

#CONSTANTS
RAW_DATA = "../Data/Binary/Twitter_Sentiment.csv"

NUM_TRAINING = 500 #total dataset has 1599998 elements
NUM_TESTING = 500

NUM_C_TRIALS_SVM = 100 #10
SVM_C_0 = 0.05 #0.00000000001
SVM_ALPHA = 1.05 #15
SVM_KERNEL = 'rbf' #'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
SVM_DEGREE = 3 #only used in poly kernel, ignored otherwise

def svmFunction(c, kernel, gamma='scale'):
    return svm.SVC(C=c,kernel=kernel, degree=SVM_DEGREE, gamma=gamma)

Xy = pd.read_csv(RAW_DATA)
y = Xy.iloc[:, 0]
X = Xy.iloc[:, -1]

#get random subsets:
X_train, y_train, X_test, y_test = DataPreprocessing.selectRandSubSet(X, y, NUM_TRAINING, NUM_TESTING)

#preprocessing (string to float)
X_train, X_test = DataPreprocessing.inputStrToFloat(X_train, X_test)

#SVM
SVMTestError = []
SVMTrainError = []
SVMRegularization = []

a = SVM_ALPHA
c = SVM_C_0
index = 0
while index < NUM_C_TRIALS_SVM:
    SVMRegularization.append(c)
    svmClf = svmFunction(c, SVM_KERNEL).fit(X_train, y_train)
    testScore = svmClf.score(X_test, y_test)
    trainScore = svmClf.score(X_train, y_train)

    #print("c: " + str(c) + " with a training score of: " + str(trainScore) + " and a testing score of: " + str(testScore))

    SVMTestError.append(testScore)
    SVMTrainError.append(trainScore)
    c = a * SVM_C_0
    a = a * SVM_ALPHA
    index += 1

#PLot Graphs
plt.figure(0)
plt.title('Error SVM')
plt.grid()
plt.plot(SVMRegularization, SVMTestError, label='Testing')
plt.plot(SVMRegularization, SVMTrainError, label='Training')
plt.xlabel('Regularization Parameter')
plt.ylabel('Performance')
plt.xscale('log')
plt.ylim(0, 1.2)
plt.xlim(min(SVMRegularization), max(SVMRegularization))
plt.legend()
plt.show()
