import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rand

import DataPreprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


#CONSTANTS
#RAW_DATA = "../Data/Binary/Twitter_Sentiment.csv"
RAW_DATA = "../Data/Binary/Review_Sentiment.csv"
REAL_TEST_DATA = "../Data/Binary/TEST.csv"


NUM_TRAINING = 500 #twitter total dataset has 1599998 elements
NUM_TESTING = 500 #review total has 2000

NUM_C_TRIALS_LOG_REG = 100 #10
LOG_REG_C_0 = 0.001 #0.00000001
LOG_REG_ALPHA = 1.15 #2.5
LOG_REG_SOLVER = 'lbfgs' #'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
LOG_REG_MAX_ITER = 10000000 #10000000


def logRegFunction(c):
    return LogisticRegression(penalty='l2', C=c, solver=LOG_REG_SOLVER, max_iter=LOG_REG_MAX_ITER)

Xy = pd.read_csv(RAW_DATA, header=None, index_col=False)
y = Xy.iloc[:, 0]
X = Xy.iloc[:, -1]

#get random subsets:
X_train_str, y_train, X_test_str, y_test = DataPreprocessing.selectRandSubSet(X, y, NUM_TRAINING, NUM_TESTING)

#preprocessing (string to float)
X_train, X_test = DataPreprocessing.inputStrToFloat(X_train_str, X_test_str)

#load REAL_TEST_DATA:
real = pd.read_csv(REAL_TEST_DATA, header=None, index_col=False)

X_real_str = real.iloc[:, -1]
y_real = real.iloc[:, 0]

#vectorize REAL_TEST_DATA with training data:
X_train_real, X_test_real = DataPreprocessing.inputStrToFloat(X_train_str, X_real_str)


logRegClf = logRegFunction(70).fit(X_train_real, y_train)
trainScore = logRegClf.score(X_train_real, y_train)
testScore = logRegClf.score(X_test_real, y_real)
print("c: " + str(70) + " with a training score of: " + str(trainScore) + " and a testing score of: " + str(testScore))




#raise SystemExit(0) #terminate and do not run code below

#Logistic regression
logRegTestError = []
logRegTrainError = []
logRegRegularization = []

a = LOG_REG_ALPHA
c = LOG_REG_C_0
index = 0
while index < NUM_C_TRIALS_LOG_REG:
    logRegRegularization.append(c)
    logRegClf = logRegFunction(c).fit(X_train, y_train)
    testScore = logRegClf.score(X_test, y_test)
    trainScore = logRegClf.score(X_train, y_train)

    print("c: " + str(c) + " with a training score of: " + str(trainScore) + " and a testing score of: " + str(testScore))

    logRegTestError.append(testScore)
    logRegTrainError.append(trainScore)
    c = a * LOG_REG_C_0
    a = a * LOG_REG_ALPHA
    index += 1

#PLot Graphs
plt.figure(0)
plt.title('Error Log_Reg')
plt.grid()
plt.plot(logRegRegularization, logRegTestError, label='Testing')
plt.plot(logRegRegularization, logRegTrainError, label='Training')
plt.xlabel('Regularization Parameter')
plt.ylabel('Performance')
plt.xscale('log')
plt.ylim(0, 1.2)
plt.xlim(min(logRegRegularization), max(logRegRegularization))
plt.legend()
plt.show()

















print(" ")
