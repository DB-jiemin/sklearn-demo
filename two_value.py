
from sklearn import tree
import pandas as pd
import numpy as np
import json

#dataMat = pd.read_csv("adult/adult.data")
dataMat = pd.read_csv("tmp")
dataLabel = dataMat.pop("classes")

pop_con_cols = ['age', 'fnlwgt', 'marital-status', 'capital-gain', 'capital-loss']
data_con_cols = dataMat[pop_con_cols]
dataLabel = np.reshape(dataLabel,(data_con_cols.shape[0],1))
threshold  = {}

for feat in pop_con_cols:
    X = np.reshape(data_con_cols[feat],(data_con_cols.shape[0],1))
    clf = tree.DecisionTreeClassifier().fit(X, dataLabel)
    threshold[feat] = clf.tree_.threshold

for feat in threshold.keys():
    tmp = list(set(threshold[feat]))
    tmp.sort()
    threshold[feat] = tmp

with open("a1_threshold", "w") as f:
    json.dump(threshold, f)
#print threshold
