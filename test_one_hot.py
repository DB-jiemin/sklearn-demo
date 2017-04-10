# coding:utf-8
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier

df = pd.read_csv("adult/adult.data")
y = df.pop("classes") # 将类别存放在ｙ中
X = df
cont_feat = [] # 连续型变量
cat_feat = [] # 类别型变量
cols = list(X.columns) # 所有的特征列

for col in cols:
    if X[col].dtype in ["int64"]: # 如果类型是int64则表示是连续型变量
        cont_feat.append(col)
    else:                         # 否则是类别型变量
        cat_feat.append(col)

print cat_feat
print cont_feat
