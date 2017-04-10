#!/usr/bin/env python
# -*- coding: utf-8 -*- # @Time    : 17-3-17 下午3:21
# @Author  : 张杰民
# @Site    : 
# @File    : test_one_hot.py # @Software: PyCharm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler #数据归一化
from sklearn.linear_model import LogisticRegression #逻辑回归模型

# from sklearn.model_selection import cross_val_score #交叉验证
from sklearn.model_selection import ShuffleSplit #数据集分裂
from sklearn.model_selection import RandomizedSearchCV #grid search提升版本
from sklearn.model_selection import learning_curve #计算学习率,训练集合与验证集合
from sklearn.externals import joblib #模型持久化
from sklearn.utils import shuffle #洗牌
from sklearn.metrics import roc_auc_score #计算auc
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt #绘制图形

xgbc = XGBClassifier(max_depth=3, learning_rate=0.1,n_estimators=100, objective="binary:logistic", gamma=0)
xgb_param ={"max_depth" : [3, 4, 5], "learning_rate":np.logspace(-3,3,100) ,"n_estimators":range(20,101,10), "gamma":np.arange(0.01,1.0,0.01)}
cv = ShuffleSplit(n_splits=10, test_size=0.2, train_size=0.8, random_state=5)
rscv = RandomizedSearchCV(xgbc, xgb_param,10, "roc_auc", cv=cv)

dataMat = pd.read_csv('adult/adult.data') #使用pandas读取csv数据
data_mat_test = pd.read_csv('adult/adult.test')
data_mat_test_label = data_mat_test.pop('classes')
dataLabel = dataMat.pop('classes') #将类别列粗
# 在这里添加 train dev test三个数据集合
#下面两个列表是属性列:pop_con_cols表示属性是连续值,pop_cat_cols为离散值
pop_con_cols = ['fnlwgt','marital-status','capital-gain','capital-loss','hours-per-week']
pop_cat_cols = ['age','workclass', 'education', 'occupation', 'e-serv', 'relationship', 'race', 'sex', 'native-country']
dataMat['age']= pd.cut(dataMat['age'],8,labels=['a','b','c','d','e','f','g','h'])
#pop_con_cols = ['age','marital-status','capital-gain','hours-per-week']
dataConCols = dataMat[pop_con_cols]
dataCatCols = dataMat[pop_cat_cols]
#dt = DecisionTreeClassifier().fit(dataConCols, dataLabel)
#with open("adult.dot","w") as f:
#    ff = tree.export_graphviz(dt, out_file=f)
#os.unlink("adult.dot")
#dot_data = tree.export_graphviz(dt, out_file=None)
#graph = pydotplus.graph_from_dot_data(dot_data)
#graph.write_pdf("adult.pdf")
#dot_data = tree.export_graphviz(dt, out_file=None, feature_names=pop_con_cols,class_names=[">50K","<=50K"],filled=True,special_characters=True)
#open("a.png","w").write(graph.create_png())
ss = StandardScaler()
ss.fit(dataConCols) #对连续值进行归一化
aa = ss.fit_transform(dataConCols)
#dataCatCols.fillna(np.nan) #用na填充缺失值,原始数据的缺失值是?
x_vec_cat = pd.get_dummies(dataCatCols) #将离散值进行哑变量表示
X = np.concatenate((x_vec_cat, aa),axis=1) #合并两部分数据,刚才拆成了离散和连续的两部分处理
data_mat_test['age'] = pd.cut(data_mat_test['age'], 8, labels=['a','b','c','d','e','f','g','h'])
dataCatCols = data_mat_test[pop_cat_cols]
dataConCols = data_mat_test[pop_con_cols]
ss = StandardScaler() #实例化
ss.fit(dataConCols) #对连续值进行归一化
aa = ss.fit_transform(dataConCols)
#dataCatCols.fillna(pd.nan) #用na填充缺失值,原始数据的缺失值是?
x_vec_cat = pd.get_dummies(dataCatCols) #将离散值进行哑变量表示
X_test = np.concatenate((x_vec_cat, aa),axis=1) #合并两部分数据,刚才拆成了离散和连续的两部分处理
#
#X, test, dataLabel, y_test = train_test_split(X, dataLabel,test_size=0.3, random_state=1)
Cs = np.array(range(1,100)) / 100.0 #logistic regression回归的超参数C
test_scores = [] #交叉验证的测试分数
# cvs = cross_val_score
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0) #创建交叉验证的数据集
#循环每个C,测试哪个C最合适,每次运行可能会发现C值都不同,没有关系,因为每次的最佳评分都差不多,误差超不过0.0001

#for c in Cs:
#    clf = LogisticRegression(C=c,tol=1e-4,penalty='l2')
#    test_score = cvs(cv=cv,estimator=clf, X=X,y=dataLabel,scoring="accuracy") #这里的roc_auc很重要,因为这里是二分类所以用的这个auc进行评判,如果是多分类还有别的评价指标,具体请看model_selection.metrics
#    test_scores.append(test_score)
#test_scores = np.array(test_scores) #将test_scores转换成np格式数据
#plt.plot(Cs,test_scores.sum(axis=1) / test_scores.shape[1]) #绘制交叉验证的曲线
#plt.show() #显示图形
#下面param_grid是为RandomizedGridCV准备的参数

param_gird = {"C":Cs,"tol":np.logspace(-5,2,100),"penalty":["l1"]} #LR参数
clf = LogisticRegression()

#param_gird = {"C":Cs, "kernel":['linear','rbf']} #SVM参数
#clf = svm.SVC()

#RandomizedSearchCV会组合里面所有的组合进行cv测试,最后给出最好的一组参数
#estimator 表示拟合的算法,这里是lr
#param_distributions 表示要搜索的超参数集合,这个是dict格式
#cv 可以通过上面的ShuffleSplit产生
#scoring 这个参数和上面的cross_val_score表示的一个意思,即评价指标,这里用的也是auc
rgs = RandomizedSearchCV(estimator=clf,param_distributions=param_gird,cv=cv,scoring="roc_auc")
rgs.fit(X,dataLabel) #开始寻找最佳参数
#下面每次输出的可能都不一样,没事,看一下上面的cross_vali_score_curve.jpg就可以了
#因为选取的参数在平行x轴的地方,所以选取哪里都可以
print "=================the first grid search params======"
print "=================算法的最佳参数===================="
print rgs.best_estimator_
print "=============在这个参数下的最好得分================"
print rgs.best_score_
print "=====================最佳超参数===================="
print rgs.best_params_
print "==================================================="

C_space = np.arange(rgs.best_params_["C"] / 2.0, rgs.best_params_["C"] * 2.0, 0.0001)
tol_space = np.arange(rgs.best_params_["tol"] / 2.0, rgs.best_params_["tol"] * 2.0, 0.0001)

param_gird = {"C":C_space, "tol":tol_space, "penalty":["l1"]} #LR参数
clf = LogisticRegression(C=rgs.best_params_["C"], penalty=rgs.best_params_["penalty"], tol=rgs.best_params_["tol"])
rgs = RandomizedSearchCV(estimator=clf,param_distributions=param_gird,cv=cv,scoring="roc_auc")
rgs.fit(X,dataLabel) #开始寻找最佳参数
print
print "=================the second grid search params======"
print "=================算法的最佳参数===================="
print rgs.best_estimator_
print "=============在这个参数下的最好得分================"
print rgs.best_score_
print "=====================最佳超参数===================="
print rgs.best_params_
print "==================================================="
# 重新洗牌,就是将数据集重新打乱
# 会提示下列错误:
# ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 1.0
# 加上面这段代码就可以了,可能是因为在做cv的时候,随机抽取的sample全是一类的
X, dataLabel = shuffle(X, dataLabel) #洗牌
#计算学习率:训练集 与 验证集
train_sizes, train_scores, vali_scores = learning_curve(rgs.best_estimator_, X, dataLabel,scoring="roc_auc",cv=cv,train_sizes=np.array(range(10,1000,1)))
#绘制曲线
#为什么要使用sum 还除以shape[1]?
#因为每次交叉验证出来一组数据,对每组数据求和,然后求取平均值
plt.plot(train_sizes, train_scores.sum(axis=1)/train_scores.shape[1])
plt.plot(train_sizes, vali_scores.sum(axis=1)/vali_scores.shape[1])
plt.show() #显示曲线
#打印所有参数
print rgs.best_estimator_.coef_, rgs.best_estimator_.intercept_
#sklearn自带的模型持久化方法
joblib.dump(rgs.best_estimator_, "LR.pkl") #模型持久化
#使用训练的模型,进行预测
y_score = rgs.best_estimator_.predict(X)
y_score_test = rgs.best_estimator_.predict(X_test)
#打印输出预测的值
#print y_score
#计算auc
# auc 大概在0.76左右
print roc_auc_score(dataLabel, y_score)
print roc_auc_score(data_mat_test_label, y_score_test)

rscv.fit(X, dataLabel)
print
print "=================the thrid grid search params======"
print "=================算法的最佳参数===================="
print rscv.best_estimator_
print "=============在这个参数下的最好得分================"
print rscv.best_score_
print "=====================最佳超参数===================="
print rscv.best_params_
print "==================================================="
y_score = rscv.best_estimator_.predict(X)
y_score_test = rscv.best_estimator_.predict(X_test)
print roc_auc_score(dataLabel, y_score)
print roc_auc_score(data_mat_test_label, y_score_testLabel)

