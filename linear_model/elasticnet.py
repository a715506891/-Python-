# elasticnte回归
'''
ElasticNet是Lasso和Ridge回归技术的混合体。
它使用L1来训练并且L2优先作为正则化矩阵。
当有多个相关的特征时，ElasticNet是很有用的。
Lasso 会随机挑选他们其中的一个，而ElasticNet则会选择两个
'''
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt  # 作图
import numpy as np
from mpl_toolkits.basemap import Basemap


dataDict = datasets.()  # 获取数据字典
xTitle, yTitle = dataDict  # 获取字典中的键值
x = dataDict[xTitle]  # 获取原始数据
y = dataDict[yTitle]  # 获取目标数值
# 岭回归增加正则化项，尽量相处变量的相关性。为有偏估计。
clf = linear_model.ElasticNet(alpha=0.2, l1_ratio=0.5, fit_intercept=True,
                              normalize=True, copy_X=False)
# 设置alpha值,l1_ratio值，计算b值，回归前进行归一化处理，不复制x
clf.fit(x, y)  # 参数拟合
print(clf.coef_)  # 系数
print(clf.intercept_)  # 常量
print(clf.predict(x))  # 求预测值
print(clf.decision_function(x))  # 求预测，等同predict
print(clf.score(x, y))  # R^2，拟合优度
print(clf.get_params())  # 获取参数信息
print(clf.set_params(fit_intercept=False))  # 重新设置参数
