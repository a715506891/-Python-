# datasets自带糖尿病数据,linear_model线性模拟模型
from sklearn import datasets, linear_model


dataDict = datasets.load_diabetes()  # 获取数据字典
xTitle, yTitle = dataDict  # 获取字典中的键值
x = dataDict[xTitle]  # 获取原始数据
y = dataDict[yTitle]  # 获取目标数值
clf = linear_model.LinearRegression(
    fit_intercept=True, normalize=True, copy_X=False, n_jobs=-1)
#使用线性模拟中的最小二乘法计算， 计算b值，回归前进行归一化处理，不复制x，使用所有可用cpu
clf.fit(x, y)
print(clf.coef_)  # 系数
print(clf.intercept_)  # 常量
print(clf.predict(x))  # 求预测值
print(clf.decision_function(x))  # 求预测，等同predict
print(clf.score(x, y))  # R^2
print(clf.get_params())  # 获取参数信息
print(clf.set_params(fit_intercept=False))  # 重新设置参数
