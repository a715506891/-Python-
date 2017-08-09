# datasets自带糖尿病数据,linear_model线性模拟模型discriminant_analysis, cross_validation
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt  # 作图

dataDict = datasets.load_diabetes()  # 获取数据字典
xTitle, yTitle = dataDict  # 获取字典中的键值
x = dataDict[xTitle]  # 获取原始数据
y = dataDict[yTitle]  # 获取目标数值
# 岭回归增加正则化项，尽量相处变量的相关性。为有偏估计。
clf = linear_model.Ridge(alpha=0.2, fit_intercept=True,
                         normalize=True, copy_X=False)
# 设置alpha值,计算b值，回归前进行归一化处理，不复制x
clf.fit(x, y)  # 参数拟合
print(clf.coef_)  # 系数
print(clf.intercept_)  # 常量
print(clf.predict(x))  # 求预测值
print(clf.decision_function(x))  # 求预测，等同predict
print(clf.score(x, y))  # R^2，拟合优度
print(clf.get_params())  # 获取参数信息
print(clf.set_params(fit_intercept=False))  # 重新设置参数


def testRidge(al):
    clfTest = linear_model.Ridge(alpha=al, fit_intercept=True,
                                 normalize=True, copy_X=False)
    clfTest.fit(x, y)  # 参数拟合
    return clfTest.score(x, y)


alist = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1,
         2, 5, 10, 20, 50, 100, 200, 500, 1000]
scores = [testRidge(x) for x in alist]  # 获取每个alpha的r方
plt.plot(alist, scores, color='red', linewidth=2)
plt.xlabel('alist')  # 横轴
plt.ylabel('scores')  # 纵轴
plt.xscale('log')  # 设置对数坐标
plt.title('ridge')  # 标题
plt.show()

# 岭回归广义交叉验证
dataDict = datasets.load_diabetes()  # 获取数据字典
xTitle, yTitle = dataDict  # 获取字典中的键值
x = dataDict[xTitle]  # 获取原始数据
y = dataDict[yTitle]  # 获取目标数值
# alphas ,列表数组均可使用，获取最好硅谷结果
clf = linear_model.RidgeCV(alphas=[0.1, 0.2, 1, 10])
clf.fit(x, y)
print(clf.coef_)
print(clf.intercept_)
print(clf.predict(x))
print(clf.decision_function(x))
print(clf.score(x, y))
print(clf.get_params())
print(clf.set_params(fit_intercept=False))
