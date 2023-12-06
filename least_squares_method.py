# -*-coding: UTF-8 -*- 

#region 线性回归
# import numpy as np
# import matplotlib.pyplot as plt

# # 生成随机数据
# np.random.seed(0)
# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)
# print(y)

# # # 可视化数据
# # plt.scatter(X, y, color='blue')
# # plt.title('Generated Data')
# # plt.xlabel('X')
# # plt.ylabel('y')
# # plt.show()

# # 在 X 前面添加一列 1，以便使用矩阵运算表示截距
# X_b = np.c_[np.ones((100, 1)), X]

# # 计算最优系数（使用最小二乘法）
# theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# # 打印最优系数
# print("最优系数:", theta_best)

# # 预测值
# y_pred = X_b.dot(theta_best)

# # 可视化拟合线
# plt.scatter(X, y, color='blue')
# plt.plot(X, y_pred, color='red', linewidth=3)
# plt.title('Linear Regression with Least Squares')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.show()
#endregion


#region 梯度下降算法
# import numpy as np
# import matplotlib.pyplot as plt

# # 生成随机数据
# np.random.seed(0)
# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)

# # 添加偏置列
# X_b = np.c_[np.ones((100, 1)), X]

# # 初始化参数
# theta = np.random.randn(2, 1)

# # 设置学习率和迭代次数
# learning_rate = 0.01
# iterations = 1000

# # 梯度下降算法
# for iteration in range(iterations):
#     # 计算梯度
#     gradients = 2/100 * X_b.T.dot(X_b.dot(theta) - y)

#     # 更新参数
#     theta = theta - learning_rate * gradients

# # 打印最终参数值
# print("最优参数:", theta)

# # 预测
# X_new = np.array([[0], [2]])
# X_new_b = np.c_[np.ones((2, 1)), X_new]
# y_predict = X_new_b.dot(theta)

# # 可视化结果
# plt.scatter(X, y, color='blue')
# plt.plot(X_new, y_predict, color='red', linewidth=2)
# plt.title('Linear Regression with Gradient Descent')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.show()
#endregion


#region 计算均方误差
# import numpy as np

# # 生成样本数据
# np.random.seed(0)
# X = 2 * np.random.rand(100, 1)
# y = 1 + 2 * X + np.random.randn(100, 1)

# # 假设我们的模型预测值为： y_pred = 2 * X + 1

# # 计算均方误差
# mse = np.mean((y - (2 * X + 1))**2)

# # 打印均方误差
# print("均方误差:", mse)
#endregion


#region 计算均方根误差
# import numpy as np
# from sklearn.metrics import mean_squared_error

# # 生成样本数据
# np.random.seed(0)
# X = 2 * np.random.rand(100, 1)
# y = 1 + 2 * X + np.random.randn(100, 1)

# # 假设我们的模型预测值为： y_pred = 2 * X + 1
# y_pred = 2 * X + 1

# # 计算均方误差
# mse = mean_squared_error(y, y_pred)

# # 计算均方根误差
# rmse = np.sqrt(mse)

# # 打印结果
# print("均方误差 (MSE):", mse)
# print("均方根误差 (RMSE):", rmse)
#endregion


#region 决定系数
# import numpy as np

# # 生成样本数据
# np.random.seed(42)
# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)

# from sklearn.linear_model import LinearRegression

# # 创建线性回归模型
# model = LinearRegression()

# # 拟合模型
# model.fit(X, y)

# # 预测
# y_pred = model.predict(X)

# from sklearn.metrics import r2_score

# # 计算 R^2
# r2 = r2_score(y, y_pred)
# print("决定系数 R^2:", r2)
#endregion


#region 分类准确度
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # 生成样本数据
# np.random.seed(42)
# X = 2 * np.random.rand(100, 1)
# y = (X > 1).astype(int)  # 二分类问题，当 X > 1 时，y 为 1，否则为 0

# # 将数据集分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 使用 ravel() 将 y 转换为一维数组
# y_train = y_train.ravel()
# y_test = y_test.ravel()

# # 创建逻辑回归模型
# model = LogisticRegression()

# # 拟合模型
# model.fit(X_train, y_train)

# # 预测测试集
# y_pred = model.predict(X_test)

# # 计算分类准确度
# accuracy = accuracy_score(y_test, y_pred)

# # 打印结果
# print("分类准确度:", accuracy)
#endregion


#region 混淆矩阵
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # plt.rcParams['font.family'] = 'Arial Unicode MS'
# sns.set(font='SimHei')  # 设置 seaborn 使用宋体（黑体）字体

# # 生成样本数据
# np.random.seed(42)
# X = 2 * np.random.rand(100, 1)
# y = (X > 1).astype(int)  # 二分类问题，当 X > 1 时，y 为 1，否则为 0

# # 将数据集分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 使用 ravel() 将 y 转换为一维数组
# y_train = y_train.ravel()
# y_test = y_test.ravel()

# # 创建逻辑回归模型
# model = LogisticRegression()

# # 拟合模型
# model.fit(X_train, y_train)

# # 预测测试集
# y_pred = model.predict(X_test)

# # 计算混淆矩阵
# cm = confusion_matrix(y_test, y_pred)

# # 使用 seaborn 库绘制热力图
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
# plt.xlabel("预测标签")
# plt.ylabel("真实标签")
# plt.title("混淆矩阵")
# plt.show()
#endregion


#region 交叉验证
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LinearRegression
# import numpy as np

# # 假设 X 和 y 是你的特征矩阵和目标变量
# X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
# y = np.array([2, 4, 6, 8])

# # 创建线性回归模型
# model = LinearRegression()

# # 使用 k=3 的交叉验证
# scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')

# # 由于 cross_val_score 返回的是负的均方误差，因此需要取反
# mse_scores = -scores

# # 打印每次的均方误差
# print("每次的均方误差:", mse_scores)

# # 打印均值作为最终性能指标
# print("平均均方误差:", np.mean(mse_scores))
#endregion


#region 学习曲线
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve
# from sklearn.linear_model import LinearRegression

# import seaborn as sns
# sns.set(font='SimHei')  # 设置 seaborn 使用宋体（黑体）字体

# # 假设 X 和 y 是你的特征矩阵和目标变量
# X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
# y = np.array([2, 4, 6, 8])

# # 创建线性回归模型
# model = LinearRegression()

# # 获取学习曲线数据
# train_sizes, train_scores, test_scores = learning_curve(
#     model, X, y, cv=3, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10))

# # 计算平均的训练集和验证集性能
# train_scores_mean = -np.mean(train_scores, axis=1)
# test_scores_mean = -np.mean(test_scores, axis=1)

# # 绘制学习曲线
# plt.figure(figsize=(10, 6))
# plt.plot(train_sizes, train_scores_mean, label='训练集性能')
# plt.plot(train_sizes, test_scores_mean, label='验证集性能')
# plt.xlabel('训练样本数量')
# plt.ylabel('均方误差')
# plt.title('学习曲线')
# plt.legend()
# plt.show()
#endregion


#region ROC 曲线（Receiver Operating Characteristic Curve）和 AUC（Area Under the Curve）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import seaborn as sns
sns.set(font='SimHei')  # 设置 seaborn 使用宋体（黑体）字体

# 生成随机样本数据
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建逻辑回归模型
model = LogisticRegression()

# 拟合模型
model.fit(X_train, y_train)

# 预测概率
y_prob = model.predict_proba(X_test)[:, 1]

# 计算 ROC 曲线和 AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC 曲线 (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
plt.xlabel('假正例率 (FPR)')
plt.ylabel('真正例率 (TPR)')
plt.title('ROC 曲线')
plt.legend(loc='lower right')
plt.show()
#endregion

